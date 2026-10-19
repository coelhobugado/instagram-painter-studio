"""
Modo Texto: escreve um texto no canvas com fontes TrueType.

O texto não passa pelo caminho genérico de imagem. A diagramação acontece já
em pixels da tela, dentro da área do canvas, e cada letra vira a linha
central da própria forma, desenhada com um pincel da largura da haste da
fonte. Assim uma fonte fina sai com o pincel mínimo e uma fonte grossa sai
com o pincel largo, sem preenchimento por varredura.

Limites medidos no SM-A736B: o pincel mínimo tem ~10 px. Abaixo de ~5 pincéis
de altura de maiúscula as letras embolam, então o ajuste automático nunca
diminui o texto além desse piso; se não couber, o plano não é gerado.
"""
import os
import glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.models import CanvasTransform, StrokePlan
from ..device.brush_width import width_level, FINE_LEVEL, FINE_DRAWN_PX, LETTER_DRAWN_GAIN_PX
from ..device.calibration import HardwareCalibrationEngine
from .line_art_engine import LineArtEngine
from .path_sanitizer import PathSanitizer

def cls_below_baseline(stroke, baselines, margin_px: float = 6.0) -> bool:
    """O traço fica na faixa abaixo da linha de base mais próxima (cedilha, descendente)."""
    ys = [p[1] for p in stroke]
    if not baselines:
        return False
    base = min(baselines, key=lambda b: abs(b - sum(ys) / len(ys)))
    return min(ys) >= base - margin_px and max(ys) > base + 3


FONTS_DIR = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")

# Fontes do Windows testadas no modo Texto (nome exibido -> arquivo).
CURATED_FONTS = {
    "Arial": "arial.ttf",
    "Arial Negrito": "arialbd.ttf",
    "Calibri": "calibri.ttf",
    "Verdana": "verdana.ttf",
    "Georgia": "georgia.ttf",
    "Times New Roman": "times.ttf",
    "Comic Sans": "comic.ttf",
    "Segoe Print": "segoepr.ttf",
    "Segoe Script": "segoesc.ttf",
    "Ink Free": "Inkfree.ttf",
    "Impact": "impact.ttf",
    "Consolas": "consola.ttf",
}

# Altura mínima da maiúscula, em larguras de traço, para as letras não embolarem.
MIN_CAP_IN_BRUSHES = 5.0
# Larguras medidas no aparelho (ver studio/device/brush_width.py)
LETTER_WIDTH_GAIN_PX = LETTER_DRAWN_GAIN_PX
FINE_SLIDER_LEVEL = FINE_LEVEL
# Com traço fino o limite de legibilidade é o tamanho da letra na tela, não o traço.
MIN_CAP_PX = 40.0
STROKE_STYLES = ("fine", "font")
# O esqueleto é extraído numa resolução maior: o de uma letra pequena
# rasterizada em 1x sai cheio de quinas e ramos falsos. Basta a haste ter
# ~SUPERSAMPLE_STEM_PX px; letras grossas usam menos ampliação (mais rápido).
SUPERSAMPLE_MAX = 4
SUPERSAMPLE_STEM_PX = 40.0


class TextFitError(ValueError):
    """O texto não cabe na área sem ficar menor que o mínimo legível."""


@dataclass
class TextLayout:
    font_path: str
    font_size: int
    lines: List[str]
    cap_height: float
    line_height: int
    width: int
    height: int
    align: str


def available_fonts() -> Dict[str, str]:
    """Fontes testadas que existem nesta máquina, mais as demais TTF/OTF da pasta de fontes."""
    fonts = {name: os.path.join(FONTS_DIR, f) for name, f in CURATED_FONTS.items()
             if os.path.exists(os.path.join(FONTS_DIR, f))}
    known = {os.path.basename(p).lower() for p in fonts.values()}
    for path in sorted(glob.glob(os.path.join(FONTS_DIR, "*.ttf")) + glob.glob(os.path.join(FONTS_DIR, "*.otf"))):
        if os.path.basename(path).lower() not in known:
            fonts.setdefault(os.path.splitext(os.path.basename(path))[0], path)
    return fonts


def resolve_font(font: str) -> str:
    """Aceita caminho, nome exibido ou nome de arquivo da pasta de fontes."""
    if os.path.isfile(font):
        return font
    fonts = available_fonts()
    if font in fonts:
        return fonts[font]
    candidate = os.path.join(FONTS_DIR, font)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Fonte não encontrada: {font}")


class TextEngine:

    @staticmethod
    def _cap_height(font: ImageFont.FreeTypeFont) -> float:
        b = font.getbbox("H")
        return float(b[3] - b[1])

    @staticmethod
    def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> Optional[List[str]]:
        """Quebra por palavras; None se uma palavra sozinha não couber na largura."""
        lines: List[str] = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current = words[0]
            if font.getlength(current) > max_w:
                return None
            for word in words[1:]:
                if font.getlength(word) > max_w:
                    return None
                trial = f"{current} {word}"
                if font.getlength(trial) <= max_w:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        return lines

    @classmethod
    def _measure(cls, text: str, font_path: str, size: int, max_w: float, line_gap: float):
        font = ImageFont.truetype(font_path, size)
        lines = cls._wrap(text, font, max_w)
        if lines is None:
            return None
        ascent, descent = font.getmetrics()
        line_h = int(round((ascent + descent) * (1.0 + line_gap)))
        height = line_h * (len(lines) - 1) + ascent + descent
        width = max((font.getlength(l) for l in lines), default=0.0)
        return font, lines, line_h, int(np.ceil(width)), int(height)

    @classmethod
    def fit_layout(cls, text: str, font_path: str, max_w: int, max_h: int, brush_px: float,
                   max_cap_px: Optional[float] = None, line_gap: float = 0.15,
                   align: str = "center") -> TextLayout:
        """Maior tamanho de fonte que cabe em max_w x max_h, com quebra de linha automática.

        A borda reserva meio pincel de cada lado, porque o traço transborda a forma.
        """
        text = text.strip("\n")
        if not text.strip():
            raise TextFitError("Digite um texto.")
        inner_w = max_w - brush_px
        inner_h = max_h - brush_px
        min_cap = max(MIN_CAP_IN_BRUSHES * brush_px, MIN_CAP_PX)

        def fits(size):
            m = cls._measure(text, font_path, size, inner_w, line_gap)
            return m if m is not None and m[4] <= inner_h else None

        lo, hi, best = 4, 2000, None
        while lo <= hi:
            mid = (lo + hi) // 2
            m = fits(mid)
            cap = cls._cap_height(ImageFont.truetype(font_path, mid))
            if m is not None and (max_cap_px is None or cap <= max_cap_px):
                best = (mid, m)
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise TextFitError("Há uma palavra longa demais para a largura da área.")
        size, (font, lines, line_h, width, height) = best
        cap = cls._cap_height(font)
        if cap < min_cap:
            raise TextFitError(
                f"O texto só caberia com letras de {cap:.0f} px de altura; o mínimo legível com "
                f"pincel de {brush_px:.0f} px é {min_cap:.0f} px. Encurte o texto ou aumente a área.")
        return TextLayout(font_path, size, lines, cap, line_h, width, height, align)

    @classmethod
    def render_mask(cls, layout: TextLayout, canvas_w: int, canvas_h: int, scale: int = 1) -> np.ndarray:
        """Máscara do texto (255 = tinta) centralizada no canvas, na escala pedida."""
        font = ImageFont.truetype(layout.font_path, layout.font_size * scale)
        img = Image.new("L", (canvas_w * scale, canvas_h * scale), 0)
        draw = ImageDraw.Draw(img)
        top = (canvas_h - layout.height) / 2.0 * scale
        for i, line in enumerate(layout.lines):
            lw = font.getlength(line)
            if layout.align == "left":
                x = (canvas_w - layout.width) / 2.0 * scale
            elif layout.align == "right":
                x = (canvas_w + layout.width) / 2.0 * scale - lw
            else:
                x = (canvas_w * scale - lw) / 2.0
            draw.text((x, top + i * layout.line_height * scale), line, font=font, fill=255)
        return np.array(img)

    @staticmethod
    def measure_stem_width(mask: np.ndarray) -> float:
        """Largura típica da haste das letras (2x a distância mediana do esqueleto à borda)."""
        if not np.count_nonzero(mask):
            return 0.0
        dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        skel = LineArtEngine.skeletonize((mask > 0).astype(np.uint8) * 255)
        values = dist[skel > 0]
        return float(2.0 * np.median(values)) if values.size else 0.0

    @staticmethod
    def prune_corner_spurs(skel: np.ndarray, dist: np.ndarray, reach: float = 1.45) -> np.ndarray:
        """Remove, em uma única passada, os ramos que nascem nas quinas da letra.

        Numa haste grossa, o esqueleto termina em um "Y" cujos braços vão até
        os cantos. O pincel na ponta de um braço de quina alcança no máximo
        ~1,41 raio a partir da junção, quase o que o pincel parado na junção
        já cobre, então o braço sai. Uma haste curta de verdade (o topo do "d"
        na Impact, os braços do "t") alcança 2 raios ou mais e fica. Podar em
        várias passadas, como no line-art, consome hastes inteiras.
        """
        sk = (skel > 0).astype(np.uint8)
        h, w = sk.shape
        # Vizinhos em ordem circular. O número de cruzamentos 0->1 em volta do
        # pixel distingue ponta (1), linha (2) e junção (3+); contar vizinhos
        # confundiria os "degraus" de uma diagonal com junções.
        ring = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        pad = np.pad(sk, 1)
        nbrs = [pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy, dx in ring]
        crossings = sum(((nbrs[k] == 0) & (nbrs[(k + 1) % 8] == 1)).astype(np.uint8) for k in range(8))
        crossings[sk == 0] = 0
        # 4-vizinhos primeiro: a caminhada passa por todos os pixels do degrau
        order = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        out = sk.copy()
        limit = 4.0 * float(dist.max()) + 4.0
        for ey, ex in np.argwhere((sk > 0) & (crossings == 1)):
            path = [(int(ey), int(ex))]
            seen = {path[0]}
            junction = None
            while len(path) <= limit:
                y, x = path[-1]
                nxt = [(y + dy, x + dx) for dy, dx in order
                       if 0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx] and (y + dy, x + dx) not in seen]
                if not nxt:
                    break
                hit = [q for q in nxt if crossings[q] >= 3]
                if hit:
                    junction = hit[0]
                    break
                path.append(nxt[0])
                seen.add(nxt[0])
            if junction is None:
                continue
            # Alcance do pincel na ponta do ramo, medido a partir da junção
            tip_reach = np.hypot(path[0][0] - junction[0], path[0][1] - junction[1]) + float(dist[path[0]])
            if tip_reach <= reach * float(dist[junction]) + 1.0:
                for q in path:
                    out[q] = 0
        return out * 255

    @classmethod
    def vectorize(cls, mask_ss: np.ndarray, scale: int, brush_px: float) -> List[List[Tuple[float, float]]]:
        """Linha central de cada letra, na ordem de escrita, em pixels da tela.

        Cada componente conectado (letra, pingo, acento) é processado em um
        recorte próprio: o esqueleto fica rápido e a ordem segue a leitura.
        """
        h, w = mask_ss.shape[:2]
        n, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_ss > 0).astype(np.uint8))
        comps = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < scale * scale:  # resíduo de antisserrilhado
                continue
            comps.append((i, x, y, cw, ch))
        # Ordem de leitura: linha (pelo centro vertical) e depois da esquerda para a direita
        line_h = max(1.0, float(np.median([c[4] for c in comps])) if comps else 1.0)
        comps.sort(key=lambda c: (int((c[2] + c[4] / 2.0) // (line_h * 1.2)), c[1]))

        strokes: List[List[Tuple[float, float]]] = []
        for i, x, y, cw, ch in comps:
            pad = 3
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + cw + pad), min(h, y + ch + pad)
            crop = ((labels[y0:y1, x0:x1] == i) * 255).astype(np.uint8)
            dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
            skel = cls.prune_corner_spurs(LineArtEngine.skeletonize(crop), dist)
            paths = LineArtEngine.trace_skeleton_paths(skel, min_length=1.0, approx_epsilon=0.6 * scale)
            local = []
            for p in paths:
                pts = [((px + x0) / scale, (py + y0) / scale) for px, py in p]
                length = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
                # Restos de esqueleto mais curtos que meio pincel ficam cobertos pelo vizinho
                if length >= max(3.0, brush_px * 0.5):
                    local.append(pts)
            if not local:
                # Pingo, ponto, acento pequeno: um toque de 4 px no centro da forma.
                # Coordenadas inteiras para o sanitizador não encurtar o toque.
                cx, cy = (int(round(v)) for v in centroids[i] / scale)
                local = [[(cx - 2, cy), (cx + 2, cy)]]
            strokes.extend(local)
        return strokes

    @staticmethod
    def remove_retraced(strokes: List[List[Tuple[float, float]]], shape: Tuple[int, int],
                        tol_px: float = 2.0, min_piece_px: float = 4.0) -> List[List[Tuple[float, float]]]:
        """Tira de cada traço os trechos que repassam um traço anterior.

        O esqueleto fecha a barriga do "a" subindo pela haste, que depois é
        desenhada de novo; no "ç" a base do "c" sai duas vezes. Com traço fino
        o repasse aparece (duas linhas quase sobrepostas e mais bolinhas de
        início). Cada traço é amostrado a cada 1 px; o que cai a até tol_px
        de um traço anterior sai, e o restante vira pedaços contínuos.
        """
        h, w = shape
        covered = np.zeros((h, w), np.uint8)
        thickness = max(1, int(round(2 * tol_px)))
        out: List[List[Tuple[float, float]]] = []
        for stroke in strokes:
            pts = np.asarray(stroke, dtype=np.float64)
            if len(pts) < 2:
                continue
            # amostras a cada ~1 px, lembrando de qual vértice vieram
            samples, keep = [], []
            for a, b in zip(pts, pts[1:]):
                n = max(1, int(np.ceil(np.hypot(*(b - a)))))
                for t in np.arange(n) / n:
                    samples.append(a + (b - a) * t)
            samples.append(pts[-1])
            samples = np.array(samples)
            xi = np.clip(np.round(samples[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(samples[:, 1]).astype(int), 0, h - 1)
            free = covered[yi, xi] == 0
            # Cruzar um traço (travessão do "t" sobre a haste) cobre poucos px
            # seguidos: só um trecho coberto mais longo é repasse.
            i = 0
            while i < len(free):
                if free[i]:
                    i += 1
                    continue
                j = i
                while j < len(free) and not free[j]:
                    j += 1
                if j - i <= max(8, 2 * thickness) and 0 < i and j < len(free):
                    free[i:j] = True
                i = j
            if free.all():
                out.append(stroke)
            else:
                # Trechos livres; os vértices originais dentro de cada trecho preservam a forma
                start = None
                for i in range(len(samples) + 1):
                    if i < len(samples) and free[i]:
                        if start is None:
                            start = i
                        continue
                    if start is not None:
                        seg = samples[start:i]
                        length = float(np.sum(np.hypot(*np.diff(seg, axis=0).T))) if len(seg) > 1 else 0.0
                        if length >= min_piece_px:
                            simplified = cv2.approxPolyDP(seg.astype(np.float32).reshape(-1, 1, 2), 0.5, False)
                            out.append([tuple(map(float, p)) for p in simplified.reshape(-1, 2)])
                        start = None
            cv2.polylines(covered, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 255, thickness)
        return out

    @staticmethod
    def join_small_fragments(strokes: List[List[Tuple[float, float]]], max_extent_px: float,
                             baselines: List[float], tol_px: float = 9.0) -> List[List[Tuple[float, float]]]:
        """Junta pedaços consecutivos de uma cedilha (abaixo da linha de base) num só gesto.

        O esqueleto parte a cedilha em 3 pedaços de poucos px; no celular cada
        início de traço vira uma bolinha de ~8 px e a cedilha sai embolada.
        Pedaços seguidos cujas pontas ficam a até tol_px se juntam enquanto o
        conjunto couber em max_extent_px. Só abaixo da linha de base: acima
        dela, juntar o topo do "t" com o braço do travessão criaria uma quina
        que o Instagram arredonda.
        """
        out: List[List[Tuple[float, float]]] = []
        for stroke in strokes:
            s = [tuple(p) for p in stroke]
            if out and cls_below_baseline(s, baselines) and cls_below_baseline(out[-1], baselines):
                prev = out[-1]
                pairs = [(np.hypot(prev[-1][0] - s[0][0], prev[-1][1] - s[0][1]), prev, s),
                         (np.hypot(prev[-1][0] - s[-1][0], prev[-1][1] - s[-1][1]), prev, s[::-1]),
                         (np.hypot(prev[0][0] - s[0][0], prev[0][1] - s[0][1]), prev[::-1], s),
                         (np.hypot(prev[0][0] - s[-1][0], prev[0][1] - s[-1][1]), prev[::-1], s[::-1])]
                dist, a, b = min(pairs, key=lambda t: t[0])
                merged = a + b
                xs = [p[0] for p in merged]
                ys = [p[1] for p in merged]
                if dist <= tol_px and max(max(xs) - min(xs), max(ys) - min(ys)) <= max_extent_px:
                    out[-1] = TextEngine._remove_spikes(merged)
                    continue
            out.append(s)
        return out

    @staticmethod
    def _remove_spikes(stroke: List[Tuple[float, float]], max_arm_px: float = 10.0,
                       min_turn_deg: float = 140.0) -> List[Tuple[float, float]]:
        """Tira vai-e-voltas curtos (ponta de ramo falso do esqueleto) de um gesto juntado."""
        pts = list(stroke)
        changed = True
        while changed and len(pts) > 2:
            changed = False
            for i in range(1, len(pts) - 1):
                a = np.subtract(pts[i], pts[i - 1])
                b = np.subtract(pts[i + 1], pts[i])
                na, nb = np.hypot(*a), np.hypot(*b)
                if 0 < na <= max_arm_px and 0 < nb <= max_arm_px:
                    turn = np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
                    if turn >= min_turn_deg:
                        del pts[i]
                        changed = True
                        break
        return pts

    @staticmethod
    def split_at_corners(stroke: List[Tuple[float, float]], window_px: float = 8.0,
                         max_turn_deg: float = 70.0) -> List[List[Tuple[float, float]]]:
        """Quebra o traço nas quinas fechadas (ex.: do travessão para a haste do "T").

        Medido no SM-A736B com traço fino: o Instagram suaviza os pontos
        recebidos e arredonda uma virada de 90° dentro do mesmo gesto; a haste
        do "T" saiu curva como um "J". Levantar o dedo na quina e recomeçar
        dali mantém o ângulo. A virada é medida entre os pontos a window_px
        antes e depois de cada vértice, para pegar quinas divididas em vários
        vértices sem confundir com curvas suaves.
        """
        pts = np.asarray(stroke, dtype=np.float64)
        if len(pts) < 3:
            return [stroke]
        seg = np.hypot(*np.diff(pts, axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])

        def point_at(s):
            s = float(np.clip(s, 0.0, arc[-1]))
            k = min(int(np.searchsorted(arc, s, side="right") - 1), len(seg) - 1)
            t = (s - arc[k]) / seg[k] if seg[k] > 0 else 0.0
            return pts[k] + (pts[k + 1] - pts[k]) * t

        turns = np.zeros(len(pts))
        for i in range(1, len(pts) - 1):
            if arc[i] < 3.0 or arc[-1] - arc[i] < 3.0:
                continue  # rente à ponta, a virada é só o arremate do traço
            a = pts[i] - point_at(arc[i] - window_px)
            b = point_at(arc[i] + window_px) - pts[i]
            na, nb = np.hypot(*a), np.hypot(*b)
            if na > 0 and nb > 0:
                turns[i] = np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
        pieces, start = [], 0
        for c in range(1, len(pts) - 1):
            if (turns[c] > max_turn_deg and turns[c] >= turns[c - 1] and turns[c] >= turns[c + 1]
                    and arc[c] - arc[start] >= 3.0):
                pieces.append([tuple(p) for p in pts[start:c + 1]])
                start = c
        tail = [tuple(p) for p in pts[start:]]
        if pieces and arc[-1] - arc[start] < 3.0:
            pieces[-1].extend(tail[1:])  # sobra curta fica com o pedaço anterior
        else:
            pieces.append(tail)
        return pieces

    @classmethod
    def generate_plan(cls, text: str, font: str, color_engine: Any, simulator: Any,
                      bounds: Dict[str, int], ink_rgb: Tuple[int, int, int] = (255, 255, 255),
                      brush_px: Optional[float] = None, max_cap_px: Optional[float] = None,
                      align: str = "center", seed: int = 42,
                      calibrator: Optional[HardwareCalibrationEngine] = None,
                      stroke: str = "fine") -> Dict[str, Any]:
        """Plano completo do modo Texto dentro de bounds (pixels da tela).

        stroke="fine" escreve com o traço mais fino da caneta sólida (~4 px),
        como uma caneta. stroke="font" segue o peso da fonte: brush_px=None
        escolhe o pincel pela haste da fonte no tamanho final, nunca abaixo do
        pincel mínimo do trilho.
        """
        if stroke not in STROKE_STYLES:
            raise ValueError(f"Traço inválido: {stroke}")
        calib = calibrator if calibrator is not None else HardwareCalibrationEngine()
        min_brush = float(calib.get_effective_brush_width("min", "solid"))
        min_drawn = min_brush + LETTER_WIDTH_GAIN_PX  # traço mais fino que uma letra realmente tem
        font_path = resolve_font(font)
        left = int(bounds.get("left", 80))
        top = int(bounds.get("top", 380))
        canvas_w = int(bounds.get("max_width", int(bounds.get("right", 1000)) - left))
        canvas_h = int(bounds.get("max_height", int(bounds.get("bottom", 1980)) - top))

        # brush_px é a largura desejada no celular, não a posição nominal do slider
        if stroke == "fine":
            fit_brush = FINE_DRAWN_PX
        else:
            fit_brush = max(min_drawn, float(brush_px)) if brush_px else min_drawn
        layout = cls.fit_layout(text, font_path, canvas_w, canvas_h, fit_brush, max_cap_px=max_cap_px, align=align)

        mask = cls.render_mask(layout, canvas_w, canvas_h, 1)
        stem_px = cls.measure_stem_width(cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1])
        ss = int(np.clip(round(SUPERSAMPLE_STEM_PX / max(stem_px, 1.0)), 1, SUPERSAMPLE_MAX))
        mask_ss = cls.render_mask(layout, canvas_w, canvas_h, ss)
        if stroke == "fine":
            level, width = FINE_SLIDER_LEVEL, FINE_DRAWN_PX
        else:
            drawn = max(min_drawn, float(brush_px) if brush_px else float(round(stem_px)))
            nominal = drawn - LETTER_WIDTH_GAIN_PX
            level = "min" if nominal <= min_brush else width_level(nominal)
            width = drawn

        strokes = cls.vectorize(mask_ss, ss, width)
        if stroke == "fine":
            # No traço grosso a virada arredondada some dentro da largura (medido);
            # quebrar ali só acrescentaria bolinhas de início de traço.
            strokes = cls.remove_retraced(strokes, (canvas_h, canvas_w))
            # Cedilhas e acentos: um gesto só, sem quebra nas quinas (a curva
            # arredondada ali fica natural; bolinhas de início não).
            small = 0.35 * layout.cap_height
            ascent = ImageFont.truetype(layout.font_path, layout.font_size).getmetrics()[0]
            text_top = (canvas_h - layout.height) / 2.0
            baselines = [text_top + i * layout.line_height + ascent for i in range(len(layout.lines))]
            strokes = cls.join_small_fragments(strokes, small, baselines)
            strokes = [piece for s in strokes
                       for piece in ([s] if cls_below_baseline(s, baselines) else cls.split_at_corners(s))]
        strokes = PathSanitizer.sanitize_all_strokes(strokes, x_min=0, x_max=canvas_w - 1, y_min=0, y_max=canvas_h - 1)
        strokes = [[(float(x), float(y)) for x, y in s] for s in strokes if len(s) >= 2]

        touch = color_engine.match_color(ink_rgb)
        matched_rgb = tuple(touch.get("matched_rgb") or touch.get("rgb") or ink_rgb)
        hex_rgb = f"#{ink_rgb[0]:02X}{ink_rgb[1]:02X}{ink_rgb[2]:02X}"
        layer = {
            "name": f"Camada 1: Texto ({touch['name']})",
            "color_rgb": tuple(ink_rgb),
            "render_color_rgb": matched_rgb,
            "target_color_rgb": tuple(ink_rgb),
            "color_hex": hex_rgb,
            "brush": "solid",
            "purpose": "text_centerline",
            "touch_info": touch,
            "strokes": strokes,
            "stroke_count": len(strokes),
            "point_count": sum(len(s) for s in strokes),
            "thickness": level,
            "drawn_width_px": width,
            "palette_index": 0,
        }
        layers = [layer]
        palette_colors = [{"name": touch["name"], "hex": hex_rgb, "rgb": tuple(ink_rgb),
                           "matched_hex": touch.get("matched_hex", hex_rgb), "delta_e": touch.get("delta_e", 0.0)}]

        bg_bgr = (22, 22, 22)  # fundo nativo do canvas do Direct
        ink_bgr = (int(matched_rgb[2]), int(matched_rgb[1]), int(matched_rgb[0]))
        fit_img = np.full((canvas_h, canvas_w, 3), bg_bgr, dtype=np.uint8)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        fit_img = (fit_img * (1 - alpha) + np.array(ink_bgr, np.float32) * alpha).astype(np.uint8)

        sim_canvas, cum_buffers = simulator.render_simulation((canvas_h, canvas_w, 3), layers,
                                                              background_bgr=bg_bgr, seed=seed)
        target_executable = np.full((canvas_h, canvas_w, 3), bg_bgr, dtype=np.uint8)
        w_px = max(1, int(round(width)))
        for s in strokes:
            pts = np.round(np.array(s)).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(target_executable, [pts], False, ink_bgr, w_px, lineType=cv2.LINE_AA)

        transform = CanvasTransform(offset_x=left, offset_y=top, scale=1.0)
        total_pts = layer["point_count"]
        time_est_s = len(strokes) * 0.35 + total_pts * 0.003
        return {
            "mode": "text",
            "plan_hash": StrokePlan.compute_plan_hash(seed, "TEXT", layers, transform=transform),
            "transform": transform,
            "transform_dict": {"offset_x": left, "offset_y": top, "scale": 1.0},
            "layers": layers,
            "total_strokes": len(strokes),
            "total_points": total_pts,
            "estimated_time_s": round(time_est_s, 1),
            "simulation_image": sim_canvas,
            "target_executable": target_executable,
            "cumulative_layer_images": cum_buffers,
            "comparison_image": np.hstack([fit_img, sim_canvas]),
            "fit_image": fit_img,
            "background_rgb": (22, 22, 22),
            "ink_rgb": tuple(ink_rgb),
            "estimated_thickness_px": round(stem_px, 1),
            "custom_bounds": bounds,
            "desired_widths": [width] * len(strokes),
            "palette_colors": palette_colors,
            "text_layout": {
                "text": text, "font_path": font_path, "font_size": layout.font_size,
                "lines": layout.lines, "cap_height_px": round(layout.cap_height, 1),
                "brush_px": width, "slider_level": level, "stroke": stroke, "stem_px": round(stem_px, 1), "align": align,
            },
        }
