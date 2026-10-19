"""
Preenchimento de áreas para o pincel sólido do Instagram, medido no SM-A736B.

No aparelho, muitos traços curtos (varredura, hachura) deixam uma bolinha em
cada início e o Instagram arredonda as pontas: a área incha e detalhes somem.
O que funciona são anéis fechados e longos que acompanham a forma, recuados
meio pincel da borda. Para cobrir áreas grandes rápido, os anéis começam com
o pincel grosso e as sobras ficam para os pincéis menores; partes finas demais
para qualquer anel viram uma linha central com o pincel fino.
"""
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

Stroke = List[Tuple[int, int]]

# Medições de tempo do injetor no aparelho (ver memória do projeto)
FRAME_S = 1.0 / 60.0
PX_PER_FRAME = 18.0
STROKE_OVERHEAD_S = 0.05
COLOR_SWITCH_S = 1.0
THICKNESS_SWITCH_S = 1.5
# Entrar no traço fino: troca neon -> mínimo -> sólida, com conferência na tela
FINE_SWITCH_EXTRA_S = 3.0


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def concentric_rings(core: np.ndarray, step: int, min_area: float = 2.0) -> List[Stroke]:
    """Contornos fechados de ``core``, recuando ``step`` px a cada volta (furos preservados)."""
    rings: List[Stroke] = []
    current = core.copy()
    kernel = _disk(step)
    while np.count_nonzero(current):
        contours, _ = cv2.findContours(current, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            pts = [(int(x), int(y)) for x, y in cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2)]
            if len(pts) < 3 or cv2.contourArea(c) < min_area:
                # Miolo pequeno: um toque no centro cobre. Com 4 px de percurso; um
                # toque de 1 px era descartado pelo sanitizador (< 2,5 px) e deixava
                # o miolo sem tinta (buraco no meio da gola branca do retrato).
                x, y = (int(round(v)) for v in np.mean(c.reshape(-1, 2), axis=0))
                rings.append([(x - 2, y), (x + 2, y)])
            else:
                rings.append(pts + [pts[0]])
        current = cv2.erode(current, kernel)
    return rings


def _is_tap(stroke: Stroke) -> bool:
    """Toque curto no miolo de um anel (ver concentric_rings)."""
    return len(stroke) == 2 and abs(stroke[1][0] - stroke[0][0]) + abs(stroke[1][1] - stroke[0][1]) <= 4


def _draw(shape, strokes: Sequence[Stroke], width: int) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    for s in strokes:
        if len(s) >= 2:
            cv2.polylines(canvas, [np.asarray(s, np.int32).reshape(-1, 1, 2)], False, 255, max(1, int(width)))
    return canvas


def centerline_strokes(mask: np.ndarray, min_length: float = 3.0) -> List[Stroke]:
    """Linha central de partes finas (onde nenhum anel cabe)."""
    if np.count_nonzero(mask) == 0:
        return []
    from .skeleton_graph import zhang_suen_thinning, SkeletonGraph
    skel = zhang_suen_thinning(mask)
    if np.count_nonzero(skel) == 0:
        return []
    strokes = SkeletonGraph(skel).extract_continuous_strokes(approx_epsilon=0.8, min_length=min_length)
    # Pontinhos isolados que o esqueleto não transforma em traço
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    covered = _draw(mask.shape, strokes, 3)
    for i in range(1, n):
        comp = labels == i
        if stats[i, cv2.CC_STAT_AREA] >= 4 and not covered[comp].any():
            cx, cy = (int(round(v)) for v in centroids[i])
            strokes.append([(cx - 2, cy), (cx + 2, cy)])
    return strokes


def fill_region(mask: np.ndarray, brush_levels: Sequence[Tuple[str, float]],
                thin_level: Tuple[str, float] = None, leftover_min_area: int = 12) -> Dict[str, List[Stroke]]:
    """Cobre ``mask`` sem sair dela, do pincel mais grosso para o mais fino.

    ``brush_levels``: [(nível, largura_px), ...] em qualquer ordem.
    Retorna {nível: traços}. As sobras finas demais para anéis viram linha
    central no nível mais fino (``thin_level``).
    """
    result: Dict[str, List[Stroke]] = {}
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return result
    # Trabalha só na caixa da região: erosões com discos de até 104 px na
    # imagem inteira, para cada região, deixavam a prévia dezenas de vezes mais lenta.
    pad = int(max(w for _lvl, w in brush_levels)) + 4
    y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
    y1, x1 = min(mask.shape[0], ys.max() + pad + 1), min(mask.shape[1], xs.max() + pad + 1)
    if (y0, x0, y1, x1) != (0, 0) + mask.shape[:2]:
        local = fill_region(np.ascontiguousarray(mask[y0:y1, x0:x1]), brush_levels, thin_level, leftover_min_area)
        return {lvl: [[(x + x0, y + y0) for x, y in s] for s in strokes] for lvl, strokes in local.items()}
    levels = sorted(brush_levels, key=lambda lv: -lv[1])
    thin_level = thin_level or levels[-1]
    remaining = (mask > 0).astype(np.uint8) * 255
    for level, width in levels:
        r = max(1, int(round(width / 2.0)))
        # Onde o pincel cabe inteiro dentro da área e ainda há algo a cobrir por perto
        core = cv2.erode(mask, _disk(r))
        core = cv2.bitwise_and(core, cv2.dilate(remaining, _disk(r)))
        if np.count_nonzero(core) == 0:
            continue
        rings = concentric_rings(core, max(2, int(round(width * 0.75))))
        if not rings:
            continue
        loops = [s for s in rings if not _is_tap(s)]
        remaining = cv2.bitwise_and(remaining, cv2.bitwise_not(_draw(mask.shape, loops, width)))
        # Toques nos miolos pequenos só quando cobrem algo que os anéis deixaram:
        # a maioria cai em área já pintada e só alongaria a pintura.
        kept = []
        disk_area = np.pi * r * r
        for tap in (s for s in rings if _is_tap(s)):
            x, y = (int(round(v)) for v in np.mean(np.asarray(tap, np.float32), axis=0))
            spot = np.zeros_like(remaining)
            cv2.circle(spot, (x, y), r, 255, -1)
            if np.count_nonzero(cv2.bitwise_and(spot, remaining)) >= max(leftover_min_area, 0.15 * disk_area):
                kept.append(tap)
                remaining = cv2.bitwise_and(remaining, cv2.bitwise_not(spot))
        if loops or kept:
            result.setdefault(level, []).extend(loops + kept)
    # Sobras: só as que valem um traço (bordas de 1 px ficam sob o contorno)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(remaining)
    leftover = np.zeros_like(remaining)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= leftover_min_area:
            leftover[labels == i] = 255
    thin = centerline_strokes(leftover)
    if thin:
        result.setdefault(thin_level[0], []).extend(thin)
    return result


def stroke_length(stroke: Stroke) -> float:
    if len(stroke) < 2:
        return 0.0
    pts = np.asarray(stroke, np.float32)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def estimate_paint_time(layers: Sequence[dict]) -> float:
    """Tempo no aparelho com o injetor: ~1 ponto por quadro, trocas de cor e de espessura."""
    total = 0.0
    last_color = last_thickness = None
    for layer in layers:
        strokes = layer.get("strokes", [])
        if not strokes:
            continue
        color = tuple(layer.get("render_color_rgb") or layer.get("color_rgb") or ())
        thickness = layer.get("thickness")
        if color != last_color:
            total += COLOR_SWITCH_S
        if thickness != last_thickness:
            total += THICKNESS_SWITCH_S
            from ..device.brush_width import is_fine_level
            if is_fine_level(thickness):
                total += FINE_SWITCH_EXTRA_S
        last_color, last_thickness = color, thickness
        for s in strokes:
            total += STROKE_OVERHEAD_S + max(2, stroke_length(s) / PX_PER_FRAME) * FRAME_S
    return round(total, 1)
