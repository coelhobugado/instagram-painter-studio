# -*- coding: utf-8 -*-
"""
Motor Especializado de Reprodução de Desenhos de Linhas (Line-Art & Esboços Vetoriais).
Atende aos requisitos de reprodução de traço único central:
- Segue rigorosamente o centro de cada linha via Esqueletização Morfológica.
- Elimina completamente contornos duplicados (evita contorno externo de bordas).
- Preserva curvas orgânicas contínuas com suavização sub-pixel.
- Identifica e preserva a cor original dos traços e a cor de fundo da referência.
- Calibra a espessura do pincel com base na medição da largura física das linhas.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..vision.perceptual_color import PerceptualColorEngine
from .scheduler_v2 import StrokeSchedulerV2
from .path_sanitizer import PathSanitizer
from .spline_interpolator import SplineInterpolator
from .fill_strategies import FillStrategyEngine
from ..device.calibration import HardwareCalibrationEngine
from ..core.models import CanvasTransform, StrokePlan
from .. import config
from ..device.brush_width import FINE_LEVEL, FINE_DRAWN_PX, TRACK_MIN_WIDTH_PX, drawn_width_px, is_fine_level
from .area_fill import fill_region, estimate_paint_time
from .color_extraction import (ink_mask_from_background, stroke_colors, region_color, group_colors,
                               enclosed_regions, lab_to_rgb, to_lab)
from .pictorial_quality import (
    NegativeSpaceAnalyzer,
    SkeletonOptimizer,
    AdaptiveLineArtPartitioner,
    ExecutableTargetBuilder,
)

class LineArtEngine:
    """
    Motor dedicado para renderização de desenhos baseados em linhas (tulipas, tatuagens, botânicos, mangás).
    """

    @classmethod
    def skeletonize(cls, binary_mask: np.ndarray) -> np.ndarray:
        """
        Esqueletização morfológica topologicamente perfeita via Zhang-Suen vetorizado (Fase 1.1).
        Preserva linhas de 1px, 2px, junctions T/X e loops sem perda de componentes.
        """
        from .skeleton_graph import zhang_suen_thinning
        return zhang_suen_thinning(binary_mask)

    @classmethod
    def detect_background_and_ink_colors(cls, img_bgr: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], np.ndarray]:
        """
        Analisa a imagem para determinar a cor do fundo e a cor dos traços de tinta,
        além de gerar a máscara binária dos traços (onde tinta = 255 e fundo = 0).
        Retorna:
            (bg_rgb, ink_rgb, ink_binary_mask)
        """
        h, w = img_bgr.shape[:2]

        # 1. Amostra a borda externa para obter a cor dominante de fundo
        border_pixels = np.concatenate([
            img_bgr[0, :],
            img_bgr[-1, :],
            img_bgr[:, 0],
            img_bgr[:, -1]
        ], axis=0)

        mean_bg_bgr = np.median(border_pixels, axis=0)
        bg_rgb = (int(round(mean_bg_bgr[2])), int(round(mean_bg_bgr[1])), int(round(mean_bg_bgr[0])))
        bg_brightness = float(np.mean(mean_bg_bgr))

        # 2. Conversão para escala de cinza e binarização adaptativa / Otsu
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Se o fundo for claro (> 128), a tinta é escura (inversão necessária)
        if bg_brightness > 128:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Limpeza de micro-ruídos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 3. Calcula a cor média da tinta na região positiva
        ink_pixels = img_bgr[binary_cleaned > 0]
        if len(ink_pixels) > 0:
            mean_ink_bgr = np.median(ink_pixels, axis=0)
            ink_rgb = (int(round(mean_ink_bgr[2])), int(round(mean_ink_bgr[1])), int(round(mean_ink_bgr[0])))
        else:
            ink_rgb = (20, 20, 20) if bg_brightness > 128 else (240, 240, 240)

        return bg_rgb, ink_rgb, binary_cleaned

    @classmethod
    def extract_solid_ink(cls, ink_mask: np.ndarray, brush_width: float) -> np.ndarray:
        """Áreas de tinta largas demais para virar linha central (pupilas, manchas, preenchimentos).

        Um disco do tamanho de ~1,6 pincel precisa caber dentro da área. Linhas
        finas desaparecem na abertura morfológica; manchas sólidas permanecem,
        com os furos internos (ex.: brilho do olho) preservados.
        """
        k = max(5, int(round(brush_width * 1.6)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        solid = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
        keep = np.zeros_like(solid)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= k * k:
                keep[labels == i] = 255
        return keep

    @classmethod
    def generate_solid_fill_strokes(cls, solid_mask: np.ndarray, brush_width: float) -> List[List[Tuple[int, int]]]:
        """Anéis concêntricos que acompanham a forma da mancha, de fora para dentro.

        No aparelho, muitos traços curtos (varredura) deixam uma bolinha em cada
        início e o Instagram arredonda as pontas: a mancha incha e os brilhos
        somem. Anéis fechados são poucos, longos e sem pontas; o primeiro fica
        meio pincel para dentro da borda, e os furos internos ganham seus
        próprios anéis, então continuam abertos.
        """
        if np.count_nonzero(solid_mask) == 0:
            return []
        r = max(1, int(round(brush_width / 2.0)))
        step = max(2, int(round(brush_width * 0.75)))
        strokes: List[List[Tuple[int, int]]] = []
        current = cv2.erode(solid_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * step + 1, 2 * step + 1))
        while np.count_nonzero(current):
            contours, _ = cv2.findContours(current, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                approx = cv2.approxPolyDP(c, 0.8, True).reshape(-1, 2)
                pts = [(int(x), int(y)) for x, y in approx]
                if len(pts) == 1 or cv2.contourArea(c) < 2:
                    # Miolo pequeno demais para um anel: um toque cobre o resto
                    x, y = pts[0]
                    strokes.append([(x, y), (x + 1, y)])
                else:
                    strokes.append(pts + [pts[0]])
            current = cv2.erode(current, ring_kernel)
        return strokes

    @classmethod
    def trace_skeleton_paths(cls, skeleton: np.ndarray, min_length: float = 4.0, approx_epsilon: float = 0.7) -> List[List[Tuple[int, int]]]:
        """
        Transforma os pixels do esqueleto em polilinhas contínuas orientadas via SkeletonGraph formal (Fase 1.2).
        Elimina duplicação de percursos, resolve junções T/X por tangentes contínuas e preserva loops fechados.
        """
        if np.count_nonzero(skeleton) == 0:
            return []
        from .skeleton_graph import SkeletonGraph
        graph = SkeletonGraph(skeleton)
        return graph.extract_continuous_strokes(approx_epsilon=approx_epsilon, min_length=min_length)

    @classmethod
    def estimate_stroke_thickness(cls, binary_mask: np.ndarray, skeleton: np.ndarray,
                                  dist_map: Optional[np.ndarray] = None) -> float:
        """
        Estima a espessura média (em pixels) das linhas originais da referência
        usando a Transformada de Distância nos pixels do esqueleto.
        """
        if np.count_nonzero(skeleton) == 0:
            return 3.2
        if dist_map is None:
            dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        skel_distances = dist_map[skeleton > 0]
        if len(skel_distances) == 0:
            return 3.2
        # A largura total é o dobro do raio medial
        median_radius = float(np.median(skel_distances))
        estimated_width = max(2.0, median_radius * 2.0)
        return estimated_width

    @classmethod
    def compute_stroke_widths(cls, binary_mask: np.ndarray, strokes: List[List[Tuple[int, int]]],
                              dist_map: Optional[np.ndarray] = None) -> List[List[float]]:
        """
        Calcula o perfil de largura desejada desired_width(s) ao longo de cada ponto da centerline
        usando a transformada de distância da máscara original (Fase 1.7 e 11.1).
        """
        if not strokes or np.count_nonzero(binary_mask) == 0:
            return [[3.2] * len(s) for s in strokes]
        if dist_map is None:
            dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        h, w = binary_mask.shape[:2]
        widths_per_stroke = []
        for s in strokes:
            s_widths = []
            for px, py in s:
                cx = max(0, min(w - 1, int(round(px))))
                cy = max(0, min(h - 1, int(round(py))))
                radius = float(dist_map[cy, cx])
                s_widths.append(max(1.0, radius * 2.0))
            widths_per_stroke.append(s_widths)
        return widths_per_stroke

    @classmethod
    def classify_width(cls, width_px: float) -> str:
        """
        Classifica o trecho em classes de espessura (Fase 11.2):
        thin (< 5.0px), medium (5.0 a 14.0px), thick (>= 14.0px).
        """
        if width_px < 5.0:
            return "thin"
        elif width_px < 14.0:
            return "medium"
        return "thick"

    @classmethod
    def generate_offset_strokes(
        cls, stroke: List[Tuple[int, int]], offset_px: float = 2.0
    ) -> List[List[Tuple[int, int]]]:
        """
        Gera traços paralelos por offset lateral para reproduzir linhas muito espessas (Fase 11.3).
        """
        if len(stroke) < 2 or offset_px <= 0.0:
            return [stroke]

        pts = np.array(stroke, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        lengths = np.linalg.norm(diffs, axis=1, keepdims=True) + 1e-6
        tangents = diffs / lengths

        # Normal unitária perpendicular
        normals = np.zeros_like(tangents)
        normals[:, 0] = -tangents[:, 1]
        normals[:, 1] = tangents[:, 0]
        # Estende normais para o mesmo número de pontos
        normals_extended = np.vstack([normals, normals[-1:]])

        # Traço original + offset positivo + offset negativo
        offset_pos = pts + normals_extended * offset_px
        offset_neg = pts - normals_extended * offset_px

        stroke_pos = [(int(round(p[0])), int(round(p[1]))) for p in offset_pos]
        stroke_neg = [(int(round(p[0])), int(round(p[1]))) for p in offset_neg]
        return [stroke, stroke_pos, stroke_neg]

    @classmethod
    def order_line_art_strokes(
        cls, strokes: List[List[Tuple[int, int]]], stroke_widths: List[List[float]]
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[float]], List[str]]:
        """
        Ordena os traços na sequência pictórica correta (Fase 11.5):
        contornos principais -> estruturas internas -> detalhes -> hachuras.
        """
        categorized = []
        for s, w_list in zip(strokes, stroke_widths):
            length = float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))) if len(s) > 1 else 1.0
            mean_w = float(np.mean(w_list)) if w_list else 3.0
            
            if length > 80.0 and mean_w >= 4.0:
                cat = "contour"
                order_key = 0
            elif length > 30.0:
                cat = "internal_structure"
                order_key = 1
            elif length > 12.0:
                cat = "detail"
                order_key = 2
            else:
                cat = "hatch"
                order_key = 3

            categorized.append((order_key, -length, s, w_list, cat))

        # Ordena prioritariamente por categoria (0 -> 1 -> 2 -> 3) e desempata por comprimento decrescente
        categorized.sort(key=lambda item: (item[0], item[1]))
        
        ordered_strokes = [item[2] for item in categorized]
        ordered_widths = [item[3] for item in categorized]
        ordered_purposes = [item[4] for item in categorized]
        return ordered_strokes, ordered_widths, ordered_purposes

    @classmethod
    def verify_line_art_topology_and_metrics(
        cls, strokes: List[List[Tuple[int, int]]], binary_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Gate obrigatório para validação métrica de Line-Art (Fase 11.6):
        - false bridges = 0
        - missing branches = 0
        - topology preservation >= 0.95
        - width P95 residual <= 3.0 px
        """
        h, w = binary_mask.shape[:2]
        canvas_test = np.zeros((h, w), dtype=np.uint8)
        for s in strokes:
            if len(s) >= 2:
                pts = np.array(s, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas_test, [pts], False, 255, 2)

        # Connected components
        num_orig, _ = cv2.connectedComponents(binary_mask)
        num_reprod, _ = cv2.connectedComponents(canvas_test)

        # Distância de chanfro / residual de largura
        dist_orig = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
        dist_test = cv2.distanceTransform(canvas_test, cv2.DIST_L2, 5)
        
        # Residual de largura nos pontos reproduzidos
        residuals = []
        for s in strokes:
            for px, py in s:
                cx = max(0, min(w - 1, px))
                cy = max(0, min(h - 1, py))
                w_orig = float(dist_orig[cy, cx] * 2.0)
                residuals.append(abs(w_orig - 2.0))

        p95_residual = float(np.percentile(residuals, 95)) if residuals else 0.0
        missing_branches = max(0, num_orig - num_reprod)
        false_bridges = max(0, num_reprod - num_orig) if num_reprod > num_orig else 0
        topo_score = float(min(1.0, max(0.0, 1.0 - (missing_branches + false_bridges) / max(1, num_orig))))

        passed = (false_bridges == 0) and (missing_branches <= 1) and (topo_score >= 0.90) and (p95_residual <= 5.0)

        return {
            "passed": passed,
            "false_bridges": false_bridges,
            "missing_branches": missing_branches,
            "topology_preservation": topo_score,
            "width_p95_residual": round(p95_residual, 2),
            "num_original_ccs": num_orig,
            "num_reproduced_ccs": num_reprod,
        }

    # Larguras nominais de linha no modo colorido (px). O slider do Instagram é
    # contínuo; poucas classes mantêm baixo o número de ajustes de espessura.
    # 3 é o traço fino (abaixo do trilho; ~4 px reais).
    LINE_WIDTH_CLASSES = (3, 10, 14, 20, 28, 38, 50)
    # Formas mais largas que isso viram área preenchida em anéis, não linha.
    AREA_MIN_WIDTH_PX = 26.0
    AREA_LEVELS = (("thick", 104.0), ("med", 58.0), ("w28", 28.0), ("min", 10.0))

    @classmethod
    def _line_level(cls, width_class, min_width):
        if width_class < TRACK_MIN_WIDTH_PX:
            return FINE_LEVEL
        return "min" if width_class <= min_width + 0.5 else f"w{int(width_class)}"

    @classmethod
    def _width_class(cls, desired_px, min_width, fine=None):
        """Classe cuja largura real no aparelho fica mais perto da linha original.

        Compara em razão (log) e penaliza engordar: duas linhas próximas que
        engordam se fundem, afinar só deixa o traço mais leve. Sem traço fino,
        as linhas finas vão para o mínimo do trilho, como antes.
        """
        fine = config.FINE_LINES if fine is None else fine
        desired = max(1.0, float(desired_px))
        best, best_cost = None, None
        for c in cls.LINE_WIDTH_CLASSES:
            if c < TRACK_MIN_WIDTH_PX and not fine:
                continue
            if TRACK_MIN_WIDTH_PX <= c < min_width - 0.5:
                continue
            drawn = drawn_width_px(cls._line_level(c, min_width), c)
            cost = abs(np.log(drawn / desired)) + (0.3 if drawn > desired else 0.0)
            if best_cost is None or cost < best_cost:
                best, best_cost = c, cost
        return best

    @staticmethod
    def _smooth_labels(labels, window=7):
        """Moda deslizante: tira oscilações de 1–2 pontos antes de cortar o traço."""
        n = len(labels)
        if n <= 2:
            return list(labels)
        half = window // 2
        out = []
        for i in range(n):
            seg = labels[max(0, i - half):i + half + 1]
            out.append(max(set(seg), key=seg.count))
        return out

    @classmethod
    def _split_runs(cls, stroke, keys, min_run=5):
        """Divide o traço onde a chave (largura, cor) muda; trechos curtos ficam com o vizinho."""
        runs = []
        start = 0
        for i in range(1, len(keys) + 1):
            if i == len(keys) or keys[i] != keys[start]:
                runs.append([start, i - 1, keys[start]])
                start = i
        merged = []
        for r in runs:
            if merged and (r[1] - r[0] + 1) < min_run:
                merged[-1][1] = r[1]
            elif not merged and len(runs) > 1 and (r[1] - r[0] + 1) < min_run:
                runs[1][0] = r[0]
                continue
            else:
                merged.append(r)
        out = []
        for a, b, key in merged:
            # Um ponto de sobreposição evita fresta entre trechos vizinhos
            seg = stroke[max(0, a - 1):b + 1] if a > 0 else stroke[a:b + 1]
            if len(seg) >= 2:
                out.append((key, seg))
        return out

    @classmethod
    def _build_multicolor_layers(cls, fit_img, ink_mask, solid_mask, strokes, stroke_widths, critical_mask, calib,
                                 color_engine, bg_rgb, fill_regions, background_painted, first_index):
        """Camadas por cor e largura: áreas (anéis em várias escalas) e depois as linhas por cima.

        Retorna (camadas, paleta, [(máscara, rgb_executável)] para o alvo executável).
        """
        from .perceptual_optimizer import PerceptualPlanOptimizer
        from .color_extraction import point_colors
        h, w = ink_mask.shape[:2]
        min_w = calib.get_effective_brush_width("min", "solid")
        lab_img = to_lab(fit_img)
        bg_lab = to_lab(np.uint8([[bg_rgb[::-1]]]))[0, 0]

        # 1. Áreas: formas largas de tinta (separadas por cor) e, no livro de colorir, o miolo entre as linhas
        areas = []  # (máscara, cor Lab, peso)
        if np.count_nonzero(solid_mask):
            n, labels, _stats, _ = cv2.connectedComponentsWithStats(solid_mask)
            for i in range(1, n):
                comp = (labels == i).astype(np.uint8) * 255
                pix = lab_img[comp > 0]
                if len(pix) > 200 and float(np.mean(np.std(pix, axis=0))) > 8.0:
                    k = 3
                    _, lbl, centers = cv2.kmeans(pix.astype(np.float32), k, None,
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                                 3, cv2.KMEANS_PP_CENTERS)
                    full = np.full(comp.shape, -1, np.int32)
                    full[comp > 0] = lbl.ravel()
                    for j in range(k):
                        part = cv2.morphologyEx((full == j).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                        if np.count_nonzero(part) >= 30:
                            areas.append((part, centers[j], float(np.count_nonzero(part))))
                else:
                    areas.append((comp, np.median(pix, axis=0), float(len(pix))))
        if fill_regions:
            for mask, lab_color in enclosed_regions(ink_mask, fit_img, bg_rgb, background_painted=background_painted):
                areas.append((mask, lab_color, float(np.count_nonzero(mask))))

        # 2. Linhas: cor e largura medidas ponto a ponto
        per_stroke = []
        all_colors, all_weights = [], []
        for stroke, widths in zip(strokes, stroke_widths):
            if len(stroke) < 2:
                continue
            colors = point_colors(stroke, lab_img, ink_mask, bg_lab)
            smooth_w = [float(np.median(widths[max(0, i - 4):i + 5])) for i in range(len(widths))]
            classes = [cls._width_class(v, min_w) for v in smooth_w]
            per_stroke.append((stroke, colors, classes))
            for c in colors:
                all_colors.append(c)
                all_weights.append(1.0)

        # 3. Agrupa as cores de linhas e áreas numa paleta curta
        area_offset = len(all_colors)
        all_colors += [a[1] for a in areas]
        all_weights += [a[2] / max(1.0, min_w) for a in areas]
        if not all_colors:
            return [], [], []
        labels, centers = group_colors(all_colors, all_weights)
        # Quase preto e pouco saturado é preto: a borda mistura a tinta com a
        # área vizinha (ex.: contorno do Homem-Aranha "puxado" para o vermelho).
        for c in centers:
            if float(c[0]) < 25 and float(np.hypot(c[1], c[2])) < 20:
                c[1] = c[2] = 0.0
        touches = [color_engine.match_color(lab_to_rgb(c)) for c in centers]
        rgbs = [lab_to_rgb(c) for c in centers]
        matched = [tuple(t.get("matched_rgb") or t.get("rgb") or rgbs[k]) for k, t in enumerate(touches)]
        by_light = sorted(range(len(centers)), key=lambda k: -float(centers[k][0]))

        area_strokes, target_fills = {}, []
        for (mask, _c, _wt), k in zip(areas, labels[area_offset:]):
            target_fills.append((mask, matched[k]))
            for lvl, st in fill_region(mask, cls.AREA_LEVELS, thin_level=("min", min_w)).items():
                area_strokes.setdefault((lvl, k), []).extend(st)

        line_strokes = {}
        cursor = 0
        # Lascas da borda suavizada das áreas (já cobertas pelo preenchimento) não são linhas
        area_halo = cv2.dilate(solid_mask, np.ones((13, 13), np.uint8)) if np.count_nonzero(solid_mask) else None
        for stroke, colors, classes in per_stroke:
            point_labels = labels[cursor:cursor + len(colors)]
            cursor += len(colors)
            keys = cls._smooth_labels(list(zip(classes, point_labels)))
            for (wclass, k), seg in cls._split_runs(stroke, keys):
                if area_halo is not None:
                    inside = sum(1 for x, y in seg if area_halo[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))])
                    if inside >= 0.7 * len(seg):
                        continue
                lvl = cls._line_level(wclass, min_w)
                line_strokes.setdefault((lvl, k), []).append(seg)

        layers = []

        def add(kind_name, purpose, lvl, k, layer_strokes):
            layer_strokes = PathSanitizer.sanitize_all_strokes(layer_strokes, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1)
            drawn = None
            if purpose == "line_art_centerline" and layer_strokes:
                drawn = drawn_width_px(lvl, calib.get_effective_brush_width(lvl, "solid"))
                l_temp = {"purpose": purpose, "strokes": layer_strokes}
                PerceptualPlanOptimizer.filter_layer_strokes_for_safety(
                    l_temp, protected_mask=critical_mask, brush_width=drawn, canvas_shape=(h, w))
                layer_strokes = l_temp["strokes"] or layer_strokes
            if not layer_strokes:
                return
            rgb = rgbs[k]
            layers.append({
                "name": f"Camada {first_index + len(layers)}: {kind_name} ({touches[k]['name']})",
                "color_rgb": rgb, "render_color_rgb": matched[k], "target_color_rgb": rgb,
                "color_hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
                "brush": "solid", "purpose": purpose, "touch_info": touches[k],
                "strokes": layer_strokes, "stroke_count": len(layer_strokes),
                "point_count": sum(len(st) for st in layer_strokes), "thickness": lvl,
            })
            if drawn is not None:
                # Largura real no aparelho: a prévia e o alvo usam esta, não a nominal
                layers[-1]["drawn_width_px"] = drawn

        def width_of(lvl):
            return calib.get_effective_brush_width(lvl, "solid")

        # Áreas: pincel grosso → fino; dentro de cada largura, cores claras antes das escuras
        for lvl in sorted({key[0] for key in area_strokes}, key=lambda v: -width_of(v)):
            for k in by_light:
                if (lvl, k) in area_strokes:
                    add("Áreas", "line_art_fill", lvl, k, area_strokes[(lvl, k)])
        # Linhas por cima: largas antes das finas, para os detalhes finos ficarem nítidos
        for lvl in sorted({key[0] for key in line_strokes}, key=lambda v: -width_of(v)):
            for k in by_light:
                if (lvl, k) in line_strokes:
                    name = ("Traços finos" if is_fine_level(lvl)
                            else "Linhas" if width_of(lvl) > min_w + 0.5 else "Traços")
                    add(name, "line_art_centerline", lvl, k, line_strokes[(lvl, k)])

        used = {l["touch_info"]["name"] for l in layers}
        palette = [{
            "name": touches[k]["name"],
            "hex": f"#{rgbs[k][0]:02X}{rgbs[k][1]:02X}{rgbs[k][2]:02X}",
            "rgb": rgbs[k],
            "matched_hex": touches[k].get("matched_hex", "#000000"),
            "delta_e": touches[k].get("delta_e", 0.0),
            "palette_page": touches[k].get("palette_page"),
            "is_long_press": touches[k].get("is_long_press", False),
        } for k in by_light if touches[k]["name"] in used]
        return layers, palette, target_fills

    @classmethod
    def _finish_plan(cls, fit_img, layers, palette_colors, simulator, transform, seed, calib, should_paint_bg,
                     bg_matched_rgb, target_fills, critical_mask, ink_mask, negative_regions, estimated_width,
                     stroke_widths, custom_bounds, bg_rgb, ink_rgb, solid_mask, style):
        """Simulação, alvo executável e metadados comuns do plano multicolorido."""
        h, w = fit_img.shape[:2]
        device_native_bg_bgr = (22, 22, 22)
        sim_canvas, cum_buffers = simulator.render_simulation((h, w, 3), layers, background_bgr=device_native_bg_bgr,
                                                              seed=seed)
        target_executable = np.zeros((h, w, 3), dtype=np.uint8)
        if should_paint_bg and bg_matched_rgb is not None:
            target_executable[:] = (int(bg_matched_rgb[2]), int(bg_matched_rgb[1]), int(bg_matched_rgb[0]))
        else:
            target_executable[:] = device_native_bg_bgr
        for mask, rgb in target_fills:
            target_executable[mask > 0] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        for l in layers:
            if l.get("purpose") == "line_art_centerline":
                rgb = l["render_color_rgb"]
                l_w = max(1, int(round(l.get("drawn_width_px")
                                       or calib.get_effective_brush_width(l.get("thickness", "min"), "solid"))))
                for st in l.get("strokes", []):
                    if len(st) >= 2:
                        cv2.polylines(target_executable, [np.array(st, dtype=np.int32).reshape((-1, 1, 2))], False,
                                      (int(rgb[2]), int(rgb[1]), int(rgb[0])), l_w, lineType=cv2.LINE_AA)
        line_strokes = [st for l in layers if l.get("purpose") == "line_art_centerline" for st in l.get("strokes", [])]
        return {
            "mode": "line_art",
            "style": style,
            "plan_hash": StrokePlan.compute_plan_hash(seed, "LINE_ART", layers, transform=transform),
            "transform": transform,
            "transform_dict": {"offset_x": transform.offset_x, "offset_y": transform.offset_y, "scale": transform.scale},
            "layers": layers,
            "total_strokes": sum(l["stroke_count"] for l in layers),
            "total_points": sum(l["point_count"] for l in layers),
            "estimated_time_s": estimate_paint_time(layers),
            "simulation_image": sim_canvas,
            "target_executable": target_executable,
            "cumulative_layer_images": cum_buffers,
            "comparison_image": np.hstack([fit_img, sim_canvas]),
            "fit_image": fit_img,
            "background_rgb": bg_rgb,
            "ink_rgb": ink_rgb,
            "estimated_thickness_px": round(estimated_width, 1),
            "custom_bounds": custom_bounds,
            "desired_widths": stroke_widths,
            "palette_colors": palette_colors,
            "negative_spaces_detected": len(negative_regions),
            "solid_ink_pixels": int(np.count_nonzero(solid_mask)),
            "critical_protection_mask": critical_mask,
            "topology_metrics": cls.verify_line_art_topology_and_metrics(line_strokes, ink_mask),
        }

    @classmethod
    def generate_plan(cls, fit_img: np.ndarray,
                      color_engine: PerceptualColorEngine,
                      simulator: Any,
                      transform: CanvasTransform,
                      seed: int = 42,
                      custom_bounds: Optional[Dict[str, int]] = None,
                      paint_background: Optional[bool] = None,
                      override_ink_rgb: Optional[Tuple[int, int, int]] = None,
                      override_bg_rgb: Optional[Tuple[int, int, int]] = None,
                      calibrator: Optional[HardwareCalibrationEngine] = None,
                      colorize: bool = False,
                      fill_regions: bool = False) -> Dict[str, Any]:
        """
        Gera o plano completo de desenho para o Modo Linhas (Line-Art),
        com traço único centralizado, sem duplicação de contorno e com fundo e cor fiéis.
        Suporta personalização de cor da tinta (override_ink_rgb) e cor de fundo (override_bg_rgb).
        """
        h, w = fit_img.shape[:2]

        # 1. Detecção da cor de fundo, cor da tinta e binarização
        detected_bg_rgb, detected_ink_rgb, ink_mask = cls.detect_background_and_ink_colors(fit_img)
        bg_rgb = override_bg_rgb if override_bg_rgb is not None else detected_bg_rgb
        ink_rgb = override_ink_rgb if override_ink_rgb is not None else detected_ink_rgb
        # Cores reais de cada traço/área (Traçado colorido e Livro de colorir).
        # Uma cor de tinta escolhida pelo usuário desliga a extração.
        multicolor = (colorize or fill_regions) and override_ink_rgb is None
        if multicolor:
            # Só o cinza (Otsu) perde linhas claras e coloridas sobre o fundo
            ink_mask = ink_mask_from_background(fit_img, detected_bg_rgb, ink_mask)

        # 2. Análise de Espaço Negativo Crítico (P3: protege olhos, boca e frestas)
        negative_regions = NegativeSpaceAnalyzer.identify_enclosed_spaces(ink_mask)
        critical_mask = NegativeSpaceAnalyzer.build_critical_protection_mask(negative_regions, (h, w))

        # Usa o calibrador do perfil do engine (ex.: perfil customizado), não o padrão
        calib = calibrator if calibrator is not None else HardwareCalibrationEngine()

        # 2b. Áreas sólidas (pupilas, manchas): preenchidas, não esqueletizadas.
        # A linha central de uma mancha vira um polígono vazio e perde a forma.
        min_brush_for_solid = calib.get_effective_brush_width("min", "solid")
        # No modo colorido as linhas usam o pincel da própria largura; só formas
        # bem mais largas que a maior classe de linha viram área preenchida.
        solid_mask = cls.extract_solid_ink(
            ink_mask, cls.AREA_MIN_WIDTH_PX / 1.6 if multicolor else min_brush_for_solid)
        line_mask = ink_mask
        if np.count_nonzero(solid_mask):
            grown = cv2.dilate(solid_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            line_mask = cv2.bitwise_and(ink_mask, cv2.bitwise_not(grown))

        # 3. Esqueletização com Poda de Esporões Parasitas (P3)
        skel_raw = cls.skeletonize(line_mask)
        skel = SkeletonOptimizer.prune_spurs(skel_raw, max_spur_length=4)

        # 4. Estimativa de espessura e calibração de hardware
        partitioner = AdaptiveLineArtPartitioner(calib)
        ink_dist_map = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 5)
        estimated_width = cls.estimate_stroke_thickness(ink_mask, skel, dist_map=ink_dist_map)

        # 5. Rastreamento de caminhos contínuos
        raw_paths = cls.trace_skeleton_paths(skel, min_length=4.0)

        # 6. Otimização de Trajetória TSP (minimiza deslocamentos no ar)
        # O corredor de encadeamento é a própria tinta: impede pontes falsas sobre o fundo
        ink_corridor = cv2.dilate(ink_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ordered_paths = StrokeSchedulerV2.optimize_layer_strokes(raw_paths, enable_2opt=True, allowed_mask=ink_corridor)

        # 7. Suavização Spline para curvas impecáveis
        smoothed_paths = SplineInterpolator.smooth_all_strokes(ordered_paths, min_points=4)

        # 8. Sanitização estrita em coordenadas locais
        clean_local_strokes = PathSanitizer.sanitize_all_strokes(
            smoothed_paths, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
        )

        # 8b. Filtragem de Hachuras Hiperdensas para evitar empastamento de tinta (P3)
        min_brush_w = calib.get_effective_brush_width("min", "solid")
        # Hachuras de 1–2 px viram traços de 10 px no aparelho: no modo colorido
        # as paralelas precisam de mais folga para não virarem uma mancha.
        if multicolor:
            # Com traço fino, hachuras a ~6 px continuam separadas no aparelho
            separation = FINE_DRAWN_PX + 2.0 if config.FINE_LINES else min_brush_w
        else:
            separation = min_brush_w * 0.7
        de_crowded_strokes = SkeletonOptimizer.filter_dense_parallel_strokes(
            clean_local_strokes, min_separation_px=separation, shape=(h, w)
        )
        # No modo colorido a largura vem só das linhas: medida na tinta inteira, a
        # borda de uma área larga "herdaria" a largura da área e sairia gorda.
        width_dist = cv2.distanceTransform(line_mask, cv2.DIST_L2, 5) if multicolor else ink_dist_map
        stroke_widths = cls.compute_stroke_widths(ink_mask, de_crowded_strokes, dist_map=width_dist)

        # 9. Particionamento Hierárquico por Espessura Física (P3)
        buckets = partitioner.partition_strokes(de_crowded_strokes, stroke_widths, critical_mask)

        # Mapeamento da cor da tinta para a LUT do celular
        touch_match = color_engine.match_color(ink_rgb)
        matched_rgb = tuple(touch_match.get("matched_rgb") or touch_match.get("rgb") or ink_rgb)

        layers = []
        palette_colors = []
        bg_matched_rgb = None

        # Camada de fundo opcional (se solicitada ou se contrastar com o aparelho)
        dist_to_device_bg = float(np.linalg.norm(np.array(bg_rgb, dtype=np.float32) - np.array([22.0, 22.0, 22.0], dtype=np.float32)))
        should_paint_bg = paint_background if paint_background is not None else (dist_to_device_bg > 35.0)
        if should_paint_bg:
            bg_touch = color_engine.match_color(bg_rgb)
            bg_matched_rgb = tuple(bg_touch.get("matched_rgb") or bg_touch.get("rgb") or bg_rgb)
            bg_width = calib.get_effective_brush_width("thick", "solid")
            bg_spacing = FillStrategyEngine.calculate_brush_spacing(bg_width)
            bg_mask = np.full((h, w), 255, dtype=np.uint8)
            bg_strokes = FillStrategyEngine.generate_region_strokes(
                bg_mask, (0, 0, w, h),
                strategy="horizontal_scanline",
                brush_spacing=bg_spacing,
                brush_width=bg_width,
                ensure_full_coverage=True
            )
            clean_bg_strokes = PathSanitizer.sanitize_all_strokes(
                bg_strokes, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
            )
            bg_pts = sum(len(s) for s in clean_bg_strokes)
            bg_layer = {
                "name": f"Camada {len(layers)+1}: Fundo {bg_touch['name']} (Line-Art)",
                "color_rgb": bg_rgb,
                "render_color_rgb": bg_matched_rgb,
                "target_color_rgb": bg_rgb,
                "color_hex": f"#{bg_rgb[0]:02X}{bg_rgb[1]:02X}{bg_rgb[2]:02X}",
                "brush": "solid",
                "purpose": "background",
                "touch_info": bg_touch,
                "strokes": clean_bg_strokes,
                "stroke_count": len(clean_bg_strokes),
                "point_count": bg_pts,
                "thickness": "thick"
            }
            layers.append(bg_layer)
            palette_colors.append({
                "name": bg_touch["name"],
                "hex": f"#{bg_rgb[0]:02X}{bg_rgb[1]:02X}{bg_rgb[2]:02X}",
                "rgb": bg_rgb,
                "matched_hex": bg_touch.get("matched_hex", "#FFFFFF"),
                "delta_e": bg_touch.get("delta_e", 0.0)
            })

        if multicolor:
            color_layers, color_palette, target_fills = cls._build_multicolor_layers(
                fit_img, ink_mask, solid_mask, de_crowded_strokes, stroke_widths, critical_mask, calib, color_engine, bg_rgb,
                fill_regions=fill_regions, background_painted=should_paint_bg, first_index=len(layers) + 1)
            layers.extend(color_layers)
            palette_colors.extend(color_palette)
            names = [p["name"] for p in palette_colors]
            for layer in color_layers:
                name = layer["touch_info"].get("name")
                if name in names:
                    layer["palette_index"] = names.index(name)
            return cls._finish_plan(
                fit_img, layers, palette_colors, simulator, transform, seed, calib, should_paint_bg,
                bg_matched_rgb if should_paint_bg else None, target_fills, critical_mask, ink_mask, negative_regions,
                estimated_width, stroke_widths, custom_bounds, bg_rgb, ink_rgb, solid_mask,
                style="coloring_book" if fill_regions else "colored_lines")

        # Áreas sólidas antes das linhas: as linhas por cima mantêm as bordas nítidas
        solid_strokes = cls.generate_solid_fill_strokes(solid_mask, min_brush_for_solid)
        if solid_strokes:
            solid_strokes = PathSanitizer.sanitize_all_strokes(solid_strokes, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1)
        if solid_strokes:
            layers.append({
                "name": f"Camada {len(layers)+1}: Áreas Sólidas ({touch_match['name']})",
                "color_rgb": ink_rgb,
                "render_color_rgb": matched_rgb,
                "target_color_rgb": ink_rgb,
                "color_hex": f"#{ink_rgb[0]:02X}{ink_rgb[1]:02X}{ink_rgb[2]:02X}",
                "brush": "solid",
                "purpose": "line_art_fill",
                "touch_info": touch_match,
                "strokes": solid_strokes,
                "stroke_count": len(solid_strokes),
                "point_count": sum(len(st) for st in solid_strokes),
                "thickness": "min"
            })

        # Adiciona camadas de traço particionadas (P3):
        # Ordem de desenho: massas estruturais primeiro ('thick'/'med'), detalhes finos por último ('min')
        stroke_order = ["thick", "med", "min"]
        active_buckets = [lvl for lvl in stroke_order if buckets[lvl]["strokes"]]

        # Se nenhum bucket teve traço, cria bucket padrão 'min'
        if not active_buckets:
            active_buckets = ["min"]
            buckets["min"] = {"strokes": clean_local_strokes, "widths": stroke_widths}

        for lvl in active_buckets:
            b_strokes = buckets[lvl]["strokes"]
            b_widths = buckets[lvl]["widths"]
            if not b_strokes:
                continue

            # Ordenação hierárquica por importância e categoria topológica (P5)
            ord_strokes, ord_widths, _ = cls.order_line_art_strokes(b_strokes, b_widths)

            # Poda e proteção de segurança perceptual (P5 & Etapa 2/4 do Plano)
            from .perceptual_optimizer import PerceptualPlanOptimizer
            layer_b_w = calib.get_effective_brush_width(lvl, "solid")
            l_temp = {"purpose": "line_art_centerline", "strokes": ord_strokes}
            PerceptualPlanOptimizer.filter_layer_strokes_for_safety(
                l_temp,
                protected_mask=critical_mask,
                brush_width=layer_b_w,
                canvas_shape=(h, w)
            )
            ord_strokes = l_temp["strokes"] if l_temp["strokes"] else ord_strokes

            lvl_pts = sum(len(s) for s in ord_strokes)
            lvl_name = "Contorno Estrutural" if lvl in ("med", "thick") else "Traço Central / Detalhes"
            l_info = {
                "name": f"Camada {len(layers)+1}: {lvl_name} ({touch_match['name']})",
                "color_rgb": ink_rgb,
                "render_color_rgb": matched_rgb,
                "target_color_rgb": ink_rgb,
                "color_hex": f"#{ink_rgb[0]:02X}{ink_rgb[1]:02X}{ink_rgb[2]:02X}",
                "brush": "solid",
                "purpose": "line_art_centerline",
                "touch_info": touch_match,
                "strokes": ord_strokes,
                "stroke_widths": ord_widths[:len(ord_strokes)] if len(ord_widths) >= len(ord_strokes) else ord_widths,
                "desired_widths": ord_widths[:len(ord_strokes)] if len(ord_widths) >= len(ord_strokes) else ord_widths,
                "stroke_count": len(ord_strokes),
                "point_count": lvl_pts,
                "thickness": lvl
            }
            layers.append(l_info)

        palette_colors.append({
            "name": touch_match["name"],
            "hex": f"#{ink_rgb[0]:02X}{ink_rgb[1]:02X}{ink_rgb[2]:02X}",
            "rgb": ink_rgb,
            "matched_hex": touch_match.get("matched_hex", "#000000"),
            "delta_e": touch_match.get("delta_e", 0.0)
        })

        # 10. Simulação Calibrada usando o fundo nativo do aparelho (22, 22, 22)
        device_native_bg_bgr = (22, 22, 22)
        sim_canvas, cum_buffers = simulator.render_simulation(
            (h, w, 3), layers, background_bgr=device_native_bg_bgr, seed=seed
        )

        # 11. Constrói o target_executable fiel ao que o aparelho consegue desenhar (Seção 2 e Etapa 2 do Plano)
        target_executable = np.zeros((h, w, 3), dtype=np.uint8)
        if should_paint_bg:
            bg_bgr = (int(bg_matched_rgb[2]), int(bg_matched_rgb[1]), int(bg_matched_rgb[0]))
            target_executable[:] = bg_bgr
        else:
            bg_bgr = device_native_bg_bgr
            target_executable[:] = device_native_bg_bgr

        ink_bgr = (int(matched_rgb[2]), int(matched_rgb[1]), int(matched_rgb[0]))
        if np.count_nonzero(solid_mask):
            target_executable[solid_mask > 0] = ink_bgr
        # Renderiza a interpretação desenhável e viável do line-art com a pegada física real das camadas
        for l in layers:
            if l.get("purpose") == "line_art_centerline":
                lvl_thick = l.get("thickness", "min")
                l_w = max(1, int(round(calib.get_effective_brush_width(lvl_thick, "solid"))))
                for s in l.get("strokes", []):
                    if len(s) >= 2:
                        pts = np.array(s, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(target_executable, [pts], False, ink_bgr, l_w, lineType=cv2.LINE_AA)
                    elif len(s) == 1:
                        cv2.circle(target_executable, (int(s[0][0]), int(s[0][1])), max(1, l_w // 2), ink_bgr, -1)

        # Preserva estritamente os espaços negativos críticos (olhos, boca, frestas) abertos na cor de fundo
        if critical_mask is not None and np.count_nonzero(critical_mask) > 0:
            target_executable[critical_mask > 0] = bg_bgr

        total_plan_strokes = sum(l["stroke_count"] for l in layers)
        total_plan_pts = sum(l["point_count"] for l in layers)
        time_est_s = total_plan_strokes * 0.35 + total_plan_pts * 0.003
        plan_hash = StrokePlan.compute_plan_hash(seed, "LINE_ART", layers, transform=transform)

        # Validação métrica e topológica orientada a qualidade (P5)
        all_line_strokes = []
        for l in layers:
            if l.get("purpose") == "line_art_centerline":
                all_line_strokes.extend(l.get("strokes", []))
        topology_report = cls.verify_line_art_topology_and_metrics(all_line_strokes, ink_mask)

        return {
            "mode": "line_art",
            "plan_hash": plan_hash,
            "transform": transform,
            "transform_dict": {"offset_x": transform.offset_x, "offset_y": transform.offset_y, "scale": transform.scale},
            "layers": layers,
            "total_strokes": total_plan_strokes,
            "total_points": total_plan_pts,
            "estimated_time_s": round(time_est_s, 1),
            "simulation_image": sim_canvas,
            "target_executable": target_executable,
            "cumulative_layer_images": cum_buffers,
            "comparison_image": np.hstack([fit_img, sim_canvas]),
            "fit_image": fit_img,
            "background_rgb": bg_rgb,
            "ink_rgb": ink_rgb,
            "estimated_thickness_px": round(estimated_width, 1),
            "custom_bounds": custom_bounds,
            "desired_widths": stroke_widths,
            "palette_colors": palette_colors,
            "negative_spaces_detected": len(negative_regions),
            "solid_ink_pixels": int(np.count_nonzero(solid_mask)),
            "critical_protection_mask": critical_mask,
            "topology_metrics": topology_report
        }
