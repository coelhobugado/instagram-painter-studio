"""
Ilustração chapada: a imagem vira poucas cores planas, cada uma pintada em anéis.

Substitui a antiga "Pintura em cores" (varredura e hachura), que no aparelho
inchava e borrava. Passos:
1. suaviza a imagem preservando bordas (mean shift) para regiões limpas;
2. agrupa as cores (k-means em Lab) e troca cada grupo pela tinta executável
   mais próxima, unindo grupos que caem na mesma tinta;
3. limpa a máscara de rótulos (moda local; manchas pequenas vão para o vizinho);
4. pinta as regiões grandes primeiro, cada uma crescida alguns pixels para não
   sobrar fresta entre vizinhas, com anéis do pincel grosso ao fino;
5. opcionalmente, um contorno escuro fino por cima das bordas mais fortes.
"""
from typing import Any, Dict, List, Optional, Tuple

import re

import cv2
import numpy as np

from .area_fill import fill_region, estimate_paint_time, centerline_strokes
from .color_extraction import to_lab, lab_to_rgb, uniform_color_mask, labels_from_nearest_pure
from .path_sanitizer import PathSanitizer
from ..core.models import CanvasTransform, StrokePlan
from .. import config
from ..device.brush_width import FINE_LEVEL, drawn_width_px

AREA_LEVELS = (("thick", 104.0), ("med", 58.0), ("w28", 28.0), ("min", 10.0))
DEVICE_BG_BGR = (22, 22, 22)


def _mode_filter(labels: np.ndarray, n_labels: int, size: int = 7) -> np.ndarray:
    """Rótulo mais frequente na vizinhança: some com pixels soltos nas bordas."""
    best = np.zeros(labels.shape, np.float32) - 1
    out = labels.copy()
    kernel = np.ones((size, size), np.float32)
    for k in range(n_labels):
        votes = cv2.filter2D((labels == k).astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        better = votes > best
        out[better] = k
        best[better] = votes[better]
    return out


def _absorb_small_regions(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Manchas menores que ``min_area`` assumem o rótulo predominante ao redor."""
    out = labels.copy()
    for k in np.unique(labels):
        mask = (out == k).astype(np.uint8)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                continue
            blob = (comp == i).astype(np.uint8)
            ring = cv2.dilate(blob, np.ones((3, 3), np.uint8)) - blob
            neighbours = out[ring > 0]
            neighbours = neighbours[neighbours != k]
            if len(neighbours):
                out[blob > 0] = np.bincount(neighbours).argmax()
    return out


class FlatIllustrationEngine:
    @staticmethod
    def _dark_details_layers(fit_img, color_engine, number, palette_colors, w, h, min_w):
        lab = to_lab(fit_img)
        L = lab[..., 0]
        local = cv2.GaussianBlur(L, (0, 0), 7)
        detail = ((local - L > 14.0) & (L < 55.0)).astype(np.uint8) * 255
        detail = cv2.morphologyEx(detail, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, comp, stats, _ = cv2.connectedComponentsWithStats(detail)
        keep = np.zeros_like(detail)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= 25:
                keep[comp == i] = 255
        strokes = centerline_strokes(keep, min_length=8)
        strokes = PathSanitizer.sanitize_all_strokes(strokes, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1)
        if not strokes:
            return []
        color = lab_to_rgb(np.median(lab[keep > 0], axis=0))
        t = color_engine.match_color(color)
        rgb = tuple(int(v) for v in (t.get("matched_rgb") or t.get("rgb")))
        names = [p["name"] for p in palette_colors]
        if t["name"] not in names:
            palette_colors.append({"name": t["name"], "hex": "#%02X%02X%02X" % color, "rgb": color,
                                   "matched_hex": t.get("matched_hex", "#000000"), "delta_e": t.get("delta_e", 0.0),
                                   "palette_page": t.get("palette_page"), "is_long_press": t.get("is_long_press", False)})
            names.append(t["name"])
        # Cada detalhe com a largura dele: os estreitos (sobrancelha, cílio, canto
        # da boca) no traço fino, os largos (pupila) no mínimo do trilho.
        from .line_art_engine import LineArtEngine
        dist = cv2.distanceTransform(keep, cv2.DIST_L2, 5)
        by_level: Dict[str, List] = {}
        for s in strokes:
            widths = [2.0 * float(dist[min(h - 1, max(0, int(y))), min(w - 1, max(0, int(x)))]) for x, y in s]
            wclass = LineArtEngine._width_class(float(np.median(widths)), min_w)
            by_level.setdefault(LineArtEngine._line_level(wclass, min_w), []).append(s)
        layers = []
        for lvl in sorted(by_level, key=lambda v: 0 if v == FINE_LEVEL else -1):  # fino por último
            layer_strokes = by_level[lvl]
            layers.append({
                "name": f"Camada {number + len(layers)}: Detalhes{' finos' if lvl == FINE_LEVEL else ''} ({t['name']})",
                "color_rgb": color, "render_color_rgb": rgb, "target_color_rgb": color,
                "color_hex": "#%02X%02X%02X" % color, "brush": "solid", "purpose": "line_art_centerline",
                "touch_info": t, "strokes": layer_strokes, "stroke_count": len(layer_strokes),
                "point_count": sum(len(s) for s in layer_strokes), "thickness": lvl,
                "drawn_width_px": drawn_width_px(lvl, min_w if lvl == "min" else 3.0),
                "palette_index": names.index(t["name"]),
            })
        return layers

    @classmethod
    def generate_plan(cls, fit_img: np.ndarray, color_engine, simulator, transform: CanvasTransform,
                      calibrator, seed: int = 42, num_colors: int = 10, include_outline: bool = False,
                      include_details: bool = True,
                      custom_bounds: Optional[Dict[str, int]] = None, min_region_area: int = 180) -> Dict[str, Any]:
        h, w = fit_img.shape[:2]
        smooth = cv2.pyrMeanShiftFiltering(fit_img, sp=8, sr=18)
        lab = to_lab(smooth).reshape(-1, 3).astype(np.float32)

        # 1. Cores: k-means em Lab com semente fixa (prévia reprodutível).
        #    Treina só nos pixels de miolo (vizinhança de cor uniforme): as transições
        #    entre cabelo e fundo ocupavam grupos inteiros (halos cinza) e faltavam
        #    grupos para cores pequenas e distintas (broche vermelho, boca rosa).
        cv2.setRNGSeed(seed)
        k = max(2, min(int(num_colors), 24))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.3)
        pure_mask = uniform_color_mask(lab.reshape(h, w, 3))
        pure = lab[pure_mask.ravel()]
        if len(pure) < 5 * k:
            pure = lab
        rng = np.random.default_rng(seed)
        sample = pure[rng.choice(len(pure), size=min(len(pure), 60000), replace=False)]
        _, _, centers = cv2.kmeans(sample, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        # Cores pequenas que nenhum grupo representa (a mais de ~18 ΔE) ganham grupo próprio
        far = pure[((pure[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1) > 18.0 ** 2]
        if len(far) >= min_region_area:
            extra = int(min(3, max(1, len(far) // (4 * min_region_area))))
            cv2.setRNGSeed(seed)
            _, _, extra_centers = cv2.kmeans(far.astype(np.float32), extra, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            centers = np.vstack([centers, extra_centers])
            k = len(centers)
        dists = ((lab[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dists.argmin(axis=1).reshape(h, w).astype(np.int32)
        labels = labels_from_nearest_pure(labels, pure_mask)

        # 2. Cada grupo vira uma tinta executável; grupos na mesma tinta se unem
        touches, executable = [], []
        remap = np.arange(k)
        for i, c in enumerate(centers):
            t = color_engine.match_color(lab_to_rgb(c))
            rgb = tuple(int(v) for v in (t.get("matched_rgb") or t.get("rgb")))
            same = next((j for j, e in enumerate(executable) if np.linalg.norm(np.subtract(e, rgb)) < 12), None)
            if same is None:
                touches.append((t, lab_to_rgb(c)))
                executable.append(rgb)
                remap[i] = len(executable) - 1
            else:
                remap[i] = same
        labels = remap[labels]
        n_colors = len(executable)

        # 3. Limpeza: moda local e absorção de manchas pequenas
        labels = _mode_filter(labels, n_colors, 7)
        labels = _absorb_small_regions(labels, min_region_area)

        # 4. Regiões: grandes primeiro; cada uma cresce 4 px sobre as vizinhas
        min_w = calibrator.get_effective_brush_width("min", "solid")
        regions = []
        for c in range(n_colors):
            n, comp, stats, _ = cv2.connectedComponentsWithStats((labels == c).astype(np.uint8), connectivity=4)
            for i in range(1, n):
                regions.append((int(stats[i, cv2.CC_STAT_AREA]), c, (comp == i).astype(np.uint8) * 255))
        regions.sort(key=lambda r: -r[0])
        grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        per_layer: Dict[Tuple[str, int], List] = {}
        order: List[Tuple[str, int]] = []
        for _area, c, mask in regions:
            grown = cv2.dilate(mask, grow)
            for lvl, strokes in fill_region(grown, AREA_LEVELS, thin_level=("min", min_w)).items():
                key = (lvl, c)
                if key not in per_layer:
                    per_layer[key] = []
                    order.append(key)
                per_layer[key].extend(strokes)

        width = {lvl: wpx for lvl, wpx in AREA_LEVELS}
        first_seen = {key: i for i, key in enumerate(order)}
        order.sort(key=lambda key: (-width[key[0]], first_seen[key]))
        layers = []
        palette_index = {}
        palette_colors = []
        for c in sorted({key[1] for key in order}, key=lambda c: -int(np.count_nonzero(labels == c))):
            t, target_rgb = touches[c]
            palette_index[c] = len(palette_colors)
            palette_colors.append({
                "name": t["name"], "hex": "#%02X%02X%02X" % target_rgb, "rgb": target_rgb,
                "matched_hex": t.get("matched_hex", "#000000"), "delta_e": t.get("delta_e", 0.0),
                "palette_page": t.get("palette_page"), "is_long_press": t.get("is_long_press", False),
            })
        for lvl, c in order:
            strokes = PathSanitizer.sanitize_all_strokes(per_layer[(lvl, c)], x_min=0, x_max=w - 1, y_min=0, y_max=h - 1)
            if not strokes:
                continue
            t, target_rgb = touches[c]
            layers.append({
                "name": f"Camada {len(layers) + 1}: Áreas ({t['name']})",
                "color_rgb": target_rgb, "render_color_rgb": executable[c], "target_color_rgb": target_rgb,
                "color_hex": "#%02X%02X%02X" % target_rgb, "brush": "solid", "purpose": "line_art_fill",
                "touch_info": t, "strokes": strokes, "stroke_count": len(strokes),
                "point_count": sum(len(s) for s in strokes), "thickness": lvl, "palette_index": palette_index[c],
            })

        # 5. Contorno escuro opcional nas bordas de maior contraste
        if include_outline:
            gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 140)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
            outline = centerline_strokes(edges, min_length=12)
            darkest = min(range(n_colors), key=lambda c: sum(executable[c]))
            t, target_rgb = touches[darkest]
            outline = PathSanitizer.sanitize_all_strokes(outline, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1)
            if outline:
                # Bordas de 1 px: traço fino (com ele desligado, o mínimo do trilho)
                outline_level = FINE_LEVEL if config.FINE_LINES else "min"
                layers.append({
                    "name": f"Camada {len(layers) + 1}: Contorno ({t['name']})",
                    "color_rgb": target_rgb, "render_color_rgb": executable[darkest], "target_color_rgb": target_rgb,
                    "color_hex": "#%02X%02X%02X" % target_rgb, "brush": "solid", "purpose": "line_art_centerline",
                    "touch_info": t, "strokes": outline, "stroke_count": len(outline),
                    "point_count": sum(len(s) for s in outline), "thickness": outline_level,
                    "drawn_width_px": drawn_width_px(outline_level, min_w if outline_level == "min" else 3.0),
                    "palette_index": palette_index.get(darkest, 0),
                })

        # 6. Detalhes escuros finos (olhos, sobrancelhas, boca): somem na redução de
        #    cores porque são estreitos; viram traço fino por cima das áreas.
        if include_details:
            layers.extend(cls._dark_details_layers(fit_img, color_engine, len(layers) + 1, palette_colors, w, h,
                                                   min_w))

        # Traço fino por último: entrar nele custa a troca neon -> sólida no aparelho,
        # então todas as camadas finas ficam juntas no fim (ordem estável no resto).
        layers.sort(key=lambda l: l.get("thickness") == FINE_LEVEL)
        for i, layer in enumerate(layers):
            layer["name"] = re.sub(r"^Camada \d+", f"Camada {i + 1}", layer["name"])
        sim_canvas, cum_buffers = simulator.render_simulation((h, w, 3), layers, background_bgr=DEVICE_BG_BGR, seed=seed)
        target_executable = np.zeros((h, w, 3), np.uint8)
        for c in range(n_colors):
            r, g, b = executable[c]
            target_executable[labels == c] = (b, g, r)
        return {
            "mode": "flat_illustration",
            "style": "flat_illustration",
            "plan_hash": StrokePlan.compute_plan_hash(seed, "FLAT", layers, transform=transform),
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
            "custom_bounds": custom_bounds,
            "palette_colors": palette_colors,
            "num_colors_requested": k,
        }
