# -*- coding: utf-8 -*-
"""
Motor de Estrategias de Preenchimento Adaptativo com Path Safety.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..vision.structure_tensor import StructureTensorEngine
from .spline_interpolator import SplineInterpolator

class FillStrategyEngine:
    @staticmethod
    def calculate_brush_spacing(brush_width: float, overlap_ratio: float = 0.85) -> int:
        """
        Calcula o espaçamento ótimo garantindo que o espaço entre pinceladas
        seja estritamente menor que a largura efetiva do pincel (overlap_ratio < 1.0, default 0.85).
        Impede frestas e espaços vazios indesejados entre passagens adjacentes.
        """
        return max(1, int(round(float(brush_width) * overlap_ratio)))

    @classmethod
    def compute_safe_mask(cls, mask: np.ndarray, brush_radius: float, safety_margin: float = 1.0,
                          bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Path Safety Avançado com Erosão de Raio de Pincel (Seção 13 do Plano de Fidelidade Máxima).
        Garante que as bordas do pincel com espessura física não transbordem fora do polígono.
        Se a região colapsar após erosão severa, aplica fallback progressivo para raio menor.
        Otimizado com ROI BBox para evitar erosão em frame completo desnecessária.
        """
        kernel_radius = max(1, int(round(brush_radius + safety_margin)))
        ksize = kernel_radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        if bbox is not None:
            bx, by, bw, bh = bbox
            h, w = mask.shape[:2]
            pad = kernel_radius + 2
            x0 = max(0, bx - pad)
            y0 = max(0, by - pad)
            x1 = min(w, bx + bw + pad)
            y1 = min(h, by + bh + pad)

            sub = mask[y0:y1, x0:x1]
            eroded_sub = cv2.erode(sub, kernel)

            if np.count_nonzero(eroded_sub) < 15:
                fallback_radius = max(1, kernel_radius // 2)
                fb_ksize = fallback_radius * 2 + 1
                fb_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fb_ksize, fb_ksize))
                eroded_fb = cv2.erode(sub, fb_kernel)
                if np.count_nonzero(eroded_fb) >= 10:
                    out = mask.copy()
                    out[y0:y1, x0:x1] = eroded_fb
                    return out
                return mask

            out = mask.copy()
            out[y0:y1, x0:x1] = eroded_sub
            return out

        eroded = cv2.erode(mask, kernel)

        # Fallback adaptativo: se a erosão removeu quase toda a área, tenta erosão com metade do raio
        if np.count_nonzero(eroded) < 15:
            fallback_radius = max(1, kernel_radius // 2)
            fb_ksize = fallback_radius * 2 + 1
            fb_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fb_ksize, fb_ksize))
            eroded_fb = cv2.erode(mask, fb_kernel)
            if np.count_nonzero(eroded_fb) >= 10:
                return eroded_fb
            return mask  # Fallback final: usa máscara original

        return eroded

    @classmethod
    def detect_and_fill_gaps(cls, mask: np.ndarray,
                             strokes: List[List[Tuple[int, int]]],
                             brush_width: float = 3.2,
                             bbox: Optional[Tuple[int, int, int, int]] = None,
                             min_gap_area: int = 12) -> List[List[Tuple[int, int]]]:
        """
        Simula a cobertura acumulada das pinceladas sobre a máscara da região,
        detecta espaços vazios/frestas residuais e sintetiza traços complementares de fechamento
        respeitando estritamente as bordas originais da região.
        """
        h, w = mask.shape
        if bbox is not None:
            bx, by, bw, bh = bbox
            pad = max(2, int(round(brush_width * 1.5)))
            x0 = max(0, bx - pad)
            y0 = max(0, by - pad)
            x1 = min(w, bx + bw + pad)
            y1 = min(h, by + bh + pad)

            sub_mask = mask[y0:y1, x0:x1]
            if np.count_nonzero(sub_mask) == 0:
                return []

            sub_coverage = np.zeros_like(sub_mask)
            eff_w = max(1, int(round(brush_width)))

            offset_polys = []
            for s in strokes:
                if len(s) >= 2:
                    pts = np.array([[px - x0, py - y0] for px, py in s], dtype=np.int32).reshape((-1, 1, 2))
                    offset_polys.append(pts)

            if offset_polys:
                cv2.polylines(sub_coverage, offset_polys, False, 255, eff_w, cv2.LINE_8)

            sub_gaps = cv2.bitwise_and(sub_mask, cv2.bitwise_not(sub_coverage))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            gaps_cleaned = cv2.morphologyEx(sub_gaps, cv2.MORPH_OPEN, kernel)
            
            # Limiar adaptativo à física do pincel: evita repassar e criar manchas em micro-frestas que a tinta já cobre
            effective_min_gap = max(min_gap_area, int(round((brush_width ** 2) * 0.40)))
            if np.count_nonzero(gaps_cleaned) < effective_min_gap:
                return []

            num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(gaps_cleaned, connectivity=8)
            gap_strokes: List[List[Tuple[int, int]]] = []
            spacing = max(1, int(round(brush_width * 0.70)))

            for comp_id in range(1, num_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area < effective_min_gap:
                    continue

                rx = stats[comp_id, cv2.CC_STAT_LEFT]
                ry = stats[comp_id, cv2.CC_STAT_TOP]
                rw_c = stats[comp_id, cv2.CC_STAT_WIDTH]
                rh_c = stats[comp_id, cv2.CC_STAT_HEIGHT]
                comp_mask = (comp_labels == comp_id).astype(np.uint8) * 255

                c_strokes = cls._vertical_scanline(comp_mask, (rx, ry, rw_c, rh_c), spacing=spacing)
                for cs in c_strokes:
                    if len(cs) >= 2:
                        global_cs = [(px + x0, py + y0) for px, py in cs]
                        if cls._is_segment_inside_mask(mask, global_cs[0], global_cs[-1]):
                            gap_strokes.append(global_cs)
            return gap_strokes

        # Fallback sem bbox
        coverage = np.zeros((h, w), dtype=np.uint8)
        eff_w = max(1, int(round(brush_width)))

        for s in strokes:
            if len(s) >= 2:
                pts = np.array(s, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(coverage, [pts], False, 255, eff_w, cv2.LINE_8)

        # Frestas não cobertas dentro da máscara da região
        gaps = cv2.bitwise_and(mask, cv2.bitwise_not(coverage))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        gaps_cleaned = cv2.morphologyEx(gaps, cv2.MORPH_OPEN, kernel)

        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(gaps_cleaned, connectivity=8)
        gap_strokes: List[List[Tuple[int, int]]] = []

        spacing = max(1, int(round(brush_width * 0.70)))

        for comp_id in range(1, num_labels):
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < min_gap_area:
                continue

            rx = stats[comp_id, cv2.CC_STAT_LEFT]
            ry = stats[comp_id, cv2.CC_STAT_TOP]
            rw = stats[comp_id, cv2.CC_STAT_WIDTH]
            rh = stats[comp_id, cv2.CC_STAT_HEIGHT]
            comp_mask = (comp_labels == comp_id).astype(np.uint8) * 255

            # Preenchimento complementar transversal dos vazios
            c_strokes = cls._vertical_scanline(comp_mask, (rx, ry, rw, rh), spacing=spacing)
            for cs in c_strokes:
                # Valida que o traço complementar está estritamente dentro da máscara geral
                if len(cs) >= 2 and cls._is_segment_inside_mask(mask, cs[0], cs[-1]):
                    gap_strokes.append(cs)

        return gap_strokes

    @classmethod
    def generate_region_strokes(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                                strategy: str = "horizontal_scanline",
                                brush_spacing: int = 6,
                                texture_angle: float = 0.0,
                                brush_width: float = None,
                                apply_safe_mask: bool = True,
                                image: Optional[np.ndarray] = None,
                                flow_field: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                enable_spline: bool = True,
                                ensure_full_coverage: bool = True) -> List[List[Tuple[int, int]]]:
        """
        Gera traços de preenchimento para uma região poligonal segundo a estratégia escolhida.
        Garante que o espaçamento seja estritamente menor que a largura física do pincel,
        evitando frestas. Se ensure_full_coverage=True, identifica vazios remanescentes e completa a cobertura.
        """
        # Garante que o espaçamento nunca seja maior que a largura do pincel
        if brush_width is not None:
            max_safe_spacing = max(1, int(round(float(brush_width) * 0.75)))
            actual_spacing = min(brush_spacing, max_safe_spacing) if brush_spacing is not None else max_safe_spacing
        else:
            actual_spacing = max(2, int(brush_spacing)) if brush_spacing is not None else 5

        work_mask = mask
        if apply_safe_mask and brush_width is not None and brush_width > 3.0:
            work_mask = cls.compute_safe_mask(mask, brush_radius=brush_width / 2.0, bbox=bbox)

        raw_strokes: List[List[Tuple[int, int]]] = []
        if strategy == "streamline_tensor":
            raw_strokes = cls._streamline_tensor_fill(
                work_mask, bbox, spacing=actual_spacing, image=image, flow_field=flow_field, enable_spline=enable_spline
            )
        elif strategy == "chiaroscuro_cross":
            raw_strokes = cls._chiaroscuro_cross_fill(
                work_mask, bbox, spacing=actual_spacing, base_angle=texture_angle, enable_spline=enable_spline
            )
        elif strategy == "vertical_scanline":
            raw_strokes = cls._vertical_scanline(work_mask, bbox, spacing=actual_spacing)
        elif strategy == "angled_hatching" and abs(texture_angle) > 5.0:
            raw_strokes = cls._angled_hatching(work_mask, bbox, angle_deg=texture_angle, spacing=actual_spacing)
        elif strategy == "edge_aware_fill":
            raw_strokes = cls._edge_aware_fill(work_mask, bbox, spacing=actual_spacing)
        elif strategy == "contour_parallel":
            raw_strokes = cls._contour_parallel_fill(work_mask, bbox, spacing=actual_spacing)
        else:
            raw_strokes = cls._horizontal_scanline(work_mask, bbox, spacing=actual_spacing)

        # Identificação e preenchimento de frestas/espaços vazios residuais confinados à máscara segura
        if ensure_full_coverage and raw_strokes:
            b_w = brush_width if brush_width is not None else 3.2
            gap_strokes = cls.detect_and_fill_gaps(work_mask, raw_strokes, brush_width=b_w, bbox=bbox)
            if gap_strokes:
                raw_strokes.extend(gap_strokes)

        if enable_spline and raw_strokes and strategy not in ("streamline_tensor", "chiaroscuro_cross"):
            return SplineInterpolator.smooth_all_strokes(raw_strokes, mask=work_mask)

        return raw_strokes

    @classmethod
    def _streamline_tensor_fill(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                                spacing: int = 6,
                                image: Optional[np.ndarray] = None,
                                flow_field: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                enable_spline: bool = True) -> List[List[Tuple[int, int]]]:
        """
        Preenchimento curvilíneo 3D através de linhas de fluxo geradas pelo Tensor de Estrutura.
        Se a imagem ou flow_field não estiverem disponíveis, utiliza a orientação geométrica da máscara.
        """
        if flow_field is not None:
            flow_dx, flow_dy = flow_field
        elif image is not None:
            flow_dx, flow_dy, _ = StructureTensorEngine.compute_orientation_field(image)
        else:
            flow_dx, flow_dy, _ = StructureTensorEngine.compute_orientation_field(mask)

        raw_lines = StructureTensorEngine.generate_curvilinear_streamlines(
            mask, flow_dx, flow_dy, bbox=bbox, spacing=spacing, step_size=3.0, min_length=8.0
        )

        if not raw_lines:
            return cls._horizontal_scanline(mask, bbox, spacing=spacing)

        if enable_spline:
            return SplineInterpolator.smooth_all_strokes(raw_lines, mask=mask)
        return raw_lines

    @classmethod
    def _chiaroscuro_cross_fill(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                                spacing: int = 6,
                                base_angle: float = 30.0,
                                enable_spline: bool = True) -> List[List[Tuple[int, int]]]:
        """
        Preenchimento Chiaroscuro com Hachura Cruzada Multi-Pass (Pilar 4).
        Gera dois conjuntos de linhas hachuradas sobrepostas com ângulo cruzado (~75 deg).
        """
        b_angle = base_angle if abs(base_angle) > 5.0 else 35.0
        pass1 = cls._angled_hatching(mask, bbox, angle_deg=b_angle, spacing=spacing)
        cross_angle = (b_angle + 75.0) % 180.0
        pass2 = cls._angled_hatching(mask, bbox, angle_deg=cross_angle, spacing=int(round(spacing * 1.3)))

        combined = []
        combined.extend(pass1)
        combined.extend(pass2)

        if enable_spline:
            return SplineInterpolator.smooth_all_strokes(combined, mask=mask)
        return combined

    @staticmethod
    def _is_segment_inside_mask(mask: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        """
        Path Safety Ultrarrápido: Slicing em C para traços ortogonais (horizontal/vertical)
        e algoritmo de Bresenham inteiro para traços diagonais. Zero alocações e zero linspace.
        """
        h, w = mask.shape
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        # Caso trivial: ponto único
        if x1 == x2 and y1 == y2:
            return 0 <= x1 < w and 0 <= y1 < h and mask[y1, x1] > 0

        # Otimização 1: Segmentos horizontais (90% dos casos de scanline)
        if y1 == y2:
            if y1 < 0 or y1 >= h:
                return False
            min_x = min(x1, x2)
            max_x = max(x1, x2)
            if min_x < 0 or max_x >= w:
                return False
            # Slicing contíguo direto em C pelo NumPy
            row_slice = mask[y1, min_x:max_x + 1]
            return not np.any(row_slice == 0)

        # Otimização 2: Segmentos verticais
        if x1 == x2:
            if x1 < 0 or x1 >= w:
                return False
            min_y = min(y1, y2)
            max_y = max(y1, y2)
            if min_y < 0 or max_y >= h:
                return False
            col_slice = mask[min_y:max_y + 1, x1]
            return not np.any(col_slice == 0)

        # Otimização 3: Bresenham inteiro para diagonais arbitrárias
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        cx, cy = x1, y1

        while True:
            if cx < 0 or cx >= w or cy < 0 or cy >= h or mask[cy, cx] == 0:
                return False
            if cx == x2 and cy == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        return True

    @classmethod
    def _get_contiguous_runs(cls, indices: np.ndarray, max_gap: int = 3) -> List[Tuple[int, int]]:
        """Decompõe uma lista de índices ordenados em intervalos contíguos [start, end], fundindo micro-gaps."""
        if len(indices) == 0:
            return []
        diffs = np.diff(indices)
        split_points = np.where(diffs > max_gap)[0] + 1
        runs = np.split(indices, split_points)
        return [(int(r[0]), int(r[-1])) for r in runs if len(r) > 0]

    @classmethod
    def _horizontal_scanline(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                             spacing: int = 6) -> List[List[Tuple[int, int]]]:
        """Preenchimento horizontal com segmentação de runs contíguos, união fluida de bordas e verificação de buracos."""
        x, y, rw, rh = bbox
        strokes = []
        current_stroke = []
        direction = 1

        for cur_y in range(y, y + rh, spacing):
            row = mask[cur_y, x:x + rw]
            white_indices = np.where(row > 0)[0]
            runs = cls._get_contiguous_runs(white_indices, max_gap=3)
            if not runs:
                if len(current_stroke) >= 2:
                    strokes.append(current_stroke)
                    current_stroke = []
                continue

            line_runs = runs if direction == 1 else list(reversed(runs))

            for r_start, r_end in line_runs:
                x_start = x + r_start
                x_end = x + r_end

                if x_start == x_end:
                    continue

                p_in = (x_start, cur_y) if direction == 1 else (x_end, cur_y)
                p_out = (x_end, cur_y) if direction == 1 else (x_start, cur_y)

                # Verifica se o próprio segmento do run está estritamente dentro da máscara
                if not cls._is_segment_inside_mask(mask, p_in, p_out):
                    continue

                if not current_stroke:
                    current_stroke = [p_in, p_out]
                else:
                    # Path Safety: Testa se a conexão entre o traço anterior e este run não cruza buracos
                    if cls._is_segment_inside_mask(mask, current_stroke[-1], p_in):
                        current_stroke.extend([p_in, p_out])
                    else:
                        strokes.append(current_stroke)
                        current_stroke = [p_in, p_out]

            direction *= -1

        if len(current_stroke) >= 2:
            strokes.append(current_stroke)
        return strokes

    @classmethod
    def _vertical_scanline(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                           spacing: int = 6) -> List[List[Tuple[int, int]]]:
        """Preenchimento vertical com segmentação de runs contíguos, união fluida de bordas e verificação de buracos."""
        x, y, rw, rh = bbox
        strokes = []
        current_stroke = []
        direction = 1

        for cur_x in range(x, x + rw, spacing):
            col = mask[y:y + rh, cur_x]
            white_indices = np.where(col > 0)[0]
            runs = cls._get_contiguous_runs(white_indices, max_gap=3)
            if not runs:
                if len(current_stroke) >= 2:
                    strokes.append(current_stroke)
                    current_stroke = []
                continue

            col_runs = runs if direction == 1 else list(reversed(runs))

            for r_start, r_end in col_runs:
                y_start = y + r_start
                y_end = y + r_end

                if y_start == y_end:
                    continue

                p_in = (cur_x, y_start) if direction == 1 else (cur_x, y_end)
                p_out = (cur_x, y_end) if direction == 1 else (cur_x, y_start)

                if not cls._is_segment_inside_mask(mask, p_in, p_out):
                    continue

                if not current_stroke:
                    current_stroke = [p_in, p_out]
                else:
                    if cls._is_segment_inside_mask(mask, current_stroke[-1], p_in):
                        current_stroke.extend([p_in, p_out])
                    else:
                        strokes.append(current_stroke)
                        current_stroke = [p_in, p_out]

            direction *= -1

        if len(current_stroke) >= 2:
            strokes.append(current_stroke)
        return strokes

    @classmethod
    def _angled_hatching(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                         angle_deg: float = 45.0, spacing: int = 6) -> List[List[Tuple[int, int]]]:
        # Preenchimento rotacionado com otimizacao regional por Bounding Box ROI.
        # Atua na ROI compacta com padding.
        h, w = mask.shape
        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return []

        # Padding proporcional à diagonal para conter toda a rotação sem cortar bordas
        diag = int(np.ceil(np.sqrt(bw * bw + bh * bh)))
        pad = max(10, int((diag - min(bw, bh)) / 2) + 8)

        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(w, bx + bw + pad)
        y1 = min(h, by + bh + pad)

        roi = mask[y0:y1, x0:x1]
        roi_h, roi_w = roi.shape
        if roi_h <= 0 or roi_w <= 0 or np.count_nonzero(roi) == 0:
            return []

        center_x = roi_w / 2.0
        center_y = roi_h / 2.0

        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), -angle_deg, 1.0)
        inv_rot_mat = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)

        rotated_roi = cv2.warpAffine(roi, rot_mat, (roi_w, roi_h), flags=cv2.INTER_NEAREST)
        pts_rot = np.where(rotated_roi > 0)
        if len(pts_rot[0]) == 0:
            return cls._horizontal_scanline(mask, bbox, spacing)

        min_y, max_y = int(np.min(pts_rot[0])), int(np.max(pts_rot[0]))
        min_x, max_x = int(np.min(pts_rot[1])), int(np.max(pts_rot[1]))
        rot_bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)

        rot_strokes = cls._horizontal_scanline(rotated_roi, rot_bbox, spacing)

        final_strokes = []
        for stroke in rot_strokes:
            transformed = []
            for px, py in stroke:
                roi_ox = int(round(inv_rot_mat[0, 0] * px + inv_rot_mat[0, 1] * py + inv_rot_mat[0, 2]))
                roi_oy = int(round(inv_rot_mat[1, 0] * px + inv_rot_mat[1, 1] * py + inv_rot_mat[1, 2]))
                orig_x = np.clip(roi_ox + x0, 0, w - 1)
                orig_y = np.clip(roi_oy + y0, 0, h - 1)
                transformed.append((orig_x, orig_y))
            if len(transformed) >= 2:
                final_strokes.append(transformed)

        return final_strokes

    @classmethod
    def _edge_aware_fill(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                         spacing: int = 6) -> List[List[Tuple[int, int]]]:
        # Edge-Aware Fill com Transformada de Distancia acelerado por ROI.
        # Calcula a transformada de distancia estritamente no recorte da ROI.
        h, w = mask.shape
        bx, by, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            return []

        pad = 4
        x0 = max(0, bx - pad)
        y0 = max(0, by - pad)
        x1 = min(w, bx + bw + pad)
        y1 = min(h, by + bh + pad)

        roi = mask[y0:y1, x0:x1]
        dist_map = cv2.distanceTransform(roi, cv2.DIST_L2, 5)

        interior_roi = (dist_map > 10.0).astype(np.uint8) * 255
        border_roi = cv2.bitwise_and(roi, cv2.bitwise_not(interior_roi))

        roi_bbox = (bx - x0, by - y0, bw, bh)
        strokes = []
        if np.count_nonzero(interior_roi) > 50:
            int_strokes = cls._horizontal_scanline(interior_roi, roi_bbox, spacing=spacing + 2)
            for s in int_strokes:
                strokes.append([(pt[0] + x0, pt[1] + y0) for pt in s])

        if np.count_nonzero(border_roi) > 30:
            bor_strokes = cls._horizontal_scanline(border_roi, roi_bbox, spacing=max(3, spacing - 2))
            for s in bor_strokes:
                strokes.append([(pt[0] + x0, pt[1] + y0) for pt in s])

        return strokes

    @classmethod
    def _contour_parallel_fill(cls, mask: np.ndarray, bbox: Tuple[int, int, int, int],
                               spacing: int = 6) -> List[List[Tuple[int, int]]]:
        # Preenchimento por aneis concentricos (Erosoes sucessivas)
        temp_mask = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spacing, spacing))
        strokes = []

        while np.count_nonzero(temp_mask) > 25:
            contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
            for cnt in contours:
                if cv2.arcLength(cnt, True) > 15:
                    approx = cv2.approxPolyDP(cnt, 1.2, True)
                    pts = [tuple(p) for p in approx.reshape(-1, 2).tolist()]
                    if len(pts) >= 3:
                        pts.append(pts[0])  # Fecha anel
                        strokes.append(pts)
            temp_mask = cv2.erode(temp_mask, kernel)

        return strokes
