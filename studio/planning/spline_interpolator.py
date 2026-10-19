# -*- coding: utf-8 -*-
"""
Interpolador de Splines Catmull-Rom e Bézier C¹ de Alta Performance com Path Safety.
Implementa o Pilar 3 do Plano Mestre de Maturidade 100% com Otimização Algorítmica.

Transforma trajetórias poligonais em curvas suaves contínuas sem inflar nós redundantes
e com validação ultrarrápida contra a máscara binária.
"""
import math
import numpy as np
from typing import List, Tuple, Optional

class SplineInterpolator:
    """
    Interpolador de curvas caligráficas com parametrização centrípeta Catmull-Rom de alta velocidade.
    Evita auto-interseções, quinas agudas e preserva integridade de fronteiras.
    """

    @classmethod
    def catmull_rom_spline(cls, points: List[Tuple[int, int]],
                           num_samples_per_segment: int = 3,
                           alpha: float = 0.5) -> List[Tuple[int, int]]:
        """
        Interpola uma sequência de pontos usando Spline Catmull-Rom centrípeta (alpha=0.5).
        Otimizado para velocidade usando operações escalares em math.hypot e math.sqrt.
        """
        if not points or len(points) < 2:
            return points

        if len(points) == 2:
            return list(points)

        n = len(points)
        p_first = points[0]
        p_second = points[1]
        p_penult = points[-2]
        p_last = points[-1]

        # Pontos fantasmas para continuidade nas extremidades
        p_start_ghost = (2.0 * p_first[0] - p_second[0], 2.0 * p_first[1] - p_second[1])
        p_end_ghost = (2.0 * p_last[0] - p_penult[0], 2.0 * p_last[1] - p_penult[1])

        # Lista estendida de nós float
        ext_pts = [p_start_ghost] + [(float(px), float(py)) for px, py in points] + [p_end_ghost]

        smooth_points: List[Tuple[int, int]] = []

        def get_t(t_curr: float, p_curr: Tuple[float, float], p_next: Tuple[float, float]) -> float:
            d = math.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
            return t_curr + (math.sqrt(d) if d > 1e-4 else 1e-3)

        for i in range(1, n):
            p0 = ext_pts[i - 1]
            p1 = ext_pts[i]
            p2 = ext_pts[i + 1]
            p3 = ext_pts[i + 2]

            seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if seg_len < 1.0:
                continue

            # Se o segmento for muito curto (< 5px), o traço já é fisicamente contínuo
            if seg_len < 5.0:
                samples = 1
            else:
                samples = min(num_samples_per_segment, max(2, int(seg_len / 4.0)))

            t0 = 0.0
            t1 = get_t(t0, p0, p1)
            t2 = get_t(t1, p1, p2)
            t3 = get_t(t2, p2, p3)

            inv_t10 = 1.0 / (t1 - t0)
            inv_t21 = 1.0 / (t2 - t1)
            inv_t32 = 1.0 / (t3 - t2)
            inv_t20 = 1.0 / (t2 - t0)
            inv_t31 = 1.0 / (t3 - t1)

            step_dt = (t2 - t1) / float(samples)

            for step in range(samples):
                t = t1 + step * step_dt

                # Barry-Goldman simplificado
                w1_0 = (t1 - t) * inv_t10
                w1_1 = (t - t0) * inv_t10
                a1_x = w1_0 * p0[0] + w1_1 * p1[0]
                a1_y = w1_0 * p0[1] + w1_1 * p1[1]

                w2_1 = (t2 - t) * inv_t21
                w2_2 = (t - t1) * inv_t21
                a2_x = w2_1 * p1[0] + w2_2 * p2[0]
                a2_y = w2_1 * p1[1] + w2_2 * p2[1]

                w3_2 = (t3 - t) * inv_t32
                w3_3 = (t - t2) * inv_t32
                a3_x = w3_2 * p2[0] + w3_3 * p3[0]
                a3_y = w3_2 * p2[1] + w3_3 * p3[1]

                wb1_0 = (t2 - t) * inv_t20
                wb1_1 = (t - t0) * inv_t20
                b1_x = wb1_0 * a1_x + wb1_1 * a2_x
                b1_y = wb1_0 * a1_y + wb1_1 * a2_y

                wb2_1 = (t3 - t) * inv_t31
                wb2_2 = (t - t1) * inv_t31
                b2_x = wb2_1 * a2_x + wb2_2 * a3_x
                b2_y = wb2_1 * a2_y + wb2_2 * a3_y

                c_x = int(round(w2_1 * b1_x + w2_2 * b2_x))
                c_y = int(round(w2_1 * b1_y + w2_2 * b2_y))

                pt_tuple = (c_x, c_y)
                if not smooth_points or pt_tuple != smooth_points[-1]:
                    smooth_points.append(pt_tuple)

        last_orig = points[-1]
        if not smooth_points or smooth_points[-1] != last_orig:
            smooth_points.append(last_orig)

        return smooth_points

    @classmethod
    def smooth_stroke_with_safety(cls, stroke: List[Tuple[int, int]],
                                  mask: Optional[np.ndarray] = None,
                                  min_points: int = 3) -> List[Tuple[int, int]]:
        """
        Suaviza um traço com validação escalar rápida de retidão e checagem vetorizada de máscara.
        """
        if not stroke or len(stroke) < min_points:
            return stroke

        # Verificação rápida de colinearidade sem alocação de objetos NumPy
        is_straight = True
        for i in range(1, len(stroke) - 1):
            p_prev = stroke[i - 1]
            p_curr = stroke[i]
            p_next = stroke[i + 1]

            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]

            n1 = math.hypot(dx1, dy1)
            n2 = math.hypot(dx2, dy2)

            if n1 > 1e-3 and n2 > 1e-3:
                cos_angle = max(-1.0, min(1.0, (dx1 * dx2 + dy1 * dy2) / (n1 * n2)))
                # Ângulo superior a 15 graus representa curvatura relevante
                if cos_angle < 0.965:
                    is_straight = False
                    break

        if is_straight:
            return stroke

        spline_pts = cls.catmull_rom_spline(stroke, num_samples_per_segment=3, alpha=0.5)

        if mask is None:
            return spline_pts

        # Validação vetorizada em lote contra a máscara binária
        h, w = mask.shape
        pts_arr = np.array(spline_pts, dtype=np.int32)
        xs = pts_arr[:, 0]
        ys = pts_arr[:, 1]

        # Verifica se todos os pontos estão dentro dos limites da tela
        in_bounds = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.all(in_bounds):
            return stroke

        # Verifica se todos os pontos estão sobre pixels brancos da máscara
        if np.all(mask[ys, xs] > 0):
            return spline_pts

        return stroke

    @classmethod
    def smooth_all_strokes(cls, strokes_list: List[List[Tuple[int, int]]],
                           mask: Optional[np.ndarray] = None,
                           min_points: int = 3) -> List[List[Tuple[int, int]]]:
        """Aplica suavização caligráfica a todas as pinceladas da lista de forma eficiente."""
        result = []
        for s in strokes_list:
            if len(s) >= min_points:
                smoothed = cls.smooth_stroke_with_safety(s, mask=mask, min_points=min_points)
                if len(smoothed) >= 2:
                    result.append(smoothed)
            elif len(s) >= 2:
                result.append(s)
        return result
