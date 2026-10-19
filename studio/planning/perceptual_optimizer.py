# -*- coding: utf-8 -*-
"""
Otimizador Perceptual Global e Motor de Poda Ativa de Traços (P5).
Implementa os requisitos da Etapa 4 e 5 do Plano de Evolução da Qualidade:
1. Poda ativa de traços redundantes e nocivos (redução de excesso de tinta e nós engrossados).
2. Encurtamento e recorte de traços que invadem feições protegidas (olhos, dentes, cavidades).
3. Seleção e poda orientada a Ganho por Segundo (GlobalObjectiveLoss) sob restrição de tempo.
4. Refinamento bidirecional: adição seletiva e eliminação de traços com retorno visual negativo.
"""
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from .global_objective import GlobalObjectiveLoss, LossWeights


class PerceptualPlanOptimizer:
    """
    Motor de otimização global que governa a densidade, precedência e poda de traços
    com base no ganho perceptual por segundo e proteção estrita de feições essenciais.
    """

    @classmethod
    def clip_stroke_against_mask(
        cls,
        stroke: List[Tuple[int, int]],
        forbidden_mask: np.ndarray,
        interior_mask: Optional[np.ndarray] = None
    ) -> List[List[Tuple[int, int]]]:
        """
        Recorta um traço contra uma máscara proibida (espaço negativo crítico ou olhos).
        Divide o traço em sub-segmentos úteis que fiquem estritamente fora da região proibida.
        """
        if not stroke or len(stroke) < 2 or forbidden_mask is None or np.count_nonzero(forbidden_mask) == 0:
            return [stroke] if len(stroke) >= 2 else []

        h, w = forbidden_mask.shape[:2]
        valid_sub_strokes: List[List[Tuple[int, int]]] = []
        current_sub: List[Tuple[int, int]] = []
        last_free: Optional[Tuple[int, int]] = None

        def close_current():
            nonlocal current_sub
            if current_sub and last_free is not None and current_sub[-1] != last_free:
                current_sub.append(last_free)
            # Descarta lascas < 3px junto à borda protegida (o executor e o QualityGate rejeitam traços < 2.5px)
            if len(current_sub) >= 2 and sum(
                    np.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(current_sub, current_sub[1:])) >= 3.0:
                valid_sub_strokes.append(current_sub)
            current_sub = []

        # Amostra cada segmento a cada ~1px: um traço esparso (ex.: scanline com 2 pontos)
        # não pode atravessar a região protegida só porque seus vértices caem fora dela.
        # Entre vértices usa a máscara erodida em 1px, para que cordas de curvas que só
        # tangenciam a borda (erro de aproximação/spline) não fragmentem o traço.
        if interior_mask is None:
            interior_mask = cv2.erode(forbidden_mask, np.ones((3, 3), np.uint8))
        for seg_idx in range(len(stroke) - 1):
            (x0, y0), (x1, y1) = stroke[seg_idx], stroke[seg_idx + 1]
            n = max(1, int(np.ceil(np.hypot(x1 - x0, y1 - y0))))
            t = np.linspace(0.0, 1.0, n + 1)
            xs = np.clip(np.round(x0 + (x1 - x0) * t).astype(np.int32), 0, w - 1)
            ys = np.clip(np.round(y0 + (y1 - y0) * t).astype(np.int32), 0, h - 1)
            free = interior_mask[ys, xs] == 0
            free[0] = forbidden_mask[ys[0], xs[0]] == 0
            free[-1] = forbidden_mask[ys[-1], xs[-1]] == 0
            start_k = 0 if seg_idx == 0 else 1

            if free[start_k:].all():
                if not current_sub:
                    current_sub = [(int(xs[start_k]), int(ys[start_k]))]
                end_pt = (int(xs[-1]), int(ys[-1]))
                if current_sub[-1] != end_pt:
                    current_sub.append(end_pt)
                last_free = end_pt
                continue

            for k in range(start_k, n + 1):
                pt = (int(xs[k]), int(ys[k]))
                if free[k]:
                    if not current_sub:
                        current_sub = [pt]
                    elif k == n and current_sub[-1] != pt:
                        current_sub.append(pt)
                    last_free = pt
                else:
                    close_current()

        close_current()
        return valid_sub_strokes

    @classmethod
    def prune_redundant_strokes(
        cls,
        strokes: List[List[Tuple[int, int]]],
        brush_width: float,
        shape: Tuple[int, int],
        max_overlap_ratio: float = 0.82
    ) -> List[List[Tuple[int, int]]]:
        """
        Remove traços redundantes cuja área já foi coberta em mais de max_overlap_ratio
        por traços anteriores da mesma camada ou de camadas adjacentes, prevenindo manchas escuras.
        """
        if len(strokes) <= 1:
            return strokes

        h, w = shape[:2]
        accumulated_coverage = np.zeros((h, w), dtype=np.uint8)
        accepted_strokes: List[List[Tuple[int, int]]] = []
        stroke_thickness = max(1, int(round(brush_width)))

        for s in strokes:
            if len(s) < 2:
                continue

            pts = np.array(s, dtype=np.int32).reshape((-1, 2))
            # Trabalha apenas na bbox do traço (+ espessura), sem alocar o frame inteiro por traço
            pad = stroke_thickness + 1
            x0 = max(0, int(pts[:, 0].min()) - pad)
            y0 = max(0, int(pts[:, 1].min()) - pad)
            x1 = min(w, int(pts[:, 0].max()) + pad + 1)
            y1 = min(h, int(pts[:, 1].max()) + pad + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            stroke_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cv2.polylines(stroke_mask, [(pts - (x0, y0)).reshape((-1, 1, 2))], False, 255, stroke_thickness)

            stroke_area = int(np.count_nonzero(stroke_mask))
            if stroke_area == 0:
                continue

            # Calcula sobreposição com traços já desenhados
            acc_roi = accumulated_coverage[y0:y1, x0:x1]
            overlap_pixels = int(np.count_nonzero(cv2.bitwise_and(stroke_mask, acc_roi)))
            overlap_ratio = overlap_pixels / float(stroke_area)

            # Se mais de 82% do traço for redundante sobre tinta já existente, descarta
            if overlap_ratio >= max_overlap_ratio:
                continue

            accepted_strokes.append(s)
            cv2.bitwise_or(acc_roi, stroke_mask, dst=acc_roi)

        return accepted_strokes

    @classmethod
    def filter_layer_strokes_for_safety(
        cls,
        layer: Dict[str, Any],
        protected_mask: Optional[np.ndarray],
        brush_width: float,
        canvas_shape: Tuple[int, int]
    ) -> None:
        """
        Aplica poda de segurança e recorte em uma camada específica:
        1. Recorta traços contra a máscara protegida se a camada for de preenchimento ou escura.
        2. Remove traços hiper-redundantes da camada.
        """
        purpose = layer.get("purpose", "")
        orig_strokes = layer.get("strokes", [])
        if not orig_strokes:
            return

        # Camadas que não podem invadir cavidades claras/olhos
        clipped_strokes: List[List[Tuple[int, int]]] = []
        should_clip = purpose not in ("highlight", "catchlight") and protected_mask is not None

        if should_clip and np.count_nonzero(protected_mask) > 0:
            interior = cv2.erode(protected_mask, np.ones((3, 3), np.uint8))
            for s in orig_strokes:
                subs = cls.clip_stroke_against_mask(s, protected_mask, interior_mask=interior)
                clipped_strokes.extend(subs)
        else:
            clipped_strokes = orig_strokes

        # Poda traços redundantes
        pruned_strokes = cls.prune_redundant_strokes(
            clipped_strokes, brush_width=brush_width, shape=canvas_shape
        )

        layer["strokes"] = pruned_strokes
        layer["stroke_count"] = len(pruned_strokes)
        layer["point_count"] = sum(len(s) for s in pruned_strokes)

    # Mesmo modelo de custo usado na estimativa do plano (painter_engine): s/traço + s/ponto
    SECONDS_PER_STROKE = 0.4
    SECONDS_PER_POINT = 0.003

    @classmethod
    def estimate_layer_time_s(cls, layer: Dict[str, Any]) -> float:
        return (layer.get("stroke_count", 0) * cls.SECONDS_PER_STROKE
                + layer.get("point_count", 0) * cls.SECONDS_PER_POINT)

    @staticmethod
    def _layer_ink_length(layer: Dict[str, Any]) -> float:
        """Comprimento total de tinta da camada (px), proxy da área coberta/redução de erro."""
        total = 0.0
        for s in layer.get("strokes", []):
            if len(s) >= 2:
                arr = np.asarray(s, dtype=np.float32)
                total += float(np.sum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))))
        return total

    @classmethod
    def optimize_plan_budget_and_gain(
        cls,
        plan: Dict[str, Any],
        time_budget_s: float,
        image_class: str = "general"
    ) -> bool:
        """
        Governa o orçamento de execução:
        Se o plano ultrapassar o orçamento físico em segundos, prioriza camadas estruturais
        (LOD 0, contornos, fundo e highlights de olhos) e poda traços secundários
        com base no Ganho por Segundo de GlobalObjectiveLoss.
        Retorna True se alguma camada foi removida (a simulação precisa ser re-renderizada).
        """
        layers = plan.get("layers", [])
        if not layers:
            return False

        current_time = float(plan.get("estimated_time_s", 0.0))

        # Se já estiver dentro do orçamento, não precisa podar camadas
        if current_time <= time_budget_s:
            return False

        # Classifica camadas entre Essenciais e Opcionais
        essential_purposes = {"background", "dark_base", "contour", "highlight", "catchlight"}
        essential_layers = []
        optional_layers = []

        for l in layers:
            p = l.get("purpose", "")
            if p in essential_purposes or "olho" in l.get("name", "").lower() or "catchlight" in l.get("name", "").lower():
                essential_layers.append(l)
            else:
                # Ganho aproximado por segundo: tinta depositada (px de traço x largura relativa) / custo
                cost_s = cls.estimate_layer_time_s(l)
                width_factor = 2.0 if l.get("thickness") in ("med", "thick", "max") else 1.0
                predicted_reduction = cls._layer_ink_length(l) * width_factor
                score = GlobalObjectiveLoss.compute_gain_per_second(predicted_reduction, cost_s)
                optional_layers.append((score, l))

        # Ordena camadas opcionais por maior ganho por segundo primeiro
        optional_layers.sort(key=lambda item: item[0], reverse=True)

        accepted_layers = list(essential_layers)
        accum_time = sum(cls.estimate_layer_time_s(l) for l in essential_layers)

        for score, opt_l in optional_layers:
            layer_cost = cls.estimate_layer_time_s(opt_l)
            if accum_time + layer_cost <= time_budget_s:
                accepted_layers.append(opt_l)
                accum_time += layer_cost

        # Reordena conforme a ordem original de execução
        layer_indices = {id(l): idx for idx, l in enumerate(layers)}
        accepted_layers.sort(key=lambda l: layer_indices.get(id(l), 0))

        pruned = len(accepted_layers) != len(layers)
        plan["layers"] = accepted_layers
        plan["total_strokes"] = sum(l.get("stroke_count", 0) for l in accepted_layers)
        plan["total_points"] = sum(l.get("point_count", 0) for l in accepted_layers)
        plan["estimated_time_s"] = round(accum_time, 1)
        return pruned
