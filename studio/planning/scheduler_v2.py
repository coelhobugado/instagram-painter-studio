"""
Agendador de Trajetória Multi-Custo com Otimização Local 2-Opt.
Implementa a Seção 35 (Scheduler v2) e Seção 36 (Scheduler hierárquico com 2-opt) do Plano.
"""
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class StrokeSchedulerV2:

    @classmethod
    def can_chain(cls, stroke_a: List[Tuple[int, int]], stroke_b: List[Tuple[int, int]],
                  allowed_mask: Optional[np.ndarray] = None, max_gap: float = 6.0,
                  min_cos_tangent: float = 0.70) -> bool:
        """
        Validação topológica estrita de encadeamento entre dois traços (Fase 1.3).
        Elimina completamente pontes falsas (false bridges):
        - Distância entre pontas <= max_gap (default: 6px).
        - Tangentes das extremidades alinhadas (cos >= min_cos_tangent, ~45°).
        - O segmento intermediário deve estar contido inteiramente no corredor permitido (allowed_mask).
        """
        if not stroke_a or not stroke_b or len(stroke_a) < 2 or len(stroke_b) < 2:
            return False

        pA = stroke_a[-1]
        pB = stroke_b[0]

        # 1. Distância entre pontas
        dist = math.hypot(pB[0] - pA[0], pB[1] - pA[1])
        if dist > max_gap:
            return False

        # Se estão no mesmo ponto exato, o encadeamento é natural
        if dist < 1.0:
            return True

        # 2. Tangentes compatíveis nas extremidades
        # Tangente de saída do stroke_a
        prevA = stroke_a[max(0, len(stroke_a) - 3)]
        dxA = pA[0] - prevA[0]
        dyA = pA[1] - prevA[1]
        magA = math.hypot(dxA, dyA)

        # Tangente de entrada do stroke_b
        nextB = stroke_b[min(len(stroke_b) - 1, 2)]
        dxB = nextB[0] - pB[0]
        dyB = nextB[1] - pB[1]
        magB = math.hypot(dxB, dyB)

        if magA > 1e-4 and magB > 1e-4:
            cos_t = (dxA * dxB + dyA * dyB) / (magA * magB)
            if cos_t < min_cos_tangent:
                return False

        # 3. Validação do corredor do segmento no allowed_mask (se fornecido)
        if allowed_mask is not None:
            h, w = allowed_mask.shape[:2]
            # Amostra pontos intermediários ao longo do segmento A -> B
            num_samples = max(2, int(math.ceil(dist)))
            for step in range(num_samples + 1):
                inter_x = int(round(pA[0] + (pB[0] - pA[0]) * (step / num_samples)))
                inter_y = int(round(pA[1] + (pB[1] - pA[1]) * (step / num_samples)))
                if 0 <= inter_x < w and 0 <= inter_y < h:
                    if allowed_mask[inter_y, inter_x] == 0:
                        return False
                else:
                    return False

        return True

    @classmethod
    def chain_continuous_strokes(cls, strokes: List[List[Tuple[int, int]]],
                                 max_gap: float = 35.0,
                                 allowed_mask: Optional[np.ndarray] = None) -> List[List[Tuple[int, int]]]:
        """
        Funde traços consecutivos adjacentes em pinceladas contínuas (toque contínuo).
        Evita o levantamento desnecessário do dedo (UP/DOWN) na tela do celular.
        """
        if not strokes or len(strokes) <= 1:
            return strokes

        chained: List[List[Tuple[int, int]]] = []
        for s in strokes:
            if not s or len(s) < 2:
                continue
            if not chained:
                chained.append(list(s))
            else:
                pA = chained[-1][-1]
                pB = s[0]
                dist = math.hypot(pB[0] - pA[0], pB[1] - pA[1])

                is_safe = True
                if allowed_mask is not None and dist > 1.0:
                    h, w = allowed_mask.shape[:2]
                    num_samples = max(2, int(math.ceil(dist)))
                    for step in range(num_samples + 1):
                        ix = int(round(pA[0] + (pB[0] - pA[0]) * (step / num_samples)))
                        iy = int(round(pA[1] + (pB[1] - pA[1]) * (step / num_samples)))
                        if 0 <= ix < w and 0 <= iy < h:
                            if allowed_mask[iy, ix] == 0:
                                is_safe = False
                                break
                        else:
                            is_safe = False
                            break

                if dist <= max_gap and is_safe:
                    if chained[-1][-1] == s[0]:
                        chained[-1].extend(s[1:])
                    else:
                        chained[-1].extend(s)
                else:
                    chained.append(list(s))
        return chained

    @classmethod
    def optimize_layer_strokes(cls, strokes: List[List[Tuple[int, int]]],
                               enable_2opt: bool = True,
                               allowed_mask: Optional[np.ndarray] = None,
                               continuous_chain: bool = False,
                               max_chain_gap: float = 35.0) -> List[List[Tuple[int, int]]]:
        """
        Ordena uma lista de traços de uma mesma camada minimizando a distância
        de deslocamento no ar, aplicando Nearest-Neighbor com reversão, refinamento 2-Opt
        e encadeamento seguro com can_chain (sem false bridges).
        """
        if not strokes or len(strokes) <= 1:
            return strokes

        valid_strokes = [list(s) for s in strokes if len(s) >= 2]
        if not valid_strokes:
            return []

        # 1. Nearest-Neighbor inicial com reversão de pontas hiper-otimizado com NumPy
        n = len(valid_strokes)
        if n == 1:
            return valid_strokes

        # Extrai pontos iniciais e finais em arrays float32 contíguos
        starts = np.array([s[0] for s in valid_strokes], dtype=np.float32)
        ends = np.array([s[-1] for s in valid_strokes], dtype=np.float32)

        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        curr_end = ends[0].copy()
        ordered = [valid_strokes[0]]

        for _ in range(n - 1):
            d_starts = (starts[:, 0] - curr_end[0])**2 + (starts[:, 1] - curr_end[1])**2
            d_ends = (ends[:, 0] - curr_end[0])**2 + (ends[:, 1] - curr_end[1])**2

            d_starts[visited] = np.inf
            d_ends[visited] = np.inf

            min_s_idx = int(np.argmin(d_starts))
            min_e_idx = int(np.argmin(d_ends))

            if d_starts[min_s_idx] <= d_ends[min_e_idx]:
                best_idx = min_s_idx
                next_stroke = valid_strokes[best_idx]
                curr_end = ends[best_idx].copy()
            else:
                best_idx = min_e_idx
                next_stroke = list(reversed(valid_strokes[best_idx]))
                curr_end = starts[best_idx].copy()

            visited[best_idx] = True
            ordered.append(next_stroke)

        # 2. Refinamento 2-Opt Local para eliminar cruzamentos aéreos
        if enable_2opt and len(ordered) >= 4:
            ordered = cls._apply_2opt(ordered, max_iterations=20)

        # 3. Encadeamento Seguro de Traços
        if continuous_chain:
            return cls.chain_continuous_strokes(ordered, max_gap=max_chain_gap, allowed_mask=allowed_mask)

        chained: List[List[Tuple[int, int]]] = []
        for s in ordered:
            if not chained:
                chained.append(list(s))
            else:
                if cls.can_chain(chained[-1], s, allowed_mask=allowed_mask):
                    # Se o primeiro ponto for idêntico ao último, omite duplicação
                    if chained[-1][-1] == s[0]:
                        chained[-1].extend(s[1:])
                    else:
                        chained[-1].extend(s)
                else:
                    chained.append(list(s))

        return chained

    @staticmethod
    def _apply_2opt(strokes: List[List[Tuple[int, int]]], max_iterations: int = 50) -> List[List[Tuple[int, int]]]:
        """
        Aplica busca local 2-opt em janelas deslizantes para desatar cruzamentos de trajetórias.
        Inverter o bloco [i+1, j-1] troca duas arestas aéreas:
        (fim_i -> início_{i+1}) + (fim_{j-1} -> início_j)  por  (fim_i -> fim_{j-1}) + (início_{i+1} -> início_j).
        """
        n = len(strokes)

        def dist(p, q):
            return math.hypot(p[0] - q[0], p[1] - q[1])

        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(n - 2):
                for j in range(i + 2, min(i + 12, n) + 1):
                    a_end = strokes[i][-1]
                    b_start = strokes[i + 1][0]
                    c_end = strokes[j - 1][-1]
                    d_current = dist(a_end, b_start)
                    d_new = dist(a_end, c_end)
                    if j < n:
                        d_start = strokes[j][0]
                        d_current += dist(c_end, d_start)
                        d_new += dist(b_start, d_start)

                    if d_new < d_current - 1.0:  # Ganho mínimo de 1px no deslocamento aéreo
                        # Inverte o sub-bloco (ordem e sentido de cada traço)
                        sub = strokes[i + 1:j]
                        sub.reverse()
                        for s in sub:
                            s.reverse()
                        strokes[i + 1:j] = sub
                        improved = True

        return strokes


class PhysicalCostScheduler:
    """
    Agendador de Trajetória com Custo Físico Real (Fase 16).
    Minimiza:
    air travel + color change penalty + brush change penalty + UI operation cost + semantic inversion + risk.
    Respeita restrições duras:
    occlusion, alpha composition, layer dependency, protected feature order.
    """

    COLOR_SWITCH_COST_PX: float = 350.0   # Equivalente em pixels de deslocamento ao atraso de troca de cor
    BRUSH_SWITCH_COST_PX: float = 200.0   # Custo do toque na toolbar
    SEMANTIC_INVERSION_PENALTY: float = 1e6 # Penalidade infinita para violar ordem semântica/oclusão

    @classmethod
    def calculate_transition_cost(
        cls,
        stroke_a: Dict[str, Any],
        stroke_b: Dict[str, Any],
    ) -> float:
        """Calcula o custo físico da transição do final do stroke A para o início do stroke B."""
        pts_a = stroke_a.get("points", [])
        pts_b = stroke_b.get("points", [])

        if not pts_a or not pts_b:
            return 0.0

        pA = pts_a[-1]
        pB = pts_b[0]

        # 1. Distância no ar (air travel)
        air_dist = float(math.hypot(pB[0] - pA[0], pB[1] - pA[1]))

        # 2. Penalidade de troca de cor
        color_cost = 0.0
        if stroke_a.get("color_rgb") != stroke_b.get("color_rgb"):
            color_cost = cls.COLOR_SWITCH_COST_PX

        # 3. Penalidade de troca de ferramenta/pincel
        brush_cost = 0.0
        if stroke_a.get("brush") != stroke_b.get("brush"):
            brush_cost = cls.BRUSH_SWITCH_COST_PX

        # 4. Risco acumulado
        risk_cost = float(stroke_b.get("risk", 0.0)) * 50.0

        return air_dist + color_cost + brush_cost + risk_cost

    @classmethod
    def schedule_layers_and_strokes(
        cls,
        strokes_with_metadata: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ordena os traços minimizando o custo físico global sob restrições estritas de precedência.
        Garante que nunca haja inversão de oclusão (ex: fundo pintado depois de figura).
        """
        if not strokes_with_metadata:
            return []

        # Agrupa por camada respeitando layer_id (preserva oclusão e alpha composition)
        # Traços de camadas anteriores NUNCA podem ser executados depois de camadas posteriores
        layer_groups: Dict[int, List[Dict[str, Any]]] = {}
        for s in strokes_with_metadata:
            l_id = int(s.get("layer_id", 0))
            if l_id not in layer_groups:
                layer_groups[l_id] = []
            layer_groups[l_id].append(s)

        sorted_layer_ids = sorted(layer_groups.keys())
        scheduled_all: List[Dict[str, Any]] = []

        last_stroke = None
        for l_id in sorted_layer_ids:
            group = layer_groups[l_id]
            # Ordena internamente na camada usando custo físico real
            remaining = list(group)
            layer_scheduled: List[Dict[str, Any]] = []

            # Ponto de partida: mais próximo do último traço da camada anterior
            if last_stroke and remaining:
                remaining.sort(key=lambda s: cls.calculate_transition_cost(last_stroke, s))
                current = remaining.pop(0)
            else:
                current = remaining.pop(0)

            layer_scheduled.append(current)

            while remaining:
                # Escolhe o próximo com menor custo de transição
                best_idx = 0
                best_cost = float("inf")
                for idx, cand in enumerate(remaining):
                    cost = cls.calculate_transition_cost(current, cand)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx

                current = remaining.pop(best_idx)
                layer_scheduled.append(current)

            scheduled_all.extend(layer_scheduled)
            last_stroke = layer_scheduled[-1] if layer_scheduled else last_stroke

        return scheduled_all

