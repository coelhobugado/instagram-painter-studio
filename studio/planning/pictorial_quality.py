"""
Módulo de Qualidade Pictórica, Preservação de Espaço Negativo e Particionamento de Traços (P3).
Implementa as soluções técnicas das Etapas 0, 1 e 2 do Plano de Evolução da Qualidade do Desenho:
- Análise de viabilidade física contra as pegadas reais do hardware.
- Detecção e proteção estrita de espaços negativos críticos (olhos, feições, frestas, pétalas).
- Poda morfológica de esporões espúrios (spurs) do esqueleto sem perda topológica.
- Particionamento hierárquico multi-camada por classes reais de espessura de pincel.
"""

from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np

from ..device.calibration import HardwareCalibrationEngine


class NegativeSpaceRegion:
    """Representa uma cavidade ou passagem de espaço negativo (fundo cercado por traços)."""
    def __init__(self, region_id: int, mask: np.ndarray, area: int, bbox: Tuple[int, int, int, int], is_critical: bool):
        self.region_id = region_id
        self.mask = mask
        self.area = area
        self.x, self.y, self.w, self.h = bbox
        self.is_critical = is_critical
        # Calcula raio máximo interno (distância máxima ao contorno de tinta mais próximo)
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        self.max_radius = float(np.max(dist_map)) if np.any(mask) else 0.0


class NegativeSpaceAnalyzer:
    """
    Analisa os vazios e espaços negativos da referência para impedir que pincéis físicos
    fechem olhos, bocas, letras ou separações de pétalas por excesso de tinta (ink bleed).
    """

    @classmethod
    def identify_enclosed_spaces(cls, ink_mask: np.ndarray, min_area: int = 8, max_area_ratio: float = 0.5) -> List[NegativeSpaceRegion]:
        """
        Identifica regiões de fundo (espaço negativo) cercadas ou semifechadas por tinta.
        """
        h, w = ink_mask.shape[:2]
        total_pixels = h * w
        background_mask = np.where(ink_mask == 0, 255, 0).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background_mask, connectivity=8)
        regions: List[NegativeSpaceRegion] = []

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            bx = stats[i, cv2.CC_STAT_LEFT]
            by = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # Ignora o fundo externo infinito que toca as bordas da tela inteira
            touches_all_borders = (bx == 0 and by == 0 and (bx + bw) >= w and (by + bh) >= h)
            if touches_all_borders or area > (total_pixels * max_area_ratio):
                continue

            if area < min_area:
                continue

            region_mask = np.where(labels == i, 255, 0).astype(np.uint8)
            # Cavidades pequenas a médias internas (como olhos, boca, frestas) são críticas
            is_critical = area <= (total_pixels * 0.15)

            regions.append(NegativeSpaceRegion(
                region_id=i,
                mask=region_mask,
                area=area,
                bbox=(bx, by, bw, bh),
                is_critical=is_critical
            ))

        return regions

    @classmethod
    def build_critical_protection_mask(cls, negative_regions: List[NegativeSpaceRegion], shape: Tuple[int, int]) -> np.ndarray:
        """
        Gera uma máscara booleana/binária unificada de todas as cavidades que não podem ser invadidas por tinta.
        """
        prot_mask = np.zeros(shape, dtype=np.uint8)
        for r in negative_regions:
            if r.is_critical:
                prot_mask[r.mask > 0] = 255
        return prot_mask


class SkeletonOptimizer:
    """
    Otimiza e limpa esqueletos morfológicos brutos eliminando esporões espúrios (spurs)
    sem quebrar a topologia dos ramos mestres e fechamentos de loops.
    """

    @classmethod
    def prune_spurs(cls, skeleton: np.ndarray, max_spur_length: int = 5) -> np.ndarray:
        """
        Remove pequenos ramos mortos (espinhos de afinamento) de comprimento <= max_spur_length
        que saem de junções e apenas engrossam nós sem acrescentar traço útil.
        Preserva rigorosamente as pontas e o comprimento de linhas mestras abertas.
        """
        skel = (skeleton > 0).astype(np.uint8)
        h, w = skel.shape[:2]
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.uint8)

        # Itera para remover spurs
        for _ in range(max_spur_length):
            neighbor_count = cv2.filter2D(skel, -1, kernel)
            endpoints = np.argwhere((skel > 0) & (neighbor_count == 1))
            if len(endpoints) == 0:
                break

            # Branch points são nós onde convergem 3 ou mais ramos
            branch_mask = (skel > 0) & (neighbor_count >= 3)
            pixels_to_remove = set()

            for ep_y, ep_x in endpoints:
                path = [(int(ep_y), int(ep_x))]
                curr_y, curr_x = int(ep_y), int(ep_x)
                found_branch = False

                for step in range(max_spur_length):
                    prev_y, prev_x = path[-2] if len(path) >= 2 else (-1, -1)
                    next_node = None

                    # Procura o próximo vizinho 8-conectado
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = curr_y + dy, curr_x + dx
                            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                                if (ny, nx) != (prev_y, prev_x):
                                    if branch_mask[ny, nx]:
                                        found_branch = True
                                        break
                                    elif neighbor_count[ny, nx] == 2:
                                        next_node = (ny, nx)
                                        break
                        if found_branch or next_node:
                            break

                    if found_branch:
                        # O caminho do endpoint até a junção é um spur a ser eliminado
                        for p in path:
                            pixels_to_remove.add(p)
                        break
                    elif next_node is not None:
                        path.append(next_node)
                        curr_y, curr_x = next_node
                    else:
                        # Ramo isolado ou sem junção nesta janela
                        break

            if not pixels_to_remove:
                break

            for py, px in pixels_to_remove:
                skel[py, px] = 0

        return (skel * 255).astype(np.uint8)

    @classmethod
    def filter_dense_parallel_strokes(
        cls,
        strokes: List[List[Tuple[int, int]]],
        min_separation_px: float = 6.0,
        shape: Tuple[int, int] = (1000, 1000)
    ) -> List[List[Tuple[int, int]]]:
        """
        Filtra hachuras e traços secundários hiperdensos que colapsariam em uma mancha escura.
        Mantém os traços estruturais longos e descarta micro-linhas paralelas que invadem o espaço livre do pincel.
        """
        if len(strokes) <= 1:
            return strokes

        # Ordena traços por comprimento decrescente (prioriza traços longos e estruturais)
        indexed_strokes = sorted(
            enumerate(strokes),
            key=lambda item: sum(
                np.hypot(item[1][i+1][0] - item[1][i][0], item[1][i+1][1] - item[1][i][1])
                for i in range(len(item[1]) - 1)
            ) if len(item[1]) > 1 else 0.0,
            reverse=True
        )

        h, w = shape[:2]
        occupancy_grid = np.zeros((h, w), dtype=np.uint8)
        accepted_indices = set()
        brush_radius = max(2, int(round(min_separation_px / 2.0)))

        for original_idx, stroke in indexed_strokes:
            if not stroke or len(stroke) < 2:
                continue

            stroke_len = sum(
                np.hypot(stroke[i+1][0] - stroke[i][0], stroke[i+1][1] - stroke[i][1])
                for i in range(len(stroke) - 1)
            )

            # Mede a fração de pontos que caem na pegada de outro traço já aceito
            overlap_pts = 0
            for px, py in stroke:
                cx = max(0, min(w - 1, int(round(px))))
                cy = max(0, min(h - 1, int(round(py))))
                if occupancy_grid[cy, cx] > 0:
                    overlap_pts += 1

            overlap_ratio = overlap_pts / float(len(stroke))

            # Para traços longos (estruturais), aceita a menos que quase todo o traço (>= 88%)
            # seja uma duplicata colada paralela que fundiria caules/pétalas (Etapa 2 do Plano)
            if stroke_len >= 40.0:
                if overlap_ratio < 0.88:
                    accepted_indices.add(original_idx)
                    for i in range(len(stroke) - 1):
                        p1, p2 = stroke[i], stroke[i+1]
                        cv2.line(occupancy_grid, p1, p2, 255, thickness=brush_radius * 2)
                continue

            # Para traços menores e hachuras, se menos de 70% do traço sobrepõe a pegada existente, aceita
            if overlap_ratio < 0.70:
                accepted_indices.add(original_idx)
                for i in range(len(stroke) - 1):
                    p1, p2 = stroke[i], stroke[i+1]
                    cv2.line(occupancy_grid, p1, p2, 255, thickness=brush_radius * 2)

        # Restaura a ordem original
        filtered = [s for idx, s in enumerate(strokes) if idx in accepted_indices]
        return filtered if filtered else strokes


class AdaptiveLineArtPartitioner:
    """
    Particiona traços contínuos de line-art em camadas especializadas de acordo com as
    larguras físicas reais medidas no hardware (Galaxy A73 / perfis de dispositivo).
    """

    def __init__(self, calib_engine: Optional[HardwareCalibrationEngine] = None):
        self.calib = calib_engine or HardwareCalibrationEngine()
        self.w_min = self.calib.get_effective_brush_width("min", "solid")
        self.w_med = self.calib.get_effective_brush_width("med", "solid")
        self.w_thick = self.calib.get_effective_brush_width("thick", "solid")

        # Limiares de decisão ótimos (ponto médio entre os pincéis físicos reais)
        self.thresh_min_med = (self.w_min + self.w_med) / 2.0  # Tipicamente (10.0 + 52.0) / 2 = 31.0 px
        self.thresh_med_thick = (self.w_med + self.w_thick) / 2.0  # Tipicamente (52.0 + 98.0) / 2 = 75.0 px

    def partition_strokes(
        self,
        strokes: List[List[Tuple[int, int]]],
        stroke_widths: List[List[float]],
        critical_mask: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Distribui os traços em camadas hierárquicas ('min', 'med', 'thick').
        Traços finos ou que passam próximos a cavidades de espaço negativo são forçados em 'min'.
        """
        buckets = {
            "min": {"strokes": [], "widths": []},
            "med": {"strokes": [], "widths": []},
            "thick": {"strokes": [], "widths": []},
        }

        h, w = (critical_mask.shape[:2]) if critical_mask is not None else (2000, 2000)

        for s, w_list in zip(strokes, stroke_widths):
            if not s:
                continue

            median_w = float(np.median(w_list)) if w_list else self.w_min

            # Verifica proximidade com espaço negativo crítico
            near_critical_negative_space = False
            if critical_mask is not None and np.any(critical_mask):
                for px, py in s:
                    cx = max(0, min(w - 1, int(round(px))))
                    cy = max(0, min(h - 1, int(round(py))))
                    # Raio de busca ao redor do traço
                    r = int(np.ceil(self.w_min))
                    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                    if np.any(critical_mask[y0:y1, x0:x1] > 0):
                        near_critical_negative_space = True
                        break

            # Se estiver perto de espaços vazios protegidos (olhos, boca, frestas),
            # usa OBRIGATORIAMENTE a caneta mínima para não engolir o espaço negativo!
            if near_critical_negative_space:
                target_level = "min"
            elif median_w < self.thresh_min_med:
                target_level = "min"
            elif median_w < self.thresh_med_thick:
                target_level = "med"
            else:
                target_level = "thick"

            buckets[target_level]["strokes"].append(s)
            buckets[target_level]["widths"].append(w_list)

        return buckets


class ExecutableTargetBuilder:
    """
    Constrói a imagem 'target_executable' adaptada à física do hardware.
    Representa o que o celular deve almejar de forma viável, sem penalizar métricas
    por detalhes sub-físicos impossíveis de reproduzir.
    """

    @classmethod
    def build_executable_target(
        cls,
        shape: Tuple[int, int, int],
        layers: List[Dict[str, Any]],
        background_rgb: Tuple[int, int, int],
        ink_rgb: Tuple[int, int, int],
        simulator: Any,
        seed: int = 42
    ) -> np.ndarray:
        """
        Renderiza a interpretação desenhável alvo com as espessuras reais dos traços compilados.
        """
        h, w = shape[:2]
        bg_bgr = (int(background_rgb[2]), int(background_rgb[1]), int(background_rgb[0]))
        target_canvas, _ = simulator.render_simulation(shape, layers, background_bgr=bg_bgr, seed=seed)
        return target_canvas
