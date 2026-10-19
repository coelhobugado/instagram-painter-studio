"""
Pintura Baseada em Linhas de Fluxo (Streamlines RK4) e Tensor de Estrutura Condicional.
Implementa a Fase 14 (14.1 Frame Field Confiável, 14.2 RK4 Streamlines e 14.3/14.4 Uso Condicional) do Roadmap.
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from ..vision.structure_tensor import StructureTensorEngine


class FlowFieldPainter:
    """
    Gerador de Pinceladas Orgânicas Orientadas por Fluxo Vetorial (Streamlines).
    Usa RK4 (Runge-Kutta de 4ª ordem) com confinamento estrito de máscara.
    """

    @classmethod
    def should_use_flow_strategy(
        cls,
        region_mask: np.ndarray,
        semantic_type: str = "general",
        min_coherence: float = 0.35,
    ) -> Tuple[bool, float, str]:
        """
        Critério de Aceite da Fase 14.3:
        Decide se a região se beneficia de FLOW_FILL contra alternativas (MASS_FILL ou CONTOUR).
        Só recomenda se a coerência de orientação for significativa e o tipo for orgânico.
        """
        organic_types = {"hair", "fur", "cloth", "petal", "flower", "organic", "skin", "foliage"}
        is_organic = semantic_type.lower() in organic_types

        # Avalia a coerência do tensor de estrutura na região
        flow_dx, flow_dy, coherence_map = StructureTensorEngine.compute_orientation_field(region_mask)

        mask_bool = region_mask > 0
        if np.count_nonzero(mask_bool) < 50:
            return False, 0.0, "Área muito pequena para linhas de fluxo."

        mean_coherence = float(np.mean(coherence_map[mask_bool]))

        if is_organic and mean_coherence >= min_coherence:
            return True, mean_coherence, f"Recomendado FLOW_FILL: Região orgânica '{semantic_type}' com alta coerência ({mean_coherence:.2f})."
        elif mean_coherence >= 0.60:
            return True, mean_coherence, f"Recomendado FLOW_FILL: Forte anisotropia direcional ({mean_coherence:.2f})."

        return False, mean_coherence, f"Recomendado MASS_FILL: Coerência baixa ({mean_coherence:.2f}) ou superfície isotrópica."

    @classmethod
    def compute_frame_field(
        cls, region_mask: np.ndarray, coherence_thresh: float = 0.35
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula o campo de vetores diretores unitários (vx, vy) e coerência (Fase 14.1).
        Onde coerência < limiar, faz fallback para a tangente do contorno mais próximo ou direção principal.
        """
        h, w = region_mask.shape[:2]
        vx, vy, coherence = StructureTensorEngine.compute_orientation_field(region_mask)

        # Onde a coerência for baixa, aplica a direção principal calculada por PCA/Momentos
        low_coh_mask = (coherence < coherence_thresh) & (region_mask > 0)
        if np.any(low_coh_mask):
            moments = cv2.moments(region_mask)
            mu20 = moments['mu20']
            mu02 = moments['mu02']
            mu11 = moments['mu11']
            principal_angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02) if (mu20 + mu02) > 0 else 0.0
            pvx = float(np.cos(principal_angle))
            pvy = float(np.sin(principal_angle))
            vx[low_coh_mask] = pvx
            vy[low_coh_mask] = pvy

        return vx, vy, coherence

    @classmethod
    def integrate_rk4_streamline(
        cls,
        seed_x: float,
        seed_y: float,
        vx: np.ndarray,
        vy: np.ndarray,
        allowed_mask: np.ndarray,
        step_size: float = 2.0,
        max_steps: int = 150,
    ) -> List[Tuple[int, int]]:
        """
        Integração numérica Runge-Kutta de 4ª ordem (RK4) (Fase 14.2).
        Confinada estritamente dentro de allowed_mask.
        """
        h, w = allowed_mask.shape[:2]

        def sample_vector(px: float, py: float) -> Tuple[float, float]:
            ix = int(np.clip(round(px), 0, w - 1))
            iy = int(np.clip(round(py), 0, h - 1))
            return float(vx[iy, ix]), float(vy[iy, ix])

        def is_valid(px: float, py: float) -> bool:
            ix = int(round(px))
            iy = int(round(py))
            if 0 <= ix < w and 0 <= iy < h:
                return bool(allowed_mask[iy, ix] > 0)
            return False

        if not is_valid(seed_x, seed_y):
            return []

        # Integração bidirecional (para frente e para trás)
        streamline_pts: List[Tuple[int, int]] = [(int(round(seed_x)), int(round(seed_y)))]

        for direction in [1.0, -1.0]:
            cx, cy = seed_x, seed_y
            collected = []
            for _ in range(max_steps):
                # RK4
                k1_x, k1_y = sample_vector(cx, cy)
                k1_x *= direction
                k1_y *= direction

                mid1_x = cx + 0.5 * step_size * k1_x
                mid1_y = cy + 0.5 * step_size * k1_y
                k2_x, k2_y = sample_vector(mid1_x, mid1_y)
                k2_x *= direction
                k2_y *= direction

                mid2_x = cx + 0.5 * step_size * k2_x
                mid2_y = cy + 0.5 * step_size * k2_y
                k3_x, k3_y = sample_vector(mid2_x, mid2_y)
                k3_x *= direction
                k3_y *= direction

                end_x = cx + step_size * k3_x
                end_y = cy + step_size * k3_y
                k4_x, k4_y = sample_vector(end_x, end_y)
                k4_x *= direction
                k4_y *= direction

                dx = (step_size / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
                dy = (step_size / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

                nx = cx + dx
                ny = cy + dy

                # Checa se o novo ponto continua estritamente dentro da máscara
                if not is_valid(nx, ny):
                    break

                # Evita andar em círculos muito curtos
                if len(collected) > 2:
                    dist_to_prev = np.sqrt((nx - collected[-1][0]) ** 2 + (ny - collected[-1][1]) ** 2)
                    if dist_to_prev < 0.5:
                        break

                cx, cy = nx, ny
                collected.append((int(round(cx)), int(round(cy))))

            if direction == 1.0:
                streamline_pts.extend(collected)
            else:
                streamline_pts = list(reversed(collected)) + streamline_pts

        return streamline_pts

    @classmethod
    def generate_flow_strokes(
        cls,
        region_mask: np.ndarray,
        spacing: int = 12,
        step_size: float = 2.5,
        min_length: int = 4,
    ) -> List[List[Tuple[int, int]]]:
        """
        Gera conjunto de traços cobrindo a região com streamlines RK4.
        """
        h, w = region_mask.shape[:2]
        vx, vy, _ = cls.compute_frame_field(region_mask)

        # Amostragem em grade de sementes (seeds)
        ys, xs = np.where(region_mask > 0)
        if len(xs) == 0:
            return []

        visited_mask = np.zeros((h, w), dtype=np.uint8)
        strokes: List[List[Tuple[int, int]]] = []

        # Ordena pontos por espaçamento
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        x_min, x_max = int(np.min(xs)), int(np.max(xs))

        for sy in range(y_min, y_max, max(2, spacing)):
            for sx in range(x_min, x_max, max(2, spacing)):
                if region_mask[sy, sx] > 0 and visited_mask[sy, sx] == 0:
                    pts = cls.integrate_rk4_streamline(
                        sx, sy, vx, vy, region_mask, step_size=step_size
                    )
                    if len(pts) >= min_length:
                        strokes.append(pts)
                        # Marca vizinhança como visitada para manter espaçamento
                        for px, py in pts:
                            cv2.circle(visited_mask, (px, py), max(1, spacing // 2), 255, -1)

        return strokes
