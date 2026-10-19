# -*- coding: utf-8 -*-
"""
Motor de Análise de Tensor de Estrutura e Campo de Vetores Curvilíneos.
Implementa o Pilar 2 do Plano Mestre de Maturidade 100%.

Calcula o tensor de estrutura local J, extrai direções coerentes de fluxo tangencial (autovetores)
e integra streamlines curvilíneas 3D (Runge-Kutta) para pinceladas anatômicas orgânicas.
"""
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional

class StructureTensorEngine:
    """
    Calculador de Tensor de Estrutura de Imagem e Gerador de Streamlines Curvilíneas.
    """

    @classmethod
    def compute_orientation_field(cls, image: np.ndarray,
                                  sigma_inner: float = 1.2,
                                  sigma_outer: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula o campo de orientação de fluxo tangencial e mapa de coerência via Tensor de Estrutura.
        Retorna:
            - flow_dx: componente X do vetor unitário de fluxo tangencial (H, W)
            - flow_dy: componente Y do vetor unitário de fluxo tangencial (H, W)
            - coherence: medida de anisotropia/confiança da orientação no intervalo [0, 1] (H, W)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        gray_f = gray.astype(np.float32) / 255.0

        # 1. Gradientes de escala fina (escala de diferenciação)
        if sigma_inner > 0:
            ksize = int(math.ceil(sigma_inner * 3)) * 2 + 1
            gray_f = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma_inner)

        Ix = cv2.Scharr(gray_f, cv2.CV_32F, 1, 0)
        Iy = cv2.Scharr(gray_f, cv2.CV_32F, 0, 1)

        # 2. Produtos de gradiente
        Jxx = Ix * Ix
        Jyy = Iy * Iy
        Jxy = Ix * Iy

        # 3. Integração espacial Gaussiana (escala de integração)
        if sigma_outer > 0:
            ksize_out = int(math.ceil(sigma_outer * 3)) * 2 + 1
            Jxx = cv2.GaussianBlur(Jxx, (ksize_out, ksize_out), sigma_outer)
            Jyy = cv2.GaussianBlur(Jyy, (ksize_out, ksize_out), sigma_outer)
            Jxy = cv2.GaussianBlur(Jxy, (ksize_out, ksize_out), sigma_outer)

        # 4. Decomposição espectral dos autovetores
        diff = Jxx - Jyy
        delta = np.sqrt(np.maximum(0.0, diff * diff + 4.0 * Jxy * Jxy))
        trace = Jxx + Jyy

        # Autovalores
        lambda1 = (trace + delta) * 0.5
        lambda2 = np.maximum(0.0, (trace - delta) * 0.5)

        # Coerência normalizada [0, 1]
        coherence = np.zeros_like(trace)
        denom = lambda1 + lambda2 + 1e-6
        coherence = ((lambda1 - lambda2) / denom) ** 2
        coherence = np.clip(coherence, 0.0, 1.0)

        # Ângulo normal (gradiente) e ângulo tangencial de fluxo
        theta = 0.5 * np.arctan2(2.0 * Jxy, diff)
        flow_theta = theta + (np.pi / 2.0)

        flow_dx = np.cos(flow_theta).astype(np.float32)
        flow_dy = np.sin(flow_theta).astype(np.float32)

        return flow_dx, flow_dy, coherence

    @classmethod
    def generate_curvilinear_streamlines(cls, mask: np.ndarray,
                                         flow_dx: np.ndarray,
                                         flow_dy: np.ndarray,
                                         bbox: Tuple[int, int, int, int],
                                         spacing: int = 6,
                                         step_size: float = 3.0,
                                         min_length: float = 12.0,
                                         max_length: float = 600.0) -> List[List[Tuple[int, int]]]:
        """
        Integra linhas de fluxo contínuas (streamlines) no campo de vetores tangenciais dentro da máscara.
        Gera pinceladas curvilíneas que abraçam a geometria e o volume 3D da figura.
        """
        bx, by, bw, bh = bbox
        h, w = mask.shape
        if bw <= 0 or bh <= 0 or np.count_nonzero(mask) == 0:
            return []

        # Mapa de ocupação para garantir espaçamento uniforme entre traços vizinhos
        grid_scale = max(2, spacing // 2)
        grid_h = int(math.ceil(h / grid_scale))
        grid_w = int(math.ceil(w / grid_scale))
        occupied = np.zeros((grid_h, grid_w), dtype=bool)

        min_dist_sq = (spacing * 0.75) ** 2

        # Gera sementes de integração distribuídas pela máscara
        seed_points: List[Tuple[int, int]] = []
        for sy in range(by, by + bh, spacing):
            for sx in range(bx, bx + bw, spacing):
                if 0 <= sx < w and 0 <= sy < h and mask[sy, sx] > 0:
                    seed_points.append((sx, sy))

        # Embaralha ou organiza sementes por gradiente/centro para cobertura uniforme
        streamlines: List[List[Tuple[int, int]]] = []

        # Raio de vizinhança (em células) equivalente à separação mínima entre streamlines
        min_dist_cells = int(math.sqrt(min_dist_sq) // grid_scale)

        def is_too_close(px: float, py: float) -> bool:
            gx = int(px / grid_scale)
            gy = int(py / grid_scale)
            if not (0 <= gx < grid_w and 0 <= gy < grid_h):
                return True
            y0, y1 = max(0, gy - min_dist_cells), min(grid_h, gy + min_dist_cells + 1)
            x0, x1 = max(0, gx - min_dist_cells), min(grid_w, gx + min_dist_cells + 1)
            return bool(occupied[y0:y1, x0:x1].any())

        def mark_occupied(pts: List[Tuple[int, int]]) -> None:
            for px, py in pts:
                gx = int(px / grid_scale)
                gy = int(py / grid_scale)
                if 0 <= gx < grid_w and 0 <= gy < grid_h:
                    occupied[gy, gx] = True

        def get_flow_vector(x: float, y: float) -> Tuple[float, float]:
            ix = int(round(x))
            iy = int(round(y))
            ix = max(0, min(w - 1, ix))
            iy = max(0, min(h - 1, iy))
            return float(flow_dx[iy, ix]), float(flow_dy[iy, ix])

        for sx, sy in seed_points:
            if is_too_close(sx, sy):
                continue

            # Integração bidirecional (frente e verso)
            line_forward = cls._integrate_half_line(
                sx, sy, 1.0, mask, get_flow_vector, step_size, max_length / 2.0, w, h, stop_fn=is_too_close
            )
            line_backward = cls._integrate_half_line(
                sx, sy, -1.0, mask, get_flow_vector, step_size, max_length / 2.0, w, h, stop_fn=is_too_close
            )

            # Combina: backward invertido + semente + forward
            full_line = list(reversed(line_backward)) + [(sx, sy)] + line_forward

            # Elimina duplicatas consecutivas
            filtered_line: List[Tuple[int, int]] = []
            for pt in full_line:
                if not filtered_line or pt != filtered_line[-1]:
                    filtered_line.append(pt)

            # Calcula comprimento total
            length = 0.0
            for i in range(len(filtered_line) - 1):
                p1 = filtered_line[i]
                p2 = filtered_line[i + 1]
                length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

            if length >= min_length and len(filtered_line) >= 2:
                streamlines.append(filtered_line)
                mark_occupied(filtered_line)

        return streamlines

    @staticmethod
    def _integrate_half_line(start_x: float, start_y: float, direction: float,
                             mask: np.ndarray,
                             vector_fn,
                             step_size: float,
                             max_len: float,
                             width: int, height: int,
                             stop_fn=None) -> List[Tuple[int, int]]:
        """
        Integra um ramo da streamline usando Runge-Kutta de 2ª ordem (Heun).
        stop_fn(x, y) -> True interrompe o ramo (ex.: aproximou-se de uma streamline já aceita).
        """
        pts: List[Tuple[int, int]] = []
        curr_x = float(start_x)
        curr_y = float(start_y)
        accum_len = 0.0

        prev_vx, prev_vy = 0.0, 0.0

        while accum_len < max_len:
            # Vetor em (curr_x, curr_y)
            vx, vy = vector_fn(curr_x, curr_y)
            vx *= direction
            vy *= direction

            # Garante consistência de sinal para evitar inversão brusca de sentido (180º)
            if prev_vx != 0.0 or prev_vy != 0.0:
                dot = vx * prev_vx + vy * prev_vy
                if dot < 0:
                    vx = -vx
                    vy = -vy

            prev_vx, prev_vy = vx, vy

            # Ponto intermediário (Euler predictor)
            mid_x = curr_x + 0.5 * step_size * vx
            mid_y = curr_y + 0.5 * step_size * vy

            # Vetor no ponto intermediário
            mid_vx, mid_vy = vector_fn(mid_x, mid_y)
            mid_vx *= direction
            mid_vy *= direction
            if mid_vx * prev_vx + mid_vy * prev_vy < 0:
                mid_vx = -mid_vx
                mid_vy = -mid_vy

            # Passo final (Corrector)
            next_x = curr_x + step_size * mid_vx
            next_y = curr_y + step_size * mid_vy

            ix = int(round(next_x))
            iy = int(round(next_y))

            # Verifica limites da tela e da máscara
            if not (0 <= ix < width and 0 <= iy < height):
                break
            if mask[iy, ix] == 0:
                break
            if stop_fn is not None and stop_fn(ix, iy):
                break

            pts.append((ix, iy))
            accum_len += step_size
            curr_x = next_x
            curr_y = next_y

        return pts
