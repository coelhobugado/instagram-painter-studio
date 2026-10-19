"""
Tensor de Estrutura 2D para Orientação Direcional de Textura e Pinceladas.
Implementa a Seção 22 do Plano de Evolução Completo (Texture Engine).
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class TextureTensorEngine:
    @staticmethod
    def compute_orientation_field(img_bgr: np.ndarray, blur_sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula o campo de orientação de textura baseado no Tensor de Estrutura.
        Retorna:
            - angles_deg: Matriz com o ângulo dominante do fluxo (-90° a +90°)
            - coherence: Matriz com o grau de anisotropia/alinhamento (0.0 = isotrópico/liso, 1.0 = estritamente alinhado/fios de cabelo)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 1. Gradientes espaciais
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # 2. Componentes do tensor
        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        # 3. Integração em vizinhança Gaussiana
        ksize = int(round(blur_sigma * 4)) | 1
        Jxx = cv2.GaussianBlur(Ixx, (ksize, ksize), blur_sigma)
        Jyy = cv2.GaussianBlur(Iyy, (ksize, ksize), blur_sigma)
        Jxy = cv2.GaussianBlur(Ixy, (ksize, ksize), blur_sigma)

        # 4. Análise de autovetores para encontrar o fluxo da textura
        # Ângulo ortogonal ao gradiente (direção do fluxo dos fios)
        angles_rad = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + (np.pi / 2.0)
        angles_deg = np.degrees(angles_rad) % 180.0 - 90.0

        # Coerência (grau de anisotropia)
        term1 = Jxx + Jyy
        term2 = np.sqrt((Jxx - Jyy)**2 + 4.0 * (Jxy**2))
        coherence = np.where(term1 > 1e-6, term2 / (term1 + 1e-6), 0.0)
        coherence = np.clip(coherence, 0.0, 1.0)

        return angles_deg, coherence

    @staticmethod
    def get_region_dominant_angle(angles_deg: np.ndarray, coherence: np.ndarray, mask: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
        """Determina o ângulo de preenchimento ótimo para a região."""
        if bbox is not None:
            bx, by, bw, bh = bbox
            sub_coh = coherence[by:by + bh, bx:bx + bw]
            sub_ang = angles_deg[by:by + bh, bx:bx + bw]
            sub_mask = mask[by:by + bh, bx:bx + bw] if mask.shape == coherence.shape else mask
            reg_coherence = sub_coh[sub_mask > 0]
            reg_angles = sub_ang[sub_mask > 0]
        else:
            reg_coherence = coherence[mask > 0]
            reg_angles = angles_deg[mask > 0]
        
        if len(reg_coherence) == 0:
            return 0.0
            
        mean_coherence = np.mean(reg_coherence)
        if mean_coherence < 0.25:
            # Baixa textura: usa varredura horizontal padrão (0 graus)
            return 0.0
            
        # Média circular ponderada pela coerência
        weights = reg_coherence / (np.sum(reg_coherence) + 1e-8)
        rads = 2.0 * np.radians(reg_angles)
        sin_sum = np.sum(weights * np.sin(rads))
        cos_sum = np.sum(weights * np.cos(rads))
        
        dom_angle = 0.5 * np.degrees(np.arctan2(sin_sum, cos_sum))
        return float(dom_angle)
