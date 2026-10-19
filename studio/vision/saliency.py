"""
Gerador de Mapas de Saliência Visual e Orçamento Ponderado de Detalhes.
Implementa a Seção 24 do Plano de Evolução Completo (Saliency Map).
"""
import cv2
import numpy as np
from typing import Tuple, Optional

class SaliencyEngine:
    @staticmethod
    def compute_saliency_map(img_bgr: np.ndarray) -> np.ndarray:
        """
        Calcula o mapa de saliência visual perceptivo em escala de 0.0 a 1.0.
        Combina o método de Resíduo Espectral (Frequency Domain) com gradientes locais.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        # 1. Resíduo Espectral (Spectral Residual Saliency)
        # Redimensiona para resolução ideal de análise FFT
        scale_w = 64
        scale_h = int(round(h * (scale_w / float(w))))
        small = cv2.resize(gray, (scale_w, scale_h), interpolation=cv2.INTER_AREA)

        # FFT 2D
        dft = np.fft.fft2(small)
        amplitude = np.abs(dft)
        log_amplitude = np.log(amplitude + 1e-8)
        phase = np.angle(dft)

        # Filtro de média para obter a curva espectral média
        avg_log = cv2.blur(log_amplitude, (3, 3))
        spectral_residual = log_amplitude - avg_log

        # IFFT com a fase original
        combined = np.exp(spectral_residual) * (np.cos(phase) + 1j * np.sin(phase))
        ifft = np.fft.ifft2(combined)
        saliency_small = np.abs(ifft) ** 2
        saliency_small = cv2.GaussianBlur(saliency_small, (7, 7), 2.5)

        # Normaliza e redimensiona de volta à resolução original
        saliency_spec = cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_CUBIC)
        saliency_spec = (saliency_spec - saliency_spec.min()) / (saliency_spec.max() - saliency_spec.min() + 1e-8)

        # 2. Gradientes Locais (Sobel) para reforçar bordas anatômicas e texturas
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)

        # 3. Combinação Ponderada (65% Saliência Espectral + 35% Gradiente de Bordas)
        combined_saliency = 0.65 * saliency_spec + 0.35 * grad_mag
        combined_saliency = np.clip(combined_saliency, 0.0, 1.0)
        return combined_saliency

    @staticmethod
    def get_region_saliency(saliency_map: np.ndarray, mask: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
        """Calcula a média de saliência dentro de uma região delimitada por máscara ou ROI."""
        if bbox is not None:
            bx, by, bw, bh = bbox
            sub_sal = saliency_map[by:by + bh, bx:bx + bw]
            sub_mask = mask[by:by + bh, bx:bx + bw] if mask.shape == saliency_map.shape else mask
            pixels = sub_sal[sub_mask > 0]
        else:
            pixels = saliency_map[mask > 0]
        if len(pixels) == 0:
            return 0.5
        return float(np.mean(pixels))
