"""
Calculador de Mapas de Erro Perceptual CIE Lab e Métricas Quantitativas (SSIM / True DeltaE CIEDE2000).
Implementa a Seção 29 (Error Map) e Seção 30 (Métricas) do Plano de Evolução Completo com formulação analítica exata.
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple

class ErrorMapEngine:
    @staticmethod
    def ciede2000_vectorized(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
        """
        Cálculo vetorizado estrito da fórmula CIEDE2000 (ISO/CIE 11664-6:2014) sobre imagens 2D float32.
        Entrada:
            lab1, lab2: matrizes (H, W, 3) com L* em [0, 100], a* e b* com sinais reais.
        Retorno:
            delta_e_map: matriz (H, W) com a distância perceptual exata CIEDE2000 em cada pixel.
        """
        L1, a1, b1 = lab1[:, :, 0], lab1[:, :, 1], lab1[:, :, 2]
        L2, a2, b2 = lab2[:, :, 0], lab2[:, :, 1], lab2[:, :, 2]

        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_bar = (C1 + C2) / 2.0

        C_bar7 = C_bar**7
        G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7 + 1e-12)))

        a1_prime = (1.0 + G) * a1
        a2_prime = (1.0 + G) * a2

        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)

        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime

        # Diferença angular do matiz
        h_diff = h2_prime - h1_prime
        abs_h_diff = np.abs(h_diff)

        delta_h_prime = np.where(
            (C1_prime * C2_prime) == 0,
            0.0,
            np.where(
                abs_h_diff <= 180.0,
                h_diff,
                np.where(h2_prime <= h1_prime, h_diff + 360.0, h_diff - 360.0)
            )
        )

        delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

        L_bar_prime = (L1 + L2) / 2.0
        C_bar_prime = (C1_prime + C2_prime) / 2.0

        # Média angular do matiz
        h_sum = h1_prime + h2_prime
        h_bar_prime = np.where(
            (C1_prime * C2_prime) == 0,
            h_sum,
            np.where(
                abs_h_diff <= 180.0,
                h_sum / 2.0,
                np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0)
            )
        )

        # Funções de ponderação
        T = (1.0 - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
             + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0)))

        delta_theta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0)**2)
        C_bar_prime7 = C_bar_prime**7
        R_C = 2.0 * np.sqrt(C_bar_prime7 / (C_bar_prime7 + 25.0**7 + 1e-12))
        S_L = 1.0 + (0.015 * (L_bar_prime - 50.0)**2) / np.sqrt(20.0 + (L_bar_prime - 50.0)**2)
        S_C = 1.0 + 0.045 * C_bar_prime
        S_H = 1.0 + 0.015 * C_bar_prime * T
        R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

        term_L = delta_L_prime / S_L
        term_C = delta_C_prime / S_C
        term_H = delta_H_prime / S_H

        delta_E00 = np.sqrt(term_L**2 + term_C**2 + term_H**2 + R_T * term_C * term_H)
        return delta_E00.astype(np.float32)

    @classmethod
    def compute_error_map(cls, target_bgr: np.ndarray,
                          rendered_bgr: np.ndarray,
                          saliency_map: np.ndarray = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calcula o mapa de calor de erro perceptual estrito e métricas estatísticas exatas (True CIEDE2000).
        Retorna:
            - error_heatmap_bgr: Imagem colorida em escala de calor (Turbo) representando o erro
            - metrics: Dicionário com Mean DeltaE, P95 DeltaE, SSIM e Saliency-Weighted Error
        """
        target_h, target_w = target_bgr.shape[:2]
        rend_resized = cv2.resize(rendered_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 1. Conversão para espaço CIE Lab analítico em float32 (L: [0, 100], a,b assinados)
        target_rgb_f = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rend_rgb_f = cv2.cvtColor(rend_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        target_lab = cv2.cvtColor(target_rgb_f, cv2.COLOR_RGB2Lab)
        rend_lab = cv2.cvtColor(rend_rgb_f, cv2.COLOR_RGB2Lab)

        # 2. Cálculo do Mapa Exato de CIEDE2000
        delta_e_map = cls.ciede2000_vectorized(target_lab, rend_lab)

        mean_de = float(np.mean(delta_e_map))
        p95_de = float(np.percentile(delta_e_map, 95))

        # 3. Erro Ponderado por Saliência (Seção 29: saliency-weighted error)
        if saliency_map is not None:
            sal_resized = cv2.resize(saliency_map, (target_w, target_h), interpolation=cv2.INTER_AREA)
            weighted_error = float(np.sum(delta_e_map * sal_resized) / (np.sum(sal_resized) + 1e-8))
        else:
            weighted_error = mean_de

        # 4. SSIM (Structural Similarity Index) com janela gaussiana 11x11
        c1_ssim = (0.01 * 255.0)**2
        c2_ssim = (0.03 * 255.0)**2
        t_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r_gray = cv2.cvtColor(rend_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

        mu1 = cv2.GaussianBlur(t_gray, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(r_gray, (11, 11), 1.5)
        sigma1_sq = cv2.GaussianBlur(t_gray**2, (11, 11), 1.5) - mu1**2
        sigma2_sq = cv2.GaussianBlur(r_gray**2, (11, 11), 1.5) - mu2**2
        sigma12 = cv2.GaussianBlur(t_gray * r_gray, (11, 11), 1.5) - mu1 * mu2

        ssim_map = ((2.0 * mu1 * mu2 + c1_ssim) * (2.0 * sigma12 + c2_ssim)) / \
                   ((mu1**2 + mu2**2 + c1_ssim) * (sigma1_sq + sigma2_sq + c2_ssim))
        ssim_val = float(np.clip(np.mean(ssim_map), 0.0, 1.0))

        # 5. Geração do Heatmap Visual (0 a 30 de DeltaE00 mapeado para cores Turbo)
        norm_de = np.clip(delta_e_map / 30.0, 0.0, 1.0) * 255.0
        norm_de_uint8 = norm_de.astype(np.uint8)
        error_heatmap = cv2.applyColorMap(norm_de_uint8, cv2.COLORMAP_TURBO)

        metrics = {
            "mean_delta_e": round(mean_de, 2),
            "p95_delta_e": round(p95_de, 2),
            "saliency_weighted_de": round(weighted_error, 2),
            "ssim": round(ssim_val, 3)
        }

        return error_heatmap, metrics
