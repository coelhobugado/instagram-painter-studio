"""
Extrator de Stencil Estrutural com Ramer-Douglas-Peucker (RDP) Adaptativo por Saliência.
Implementa a Seção 25 (RDP Adaptativo) e Seção 26 (Edge Hierarchy) do Plano de Evolução Completo.
"""
import cv2
import numpy as np
from typing import List, Tuple

class AdaptiveStencilExtractor:
    @staticmethod
    def extract_contours(img_bgr: np.ndarray, saliency_map: np.ndarray = None) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int, int]]]:
        """
        Extrai contornos estruturais finos com simplificação poligonal RDP dinâmica.
        Retorna lista de tuplas: (pontos_do_traço, cor_rgb_do_traço)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Filtro bilateral preservando bordas
        filtered = cv2.bilateralFilter(gray, 7, 50, 50)
        
        # 2. CLAHE para reforçar detalhes faciais locais
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        # 3. Difference of Gaussians (DoG)
        g1 = cv2.GaussianBlur(enhanced, (3, 3), 1.0)
        g2 = cv2.GaussianBlur(enhanced, (9, 9), 2.5)
        dog = cv2.subtract(g1, g2)
        _, dog_th = cv2.threshold(dog, 12, 255, cv2.THRESH_BINARY)

        canny = cv2.Canny(enhanced, 60, 160)
        combined = cv2.bitwise_or(dog_th, canny)

        # Limpeza morfológica
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        result_strokes = []
        h, w = gray.shape

        for cnt in contours:
            arc_len = cv2.arcLength(cnt, False)
            if arc_len < 15:
                continue

            # Calcula a saliência média ao longo do contorno
            pts_sample = cnt.reshape(-1, 2)
            if saliency_map is not None:
                ys = np.clip(pts_sample[:, 1], 0, h - 1)
                xs = np.clip(pts_sample[:, 0], 0, w - 1)
                mean_sal = float(np.mean(saliency_map[ys, xs]))
                if mean_sal < 0.20:
                    continue
            else:
                mean_sal = 0.5

            # RDP Adaptativo (Seção 25):
            # Saliência alta -> epsilon 0.8 (preserva máxima curvatura de olhos e boca)
            # Saliência média -> epsilon 1.4
            # Saliência baixa -> epsilon 2.2
            if mean_sal > 0.65:
                epsilon = 0.8
            elif mean_sal > 0.35:
                epsilon = 1.3
            else:
                epsilon = 2.0

            approx = cv2.approxPolyDP(cnt, epsilon, False)
            pts_list = [tuple(p) for p in approx.reshape(-1, 2).tolist()]
            if len(pts_list) < 2:
                continue

            # Edge Hierarchy Harmônica:
            # Extrai a cor local média escurecida de forma sutil (preserva matiz, sem riscar de branco)
            ys = np.clip(pts_sample[:, 1], 0, h - 1)
            xs = np.clip(pts_sample[:, 0], 0, w - 1)
            local_bgr = np.mean(img_bgr[ys, xs], axis=0)
            b, g, r = float(local_bgr[0]), float(local_bgr[1]), float(local_bgr[2])

            dr = int(np.clip(r * 0.45, 10, 220))
            dg = int(np.clip(g * 0.45, 10, 220))
            db = int(np.clip(b * 0.45, 10, 220))
            stroke_rgb = (dr, dg, db)

            result_strokes.append((pts_list, stroke_rgb))

        return result_strokes
