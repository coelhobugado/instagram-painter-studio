"""
Módulo de Formas Negativas e Pintura Subtrativa via Borracha.
Implementa a Fase 15 (Pintura All-Inclusiva + Subtração por Borracha para Brilhos e Acentos Críticos) do Roadmap.
"""
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class NegativeShapeEngine:
    """
    Motor de Detalhes Negativos (Fase 15).
    Identifica áreas onde a tinta deve ser removida pela ferramenta 'eraser' (borracha)
    para abrir passagens de luz impossíveis de atingir por superposição com tinta translúcida.
    """

    @classmethod
    def identify_negative_regions(
        cls,
        target_img_bgr: np.ndarray,
        base_painted_canvas: np.ndarray,
        contrast_threshold: float = 40.0,
    ) -> np.ndarray:
        """
        Identifica regiões onde o alvo é substancialmente mais claro que a base já pintada,
        mas o pigmento sobreposto não consegue cobrir adequadamente.
        Retorna uma máscara binária (uint8) das regiões negativas a apagar.
        """
        target_gray = cv2.cvtColor(target_img_bgr, cv2.COLOR_BGR2GRAY)
        base_gray = cv2.cvtColor(base_painted_canvas, cv2.COLOR_BGR2GRAY)

        # Regiões onde o alvo é muito mais brilhante que a base
        diff_bright = np.maximum(0, target_gray.astype(np.int16) - base_gray.astype(np.int16))
        neg_mask = np.where(diff_bright > contrast_threshold, 255, 0).astype(np.uint8)

        # Filtra apenas pequenas ilhas, frestas e reflexos (morfologia)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        neg_cleaned = cv2.morphologyEx(neg_mask, cv2.MORPH_OPEN, kernel)
        return neg_cleaned

    @classmethod
    def generate_negative_eraser_strokes(
        cls,
        negative_mask: np.ndarray,
        eraser_width: float = 8.0,
        max_stroke_length: int = 50,
    ) -> List[List[Tuple[int, int]]]:
        """
        Gera os traços de borracha (pen_eraser) para limpar as ilhas negativas.
        """
        if np.count_nonzero(negative_mask) == 0:
            return []

        contours, _ = cv2.findContours(negative_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        eraser_strokes: List[List[Tuple[int, int]]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:
                continue

            # Se for uma ilha pequena compacta, gera um ponto/toque central
            if area < 60:
                m = cv2.moments(cnt)
                if m["m00"] > 0:
                    cx = int(round(m["m10"] / m["m00"]))
                    cy = int(round(m["m01"] / m["m00"]))
                    eraser_strokes.append([(cx - 1, cy), (cx + 1, cy)])
            else:
                # Se for linear ou fresta, segue o esqueleto interno
                skel_local = np.zeros(negative_mask.shape, dtype=np.uint8)
                cv2.drawContours(skel_local, [cnt], -1, 255, -1)
                # Centroide e linha principal
                rect = cv2.minAreaRect(cnt)
                (rx, ry), (rw, rh), angle = rect
                cx, cy = int(round(rx)), int(round(ry))
                rad = np.radians(angle)
                half_l = min(max_stroke_length, max(rw, rh) / 2.0)
                dx = int(round(np.cos(rad) * half_l))
                dy = int(round(np.sin(rad) * half_l))
                eraser_strokes.append([(cx - dx, cy - dy), (cx + dx, cy + dy)])

        return eraser_strokes

    @classmethod
    def build_eraser_layer(
        cls, eraser_strokes: List[List[Tuple[int, int]]], eraser_thickness: str = "min"
    ) -> Optional[Dict[str, Any]]:
        """
        Monta a camada estruturada de borracha com a ferramenta 'eraser'.
        """
        if not eraser_strokes:
            return None

        total_pts = sum(len(s) for s in eraser_strokes)
        return {
            "name": "Camada Subtrativa: Borracha (Negative Shapes)",
            "brush": "eraser",
            "purpose": "negative_stroke",
            "color_rgb": (255, 255, 255),
            "thickness": eraser_thickness,
            "strokes": eraser_strokes,
            "stroke_count": len(eraser_strokes),
            "point_count": total_pts,
        }
