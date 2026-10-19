"""
Planejador de Correções Autônomo com Máscara de Proteção e Análise de Risco.
Implementa a Seção 31 (Correction Planner), Seção 32 (Protected Mask) e Seção 33/34 (Risk & Stop Criterion).
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ..vision.perceptual_color import PerceptualColorEngine
from ..planning.fill_strategies import FillStrategyEngine

class CorrectionPlanner:
    def __init__(self, color_engine: PerceptualColorEngine = None):
        self.color_engine = color_engine or PerceptualColorEngine()

    def plan_corrections(self, target_bgr: np.ndarray,
                         observed_bgr: np.ndarray,
                         offset_xy: Tuple[int, int],
                         max_correction_strokes: int = 40) -> Dict[str, Any]:
        """
        Analisa discrepâncias entre o que foi observado na tela do celular e o alvo,
        congelando regiões já perfeitas e sintetizando traços de retoque estritamente necessários.
        """
        h, w = target_bgr.shape[:2]
        obs_resized = cv2.resize(observed_bgr, (w, h), interpolation=cv2.INTER_AREA)

        # 1. Mapa de Erro Delta E
        target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        obs_lab = cv2.cvtColor(obs_resized, cv2.COLOR_BGR2Lab).astype(np.float32)

        diff = np.sqrt(np.sum((target_lab - obs_lab)**2, axis=2))

        # 2. Máscara de Proteção (Seção 32): Áreas com erro muito baixo são blindadas
        protected_mask = (diff < 4.0).astype(np.uint8) * 255

        # 3. Máscara de Falhas Críticas (onde há falta de tinta ou erro alto e NÃO está protegida)
        error_mask = ((diff > 8.0) & (protected_mask == 0)).astype(np.uint8) * 255

        # Limpeza morfológica para agrupar micro-falhas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        error_cleaned = cv2.morphologyEx(error_mask, cv2.MORPH_OPEN, kernel)

        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(error_cleaned, connectivity=8)

        correction_layers = []
        total_correction_strokes = 0
        off_x, off_y = offset_xy

        for comp_id in range(1, num_labels):
            if total_correction_strokes >= max_correction_strokes:
                break

            area = stats[comp_id, cv2.CC_STAT_AREA]
            # Correction Risk (Seção 33): Descarta ilhas minúsculas que trariam risco sem ganho
            if area < 30:
                continue

            comp_mask = (comp_labels == comp_id).astype(np.uint8) * 255
            rx = stats[comp_id, cv2.CC_STAT_LEFT]
            ry = stats[comp_id, cv2.CC_STAT_TOP]
            rw = stats[comp_id, cv2.CC_STAT_WIDTH]
            rh = stats[comp_id, cv2.CC_STAT_HEIGHT]

            # Cor média alvo na região falha
            target_patch = target_bgr[comp_mask > 0]
            mean_b = int(np.mean(target_patch[:, 0]))
            mean_g = int(np.mean(target_patch[:, 1]))
            mean_r = int(np.mean(target_patch[:, 2]))
            target_rgb = (mean_r, mean_g, mean_b)

            match_info = self.color_engine.match_color(target_rgb)

            # Gera traços de preenchimento serpenteante de correção
            local_strokes = FillStrategyEngine.generate_region_strokes(
                comp_mask, (rx, ry, rw, rh), strategy="horizontal_scanline", brush_spacing=4
            )

            scr_strokes = []
            for stroke in local_strokes:
                scr_s = [(off_x + pt[0], off_y + pt[1]) for pt in stroke]
                scr_strokes.append(scr_s)

            if scr_strokes:
                total_correction_strokes += len(scr_strokes)
                correction_layers.append({
                    "name": f"Retoque: {match_info['name']} (Área {area}px)",
                    "color_rgb": target_rgb,
                    "color_hex": match_info["matched_hex"],
                    "touch_info": match_info,
                    "strokes": scr_strokes,
                    "stroke_count": len(scr_strokes),
                    "thickness": "min"
                })

        # Critério de parada (Seção 25/34 do Plano): se não há correções necessárias ou erro médio baixo
        mean_error = float(np.mean(diff))
        should_stop = len(correction_layers) == 0 or mean_error < 4.0

        return {
            "should_stop": should_stop,
            "mean_error": round(mean_error, 2),
            "total_correction_strokes": total_correction_strokes,
            "layers": correction_layers,               # Contrato unificado (Seção 22 do Plano)
            "correction_layers": correction_layers,    # Compatibilidade retroativa
            "protected_mask": protected_mask,
            "error_mask": error_cleaned
        }
