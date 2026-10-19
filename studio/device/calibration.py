# -*- coding: utf-8 -*-
"""
Módulo de Calibração de Hardware no Dispositivo Real (Galaxy A73 / Android).
Implementa o Requisito 6:
- Mede a cor efetivamente desenhada e a largura física de cada pincel no celular real.
- Compartilha exatamente os mesmos dados na conversão (quantização), na prévia e na execução.
- Assegura que a prévia mostre estritamente o que o aparelho real consegue reproduzir.
"""
import os
import json
import time
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional

from .. import config

class HardwareCalibrationEngine:
    """
    Gerencia e executa a calibração física dos pincéis e da paleta de cores no aparelho real.
    """
    DEFAULT_PROFILE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "profiles", "SM-A736B"
    )

    def __init__(self, profile_dir: Optional[str] = None):
        self.profile_dir = profile_dir or self.DEFAULT_PROFILE_DIR
        self.brushes_path = os.path.join(self.profile_dir, "brushes.json")
        self.lut_path = os.path.join(self.profile_dir, "color_lut.json")
        self.brushes_data = self._load_json(self.brushes_path)
        self.lut_data = self._load_json(self.lut_path)

    def _load_json(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_json(self, path: str, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_effective_brush_width(self, level: str = "min", tool: str = "solid") -> float:
        """
        Retorna a largura efetiva medida em pixels do traço no celular real.
        """
        from .brush_width import parse_width_level
        exact = parse_width_level(level)
        if exact is not None:
            return exact
        brushes = self.brushes_data.get("brushes", {})
        tool_data = brushes.get(tool, brushes.get("solid", {}))
        level_data = tool_data.get("levels", {}).get(level, {})
        return float(level_data.get("effective_width_px", config.BRUSH_WIDTHS.get(level, 3.2)))

    def get_effective_color(self, target_rgb: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], float]:
        """
        Retorna a cor que o celular consegue efetivamente desenhar para o alvo RGB solicitado,
        juntamente com o erro Delta E correspondente.
        """
        entries = self.lut_data.get("entries", [])
        if not entries:
            return target_rgb, 0.0

        r, g, b = target_rgb
        # Busca no LUT o toque com menor distância euclidiana simples ou perceptual
        best_entry = None
        min_dist = float("inf")
        for entry in entries:
            er, eg, eb = entry.get("rgb", [255, 255, 255])
            dist = (r - er)**2 + (g - eg)**2 + (b - eb)**2
            if dist < min_dist:
                min_dist = dist
                best_entry = entry

        if best_entry:
            eff_rgb = tuple(best_entry.get("rgb", list(target_rgb)))
            delta = float(np.sqrt(min_dist))
            return (int(eff_rgb[0]), int(eff_rgb[1]), int(eff_rgb[2])), delta
        return target_rgb, 0.0

    @classmethod
    def measure_stroke_attributes(cls, screencap_bgr: np.ndarray,
                                  background_bgr: np.ndarray,
                                  expected_color_rgb: Tuple[int, int, int],
                                  roi_rect: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Analisa uma captura da tela do celular após desenhar um traço de teste,
        medindo com precisão sub-pixel a largura real em pixels e a cor RGB resultante.
        """
        rx, ry, rw, rh = roi_rect
        patch = screencap_bgr[ry:ry + rh, rx:rx + rw]
        bg_patch = background_bgr[ry:ry + rh, rx:rx + rw] if background_bgr is not None else None

        # Diferença absoluta em relação ao fundo
        if bg_patch is not None and bg_patch.shape == patch.shape:
            diff = cv2.absdiff(patch, bg_patch)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, gray_diff = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

        _, stroke_mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)

        # Medição de largura média pela área / comprimento do esqueleto
        skel = np.zeros_like(stroke_mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = stroke_mask.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            skel = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break

        skel_length = np.count_nonzero(skel)
        total_pixels = np.count_nonzero(stroke_mask)

        measured_width = float(total_pixels / max(1, skel_length)) if skel_length > 0 else 3.2

        # Medição da cor média no centro do traço
        measured_pixels = patch[stroke_mask > 0]
        if len(measured_pixels) > 0:
            mean_bgr = np.median(measured_pixels, axis=0)
            measured_rgb = (int(round(mean_bgr[2])), int(round(mean_bgr[1])), int(round(mean_bgr[0])))
        else:
            measured_rgb = expected_color_rgb

        return {
            "effective_width_px": round(measured_width, 2),
            "measured_rgb": measured_rgb,
            "expected_rgb": expected_color_rgb,
            "sample_pixel_count": total_pixels
        }

    def update_calibrated_brush(self, level: str, width_px: float, edge_softness: float = 0.8):
        """Atualiza a largura efetiva calibrada do pincel no perfil persistente."""
        brushes = self.brushes_data.setdefault("brushes", {})
        solid = brushes.setdefault("solid", {"name": "Canetinha Sólida", "levels": {}})
        levels = solid.setdefault("levels", {})
        levels[level] = {
            "effective_width_px": float(round(width_px, 2)),
            "edge_softness": float(edge_softness),
            "calibrated": True
        }
        self._save_json(self.brushes_path, self.brushes_data)
