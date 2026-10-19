# -*- coding: utf-8 -*-
"""
Brush Response Model — Bancada e Modelo de Calibração Física do Pincel (FASE 4 do Roadmap).
Mede e prediz o comportamento físico das ferramentas do Instagram:
- Curvas de dependência velocidade -> largura (100 a 1200 px/s).
- Acúmulo de passadas sucessivas (Marker multi-pass).
- Perfil transversal do traço (brush_profile(x)).
- Decomposição física de Neon (Core intenso, Halo difuso e Falloff).
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional

@dataclass
class PhysicalBrushProfile:
    """Perfil transversal de intensidade de uma pincelada a partir do centro x=0."""
    radius_px: float
    samples: np.ndarray  # Vetor 1D de opacidade [0.0 a 1.0] do centro à borda

    def sample_at(self, dist_from_center: float) -> float:
        if dist_from_center >= self.radius_px or len(self.samples) == 0:
            return 0.0
        idx = int((dist_from_center / self.radius_px) * (len(self.samples) - 1))
        idx = max(0, min(len(self.samples) - 1, idx))
        return float(self.samples[idx])

@dataclass
class BrushResponse:
    """Saída física prevista do modelo de resposta do pincel."""
    predicted_width: float
    opacity: float
    profile: PhysicalBrushProfile
    result_color_bgr: Tuple[int, int, int]
    neon_components: Optional[Dict[str, Any]] = None

class BrushResponseModel:
    """
    Modelo calibrado que simula o Instagram como um sistema físico real.
    Entrada: tool, slider, velocity, passes, color, background.
    Saída: predicted_width, opacity, profile, result_color.
    """

    # Larguras nominais de base calibradas no hardware (SM-A736B)
    BASE_WIDTHS = {
        "solid": {"min": 10.0, "med": 52.0, "thick": 98.0},
        "marker": {"min": 8.0, "med": 24.0, "thick": 55.0},
        "neon": {"min": 4.5, "med": 14.0, "thick": 32.0},
        "eraser": {"min": 10.0, "med": 35.0, "thick": 80.0}
    }

    # Opacidades base por passada
    BASE_OPACITY = {
        "solid": 1.0,
        "marker": 0.52,
        "neon": 0.95,
        "eraser": 1.0
    }

    @classmethod
    def calculate_velocity_width_factor(cls, tool: str, velocity_px_s: float) -> float:
        """
        Fase 4.1: Curva velocidade -> largura.
        Em velocidades muito altas (> 600 px/s), o traço sofre ligeiro afilamento hidrodinâmico no Instagram.
        """
        v = max(0.0, float(velocity_px_s))
        if tool == "marker":
            # Marker afina cerca de 10% a 1000 px/s
            if v > 600:
                factor = 1.0 - min(0.18, (v - 600) / 3000.0)
                return max(0.82, factor)
        elif tool == "solid":
            if v > 800:
                factor = 1.0 - min(0.08, (v - 800) / 4000.0)
                return max(0.92, factor)
        return 1.0

    @classmethod
    def calculate_multi_pass_opacity(cls, tool: str, passes: int) -> float:
        """
        Fase 4.2: Acúmulo de passadas para Marker via saturação exponencial (Beer-Lambert).
        """
        base_alpha = cls.BASE_OPACITY.get(tool, 1.0)
        if tool != "marker" or passes <= 1:
            return base_alpha

        # Saturação progressiva: 1 - (1 - alpha)^p
        accumulated = 1.0 - ((1.0 - base_alpha) ** passes)
        return min(0.98, float(accumulated))

    @classmethod
    def generate_transversal_profile(cls, tool: str, radius_px: float) -> PhysicalBrushProfile:
        """
        Fase 4.3 & 4.4: Gera o perfil transversal contínuo do centro à borda.
        """
        num_samples = 32
        r_grid = np.linspace(0.0, 1.0, num_samples)

        if tool == "solid":
            # Perfil firme com difusão suave de 1px na borda externa
            samples = np.where(r_grid < 0.85, 1.0, (1.0 - r_grid) / 0.15)
        elif tool == "marker":
            # Perfil ligeiramente plano com bordas semi-transparentes
            samples = np.where(r_grid < 0.70, 0.95, 0.95 * (1.0 - (r_grid - 0.70) / 0.30))
        elif tool == "neon":
            # Núcleo denso + halo exponencial
            samples = np.exp(-3.0 * (r_grid ** 2))
        else:
            samples = np.ones(num_samples)

        return PhysicalBrushProfile(radius_px=radius_px, samples=np.clip(samples, 0.0, 1.0))

    @classmethod
    def predict_response(cls, tool: str = "solid",
                         slider_thickness: str = "min",
                         velocity_px_s: float = 300.0,
                         passes: int = 1,
                         color_rgb: Tuple[int, int, int] = (255, 255, 255),
                         background_bgr: Tuple[int, int, int] = (22, 22, 22)) -> BrushResponse:
        """
        Prediz rigorosamente largura, opacidade, perfil e cor resultante (Fase 4.5).
        """
        from .brush_width import parse_width_level
        exact = parse_width_level(slider_thickness)
        base_w = exact if exact is not None else cls.BASE_WIDTHS.get(tool, {}).get(slider_thickness, 3.2)
        v_factor = cls.calculate_velocity_width_factor(tool, velocity_px_s)
        predicted_w = round(base_w * v_factor, 2)
        radius = predicted_w / 2.0

        opacity = cls.calculate_multi_pass_opacity(tool, passes)
        profile = cls.generate_transversal_profile(tool, radius)

        # Cor BGR e Composição Alfa sobre o Fundo
        r, g, b = color_rgb
        fg_bgr = np.array([b, g, r], dtype=np.float32)
        bg_bgr = np.array(background_bgr, dtype=np.float32)

        # Mistura alfa
        blended = fg_bgr * opacity + bg_bgr * (1.0 - opacity)
        res_color = (
            int(round(np.clip(blended[0], 0, 255))),
            int(round(np.clip(blended[1], 0, 255))),
            int(round(np.clip(blended[2], 0, 255)))
        )

        neon_meta = None
        if tool == "neon":
            neon_meta = {
                "core_radius_px": round(radius * 0.35, 1),
                "halo_radius_px": round(radius * 2.6, 1),
                "core_color_bgr": (255, 255, 255),
                "falloff_rate": 2.8
            }

        return BrushResponse(
            predicted_width=predicted_w,
            opacity=opacity,
            profile=profile,
            result_color_bgr=res_color,
            neon_components=neon_meta
        )
