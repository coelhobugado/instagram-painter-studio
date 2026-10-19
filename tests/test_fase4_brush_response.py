# -*- coding: utf-8 -*-
"""
Testes Unitários da FASE 4: Bancada e Modelo de Resposta Física do Pincel.
Em estrita conformidade com o INSTAGRAM_PAINTER_ROADMAP_APICE.md:
- 4.1: Curvas velocidade -> largura.
- 4.2: Acúmulo de passadas para Marker (1, 2, 3 passes).
- 4.3: Perfil transversal brush_profile(x).
- 4.4: Modelagem de Neon (core, halo, falloff).
- 4.5: Estrutura e predição BrushResponseModel.
"""
import unittest
from studio.device.brush_response import BrushResponseModel

class TestFase4BrushResponse(unittest.TestCase):

    def test_4_1_velocity_width_factor(self):
        """4.1: Valida que velocidade elevada provoca redução controlada de largura no Marker."""
        w_slow = BrushResponseModel.calculate_velocity_width_factor("marker", 100.0)
        w_fast = BrushResponseModel.calculate_velocity_width_factor("marker", 1200.0)
        self.assertEqual(w_slow, 1.0)
        self.assertLess(w_fast, 1.0)
        self.assertGreaterEqual(w_fast, 0.80)

    def test_4_2_multi_pass_marker_accumulation(self):
        """4.2: Valida que passadas adicionais aumentam a opacidade acumulada."""
        alpha_1 = BrushResponseModel.calculate_multi_pass_opacity("marker", 1)
        alpha_2 = BrushResponseModel.calculate_multi_pass_opacity("marker", 2)
        alpha_4 = BrushResponseModel.calculate_multi_pass_opacity("marker", 4)

        self.assertAlmostEqual(alpha_1, 0.52, places=2)
        self.assertGreater(alpha_2, alpha_1)
        self.assertGreater(alpha_4, alpha_2)
        self.assertLessEqual(alpha_4, 1.0)

    def test_4_3_transversal_profile(self):
        """4.3: Valida a amostragem do perfil transversal de intensidade do traço."""
        profile = BrushResponseModel.generate_transversal_profile("solid", radius_px=10.0)
        val_center = profile.sample_at(0.0)
        val_edge = profile.sample_at(9.8)
        val_outside = profile.sample_at(10.5)

        self.assertAlmostEqual(val_center, 1.0, places=2)
        self.assertLess(val_edge, val_center)
        self.assertEqual(val_outside, 0.0)

    def test_4_4_neon_decomposition(self):
        """4.4: Valida separação entre núcleo brilhante e halo difuso do Neon."""
        resp = BrushResponseModel.predict_response(
            tool="neon",
            slider_thickness="med",
            color_rgb=(0, 255, 255)  # Ciano
        )
        self.assertIsNotNone(resp.neon_components)
        meta = resp.neon_components
        self.assertIn("core_radius_px", meta)
        self.assertIn("halo_radius_px", meta)
        self.assertGreater(meta["halo_radius_px"], meta["core_radius_px"])

    def test_4_5_predict_response_complete_schema(self):
        """4.5: Valida predição integral e composição alfa com o fundo."""
        resp = BrushResponseModel.predict_response(
            tool="solid",
            slider_thickness="min",
            color_rgb=(255, 0, 0),  # Vermelho puro
            background_bgr=(0, 0, 0)
        )
        self.assertEqual(resp.predicted_width, 10.0)
        self.assertEqual(resp.opacity, 1.0)
        # BGR: (0, 0, 255)
        self.assertEqual(resp.result_color_bgr, (0, 0, 255))

if __name__ == "__main__":
    unittest.main()
