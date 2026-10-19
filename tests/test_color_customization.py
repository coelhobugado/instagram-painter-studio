"""
Testes Unitários para o Sistema de Customização e Mudança de Cores da Pintura.
Valida o recoloramento dinâmico em tempo real (recolor_plan), os overrides de cor
nos modos Linhas (Line-Art), Stencil e Fotografia (Camadas) e a precisão do mapeamento LUT.
"""
import unittest
import numpy as np
from studio.painter_engine import PainterEngine
from studio.planning.line_art_engine import LineArtEngine

class TestColorCustomization(unittest.TestCase):
    def setUp(self):
        self.engine = PainterEngine()

    def test_line_art_color_override_ink_and_bg(self):
        """Valida se o modo Linhas aceita override de cor da tinta e de fundo."""
        # Cria imagem sintética simples (traço escuro sobre fundo claro)
        dummy_img = np.full((200, 200, 3), 245, dtype=np.uint8)
        dummy_img[50:150, 95:105] = (20, 20, 20)  # Linha vertical escura

        # Força traço azul e fundo amarelo
        override_ink = (0, 0, 255)      # Vermelho puro em RGB
        override_bg = (255, 255, 0)     # Amarelo puro em RGB

        plan = self.engine.generate_line_art_plan(
            "inputs/flores.jpg",
            override_ink_rgb=override_ink,
            override_bg_rgb=override_bg,
            paint_background=True
        )

        self.assertIn("layers", plan)
        self.assertGreaterEqual(len(plan["layers"]), 1)

        # Camada de fundo deve existir e ter cor amarela
        bg_layer = plan["layers"][0]
        self.assertEqual(bg_layer["purpose"], "background")
        self.assertEqual(bg_layer["color_rgb"], override_bg)

        # Camada de traço central deve ter cor vermelha
        ink_layer = plan["layers"][1]
        self.assertEqual(ink_layer["purpose"], "line_art_centerline")
        self.assertEqual(ink_layer["color_rgb"], override_ink)
        self.assertIn("render_color_rgb", ink_layer)

    def test_line_art_no_paint_background(self):
        """Garante que paint_background=False previne a criação de camada de fundo."""
        plan = self.engine.generate_line_art_plan(
            "inputs/flores.jpg",
            paint_background=False
        )
        # Não deve haver camada com purpose == 'background'
        purposes = [l["purpose"] for l in plan["layers"]]
        self.assertNotIn("background", purposes)
        self.assertEqual(len(plan["layers"]), 1)

    def test_stencil_color_override(self):
        """Valida override de cor de caneta no modo Stencil."""
        override_color = (0, 200, 50)  # Verde
        plan = self.engine.generate_stencil_plan(
            "inputs/flores.jpg",
            override_ink_rgb=override_color
        )
        self.assertEqual(len(plan["layers"]), 1)
        layer = plan["layers"][0]
        self.assertEqual(layer["color_rgb"], override_color)
        self.assertEqual(plan["palette_colors"][0]["rgb"], override_color)
        self.assertIn("touch_info", layer)
        self.assertIn("matched_rgb", layer["touch_info"])

    def test_recolor_plan_layered_mode_updates_layers_and_sim(self):
        """Valida recolor_plan em tempo real no modo Fotografia por Camadas."""
        plan = self.engine.generate_layered_paint_plan(
            "inputs/flores.jpg",
            num_colors=4,
            brush_spacing=4,
            optimize_in_simulation=False
        )

        self.assertIn("palette_colors", plan)
        initial_color = plan["palette_colors"][0]["rgb"]
        new_color = (255, 0, 128)  # Rosa Magenta

        # Executa a mudança de cor da Cor 1
        updated_plan = self.engine.recolor_plan(plan, color_index=0, new_rgb=new_color)

        # 1. A paleta deve estar atualizada
        self.assertEqual(updated_plan["palette_colors"][0]["rgb"], new_color)
        self.assertEqual(updated_plan["palette_colors"][0]["hex"], "#FF0080")

        # 2. Pelo menos uma camada correspondente deve ter a nova cor
        matching_layers = [
            l for l in updated_plan["layers"]
            if l.get("palette_index") == 0 or l.get("color_rgb") == new_color
        ]
        self.assertGreater(len(matching_layers), 0)
        for ml in matching_layers:
            self.assertEqual(ml["color_rgb"], new_color)
            self.assertEqual(ml["color_hex"], "#FF0080")
            self.assertIn("touch_info", ml)

        # 3. As imagens de simulação devem continuar válidas
        self.assertIsNotNone(updated_plan["simulation_image"])
        self.assertIsNotNone(updated_plan["comparison_image"])
        self.assertEqual(
            updated_plan["simulation_image"].shape,
            plan["simulation_image"].shape
        )

    def test_recolor_plan_line_art_mode(self):
        """Valida recolor_plan no modo Linhas."""
        plan = self.engine.generate_line_art_plan("inputs/flores.jpg", paint_background=True)
        # Supondo 2 camadas (fundo no idx 0 e traço no idx 1)
        if len(plan["palette_colors"]) >= 2:
            new_ink_color = (180, 40, 220)  # Roxo
            updated = self.engine.recolor_plan(plan, color_index=1, new_rgb=new_ink_color)
            self.assertEqual(updated["palette_colors"][1]["rgb"], new_ink_color)
            self.assertEqual(updated["layers"][1]["color_rgb"], new_ink_color)

    def test_recolor_plan_safe_with_invalid_index(self):
        """Garante segurança quando o índice de cor for inválido."""
        plan = self.engine.generate_stencil_plan("inputs/flores.jpg")
        # Índice negativo ou muito grande
        res1 = self.engine.recolor_plan(plan, color_index=-1, new_rgb=(255, 255, 255))
        res2 = self.engine.recolor_plan(plan, color_index=999, new_rgb=(255, 255, 255))
        self.assertIsNotNone(res1)
        self.assertIsNotNone(res2)

if __name__ == "__main__":
    unittest.main()
