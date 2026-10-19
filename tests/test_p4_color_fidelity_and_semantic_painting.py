# -*- coding: utf-8 -*-
"""
Testes de Qualidade P4: Fidelidade Cromática, Preservação de Matiz e Pintura Semântica por Objeto.
Validações essenciais:
1. Consistência e Fidelidade de Matiz (Hue Consistency): Azul nunca é convertido para Roxo.
2. Extração e Proteção de Cavidades Claras Internas (Olhos de Personagens em Cartoons).
3. Preenchimento Confinado com Insetting e Fechamento de Frestas Seguro.
4. Ordem e Oclusão Semântica no Caso 'personagem.jpg'.
"""
import unittest
import os
import cv2
import numpy as np

from studio.vision.perceptual_color import PerceptualColorEngine
from studio.vision.semantic_features import SemanticFeatureExtractor
from studio.planning.fill_strategies import FillStrategyEngine
from studio.painter_engine import PainterEngine


class TestP4ColorAndSemanticPainting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.miranha_path = os.path.join("inputs", "personagem.jpg")
        cls.color_engine = PerceptualColorEngine()
        cls.painter_engine = PainterEngine()

    def test_hue_consistency_penalty_preserves_blue(self):
        """Valida que o motor de cores impede a conversão de azul marinho/saturado em roxo violeta."""
        # Azul do traje do Homem-Aranha em inputs/personagem.jpg
        miranha_blue = (45, 49, 140)
        match_result = self.color_engine.match_color(miranha_blue)

        matched_name = match_result.get("name", "")
        matched_rgb = match_result.get("matched_rgb", (0, 0, 0))

        # A cor casada deve ser da família azul (B deve ser dominante em relação a R)
        self.assertNotIn("Roxo", matched_name, f"O azul do traje virou roxo: {matched_name}")
        self.assertNotIn("Violeta", matched_name, f"O azul do traje virou violeta: {matched_name}")
        self.assertGreater(matched_rgb[2], matched_rgb[0], "No espaço RGB, o componente azul (B) deve superar o vermelho (R).")

        # Azul vibrante de cartoon
        vibrant_blue = (35, 110, 225)
        vibrant_match = self.color_engine.match_color(vibrant_blue)
        v_rgb = vibrant_match.get("matched_rgb", (0, 0, 0))
        v_name = vibrant_match.get("name", "")
        self.assertNotIn("Roxo", v_name)
        self.assertNotIn("Violeta", v_name)
        self.assertGreater(v_rgb[2], v_rgb[0] + 50, "O azul vibrante casado deve ter componente B muito superior ao R.")

    def test_internal_white_features_detection_on_miranha(self):
        """Valida a detecção e isolamento das cavidades brancas internas (olhos do Homem-Aranha)."""
        if not os.path.exists(self.miranha_path):
            self.skipTest(f"Imagem ausente: {self.miranha_path}")

        img = cv2.imread(self.miranha_path)
        internal_whites = SemanticFeatureExtractor.extract_internal_white_features(img, min_val=220)

        total_white_pixels = int(np.count_nonzero(internal_whites))
        self.assertGreater(total_white_pixels, 3000,
                           f"Os olhos do Homem-Aranha devem ser detectados! Apenas {total_white_pixels}px encontrados.")

        # Valida que o fundo externo da imagem (cantos) NÃO foi incluído como feição interna
        h, w = img.shape[:2]
        self.assertEqual(int(internal_whites[0, 0]), 0, "O canto superior esquerdo (fundo) não pode ser feição interna.")
        self.assertEqual(int(internal_whites[h - 1, w - 1]), 0, "O canto inferior direito não pode ser feição interna.")

    def test_fill_insetting_confines_strokes(self):
        """Valida que o preenchimento com safe_mask respeita estritamente o insetting do pincel."""
        # Cria máscara circular sólida de raio 50px centralizada em (100, 100)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 50, 255, -1)

        brush_width = 16.0
        strokes = FillStrategyEngine.generate_region_strokes(
            mask, (50, 50, 100, 100),
            strategy="horizontal_scanline",
            brush_width=brush_width,
            brush_spacing=10,
            apply_safe_mask=True,
            ensure_full_coverage=True
        )

        self.assertGreater(len(strokes), 0, "Devem ser geradas pinceladas de preenchimento.")

        # Verifica se nenhum ponto gerado atinge a borda extrema (raio > 47px)
        for s in strokes:
            for pt in s:
                dist = np.hypot(pt[0] - 100, pt[1] - 100)
                self.assertLess(dist, 48.0,
                                f"Ponto {pt} a distância {dist:.1f}px ultrapassou o insetting do raio do pincel!")

    def test_miranha_layered_plan_structure(self):
        """Valida o plano de pintura completo do Homem-Aranha com preservação dos olhos e cores corretas."""
        if not os.path.exists(self.miranha_path):
            self.skipTest(f"Imagem ausente: {self.miranha_path}")

        plan = self.painter_engine.generate_layered_paint_plan(
            self.miranha_path, num_colors=6, seed=42, optimize_in_simulation=False
        )

        layers = plan.get("layers", [])
        self.assertGreater(len(layers), 0)

        # 1. Verifica que há camada de highlight/catchlight para os olhos
        highlight_layers = [l for l in layers if l.get("purpose") == "highlight"]
        self.assertGreaterEqual(len(highlight_layers), 1, "Deve existir camada de highlight para os olhos brancos.")
        eye_layer = highlight_layers[0]
        self.assertGreater(eye_layer.get("stroke_count", 0), 2, "Os olhos devem ter preenchimento de traços reais.")

        # 2. Verifica que a camada do traje azul não contém nome de roxo/violeta
        for l in layers:
            rgb = l.get("color_rgb", (0, 0, 0))
            # Se for uma camada com B dominante sobre R
            if rgb[2] > rgb[0] + 30 and rgb[2] > 90:
                t_name = l.get("touch_info", {}).get("name", "")
                self.assertNotIn("Roxo", t_name, f"Traje azul foi nomeado como {t_name}")
                self.assertNotIn("Violeta", t_name, f"Traje azul foi nomeado como {t_name}")


if __name__ == "__main__":
    unittest.main()
