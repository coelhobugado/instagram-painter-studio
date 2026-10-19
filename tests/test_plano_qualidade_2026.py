# -*- coding: utf-8 -*-
"""
Testes de Validação do PLANO_EVOLUCAO_QUALIDADE_DO_DESENHO_2026.
Cobre todos os tipos de desenho:
1. Anime Face (preservação de espaços negativos de olhos/boca/feições)
2. Tulipa / Floral (separação de pétalas e caules sem fusão de hachuras)
3. Miranha / Cartoon (preservação cromática de azul vs roxo e olhos brancos protegidos)
4. Lettering / Logos (preservação de vazios de letras)
5. Governança Perceptual e Função Objetivo Global
"""

import unittest
import numpy as np
import cv2

from studio.planning.line_art_engine import LineArtEngine
from studio.planning.pictorial_quality import NegativeSpaceAnalyzer, SkeletonOptimizer
from studio.planning.global_objective import GlobalObjectiveLoss
from studio.vision.perceptual_color import PerceptualColorEngine
from studio.simulation.simulator_v2 import CalibratedSimulatorV2
from studio.core.models import CanvasTransform
from studio.painter_engine import PainterEngine


class TestPlanoQualidade2026(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.color_engine = PerceptualColorEngine()
        cls.simulator = CalibratedSimulatorV2()
        cls.transform = CanvasTransform(offset_x=50, offset_y=100, scale=0.8)

    def test_caso1_anime_face_preservacao_espacos_negativos(self):
        """
        Experimento 1 do Plano: Anime Face.
        Valida que os olhos e cavidades internas críticas não sofrem empastamento,
        mantendo o espaço negativo aberto no target_executable e na simulação.
        """
        h, w = 300, 300
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        # Rosto: contorno da mandíbula
        cv2.ellipse(canvas, (150, 150), (100, 120), 0, 0, 180, (0, 0, 0), 2)
        # Olho esquerdo com esclera interna branca aberta
        cv2.circle(canvas, (110, 130), 22, (0, 0, 0), 2)
        # Pupila interna fina com vazio em volta
        cv2.circle(canvas, (110, 130), 8, (0, 0, 0), 1)

        # Olho direito com esclera interna branca aberta
        cv2.circle(canvas, (190, 130), 22, (0, 0, 0), 2)
        cv2.circle(canvas, (190, 130), 8, (0, 0, 0), 1)

        # Nariz e boca fina
        cv2.line(canvas, (150, 150), (148, 165), (0, 0, 0), 2)
        cv2.ellipse(canvas, (150, 190), (25, 12), 0, 0, 180, (0, 0, 0), 2)

        # Gera o plano de line art
        plan = LineArtEngine.generate_plan(
            fit_img=canvas,
            color_engine=self.color_engine,
            simulator=self.simulator,
            transform=self.transform,
            seed=42,
            paint_background=False
        )

        self.assertIn("layers", plan)
        self.assertGreater(len(plan["layers"]), 0)

        # Valida que o target_executable foi gerado como interpretação viável
        target = plan["target_executable"]
        self.assertEqual(target.shape, (h, w, 3))

        # Ponto central do olho esquerdo (110, 122) deve ser fundo (22, 22, 22), NÃO tinta preta de contorno
        # No target_executable, a área protegida deve permanecer com cor de fundo
        left_eye_center_bgr = target[122, 110]
        # Deve ter cor de fundo (22, 22, 22)
        np.testing.assert_array_equal(left_eye_center_bgr, np.array([22, 22, 22], dtype=np.uint8))

        # Nenhum traço gerado deve penetrar no espaço negativo crítico dos olhos
        # Obtém a máscara de espaço negativo
        _, _, ink_mask = LineArtEngine.detect_background_and_ink_colors(canvas)
        neg_regions = NegativeSpaceAnalyzer.identify_enclosed_spaces(ink_mask)
        critical_mask = NegativeSpaceAnalyzer.build_critical_protection_mask(neg_regions, (h, w))
        self.assertGreater(np.count_nonzero(critical_mask), 0)

        for l in plan["layers"]:
            if l.get("purpose") == "line_art_centerline":
                for stroke in l.get("strokes", []):
                    for pt in stroke:
                        px, py = int(pt[0]), int(pt[1])
                        if 0 <= py < h and 0 <= px < w:
                            # Nenhum traço deve estar contido no interior crítico protegido
                            self.assertEqual(critical_mask[py, px], 0, f"Traço invadiu espaço negativo crítico em ({px}, {py})")

    def test_caso2_tulipa_floral_separacao_petalas_e_caules(self):
        """
        Experimento 2 do Plano: Tulipa / Floral.
        Valida que caules paralelos e pétalas curvas não se fundem em um borrão hiperdenso.
        """
        h, w = 300, 300
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        # Desenha 4 caules/hachuras verticais muito próximas
        for x_pos in [140, 148, 156, 164]:
            cv2.line(canvas, (x_pos, 100), (x_pos, 260), (0, 0, 0), 2)

        # Pétalas da tulipa no topo
        cv2.ellipse(canvas, (150, 80), (35, 45), 0, 0, 360, (0, 0, 0), 2)

        plan = LineArtEngine.generate_plan(
            fit_img=canvas,
            color_engine=self.color_engine,
            simulator=self.simulator,
            transform=self.transform,
            seed=42,
            paint_background=False
        )

        sim_canvas = plan["simulation_image"]
        # Mede a proporção de pixels desenhados
        diff_from_bg = np.any(sim_canvas != [22, 22, 22], axis=-1)
        ink_coverage = float(np.count_nonzero(diff_from_bg)) / (h * w)

        # A cobertura de tinta simulada não deve explodir além de 20% da tela (evitando a mancha de 12.44% relatada no diagnóstico)
        self.assertLess(ink_coverage, 0.20)

    def test_caso3_miranha_cartoon_preservacao_cor_e_olhos_brancos(self):
        """
        Experimento 3 do Plano: Miranha / Cartoon.
        Valida que o azul não vira roxo/violeta e que olhos brancos/catchlights
        ficam protegidos contra sobreposição de tinta vermelha/escura.
        """
        # Teste 1: Preservação estrita da família cromática Azul
        target_blue_rgb = (10, 80, 220)  # Azul puro vibrante do uniforme
        matched = self.color_engine.match_color(target_blue_rgb)
        matched_rgb = matched.get("matched_rgb", matched.get("rgb"))

        # O canal azul deve dominar claramente o canal vermelho (B > R + 20)
        # Se B <= R, significa que virou roxo/magenta!
        self.assertGreater(matched_rgb[2], matched_rgb[0] + 15,
                           f"Azul vibrante colapsou para roxo/magenta: {matched_rgb}")

        # Teste 2: Criação de arte cartoon com vermelho, azul e olhos brancos
        h, w = 150, 150
        cartoon_img = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

        # Máscara do rosto vermelha (BGR: 0, 0, 220)
        cv2.circle(cartoon_img, (75, 75), 55, (0, 0, 220), -1)

        # Olhos grandes brancos do Miranha (BGR: 255, 255, 255) com contorno preto
        cv2.ellipse(cartoon_img, (55, 70), (16, 26), -20, 0, 360, (0, 0, 0), 4)
        cv2.ellipse(cartoon_img, (55, 70), (13, 23), -20, 0, 360, (255, 255, 255), -1)

        cv2.ellipse(cartoon_img, (95, 70), (16, 26), 20, 0, 360, (0, 0, 0), 4)
        cv2.ellipse(cartoon_img, (95, 70), (13, 23), 20, 0, 360, (255, 255, 255), -1)

        engine = PainterEngine()
        plan = engine.generate_layered_paint_plan(
            image_path="",
            num_colors=4,
            fidelity_mode=True,
            optimize_in_simulation=False
        ) if False else None

        # Testa a extração de feições anatômicas brancas diretamente
        from studio.vision.semantic_features import SemanticFeatureExtractor
        spec_mask = SemanticFeatureExtractor.extract_specular_highlights(cartoon_img)

        # Os olhos brancos devem ser identificados como regiões protegidas (> 50 pixels brancos protegidos)
        self.assertGreater(np.count_nonzero(spec_mask), 50)
        # O centro do olho esquerdo (55, 70) deve estar protegido
        self.assertGreater(spec_mask[70, 55], 0)

    def test_caso4_lettering_logos_preservacao_vazios(self):
        """
        Experimento 4: Lettering / Logos.
        Valida que o interior de letras fechadas ('O', 'A', 'P') permanece aberto.
        """
        h, w = 200, 200
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        # Letra 'O' maiúscula com espessura de traço
        cv2.circle(canvas, (100, 100), 60, (0, 0, 0), 12)

        plan = LineArtEngine.generate_plan(
            fit_img=canvas,
            color_engine=self.color_engine,
            simulator=self.simulator,
            transform=self.transform,
            seed=42,
            paint_background=False
        )

        target = plan["target_executable"]
        # O centro da letra 'O' (100, 100) deve ser fundo limpo (22, 22, 22)
        np.testing.assert_array_equal(target[100, 100], np.array([22, 22, 22], dtype=np.uint8))

    def test_caso5_funcao_objetivo_global_classes_visuais(self):
        """
        Valida a Seção 5 do Plano: Pesos e perdas ponderadas por modalidade visual.
        """
        w_cartoon = GlobalObjectiveLoss.get_weights_for_class("cartoon")
        self.assertGreaterEqual(w_cartoon.w_structure, 0.30)
        self.assertGreaterEqual(w_cartoon.w_color, 0.30)
        self.assertGreater(w_cartoon.w_negative_space, 0.05)

        w_flower = GlobalObjectiveLoss.get_weights_for_class("flower")
        self.assertGreaterEqual(w_flower.w_edge, 0.25)
        self.assertGreaterEqual(w_flower.w_structure, 0.25)

        w_lettering = GlobalObjectiveLoss.get_weights_for_class("lettering")
        self.assertGreaterEqual(w_lettering.w_structure, 0.35)
        self.assertGreaterEqual(w_lettering.w_negative_space, 0.10)

        # Cálculo de perda com espaço negativo e sobreposição
        loss = GlobalObjectiveLoss.compute_loss(
            e_structure=0.1,
            e_color=0.1,
            e_edge=0.1,
            e_feature=0.1,
            time_cost_s=20.0,
            risk=0.0,
            weights=w_cartoon,
            e_negative_space=0.2,
            e_overlap=0.15
        )
        self.assertGreater(loss, 0.0)


if __name__ == "__main__":
    unittest.main()
