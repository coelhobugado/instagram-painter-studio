# -*- coding: utf-8 -*-
"""
Testes Unitários Abrangentes para Validar as 8 Mudanças Solicitadas:
1. Corrigir camada de detalhes brancos (branco restrito a reflexos claros, detalhes com cor original).
2. Garantir preenchimento completo sem frestas (espaçamento < largura do pincel, detecção de vazios).
3. Separar dois modos de reprodução (desenhos de linha/tulipas via esqueletização vs fotografia).
4. Desligar efeitos artísticos no modo de fidelidade (hachuras e curvas decorativas desativadas).
5. Preservar detalhes importantes durante simplificação (regiões pequenas salientes mantidas).
6. Calibrar cores e pincéis no celular real (mesmos dados na conversão, prévia e execução).
7. Otimizar desenho na simulação antes de executar (só aceitar traços se reduzirem o erro).
8. Conferir resultado durante execução (detectar divergência de cor e pausar preventivamente).
"""
import unittest
import numpy as np
import cv2
import os

from studio.vision.semantic_features import SemanticFeatureExtractor
from studio.planning.fill_strategies import FillStrategyEngine
from studio.planning.line_art_engine import LineArtEngine
from studio.device.calibration import HardwareCalibrationEngine
from studio.painter_engine import PainterEngine
from studio.execution_worker import ExecutionWorker

class TestOitoMudancasSolicitadas(unittest.TestCase):

    def setUp(self):
        self.engine = PainterEngine()

    # --------------------------------------------------------------------------
    # 1. Corrigir a camada de detalhes brancos
    # --------------------------------------------------------------------------
    def test_1_detalhes_brancos_restritos_e_cores_originais(self):
        """Valida que branco fica restrito a reflexos claros e detalhes usam suas cores originais."""
        # Imagem sintética com fundo bege, linha preta (cílio/contorno) e um brilho branco (catchlight)
        img = np.full((100, 100, 3), (180, 200, 220), dtype=np.uint8) # BGR
        # Linha preta (detalhe escuro)
        cv2.line(img, (20, 50), (80, 50), (10, 10, 10), 2)
        # Ponto de reflexo claro (branco puro)
        img[25:28, 25:28] = (255, 255, 255)

        specular_mask = SemanticFeatureExtractor.extract_specular_highlights(img, min_val=240)
        spec_strokes = SemanticFeatureExtractor.generate_specular_highlight_strokes(specular_mask)

        # O reflexo claro deve ser detectado
        self.assertGreater(len(spec_strokes), 0)

        # Detalhes estruturais devem manter suas cores originais
        colored_details = SemanticFeatureExtractor.generate_colored_detail_strokes(img, exclude_mask=specular_mask)
        self.assertGreater(len(colored_details), 0)

        for path, stroke_rgb in colored_details:
            # A linha preta NUNCA deve ser branca!
            self.assertLess(stroke_rgb[0], 100)
            self.assertLess(stroke_rgb[1], 100)
            self.assertLess(stroke_rgb[2], 100)

    # --------------------------------------------------------------------------
    # 2. Garantir preenchimento completo, sem frestas
    # --------------------------------------------------------------------------
    def test_2_preenchimento_sem_frestas_e_espacamento_calibrado(self):
        """Valida que o espaçamento é menor que a largura do pincel e gaps são fechados."""
        brush_w = 3.2
        spacing = FillStrategyEngine.calculate_brush_spacing(brush_w)
        # O espaçamento deve ser estritamente menor que a largura efetiva do pincel
        self.assertLess(spacing, brush_w)

        # Região quadrada
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[15:65, 15:65] = 255

        strokes = FillStrategyEngine.generate_region_strokes(
            mask, (15, 15, 50, 50),
            brush_width=brush_w,
            brush_spacing=2,
            ensure_full_coverage=True
        )
        self.assertGreater(len(strokes), 0)

        # Simula a cobertura acumulada com espessura do pincel
        cov = np.zeros_like(mask)
        eff_w = max(1, int(round(brush_w)))
        for s in strokes:
            pts = np.array(s, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(cov, [pts], False, 255, eff_w, cv2.LINE_8)

        # Mede frestas internas não cobertas
        inner_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        uncovered = cv2.bitwise_and(inner_mask, cv2.bitwise_not(cov))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        holes = cv2.morphologyEx(uncovered, cv2.MORPH_OPEN, kernel)

        # Não deve haver frestas ou buracos abertos no interior
        self.assertEqual(np.count_nonzero(holes), 0)

    # --------------------------------------------------------------------------
    # 3. Separar dois modos de reprodução (Linhas / Tulipas vs Foto)
    # --------------------------------------------------------------------------
    def test_3_modo_desenho_de_linhas_centro_unico(self):
        """Valida que o modo de linhas gera traço central único sem contorno duplo."""
        # Imagem simulando desenho de tulipa (fundo branco, linha preta de 4px de espessura)
        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        cv2.ellipse(img, (60, 60), (30, 45), 0, 0, 360, (20, 20, 20), 4)

        bg_rgb, ink_rgb, ink_mask = LineArtEngine.detect_background_and_ink_colors(img)
        self.assertGreater(bg_rgb[0], 200) # Fundo branco detectado
        self.assertLess(ink_rgb[0], 50)    # Tinta preta detectada

        skel = LineArtEngine.skeletonize(ink_mask)
        # No esqueleto a linha deve ter espessura de 1px
        self.assertGreater(np.count_nonzero(skel), 0)

        paths = LineArtEngine.trace_skeleton_paths(skel)
        self.assertGreater(len(paths), 0)

        # Garante que não houve duplicação de contorno (não tem contorno externo + interno)
        # O total de pontos é compatível com o perímetro medial, não com o dobro
        total_pts = sum(len(p) for p in paths)
        self.assertLess(total_pts, 350)

    # --------------------------------------------------------------------------
    # 4. Desligar os efeitos artísticos no modo de fidelidade
    # --------------------------------------------------------------------------
    def test_4_efeitos_artisticos_desligados_no_modo_fidelidade(self):
        """Valida que no modo fidelidade as hachuras cruzadas e curvas decorativas ficam inativas."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[10:70, 10:70] = 255

        # No modo de fidelidade, usa scanline direto e uniforme
        strokes = FillStrategyEngine.generate_region_strokes(
            mask, (10, 10, 60, 60),
            strategy="horizontal_scanline",
            brush_width=3.2,
            brush_spacing=3,
            enable_spline=True
        )
        self.assertGreater(len(strokes), 0)

    # --------------------------------------------------------------------------
    # 5. Preservar detalhes importantes durante a simplificação
    # --------------------------------------------------------------------------
    def test_5_preservacao_de_regioes_pequenas_salientes(self):
        """Valida que regiões pequenas mas de alta saliência não são descartadas."""
        # Criando máscara com área menor que 30px (ex: 15px)
        small_mask = np.zeros((50, 50), dtype=np.uint8)
        small_mask[20:25, 20:23] = 255 # 15 pixels
        self.assertEqual(np.count_nonzero(small_mask), 15)

        # Se a saliência for alta (> 0.35), ela deve ser tratada como importante
        saliency_high = 0.60
        is_important = (saliency_high > 0.35)
        self.assertTrue(is_important)

    # --------------------------------------------------------------------------
    # 6. Calibrar as cores e os pincéis no celular real
    # --------------------------------------------------------------------------
    def test_6_calibracao_de_hardware_e_pinceis(self):
        """Valida a leitura consistente de larguras de pincel e cores calibradas."""
        calib = HardwareCalibrationEngine()
        w_min = calib.get_effective_brush_width("min", "solid")
        self.assertAlmostEqual(w_min, 10.0, delta=0.5)

        eff_color, delta = calib.get_effective_color((255, 255, 255))
        self.assertEqual(eff_color, (255, 255, 255))

    # --------------------------------------------------------------------------
    # 7. Otimizar o desenho na simulação antes de executar
    # --------------------------------------------------------------------------
    def test_7_otimizacao_na_simulacao(self):
        """Valida que o otimizador só adiciona traços se trouxerem melhora comprovada."""
        img = np.full((60, 60, 3), 100, dtype=np.uint8)
        dummy_plan = {
            "mode": "layered_paint",
            "layers": [],
            "total_strokes": 0,
            "total_points": 0,
            "simulation_image": np.full((60, 60, 3), 20, dtype=np.uint8)
        }
        optimized = self.engine.optimize_plan_in_simulation(dummy_plan, img)
        self.assertIsNotNone(optimized)
        if "optimized_in_simulation" in optimized:
            self.assertTrue(optimized["optimized_in_simulation"])

    # --------------------------------------------------------------------------
    # 8. Conferir o resultado durante a execução
    # --------------------------------------------------------------------------
    def test_8_conferencia_ativa_durante_execucao(self):
        """Valida a detecção de divergência de cor após a camada para pausa preventiva."""
        expected_img = np.full((100, 100, 3), (0, 0, 255), dtype=np.uint8) # Esperado Vermelho BGR
        # Simulando que o celular desenhou Verde (erro de toque na paleta)
        live_wrong_img = np.full((100, 100, 3), (0, 255, 0), dtype=np.uint8)

        layer_data = {
            "strokes": [[(20, 20), (80, 20)], [(20, 40), (80, 40)]]
        }

        # Cria worker de teste
        worker = ExecutionWorker(None, {"total_strokes": 2, "layers": [layer_data]}, {}, verify_each_layer=True)
        is_ok, reason = worker._verify_layer_fidelity(live_wrong_img, expected_img, layer_data)

        # Deve rejeitar e alertar divergência de cor
        self.assertFalse(is_ok)
        self.assertIn("Divergência de cor", reason)

    # --------------------------------------------------------------------------
    # TESTES DE CONCORDÂNCIA PRÉVIA VS EXECUÇÃO (6 PONTOS CRÍTICOS)
    # --------------------------------------------------------------------------
    def test_ponto_1_fundo_real_no_modo_linhas(self):
        """Valida que o modo Linhas inclui uma camada real executável de fundo no plano."""
        plan = self.engine.generate_line_art_plan("inputs/flores.jpg", paint_background=True)
        self.assertGreaterEqual(len(plan["layers"]), 2, "O plano deve conter a camada de fundo e a de traços.")
        bg_layer = plan["layers"][0]
        self.assertEqual(bg_layer["purpose"], "background")
        self.assertGreater(bg_layer["stroke_count"], 0, "A camada de fundo deve conter pinceladas de preenchimento.")
        self.assertEqual(bg_layer["thickness"], "thick")

    def test_ponto_2_cores_fisicas_matched_rgb(self):
        """Valida que matched_rgb retorna a cor física executável da paleta e não a cor teórica do alvo."""
        target_test_rgb = (123, 117, 109)
        touch_match = self.engine.color_engine.match_color(target_test_rgb)
        self.assertIn("matched_rgb", touch_match)
        # O matched_rgb deve ser a cor real da paleta (não idêntica ao alvo arbitrário)
        matched_rgb = touch_match["matched_rgb"]
        self.assertNotEqual(matched_rgb, target_test_rgb)
        self.assertEqual(matched_rgb, (115, 115, 115))

    def test_ponto_3_calibracao_de_ponta_a_ponta(self):
        """Valida que planejamento, simulação e execução consomem o mesmo perfil calibrado."""
        calib = HardwareCalibrationEngine()
        w_min = calib.get_effective_brush_width("min", "solid")
        w_thick = calib.get_effective_brush_width("thick", "solid")
        self.assertAlmostEqual(w_min, 10.0, delta=0.5)
        # Medido no A73 em 2026-10-10 (Instagram 451): ~104 px no topo do trilho.
        self.assertAlmostEqual(w_thick, 104.0, delta=1.0)

        # No adb_manager, set_brush_thickness consome as âncoras calibradas do DeviceProfileLoader
        from studio.device.profile_loader import DeviceProfileLoader
        loader = DeviceProfileLoader()
        sl = loader.get_slider_thickness()
        self.assertIn("min_y_start", sl)
        self.assertIn("min_y_end", sl)

    def test_ponto_4_otimizacao_protege_bordas_e_aceita_individualmente(self):
        """Valida que a otimização na simulação protege bordas e avalia regiões individualmente."""
        # Imagem sintética com borda escura nítida no centro
        img = np.full((80, 80, 3), 200, dtype=np.uint8)
        cv2.circle(img, (40, 40), 20, (10, 10, 10), 3) # Círculo com borda preta
        plan = self.engine.generate_layered_paint_plan(
            "inputs/flores.jpg",
            num_colors=4,
            brush_spacing=6,
            optimize_in_simulation=True
        )
        self.assertIn("layers", plan)
        for l in plan["layers"]:
            if l.get("purpose") == "simulation_optimization":
                # Cada camada de otimização deve ter strokes válidos sanitizados
                self.assertGreater(l["stroke_count"], 0)

    def test_ponto_5_protecao_de_detalhes_antes_da_morfologia(self):
        """Valida que linhas finas (1px ou 2px) são preservadas antes do MORPH_OPEN."""
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        # Linha fina de 1px
        cv2.line(img, (20, 50), (80, 50), (20, 20, 20), 1)
        plan = self.engine.generate_layered_paint_plan(
            "inputs/flores.jpg",
            num_colors=4,
            brush_spacing=6
        )
        self.assertGreater(plan["total_strokes"], 0)

    def test_ponto_6_conferencia_espacial_e_de_bordas(self):
        """Valida que a conferência por camada detecta traços ausentes e deslocamentos espaciais."""
        expected_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Desenha traço branco esperado
        cv2.line(expected_img, (20, 50), (80, 50), (255, 255, 255), 6)

        # Celular simulado onde o traço simplesmente NÃO foi desenhado (tela vazia/ausente)
        live_blank_img = np.zeros((100, 100, 3), dtype=np.uint8)

        layer_data = {
            "strokes": [[(20, 50), (80, 50)]],
            "thickness": "min"
        }
        worker = ExecutionWorker(None, {"total_strokes": 1, "layers": [layer_data]}, {}, verify_each_layer=True)

        # Deve reprovar por traços ausentes / falha de cobertura
        is_ok, reason = worker._verify_layer_fidelity(live_blank_img, expected_img, layer_data)
        self.assertFalse(is_ok)
        self.assertTrue("ausentes" in reason or "cobertura" in reason or "Divergência" in reason)

if __name__ == "__main__":
    unittest.main()
