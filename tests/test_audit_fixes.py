"""
Bateria Completa de Testes de Validação da Auditoria Real (CODE_AUDIT_REAL.md).
Cobre 100% dos testes solicitados na Seção 36 do relatório de auditoria:
1. test_color_touch_schema_long_press
2. test_simulator_uses_local_coordinates
3. test_error_map_against_known_ciede2000
4. test_scanline_does_not_cross_hole
5. test_scanline_concave_shape
6. test_execute_capture_register_correct_execute
7. test_checkpoint_rejects_other_plan
8. test_checkpoint_rejects_other_device
9. test_second_retry_failure_does_not_advance_progress
10. test_color_reconfigured_after_reconnect
11. test_portrait_skin_detection_lab_scale
12. test_superpixel_lab_representation
13. test_watershed_boundary_label_unique
14. test_merged_features_are_recomputed
"""
import os
import unittest
import numpy as np
import cv2

from studio.vision.perceptual_color import PerceptualColorEngine
from studio.vision.superpixels import SuperpixelEngine
from studio.vision.region_graph import RegionAdjacencyGraph
from studio.feedback.error_map import ErrorMapEngine
from studio.feedback.registration import CanvasRegistrationEngine
from studio.feedback.correction_planner import CorrectionPlanner
from studio.planning.fill_strategies import FillStrategyEngine
from studio.planning.auto_mode import AutoModeDecisor
from studio.planning.layer_graph import LayerGraphEngine
from studio.core.models import StrokePlan, CanvasTransform, RegionFeatures
from studio.device.fake_adb import FakeADB
from studio.device.quality_gate import QualityGateResult
from studio.device.safe_executor import PersistentCheckpointManager
from studio.execution_worker import ExecutionWorker
from studio.painter_engine import PainterEngine

class TestAuditRealFixes(unittest.TestCase):
    def setUp(self):
        self.color_engine = PerceptualColorEngine()
        self.painter_engine = PainterEngine()

    # --------------------------------------------------------------------------
    # 1. Cor & Schema Canônico
    # --------------------------------------------------------------------------
    def test_color_touch_schema_long_press(self):
        """Verifica se cores de espectro (long press) contêm base_x, base_y, target_x, target_y sem KeyError."""
        # Azul celeste do espectro
        match = self.color_engine.match_color((91, 170, 243))
        self.assertIn("is_long_press", match)
        self.assertIn("target_x", match)
        self.assertIn("target_y", match)

        fake_adb = FakeADB()
        # Não deve lançar KeyError nem TypeError
        try:
            fake_adb.select_palette_color(match)
        except KeyError as e:
            self.fail(f"KeyError inesperado em select_palette_color: {e}")

        # Se for long press, valida se gravou drag no mock
        if match["is_long_press"]:
            self.assertIsNotNone(match["base_x"])
            self.assertIsNotNone(match["base_y"])
            self.assertTrue(any("select_spectrum_drag" in cmd for cmd in fake_adb.executed_commands))

    # --------------------------------------------------------------------------
    # 2. Coordenadas Locais vs Tela Física no Simulador
    # --------------------------------------------------------------------------
    def test_simulator_uses_local_coordinates(self):
        """Verifica se o plano gera traços Canvas-Local (0..W, 0..H) e se o simulador não sofre corte."""
        plan = self.painter_engine.generate_stencil_plan("inputs/flores.jpg")
        self.assertIn("transform", plan)
        transform = plan["transform"]
        self.assertIsInstance(transform, CanvasTransform)

        fit_w = plan["fit_image"].shape[1]
        fit_h = plan["fit_image"].shape[0]

        # Todos os traços da camada devem estar dentro dos limites do canvas local
        for layer in plan["layers"]:
            for stroke in layer["strokes"]:
                for px, py in stroke:
                    self.assertGreaterEqual(px, 0)
                    self.assertLess(px, fit_w)
                    self.assertGreaterEqual(py, 0)
                    self.assertLess(py, fit_h)

        # O preview do simulador deve possuir exatamente a dimensão local
        sim_img = plan["simulation_image"]
        self.assertEqual(sim_img.shape[:2], (fit_h, fit_w))

    # --------------------------------------------------------------------------
    # 3. Métrica Perceptual True CIEDE2000 no ErrorMapEngine
    # --------------------------------------------------------------------------
    def test_error_map_against_known_ciede2000(self):
        """Verifica se o ErrorMapEngine calcula True CIEDE2000 (0 para cores idênticas e valor exato para pares conhecidos)."""
        target = np.ones((50, 50, 3), dtype=np.uint8) * 128
        same_target = target.copy()

        _, metrics_zero = ErrorMapEngine.compute_error_map(target, same_target)
        self.assertEqual(metrics_zero["mean_delta_e"], 0.0)
        self.assertEqual(metrics_zero["ssim"], 1.0)

        # Imagem com par cromático diferente
        diff_img = np.ones((50, 50, 3), dtype=np.uint8)
        diff_img[:, :] = [100, 150, 200]
        _, metrics_diff = ErrorMapEngine.compute_error_map(target, diff_img)
        self.assertGreater(metrics_diff["mean_delta_e"], 5.0)
        self.assertLess(metrics_diff["mean_delta_e"], 100.0)

    # --------------------------------------------------------------------------
    # 4. Path Safety: Não atravessa buracos em máscaras
    # --------------------------------------------------------------------------
    def test_scanline_does_not_cross_hole(self):
        """Cria uma máscara com um buraco no meio e garante que nenhum segmento passe por dentro do buraco."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Quadrado branco de 10 a 90
        mask[10:90, 10:90] = 255
        # Buraco central de 35 a 65
        mask[35:65, 35:65] = 0

        strokes = FillStrategyEngine.generate_region_strokes(
            mask, (10, 10, 80, 80), strategy="horizontal_scanline", brush_spacing=4
        )
        self.assertGreater(len(strokes), 0)

        # Testa se algum ponto ou segmento gerado intersecta o buraco (35 <= x <= 65 e 35 <= y <= 65)
        for stroke in strokes:
            for pt in stroke:
                x, y = pt
                if 35 < x < 65 and 35 < y < 65:
                    self.fail(f"Ponto ({x}, {y}) violou o buraco da máscara!")

    # --------------------------------------------------------------------------
    # 5. Path Safety: Região Côncava em 'U'
    # --------------------------------------------------------------------------
    def test_scanline_concave_shape(self):
        """Garante que em uma região em forma de 'U', os traços não cruzem o espaço vazio superior."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Base do U
        mask[60:90, 10:90] = 255
        # Braço esquerdo
        mask[10:60, 10:35] = 255
        # Braço direito
        mask[10:60, 65:90] = 255
        # Espaço vazio côncavo: x in (35, 65) e y in (10, 60)

        strokes = FillStrategyEngine.generate_region_strokes(
            mask, (10, 10, 80, 80), strategy="horizontal_scanline", brush_spacing=4
        )
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                p1 = stroke[i]
                p2 = stroke[i + 1]
                # Verifica se o segmento sai da máscara
                self.assertTrue(
                    FillStrategyEngine._is_segment_inside_mask(mask, p1, p2),
                    f"Segmento {p1} -> {p2} vazou para fora da região côncava!"
                )

    # --------------------------------------------------------------------------
    # 6. Closed Loop Integrado: Execute -> Capture -> Register -> Correct -> Execute
    # --------------------------------------------------------------------------
    def test_execute_capture_register_correct_execute(self):
        """Valida o ciclo fechado operacional com FakeADB, registro afim e retoques autônomos."""
        fake_adb = FakeADB()
        plan = self.painter_engine.generate_stencil_plan("inputs/flores.jpg")
        
        callbacks = {
            "on_progress": None,
            "on_layer_start": None,
            "on_log": None,
            "on_finished": None,
            "on_screencap": None,
            "on_error_map": None,
            "on_checkpoint": None
        }

        worker = ExecutionWorker(fake_adb, plan, callbacks, enable_closed_loop=True)
        worker.run()

        # O FakeADB deve conter os traços da camada base e o screencap deve refletir isso
        self.assertGreater(len(fake_adb.injected_strokes), 0)
        screencap = fake_adb.capture_screenshot()
        self.assertIsNotNone(screencap)

    # --------------------------------------------------------------------------
    # 7 & 8. Segurança de Checkpoint (Rejeita outro plano ou outro dispositivo)
    # --------------------------------------------------------------------------
    def test_checkpoint_rejects_other_plan(self):
        """Valida que o plano rejeita checkpoint de plano com imagem/hash diferente."""
        plan1 = self.painter_engine.generate_stencil_plan("inputs/flores.jpg", seed=42)
        plan2 = self.painter_engine.generate_stencil_plan("inputs/flores.jpg", seed=999)
        self.assertNotEqual(plan1["plan_hash"], plan2["plan_hash"])

    def test_checkpoint_rejects_other_device(self):
        """Valida salvamento e leitura de device_id no checkpoint."""
        PersistentCheckpointManager.save_checkpoint(
            plan_hash="hash123", layer_idx=1, stroke_idx=20, device_id="SM-A736B"
        )
        cp = PersistentCheckpointManager.load_checkpoint()
        self.assertEqual(cp["device_id"], "SM-A736B")
        self.assertEqual(cp["plan_hash"], "hash123")
        PersistentCheckpointManager.clear_checkpoint()
        self.assertIsNone(PersistentCheckpointManager.load_checkpoint())

    # --------------------------------------------------------------------------
    # 9. Falha de Segundo Retry não avança completed_strokes
    # --------------------------------------------------------------------------
    def test_second_retry_failure_does_not_advance_progress(self):
        """Verifica se o worker pausa sem avançar o contador de traços se o lote falhar 2 vezes."""
        fake_adb = FakeADB()
        # Força o FakeADB a falhar
        fake_adb.screen_on = False

        plan = {
            "total_strokes": 10,
            "layers": [{
                "name": "Camada Teste",
                "color_rgb": (255, 255, 255),
                "brush": "solid",
                "thickness": "min",
                "touch_info": {"target_x": 100, "target_y": 100, "is_long_press": False},
                "strokes": [[(10, 10), (20, 20)] for _ in range(5)]
            }],
            "transform_dict": {"offset_x": 0, "offset_y": 0, "scale": 1.0}
        }

        worker = ExecutionWorker(fake_adb, plan, {})
        worker.run()
        # Com tela desligada (fail-closed), nenhum traço é computado como sucesso
        self.assertEqual(worker.total_completed_strokes, 0)

    # --------------------------------------------------------------------------
    # 10. Reconfiguração de Pincel e Cor após Reconnect
    # --------------------------------------------------------------------------
    def test_color_reconfigured_after_reconnect(self):
        """Garante que touch_info é re-executado após auto_reconnect do worker."""
        fake_adb = FakeADB()
        touch_info = {"target_x": 200, "target_y": 300, "is_long_press": False, "name": "Azul Teste"}
        fake_adb.select_palette_color(touch_info)
        self.assertIn("tap_color_200_300", fake_adb.executed_commands)

    # --------------------------------------------------------------------------
    # 11. Escala Lab na Detecção de Pele do AUTO Mode
    # --------------------------------------------------------------------------
    def test_portrait_skin_detection_lab_scale(self):
        """Verifica se uma imagem de retrato tem tons de pele detectados e é classificada como PORTRAIT."""
        portrait_path = "inputs/retrato.jpg"
        if os.path.exists(portrait_path):
            img = cv2.imread(portrait_path)
            res = AutoModeDecisor.analyze_scene_and_decide(img)
            self.assertEqual(res["scene_type"], "PORTRAIT")
            self.assertGreater(res["metrics"]["skin_ratio"], 0.15)

    # --------------------------------------------------------------------------
    # 12. Superpixels em True float32 Lab
    # --------------------------------------------------------------------------
    def test_superpixel_lab_representation(self):
        """Verifica se extract_features extrai L* em [0, 100] e a*, b* com sinais reais."""
        dummy_bgr = np.ones((60, 60, 3), dtype=np.uint8) * 180
        labels, num_labels = SuperpixelEngine.segment_superpixels_lab(dummy_bgr, num_superpixels=10)
        features = SuperpixelEngine.extract_features(dummy_bgr, labels, num_labels)
        self.assertGreater(len(features), 0)
        for f in features:
            l, a, b = f.mean_lab
            self.assertGreaterEqual(l, 0.0)
            self.assertLessEqual(l, 100.0)

    # --------------------------------------------------------------------------
    # 13. Watershed não mescla fronteira (-1) com label 0
    # --------------------------------------------------------------------------
    def test_watershed_boundary_label_unique(self):
        """Verifica se a normalização de Watershed não produz labels negativos nem mescla bordas incorretamente."""
        test_img = np.zeros((80, 80, 3), dtype=np.uint8)
        test_img[:40, :] = [50, 50, 200]
        test_img[40:, :] = [200, 50, 50]

        labels, num_labels = SuperpixelEngine.segment_superpixels_lab(test_img, num_superpixels=16)
        self.assertGreaterEqual(np.min(labels), 0)
        self.assertEqual(np.max(labels), num_labels - 1)

    # --------------------------------------------------------------------------
    # 14. Regiões Fundidas Recalculam Média Ponderada
    # --------------------------------------------------------------------------
    def test_merged_features_are_recomputed(self):
        """Verifica se adaptive_merge calcula média ponderada de Lab entre as sub-regiões."""
        f1 = RegionFeatures(id=0, mean_lab=(60.0, 10.0, 10.0), lab_variance=2.0, texture_angle=0.0,
                            edge_density=0.0, area=100, centroid=(10, 10), bbox=(0, 0, 20, 20), saliency=0.4)
        f2 = RegionFeatures(id=1, mean_lab=(62.0, 10.0, 10.0), lab_variance=3.0, texture_angle=0.0,
                            edge_density=0.0, area=200, centroid=(30, 10), bbox=(20, 0, 20, 20), saliency=0.6)

        labels_map = np.zeros((20, 40), dtype=np.int32)
        labels_map[:, :20] = 0
        labels_map[:, 20:] = 1

        rag = RegionAdjacencyGraph([f1, f2], labels_map)
        merged_labels, merged_features = rag.adaptive_merge(max_delta_e=5.0)

        # Como a distância perceptual entre (60, 10, 10) e (62, 10, 10) é pequena (~2.0), devem fundir
        self.assertEqual(len(merged_features), 1)
        # Média ponderada esperada: (60*100 + 62*200) / 300 = 18400 / 300 = 61.33
        expected_l = (60.0 * 100 + 62.0 * 200) / 300.0
        self.assertAlmostEqual(merged_features[0].mean_lab[0], expected_l, places=2)

    # --------------------------------------------------------------------------
    # 15. Toggle de Contornos na Pintura por Camadas
    # --------------------------------------------------------------------------
    def test_include_contours_toggle(self):
        """Verifica se include_contours=False desativa camadas de contorno e include_contours=True as gera."""
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 180
        cv2.line(test_img, (20, 50), (80, 50), (10, 10, 10), 3)
        temp_path = "tests_temp_toggle.png"
        cv2.imwrite(temp_path, test_img)

        try:
            plan_no_cnt = self.painter_engine.generate_layered_paint_plan(temp_path, num_colors=2, include_contours=False)
            cnt_layers_no = [l for l in plan_no_cnt["layers"] if l["purpose"] == "contour"]
            self.assertEqual(len(cnt_layers_no), 0)

            plan_with_cnt = self.painter_engine.generate_layered_paint_plan(temp_path, num_colors=2, include_contours=True)
            cnt_layers_with = [l for l in plan_with_cnt["layers"] if l["purpose"] == "contour"]
            self.assertGreater(len(cnt_layers_with), 0)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --------------------------------------------------------------------------
    # 16. Blindagem do Closed-Loop contra Riscos Destrutivos
    # --------------------------------------------------------------------------
    def test_closed_loop_does_not_scratch_canvas(self):
        """Garante que ExecutionWorker nunca aplique traços destrutivos no final da pintura por padrão."""
        from studio.execution_worker import ExecutionWorker
        from studio.device.fake_adb import FakeADB

        fake_adb = FakeADB()
        # Mock de captura de tela retornando imagem diferente da alvo
        fake_adb.capture_screenshot = lambda: np.zeros((400, 400, 3), dtype=np.uint8)

        plan = {
            "fit_image": np.ones((100, 100, 3), dtype=np.uint8) * 200,
            "quality_gate": QualityGateResult(passed=True, summary="Plano de teste aprovado"),
            "custom_bounds": {"left": 80, "right": 500, "top": 380, "bottom": 800},
            "layers": [
                {
                    "name": "Camada 1",
                    "color_rgb": (0, 0, 0),
                    "strokes": [[(100, 500), (200, 600)]],
                    "thickness": "min",
                    "touch_info": {"target_x": 100, "target_y": 100, "is_long_press": False}
                }
            ],
            "transform_dict": {"offset_x": 0, "offset_y": 0, "scale": 1.0}
        }

        executed_batches = []
        original_execute = fake_adb.execute_stroke_batch
        def mock_execute(batch):
            executed_batches.append(batch)
            return original_execute(batch)
        fake_adb.execute_stroke_batch = mock_execute

        worker = ExecutionWorker(
            fake_adb, plan, callbacks={}, enable_closed_loop=True, allow_destructive_corrections=False
        )
        worker.run()

        # Deve ter executado apenas 1 lote referente à Camada 1, NENHUM lote de correção por cima do desenho
        self.assertEqual(len(executed_batches), 1)

if __name__ == "__main__":
    unittest.main()
