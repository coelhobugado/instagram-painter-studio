# -*- coding: utf-8 -*-
"""
Testes de Regressão e Validação das Correções:
1. Modo Automático do MCP (AutoModeDecisor.decide_mode).
2. FakeADB sem UI sintética sobre a arte (init_ui=False) e espessuras/cores calibradas.
3. BenchmarkSuite: configuração de cor/espessura por camada, transparência de hardware real vs mock.
"""
import unittest
import os
import cv2
import numpy as np

from studio.planning.auto_mode import AutoModeDecisor, AutoModeDecision
from studio.device.fake_adb import FakeADB
from studio.benchmarks.benchmark_suite import BenchmarkSuite
from studio.mcp.handlers_planning import handle_plan_artwork

class TestValidationFixes(unittest.TestCase):
    def setUp(self):
        self.sample_img = "inputs/flores.jpg"

    def test_auto_mode_decisor_decide_mode(self):
        """Valida que decide_mode aceita caminho de arquivo e numpy array, retornando AutoModeDecision."""
        if os.path.exists(self.sample_img):
            # Teste via path
            decision_from_path = AutoModeDecisor.decide_mode(self.sample_img)
            self.assertIsInstance(decision_from_path, AutoModeDecision)
            self.assertEqual(decision_from_path.mode, "line_art")
            self.assertEqual(decision_from_path["mode"], "line_art")
            self.assertEqual(decision_from_path.get("recommended_mode"), "line_art")

            # Teste via array BGR
            img = cv2.imread(self.sample_img)
            decision_from_arr = AutoModeDecisor.decide_mode(img)
            self.assertEqual(decision_from_arr.mode, "line_art")
            self.assertIn("mode", decision_from_arr)
            self.assertIn("recommended_k_colors", decision_from_arr)

    def test_mcp_handle_plan_artwork_auto_mode(self):
        """Valida que o modo automático do MCP não falha e retorna o plano com auto_decision."""
        if os.path.exists(self.sample_img):
            res = handle_plan_artwork({
                "image_path": self.sample_img,
                "style": "auto"
            })
            self.assertTrue(res["success"])
            self.assertEqual(res["mode"], "line_art")
            self.assertIn("auto_decision", res)
            self.assertEqual(res["auto_decision"]["scene_type"], "LOGO_OR_LINEART")

    def test_fake_adb_clean_canvas_and_thickness(self):
        """Valida FakeADB com init_ui=False, espessura calibrada (thick=28) e set_color_rgb."""
        adb = FakeADB(width=400, height=400, init_ui=False, background_bgr=(255, 255, 255))
        # Verifica que o canvas está limpo, sem toolbar ou botões sintéticos
        self.assertTrue(np.all(adb.virtual_screen == 255))

        # Testa espessura calibrada
        adb.set_brush_thickness("thick")
        self.assertEqual(adb.current_width_px, 98)

        adb.set_brush_thickness("min")
        self.assertEqual(adb.current_width_px, 10)

        # Testa set_color_rgb
        adb.set_color_rgb((10, 20, 30))
        self.assertEqual(adb.current_color_bgr, (30, 20, 10))

    def test_benchmark_layer_configuration_and_transparency(self):
        """Valida que o benchmark configura cores e espessuras por camada e reporta backend com precisão."""
        suite = BenchmarkSuite()
        rep = suite.run_canonical_benchmark(max_cases=1)
        self.assertIn("execution_backend", rep)
        self.assertIn("is_real_device", rep)
        self.assertEqual(rep["is_real_device"], False)
        self.assertEqual(rep["validation_status"], "MOCK_SEM_HARDWARE_FISICO")

        # Verifica caso individual
        case = rep["results"][0]
        self.assertEqual(case["device_backend"], "fake_adb_mock")
        self.assertIn("device_ssim", case)
        self.assertGreater(case["device_ssim"], 0.9)

if __name__ == "__main__":
    unittest.main()
