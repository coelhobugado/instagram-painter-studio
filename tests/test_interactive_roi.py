"""
Testes Unitários para Seleção de Área Interativa (ROI) e Bounds Customizados no Canvas.
Garante que os motores de planejamento e execução respeitem rigorosamente o retângulo
desenhado interativamente pelo usuário no print do celular.
"""
import unittest
import numpy as np
from studio.painter_engine import PainterEngine
from studio.device.fake_adb import FakeADB
from studio.execution_worker import ExecutionWorker
from studio.core.models import CanvasTransform

class TestInteractiveROI(unittest.TestCase):
    def setUp(self):
        self.engine = PainterEngine()
        self.fake_adb = FakeADB()

    def test_preprocess_fit_with_custom_bounds(self):
        """Verifica se o enquadramento aspect-ratio é contido rigorosamente dentro de custom_bounds."""
        dummy_img = np.zeros((400, 300, 3), dtype=np.uint8)
        custom_bounds = {
            "left": 200,
            "top": 500,
            "right": 800,
            "bottom": 1300,
            "center_x": 500,
            "center_y": 900,
            "max_width": 600,
            "max_height": 800
        }
        fitted, scale, (off_x, off_y), (new_w, new_h) = self.engine.preprocess_fit(
            dummy_img, custom_bounds=custom_bounds
        )
        # Deve caber dentro de 600x800
        self.assertLessEqual(new_w, 600)
        self.assertLessEqual(new_h, 800)
        # Deve estar centralizado em (500, 900)
        self.assertEqual(off_x, int(round(500 - new_w / 2.0)))
        self.assertEqual(off_y, int(round(900 - new_h / 2.0)))
        # Os limites físicos não podem ultrapassar custom_bounds
        self.assertGreaterEqual(off_x, custom_bounds["left"])
        self.assertLessEqual(off_x + new_w, custom_bounds["right"])
        self.assertGreaterEqual(off_y, custom_bounds["top"])
        self.assertLessEqual(off_y + new_h, custom_bounds["bottom"])

    def test_stencil_plan_with_custom_bounds(self):
        """Verifica se o plano de stencil propaga e respeita os custom_bounds definidos."""
        custom_bounds = {
            "left": 150, "top": 400, "right": 750, "bottom": 1200,
            "center_x": 450, "center_y": 800, "max_width": 600, "max_height": 800
        }
        plan = self.engine.generate_stencil_plan("inputs/flores.jpg", custom_bounds=custom_bounds)
        self.assertIn("custom_bounds", plan)
        self.assertEqual(plan["custom_bounds"], custom_bounds)
        td = plan["transform_dict"]
        self.assertGreaterEqual(td["offset_x"], custom_bounds["left"])
        self.assertGreaterEqual(td["offset_y"], custom_bounds["top"])

    def test_execution_worker_sanitizes_within_custom_bounds(self):
        """Garante que o ExecutionWorker use os custom_bounds para isolamento total de traços."""
        custom_bounds = {
            "left": 300, "top": 600, "right": 700, "bottom": 1000,
            "center_x": 500, "center_y": 800, "max_width": 400, "max_height": 400
        }
        # Traços com pontos propositalmente fora dos limites da área customizada
        out_of_bounds_strokes = [
            [(100, 200), (350, 650), (950, 1500)]
        ]
        dummy_plan = {
            "total_strokes": 1,
            "layers": [{
                "name": "Test Layer",
                "color_rgb": (255, 255, 255),
                "thickness": "med",
                "touch_info": {"target_x": 389, "target_y": 2184, "is_long_press": False},
                "strokes": out_of_bounds_strokes
            }],
            "transform": CanvasTransform(offset_x=0, offset_y=0, scale=1.0),
            "custom_bounds": custom_bounds
        }
        worker = ExecutionWorker(self.fake_adb, dummy_plan, {})
        sanitized = worker._convert_and_sanitize_batch(out_of_bounds_strokes)
        for stroke in sanitized:
            for pt in stroke:
                self.assertGreaterEqual(pt[0], custom_bounds["left"])
                self.assertLessEqual(pt[0], custom_bounds["right"])
                self.assertGreaterEqual(pt[1], custom_bounds["top"])
                self.assertLessEqual(pt[1], custom_bounds["bottom"])

if __name__ == "__main__":
    unittest.main()
