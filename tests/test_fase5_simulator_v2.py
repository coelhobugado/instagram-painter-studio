# -*- coding: utf-8 -*-
"""
Testes Unitários da FASE 5: Simulador Baseado no Motion Plan e Diagnóstico Tríplice.
Em estrita conformidade com o INSTAGRAM_PAINTER_ROADMAP_APICE.md:
- 5.1: Simulação consumindo exatamente MotionStroke[] (mesma entrada do executor).
- 5.2: Geração das três visualizações (Target, Predição Física, Resultado Real).
- 5.3: Cálculo e diagnóstico da tríade de erros (Original↔Target, Target↔Predição, Predição↔Real).
"""
import unittest
import numpy as np
import cv2
from studio.simulation.simulator_v2 import CalibratedSimulatorV2
from studio.planning.motion_compiler import MotionStroke, MotionEvent, TouchAction

class TestFase5Simulator(unittest.TestCase):

    def setUp(self):
        self.simulator = CalibratedSimulatorV2()

    def test_5_1_render_motion_plan_directly(self):
        """5.1: Garante que o simulador renderiza diretamente a partir de MotionStroke[]."""
        events = [
            MotionEvent(TouchAction.DOWN, 20, 20, 0),
            MotionEvent(TouchAction.MOVE, 50, 20, 8),
            MotionEvent(TouchAction.MOVE, 80, 20, 16),
            MotionEvent(TouchAction.UP, 80, 20, 24)
        ]
        stroke = MotionStroke(events=events, tool="solid", thickness_mode="med", color_rgb=(0, 255, 0))

        sim_canvas, cum = self.simulator.render_motion_plan(
            canvas_shape=(100, 100, 3),
            motion_strokes=[stroke],
            background_bgr=(0, 0, 0)
        )

        self.assertEqual(sim_canvas.shape, (100, 100, 3))
        # O traço verde desenhado deve ter pixels verdes ativos
        green_pixels = np.count_nonzero(sim_canvas[:, :, 1] > 100)
        self.assertGreater(green_pixels, 50)

    def test_5_3_compute_triplet_errors_and_diagnosis(self):
        """5.3: Valida cálculo das 3 métricas de erro e árvore diagnóstica."""
        # Cria 3 imagens sintéticas: Original, Target idêntico, Predição ligeiramente diferente, Real com desvio
        original = np.full((100, 100, 3), 150, dtype=np.uint8)
        target = original.copy()
        pred = original.copy()
        pred[20:40, 20:40] = 160  # Pequena diferença
        real = original.copy()
        real[0:50, 0:50] = 20     # Grande desvio simulando falha de hardware

        triplet = CalibratedSimulatorV2.compute_triplet_errors(
            original_img=original,
            target_executable=target,
            predicted_device=pred,
            observed_device=real
        )

        self.assertIn("error_original_target", triplet)
        self.assertIn("error_target_prediction", triplet)
        self.assertIn("error_prediction_real", triplet)

        # Original e Target são idênticos -> planning ok
        self.assertEqual(triplet["diagnosis"]["planning"], "ok")
        # Target e Predição muito próximos -> motion_compiler_simulator ok
        self.assertEqual(triplet["diagnosis"]["motion_compiler_simulator"], "ok")
        # Predição e Real muito distantes -> alerta de divergência de hardware
        self.assertEqual(triplet["diagnosis"]["physical_calibration"], "high_device_divergence")

if __name__ == "__main__":
    unittest.main()
