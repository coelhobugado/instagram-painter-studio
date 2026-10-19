"""
Testes Unitários do Simulador Calibrado v2 e Determinismo de Seed (Seções 10, 54 e 82).
"""
import unittest
import numpy as np
from studio.simulation.simulator_v2 import CalibratedSimulatorV2

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = CalibratedSimulatorV2()

    def test_strict_determinism_seed(self):
        """Valida que a mesma seed gera exatamente os mesmos pixels bit-a-bit."""
        shape = (400, 400, 3)
        layers = [
            {
                "name": "Camada 1",
                "color_rgb": (200, 50, 50),
                "brush": "solid",
                "thickness": "thick",
                "strokes": [[(50, 50), (350, 50)], [(50, 100), (350, 100)]]
            },
            {
                "name": "Camada 2",
                "color_rgb": (50, 200, 50),
                "brush": "solid",
                "thickness": "min",
                "strokes": [[(50, 50), (50, 350)], [(100, 50), (100, 350)]]
            }
        ]

        canvas1, _ = self.simulator.render_simulation(shape, layers, seed=42)
        canvas2, _ = self.simulator.render_simulation(shape, layers, seed=42)

        # Deve ser estritamente idêntico (diferença máxima = 0)
        max_diff = np.max(np.abs(canvas1.astype(np.int32) - canvas2.astype(np.int32)))
        self.assertEqual(max_diff, 0)

    def test_calibrated_widths(self):
        """Valida que as espessuras carregadas correspondem às especificadas no profile."""
        w_min = self.simulator.get_calibrated_width("min")
        w_med = self.simulator.get_calibrated_width("med")
        w_thick = self.simulator.get_calibrated_width("thick")

        self.assertLess(w_min, w_med)
        self.assertLess(w_med, w_thick)
        self.assertGreaterEqual(w_min, 1)

if __name__ == "__main__":
    unittest.main()
