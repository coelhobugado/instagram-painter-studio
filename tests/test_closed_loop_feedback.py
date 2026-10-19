"""
Testes Unitários do Sistema de Feedback Visual, Error Map e Closed-Loop (Seções 27-34).
"""
import unittest
import numpy as np
from studio.feedback.error_map import ErrorMapEngine
from studio.feedback.correction_planner import CorrectionPlanner

class TestClosedLoop(unittest.TestCase):
    def test_error_map_zero_for_identical_images(self):
        """Imagens idênticas devem apresentar Mean Delta E próximo de 0 e SSIM = 1.0."""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128
        heatmap, metrics = ErrorMapEngine.compute_error_map(img, img)

        self.assertAlmostEqual(metrics["mean_delta_e"], 0.0, places=1)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=2)

    def test_correction_planner_protects_accurate_areas(self):
        """Valida que regiões já coincidentes com o alvo sejam preservadas pela máscara de proteção."""
        target = np.ones((200, 200, 3), dtype=np.uint8) * 180
        # Cria observação onde metade é idêntica e metade tem erro
        observed = target.copy()
        observed[:, 100:] = 20  # Metade direita com erro grave

        planner = CorrectionPlanner()
        plan_res = planner.plan_corrections(target, observed, offset_xy=(0, 0))

        # A metade esquerda deve estar na máscara de proteção (valor 255)
        prot_mask = plan_res["protected_mask"]
        self.assertEqual(prot_mask[50, 50], 255)
        # Deve ter gerado traços de correção para a área com falha
        self.assertGreater(plan_res["total_correction_strokes"], 0)

if __name__ == "__main__":
    unittest.main()
