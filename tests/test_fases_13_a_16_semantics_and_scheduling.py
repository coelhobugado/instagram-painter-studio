"""
Testes unitários para Fases 13, 14, 15 e 16.
Cobre:
- Fase 13: Classificador de imagem, faces, subject e mapa de importância
- Fase 14: Linhas de fluxo RK4 e tensor de estrutura condicional
- Fase 15: Formas negativas e camada subtrativa de borracha
- Fase 16: Scheduler com custo físico real e preservação estrita de oclusão
"""
import unittest
import numpy as np
import cv2
from studio.vision.scene_classifier import SceneUnderstandingEngine, ImageClass, FacialLandmarks
from studio.planning.flow_streamlines import FlowFieldPainter
from studio.planning.negative_shapes import NegativeShapeEngine
from studio.planning.scheduler_v2 import PhysicalCostScheduler


class TestFases13a16SemanticsAndScheduling(unittest.TestCase):
    def test_fase13_image_classification_and_importance_map(self):
        """Valida classificação de cena, segmentação de subject e mapa de importância multimodal."""
        # Imagem sintética de line-art (fundo branco, linha preta)
        lineart_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(lineart_img, (100, 100), 40, (0, 0, 0), 2)
        cls_res, conf, metrics = SceneUnderstandingEngine.classify_image(lineart_img)
        self.assertEqual(cls_res, ImageClass.LINE_ART)
        self.assertGreater(conf, 0.8)

        # Subject segmentation
        subject_mask = SceneUnderstandingEngine.segment_subject_background(lineart_img)
        self.assertEqual(subject_mask.shape, (200, 200))

        # Mapa de importância com landmarks
        landmarks = FacialLandmarks(
            face_bbox=(50, 50, 100, 100),
            left_eye=(80, 85),
            right_eye=(120, 85),
            mouth=(100, 125),
        )
        importance = SceneUnderstandingEngine.build_importance_map(lineart_img, landmarks, subject_mask)
        self.assertEqual(importance.shape, (200, 200))
        # O olho deve ter importância superior à borda distante da imagem
        self.assertGreater(importance[85, 80], importance[10, 10])

    def test_fase14_flow_painter_and_rk4_streamlines(self):
        """Valida integração RK4 confinada na máscara e recomendação condicional de fluxo."""
        mask = np.zeros((150, 150), dtype=np.uint8)
        cv2.ellipse(mask, (75, 75), (50, 25), 30, 0, 360, 255, -1)

        # Recomendação para tipo orgânico "hair" com orientação elíptica
        should_flow, coh, reason = FlowFieldPainter.should_use_flow_strategy(mask, semantic_type="hair")
        self.assertTrue(should_flow)
        self.assertIn("FLOW_FILL", reason)

        # Vetores diretores
        vx, vy, coherence = FlowFieldPainter.compute_frame_field(mask)
        self.assertEqual(vx.shape, (150, 150))
        self.assertEqual(vy.shape, (150, 150))

        # Streamline RK4 a partir do centroide
        streamline = FlowFieldPainter.integrate_rk4_streamline(75.0, 75.0, vx, vy, mask, step_size=2.0)
        self.assertGreater(len(streamline), 5)
        # Todos os pontos da streamline devem estar estritamente dentro da máscara
        for px, py in streamline:
            self.assertGreater(mask[py, px], 0)

    def test_fase15_negative_shapes_and_eraser_layer(self):
        """Valida identificação de regiões de brilho para apagamento por borracha."""
        target = np.ones((100, 100, 3), dtype=np.uint8) * 50
        # Catchlight alvo muito brilhante no centro
        cv2.circle(target, (50, 50), 5, (255, 255, 255), -1)

        painted_base = np.ones((100, 100, 3), dtype=np.uint8) * 50

        neg_mask = NegativeShapeEngine.identify_negative_regions(target, painted_base, contrast_threshold=50.0)
        self.assertGreater(np.count_nonzero(neg_mask), 0)

        eraser_strokes = NegativeShapeEngine.generate_negative_eraser_strokes(neg_mask)
        self.assertGreater(len(eraser_strokes), 0)

        eraser_layer = NegativeShapeEngine.build_eraser_layer(eraser_strokes)
        self.assertIsNotNone(eraser_layer)
        self.assertEqual(eraser_layer["brush"], "eraser")
        self.assertEqual(eraser_layer["purpose"], "negative_stroke")

    def test_fase16_physical_cost_scheduler_and_occlusion_preservation(self):
        """Valida cálculo de custo físico e respeito estrito da ordem de oclusão de camadas."""
        stroke1 = {
            "id": "s1", "layer_id": 0, "color_rgb": (255, 0, 0), "brush": "solid",
            "points": [(10, 10), (10, 50)]
        }
        stroke2_same_color = {
            "id": "s2", "layer_id": 0, "color_rgb": (255, 0, 0), "brush": "solid",
            "points": [(12, 50), (12, 100)]
        }
        stroke3_diff_color = {
            "id": "s3", "layer_id": 0, "color_rgb": (0, 255, 0), "brush": "solid",
            "points": [(12, 50), (12, 100)]
        }
        stroke4_layer1 = {
            "id": "s4", "layer_id": 1, "color_rgb": (0, 0, 255), "brush": "solid",
            "points": [(20, 20), (20, 40)]
        }

        # Transição para mesma cor deve ser muito mais barata que para cor diferente
        cost_same = PhysicalCostScheduler.calculate_transition_cost(stroke1, stroke2_same_color)
        cost_diff = PhysicalCostScheduler.calculate_transition_cost(stroke1, stroke3_diff_color)
        self.assertLess(cost_same, cost_diff)

        # Agendamento respeitando camadas (layer_id 0 DEVE preceder layer_id 1)
        unordered = [stroke4_layer1, stroke3_diff_color, stroke1, stroke2_same_color]
        scheduled = PhysicalCostScheduler.schedule_layers_and_strokes(unordered)
        
        scheduled_layer_ids = [s["layer_id"] for s in scheduled]
        self.assertEqual(scheduled_layer_ids, [0, 0, 0, 1])  # Layer 0 sempre antes de Layer 1!


if __name__ == "__main__":
    unittest.main()
