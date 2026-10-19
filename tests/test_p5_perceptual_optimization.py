# -*- coding: utf-8 -*-
"""
Testes de Qualidade P5: Otimização Perceptual Global e Poda Ativa de Pinceladas.
Validações essenciais:
1. Poda de redundância e recorte de traços contra feições proibidas/olhos.
2. Governança de orçamento de execução por Ganho por Segundo (GlobalObjectiveLoss).
3. Integração de métricas topológicas e ordenação em LineArtEngine.
4. Limite de camadas de refinamento em simulação para evitar poluição visual.
"""
import unittest
import os
import cv2
import numpy as np

from studio.planning.perceptual_optimizer import PerceptualPlanOptimizer
from studio.planning.line_art_engine import LineArtEngine
from studio.painter_engine import PainterEngine
from studio.core.models import CanvasTransform
from studio.vision.perceptual_color import PerceptualColorEngine
from studio.simulation.simulator_v2 import CalibratedSimulatorV2


class TestP5PerceptualOptimization(unittest.TestCase):
    def test_clip_stroke_against_forbidden_mask(self):
        """Valida que traços que penetram em feições protegidas são recortados/divididos."""
        forbidden_mask = np.zeros((100, 100), dtype=np.uint8)
        # Cria uma zona proibida circular de raio 15 centralizada em (50, 50)
        cv2.circle(forbidden_mask, (50, 50), 15, 255, -1)

        # Traço horizontal que cruza a zona proibida de (10, 50) até (90, 50)
        crossing_stroke = [(x, 50) for x in range(10, 91, 5)]

        clipped = PerceptualPlanOptimizer.clip_stroke_against_mask(crossing_stroke, forbidden_mask)

        # O traço deve ser dividido em 2 sub-traços úteis (antes e depois do círculo)
        self.assertEqual(len(clipped), 2, "O traço deve ser dividido em dois segmentos fora da zona proibida.")

        # Valida que nenhum ponto restante está dentro da zona proibida
        for sub in clipped:
            for px, py in sub:
                self.assertEqual(int(forbidden_mask[py, px]), 0, f"Ponto ({px}, {py}) permaneceu dentro da zona proibida!")

    def test_prune_redundant_strokes(self):
        """Valida que traços redundantes na mesma área são podados para evitar manchas escuras."""
        shape = (100, 100)
        stroke1 = [(x, 30) for x in range(20, 80, 2)]
        # stroke2 é quase idêntico ao stroke1, apenas 1px deslocado (sobreposição > 85%)
        stroke2 = [(x, 31) for x in range(20, 80, 2)]
        # stroke3 é independente em outra posição
        stroke3 = [(x, 70) for x in range(20, 80, 2)]

        strokes = [stroke1, stroke2, stroke3]
        pruned = PerceptualPlanOptimizer.prune_redundant_strokes(strokes, brush_width=8.0, shape=shape)

        self.assertEqual(len(pruned), 2, "O traço redundante stroke2 deve ser descartado.")
        self.assertIn(stroke1, pruned)
        self.assertIn(stroke3, pruned)

    def test_line_art_topology_metrics_and_ordering(self):
        """Valida que LineArtEngine integra topology_metrics e ordenação hierárquica de traços."""
        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        cv2.circle(img, (60, 60), 30, (0, 0, 0), 3)

        color_engine = PerceptualColorEngine()
        simulator = CalibratedSimulatorV2()
        transform = CanvasTransform(scale=1.0, offset_x=0, offset_y=0)

        plan = LineArtEngine.generate_plan(img, color_engine, simulator, transform)

        self.assertIn("topology_metrics", plan, "O plano de line art deve conter topology_metrics.")
        metrics = plan["topology_metrics"]
        self.assertTrue(metrics["passed"], "A validação métrica e topológica deve ser aprovada.")
        self.assertGreaterEqual(metrics["topology_preservation"], 0.90)

    def test_optimize_plan_budget_and_gain(self):
        """Valida que a governança de orçamento corta traços secundários respeitando o time budget."""
        dummy_plan = {
            "estimated_time_s": 500.0,
            "layers": [
                {"name": "Fundo", "purpose": "background", "stroke_count": 50, "point_count": 200},
                {"name": "Contorno Principal", "purpose": "contour", "stroke_count": 60, "point_count": 300},
                {"name": "Olhos", "purpose": "highlight", "stroke_count": 20, "point_count": 100},
                {"name": "Hachura Opcional 1", "purpose": "colored_detail", "stroke_count": 400, "point_count": 1500},
                {"name": "Hachura Opcional 2", "purpose": "colored_detail", "stroke_count": 500, "point_count": 2000},
            ]
        }

        # Orçamento apertado de 100 segundos
        PerceptualPlanOptimizer.optimize_plan_budget_and_gain(dummy_plan, time_budget_s=100.0)

        # As camadas essenciais (Fundo, Contorno, Olhos) devem ser preservadas
        purposes = [l["purpose"] for l in dummy_plan["layers"]]
        self.assertIn("background", purposes)
        self.assertIn("contour", purposes)
        self.assertIn("highlight", purposes)

        # O tempo total deve estar contido no orçamento
        self.assertLessEqual(dummy_plan["estimated_time_s"], 105.0)

    def test_refinement_layers_capped_in_simulation(self):
        """Valida que optimize_plan_in_simulation não empilha mais de 2 camadas de refinamento."""
        engine = PainterEngine()
        img = np.ones((200, 200, 3), dtype=np.uint8) * 220
        cv2.rectangle(img, (40, 40), (160, 160), (30, 30, 200), -1)

        import tempfile
        td = tempfile.TemporaryDirectory()
        p = os.path.join(td.name, "box.png")
        cv2.imwrite(p, img)

        plan = engine.generate_layered_paint_plan(p, num_colors=3, seed=42, optimize_in_simulation=True)

        refinement_layers = [l for l in plan["layers"] if l.get("purpose") == "simulation_optimization"]
        self.assertLessEqual(len(refinement_layers), 2, "Não devem existir mais de 2 camadas de refinamento em simulação.")
        td.cleanup()


if __name__ == "__main__":
    unittest.main()
