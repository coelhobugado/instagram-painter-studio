# -*- coding: utf-8 -*-
"""
Testes Unitários da FASE 2: Motion Compiler e Eventos de Touchscreen.
Em estrita conformidade com o INSTAGRAM_PAINTER_ROADMAP_APICE.md:
- 2.1: Novas estruturas (StrokeIntent, GeometricStroke, MotionStroke, ExecutionBatch).
- 2.2: Reamostragem por comprimento de arco com max_gap configurável e curvatura adaptativa.
- 2.3: Quantização inteira única.
- 2.4: Remoção de nós duplicados consecutivos.
- 2.5: Remoção de colineares redundantes preservando gap máximo.
- 2.6: Timing explícito com timestamps monotônicos e hash determinístico estável.
"""
import unittest
import math
from studio.core.models import CanvasTransform
from studio.planning.motion_compiler import (
    MotionCompiler,
    StrokeIntent,
    GeometricStroke,
    TouchAction,
    CompiledMotionPlan
)

class TestFase2MotionCompiler(unittest.TestCase):

    def setUp(self):
        self.transform = CanvasTransform(offset_x=100, offset_y=200)

    def test_2_1_structures_instantiation(self):
        """2.1: Valida criação e integridade das estruturas de dados."""
        intent = StrokeIntent(region_id="eye_outline", purpose="contour", target_width=2.5, priority=0.9)
        geom = GeometricStroke(points=[(10.0, 10.0), (50.0, 50.0)], stroke_type="spline", intent=intent)
        self.assertEqual(geom.intent.purpose, "contour")
        self.assertEqual(len(geom.points), 2)

    def test_2_2_resample_by_arc_length_respects_max_gap(self):
        """2.2: Garante que nenhum segmento resultante tenha gap maior que max_gap."""
        # Linha longa de 100px com apenas 2 pontos de entrada
        raw_points = [(0.0, 0.0), (100.0, 0.0)]
        max_gap = 2.0
        resampled = MotionCompiler.resample_by_arc_length(raw_points, max_gap=max_gap)

        self.assertGreater(len(resampled), 50)
        for i in range(len(resampled) - 1):
            p1, p2 = resampled[i], resampled[i + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self.assertLessEqual(dist, max_gap + 1e-4)

    def test_2_2_adaptive_curvature_sampling(self):
        """2.2: Garante que trechos com curvatura acentuada recebam maior densidade de amostragem."""
        # Curva em 90 graus: (0, 0) -> (10, 0) -> (10, 10)
        corner = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        resampled = MotionCompiler.resample_by_arc_length(corner, max_gap=3.0, adaptive_curvature=True)
        # Deve conter mais pontos que uma linha reta de mesmo comprimento total (20px)
        straight = [(0.0, 0.0), (20.0, 0.0)]
        resampled_straight = MotionCompiler.resample_by_arc_length(straight, max_gap=3.0, adaptive_curvature=True)
        self.assertGreater(len(resampled), len(resampled_straight))

    def test_2_4_deduplication_removes_consecutive_duplicates(self):
        """2.4: Garante que coordenadas consecutivas idênticas após arredondamento viram evento único."""
        screen_pts = [(10.1, 20.2), (10.2, 20.1), (10.0, 20.0), (50.0, 60.0)]
        # Os três primeiros arredondam para (10, 20)
        quantized = MotionCompiler.quantize_and_deduplicate(screen_pts)
        self.assertEqual(len(quantized), 2)
        self.assertEqual(quantized[0], (10, 20))
        self.assertEqual(quantized[1], (50, 60))

    def test_2_5_remove_redundant_collinears(self):
        """2.5: Remove nós colineares redundantes desde que o gap máximo continue respeitado."""
        # Linha reta com 5 pontos colineares: (0,0), (1,0), (2,0), (3,0), (4,0)
        collinear = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        filtered = MotionCompiler.remove_redundant_collinears(collinear, max_gap=5.0)
        # Deve reduzir apenas para as extremidades (0,0) e (4,0)
        self.assertEqual(filtered, [(0, 0), (4, 0)])

    def test_2_6_explicit_timing_monotonicity(self):
        """2.6: Garante que MotionEvents possuem timestamps estritamente monotônicos e sequenciais."""
        geom = GeometricStroke(points=[(0.0, 0.0), (30.0, 0.0), (60.0, 20.0)])
        m_stroke, next_t = MotionCompiler.compile_stroke(
            geom, transform=self.transform, start_time_ms=100, sample_interval_ms=8, max_gap=3.0
        )
        self.assertGreater(len(m_stroke.events), 2)
        self.assertEqual(m_stroke.events[0].action, TouchAction.DOWN)
        self.assertEqual(m_stroke.events[-1].action, TouchAction.UP)

        for i in range(len(m_stroke.events) - 1):
            t_curr = m_stroke.events[i].timestamp_ms
            t_next = m_stroke.events[i + 1].timestamp_ms
            self.assertGreater(t_next, t_curr, "Timestamps devem ser estritamente monotônicos crescentes.")

    def test_fase2_plan_determinism_and_stable_hash(self):
        """Critério de Aceite: Mesma entrada gera exatamente o mesmo Motion Plan com hash idêntico."""
        layers = [
            {
                "strokes": [[(10, 10), (50, 10), (50, 50)]],
                "brush": "solid",
                "thickness": "min",
                "color_rgb": (255, 0, 0)
            }
        ]
        plan1 = MotionCompiler.compile_plan(layers, self.transform, target_hz=120, max_gap=2.0)
        plan2 = MotionCompiler.compile_plan(layers, self.transform, target_hz=120, max_gap=2.0)

        self.assertEqual(plan1.plan_hash, plan2.plan_hash)
        self.assertEqual(plan1.total_events, plan2.total_events)
        self.assertEqual(plan1.total_duration_ms, plan2.total_duration_ms)
        self.assertGreater(len(plan1.batches), 0)

if __name__ == "__main__":
    unittest.main()
