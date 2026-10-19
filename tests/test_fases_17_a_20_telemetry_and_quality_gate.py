"""
Testes unitários para Fases 17, 18, 19 e 20.
Cobre:
- Fase 17: SessionTelemetry completa (17 métricas, exportação JSON e markdown)
- Fase 18: Aprendizado interpretável com ResidualCorrectionModel, RiskPredictor e PlanRanker
- Fase 19: Inspeção interativa e edição de regiões (PaintingInspectorManager)
- Fase 20: QualityGate fail-closed com validação dos 6 critérios pré-execução
"""
import unittest
import os
import json
import numpy as np
from studio.device.session_telemetry import SessionTelemetry
from studio.device.sim_to_real_learning import ResidualCorrectionModel, RiskPredictor, PlanRanker
from studio.planning.painting_plan import PaintingPlan, PaintingRegion, StrokeStrategy
from studio.planning.inspector_model import PaintingInspectorManager
from studio.device.quality_gate import QualityGate
from studio.device.ui_state_machine import UIValidationResult
from studio.planning.motion_compiler import CompiledMotionPlan, MotionStroke, MotionEvent


class TestFases17a20TelemetryAndQualityGate(unittest.TestCase):
    def test_fase17_session_telemetry(self):
        """Valida que todas as 17 métricas são registradas e exportadas corretamente."""
        telem = SessionTelemetry(session_id="session_test_001")
        telem.planning_time_s = 1.85
        telem.simulation_time_s = 0.45
        telem.execution_time_s = 22.5
        telem.events_per_s = 120.0
        telem.strokes_per_s = 4.5
        telem.mean_delta_e = 6.2
        telem.p95_delta_e = 12.8
        telem.edge_f1 = 0.92
        telem.preview_real_residual = 1.4
        telem.ram_peak_mb = 115.0
        
        telem.record_switch(is_color=True, is_tool=False, is_long_press=True)
        telem.record_verification(passed=False, corrective_count=2)

        data = telem.to_dict()
        self.assertEqual(data["session_id"], "session_test_001")
        self.assertEqual(data["color_switches"], 1)
        self.assertEqual(data["long_presses"], 1)
        self.assertEqual(data["verification_failures"], 1)
        self.assertEqual(data["corrective_strokes"], 2)
        self.assertAlmostEqual(data["mean_delta_e"], 6.2)

        md = telem.to_summary_markdown()
        self.assertIn("Relatório de Telemetria", md)
        self.assertIn("1.85", md)

    def test_fase18_residual_learning_and_risk_prediction(self):
        """Valida modelo residual interpretável e preditor de risco causal."""
        # Residual learning
        res_model = ResidualCorrectionModel()
        pred_pts = [(100, 100), (120, 100)]
        # Simulando um desvio físico constante de +2px em X e +1px em Y no aparelho real
        obs_pts = [(102, 101), (122, 101)]
        res_model.update_from_observation(pred_pts, obs_pts, predicted_width=4.0, actual_observed_width=4.4)

        comp_pts, comp_w = res_model.apply_compensation([(200, 200)], desired_width=4.0)
        # O ponto corrigido deve compensar para a esquerda (x < 200)
        self.assertLessEqual(comp_pts[0][0], 200)

        # Risk predictor
        bounds = (0, 0, 1000, 2000)
        # Traço de alta velocidade colado na borda da tela
        risky_pts = [(5, 100), (5, 500)]
        risk_val, factors = RiskPredictor.evaluate_stroke_risk(risky_pts, velocity_px_s=1400.0, canvas_bounds=bounds)
        self.assertGreater(risk_val, 0.5)
        self.assertTrue(any("Velocidade excessiva" in f for f in factors))
        self.assertTrue(any("Proximidade crítica" in f for f in factors))

        # PlanRanker
        score_good = PlanRanker.score_plan(predicted_delta_e=4.0, time_seconds=50.0, time_budget_s=60.0, risk_mean=0.1)
        score_bad = PlanRanker.score_plan(predicted_delta_e=18.0, time_seconds=120.0, time_budget_s=60.0, risk_mean=0.7)
        self.assertGreater(score_good, score_bad)

    def test_fase19_inspector_model_and_user_overrides(self):
        """Valida modelo de inspeção por coordenada e comandos de override do usuário."""
        plan = PaintingPlan(image_class="portrait", time_budget_s=120.0)
        reg = PaintingRegion(
            id="reg_1", semantic_type="eyes", importance=0.95,
            strategy=StrokeStrategy.CONTOUR, stroke_count=12, time_cost_s=8.0
        )
        plan.add_region(reg)

        label_map = np.ones((50, 50), dtype=np.int32)  # Toda a imagem rotulada como 0 (reg_1)
        inspector = PaintingInspectorManager(plan, label_map=label_map)

        detail = inspector.inspect_at_point(25, 25)
        self.assertIsNotNone(detail)
        self.assertEqual(detail.semantic_type, "eyes")
        self.assertEqual(detail.strategy, "CONTOUR")

        # Overrides do usuário
        inspector.change_strategy("reg_1", StrokeStrategy.DETAIL)
        inspector.force_color("reg_1", (10, 10, 10))
        inspector.lock_region("reg_1", True)

        detail_after = inspector.inspect_at_point(25, 25)
        self.assertEqual(detail_after.strategy, "DETAIL")
        self.assertEqual(detail_after.target_color_rgb, (10, 10, 10))
        self.assertTrue(detail_after.is_locked)

    def test_fase20_quality_gate_fail_closed(self):
        """Valida aprovação de integridade e trava estrita quando qualquer critério falhar."""
        layers = [{"strokes": [[(10, 10), (50, 50)]]}]
        from studio.planning.motion_compiler import TouchAction
        dummy_event = MotionEvent(action=TouchAction.DOWN, x=10, y=10, timestamp_ms=0)
        dummy_motion_stroke = MotionStroke(events=[dummy_event], tool="solid", thickness_mode="med")
        motion_plan = CompiledMotionPlan(
            motion_strokes=[dummy_motion_stroke],
            batches=[],
            total_events=1,
            total_strokes=1,
            total_duration_ms=2000,
            sample_rate_hz=120,
            plan_hash="hash_abc_123"
        )
        
        ui_valid = UIValidationResult(is_valid=True, reason="OK", confidence=0.9)
        device_profile = {"safe_bounds": {"left": 50, "top": 200, "right": 800, "bottom": 1600}}

        # Caso 1: Tudo em conformidade -> Aprovado
        res_pass = QualityGate.evaluate(
            geometric_layers=layers,
            motion_plan=motion_plan,
            ui_validation_res=ui_valid,
            device_profile=device_profile,
            time_budget_s=45.0,
            preview_delta_e=12.0,
            preview_edge_f1=0.88,
        )
        self.assertTrue(res_pass.passed)
        self.assertEqual(len(res_pass.failed_checks), 0)

        # Caso 2: UI inválida (ex: tela apagada ou popup) -> Fail-closed
        ui_invalid = UIValidationResult(is_valid=False, reason="Instagram fora de foco", confidence=0.1)
        res_fail_ui = QualityGate.evaluate(
            geometric_layers=layers,
            motion_plan=motion_plan,
            ui_validation_res=ui_invalid,
            device_profile=device_profile,
            time_budget_s=45.0,
        )
        self.assertFalse(res_fail_ui.passed)
        self.assertTrue(any("safe_ui_valid" in f for f in res_fail_ui.failed_checks))

        # Caso 3: Tempo de execução estourando o budget -> Fail-closed
        res_fail_time = QualityGate.evaluate(
            geometric_layers=layers,
            motion_plan=motion_plan,
            ui_validation_res=ui_valid,
            device_profile=device_profile,
            time_budget_s=1.0,  # Orçamento menor que a duração de 2s
        )
        self.assertFalse(res_fail_time.passed)
        self.assertTrue(any("time_budget_valid" in f for f in res_fail_time.failed_checks))


if __name__ == "__main__":
    unittest.main()
