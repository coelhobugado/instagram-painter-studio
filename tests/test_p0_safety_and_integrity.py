# -*- coding: utf-8 -*-
"""
Suíte de Testes Automatizados de Validação e Bloqueio Fail-Closed para todo P0.
Cobre exaustivamente as correções P0.1 a P0.6 apontadas na Auditoria:
- P0.1: Canvas safety requer captura válida e não aceita screenshot=None
- P0.2: QualityGate fail-closed por padrão no MCP, limites estritos e métricas reais
- P0.3: Divergência visual ou ausência de imagem pós-camada retorna success=False
- P0.4 e P0.5: QualityGate estritamente obrigatório em todos os executores e MCP
- P0.6: MCPServer responsivo com atendimento imediato a ig_emergency_kill
"""
import unittest
import math
import time
import threading
import numpy as np
from unittest.mock import MagicMock, patch

from studio.device.ui_state_machine import UIStateMachine, UIState
from studio.device.quality_gate import QualityGate, QualityGateResult
from studio.execution_orchestrator import ExecutionOrchestrator
from studio.execution_worker import ExecutionWorker
from studio.device.fake_adb import FakeADB
from studio.mcp.server import MCPServer
from studio.mcp.handlers_planning import handle_plan_artwork, PLANS_CACHE
from studio.mcp.handlers_execution import handle_execute_layer
from studio.mcp.protocol import MCPProtocol
from studio.storage.execution_artifact import ExecutionArtifact
from studio.core.models import CanvasTransform, StrokePlan


class TestP0SafetyAndIntegrity(unittest.TestCase):

    def setUp(self):
        self.fake_adb = FakeADB()
        self.state_machine = UIStateMachine()

    # --------------------------------------------------------------------------
    # P0.1: Canvas safety requer captura válida e não aceita screenshot=None
    # --------------------------------------------------------------------------
    def test_p0_1_validate_safety_rejects_none_screenshot(self):
        """P0.1: validate_safety_for_drawing deve rejeitar screenshot=None quando não é possível capturar."""
        mock_adb = MagicMock()
        mock_adb.is_connected = True
        mock_adb.is_screen_on.return_value = True
        mock_adb.is_instagram_foreground.return_value = True
        mock_adb.capture_screenshot.return_value = None  # Captura falhou

        is_safe, reason = self.state_machine.validate_safety_for_drawing(mock_adb, screenshot_bgr=None)
        self.assertFalse(is_safe)
        self.assertEqual(self.state_machine.current_state, UIState.ERROR)
        self.assertIn("Captura de tela ausente ou nula", reason)

    def test_p0_1_validate_safety_rejects_empty_screenshot(self):
        """P0.1: validate_safety_for_drawing deve rejeitar screenshot vazio (size == 0)."""
        mock_adb = MagicMock()
        mock_adb.is_connected = True
        mock_adb.is_screen_on.return_value = True
        mock_adb.is_instagram_foreground.return_value = True

        empty_img = np.array([])
        is_safe, reason = self.state_machine.validate_safety_for_drawing(mock_adb, screenshot_bgr=empty_img)
        self.assertFalse(is_safe)
        self.assertEqual(self.state_machine.current_state, UIState.ERROR)

    def test_p0_1_orchestrator_validate_safety_fails_closed_when_capture_is_none(self):
        """P0.1: ExecutionOrchestrator.validate_safety retorna False se capture_screenshot() retornar None."""
        mock_adb = MagicMock()
        mock_adb.capture_screenshot.return_value = None
        orchestrator = ExecutionOrchestrator(mock_adb)

        is_safe, reason, confidence = orchestrator.validate_safety()
        self.assertFalse(is_safe)
        self.assertIn("ausente ou nula", reason)
        self.assertEqual(confidence, 0.0)

    # --------------------------------------------------------------------------
    # P0.2: QualityGate Fail-Closed por padrão, limites estritos e métricas reais
    # --------------------------------------------------------------------------
    def test_p0_2_mcp_handler_uses_fail_closed_qg_by_default(self):
        """P0.2: handle_execute_layer deve ter fail_closed_qg=True por padrão e bloquear plano reprovado."""
        bad_plan = {
            "layers": [
                {
                    "name": "Camada Inválida",
                    "color_rgb": (0, 0, 0),
                    "strokes": [[(10, 10), (20, 20)]],
                    "thickness": "min"
                }
            ],
            "quality_gate": QualityGateResult(
                passed=False,
                summary="QualityGate Reprovado por teste",
                failed_checks=["preview_quality_pass"]
            )
        }
        PLANS_CACHE["bad_plan_test"] = bad_plan

        res = handle_execute_layer({"plan_id": "bad_plan_test", "layer_index": 0})
        self.assertFalse(res["success"])
        self.assertEqual(res["stage"], "quality_gate")
        self.assertIn("QualityGate REPROVADO", res["error"])

    def test_mcp_cannot_disable_execution_safety(self):
        plan_id = "unsafe_override_test"
        PLANS_CACHE[plan_id] = {
            "layers": [{"strokes": [[(10, 10), (20, 20)]]}],
            "quality_gate": QualityGateResult(passed=False, summary="Reprovado"),
        }
        try:
            result = handle_execute_layer({
                "plan_id": plan_id,
                "layer_index": 0,
                "fail_closed_qg": False,
                "fail_closed_ui": False,
            })
            self.assertFalse(result["success"])
            self.assertEqual(result["stage"], "quality_gate")
        finally:
            PLANS_CACHE.pop(plan_id, None)

    def test_mcp_always_requests_visual_verification(self):
        plan_id = "verify_override_test"
        PLANS_CACHE[plan_id] = {"layers": [{"strokes": [[(10, 10), (20, 20)]]}]}
        try:
            with patch("studio.mcp.handlers_execution.ExecutionOrchestrator") as orchestrator:
                handle_execute_layer({"plan_id": plan_id, "verify_fidelity": False})
                kwargs = orchestrator.return_value.execute_layer.call_args.kwargs
                self.assertTrue(kwargs["verify_fidelity"])
                self.assertTrue(kwargs["fail_closed_qg"])
                self.assertTrue(kwargs["fail_closed_ui"])
        finally:
            PLANS_CACHE.pop(plan_id, None)

    def test_orchestrator_rejects_missing_gate(self):
        orchestrator = ExecutionOrchestrator(self.fake_adb)
        result = orchestrator.execute_layer(
            {"layers": [{"strokes": [[(10, 10), (20, 20)]]}]},
            layer_index=0,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "quality_gate")

        bypass = orchestrator.execute_layer(
            {"layers": [{"strokes": [[(10, 10), (20, 20)]]}]},
            layer_index=0, fail_closed_qg=False, fail_closed_ui=False,
        )
        self.assertFalse(bypass["success"])
        self.assertEqual(bypass["stage"], "quality_gate")

    def test_quality_gate_rejects_empty_layer_and_motion_mismatch(self):
        motion = MagicMock(total_strokes=2, plan_hash="hash")
        bounds = {"safe_bounds": {"left": 0, "top": 0, "right": 100, "bottom": 100}}
        result = QualityGate.evaluate(
            geometric_layers=[
                {"strokes": [[(10, 10), (20, 20)]]},
                {"strokes": []},
            ],
            motion_plan=motion,
            ui_validation_res=None,
            device_profile=bounds,
            time_budget_s=600,
            require_live_ui=False,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.check_details["geometry_valid"])
        self.assertFalse(result.check_details["motion_valid"])

    def test_quality_gate_never_invents_preview_metrics(self):
        layers = [{"strokes": [[(10, 10), (20, 20)]]}]
        motion = MagicMock(total_strokes=1, plan_hash="hash", total_duration_ms=1000)
        result = QualityGate.evaluate(
            geometric_layers=layers, motion_plan=motion,
            ui_validation_res=None,
            device_profile={"safe_bounds": {"left": 0, "top": 0, "right": 100, "bottom": 100}},
            time_budget_s=600, require_live_ui=False,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.check_details["preview_quality_pass"])

    def test_p0_2_quality_gate_rejects_nan_and_inf_coordinates(self):
        """P0.2: QualityGate.evaluate deve rejeitar coordenadas NaN ou infinitas."""
        nan_layers = [
            {
                "strokes": [[(10, 20), (float("nan"), 40)]]
            }
        ]
        device_profile = {"safe_bounds": {"left": 0, "top": 0, "right": 1000, "bottom": 2000}}
        res = QualityGate.evaluate(
            geometric_layers=nan_layers,
            motion_plan=None,
            ui_validation_res=None,
            device_profile=device_profile,
            time_budget_s=60.0
        )
        self.assertFalse(res.passed)
        self.assertIn("geometry_valid", res.failed_checks[0])

    def test_p0_2_quality_gate_rejects_incomplete_or_inverted_bounds(self):
        """P0.2: QualityGate.evaluate deve rejeitar perfil com safe_bounds faltando bottom ou invertido."""
        valid_layers = [{"strokes": [[(10, 20), (30, 40)]]}]

        # Faltando 'bottom'
        missing_bottom = {"safe_bounds": {"left": 0, "top": 0, "right": 1000}}
        res1 = QualityGate.evaluate(
            geometric_layers=valid_layers,
            motion_plan=None,
            ui_validation_res=None,
            device_profile=missing_bottom,
            time_budget_s=60.0
        )
        self.assertFalse(res1.passed)
        self.assertTrue(any("device_profile_valid" in f for f in res1.failed_checks))

        # Limites invertidos: left >= right
        inverted_bounds = {"safe_bounds": {"left": 1000, "top": 0, "right": 500, "bottom": 2000}}
        res2 = QualityGate.evaluate(
            geometric_layers=valid_layers,
            motion_plan=None,
            ui_validation_res=None,
            device_profile=inverted_bounds,
            time_budget_s=60.0
        )
        self.assertFalse(res2.passed)
        self.assertTrue(any("device_profile_valid" in f for f in res2.failed_checks))

    # --------------------------------------------------------------------------
    # P0.3: Divergência ou ausência de imagem pós-camada retorna success=False
    # --------------------------------------------------------------------------
    def test_p0_3_verify_layer_fidelity_returns_false_for_missing_expected_or_live_img(self):
        """P0.3: verify_layer_fidelity não deve retornar True se aligned_live ou expected_img forem None."""
        orchestrator = ExecutionOrchestrator(self.fake_adb)
        layer = {"strokes": [[(10, 10), (20, 20)]]}

        is_ok, reason, metrics = orchestrator.verify_layer_fidelity(None, np.zeros((100, 100, 3), dtype=np.uint8), layer)
        self.assertFalse(is_ok)
        self.assertEqual(metrics.get("status"), "indeterminate")

        is_ok2, reason2, metrics2 = orchestrator.verify_layer_fidelity(np.zeros((100, 100, 3), dtype=np.uint8), None, layer)
        self.assertFalse(is_ok2)
        self.assertEqual(metrics2.get("status"), "indeterminate")

        tiny = {"strokes": [[(1, 1)]]}
        ok3, _reason3, metrics3 = orchestrator.verify_layer_fidelity(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8), tiny,
        )
        self.assertFalse(ok3)
        self.assertEqual(metrics3.get("status"), "indeterminate")

    def test_p0_3_execute_layer_returns_failure_when_divergence_is_detected(self):
        """P0.3: execute_layer DEVE retornar success=False quando houver divergência visual detectada."""
        orchestrator = ExecutionOrchestrator(self.fake_adb)
        orchestrator.verify_layer_fidelity = MagicMock(return_value=(False, "Cor totalmente divergente (ΔE=55.0)", {"delta_e": 55.0}))

        plan = {
            "layers": [
                {
                    "name": "Camada 1",
                    "color_rgb": (255, 0, 0),
                    "strokes": [[(100, 500), (200, 600)]],
                    "thickness": "min"
                }
            ],
            "cumulative_layer_images": [np.zeros((800, 800, 3), dtype=np.uint8)],
            "quality_gate": QualityGateResult(passed=True, summary="OK")
        }

        res = orchestrator.execute_layer(plan, layer_index=0, verify_fidelity=True)
        self.assertFalse(res["success"], "execute_layer deve retornar success=False quando houver divergência visual")
        self.assertTrue(res.get("divergence_detected"))
        self.assertEqual(res.get("stage"), "closed_loop_divergence")
        self.assertIn("Divergência visual detectada", res.get("error"))

    def test_p0_3_execute_artifact_halts_immediately_on_divergence(self):
        """P0.3: execute_artifact interrompe a execução e não avança para camadas seguintes se houver divergência."""
        orchestrator = ExecutionOrchestrator(self.fake_adb)

        # Mock de execute_layer retornando divergência na primeira camada
        mock_divergence_report = {
            "success": False,
            "stage": "closed_loop_divergence",
            "divergence_detected": True,
            "error": "Divergência detectada na camada 0"
        }
        orchestrator.execute_layer = MagicMock(return_value=mock_divergence_report)

        artifact = MagicMock()
        artifact.layers = [{"name": "C1", "strokes": [[(10, 10)]], "risk_level": "HIGH"}, {"name": "C2", "strokes": [[(20, 20)]], "risk_level": "HIGH"}]
        artifact.transform = None
        artifact.quality_gate = QualityGateResult(passed=True, summary="OK")

        res = orchestrator.execute_artifact(artifact, selective_closed_loop=True)
        self.assertFalse(res["success"])
        self.assertTrue(res.get("divergence_detected"))
        # Garante que só executou a primeira camada e parou
        self.assertEqual(orchestrator.execute_layer.call_count, 1)

    # --------------------------------------------------------------------------
    # P0.4 e P0.5: QualityGate estritamente obrigatório em todos os executores e MCP
    # --------------------------------------------------------------------------
    def test_p0_5_handlers_planning_reports_false_when_no_quality_gate(self):
        """P0.5: handle_plan_artwork informa quality_gate_passed=False quando não houver gate."""
        with patch("studio.painter_engine.PainterEngine.generate_layered_paint_plan") as mock_plan_gen:
            mock_plan_gen.return_value = {
                "plan_hash": "test_no_qg",
                "layers": [],
                "quality_gate": None  # Sem quality gate
            }
            res = handle_plan_artwork({"image_path": "inputs/flores.jpg", "style": "layered_paint"})
            self.assertFalse(res["quality_gate_passed"])

    def test_p0_5_execution_worker_fails_closed_when_quality_gate_fails(self):
        """P0.5: ExecutionWorker deve abortar imediatamente se o QualityGate for reprovado."""
        failed_qg = QualityGateResult(passed=False, summary="Gate Reprovado por Teste", failed_checks=["geometry_valid"])
        plan = {
            "layers": [{"name": "C1", "strokes": [[(10, 10)]], "thickness": "min", "color_rgb": (0, 0, 0)}],
            "quality_gate": failed_qg
        }

        finished_events = []
        worker = ExecutionWorker(
            self.fake_adb,
            plan,
            callbacks={"on_finished": lambda ok, msg: finished_events.append((ok, msg))}
        )
        worker.run()

        self.assertEqual(len(finished_events), 1)
        self.assertFalse(finished_events[0][0])
        self.assertIn("QualityGate REPROVADO", finished_events[0][1])

    def test_execution_worker_rejects_missing_gate_without_touches(self):
        events = []
        worker = ExecutionWorker(
            self.fake_adb,
            {"layers": [{"strokes": [[(10, 10), (20, 20)]]}]},
            callbacks={"on_finished": lambda ok, msg: events.append((ok, msg))},
        )
        worker.run()
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0][0])
        self.assertEqual(self.fake_adb.injected_strokes, [])

    def test_worker_layer_fidelity_rejects_missing_capture(self):
        worker = ExecutionWorker(self.fake_adb, {"layers": []}, callbacks={})
        ok, reason = worker._verify_layer_fidelity(
            None,
            np.zeros((20, 20, 3), dtype=np.uint8),
            {"strokes": [[(1, 1), (10, 10)]]},
        )
        self.assertFalse(ok)
        self.assertIn("indeterminada", reason)

    def test_worker_reports_first_stroke_rejected_by_safe_bounds(self):
        events = []
        plan = {
            "layers": [{
                "strokes": [[(-200, -200), (-100, -100)]],
                "touch_info": {"target_x": 100, "target_y": 2200},
            }],
            "total_strokes": 1,
            "quality_gate": QualityGateResult(passed=True, summary="Aprovado"),
        }
        worker = ExecutionWorker(
            self.fake_adb, plan,
            callbacks={"on_finished": lambda ok, msg: events.append((ok, msg))},
        )
        worker.run()
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0][0])
        self.assertIn("fora da área segura", events[0][1])
        self.assertEqual(self.fake_adb.injected_strokes, [])

    def test_manual_area_cannot_expand_device_safe_bounds(self):
        worker = ExecutionWorker(
            self.fake_adb,
            {"custom_bounds": {"left": 0, "top": 0, "right": 1080, "bottom": 2400}},
            callbacks={},
        )
        self.assertEqual(worker._convert_and_sanitize_batch([[(10, 1000), (30, 1000)]]), [])
        self.assertEqual(
            worker._convert_and_sanitize_batch([[(100, 1000), (130, 1000)]]),
            [[(100, 1000), (130, 1000)]],
        )

    def test_artifact_requires_boolean_gate_approval(self):
        artifact = ExecutionArtifact(
            plan_id="test", fingerprint="test", mode="line_art",
            image_class="line_art", transform=CanvasTransform(0, 0),
            layers=[], quality_gate={"passed": "false"},
        )
        self.assertFalse(artifact.is_quality_gate_approved)
        artifact.quality_gate = QualityGateResult(passed=True, summary="OK")
        self.assertTrue(artifact.is_quality_gate_approved)

    def test_worker_rejects_plan_changed_after_approval(self):
        layers = [{"strokes": [[(100, 500), (130, 500)]]}]
        plan = {
            "mode": "line_art", "seed": 42, "layers": layers,
            "plan_hash": StrokePlan.compute_plan_hash(42, "LINE_ART", layers),
            "quality_gate": QualityGateResult(passed=True, summary="OK"),
        }
        layers[0]["strokes"][0][1] = (140, 500)
        events = []
        ExecutionWorker(
            self.fake_adb, plan,
            callbacks={"on_finished": lambda ok, msg: events.append((ok, msg))},
        ).run()
        self.assertFalse(events[0][0])
        self.assertIn("Plano alterado", events[0][1])
        self.assertEqual(self.fake_adb.injected_strokes, [])

    def test_worker_rejects_checkpoint_beyond_plan(self):
        events = []
        plan = {
            "layers": [{"strokes": [[(100, 500), (130, 500)]]}],
            "total_strokes": 1,
            "quality_gate": QualityGateResult(passed=True, summary="OK"),
        }
        ExecutionWorker(
            self.fake_adb, plan,
            callbacks={"on_finished": lambda ok, msg: events.append((ok, msg))},
            start_layer=2,
        ).run()
        self.assertFalse(events[0][0])
        self.assertIn("checkpoint inválido", events[0][1])
        self.assertEqual(self.fake_adb.injected_strokes, [])

    # --------------------------------------------------------------------------
    # P0.6: MCPServer responsivo e atendimento imediato de ig_emergency_kill
    # --------------------------------------------------------------------------
    def test_p0_6_mcp_server_dispatches_tools_asynchronously_and_handles_emergency_kill_immediately(self):
        """P0.6: MCPServer não bloqueia o loop de leitura e executa ig_emergency_kill sem atraso."""
        server = MCPServer()

        # Simula uma ferramenta longa de pintura rodando no executor
        tool_started = threading.Event()
        tool_can_finish = threading.Event()

        def slow_tool_handler(args):
            tool_started.set()
            tool_can_finish.wait(timeout=2.0)
            return {"success": True, "message": "done"}

        with patch.dict("studio.mcp.server.TOOL_HANDLERS", {"slow_tool": slow_tool_handler}):
            # Dispara ferramenta longa
            msg_slow = {
                "jsonrpc": "2.0",
                "id": "req_slow",
                "method": "tools/call",
                "params": {"name": "slow_tool", "arguments": {}}
            }
            server._dispatch(msg_slow)

            # Aguarda a ferramenta iniciar na thread de trabalho
            self.assertTrue(tool_started.wait(timeout=1.0))

            # Enquanto a ferramenta longa está rodando, envia ig_emergency_kill
            kill_handled = threading.Event()
            def mock_kill(args):
                kill_handled.set()
                return {"success": True, "message": "interrompido"}

            with patch.dict("studio.mcp.server.TOOL_HANDLERS", {"ig_emergency_kill": mock_kill}):
                msg_kill = {
                    "jsonrpc": "2.0",
                    "id": "req_kill",
                    "method": "tools/call",
                    "params": {"name": "ig_emergency_kill", "arguments": {}}
                }
                # _dispatch deve processar o kill switch imediatamente na thread atual
                server._dispatch(msg_kill)
                self.assertTrue(kill_handled.is_set(), "ig_emergency_kill deve ser executado imediatamente mesmo com ferramenta longa em andamento")

            tool_can_finish.set()
            server.executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
