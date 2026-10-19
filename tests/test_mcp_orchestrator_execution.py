# -*- coding: utf-8 -*-
"""
Testes Unitários da Integração Completa do Servidor MCP com o ExecutionOrchestrator.
Valida o caminho arquitetural unificado:
MCP -> ExecutionOrchestrator -> QualityGate -> Servo Vision -> CompiledMotionPlan
    -> PersistentTouchExecutor -> Closed-Loop -> Telemetry -> Device.
"""
import unittest
import os
import cv2
import numpy as np

from studio.painter_engine import PainterEngine
from studio.device.fake_adb import FakeADB
from studio.mcp.handlers_planning import PLANS_CACHE, handle_plan_artwork
from studio.mcp.handlers_execution import (
    handle_execute_layer,
    handle_execute_freehand_path,
    handle_capture_screen_diff,
)
from studio.execution_orchestrator import ExecutionOrchestrator
from studio.device.quality_gate import QualityGateResult


class TestMCPOrchestratorExecution(unittest.TestCase):

    def setUp(self):
        self.fake_adb = FakeADB()
        # Garante que get_adb retorne o FakeADB configurado
        from studio.mcp import handlers_device
        handlers_device._SHARED_ADB = self.fake_adb

        # Gera um plano de teste leve no cache
        plan_res = handle_plan_artwork({
            "image_path": "inputs/flores.jpg",
            "style": "stencil",
            "max_colors": 3
        })
        self.assertTrue(plan_res["success"])
        self.plan_id = plan_res["plan_id"]
        self.plan = PLANS_CACHE[self.plan_id]

    def test_mcp_execute_layer_through_orchestrator(self):
        """Valida que handle_execute_layer passa por todo o pipeline do orchestrator."""
        res = handle_execute_layer({
            "plan_id": self.plan_id,
            "layer_index": 0,
            "batch_size": 4,
            "verify_fidelity": True
        })

        self.assertTrue(res["success"], f"Falha na execução da camada: {res.get('error')}")
        self.assertIn("orchestrator_pipeline", res)
        pipeline = res["orchestrator_pipeline"]

        # 1. QualityGate avaliado
        self.assertIn("quality_gate", pipeline)
        self.assertIn("passed", pipeline["quality_gate"])

        # 2. Servo-Visão validada
        self.assertIn("servo_vision", pipeline)
        self.assertTrue(pipeline["servo_vision"]["validated"])
        self.assertGreater(pipeline["servo_vision"]["confidence"], 0.0)

        # 3. MotionCompiler executado (CompiledMotionPlan a 120 Hz)
        self.assertIn("motion_compiler", pipeline)
        self.assertEqual(pipeline["motion_compiler"]["sample_rate_hz"], 120)
        self.assertGreater(pipeline["motion_compiler"]["total_events"], 0)
        self.assertTrue(bool(pipeline["motion_compiler"]["plan_hash"]))

        # 4. PersistentTouchExecutor executou os eventos
        self.assertIn("persistent_touch", pipeline)
        touch_m = pipeline["persistent_touch"]
        self.assertIn("events_sent", touch_m)

        # 5. Closed-Loop executado
        self.assertIn("closed_loop", pipeline)
        cl = pipeline["closed_loop"]
        self.assertTrue(cl["verified"])
        self.assertIn("metrics", cl)

        # 6. Pinceladas realmente injetadas no dispositivo
        self.assertGreater(len(self.fake_adb.injected_strokes), 0)

    def test_mcp_execute_freehand_path_through_orchestrator(self):
        """Valida que handle_execute_freehand_path passa pelo pipeline orquestrado."""
        strokes = [
            [[200, 500], [250, 520], [300, 550], [350, 600]],
            [[400, 700], [450, 720], [500, 750]]
        ]
        res = handle_execute_freehand_path({
            "strokes": strokes,
            "thickness": "min",
            "color_name": "preto",
            "batch_size": 2
        })

        self.assertTrue(res["success"])
        self.assertIn("orchestrator_pipeline", res)
        self.assertEqual(res["executed_strokes"], 2)
        self.assertGreater(res["total_events"], 0)
        self.assertTrue(res["orchestrator_pipeline"]["servo_vision"]["validated"])

    def test_mcp_capture_screen_diff_with_registration_and_ciede2000(self):
        """Valida que handle_capture_screen_diff gera mapa CIEDE2000 e métricas alinhadas."""
        res = handle_capture_screen_diff({"plan_id": self.plan_id})
        self.assertTrue(res["success"])
        self.assertTrue(os.path.exists(res["screenshot_path"]))

        metrics = res["metrics"]
        self.assertIn("mean_delta_e", metrics)
        self.assertIn("ssim", metrics)
        self.assertIn("heatmap_path", metrics)
        self.assertTrue(os.path.exists(metrics["heatmap_path"]))

    def test_orchestrator_fail_closed_on_screen_off(self):
        """Valida o princípio fail-closed: se a tela estiver apagada, a execução deve ser bloqueada."""
        self.fake_adb.screen_on = False
        orchestrator = ExecutionOrchestrator(self.fake_adb)

        res = orchestrator.execute_layer(self.plan, layer_index=0, fail_closed_ui=True)
        self.assertFalse(res["success"])
        self.assertEqual(res["stage"], "servo_vision")
        self.assertIn("tela do celular está apagada", res["error"])

    def test_orchestrator_fail_closed_on_instagram_not_foreground(self):
        """Valida o princípio fail-closed: se o Instagram não for o app em primeiro plano, bloqueia."""
        self.fake_adb.instagram_foreground = False
        orchestrator = ExecutionOrchestrator(self.fake_adb)

        res = orchestrator.execute_layer(self.plan, layer_index=0, fail_closed_ui=True)
        self.assertFalse(res["success"])
        self.assertEqual(res["stage"], "servo_vision")
        self.assertIn("Instagram não está ativo", res["error"])


if __name__ == "__main__":
    unittest.main()
