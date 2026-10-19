# -*- coding: utf-8 -*-
"""
Testes de Validação e Benchmarking de Latência da Camada Única de UI Timing.
Comprova a eliminação de sleeps duplicados e a adaptação automática para FakeADB (0 ms).
"""
import unittest
import time
from studio.device.ui_timing import UITimingManager, get_ui_timing
from studio.device.fake_adb import FakeADB
from studio.mcp import handlers_canvas, handlers_device
from studio.execution_orchestrator import ExecutionOrchestrator
from studio.core.models import CanvasTransform
from studio.device.quality_gate import QualityGateResult


class TestUITimingAndLatency(unittest.TestCase):
    """Garante que a política centralizada de timing é estritamente adaptativa e livre de sobreposições."""

    def test_mock_environment_has_zero_sleep(self):
        """No ambiente de testes ou emulador virtual (FakeADB), o settle deve ser estritamente 0.0s."""
        fake_adb = FakeADB()
        timing = get_ui_timing(fake_adb)
        self.assertTrue(timing.is_mock)

        t0 = time.time()
        for action in ["tool_select", "thickness_slider", "color_palette_tap", "color_palette_swipe", "undo_action", "layer_post_draw"]:
            delay = timing.settle(action, fake_adb)
            self.assertEqual(delay, 0.0)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.05, "Modo mock deve executar todas as ações em menos de 50ms (zero latência artificial).")

    def test_hardware_delays_are_calibrated_and_bounded(self):
        """No modo de hardware real, os delays devem ser mínimos e estritamente controlados pelo UITimingManager."""
        timing = UITimingManager(is_mock=False, timing_scale=1.0)
        self.assertFalse(timing.is_mock)

        # Verifica limites calibrados (entre 0.05s e 0.50s; a paginação da
        # paleta precisa de ~0.45s medidos no aparelho para não perder toques)
        for action, delay in timing.delays.items():
            self.assertGreaterEqual(delay, 0.05)
            self.assertLessEqual(delay, 0.50)

        # Com timing_scale=0, deve zerar os atrasos
        timing_fast = UITimingManager(is_mock=False, timing_scale=0.0)
        d = timing_fast.settle("tool_select")
        self.assertEqual(d, 0.0)

    def test_mcp_handlers_do_not_duplicate_sleep(self):
        """
        Garante que os handlers MCP (select_tool, set_thickness, select_color, action_undo)
        não adicionam sleep arbitrário sobre o FakeADB.
        """
        fake_adb = FakeADB()
        handlers_device.set_adb(fake_adb)

        t0 = time.time()
        res_tool = handlers_canvas.handle_select_tool({"pen_type": "marker"})
        res_thick = handlers_canvas.handle_set_thickness({"mode": "min"})
        res_color = handlers_canvas.handle_select_color({"color_name": "azul"})
        res_undo = handlers_canvas.handle_action_undo({"count": 2})
        elapsed = time.time() - t0

        self.assertTrue(res_tool["success"])
        self.assertTrue(res_thick["success"])
        self.assertTrue(res_color["success"])
        self.assertTrue(res_undo["success"])

        # Anteriormente, essas 4 chamadas somavam > 750 ms em sleeps fixos.
        # Agora com FakeADB, devem rodar em menos de 100 ms!
        self.assertLess(
            elapsed, 0.15,
            f"Handlers MCP gastaram {elapsed*1000:.1f}ms; esperava-se execução quase instantânea (<150ms) sem sleeps duplicados."
        )

    def test_orchestrator_uses_centralized_timing(self):
        """Garante que o ExecutionOrchestrator adota o UITimingManager do adb_manager."""
        fake_adb = FakeADB()
        orchestrator = ExecutionOrchestrator(fake_adb)
        self.assertTrue(orchestrator.timing.is_mock)

        dummy_plan = {
            "quality_gate": QualityGateResult(passed=True, summary="Plano de teste aprovado"),
            "version": "2.0",
            "canvas_dimensions": {"width": 1080, "height": 2400},
            "total_strokes": 2,
            "estimated_time_s": 5.0,
            "transform": CanvasTransform(offset_x=0, offset_y=0, scale=1.0),
            "layers": [
                {
                    "name": "Camada Teste",
                    "brush": "solid",
                    "thickness": "min",
                    "touch_info": {"target_x": 411, "target_y": 2185, "name": "Branco"},
                    "color_rgb": (255, 255, 255),
                    # Dentro da zona segura: y=200 cairia sobre a barra de ferramentas.
                    "strokes": [[(300, 800), (320, 820), (350, 850)]]
                }
            ]
        }

        t0 = time.time()
        res = orchestrator.execute_layer(dummy_plan, layer_index=0, verify_fidelity=False)
        elapsed = time.time() - t0

        self.assertTrue(res.get("success", False), f"Execução da camada falhou: {res}")
        self.assertLess(
            elapsed, 0.20,
            f"Orquestrador gastou {elapsed*1000:.1f}ms na camada; deve rodar sem os sleeps duplicados (<200ms)."
        )


if __name__ == "__main__":
    unittest.main()
