# -*- coding: utf-8 -*-
"""
Testes Unitários da FASE 3: Executor Touch Persistente e Métricas Físicas.
Em conformidade com o INSTAGRAM_PAINTER_ROADMAP_APICE.md:
- 3.1: Protocolo de eventos (BEGIN, DOWN, MOVE, UP, END).
- 3.2: Frequência configurável (60, 90, 120, 240 Hz).
- 3.3: Fallback gracioso para ADB/FakeADB quando persistente inacessível.
- 3.4: Métricas de latência, jitter, events/s e completion rate.
"""
import unittest
from studio.device.persistent_touch import PersistentTouchExecutor
from studio.planning.motion_compiler import MotionStroke, MotionEvent, TouchAction
from studio.device.fake_adb import FakeADB

class TestFase3PersistentTouch(unittest.TestCase):

    def setUp(self):
        self.fake_adb = FakeADB()
        # Inicializa executor com mock ativo para simulação de daemon
        self.executor = PersistentTouchExecutor(
            fallback_adb=self.fake_adb,
            target_frequency_hz=120,
            enable_mock=True
        )
        self.executor.connect()

    def test_3_1_protocol_execution_and_metrics(self):
        """3.1 & 3.4: Valida despacho de MotionStroke e coleta de métricas."""
        events = [
            MotionEvent(TouchAction.DOWN, 100, 200, 0, 1.0),
            MotionEvent(TouchAction.MOVE, 105, 205, 8, 1.0),
            MotionEvent(TouchAction.MOVE, 110, 210, 16, 1.0),
            MotionEvent(TouchAction.UP, 115, 215, 24, 0.0),
        ]
        stroke = MotionStroke(events=events, tool="solid", thickness_mode="min")
        ok = self.executor.execute_motion_stroke(stroke)
        self.assertTrue(ok)

        metrics = self.executor.get_metrics()
        self.assertEqual(metrics["events_sent"], 4)
        self.assertEqual(metrics["events_acked"], 4)
        self.assertEqual(metrics["completion_rate_pct"], 100.0)
        self.assertGreaterEqual(metrics["mean_latency_ms"], 0.0)

    def test_3_2_frequency_pacing_configuration(self):
        """3.2: Valida configuração de frequências de digitalização."""
        for hz in [60, 90, 120, 240]:
            exec_hz = PersistentTouchExecutor(target_frequency_hz=hz, enable_mock=True)
            self.assertEqual(exec_hz.target_frequency_hz, hz)

    def test_3_3_transparent_fallback_to_adb(self):
        """3.3: Garante fallback automático para FakeADB se conexão persistente estiver desconectada."""
        unconnected_executor = PersistentTouchExecutor(
            fallback_adb=self.fake_adb,
            enable_mock=False
        )
        # Não conecta -> is_connected = False
        self.assertFalse(unconnected_executor.is_connected)

        events = [
            MotionEvent(TouchAction.DOWN, 50, 50, 0),
            MotionEvent(TouchAction.MOVE, 60, 60, 8),
            MotionEvent(TouchAction.UP, 70, 70, 16),
        ]
        stroke = MotionStroke(events=events)
        ok = unconnected_executor.execute_motion_stroke(stroke)

        self.assertTrue(ok)
        metrics = unconnected_executor.get_metrics()
        self.assertEqual(metrics["fallback_invocations"], 1)
        # FakeADB deve ter recebido a pincelada
        self.assertGreater(len(self.fake_adb.injected_strokes), 0)

if __name__ == "__main__":
    unittest.main()
