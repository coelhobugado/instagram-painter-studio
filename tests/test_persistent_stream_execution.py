# -*- coding: utf-8 -*-
"""
Testes Unitários do Stream Contínuo Persistente (Persistent Shell Stream & Persistent Touch).
Valida a eliminação de subprocessos por lote, push, chmod e sleeps artificiais.
"""
import unittest
from unittest.mock import MagicMock, patch
import time
import subprocess
import io

from studio.device.persistent_touch import (
    PersistentTouchExecutor,
    TouchChannelType,
    PersistentTouchMetrics,
)
from studio.planning.motion_compiler import (
    MotionStroke,
    MotionEvent,
    TouchAction,
    ExecutionBatch,
)
from studio.adb_manager import ADBManager


class TestPersistentStreamExecution(unittest.TestCase):

    def test_persistent_touch_channel_hierarchy_and_stream(self):
        """Valida que o executor abre stream interativo único sem disparar subprocessos por lote."""
        mock_adb = MagicMock()
        mock_adb.adb_path = "adb"
        mock_adb.device_target = "TEST_DEVICE_SERIAL"
        mock_adb.is_connected = True

        executor = PersistentTouchExecutor(
            host="127.0.0.1",
            port=19999,  # Porta fechada para forçar o fallback para o Persistent Shell Stream
            fallback_adb=mock_adb,
            target_frequency_hz=120,
            enable_mock=False
        )

        # Mock do subprocess.Popen para simular a resposta imediata do stream shell do Android
        fake_stdout = io.StringIO("__PERSISTENT_SHELL_READY__\n__STREAM_ACK_1__\n__STREAM_ACK_2__\n")
        fake_stdin = io.StringIO()
        fake_proc = MagicMock()
        fake_proc.stdout = fake_stdout
        fake_proc.stdin = fake_stdin

        with patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
            ok, msg = executor.connect()

            self.assertTrue(ok)
            self.assertEqual(executor.channel_type, TouchChannelType.PERSISTENT_SHELL)
            self.assertIn("Shell Stream ADB", msg)
            mock_popen.assert_called_once()  # UM ÚNICO subprocesso iniciado

            # Cria um lote com múltiplos eventos de toque compilados
            events = [
                MotionEvent(TouchAction.DOWN, 100, 200, 0, 1.0),
                MotionEvent(TouchAction.MOVE, 120, 220, 8, 1.0),
                MotionEvent(TouchAction.UP, 140, 240, 16, 0.0)
            ]
            stroke = MotionStroke(events=events, tool="solid", thickness_mode="min", color_rgb=(255, 0, 0))
            batch = ExecutionBatch(batch_id=1, strokes=[stroke], estimated_duration_ms=16)

            # Executa o lote
            exec_ok = executor.execute_batch(batch)
            self.assertTrue(exec_ok)

            # Valida que NÃO houve chamada de adb.run_cmd nem subprocesso extra
            mock_adb.run_cmd.assert_not_called()
            mock_adb.execute_stroke_batch.assert_not_called()

            # Valida que os comandos foram injetados diretamente no stream do stdin
            stdin_content = fake_stdin.getvalue()
            self.assertIn("input swipe 100 200 140 240", stdin_content)
            self.assertIn("__STREAM_ACK_1__", stdin_content)

            # Valida que as métricas foram atualizadas sem sleeps arbitrários
            metrics = executor.get_metrics()
            self.assertEqual(metrics["channel_type"], "persistent_shell")
            self.assertEqual(metrics["events_sent"], 3)
            self.assertEqual(metrics["events_acked"], 3)
            self.assertEqual(metrics["fallback_invocations"], 0)

            executor.close()

    def test_adb_manager_interactive_shell_fallback(self):
        """Valida que o ADBManager reutiliza a sessão interativa persistente sem recriar scripts."""
        adb = ADBManager(adb_path="adb", device_target="TEST_DEVICE_SERIAL")

        fake_stdout = io.StringIO("__ADB_SHELL_STREAM_READY__\n__STREAM_BATCH_ACK_1__\n")
        fake_stdin = io.StringIO()
        fake_proc = MagicMock()
        fake_proc.stdout = fake_stdout
        fake_proc.stdin = fake_stdin
        fake_proc.poll.return_value = None

        strokes = [[(50, 50), (100, 100), (150, 150)]]

        with patch("subprocess.Popen", return_value=fake_proc) as mock_popen:
            # Mock de run_cmd para garantir que NÃO faça push nem chmod
            with patch.object(adb, "run_cmd") as mock_run_cmd:
                ok = adb.execute_stroke_batch(strokes)
                self.assertTrue(ok)
                mock_popen.assert_called_once()
                mock_run_cmd.assert_not_called() # ZERO push, ZERO chmod

                stdin_content = fake_stdin.getvalue()
                self.assertIn("input swipe 50 50 150 150", stdin_content)
                self.assertIn("__STREAM_BATCH_ACK_1__", stdin_content)

        adb.close_interactive_shell()


if __name__ == "__main__":
    unittest.main()
