"""
Testes de Integridade do Plano e do Resultado (P1 — P1.1 a P1.8).
Cobre:
- P1.1: Reavaliação obrigatória do QualityGate após encadeamento/alteração de traços.
- P1.2: Hash de integridade criptográfica contra adulteração e cobertura de perfil/LUT.
- P1.3: Carregamento binário seguro com allow_pickle=False e rejeição de manifesto JSON isolado sem traços.
- P1.4: Sanitização detalhada, bloqueio de sucesso silencioso em lote vazio e validação de dispositivo no checkpoint.
- P1.5: Metadados explícitos de canal de hardware e despacho no CompiledMotionPlan.
- P1.6: Estimativa de tempo total realista no QualityGate incluindo ciclo ADB completo.
- P1.7: Bloqueio de exportação e salvamento de plano dirty/desatualizado na GUI.
- P1.8: Bloqueio de alteração de imagem, modo ou área durante a execução da pintura.
"""
import os
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

from studio.storage.execution_artifact import ExecutionArtifact, compute_plan_fingerprint
from studio.core.models import CanvasTransform
from studio.planning.path_sanitizer import PathSanitizer, SanitizerMode
from studio.adb_manager import ADBManager
from studio.device.quality_gate import QualityGate
from studio.planning.motion_compiler import MotionCompiler
from studio.planning.scheduler_v2 import StrokeSchedulerV2


class TestP1Integrity(unittest.TestCase):

    # -------------------------------------------------------------------------
    # P1.1: Invalidação e reavaliação de QualityGate pós-encadeamento
    # -------------------------------------------------------------------------
    def test_p1_1_quality_gate_revaluated_after_stroke_chaining(self):
        # Cria conjunto de traços adjacentes
        strokes = [
            [(100, 100), (120, 100)],
            [(121, 100), (150, 100)],
            [(151, 100), (200, 100)]
        ]
        chained = StrokeSchedulerV2.chain_continuous_strokes(strokes, max_gap=5.0)
        self.assertLess(len(chained), len(strokes))

        # Simula camada após encadeamento
        layers = [{
            "name": "Linhas", "strokes": chained, "thickness": "min",
            "brush": "solid", "color_rgb": (0, 0, 0)
        }]
        dev_profile = {
            "device_model": "SM-A736B",
            "safe_bounds": {"left": 80, "top": 350, "right": 1000, "bottom": 1950}
        }
        motion_plan = MotionCompiler.compile_plan(layers, target_hz=120)
        qg = QualityGate.evaluate(
            geometric_layers=layers,
            motion_plan=motion_plan,
            ui_validation_res=None,
            device_profile=dev_profile,
            time_budget_s=1800.0,
            preview_delta_e=12.0,
            preview_edge_f1=0.88,
            require_live_ui=False
        )
        self.assertTrue(qg.passed)
        self.assertEqual(qg.metrics_snapshot["total_strokes"], len(chained))

    # -------------------------------------------------------------------------
    # P1.2: Hash de integridade criptográfica e cobertura de perfil/LUT
    # -------------------------------------------------------------------------
    def test_p1_2_fingerprint_includes_device_profile_and_lut(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake_image_bytes")
            tmp_img = f.name
        try:
            cfg = {"mode": "layered_paint", "num_colors": 5}
            fp1 = compute_plan_fingerprint(tmp_img, cfg)

            # Alterando perfil deve alterar fingerprint
            fp2 = compute_plan_fingerprint(tmp_img, cfg, device_profile={"safe_bounds": {"left": 10}})
            self.assertNotEqual(fp1, fp2)

            # Alterando LUT deve alterar fingerprint
            fp3 = compute_plan_fingerprint(tmp_img, cfg, lut_data={"black": (38, 38, 38)})
            self.assertNotEqual(fp1, fp3)
        finally:
            if os.path.exists(tmp_img):
                os.remove(tmp_img)

    def test_p1_2_content_tampering_detected_on_load(self):
        tf = CanvasTransform(0, 0, 1.0)
        art = ExecutionArtifact(
            "plan_test", "fp_123", "layered_paint", "general", tf,
            [{"name": "Camada 1", "strokes": [[(100, 100), (200, 200)]]}],
            {"passed": True}
        )
        with tempfile.TemporaryDirectory() as td:
            base_p = os.path.join(td, "test_art")
            mf, nf = art.save(base_p)

            # Carregamento legítimo funciona
            loaded = ExecutionArtifact.load(base_p)
            self.assertEqual(loaded.plan_id, "plan_test")

            # Adultera o NPZ em disco
            raw = dict(np.load(nf, allow_pickle=False))
            raw["plan_id"] = np.array("tampered_id")
            np.savez_compressed(nf, **raw)

            # Deve rejeitar categoricamente por violação de integridade
            with self.assertRaises(ValueError) as ctx:
                ExecutionArtifact.load(base_p)
            self.assertIn("Falha de integridade", str(ctx.exception))

    # -------------------------------------------------------------------------
    # P1.3: allow_pickle=False e rejeição de manifesto JSON sem binário
    # -------------------------------------------------------------------------
    def test_p1_3_manifest_json_without_npz_rejected_as_executable(self):
        tf = CanvasTransform(0, 0, 1.0)
        art = ExecutionArtifact(
            "plan_json_only", "fp_456", "stencil", "general", tf,
            [{"name": "L1", "strokes": [[(10, 10), (20, 20)]]}],
            {"passed": True}
        )
        with tempfile.TemporaryDirectory() as td:
            base_p = os.path.join(td, "json_test")
            mf, nf = art.save(base_p)

            # Remove o NPZ deixando apenas o manifesto JSON
            os.remove(nf)
            self.assertTrue(os.path.exists(mf))

            # Deve rejeitar carregar o JSON isolado como artefato executável
            with self.assertRaises(ValueError) as ctx:
                ExecutionArtifact.load(mf)
            self.assertIn("contém apenas metadados", str(ctx.exception))

    # -------------------------------------------------------------------------
    # P1.4: Sanitização detalhada, bloqueio de lote vazio e verificação de dispositivo
    # -------------------------------------------------------------------------
    def test_p1_4_sanitizer_detailed_reports_rejected_count_and_reasons(self):
        strokes = [
            [(100, 100), (200, 200)],       # Válido dentro dos limites
            [(10, 10)],                     # Menos de 2 pontos (degenerado)
            [(5000, 5000), (6000, 6000)],   # Fora dos limites do canvas
            []                              # Vazio
        ]
        clean, rejected_count, reasons = PathSanitizer.sanitize_strokes_detailed(
            strokes, x_min=50, x_max=500, y_min=50, y_max=500
        )
        self.assertEqual(len(clean), 1)
        self.assertEqual(rejected_count, 3)
        self.assertEqual(len(reasons), 3)

    def test_p1_4_adb_execute_stroke_batch_rejects_empty_strokes(self):
        with patch("studio.adb_manager.subprocess.run"):
            mgr = ADBManager("adb", "OFFLINE")
            # Envio de lista vazia deve retornar False (fim do sucesso silencioso de lote vazio)
            res = mgr.execute_stroke_batch([])
            self.assertFalse(res)

    def test_p1_4_checkpoint_resume_validates_device_id(self):
        from studio.app_gui import InstagramPainterStudioGUI
        # Testa lógica de verificação de dispositivo do checkpoint
        saved_cp = {
            "plan_hash": "hash_xyz",
            "layer_index": 2,
            "stroke_index": 15,
            "device_id": "192.168.1.50:5555",
            "timestamp": "2026-09-23T14:00:00Z"
        }
        gui = Mock()
        gui.saved_checkpoint = saved_cp
        gui.current_plan = {"plan_hash": "hash_xyz", "total_strokes": 50, "layers": [1, 2, 3]}
        gui.adb = Mock()
        gui.adb.device_target = "192.168.1.99:5555" # Dispositivo divergente!
        gui.checkpoint = {"layer": 0, "stroke": 0}
        gui.log = Mock()

        # Invoca o validador real implementado em app_gui com patch nos diálogos modais
        with patch("studio.app_gui.messagebox.showwarning"), patch("studio.app_gui.messagebox.askyesno"):
            InstagramPainterStudioGUI._validate_and_offer_resume(gui)

        # Deve manter checkpoint em (0, 0) e registrar bloqueio por dispositivo divergente
        self.assertEqual(gui.checkpoint, {"layer": 0, "stroke": 0})
        gui.log.assert_called()
        log_msg = gui.log.call_args[0][0]
        self.assertIn("pertence ao dispositivo '192.168.1.50:5555'", log_msg)

    # -------------------------------------------------------------------------
    # P1.5: Metadados explícitos de hardware no CompiledMotionPlan
    # -------------------------------------------------------------------------
    def test_p1_5_compiled_motion_plan_contains_hardware_metadata(self):
        layers = [{
            "name": "Base", "strokes": [[(100, 100), (200, 100)]],
            "thickness": "min", "brush": "solid", "color_rgb": (0, 0, 0)
        }]
        plan = MotionCompiler.compile_plan(layers, target_hz=120)
        self.assertIn("execution_channel", plan.metadata)
        self.assertIn("physical_dispatch_mode", plan.metadata)
        self.assertIn("pure_motion_duration_s", plan.metadata)
        self.assertIn("estimated_adb_runtime_s", plan.metadata)
        self.assertGreater(plan.metadata["estimated_adb_runtime_s"], plan.metadata["pure_motion_duration_s"])

    # -------------------------------------------------------------------------
    # P1.6: Estimativa de tempo total realista no QualityGate
    # -------------------------------------------------------------------------
    def test_p1_6_quality_gate_computes_realistic_full_adb_cycle_time(self):
        layers = [
            {"name": "L1", "strokes": [[(100, 100), (200, 100)]] * 10, "thickness": "min", "brush": "solid", "color_rgb": (0, 0, 0)},
            {"name": "L2", "strokes": [[(100, 100), (200, 100)]] * 10, "thickness": "min", "brush": "solid", "color_rgb": (255, 0, 0)}
        ]
        dev_profile = {
            "device_model": "SM-A736B",
            "safe_bounds": {"left": 80, "top": 350, "right": 1000, "bottom": 1950}
        }
        motion_plan = MotionCompiler.compile_plan(layers, target_hz=120)
        qg = QualityGate.evaluate(
            geometric_layers=layers,
            motion_plan=motion_plan,
            ui_validation_res=None,
            device_profile=dev_profile,
            time_budget_s=600.0,
            preview_delta_e=12.0,
            preview_edge_f1=0.88,
            require_live_ui=False
        )
        self.assertTrue(qg.passed)
        # Verifica se o tempo estimado inclui overhead de ADB e trocas de camada
        total_time = qg.check_details["estimated_total_adb_time_s"]
        pure_motion = qg.check_details["pure_motion_time_s"]
        self.assertGreater(total_time, pure_motion)
        # Com 2 camadas e 20 traços, o overhead deve ser de pelo menos 2.0 segundos
        self.assertGreaterEqual(total_time - pure_motion, 2.0)

    # -------------------------------------------------------------------------
    # P1.7: Bloqueio de exportação/salvamento com plano dirty
    # -------------------------------------------------------------------------
    def test_p1_7_dirty_plan_blocks_saving_and_exporting(self):
        from studio.app_gui import InstagramPainterStudioGUI
        gui = Mock()
        gui.current_plan = {"simulation_image": np.zeros((10, 10, 3), dtype=np.uint8)}
        gui._editing_plan = False
        gui._plan_dirty = True

        with patch("studio.app_gui.messagebox.showwarning") as warn_mock:
            InstagramPainterStudioGUI.action_save_simulation(gui)
            warn_mock.assert_called_once()
            self.assertIn("desatualizado", warn_mock.call_args[0][1])

        with patch("studio.app_gui.messagebox.showwarning") as warn_mock2:
            InstagramPainterStudioGUI.action_save_project(gui)
            warn_mock2.assert_called_once()
            self.assertIn("desatualizado", warn_mock2.call_args[0][1])

    # -------------------------------------------------------------------------
    # P1.8: Bloqueio de alteração de imagem/modo/área durante execução
    # -------------------------------------------------------------------------
    def test_p1_8_controls_locked_while_worker_running(self):
        from studio.app_gui import InstagramPainterStudioGUI
        gui = Mock()
        worker = Mock()
        worker._is_running = True
        worker.is_alive.return_value = True
        gui.worker = worker

        with patch("studio.app_gui.messagebox.showwarning") as warn_mock:
            InstagramPainterStudioGUI.load_image_file(gui, "dummy.png")
            warn_mock.assert_called_once()
            self.assertIn("Não é permitido alterar a imagem", warn_mock.call_args[0][1])

        with patch("studio.app_gui.messagebox.showwarning") as warn_mock2:
            InstagramPainterStudioGUI._on_mode_change(gui, "STENCIL")
            warn_mock2.assert_called_once()
            self.assertIn("Não é permitido alterar o modo", warn_mock2.call_args[0][1])

        with patch("studio.app_gui.messagebox.showwarning") as warn_mock3:
            InstagramPainterStudioGUI._on_roi_mouse_press(gui, Mock())
            warn_mock3.assert_called_once()
            self.assertIn("Não é permitido redefinir a área", warn_mock3.call_args[0][1])


if __name__ == "__main__":
    unittest.main()
