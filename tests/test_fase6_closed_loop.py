"""
Testes unitários para a Fase 6 — Fechar o loop real.
Cobre:
- RegistrationResult com correlação, reprojection_error, anchor_residual e fail-closed
- Unpacking retrocompatível (aligned_img, success)
- ServoVisionDetector com prior de ui_geometry.json
- ClosedLoopCorrectionManager com 3 estágios de correção e checkpoints semânticos
"""
import unittest
import numpy as np
import cv2
from studio.feedback.registration import CanvasRegistrationEngine, RegistrationResult
from studio.device.ui_state_machine import (
    UIStateMachine,
    ServoVisionDetector,
    ClosedLoopCorrectionManager,
    CorrectionStage,
    UIValidationResult,
)


class TestFase6ClosedLoop(unittest.TestCase):
    def setUp(self):
        # Cria uma imagem-alvo sintética com textura para registro óptico
        self.target = np.ones((400, 300, 3), dtype=np.uint8) * 240
        cv2.circle(self.target, (150, 200), 50, (0, 0, 255), -1)
        cv2.rectangle(self.target, (80, 80), (220, 130), (0, 200, 0), -1)
        cv2.line(self.target, (50, 350), (250, 350), (255, 0, 0), 4)

    def test_registration_result_dataclass_and_unpacking(self):
        """Valida que RegistrationResult possui todos os campos e permite unpacking retrocompatível."""
        res = RegistrationResult(
            method="ecc",
            confidence=0.88,
            correlation=0.92,
            reprojection_error=1.5,
            anchor_residual=1.2,
            registered_image=self.target,
            success=True,
        )
        self.assertTrue(res.is_reliable(0.65))
        self.assertEqual(res.method, "ecc")
        self.assertAlmostEqual(res.reprojection_error, 1.5)

        # Unpacking como tupla (aligned, ok)
        aligned, ok = res
        self.assertTrue(ok)
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.shape, self.target.shape)

    def test_registration_fail_closed_on_unaligned_geometry(self):
        """Valida que o fallback geométrico direto não retorna success=True (Fail-Closed)."""
        # Cria um screenshot completamente preto / sem textura coincidente
        black_screen = np.zeros((1920, 1080, 3), dtype=np.uint8)
        res = CanvasRegistrationEngine.align_screenshot_to_target(black_screen, self.target)
        
        # Deve falhar fechado com success=False e confiança abaixo de 0.65
        self.assertFalse(res.success)
        self.assertFalse(res.is_reliable(0.65))
        self.assertLess(res.confidence, 0.65)
        self.assertEqual(res.method, "geometric_prior_unconfirmed")

    def test_registration_ecc_convergence(self):
        """Valida convergência do ECC com pequeno deslocamento sub-pixel/pixel."""
        # Cria um screenshot grande contendo o target deslocado em uma bbox conhecida
        screen = np.zeros((1000, 800, 3), dtype=np.uint8)
        # Bbox: sx=100, sy=200, sw=300, sh=400
        # Coloca o target com pequeno offset de 3 pixels
        shifted_target = np.roll(self.target, shift=(3, 2), axis=(0, 1))
        screen[200:600, 100:400] = shifted_target

        res = CanvasRegistrationEngine.align_screenshot_to_target(
            screen, self.target, canvas_bbox=(100, 200, 300, 400)
        )
        self.assertTrue(res.success)
        self.assertTrue(res.is_reliable(0.60))
        self.assertEqual(res.method, "ecc")
        self.assertGreater(res.correlation, 0.60)

    def test_servo_vision_detector_validation(self):
        """Valida detecção fail-closed de tela do Instagram por Servo-Visão."""
        detector = ServoVisionDetector({
            "safe_bounds": {"left": 80, "right": 1000, "top": 380, "bottom": 1980},
            "ui_anchors": {
                "pen_solid": {"x": 287, "y": 318},
                "pen_marker": {"x": 539, "y": 318},
            }
        })

        # Caso 1: Imagem inválida ou dimensões incompatíveis (landscape)
        landscape_img = np.ones((600, 800, 3), dtype=np.uint8) * 128
        val_landscape = detector.detect_and_validate(landscape_img)
        self.assertFalse(val_landscape.is_valid)
        self.assertFalse(val_landscape.dimensions_valid)

        # Caso 2: Simulação de tela do Stories Canvas (1080x1920)
        sim_screen = np.ones((1920, 1080, 3), dtype=np.uint8) * 30
        # Toolbar superior com ícones e gradiente
        cv2.circle(sim_screen, (287, 318), 25, (255, 255, 255), -1)
        cv2.circle(sim_screen, (539, 318), 25, (200, 200, 255), -1)
        # Slider na lateral esquerda
        cv2.line(sim_screen, (39, 800), (39, 1400), (255, 255, 255), 4)
        # Paleta na barra inferior
        for x_c in range(200, 900, 80):
            cv2.circle(sim_screen, (x_c, 1800), 20, (0, 255, 0), -1)

        val_sim = detector.detect_and_validate(sim_screen)
        self.assertTrue(val_sim.dimensions_valid)
        self.assertTrue(val_sim.safe_bounds_valid)
        self.assertTrue(val_sim.canvas_present)
        self.assertTrue(val_sim.toolbar_present)
        self.assertTrue(val_sim.palette_present)
        self.assertTrue(val_sim.is_valid)

    def test_closed_loop_correction_stages_and_checkpoints(self):
        """Valida ciclo estrito de 3 estágios de correção e checkpoints semânticos verificados."""
        mgr = ClosedLoopCorrectionManager(max_undo_retries=2)

        # Sem divergência -> sem ação corretiva
        stage, msg = mgr.decide_next_action(divergence_detected=False, divergence_reason="")
        self.assertEqual(stage, CorrectionStage.NONE)

        # Divergência detectada 1x -> Estágio 1 (Pausar e Abortar)
        stage1, msg1 = mgr.decide_next_action(divergence_detected=True, divergence_reason="DeltaE alto")
        self.assertEqual(stage1, CorrectionStage.STAGE_1_ABORT_PAUSE)
        self.assertIn("Estágio 1", msg1)

        # Divergência mantida -> Estágio 2 (Undo tentativa 1)
        stage2, msg2 = mgr.decide_next_action(divergence_detected=True, divergence_reason="DeltaE alto")
        self.assertEqual(stage2, CorrectionStage.STAGE_2_UNDO_RETRY)
        self.assertIn("Estágio 2", msg2)

        # Divergência mantida -> Estágio 2 (Undo tentativa 2)
        stage2_b, _ = mgr.decide_next_action(divergence_detected=True, divergence_reason="DeltaE alto")
        self.assertEqual(stage2_b, CorrectionStage.STAGE_2_UNDO_RETRY)

        # Divergência mantida após limite de undos -> Estágio 3 (Corrective Strokes)
        stage3, msg3 = mgr.decide_next_action(divergence_detected=True, divergence_reason="DeltaE alto")
        self.assertEqual(stage3, CorrectionStage.STAGE_3_CORRECTIVE_STROKES)
        self.assertIn("Estágio 3", msg3)

        # Checkpoint semântico: após aprovação do estado real, reseta estágio e grava estado verificado
        mgr.commit_checkpoint(layer_idx=1, stroke_idx=25, metrics={"ciede2000": 4.2})
        self.assertEqual(len(mgr.committed_checkpoints), 1)
        self.assertTrue(mgr.committed_checkpoints[0]["verified"])
        self.assertEqual(mgr.current_stage, CorrectionStage.NONE)


if __name__ == "__main__":
    unittest.main()
