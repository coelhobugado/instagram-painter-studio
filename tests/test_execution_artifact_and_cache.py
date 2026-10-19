# -*- coding: utf-8 -*-
"""
Testes Unitários de Persistência, Caching por Hash SHA-256 e QualityGate Fail-Closed.
Valida os 20 Princípios de Engenharia aplicados ao pipeline de artefatos de pintura.
"""
import unittest
import os
import shutil
import tempfile
import numpy as np
from studio.storage.execution_artifact import (
    ExecutionArtifact,
    ArtifactCacheManager,
    compute_plan_fingerprint,
    LayerRiskLevel
)
from studio.core.models import CanvasTransform
from studio.planning.motion_compiler import MotionCompiler


class TestExecutionArtifactAndCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.prev_cache = ArtifactCacheManager.CACHE_DIR
        ArtifactCacheManager.CACHE_DIR = os.path.join(self.temp_dir, "cache")

    def tearDown(self):
        ArtifactCacheManager.CACHE_DIR = self.prev_cache
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_fingerprint_deterministic(self):
        """O fingerprint SHA-256 deve ser estável e sensível a qualquer alteração de configuração."""
        img_dummy = os.path.join(self.temp_dir, "test.png")
        with open(img_dummy, "wb") as f:
            f.write(b"PNG_FAKE_IMAGE_DATA_12345")

        conf1 = {"mode": "layered_paint", "num_colors": 5}
        conf2 = {"mode": "layered_paint", "num_colors": 6}

        fp1 = compute_plan_fingerprint(img_dummy, conf1)
        fp1_repeat = compute_plan_fingerprint(img_dummy, conf1)
        fp2 = compute_plan_fingerprint(img_dummy, conf2)

        self.assertEqual(fp1, fp1_repeat, "Hash deve ser 100% determinístico para os mesmos parâmetros.")
        self.assertNotEqual(fp1, fp2, "Alteração de configuração deve gerar hash diferente.")

    def test_quality_gate_fail_closed_by_default(self):
        """Se o QualityGate estiver ausente ou não comprovado, o artefato DEVE ser reprovado (fail-closed)."""
        artifact_fail = ExecutionArtifact(
            plan_id="p1", fingerprint="f1", mode="layered_paint", image_class="photo",
            transform=CanvasTransform(0, 0, 1.0), layers=[],
            quality_gate={"passed": False, "summary": "Violação de invariante"}
        )
        self.assertFalse(artifact_fail.is_quality_gate_approved)
        manifest = artifact_fail.to_manifest_dict()
        self.assertFalse(manifest["ready_to_send_instagram"])
        self.assertEqual(manifest["status"], "REJECTED_BY_QUALITY_GATE")

        # Com QualityGate aprovado
        artifact_ok = ExecutionArtifact(
            plan_id="p2", fingerprint="f2", mode="layered_paint", image_class="photo",
            transform=CanvasTransform(0, 0, 1.0), layers=[],
            quality_gate={"passed": True, "summary": "Invariantes cumpridas"}
        )
        self.assertTrue(artifact_ok.is_quality_gate_approved)
        self.assertTrue(artifact_ok.to_manifest_dict()["ready_to_send_instagram"])

    def test_artifact_save_and_fast_load(self):
        """Garante que a serialização binária (.npz) salva e recarrega os dados completos sem perdas."""
        layers = [
            {
                "name": "Camada Contorno",
                "color_rgb": [20, 20, 20],
                "thickness": "min",
                "purpose": "contour",
                "risk_level": LayerRiskLevel.HIGH.value,
                "strokes": [[(10, 20), (30, 40), (50, 60)]]
            }
        ]
        artifact = ExecutionArtifact(
            plan_id="plan_save_test",
            fingerprint="fp_save_test",
            mode="line_art",
            image_class="line_art",
            transform=CanvasTransform(offset_x=100, offset_y=200, scale=1.5),
            layers=layers,
            quality_gate={"passed": True, "summary": "OK"}
        )

        base_path = os.path.join(self.temp_dir, "test_art")
        manifest_f, npz_f = artifact.save(base_path)

        self.assertTrue(os.path.exists(manifest_f))
        self.assertTrue(os.path.exists(npz_f))

        # Recarrega do .npz
        loaded = ExecutionArtifact.load(base_path)
        self.assertEqual(loaded.plan_id, "plan_save_test")
        self.assertEqual(len(loaded.layers), 1)
        self.assertEqual(loaded.layers[0]["strokes"][0], [[10, 20], [30, 40], [50, 60]])
        self.assertEqual(loaded.transform.scale, 1.5)
        self.assertTrue(loaded.is_quality_gate_approved)

    def test_artifact_without_content_hash_is_rejected(self):
        artifact = ExecutionArtifact(
            plan_id="missing_hash", fingerprint="fp", mode="line_art",
            image_class="line_art", transform=CanvasTransform(0, 0),
            layers=[{"strokes": [[(10, 10), (20, 20)]]}],
            quality_gate={"passed": True},
        )
        base_path = os.path.join(self.temp_dir, "unsigned")
        _, npz_file = artifact.save(base_path)
        with np.load(npz_file, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files if key != "content_hash"}
        np.savez_compressed(npz_file, **payload)
        with self.assertRaisesRegex(ValueError, "sem hash"):
            ExecutionArtifact.load(base_path)

    def test_motion_compiler_adaptive_batches(self):
        """Garante que camadas finas recebem lotes menores e preenchimentos recebem lotes maiores."""
        layers = [
            {
                "purpose": "contour",
                "thickness": "min",
                "strokes": [[(i, i), (i + 1, i + 1)] for i in range(12)]
            },
            {
                "purpose": "color_mass",
                "thickness": "thick",
                "strokes": [[(i, i), (i + 1, i + 1)] for i in range(32)]
            }
        ]

        mp = MotionCompiler.compile_plan(layers, adaptive=True)
        # Camada 1 (contour): 12 traços com batch 4 = 3 lotes
        # Camada 2 (color_mass): 32 traços com batch 16 = 2 lotes
        # Total de lotes esperado: 3 + 2 = 5 lotes
        self.assertEqual(len(mp.batches), 5)
        self.assertEqual(len(mp.batches[0].strokes), 4, "Contorno deve ter batch adaptativo = 4.")
        self.assertEqual(len(mp.batches[3].strokes), 16, "Massa de cor deve ter batch adaptativo = 16.")


if __name__ == "__main__":
    unittest.main()
