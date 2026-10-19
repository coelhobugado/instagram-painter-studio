# -*- coding: utf-8 -*-
"""
Pipeline de Alta Eficiência e Caching Persistente para 'personagem.jpg'.
Implementa os 20 Princípios de Engenharia da Auditoria de Maturidade:
- Cache persistente por hash SHA-256 (lookup instantâneo).
- QualityGate estritamente fail-closed (default False).
- Compilação adaptativa de movimento por nível de risco de camada.
- Perfilamento de tempo de cada estágio.
- Exportação de Artefato Executável Completo (.npz + .json) para execução sem recálculo.
"""
import os
import sys
import time
import json
import cv2
import numpy as np

# Garante inclusão da raiz do projeto no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from studio.painter_engine import PainterEngine
from studio.planning.scheduler_v2 import StrokeSchedulerV2
from studio.storage.execution_artifact import (
    ExecutionArtifact,
    ArtifactCacheManager,
    compute_plan_fingerprint,
    LayerRiskLevel
)


def prepare_miranha(force_recompute: bool = False) -> ExecutionArtifact:
    t_start_total = time.perf_counter()
    profile_metrics = {}

    img_path = os.path.join("inputs", "personagem.jpg")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    # 1. Arquivamento nos inputs do projeto
    t0 = time.perf_counter()
    dest_input = os.path.join("inputs", "personagem.jpg")
    os.makedirs("inputs", exist_ok=True)
    if not os.path.exists(dest_input):
        with open(img_path, "rb") as f_in, open(dest_input, "wb") as f_out:
            f_out.write(f_in.read())
    profile_metrics["load_and_archive_s"] = time.perf_counter() - t0

    # 2. Configuração do Plano e Cálculo do Hash SHA-256
    config_dict = {
        "mode": "layered_paint",
        "num_colors": 5,
        "time_budget_s": 1800.0,
        "device_model": "SM-A736B",
        "engine_version": "3.1-enterprise"
    }

    profile_dir = os.path.join("profiles", config_dict["device_model"])
    with open(os.path.join(profile_dir, "ui_geometry.json"), encoding="utf-8") as profile_file:
        device_profile = json.load(profile_file)
    with open(os.path.join(profile_dir, "color_lut.json"), encoding="utf-8") as lut_file:
        lut_data = json.load(lut_file)
    fingerprint = compute_plan_fingerprint(
        img_path, config_dict, device_profile=device_profile, lut_data=lut_data
    )
    print(f"[*] Hash Criptográfico SHA-256 da Imagem/Config: {fingerprint[:16]}...")

    # 3. Cache Lookup Persistente (Instantâneo)
    if not force_recompute:
        t0 = time.perf_counter()
        cached = ArtifactCacheManager.lookup(fingerprint)
        profile_metrics["cache_lookup_s"] = time.perf_counter() - t0
        if cached is not None:
            print(f"[CACHE HIT] Artefato executavel carregado em {profile_metrics['cache_lookup_s']*1000:.1f}ms!")
            print(f"    - ID: {cached.plan_id}")
            print(f"    - Camadas: {len(cached.layers)} | Tracos: {cached.total_strokes}")
            print(f"    - QualityGate: {'APROVADO' if cached.is_quality_gate_approved else 'REPROVADO'}")
            return cached

    print("[* CACHE MISS] Processando geometria e segmentação com PainterEngine...")

    # 4. Geração do Plano Geométrico
    t0 = time.perf_counter()
    engine = PainterEngine()
    plan = engine.generate_layered_paint_plan(
        img_path,
        num_colors=config_dict["num_colors"],
        time_budget_s=config_dict["time_budget_s"]
    )
    profile_metrics["geometric_planning_s"] = time.perf_counter() - t0

    layers = plan.get("layers", [])

    # Encadeamento de Toques Contínuos (Continuous Touch Chaining)
    total_strokes_count = 0
    for idx, l in enumerate(layers):
        l["index"] = idx
        orig_strokes = l.get("strokes", [])
        chained = StrokeSchedulerV2.chain_continuous_strokes(orig_strokes, max_gap=35.0)
        l["strokes"] = chained
        l["stroke_count"] = len(chained)
        total_strokes_count += len(chained)

        purpose = l.get("purpose", "fill")
        thickness = l.get("thickness", "min")
        if purpose in ("contour", "line_art_centerline", "colored_detail") or thickness == "min":
            l["risk_level"] = LayerRiskLevel.HIGH.value
        elif purpose in ("color_mass", "fill", "background"):
            l["risk_level"] = LayerRiskLevel.LOW.value
        else:
            l["risk_level"] = LayerRiskLevel.MEDIUM.value

    plan["total_strokes"] = total_strokes_count

    # 5. Reconstrói prévia, hash, movimentos e gate a partir dos traços finais.
    t0 = time.perf_counter()
    plan["quality_gate"] = None
    plan = engine._refresh_edited_plan(plan)
    layers = plan["layers"]
    plan_id = plan["plan_hash"]
    transform = plan["transform"]
    motion_plan = plan["compiled_motion_plan"]
    qg = plan["quality_gate"]
    profile_metrics["rebuild_and_quality_gate_s"] = time.perf_counter() - t0

    # REGRA FAIL-CLOSED: Default deve ser False
    qg_passed = bool(getattr(qg, "passed", False))
    qg_dict = {
        "passed": qg_passed,
        "summary": str(getattr(qg, "summary", "Reprovado por default")),
        "failed_checks": list(getattr(qg, "failed_checks", [])),
        "check_details": dict(getattr(qg, "check_details", {}))
    }

    # 7. Pré-Visualização e Simulação Incremental
    t0 = time.perf_counter()
    out_dir = os.path.join("outputs", "miranha_previews")
    os.makedirs(out_dir, exist_ok=True)
    final_sim = plan.get("simulation_image")
    if final_sim is not None:
        cv2.imwrite(os.path.join(out_dir, "miranha_final_simulation.png"), final_sim)
        cv2.imwrite(os.path.join("outputs", "miranha_final_simulation.png"), final_sim)
    profile_metrics["simulation_preview_s"] = time.perf_counter() - t0

    # 8. Montagem e Serialização do Artefato Executável Completo
    t0 = time.perf_counter()
    profile_metrics["total_preparation_time_s"] = time.perf_counter() - t_start_total

    artifact = ExecutionArtifact(
        plan_id=plan_id,
        fingerprint=fingerprint,
        mode=plan.get("mode", "layered_paint"),
        image_class=plan.get("image_class", "general"),
        transform=transform,
        layers=layers,
        quality_gate=qg_dict,
        profile_metrics=profile_metrics,
        source_image=img_path,
        estimated_time_s=plan.get("estimated_time_s", 0.0),
        total_strokes=plan.get("total_strokes", 0)
    )

    # Salva no cache persistente e nas saídas do projeto
    ArtifactCacheManager.put(artifact)
    manifest_file, npz_file = artifact.save(os.path.join("outputs", "miranha"))
    profile_metrics["serialization_s"] = time.perf_counter() - t0

    print("\n" + "=" * 65)
    print("ARTEFATO EXECUTÁVEL DO MIRANHA PRONTO E PERSISTIDO!")
    print(f"Manifesto JSON: {os.path.abspath(manifest_file)}")
    print(f"Plano Binário (.npz): {os.path.abspath(npz_file)}")
    print(f"QualityGate: {'APROVADO' if qg_passed else 'REPROVADO'} ({qg_dict['summary']})")
    print(f"Total de Camadas: {len(layers)} | Traços: {artifact.total_strokes}")
    print(f"Lotes Adaptativos: {len(motion_plan.batches)}")
    print("Perfil de Tempo de Preparação:")
    for k, v in profile_metrics.items():
        print(f"  - {k}: {v*1000:.1f} ms" if v < 1.0 else f"  - {k}: {v:.2f} s")
    print("=" * 65)

    return artifact


if __name__ == "__main__":
    prepare_miranha(force_recompute=True)
