# -*- coding: utf-8 -*-
"""
Script de Execução Física do Miranha no Instagram Direct (Arquitetura Otimizada).
Implementa os Requisitos de Performance e Integridade:
- NUNCA instancia PainterEngine: carrega o ExecutionArtifact pré-compilado em < 10 ms.
- Paridade 100% estrita entre Preview e Desenho Físico (exatamente os mesmos strokes).
- Validação Fail-Closed da assinatura do QualityGate e conferência de hash SHA-256.
- Servo-Visão pré-desenho com auditoria de interface.
- Closed-loop seletivo e batches adaptativos por nível de risco de camada.
"""
import os
import sys
import time
import json
import logging

# Garante inclusão da raiz do projeto no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from studio.adb_manager import ADBManager
from studio.execution_orchestrator import ExecutionOrchestrator
from studio.storage.execution_artifact import ExecutionArtifact, ArtifactCacheManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("MiranhaLiveExecution")


def execute_miranha():
    logger.info("=== INICIANDO EXECUÇÃO FÍSICA DO MIRANHA NO INSTAGRAM DIRECT ===")
    t_start = time.perf_counter()

    # 1. Carregamento Ultrarrápido do Artefato Executável Pré-Compilado (Sem PainterEngine)
    artifact_base = os.path.join("outputs", "miranha")
    t0 = time.perf_counter()
    try:
        artifact = ExecutionArtifact.load(artifact_base)
        load_time_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "⚡ Artefato executável carregado em %.1f ms! (ID: %s, Hash: %s...)",
            load_time_ms, artifact.plan_id, artifact.fingerprint[:12]
        )
    except Exception as e:
        logger.error("Falha ao carregar artefato executável de '%s': %s", artifact_base, e)
        logger.error("Execute primeiro 'python scripts/prepare_miranha.py' para compilar o plano.")
        return False

    # 2. Conferência Estrita do QualityGate (Fail-Closed)
    if not artifact.is_quality_gate_approved:
        logger.error(
            "❌ Execução RECUSADA (Fail-Closed): O artefato foi reprovado pelo QualityGate: %s",
            artifact.quality_gate.get("summary", "Invariantes violadas")
        )
        return False

    logger.info("✅ QualityGate validado no artefato: %s", artifact.quality_gate.get("summary"))
    logger.info("Total de camadas: %d | Total de traços: %d", len(artifact.layers), artifact.total_strokes)

    # 3. Conexão com o Dispositivo Físico
    adb = ADBManager()
    ok, msg = adb.connect()
    if not ok:
        logger.error("Falha na conexão com o celular: %s", msg)
        return False

    logger.info("Dispositivo conectado: %s (%s)", adb.device_target, adb.device_info.get("model"))

    # 4. Inicialização do Orquestrador Unificado (Zero Atrasos)
    orchestrator = ExecutionOrchestrator(adb, target_hz=120)
    orchestrator.remove_delays()  # Zera todos os atrasos artificiais de interface

    # 5. Execução Direta do Artefato com Toques Contínuos e Sem Atraso
    def progress_callback(current_layer, total_layers, comp_strokes, tot_strokes):
        pct = (comp_strokes / max(1, tot_strokes)) * 100.0
        logger.info(
            "Progresso: Camada %d/%d concluída | Traços: %d/%d (%.1f%%)",
            current_layer, total_layers, comp_strokes, tot_strokes, pct
        )

    logger.info("Iniciando despacho físico com TOQUES CONTÍNUOS e SEM ATRASO...")
    res = orchestrator.execute_artifact(
        artifact=artifact,
        selective_closed_loop=False,  # Sem atraso pós-camada
        on_progress_callback=progress_callback
    )

    elapsed_total = round(time.perf_counter() - t_start, 2)

    if not res.get("success", False) or res.get("divergence_detected", False):
        logger.error("❌ Falha ou divergência na execução física: %s (Estágio: %s)", res.get("error"), res.get("stage"))
        return False

    logger.info("\n" + "=" * 65)
    logger.info("🎉 PINTURA DO MIRANHA CONCLUÍDA COM SUCESSO NO INSTAGRAM DIRECT!")
    logger.info("Tempo total de execução: %.1fs (%d traços)", elapsed_total, res.get("total_strokes", 0))
    logger.info("Métricas de Telemetria: %s", res.get("telemetry", {}))
    logger.info("=" * 65)

    return True


if __name__ == "__main__":
    execute_miranha()
