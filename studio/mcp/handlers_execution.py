# -*- coding: utf-8 -*-
"""
Handlers MCP para Execução Física de Traços no Celular e Feedback em Ciclo Fechado.
Totalmente integrados à infraestrutura avançada do Instagram Painter Studio:
MCP -> ExecutionOrchestrator -> QualityGate -> Servo Vision -> CompiledMotionPlan
    -> PersistentTouchExecutor -> Closed-Loop Feedback -> Telemetry -> Device.
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, List

from .handlers_device import get_adb
from .handlers_planning import PLANS_CACHE
from ..execution_orchestrator import ExecutionOrchestrator
from ..feedback.registration import CanvasRegistrationEngine
from ..feedback.error_map import ErrorMapEngine


def handle_execute_freehand_path(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executa traços vetoriais diretos enviados pelo agente no canvas do celular
    através do ExecutionOrchestrator (Servo-Visão, MotionCompiler, PersistentTouch e Telemetria).
    """
    raw_strokes = args.get("strokes", [])
    if not raw_strokes:
        return {"success": False, "error": "Nenhum traço fornecido em 'strokes'."}

    thickness = args.get("thickness", "min")
    color_name = args.get("color_name", "preto")
    batch_size = max(1, int(args.get("batch_size", 8)))

    adb = get_adb()
    orchestrator = ExecutionOrchestrator(adb)

    return orchestrator.execute_freehand_path(
        strokes=raw_strokes,
        thickness=thickness,
        color_name=color_name,
        batch_size=batch_size
    )


def handle_execute_layer(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executa uma camada completa de um plano pré-planejado no celular através do
    ExecutionOrchestrator, garantindo a passagem pelo funil completo:
    QualityGate -> Servo-Visão -> CompiledMotionPlan -> PersistentTouchExecutor -> Closed-Loop -> Telemetria.
    """
    plan_id = args.get("plan_id")
    if not plan_id or plan_id not in PLANS_CACHE:
        return {"success": False, "error": f"Plano com ID '{plan_id}' não encontrado no cache."}

    plan = PLANS_CACHE[plan_id]
    layers = plan.get("layers", [])
    layer_idx = int(args.get("layer_index", 0))

    if layer_idx < 0 or layer_idx >= len(layers):
        return {"success": False, "error": f"Índice de camada {layer_idx} inválido (Total: {len(layers)})."}

    adb = get_adb()
    batch_size = max(1, int(args.get("batch_size", 8)))

    orchestrator = ExecutionOrchestrator(adb)
    return orchestrator.execute_layer(
        plan=plan,
        layer_index=layer_idx,
        batch_size=batch_size,
        verify_fidelity=True,
        fail_closed_qg=True,
        fail_closed_ui=True
    )


def handle_capture_screen_diff(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Captura a tela atual do celular e computa o mapa de calor e métricas de divergência
    (True CIEDE2000 e SSIM) contra a imagem esperada do plano com registro subpixel.
    """
    adb = get_adb()
    screenshot = adb.capture_screenshot()
    if screenshot is None:
        return {"success": False, "error": "Falha ao capturar a tela do celular."}

    out_dir = os.path.join("outputs", "mcp_feedback")
    os.makedirs(out_dir, exist_ok=True)
    live_path = os.path.join(out_dir, "live_screen.png")
    cv2.imwrite(live_path, screenshot)

    plan_id = args.get("plan_id")
    metrics_summary: Dict[str, Any] = {}

    if plan_id and plan_id in PLANS_CACHE:
        plan = PLANS_CACHE[plan_id]
        expected_img = plan.get("simulation_image")
        transform = plan.get("transform")
        if expected_img is not None:
            try:
                # Alinhamento sub-pixel rigoroso via CanvasRegistrationEngine
                aligned_screen, reg_ok = CanvasRegistrationEngine.align_screenshot_to_target(
                    screenshot, expected_img, transform=transform
                )
                heatmap, metrics = ErrorMapEngine.compute_error_map(expected_img, aligned_screen)
                
                heatmap_path = os.path.join(out_dir, "error_heatmap.png")
                cv2.imwrite(heatmap_path, heatmap)

                metrics_summary = {
                    "registered": bool(reg_ok),
                    "mean_delta_e": round(float(metrics.get("mean_delta_e", 0.0)), 2),
                    "p95_delta_e": round(float(metrics.get("p95_delta_e", 0.0)), 2),
                    "ssim": round(float(metrics.get("ssim", 0.0)), 3),
                    "divergence_detected": bool(metrics.get("mean_delta_e", 0.0) > 28.0),
                    "heatmap_path": heatmap_path
                }
            except Exception as e:
                metrics_summary["diff_error"] = str(e)

    return {
        "success": True,
        "screenshot_path": live_path,
        "metrics": metrics_summary
    }
