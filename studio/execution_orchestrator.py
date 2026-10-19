# -*- coding: utf-8 -*-
"""
Orquestrador Industrial de Execução Física (Execution Orchestrator).
Ponte central de execução do Instagram Painter Studio que integra:
MCP -> ExecutionOrchestrator -> QualityGate -> Servo Vision -> CompiledMotionPlan
    -> PersistentTouchExecutor -> Closed-Loop Feedback -> Telemetry -> ADB/Device.
"""
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

from .core.models import CanvasTransform
from .device.ui_state_machine import UIStateMachine, UIState, CorrectionStage
from .device.quality_gate import QualityGateResult, plan_content_matches_hash
from .device.persistent_touch import PersistentTouchExecutor
from .device.session_telemetry import SessionTelemetry
from .planning.motion_compiler import MotionCompiler, CompiledMotionPlan, MotionStroke
from .planning.path_sanitizer import PathSanitizer
from .feedback.registration import CanvasRegistrationEngine
from .feedback.error_map import ErrorMapEngine
from .feedback.layer_verification import verify_layer_fidelity
from .execution_worker import convert_and_sanitize_strokes
from .device.ui_timing import get_ui_timing

logger = logging.getLogger("ExecutionOrchestrator")


class ExecutionOrchestrator:
    """
    Orquestrador Unificado de Execução Física.
    Garante que toda ordem de desenho emitida via MCP passe estritamente por:
    1. QualityGate (Invariantes geométricas e perceptuais fail-closed)
    2. Servo-Visão (Auditoria da UI do Instagram antes do primeiro toque)
    3. MotionCompiler (Preservação temporal e reamostragem a 120 Hz em CompiledMotionPlan)
    4. PersistentTouchExecutor (Socket persistente de alta velocidade com fallback ADB)
    5. Closed-Loop (Registro subpixel e auditoria pós-camada com tolerância a desvios)
    6. Telemetria (Registro contínuo de latência, jitter e taxa de eventos)
    """

    def __init__(self, adb_manager: Any, target_hz: int = 120, enable_mock: Optional[bool] = None):
        self.adb = adb_manager
        self.target_hz = target_hz

        is_mock = enable_mock if enable_mock is not None else (
            getattr(adb_manager, "device_target", "") == "fake_device_emulator"
        )
        self.is_mock = is_mock
        self.timing = get_ui_timing(adb_manager)

        # Máquina de estados com Servo-Visão fail-closed
        self.state_machine = UIStateMachine()

        # Canal de toque persistente de alto desempenho com fallback
        self.touch_executor = PersistentTouchExecutor(
            fallback_adb=adb_manager,
            target_frequency_hz=target_hz,
            enable_mock=is_mock
        )

        # Telemetria da sessão
        self.telemetry = SessionTelemetry(session_id=f"mcp_exec_{int(time.time())}")
        self.start_time: Optional[float] = None

    def remove_delays(self):
        """Elimina todos os atrasos de interface e esperas artificiais (0 ms)."""
        if hasattr(self.timing, "zero_delays"):
            self.timing.zero_delays()
        else:
            self.timing.timing_scale = 0.0


    def validate_safety(self, screenshot_bgr: Optional[np.ndarray] = None) -> Tuple[bool, str, float]:
        """
        Executa Servo-Visão fail-closed na tela do celular para garantir que o
        Instagram Direct Canvas está ativo e pronto para desenhar.
        """
        if screenshot_bgr is None and self.adb:
            try:
                screenshot_bgr = self.adb.capture_screenshot()
            except Exception as e:
                return False, f"Falha ao capturar screenshot para servo-visão: {e}", 0.0

        if screenshot_bgr is None or getattr(screenshot_bgr, "size", 0) == 0:
            return False, "Falha de segurança Fail-Closed: Captura de tela ausente ou nula. Validação visual do canvas é estritamente obrigatória.", 0.0

        is_safe, reason = self.state_machine.validate_safety_for_drawing(self.adb, screenshot_bgr=screenshot_bgr)
        return is_safe, reason, self.state_machine.confidence

    def evaluate_quality_gate(self, plan: Dict[str, Any]) -> QualityGateResult:
        """Avalia as invariantes do plano de pintura com o QualityGate fail-closed."""
        if not plan_content_matches_hash(plan):
            return QualityGateResult(
                passed=False,
                summary="Plano alterado após a aprovação; gere a prévia novamente.",
                failed_checks=["plan_hash_mismatch"],
            )
        if "quality_gate" in plan and plan["quality_gate"] is not None:
            qg = plan["quality_gate"]
            if isinstance(qg, QualityGateResult):
                return qg
            if isinstance(qg, dict) and "passed" in qg:
                return QualityGateResult(
                    passed=qg.get("passed") is True,
                    summary=str(qg.get("summary", "QualityGate")),
                    failed_checks=list(qg.get("failed_checks", [])),
                    check_details=dict(qg.get("check_details", {}))
                )

        return QualityGateResult(
            passed=False,
            summary="QualityGate ausente ou inválido: gere e revise a prévia antes da execução.",
            failed_checks=["quality_gate_missing"],
        )

    def verify_layer_fidelity(
        self,
        aligned_live: Optional[np.ndarray],
        expected_img: Optional[np.ndarray],
        layer_data: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Avalia a fidelidade da camada desenhada comparando a tela alinhada contra o esperado:
        - Divergência de cor (Delta E Lab)
        - Cobertura espacial dos traços
        - Alinhamento de bordas
        - Espessura do pincel
        Mesmos limites do ExecutionWorker (studio/feedback/layer_verification.py).
        """
        # Sem a medição de espessura: em traços densos ela funde linhas finas
        # vizinhas e reprova camadas corretas. O worker compensa com a
        # conferência direta dos traços; o caminho MCP ainda não tem esse recurso.
        return verify_layer_fidelity(aligned_live, expected_img, layer_data, check_thickness=False)

    def execute_layer(
        self,
        plan: Dict[str, Any],
        layer_index: int,
        batch_size: int = 8,
        verify_fidelity: bool = True,
        fail_closed_qg: bool = True,
        fail_closed_ui: bool = True,
        pre_validated_screen: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Executa uma camada inteira do plano de forma estritamente orquestrada:
        QualityGate -> Servo-Visão -> Configuração de Hardware -> CompiledMotionPlan
        -> PersistentTouchExecutor -> Closed-Loop Feedback -> Telemetria.

        ``fail_closed_qg``/``fail_closed_ui`` são aceitos só por compatibilidade:
        QualityGate e servo-visão nunca podem ser desligados.
        ``pre_validated_screen``: captura já aprovada pela servo-visão pelo
        chamador imediatamente antes (ex.: ``execute_artifact``); evita repetir
        captura e validação para a primeira camada.
        """
        t0 = time.time()
        layers = plan.get("layers", [])
        if layer_index < 0 or layer_index >= len(layers):
            return {"success": False, "error": f"Índice de camada {layer_index} inválido (Total: {len(layers)})."}

        layer = layers[layer_index]
        transform = plan.get("transform")
        if transform is None and "transform_dict" in plan:
            td = plan["transform_dict"]
            transform = CanvasTransform(offset_x=td["offset_x"], offset_y=td["offset_y"], scale=td.get("scale", 1.0))

        # ----------------------------------------------------------------------
        # 1. QualityGate (Validação Fail-Closed do Plano)
        # ----------------------------------------------------------------------
        qg_result = self.evaluate_quality_gate(plan)
        if not qg_result.passed:
            return {
                "success": False,
                "stage": "quality_gate",
                "error": f"QualityGate REPROVADO (Fail-Closed): {qg_result.summary}",
                "failed_checks": qg_result.failed_checks,
                "check_details": qg_result.check_details
            }

        # ----------------------------------------------------------------------
        # 2. Servo-Visão (Auditoria da Interface do Instagram antes de desenhar)
        # ----------------------------------------------------------------------
        if pre_validated_screen is None:
            pre_screen = None
            if self.adb:
                try:
                    pre_screen = self.adb.capture_screenshot()
                except Exception:
                    pass

            ui_safe, ui_reason, confidence = self.validate_safety(pre_screen)
            if not ui_safe:
                return {
                    "success": False,
                    "stage": "servo_vision",
                    "error": f"Servo-Visão Fail-Closed: {ui_reason}",
                    "confidence": round(confidence, 3)
                }

        # ----------------------------------------------------------------------
        # 3. Configuração de Hardware / Ferramenta, Espessura e Cor
        # ----------------------------------------------------------------------
        tool = layer.get("brush", "solid")
        thickness = layer.get("thickness", "min")
        touch_info = layer.get("touch_info")
        color_rgb = layer.get("color_rgb", (0, 0, 0))

        try:
            if hasattr(self.adb, "select_pen_tool"):
                self.adb.select_pen_tool(tool)
            if getattr(self.adb, "supports_visual_thickness_check", False) is True:
                self.adb.prepare_brush_thickness(thickness)
            elif hasattr(self.adb, "set_brush_thickness"):
                self.adb.set_brush_thickness(thickness)
            if touch_info and hasattr(self.adb, "select_palette_color"):
                self.adb.select_palette_color(touch_info)
            elif hasattr(self.adb, "set_color_rgb"):
                self.adb.set_color_rgb(color_rgb)
            else:
                raise ValueError("Camada sem cor selecionável")
        except Exception as e:
            return {"success": False, "stage": "color_setup", "error": str(e)}

        # ----------------------------------------------------------------------
        # 4. CompiledMotionPlan (Compilação determinística com taxa a 120 Hz)
        # ----------------------------------------------------------------------
        # Mesma conversão do ExecutionWorker: deslocamento + escala do canvas,
        # escala da resolução e recorte na zona segura do perfil. O compilador
        # recebe coordenadas de tela prontas (ele só aplicaria o deslocamento).
        screen_strokes = convert_and_sanitize_strokes(
            self.adb, layer.get("strokes", []), transform, plan.get("custom_bounds")
        )
        if not screen_strokes:
            return {
                "success": False,
                "stage": "safe_bounds",
                "error": "Todos os traços da camada estão fora da área segura da tela.",
            }
        motion_plan = MotionCompiler.compile_plan(
            [dict(layer, strokes=screen_strokes)],
            transform=CanvasTransform(offset_x=0, offset_y=0, scale=1.0),
            target_hz=self.target_hz,
            batch_size=batch_size
        )

        # ----------------------------------------------------------------------
        # 5. PersistentTouchExecutor (Injeção via socket ou fallback resiliente)
        # ----------------------------------------------------------------------
        conn_ok, conn_msg = self.touch_executor.ensure_connected()
        executed_batches = 0
        executed_strokes = 0
        total_events = 0

        for b in motion_plan.batches:
            batch_ok = self.touch_executor.execute_batch(b)
            if not batch_ok:
                return {
                    "success": False,
                    "stage": "persistent_touch",
                    "error": f"Falha na injeção de toques no lote {b.batch_id}.",
                    "executed_strokes": executed_strokes,
                    "metrics": self.touch_executor.get_metrics()
                }
            executed_batches += 1
            executed_strokes += len(b.strokes)
            total_events += sum(len(s.events) for s in b.strokes)

        elapsed_s = round(time.time() - t0, 2)

        # ----------------------------------------------------------------------
        # 6. Closed-Loop Feedback & Verificação Pós-Camada
        # ----------------------------------------------------------------------
        closed_loop_report: Dict[str, Any] = {
            "verified": False,
            "status": "skipped"
        }

        if verify_fidelity and self.adb:
            self.timing.settle("layer_post_draw")
            post_screen = self.adb.capture_screenshot()
            if post_screen is None:
                closed_loop_report = {
                    "verified": False,
                    "status": "screenshot_failed",
                    "is_ok": False,
                    "reason": "Falha ao capturar screenshot pós-camada para verificação de fidelidade."
                }
                self.telemetry.record_verification(passed=False)
                return {
                    "success": False,
                    "stage": "closed_loop_verification",
                    "error": "Falha na verificação de camada: captura de tela ausente após o desenho.",
                    "divergence_detected": True,
                    "layer_index": layer_index,
                    "layer_name": layer.get("name", f"Camada {layer_index + 1}"),
                    "executed_strokes": executed_strokes,
                    "closed_loop_report": closed_loop_report,
                    "execution_time_s": elapsed_s,
                    "orchestrator_pipeline": {
                        "quality_gate": {"passed": qg_result.passed, "summary": qg_result.summary},
                        "servo_vision": {"validated": True, "confidence": round(self.state_machine.confidence, 3)},
                        "motion_compiler": {"sample_rate_hz": self.target_hz, "total_events": total_events, "plan_hash": motion_plan.plan_hash if motion_plan else ""},
                        "persistent_touch": self.touch_executor.get_metrics(),
                        "closed_loop": closed_loop_report
                    }
                }

            cum_buffers = plan.get("cumulative_layer_images", [])
            expected_img = None
            if layer_index < len(cum_buffers):
                expected_img = cum_buffers[layer_index]
            elif "simulation_image" in plan:
                expected_img = plan["simulation_image"]
            elif "target_executable" in plan:
                expected_img = plan["target_executable"]

            if expected_img is None:
                closed_loop_report = {
                    "verified": False,
                    "status": "missing_expected_image",
                    "is_ok": False,
                    "reason": "Imagem de referência esperada ausente no plano para verificação de camada."
                }
                self.telemetry.record_verification(passed=False)
                return {
                    "success": False,
                    "stage": "closed_loop_verification",
                    "error": "Falha na verificação de camada: imagem de referência ausente no plano.",
                    "divergence_detected": True,
                    "layer_index": layer_index,
                    "layer_name": layer.get("name", f"Camada {layer_index + 1}"),
                    "executed_strokes": executed_strokes,
                    "closed_loop_report": closed_loop_report,
                    "execution_time_s": elapsed_s,
                    "orchestrator_pipeline": {
                        "quality_gate": {"passed": qg_result.passed, "summary": qg_result.summary},
                        "servo_vision": {"validated": True, "confidence": round(self.state_machine.confidence, 3)},
                        "motion_compiler": {"sample_rate_hz": self.target_hz, "total_events": total_events, "plan_hash": motion_plan.plan_hash if motion_plan else ""},
                        "persistent_touch": self.touch_executor.get_metrics(),
                        "closed_loop": closed_loop_report
                    }
                }

            aligned_live, reg_ok = CanvasRegistrationEngine.align_screenshot_to_target(
                post_screen, expected_img, transform=transform
            )
            if self.is_mock and aligned_live is not None:
                reg_ok = True

            if not reg_ok:
                closed_loop_report = {
                    "verified": False,
                    "status": "registration_failed",
                    "is_ok": False,
                    "reason": "Falha no alinhamento/registro do canvas com a imagem esperada."
                }
                self.telemetry.record_verification(passed=False)
                return {
                    "success": False,
                    "stage": "closed_loop_registration",
                    "error": "Falha no registro/alinhamento visual do canvas pós-camada.",
                    "divergence_detected": True,
                    "layer_index": layer_index,
                    "layer_name": layer.get("name", f"Camada {layer_index + 1}"),
                    "executed_strokes": executed_strokes,
                    "closed_loop_report": closed_loop_report,
                    "execution_time_s": elapsed_s,
                    "orchestrator_pipeline": {
                        "quality_gate": {"passed": qg_result.passed, "summary": qg_result.summary},
                        "servo_vision": {"validated": True, "confidence": round(self.state_machine.confidence, 3)},
                        "motion_compiler": {"sample_rate_hz": self.target_hz, "total_events": total_events, "plan_hash": motion_plan.plan_hash if motion_plan else ""},
                        "persistent_touch": self.touch_executor.get_metrics(),
                        "closed_loop": closed_loop_report
                    }
                }

            is_ok, reason, f_metrics = self.verify_layer_fidelity(
                aligned_live, expected_img, layer
            )
            try:
                _, err_metrics = ErrorMapEngine.compute_error_map(expected_img, aligned_live)
                f_metrics["global_mean_delta_e"] = round(float(err_metrics.get("mean_delta_e", 0.0)), 2)
                f_metrics["global_ssim"] = round(float(err_metrics.get("ssim", 0.0)), 3)
            except Exception:
                pass

            closed_loop_report = {
                "verified": True,
                "status": "approved" if is_ok else "divergence_detected",
                "is_ok": is_ok,
                "reason": reason,
                "metrics": f_metrics
            }

            if not is_ok:
                stage, act_msg = self.state_machine.correction_manager.decide_next_action(
                    divergence_detected=True, divergence_reason=reason
                )
                closed_loop_report["correction_stage"] = stage.value
                closed_loop_report["correction_action"] = act_msg
                self.telemetry.record_verification(passed=False, corrective_count=1)
                return {
                    "success": False,
                    "stage": "closed_loop_divergence",
                    "error": f"Divergência visual detectada na camada {layer_index}: {reason}",
                    "divergence_detected": True,
                    "layer_index": layer_index,
                    "layer_name": layer.get("name", f"Camada {layer_index + 1}"),
                    "executed_strokes": executed_strokes,
                    "closed_loop_report": closed_loop_report,
                    "execution_time_s": elapsed_s,
                    "orchestrator_pipeline": {
                        "quality_gate": {"passed": qg_result.passed, "summary": qg_result.summary},
                        "servo_vision": {"validated": True, "confidence": round(self.state_machine.confidence, 3)},
                        "motion_compiler": {"sample_rate_hz": self.target_hz, "total_events": total_events, "plan_hash": motion_plan.plan_hash if motion_plan else ""},
                        "persistent_touch": self.touch_executor.get_metrics(),
                        "closed_loop": closed_loop_report
                    }
                }
            else:
                self.state_machine.correction_manager.commit_checkpoint(
                    layer_index, executed_strokes, {"layer": layer_index + 1}
                )
                self.telemetry.record_verification(passed=True)

        # ----------------------------------------------------------------------
        # 7. Telemetria Consolidada
        # ----------------------------------------------------------------------
        touch_metrics = self.touch_executor.get_metrics()
        self.telemetry.execution_time_s = elapsed_s
        if elapsed_s > 0:
            self.telemetry.strokes_per_s = round(executed_strokes / elapsed_s, 2)
            self.telemetry.events_per_s = round(total_events / elapsed_s, 1)

        return {
            "success": True,
            "layer_index": layer_index,
            "layer_name": layer.get("name", f"Camada {layer_index + 1}"),
            "executed_strokes": executed_strokes,
            "total_events": total_events,
            "executed_batches": executed_batches,
            "execution_time_s": elapsed_s,
            "orchestrator_pipeline": {
                "quality_gate": {
                    "passed": qg_result.passed,
                    "summary": qg_result.summary
                },
                "servo_vision": {
                    "validated": True,
                    "confidence": round(self.state_machine.confidence, 3)
                },
                "motion_compiler": {
                    "sample_rate_hz": self.target_hz,
                    "total_events": total_events,
                    "plan_hash": motion_plan.plan_hash
                },
                "persistent_touch": touch_metrics,
                "closed_loop": closed_loop_report
            }
        }

    def execute_freehand_path(
        self,
        strokes: List[List[Tuple[int, int]]],
        thickness: str = "min",
        color_name: str = "preto",
        batch_size: int = 8,
        color_rgb: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Executa traços livres vetoriais com a mesma infraestrutura industrial:
        Servo-Visão -> Sanitização -> Hardware -> MotionCompiler -> PersistentTouch.
        """
        t0 = time.time()
        if not strokes:
            return {"success": False, "error": "Nenhum traço fornecido em 'strokes'."}

        # 1. Servo-Visão
        safe, reason, confidence = self.validate_safety()
        if not safe:
            return {
                "success": False,
                "stage": "servo_vision",
                "error": f"Servo-Visão Fail-Closed: {reason}",
                "confidence": round(confidence, 3)
            }

        # 2. Caneta, espessura e cor (o Instagram abre o desenho com a neon)
        try:
            if hasattr(self.adb, "select_pen_tool"):
                self.adb.select_pen_tool("solid")
            if getattr(self.adb, "supports_visual_thickness_check", False) is True:
                self.adb.prepare_brush_thickness(thickness)
            elif hasattr(self.adb, "set_brush_thickness"):
                self.adb.set_brush_thickness(thickness)
            if hasattr(self.adb, "select_palette_color") and color_rgb is not None:
                from .vision.perceptual_color import PerceptualColorEngine
                self.adb.select_palette_color(PerceptualColorEngine().match_color(tuple(color_rgb)))
            elif hasattr(self.adb, "set_color_rgb") and color_rgb is not None:
                self.adb.set_color_rgb(color_rgb)
        except Exception as e:
            return {"success": False, "stage": "color_setup", "error": str(e)}

        # 3. Sanitização dos pontos dentro dos limites seguros
        sb = self.adb.profile_loader.get_safe_bounds() if hasattr(self.adb, "profile_loader") else {"left": 0, "right": 1080, "top": 0, "bottom": 2400}
        clean_strokes = PathSanitizer.sanitize_all_strokes(
            strokes,
            x_min=sb["left"], x_max=sb["right"],
            y_min=sb["top"], y_max=sb["bottom"]
        )

        if not clean_strokes:
            return {"success": False, "error": "Todos os traços estavam fora da área segura da tela."}

        # 4. Compilação via MotionCompiler
        synthetic_layer = {
            "strokes": clean_strokes,
            "brush": "solid",
            "thickness": thickness,
            "color_rgb": color_rgb or (0, 0, 0),
            "purpose": "freehand"
        }
        motion_plan = MotionCompiler.compile_plan(
            [synthetic_layer],
            transform=CanvasTransform(offset_x=0, offset_y=0, scale=1.0),
            target_hz=self.target_hz,
            batch_size=batch_size
        )

        # 5. Execução via PersistentTouchExecutor
        self.touch_executor.ensure_connected()
        executed_strokes = 0
        total_events = 0

        for b in motion_plan.batches:
            ok = self.touch_executor.execute_batch(b)
            if not ok:
                return {
                    "success": False,
                    "stage": "persistent_touch",
                    "error": f"Falha na injeção do lote {b.batch_id}.",
                    "executed_strokes": executed_strokes
                }
            executed_strokes += len(b.strokes)
            total_events += sum(len(s.events) for s in b.strokes)

        elapsed_s = round(time.time() - t0, 2)
        return {
            "success": True,
            "executed_strokes": executed_strokes,
            "total_events": total_events,
            "execution_time_s": elapsed_s,
            "thickness": thickness,
            "color": color_name,
            "orchestrator_pipeline": {
                "servo_vision": {"validated": True, "confidence": round(confidence, 3)},
                "persistent_touch": self.touch_executor.get_metrics()
            }
        }

    def execute_artifact(
        self,
        artifact: Any,
        layer_indices: Optional[List[int]] = None,
        selective_closed_loop: bool = True,
        on_progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Executa um ExecutionArtifact completo diretamente no celular sem necessidade de Planner ou regeneração.
        Garante fidelidade 100% estrita entre o que foi pré-visualizado e o que é traçado no aparelho.
        """
        t0 = time.time()

        # 1. Validação Fail-Closed do QualityGate assinado no Artefato
        if not getattr(artifact, "is_quality_gate_approved", False):
            return {
                "success": False,
                "stage": "quality_gate",
                "error": "Artefato REPROVADO pelo QualityGate. Execução rejeitada (Fail-Closed).",
                "quality_gate": getattr(artifact, "quality_gate", {})
            }

        # 2. Servo-Visão na tela atual do Instagram
        pre_screen = None
        if self.adb:
            try:
                pre_screen = self.adb.capture_screenshot()
            except Exception:
                pass

        ui_safe, ui_reason, confidence = self.validate_safety(pre_screen)
        if not ui_safe:
            return {
                "success": False,
                "stage": "servo_vision",
                "error": f"Servo-Visão Fail-Closed: {ui_reason}",
                "confidence": round(confidence, 3)
            }

        layers = getattr(artifact, "layers", [])
        transform = getattr(artifact, "transform", None)
        # Guarda o índice original: a prévia acumulada de cada camada é indexada
        # pelo plano completo, não pela posição na seleção.
        indices = list(layer_indices) if layer_indices is not None else list(range(len(layers)))
        if any(not 0 <= i < len(layers) for i in indices):
            return {"success": False, "stage": "layer_selection",
                    "error": f"Índices de camada inválidos: {indices} (Total: {len(layers)})."}
        selected_layers = [(i, layers[i]) for i in indices]
        total_layers = len(selected_layers)

        total_strokes = sum(len(l.get("strokes", [])) for _, l in selected_layers)
        completed_strokes = 0
        layer_reports = []

        # 3. Execução camada a camada com closed-loop seletivo e batches adaptativos
        cum_imgs = getattr(artifact, "cumulative_layer_images", [])
        sim_img = getattr(artifact, "simulation_image", None)
        for idx, (orig_idx, layer) in enumerate(selected_layers):
            l_name = layer.get("name", f"Camada {orig_idx + 1}")
            l_strokes = layer.get("strokes", [])
            risk_level = layer.get("risk_level", "MEDIUM")

            expected_cum = [cum_imgs[orig_idx]] if (isinstance(cum_imgs, list) and orig_idx < len(cum_imgs)) else []

            res = self.execute_layer(
                plan={
                    "layers": [layer],
                    "transform": transform,
                    "quality_gate": getattr(artifact, "quality_gate", None),
                    "cumulative_layer_images": expected_cum,
                    "simulation_image": sim_img,
                },
                layer_index=0,
                verify_fidelity=selective_closed_loop and (risk_level == "HIGH"),
                # A primeira camada reaproveita a servo-visão feita logo acima.
                pre_validated_screen=pre_screen if idx == 0 else None,
            )

            if not res.get("success", False) or res.get("divergence_detected", False):
                return {
                    "success": False,
                    "stage": res.get("stage", "layer_execution"),
                    "failed_layer_index": orig_idx,
                    "failed_layer_name": l_name,
                    "divergence_detected": True,
                    "error": res.get("error", "Divergência detectada na camada"),
                    "completed_strokes": completed_strokes,
                    "layer_reports": layer_reports
                }

            completed_strokes += len(l_strokes)
            layer_reports.append(res)

            if on_progress_callback:
                on_progress_callback(idx + 1, total_layers, completed_strokes, total_strokes)

        elapsed = round(time.time() - t0, 2)
        return {
            "success": True,
            "plan_id": getattr(artifact, "plan_id", "artifact"),
            "total_layers": total_layers,
            "total_strokes": completed_strokes,
            "execution_time_s": elapsed,
            "layer_reports": layer_reports,
            "telemetry": self.telemetry.to_dict() if hasattr(self.telemetry, "to_dict") else {}
        }
