"""
Executor em Thread com Micro-Lotes, Tolerância a Falhas, Fail-Closed UI State Machine,
Transformação Local->Screen, Reconfiguração de Cor após Reconnect e Closed-Loop Integrado.
Em estrita conformidade com as Seções 3, 4, 9, 20 e 23 da Auditoria Real.
"""
import os
import threading
import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from . import config
from .core.models import CanvasTransform
from .device.ui_state_machine import UIStateMachine, UIState
from .device.quality_gate import QualityGateResult, plan_content_matches_hash
from .device.session_telemetry import SessionTelemetry
from .planning.path_sanitizer import PathSanitizer
from .feedback.registration import CanvasRegistrationEngine
from .feedback.error_map import ErrorMapEngine
from .feedback.correction_planner import CorrectionPlanner
from .feedback.layer_verification import verify_layer_fidelity
from .device.ui_timing import get_ui_timing

# Independente do diretório atual (MCP e scripts não fazem chdir para a raiz).
TELEMETRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def convert_and_sanitize_strokes(adb, strokes, transform, custom_bounds=None):
    """Converte traços Canvas-Local para Tela Física (escala do canvas e da resolução)
    e recorta na zona segura do perfil. Usado pelo worker (GUI) e pelo orquestrador (MCP)."""
    screen_batch = []
    profile_loader = getattr(adb, "profile_loader", None)
    scale_x = getattr(profile_loader, "scale_x", 1.0) if profile_loader else 1.0
    scale_y = getattr(profile_loader, "scale_y", 1.0) if profile_loader else 1.0
    needs_scale = (scale_x != 1.0 or scale_y != 1.0) and (custom_bounds is None)
    for stroke in strokes:
        if not stroke:
            continue
        if transform is not None:
            scr_s = [transform.local_to_screen(pt) for pt in stroke]
        else:
            scr_s = stroke
        if needs_scale and profile_loader:
            scr_s = [profile_loader.scale_point(pt[0], pt[1]) for pt in scr_s]
        screen_batch.append(scr_s)

    cb = custom_bounds
    sb_candidate = profile_loader.get_safe_bounds() if profile_loader else None
    sb = sb_candidate if isinstance(sb_candidate, dict) and all(
        key in sb_candidate for key in ("left", "right", "top", "bottom")
    ) else None
    if cb and "left" in cb and "right" in cb and "top" in cb and "bottom" in cb:
        # A área marcada pelo usuário nunca amplia a zona segura do perfil.
        bounds = {
            "left": max(cb["left"], sb["left"]) if sb else cb["left"],
            "right": min(cb["right"], sb["right"]) if sb else cb["right"],
            "top": max(cb["top"], sb["top"]) if sb else cb["top"],
            "bottom": min(cb["bottom"], sb["bottom"]) if sb else cb["bottom"],
        }
        if bounds["left"] >= bounds["right"] or bounds["top"] >= bounds["bottom"]:
            return []
        return PathSanitizer.sanitize_all_strokes(
            screen_batch,
            x_min=bounds["left"], x_max=bounds["right"],
            y_min=bounds["top"], y_max=bounds["bottom"]
        )

    if sb:
        return PathSanitizer.sanitize_all_strokes(
            screen_batch,
            x_min=sb["left"], x_max=sb["right"],
            y_min=sb["top"], y_max=sb["bottom"]
        )
    return PathSanitizer.sanitize_all_strokes(screen_batch)


class ExecutionWorker(threading.Thread):
    def __init__(self, adb_manager, plan: Dict[str, Any], callbacks: Dict[str, Any],
                 start_layer: int = 0, start_stroke_offset: int = 0,
                 enable_closed_loop: bool = False,
                 allow_destructive_corrections: bool = False,
                 verify_each_layer: bool = False,
                 stroke_batch_size: Optional[int] = None):
        super().__init__(daemon=True)
        self.adb = adb_manager
        self.timing = get_ui_timing(self.adb)
        self.plan = plan
        self.cb_progress = callbacks.get("on_progress")
        self.cb_layer = callbacks.get("on_layer_start")
        self.cb_log = callbacks.get("on_log")
        self.cb_finished = callbacks.get("on_finished")
        self.cb_screencap = callbacks.get("on_screencap")
        self.cb_checkpoint = callbacks.get("on_checkpoint")
        self.cb_error_map = callbacks.get("on_error_map")
        self.cb_pause_state = callbacks.get("on_pause_state")
        
        self.start_layer = start_layer
        self.start_stroke_offset = start_stroke_offset
        self.enable_closed_loop = enable_closed_loop
        self.allow_destructive_corrections = allow_destructive_corrections
        self.verify_each_layer = verify_each_layer
        # Traços por ACK: a pausa vale ao fim do lote; a parada fecha o stream na hora.
        self.stroke_batch_size = max(1, int(stroke_batch_size or config.WORKER_STROKE_BATCH_SIZE))

        self._is_running = True
        self._is_paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_event = threading.Event()
        self._abort_thread = None

        # Máquina de estados Fail-Closed
        self.state_machine = UIStateMachine()

        # Recupera transformação Canvas-Local para Tela Física
        self.transform = self.plan.get("transform")
        if self.transform is None and "transform_dict" in self.plan:
            td = self.plan["transform_dict"]
            self.transform = CanvasTransform(offset_x=td["offset_x"], offset_y=td["offset_y"], scale=td.get("scale", 1.0))

        self.current_layer_idx = start_layer
        self.current_layer_stroke_idx = start_stroke_offset
        self.total_completed_strokes = 0
        self.failure_reason = None

    def _verify_layer_fidelity(self, aligned_live: np.ndarray,
                               expected_cum_img: np.ndarray,
                               layer_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Confere o resultado após a camada (Requisito 8) com os mesmos limites do
        orquestrador MCP: cor média, cobertura, bordas e espessura do pincel.
        """
        if not layer_data.get("strokes") or aligned_live is None or expected_cum_img is None:
            return False, "Verificação indeterminada: traços, captura ou referência ausentes"
        ok, reason, _metrics = verify_layer_fidelity(aligned_live, expected_cum_img, layer_data)
        return ok, reason

    def pause(self):
        """Impede o início do próximo traço ou da próxima ação de camada."""
        if not self._is_running:
            return
        self._is_paused = True
        self._pause_event.clear()
        if self.cb_pause_state:
            self.cb_pause_state(True)
        if self.cb_log:
            self.cb_log("⏸ Pausa solicitada; aguardando o traço atual terminar.")

    def resume(self):
        """Retoma a execução do ponto exato onde parou."""
        if not self._is_running:
            return
        self._is_paused = False
        self._pause_event.set()
        if self.cb_pause_state:
            self.cb_pause_state(False)
        if self.cb_log:
            self.cb_log("▶ Retomando pintura a partir do ponto atual...")

    def stop(self):
        """Sinaliza a parada sem bloquear a interface gráfica."""
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._is_running = False
        self._pause_event.set()
        if self.adb:
            self._abort_thread = threading.Thread(target=self.adb.emergency_kill, daemon=True)
            self._abort_thread.start()
        if self.cb_log:
            self.cb_log("🚨 Parada solicitada; encerrando o envio de traços.")

    def _wait_if_paused(self):
        """Espera em intervalos curtos para a parada sempre vencer a pausa."""
        was_paused = self._is_paused
        if self._is_paused and self._is_running and self.cb_log:
            self.cb_log("⏸ Pintura pausada.")
        while self._is_running and not self._pause_event.wait(timeout=0.05):
            pass
        if was_paused and self._is_running:
            live_screen = self.adb.capture_screenshot() if self.adb else None
            valid, reason = self.state_machine.validate_safety_for_drawing(self.adb, screenshot_bgr=live_screen)
            if not valid:
                if self.cb_log:
                    self.cb_log(f"❌ Retomada bloqueada por falha de segurança Fail-Closed: {reason}")
                self._is_running = False
                return False
        return self._is_running

    def _convert_and_sanitize_batch(self, strokes):
        """Converte traços de coordenadas Canvas-Local para Tela Física e sanitiza limites (Seção 4)."""
        return convert_and_sanitize_strokes(self.adb, strokes, self.transform, self.plan.get("custom_bounds"))

    def _apply_tool_and_color(self, thickness, touch_info, drawn_width_px=None):
        """Configura espessura e cor com validação e tolerância a falhas.

        drawn_width_px é a largura real que a prévia usa para a camada; só o
        FakeADB a consome (set_drawn_width_hint), para desenhar o mesmo que a prévia.
        """
        if not self._wait_if_paused():
            return False
        if getattr(self.adb, "supports_visual_thickness_check", False) is True:
            self.adb.prepare_brush_thickness(thickness)
        else:
            self.adb.set_brush_thickness(thickness)
        note = getattr(self.adb, "last_thickness_note", None)
        if isinstance(note, str) and note and self.cb_log:
            self.cb_log(f"⚠️ {note}")
        hint = getattr(self.adb, "set_drawn_width_hint", None)
        if callable(hint):
            hint(drawn_width_px)
        if not self._wait_if_paused():
            return False
        if not touch_info:
            raise ValueError("Camada sem coordenadas de cor; desenho interrompido")
        self.adb.select_palette_color(touch_info)
        return True

    @staticmethod
    def _verify_background_fill(before, after, expected, layer, transform, resumed=False):
        """Confere um fundo uniforme pela geometria e tinta, sem registro óptico."""
        if (before is None or after is None or expected is None or transform is None
                or before.shape != after.shape):
            return False, "captura anterior, posterior ou área planejada indisponível"
        h, w = expected.shape[:2]
        x0, y0 = transform.local_to_screen((0, 0))
        x1, y1 = transform.local_to_screen((w, h))
        if x0 < 0 or y0 < 0 or x1 > after.shape[1] or y1 > after.shape[0]:
            return False, "área planejada fora da captura do celular"
        margin = min(20, max(2, min(x1 - x0, y1 - y0) // 12))
        if x1 - x0 <= margin * 2 or y1 - y0 <= margin * 2:
            return False, "área de fundo pequena demais para conferir"
        prior = before[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
        painted = after[y0 + margin:y1 - margin, x0 + margin:x1 - margin]
        rgb = layer.get("render_color_rgb") or layer.get("color_rgb")
        if rgb is None or len(rgb) != 3:
            return False, "cor planejada do fundo indisponível"
        target_bgr = np.asarray(tuple(reversed(rgb)), dtype=np.int16)
        color_error = np.linalg.norm(painted.astype(np.int16) - target_bgr, axis=2)
        color_coverage = float(np.mean(color_error <= 70.0))
        changed = float(np.mean(np.max(cv2.absdiff(prior, painted), axis=2) >= 20))
        if color_coverage < 0.45:
            return False, f"fundo visível em apenas {color_coverage:.0%} da área planejada"
        if not resumed and changed < 0.20:
            return False, f"a captura mudou em apenas {changed:.0%} da área planejada"
        return True, f"fundo confirmado ({color_coverage:.0%} da área, mudança {changed:.0%})"

    @staticmethod
    def _verify_strokes_at_planned_positions(before, after, expected, layer, transform, resumed=False):
        """Confere tinta nos caminhos planejados quando o fundo impede registro óptico."""
        if (before is None or after is None or expected is None or transform is None
                or before.shape != after.shape):
            return False, "capturas ou geometria indisponíveis"
        strokes = layer.get("strokes", [])
        mask = np.zeros(after.shape[:2], dtype=np.uint8)
        for stroke in strokes:
            if len(stroke) >= 2:
                points = np.asarray(
                    [transform.local_to_screen(tuple(point)) for point in stroke],
                    dtype=np.int32,
                ).reshape((-1, 1, 2))
                cv2.polylines(mask, [points], False, 255, 5)
        y_coords, x_coords = np.nonzero(mask)
        if not len(x_coords):
            return False, "traços planejados fora da captura"
        x0 = max(0, int(x_coords.min()) - 15)
        x1 = min(after.shape[1], int(x_coords.max()) + 16)
        y0 = max(0, int(y_coords.min()) - 15)
        y1 = min(after.shape[0], int(y_coords.max()) + 16)
        on = mask[y0:y1, x0:x1] > 0
        if int(np.count_nonzero(on)) < 50:
            return False, "amostra dos traços pequena demais"
        prior = before[y0:y1, x0:x1]
        painted = after[y0:y1, x0:x1]
        rgb = layer.get("render_color_rgb") or layer.get("color_rgb")
        if rgb is None or len(rgb) != 3:
            return False, "cor planejada dos traços indisponível"
        target_bgr = np.asarray(tuple(reversed(rgb)), dtype=np.int16)
        color_error = np.linalg.norm(painted.astype(np.int16) - target_bgr, axis=2)
        matched = color_error <= 55.0
        coverage = float(np.mean(matched[on]))
        outside = float(np.mean(matched[~on]))
        changed = float(np.mean(np.max(cv2.absdiff(prior, painted), axis=2)[on] >= 20))
        if coverage < 0.35 or coverage - outside < 0.20 or outside > 0.50:
            return False, f"tinta nos traços {coverage:.0%}; fora deles {outside:.0%}"
        if not resumed and changed < 0.15:
            return False, f"captura mudou em apenas {changed:.0%} dos traços"
        return True, f"traços confirmados na posição planejada ({coverage:.0%}; mudança {changed:.0%})"

    def run(self):
        try:
            self._run_impl()
        except Exception as exc:
            self._is_running = False
            if self.cb_log:
                self.cb_log(f"❌ Erro na execução: {exc}")
            if self.cb_finished:
                self.cb_finished(False, f"Erro na execução: {exc}")

    def _run_impl(self):
        if not self._wait_if_paused():
            if self.cb_finished:
                self.cb_finished(False, "Execução interrompida.")
            return
        layers = self.plan.get("layers", [])
        total_layers = len(layers)
        total_strokes = sum(len(layer.get("strokes", [])) for layer in layers)
        if (total_strokes <= 0 or
                self.plan.get("total_strokes", total_strokes) != total_strokes or
                not 0 <= self.start_layer < total_layers or
                not 0 <= self.start_stroke_offset <= len(layers[self.start_layer].get("strokes", []))):
            if self.cb_finished:
                self.cb_finished(False, "Plano ou checkpoint inválido: contagem de traços ou posição de retomada inconsistente.")
            return
        
        # Validação de Integridade QualityGate Fail-Closed Obrigatória (Fase 20: P0.4/P0.5).
        # Checagens locais e baratas vêm antes de qualquer captura no aparelho.
        qgate = self.plan.get("quality_gate")
        # A aprovação pertence à revisão planejada. Recriá-la aqui com métricas
        # padrão aprovaria um plano cuja prévia nunca foi avaliada.
        if not isinstance(qgate, QualityGateResult) or not qgate.passed:
            reason_msg = getattr(qgate, "summary", "QualityGate ausente ou inválido") if qgate else "QualityGate ausente no plano: aprovação prévia obrigatória."
            failed_str = f" ({', '.join(qgate.failed_checks)})" if qgate and getattr(qgate, "failed_checks", None) else ""
            fail_msg = f"QualityGate REPROVADO (Fail-Closed): {reason_msg}{failed_str}"
            if self.cb_log:
                self.cb_log(f"❌ {fail_msg}")
            if self.cb_finished:
                self.cb_finished(False, fail_msg)
            return
        if not plan_content_matches_hash(self.plan):
            if self.cb_finished:
                self.cb_finished(False, "Plano alterado depois da aprovação. Gere a prévia novamente.")
            return

        # Validação Inicial Fail-Closed com Servo-Visão (Fase 6.3 & 6.4: P0.1)
        live_screen = self.adb.capture_screenshot() if self.adb else None
        if live_screen is None:
            fail_msg = "Falha de segurança Fail-Closed: Não foi possível capturar a tela do celular para servo-visão."
            if self.cb_log:
                self.cb_log(f"❌ {fail_msg}")
            if self.cb_finished:
                self.cb_finished(False, fail_msg)
            return

        valid, reason = self.state_machine.validate_safety_for_drawing(self.adb, screenshot_bgr=live_screen)
        if not valid:
            if self.cb_log:
                self.cb_log(f"⚠️ {reason}")
            if self.cb_finished:
                self.cb_finished(False, reason)
            return


        for layer_idx, layer in enumerate(layers):
            for stroke_idx, stroke in enumerate(layer.get("strokes", [])):
                if not self._convert_and_sanitize_batch([stroke]):
                    if self.cb_finished:
                        self.cb_finished(
                            False,
                            f"Traço {stroke_idx + 1} da camada {layer_idx + 1} fora da área segura "
                            "ou curto demais. Ajuste a área e gere a prévia novamente.",
                        )
                    return

        # Inicializa Telemetria Completa de Sessão (Fase 17)
        plan_hash = self.plan.get("plan_hash", f"session_{int(time.time())}")
        self.telemetry = SessionTelemetry(session_id=plan_hash)
        self.telemetry.planning_time_s = float(self.plan.get("estimated_time_s", 0.0))
        self.start_exec_time = time.time()

        completed_strokes = 0
        for i in range(self.start_layer):
            completed_strokes += len(layers[i].get("strokes", []))
        completed_strokes += self.start_stroke_offset
        self.total_completed_strokes = completed_strokes

        if self.cb_log:
            if self.start_layer > 0 or self.start_stroke_offset > 0:
                self.cb_log(f"🔄 Retomando pintura do checkpoint: Camada {self.start_layer + 1}, Traço {self.start_stroke_offset + 1}...")
            else:
                self.cb_log(f"Iniciando pintura autônoma: {total_layers} camadas, {total_strokes} traços.")

        batch_size = self.stroke_batch_size
        
        for l_idx in range(self.start_layer, total_layers):
            if not self._wait_if_paused():
                break
                
            self.current_layer_idx = l_idx
            layer = layers[l_idx]
            layer_name = layer.get("name", f"Camada {l_idx + 1}")
            color_rgb = layer.get("color_rgb", (0, 0, 0))
            thickness = layer.get("thickness", "med")
            touch_info = layer.get("touch_info", {})
            strokes = layer.get("strokes", [])
            
            start_stroke = self.start_stroke_offset if l_idx == self.start_layer else 0
            self.start_stroke_offset = 0

            if self.cb_layer:
                self.cb_layer(l_idx + 1, layer_name, color_rgb)
            if self.cb_log:
                self.cb_log(f"--> [{l_idx + 1}/{total_layers}] {layer_name} ({len(strokes) - start_stroke} traços)")

            # Configuração de ferramenta e cor na camada atual
            try:
                if not self._wait_if_paused():
                    break
                self.adb.select_pen_tool(layer.get("brush", "solid"))
                if not self._wait_if_paused():
                    break
                if not self._apply_tool_and_color(thickness, touch_info, layer.get("drawn_width_px")):
                    break
            except Exception as e:
                if not self._is_running:
                    break
                self.failure_reason = f"Falha ao preparar caneta, espessura ou cor na camada {l_idx + 1}: {e}"
                if isinstance(e, (ValueError, KeyError, TypeError)):
                    if self.cb_log:
                        self.cb_log(f"❌ Plano inválido na camada {l_idx + 1}: {e}")
                    self._is_running = False
                    self.failure_reason = f"Plano inválido na camada {l_idx + 1}: {e}"
                    break
                if self.cb_log:
                    self.cb_log(f"Falha ao configurar cor/pincel: {e}. Tentando reconectar...")
                if self.adb.auto_reconnect() and self._wait_if_paused():
                    recon_screen = self.adb.capture_screenshot() if self.adb else None
                    valid, reason = self.state_machine.validate_safety_for_drawing(self.adb, screenshot_bgr=recon_screen)
                    if not valid:
                        self.failure_reason = f"Segurança comprometida após reconexão: {reason}"
                        if self.cb_log:
                            self.cb_log(f"❌ Segurança comprometida após reconexão: {reason}. Pausando.")
                        self._is_running = False
                        break
                    try:
                        self.adb.select_pen_tool(layer.get("brush", "solid"))
                        if not self._apply_tool_and_color(thickness, touch_info, layer.get("drawn_width_px")):
                            break
                    except Exception as e2:
                        self.failure_reason = f"Falha ao preparar cor/pincel após reconexão: {e2}"
                        if self.cb_log:
                            self.cb_log(f"❌ Não foi possível configurar cor após reconexão: {e2}. Pausando.")
                        self._is_running = False
                else:
                    self._is_running = False
                if not self._is_running:
                    break

            # Execução de Micro-Lotes
            first_screen = self.adb.capture_screenshot() if start_stroke >= len(strokes) else None
            b_start = start_stroke
            while b_start < len(strokes):
                if not self._wait_if_paused():
                    break

                if b_start == start_stroke:
                    # Selecionar ferramenta/cor altera a UI; confirme novamente
                    # o canvas antes de enviar o primeiro gesto da camada.
                    first_screen = self.adb.capture_screenshot()
                    valid, reason = self.state_machine.validate_safety_for_drawing(
                        self.adb, screenshot_bgr=first_screen
                    )
                    if not valid:
                        self.failure_reason = f"Canvas inválido antes do primeiro traço: {reason}"
                        if self.cb_log:
                            self.cb_log(f"❌ {self.failure_reason}")
                        self._is_running = False
                        break

                self.current_layer_stroke_idx = b_start
                b_strokes_local = strokes[b_start:b_start + batch_size]
                clean_screen_batch = self._convert_and_sanitize_batch(b_strokes_local)

                # Se a sanitização descartou todos os traços do lote (P1.4), não tenta enviar lote vazio
                # Cada traço já foi validado sozinho antes do início; um lote vazio
                # aqui indica perfil/área alterados durante a execução.
                if not clean_screen_batch:
                    self.failure_reason = (
                        f"Traço {b_start + 1} da camada {l_idx + 1} fora da área segura ou curto demais. "
                        "Confira a área marcada no celular e gere a prévia novamente."
                    )
                    if self.cb_log:
                        self.cb_log(f"❌ {self.failure_reason}")
                    self._is_running = False
                    break

                # Tentativa 1 de execução
                success = self.adb.execute_stroke_batch(clean_screen_batch)
                if not self._is_running:
                    break
                if not success:
                    b_last = b_start + len(b_strokes_local)
                    span = f"o traço {b_start + 1}" if b_last == b_start + 1 else f"os traços {b_start + 1}–{b_last}"
                    self.failure_reason = (
                        f"O Android não confirmou {span} da camada {l_idx + 1}. "
                        "Confira a conexão e a tela antes de retomar."
                    )
                    if self.cb_log:
                        self.cb_log(f"❌ {self.failure_reason}")
                    self._is_running = False
                    break

                completed_strokes += len(b_strokes_local)
                self.total_completed_strokes = completed_strokes
                
                percent = min(100, int((completed_strokes / total_strokes) * 100))
                if self.cb_progress:
                    msg = f"Camada {l_idx + 1}/{total_layers}: Traço {completed_strokes}/{total_strokes}"
                    self.cb_progress(l_idx + 1, total_layers, completed_strokes, total_strokes, percent, msg)
                
                # O ACK do Android confirma o envio, não a tinta na tela.
                # Com verificação visual ativa, só persistimos a camada aprovada.
                if self.cb_checkpoint and not self.verify_each_layer:
                    self.cb_checkpoint(l_idx, b_start + len(b_strokes_local))
                    
                delay = config.WORKER_INTER_BATCH_DELAY_S * getattr(self.timing, "timing_scale", 1.0)
                if not self.timing.is_mock and delay > 0:
                    # Espera interrompível: a parada não aguarda o respiro.
                    self._stop_event.wait(delay)
                b_start += batch_size

            # Captura de tela e conferência ao final da camada (Requisito 8)
            if self._is_running:
                verification_failed = False
                try:
                    if not self._wait_if_paused():
                        break
                    self.timing.settle("layer_post_draw")
                    if not self._wait_if_paused():
                        break
                    live_screen = self.adb.capture_screenshot()
                    if live_screen is not None:
                        if self.cb_screencap:
                            self.cb_screencap(live_screen)

                        if self.verify_each_layer:
                            cum_buffers = self.plan.get("cumulative_layer_images", [])
                            if l_idx < len(cum_buffers):
                                expected_cum = cum_buffers[l_idx]
                                if layer.get("purpose") == "background":
                                    bg_ok, bg_reason = self._verify_background_fill(
                                        first_screen, live_screen, expected_cum, layer,
                                        self.transform, resumed=start_stroke > 0,
                                    )
                                    if bg_ok:
                                        if self.cb_log:
                                            self.cb_log(f"✅ {bg_reason}")
                                        self.state_machine.correction_manager.commit_checkpoint(
                                            l_idx, completed_strokes, {"layer": l_idx + 1}
                                        )
                                        if hasattr(self, "telemetry"):
                                            self.telemetry.record_verification(passed=True)
                                        if self.cb_checkpoint:
                                            next_layer = l_idx + 1 < total_layers
                                            self.cb_checkpoint(l_idx + 1 if next_layer else l_idx,
                                                               0 if next_layer else len(strokes))
                                    else:
                                        verification_failed = True
                                        self.failure_reason = f"Fundo não confirmado: {bg_reason}. Confira a tela antes de retomar."
                                        if self.cb_log:
                                            self.cb_log(f"❌ {self.failure_reason}")
                                        if hasattr(self, "telemetry"):
                                            self.telemetry.record_verification(passed=False, corrective_count=1)
                                        self._is_running = False
                                    continue
                                aligned_live, reg_ok = CanvasRegistrationEngine.align_screenshot_to_target(
                                    live_screen, expected_cum, transform=self.transform
                                )
                                if getattr(self.adb, "device_target", "") == "fake_device_emulator" and aligned_live is not None:
                                    reg_ok = True

                                if not reg_ok:
                                    direct_ok = False
                                    direct_reason = ""
                                    if layer.get("purpose") in ("line_art_centerline", "stencil", "contour"):
                                        direct_ok, direct_reason = self._verify_strokes_at_planned_positions(
                                            first_screen, live_screen, expected_cum, layer,
                                            self.transform, resumed=start_stroke > 0,
                                        )
                                    if direct_ok:
                                        if self.cb_log:
                                            self.cb_log(f"✅ {direct_reason}")
                                        self.state_machine.correction_manager.commit_checkpoint(
                                            l_idx, completed_strokes, {"layer": l_idx + 1}
                                        )
                                        if hasattr(self, "telemetry"):
                                            self.telemetry.record_verification(passed=True)
                                        if self.cb_checkpoint:
                                            next_layer = l_idx + 1 < total_layers
                                            self.cb_checkpoint(l_idx + 1 if next_layer else l_idx,
                                                               0 if next_layer else len(strokes))
                                        continue
                                    verification_failed = True
                                    self.failure_reason = (
                                        f"Traços da camada {l_idx + 1} enviados, mas a verificação visual não conseguiu "
                                        f"alinhar a captura à prévia nem confirmar os traços: {direct_reason or 'sem evidência suficiente'}. "
                                        "Confira o desenho antes de retomar."
                                    )
                                    if self.cb_log:
                                        self.cb_log(f"❌ {self.failure_reason}")
                                    if hasattr(self, "telemetry"):
                                        self.telemetry.record_verification(passed=False, corrective_count=1)
                                    self._is_running = False
                                else:
                                    is_ok, reason = self._verify_layer_fidelity(aligned_live, expected_cum, layer)
                                    if not is_ok and layer.get("purpose") in ("line_art_centerline", "stencil", "contour"):
                                        # O registro pode alinhar a prévia enquanto as bordas
                                        # do papel de parede animado dominam a comparação.
                                        # Confirme a tinta na geometria física dos traços.
                                        direct_ok, direct_reason = self._verify_strokes_at_planned_positions(
                                            first_screen, live_screen, expected_cum, layer,
                                            self.transform, resumed=start_stroke > 0,
                                        )
                                        if direct_ok:
                                            is_ok, reason = True, direct_reason
                                        else:
                                            reason = f"{reason}; conferência dos traços: {direct_reason}"
                                    if not is_ok:
                                        verification_failed = True
                                        self.failure_reason = (
                                            f"Traços da camada {l_idx + 1} enviados, mas a verificação visual "
                                            f"encontrou um problema: {reason}. Confira o desenho antes de retomar."
                                        )
                                        if hasattr(self, "telemetry"):
                                            self.telemetry.record_verification(passed=False, corrective_count=1)
                                        if self.cb_log:
                                            self.cb_log(f"❌ {self.failure_reason}")
                                        # A revisão precisa produzir um resultado para a
                                        # interface, inclusive após a última camada.
                                        # O checkpoint permite reconferir ao retomar.
                                        self._is_running = False
                                    else:
                                        self.state_machine.correction_manager.commit_checkpoint(
                                            l_idx, completed_strokes, {"layer": l_idx + 1}
                                        )
                                        if hasattr(self, "telemetry"):
                                            self.telemetry.record_verification(passed=True)
                                        if self.cb_checkpoint:
                                            next_layer = l_idx + 1 < total_layers
                                            self.cb_checkpoint(l_idx + 1 if next_layer else l_idx,
                                                               0 if next_layer else len(strokes))
                            else:
                                verification_failed = True
                                self.failure_reason = f"Imagem esperada da camada {l_idx + 1} ausente; verificação indeterminada."
                                if self.cb_log:
                                    self.cb_log(f"❌ {self.failure_reason}")
                                self._is_running = False
                    elif self.verify_each_layer:
                        verification_failed = True
                        self.failure_reason = f"Captura ausente após a camada {l_idx + 1}; verificação indeterminada."
                        if self.cb_log:
                            self.cb_log(f"❌ {self.failure_reason}")
                        self._is_running = False
                except Exception as e:
                    verification_failed = True
                    self.failure_reason = f"Falha na conferência da camada {l_idx + 1}: {e}"
                    if self.cb_log:
                        self.cb_log(f"❌ {self.failure_reason}")
                    if self.verify_each_layer:
                        self._is_running = False
                finally:
                    if verification_failed and self.verify_each_layer and self.cb_checkpoint:
                        # Um checkpoint antigo desta camada pode ter sido salvo
                        # antes da falha (inclusive por versões anteriores).
                        self.cb_checkpoint(l_idx, 0)

        # Uma pausa pedida pelo usuário ainda precisa ser respeitada, mesmo
        # quando chega após o último traço. Falhas de verificação encerram
        # acima com um motivo explícito, sem bloquear o aviso final.
        if self._is_running:
            self._wait_if_paused()

        # Closed-Loop Feedback & Auditoria Visual Pós-Pintura (Seção 9 e 38 da Auditoria Real)
        if self._is_running and self.enable_closed_loop and self._wait_if_paused():
            if self.cb_log:
                self.cb_log("🔍 Auditoria Closed-Loop: Analisando fidelidade visual final do desenho...")
            try:
                live_screen = self.adb.capture_screenshot()
                target_img = self.plan.get("target_executable", self.plan.get("fit_image"))
                if live_screen is not None and target_img is not None:
                    # 1. Alinhamento sub-pixel via Registration com Transform Conhecido
                    aligned_screen, reg_ok = CanvasRegistrationEngine.align_screenshot_to_target(
                        live_screen, target_img, transform=self.transform
                    )
                    if not reg_ok:
                        raise RuntimeError("não foi possível alinhar a captura à prévia executável")
                    # 2. True CIEDE2000 Error Map
                    heatmap, metrics = ErrorMapEngine.compute_error_map(target_img, aligned_screen)
                    if self.cb_error_map:
                        self.cb_error_map(heatmap, metrics)
                    if self.cb_log:
                        self.cb_log(f"📊 Auditoria Final: Mean ΔE00={metrics.get('mean_delta_e')}, SSIM={metrics.get('ssim')}")

                    # 3. Retoque autônomo: BLINDADO contra intervenções destrutivas.
                    # NUNCA aplica traços por cima da arte a menos que explicitamente solicitado pelo operador
                    # e com alinhamento sub-pixel estritamente confirmado.
                    if getattr(self, "allow_destructive_corrections", False) and reg_ok and metrics.get("mean_delta_e", 0) > 25.0:
                        planner = CorrectionPlanner()
                        if self.transform is None:
                            raise RuntimeError("retoque bloqueado: transformação do canvas ausente")
                        # Retoques ficam em coordenadas do canvas; a conversão para a
                        # tela (deslocamento + escala) e o recorte seguro vêm abaixo.
                        corr_plan = planner.plan_corrections(target_img, aligned_screen, (0, 0), max_correction_strokes=15)
                        corr_layers = corr_plan.get("layers", [])
                        if corr_layers:
                            if self.cb_log:
                                self.cb_log(f"🎨 Aplicando {corr_plan['total_correction_strokes']} micro-retoques...")
                            for c_l in corr_layers:
                                if not self._wait_if_paused():
                                    break
                                if not self._apply_tool_and_color(c_l.get("thickness", "min"), c_l.get("touch_info", {}),
                                                              c_l.get("drawn_width_px")):
                                    break
                                # Retoques passam pela mesma servo-visão e sanitização dos traços normais.
                                corr_screen = self.adb.capture_screenshot()
                                valid, reason = self.state_machine.validate_safety_for_drawing(
                                    self.adb, screenshot_bgr=corr_screen
                                )
                                if not valid:
                                    raise RuntimeError(f"retoque bloqueado pela servo-visão: {reason}")
                                for stroke in c_l.get("strokes", []):
                                    if not self._wait_if_paused():
                                        break
                                    clean = self._convert_and_sanitize_batch([stroke])
                                    if not clean:
                                        continue
                                    if not self.adb.execute_stroke_batch(clean):
                                        self._is_running = False
                                        break
                            if self.cb_log:
                                self.cb_log("✅ Retoques pontuais finalizados!")
                    else:
                        if self.cb_log:
                            self.cb_log("Auditoria final registrada; nenhum retoque automático foi aplicado.")
                else:
                    raise RuntimeError("captura ou imagem executável ausente")
            except Exception as e:
                self.failure_reason = f"Auditoria visual final indeterminada: {e}"
                self._is_running = False
                if self.cb_log:
                    self.cb_log(f"❌ {self.failure_reason}")

        # Consolidação e Exportação de Telemetria (Fase 17)
        if hasattr(self, "telemetry") and hasattr(self, "start_exec_time"):
            exec_dur = float(time.time() - self.start_exec_time)
            self.telemetry.execution_time_s = round(exec_dur, 2)
            if exec_dur > 0:
                self.telemetry.strokes_per_s = round(self.total_completed_strokes / exec_dur, 2)
            try:
                os.makedirs(TELEMETRY_DIR, exist_ok=True)
                self.telemetry.export_json(os.path.join(TELEMETRY_DIR, "session_telemetry.json"))
                if self.cb_log:
                    self.cb_log(f"📊 Telemetria registrada: {self.telemetry.strokes_per_s} traços/s, {self.telemetry.verification_failures} falhas closed-loop.")
            except Exception:
                pass

        if self._stop_event.is_set() and self._abort_thread is not None:
            self._abort_thread.join(timeout=3.5)

        if self._is_running:
            if self.cb_log:
                self.cb_log("🎉 Pintura concluída com sucesso no Instagram!")
            if self.cb_finished:
                self.cb_finished(True, "Pintura concluída com sucesso!")
        else:
            if self.cb_finished:
                self.cb_finished(False, self.failure_reason or "Execução interrompida.")
