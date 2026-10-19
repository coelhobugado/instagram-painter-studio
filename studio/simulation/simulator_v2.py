"""
Simulador Calibrado v2 com Modelo Físico Completo de Pincéis, Alpha Compositing e Determinismo.
Implementa a Seção 10 (Simulator v2), Seção 9 (Modelo de Marcador/Opacidade), Seção 17 da Auditoria Real e Seção 54 (Determinismo).
"""
import os
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class CalibratedSimulatorV2:
    def __init__(self, profile_dir: str = None):
        if profile_dir is None:
            profile_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "profiles", "SM-A736B"
            )
        self.profile_dir = profile_dir
        self.brushes_config = self._load_json(os.path.join(profile_dir, "brushes.json"))
        self.opacity_config = self._load_json(os.path.join(profile_dir, "opacity.json"))

    def _load_json(self, path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_calibrated_width_and_softness(self, tool: str, thickness_level: str) -> Tuple[int, float]:
        """Obtém a largura efetiva em pixels e edge_softness a partir da calibração real de hardware."""
        from ..device.brush_width import parse_width_level
        exact = parse_width_level(thickness_level)
        if exact is not None:
            return max(1, int(round(exact))), 0.8
        brush = self.brushes_config.get("brushes", {}).get(tool, self.brushes_config.get("brushes", {}).get("solid", {}))
        levels = brush.get("levels", {})
        if thickness_level in levels:
            width = max(1, int(round(levels[thickness_level].get("effective_width_px", 6))))
            softness = float(levels[thickness_level].get("edge_softness", 0.8))
            return width, softness
        return 6, 0.8

    def get_calibrated_width(self, thickness_level: str, tool: str = "solid") -> int:
        """Obtém a largura efetiva em pixels a partir da calibração real de hardware."""
        width, _ = self.get_calibrated_width_and_softness(tool, thickness_level)
        return width

    def render_simulation(self, canvas_shape: Tuple[int, int, int],
                          layers: List[Dict[str, Any]],
                          background_bgr: Tuple[int, int, int] = (22, 22, 22),
                          seed: int = 42,
                          transform: Any = None,
                          initial_canvas: np.ndarray = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Executa a simulação física completa da pintura camada por camada em coordenadas do canvas local.
        Suporta acúmulo incremental de passagens (marker), núcleo e halo (neon) e difusão de borda (edge_softness).
        Retorna:
            - canvas_final: Imagem BGR final simulada
            - cumulative_buffers: Lista de imagens com o estado da pintura após cada camada
        """
        from ..adb_manager import ADBManager
        from .. import config
        from ..device.touch_injector import resample_stroke
        
        h, w = canvas_shape[:2]
        if initial_canvas is not None:
            if initial_canvas.shape != canvas_shape:
                raise ValueError("Canvas inicial incompatível com a prévia")
            canvas = initial_canvas.copy()
        else:
            canvas = np.empty((h, w, 3), dtype=np.uint8)
            canvas[:, :] = background_bgr

        cumulative_buffers = []
        pass_table = self.opacity_config.get("pass_alpha_table", {})
        alpha_single_pass = float(pass_table.get("1", 0.50))

        for layer in layers:
            strokes = layer.get("strokes", [])
            # Seção 9 do Plano: Simulador usa a cor física executável (render_color) da LUT
            render_rgb = (
                layer.get("render_color_rgb") or
                layer.get("touch_info", {}).get("rgb") or
                layer.get("color_rgb", (200, 200, 200))
            )
            color_bgr = (int(render_rgb[2]), int(render_rgb[1]), int(render_rgb[0]))
            thickness_mode = layer.get("thickness", "med")
            tool = layer.get("brush", "solid")

            width_px, edge_softness = self.get_calibrated_width_and_softness(tool, thickness_mode)
            # Largura medida no aparelho para o tipo de traço da camada (ex.: letras),
            # quando difere da nominal do slider
            if layer.get("drawn_width_px"):
                width_px = max(1, int(round(float(layer["drawn_width_px"]))))

            # Normalização de coordenadas para o canvas local
            # Se os traços vierem com offset de tela antiga, converte para local
            adjusted_strokes = []
            for s in strokes:
                if not s:
                    continue
                # Se os pontos excederem o canvas_shape, mas houver transform, ajusta
                if transform is not None:
                    adj_s = [transform.screen_to_local(pt) if pt[0] > w or pt[1] > h else pt for pt in s]
                else:
                    adj_s = s
                # Reproduz o caminho que o transporte ativo envia ao celular: o injetor
                # reamostra o traço contínuo; o modo clássico manda uma aresta por swipe.
                if config.TOUCH_TRANSPORT == "injector":
                    adjusted_strokes.append(resample_stroke(adj_s, config.INJECTOR_MAX_STEP_PX))
                else:
                    adjusted_strokes.append(ADBManager._compact_stroke(adj_s))

            if tool == "marker":
                # Modelo de Acúmulo Incremental por Pincelada (Seção 17.1 da Auditoria Real):
                # Cada traço acumula opacidade alpha sobre o fundo e sobre traços cruzados anteriores
                for stroke in adjusted_strokes:
                    if len(stroke) < 2:
                        continue
                    pts = np.array(stroke, dtype=np.int32).reshape((-1, 2))
                    ksize = int(round(edge_softness * 2)) | 1 if edge_softness > 1.0 else 0
                    # Compõe apenas na bbox do traço (+ largura e blur), não no frame inteiro
                    pad = width_px + ksize + 2
                    x0 = max(0, int(pts[:, 0].min()) - pad)
                    y0 = max(0, int(pts[:, 1].min()) - pad)
                    x1 = min(w, int(pts[:, 0].max()) + pad + 1)
                    y1 = min(h, int(pts[:, 1].max()) + pad + 1)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    stroke_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                    cv2.polylines(stroke_mask, [(pts - (x0, y0)).reshape((-1, 1, 2))], False, 255, width_px, cv2.LINE_AA)

                    if ksize:
                        stroke_mask = cv2.GaussianBlur(stroke_mask, (ksize, ksize), edge_softness * 0.5)

                    alpha_map = ((stroke_mask.astype(np.float32) / 255.0) * alpha_single_pass)[..., None]
                    roi = canvas[y0:y1, x0:x1]
                    roi[:] = np.clip(
                        roi.astype(np.float32) * (1.0 - alpha_map) + np.float32(color_bgr) * alpha_map,
                        0.0, 255.0
                    ).astype(np.uint8)

            elif tool == "neon":
                # Modelo Físico Neon (Seção 17.3 da Auditoria Real):
                # 1. Halo difuso colorido alargado
                # 2. Núcleo claro de alta intensidade
                glow_width = max(width_px + 2, int(round(width_px * 2.2)))
                core_width = max(1, int(round(width_px * 0.7)))

                glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
                for stroke in adjusted_strokes:
                    if len(stroke) >= 2:
                        pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(glow_layer, [pts], False, color_bgr, glow_width, cv2.LINE_AA)

                # Difusão gaussiana do halo
                glow_blur = cv2.GaussianBlur(glow_layer, (15, 15), max(2.5, edge_softness * 1.5))
                canvas = cv2.addWeighted(canvas, 1.0, glow_blur, 0.7, 0)

                # Desenha o núcleo claro
                core_color = (
                    int(min(255, color_bgr[0] + 190)),
                    int(min(255, color_bgr[1] + 190)),
                    int(min(255, color_bgr[2] + 190))
                )
                for stroke in adjusted_strokes:
                    if len(stroke) >= 2:
                        pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(canvas, [pts], False, core_color, core_width, cv2.LINE_AA)

            else:
                # Canetinha Sólida com antialiasing sub-pixel e suavidade de borda
                for stroke in adjusted_strokes:
                    if len(stroke) >= 2 and config.TOUCH_TRANSPORT == "injector":
                        # Gesto contínuo: sem levantar o dedo nem descartar trechos curtos
                        pts = np.array(stroke, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(canvas, [pts], False, color_bgr, width_px, cv2.LINE_AA)
                    elif len(stroke) >= 2:
                        for start, end in zip(stroke, stroke[1:]):
                            if np.hypot(end[0] - start[0], end[1] - start[1]) >= 2.5:
                                cv2.line(canvas, start, end, color_bgr, width_px, cv2.LINE_AA)

            cumulative_buffers.append(canvas.copy())

        return canvas, cumulative_buffers

    def render_motion_plan(self, canvas_shape: Tuple[int, int, int],
                           motion_strokes: List[Any],
                           background_bgr: Tuple[int, int, int] = (22, 22, 22),
                           transform: Any = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Fase 5.1: Simula rigorosamente o Motion Plan real (MotionStroke[]) - a mesma entrada do executor.
        Usa o BrushResponseModel para predizer o resultado físico no device.
        """
        from ..device.brush_response import BrushResponseModel
        h, w = canvas_shape[:2]
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas[:, :] = background_bgr

        cumulative_buffers = []

        for stroke in motion_strokes:
            events = stroke.events if hasattr(stroke, "events") else []
            if not events or len(events) < 2:
                continue

            tool = getattr(stroke, "tool", "solid")
            thickness_mode = getattr(stroke, "thickness_mode", "med")
            color_rgb = getattr(stroke, "color_rgb", (255, 255, 255))
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))

            # Calcula velocidade média do traço para o modelo físico
            duration_s = max(0.01, float(stroke.duration_ms) / 1000.0) if hasattr(stroke, "duration_ms") else 0.1
            points = [(e.x, e.y) for e in events]
            pts_arr = np.array(points, dtype=np.float32)
            diffs = pts_arr[1:] - pts_arr[:-1]
            length_px = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
            velocity_px_s = length_px / duration_s

            # Resposta física do pincel
            response = BrushResponseModel.predict_response(
                tool=tool,
                slider_thickness=thickness_mode,
                velocity_px_s=velocity_px_s,
                passes=1,
                color_rgb=color_rgb,
                background_bgr=background_bgr
            )
            width_px = max(1, int(round(response.predicted_width)))

            pts = pts_arr.astype(np.int32).reshape((-1, 1, 2))

            if tool == "marker":
                stroke_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.polylines(stroke_mask, [pts], False, 255, width_px, cv2.LINE_AA)
                alpha_map = (stroke_mask.astype(np.float32) / 255.0) * response.opacity
                for c in range(3):
                    canvas[:, :, c] = np.clip(
                        canvas[:, :, c].astype(np.float32) * (1.0 - alpha_map) + color_bgr[c] * alpha_map,
                        0.0, 255.0
                    ).astype(np.uint8)

            elif tool == "neon" and response.neon_components:
                neon = response.neon_components
                halo_w = max(width_px + 2, int(round(neon["halo_radius_px"] * 2.0)))
                core_w = max(1, int(round(neon["core_radius_px"] * 2.0)))

                glow_layer = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.polylines(glow_layer, [pts], False, color_bgr, halo_w, cv2.LINE_AA)
                glow_blur = cv2.GaussianBlur(glow_layer, (15, 15), 3.0)
                canvas = cv2.addWeighted(canvas, 1.0, glow_blur, 0.7, 0)
                cv2.polylines(canvas, [pts], False, neon["core_color_bgr"], core_w, cv2.LINE_AA)

            else:
                cv2.polylines(canvas, [pts], False, color_bgr, width_px, cv2.LINE_AA)

        cumulative_buffers.append(canvas.copy())
        return canvas, cumulative_buffers

    @staticmethod
    def compute_triplet_errors(original_img: np.ndarray,
                               target_executable: np.ndarray,
                               predicted_device: np.ndarray,
                               observed_device: Optional[np.ndarray] = None,
                               target_prediction_error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fase 5.3: Calcula a tríade de erros fundamentais do sistema:
          1. Original <-> Target (Erro de Planejamento)
          2. Target <-> Predição (Erro de Motion Compiler / Simulador)
          3. Predição <-> Real (Erro de Calibração / Execução Física)
        """
        from ..feedback.error_map import ErrorMapEngine
        h, w = target_executable.shape[:2]
        orig_fitted = cv2.resize(original_img, (w, h)) if original_img.shape[:2] != (h, w) else original_img

        _, err_orig_target = ErrorMapEngine.compute_error_map(orig_fitted, target_executable)
        if target_prediction_error is not None:
            # Métricas target x predição já calculadas pelo chamador (evita recomputar o mapa de erro)
            err_target_pred = target_prediction_error
        else:
            _, err_target_pred = ErrorMapEngine.compute_error_map(target_executable, predicted_device)

        err_pred_real = None
        if observed_device is not None:
            obs_fitted = cv2.resize(observed_device, (w, h)) if observed_device.shape[:2] != (h, w) else observed_device
            _, err_pred_real = ErrorMapEngine.compute_error_map(predicted_device, obs_fitted)

        # Interpretação diagnóstica automática (Fase 5.3)
        planning_issue = err_orig_target["mean_delta_e"] > 15.0 or err_orig_target.get("p95_delta_e", 0.0) > 30.0
        motion_sim_issue = err_target_pred["mean_delta_e"] > 10.0 or err_target_pred.get("p95_delta_e", 0.0) > 25.0
        calib_issue = (err_pred_real["mean_delta_e"] > 10.0 or err_pred_real.get("p95_delta_e", 0.0) > 25.0) if err_pred_real else False

        return {
            "error_original_target": err_orig_target,
            "error_target_prediction": err_target_pred,
            "error_prediction_real": err_pred_real,
            "diagnosis": {
                "planning": "high_planning_error" if planning_issue else "ok",
                "motion_compiler_simulator": "high_motion_simulator_divergence" if motion_sim_issue else "ok",
                "physical_calibration": "high_device_divergence" if calib_issue else "ok"
            }
        }

