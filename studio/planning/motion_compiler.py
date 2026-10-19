# -*- coding: utf-8 -*-
"""
Motion Compiler — Compilador de Movimento para Touchscreen (FASE 2 do Roadmap para o Ápice).
Separa rigorosamente a geometria ideal contínua da trajetória executável pelo digitalizador de toque.

Pipeline de Compilação:
  1. Geometria contínua float (GeometricStroke)
  2. Transformação canônica para coordenadas de tela
  3. Reamostragem por comprimento de arco com curvatura adaptativa
  4. Quantização inteira única (Round)
  5. Deduplicação de eventos idênticos consecutivos
  6. Remoção de colineares redundantes preservando gaps máximos
  7. Atribuição de timing explícito monotônico determinístico (DOWN, MOVE..., UP)
"""
import math
import hashlib
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

from ..core.models import CanvasTransform

@dataclass
class StrokeIntent:
    """Representa a intenção de desenho por trás de uma pincelada."""
    region_id: str = "default_region"
    purpose: str = "contour"
    target_width: float = 3.2
    priority: float = 1.0
    color_target: Tuple[int, int, int] = (0, 0, 0)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GeometricStroke:
    """Representação contínua ideal (float) no espaço Canvas-Local."""
    points: List[Tuple[float, float]]
    stroke_type: str = "polyline"  # "polyline", "spline", "bezier"
    intent: Optional[StrokeIntent] = None

class TouchAction(str, Enum):
    DOWN = "DOWN"
    MOVE = "MOVE"
    UP = "UP"

@dataclass(frozen=True)
class MotionEvent:
    """Evento atômico de toque com posição e timestamp explícito."""
    action: TouchAction
    x: int
    y: int
    timestamp_ms: int
    pressure: float = 1.0

@dataclass
class MotionStroke:
    """Traço final compilado e executável pelo touchscreen."""
    events: List[MotionEvent]
    intent: Optional[StrokeIntent] = None
    thickness_mode: str = "med"
    tool: str = "solid"
    color_rgb: Tuple[int, int, int] = (0, 0, 0)
    touch_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def start_time_ms(self) -> int:
        return self.events[0].timestamp_ms if self.events else 0

    @property
    def end_time_ms(self) -> int:
        return self.events[-1].timestamp_ms if self.events else 0

    @property
    def duration_ms(self) -> int:
        return self.end_time_ms - self.start_time_ms

    @property
    def points(self) -> List[Tuple[int, int]]:
        return [(e.x, e.y) for e in self.events]

    def to_points_list(self) -> List[Tuple[int, int]]:
        return [(e.x, e.y) for e in self.events]

@dataclass
class ExecutionBatch:
    """Lote de MotionStrokes despachado atomicamente."""
    batch_id: int
    strokes: List[MotionStroke]
    estimated_duration_ms: int

@dataclass
class CompiledMotionPlan:
    """Plano de Movimento Compilado e Determinístico pronto para execução ou simulação."""
    motion_strokes: List[MotionStroke]
    batches: List[ExecutionBatch]
    total_events: int
    total_strokes: int
    total_duration_ms: int
    sample_rate_hz: int
    plan_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class MotionCompiler:
    """
    Compilador de Movimento responsável por traduzir geometria pura em eventos de touchscreen
    com taxa de amostragem configurável e rigorosa preservação temporal.
    """

    @classmethod
    def resample_by_arc_length(cls, points: List[Tuple[float, float]],
                               max_gap: float = 2.0,
                               adaptive_curvature: bool = True) -> List[Tuple[float, float]]:
        """
        Reamostra a curva contínua float para garantir que a distância euclidiana entre
        pontos sucessivos nunca exceda max_gap. Em trechos de alta curvatura (curvas fechadas),
        reduz dinamicamente o passo para manter a suavidade anatômica.
        """
        if not points or len(points) < 2:
            return list(points)

        resampled: List[Tuple[float, float]] = [points[0]]

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_len = math.hypot(dx, dy)

            if seg_len < 1e-5:
                continue

            # Avalia curvatura local usando vizinhos se disponível
            current_max_gap = max_gap
            if adaptive_curvature and 0 < i < len(points) - 1:
                p0 = points[i - 1]
                v_in = (p1[0] - p0[0], p1[1] - p0[1])
                v_out = (dx, dy)
                mag_in = math.hypot(v_in[0], v_in[1])
                mag_out = seg_len
                if mag_in > 1e-4 and mag_out > 1e-4:
                    cos_theta = (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (mag_in * mag_out)
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    # Ângulo agudo -> diminui gap para aumentar amostragem na curva
                    if cos_theta < 0.85:
                        current_max_gap = max(0.8, max_gap * 0.5)

            num_sub = max(1, int(math.ceil(seg_len / current_max_gap)))
            for s in range(1, num_sub + 1):
                t = float(s) / float(num_sub)
                nx = p1[0] + t * dx
                ny = p1[1] + t * dy
                resampled.append((nx, ny))

        return resampled

    @classmethod
    def quantize_and_deduplicate(cls, screen_points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Fase 2.3 & 2.4: Quantização inteira única seguida de eliminação de duplicatas consecutivas.
        """
        if not screen_points:
            return []

        quantized: List[Tuple[int, int]] = []
        last_pt: Optional[Tuple[int, int]] = None

        for px, py in screen_points:
            ix = int(round(px))
            iy = int(round(py))
            curr = (ix, iy)
            if curr != last_pt:
                quantized.append(curr)
                last_pt = curr

        return quantized

    @classmethod
    def remove_redundant_collinears(cls, points: List[Tuple[int, int]], max_gap: float = 4.0) -> List[Tuple[int, int]]:
        """
        Fase 2.5: Remove nós intermediários colineares desde que o gap euclidiano
        resultante entre o ponto anterior e o próximo não ultrapasse max_gap.
        """
        if len(points) <= 2:
            return points

        filtered: List[Tuple[int, int]] = [points[0]]

        for i in range(1, len(points) - 1):
            p_prev = filtered[-1]
            p_curr = points[i]
            p_next = points[i + 1]

            # Vetores
            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]

            # Área do paralelogramo (produto vetorial 2D)
            cross = abs(dx1 * dy2 - dy1 * dx2)
            dist_prev_next = math.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])

            # Se colinear (área zero) E gap tolerável, pula o ponto intermediário
            if cross == 0 and dist_prev_next <= max_gap:
                continue

            filtered.append(p_curr)

        filtered.append(points[-1])
        return filtered

    @classmethod
    def compile_stroke(cls, geom: GeometricStroke,
                       transform: CanvasTransform,
                       start_time_ms: int = 0,
                       sample_interval_ms: int = 8,
                       max_gap: float = 2.0,
                       thickness_mode: str = "med",
                       tool: str = "solid",
                       color_rgb: Tuple[int, int, int] = (0, 0, 0),
                       touch_info: Dict[str, Any] = None,
                       air_gap_ms: int = 0) -> Tuple[MotionStroke, int]:
        """
        Compila uma pincelada geométrica contínua em eventos reais de touchscreen com timing monotônico.
        Retorna (MotionStroke, next_time_ms).
        """
        raw_pts = geom.points
        if not raw_pts or len(raw_pts) < 2:
            return MotionStroke([], intent=geom.intent), start_time_ms

        # 1. Transformação float para tela
        screen_float = [
            (float(pt[0] + transform.offset_x), float(pt[1] + transform.offset_y))
            for pt in raw_pts
        ]

        # 2. Reamostragem por comprimento de arco
        resampled = cls.resample_by_arc_length(screen_float, max_gap=max_gap, adaptive_curvature=True)

        # 3. Quantização única e deduplicação
        quantized = cls.quantize_and_deduplicate(resampled)

        # 4. Remoção de colineares redundantes
        final_points = cls.remove_redundant_collinears(quantized, max_gap=max_gap * 2.0)
        if len(final_points) < 2:
            final_points = quantized

        # 5. Geração de eventos com timestamps monotônicos explícitos
        events: List[MotionEvent] = []
        curr_t = start_time_ms

        # DOWN
        events.append(MotionEvent(TouchAction.DOWN, final_points[0][0], final_points[0][1], curr_t, pressure=1.0))

        # MOVEs
        for pt in final_points[1:-1]:
            curr_t += sample_interval_ms
            events.append(MotionEvent(TouchAction.MOVE, pt[0], pt[1], curr_t, pressure=1.0))

        # UP
        curr_t += sample_interval_ms
        events.append(MotionEvent(TouchAction.UP, final_points[-1][0], final_points[-1][1], curr_t, pressure=0.0))

        # Intervalo após levantar o dedo (zero delay se air_gap_ms = 0)
        next_t = curr_t + air_gap_ms

        m_stroke = MotionStroke(
            events=events,
            intent=geom.intent,
            thickness_mode=thickness_mode,
            tool=tool,
            color_rgb=color_rgb,
            touch_info=touch_info or {}
        )
        return m_stroke, next_t

    @classmethod
    def compile_plan(cls, plan_layers: List[Dict[str, Any]],
                     transform: Optional[CanvasTransform] = None,
                     target_hz: int = 120,
                     max_gap: float = 2.0,
                     batch_size: int = 8,
                     sample_rate_hz: Optional[int] = None,
                     adaptive: bool = True) -> CompiledMotionPlan:
        """
        Compila todas as camadas de desenho em um CompiledMotionPlan determinístico com hash estável.
        Suporta modo adaptativo dinâmico (batch e frequência calibrados por nível de risco visual).
        """
        if sample_rate_hz is not None:
            target_hz = sample_rate_hz

        if transform is None:
            transform = CanvasTransform(offset_x=0, offset_y=0, scale=1.0)

        all_motion_strokes: List[MotionStroke] = []
        batches: List[ExecutionBatch] = []
        current_time_ms = 0
        batch_counter = 0

        for layer in plan_layers:
            strokes = layer.get("strokes", [])
            tool = layer.get("brush", "solid")
            thickness = layer.get("thickness", "med")
            color_rgb = layer.get("color_rgb", (0, 0, 0))
            touch_info = layer.get("touch_info", {})
            purpose = layer.get("purpose", "fill")

            # Cálculo de Frequência e Batch Adaptativos
            if adaptive:
                if purpose in ("contour", "line_art_centerline", "colored_detail") or thickness == "min":
                    eff_hz = 90
                    eff_batch = 4
                elif purpose in ("color_mass", "fill", "background"):
                    eff_hz = 150
                    eff_batch = 16
                else:
                    eff_hz = target_hz
                    eff_batch = batch_size
            else:
                eff_hz = target_hz
                eff_batch = batch_size

            sample_interval_ms = max(1, int(round(1000.0 / float(eff_hz))))
            layer_strokes: List[MotionStroke] = []

            for s_idx, s in enumerate(strokes):
                if not s or len(s) < 2:
                    continue
                intent = StrokeIntent(
                    region_id=f"layer_{purpose}_{s_idx}",
                    purpose=purpose,
                    target_width=3.2 if thickness == "min" else (12.0 if thickness == "med" else 28.0),
                    priority=1.0,
                    color_target=color_rgb
                )
                geom = GeometricStroke(points=[(float(p[0]), float(p[1])) for p in s], intent=intent)
                m_stroke, next_t = cls.compile_stroke(
                    geom, transform=transform,
                    start_time_ms=current_time_ms,
                    sample_interval_ms=sample_interval_ms,
                    max_gap=max_gap,
                    thickness_mode=thickness,
                    tool=tool,
                    color_rgb=color_rgb,
                    touch_info=touch_info
                )
                layer_strokes.append(m_stroke)
                all_motion_strokes.append(m_stroke)
                current_time_ms = next_t

            # Agrupa os traços da camada com o batch size adaptativo
            for start in range(0, len(layer_strokes), eff_batch):
                chunk = layer_strokes[start:start + eff_batch]
                b_dur = chunk[-1].end_time_ms - chunk[0].start_time_ms if chunk else 0
                batches.append(ExecutionBatch(
                    batch_id=batch_counter,
                    strokes=chunk,
                    estimated_duration_ms=b_dur
                ))
                batch_counter += 1

        total_events = sum(len(s.events) for s in all_motion_strokes)
        total_duration = current_time_ms

        # Hash estável determinístico baseado em todos os eventos
        hasher = hashlib.sha256()
        hasher.update(f"HZ:{target_hz};GAP:{max_gap};STROKES:{len(all_motion_strokes)};".encode("utf-8"))
        for s in all_motion_strokes:
            for e in s.events:
                hasher.update(f"{e.action}:{e.x},{e.y}@{e.timestamp_ms};".encode("utf-8"))

        plan_hash = hasher.hexdigest()[:16]

        return CompiledMotionPlan(
            motion_strokes=all_motion_strokes,
            batches=batches,
            total_events=total_events,
            total_strokes=len(all_motion_strokes),
            total_duration_ms=total_duration,
            sample_rate_hz=target_hz,
            plan_hash=plan_hash,
            metadata={
                "batch_count": len(batches),
                "execution_channel": "adb_persistent_stream",
                "physical_dispatch_mode": "input_swipe" if target_hz <= 60 else "interpolated_stream",
                "sample_rate_hz": target_hz,
                "pure_motion_duration_s": round(total_duration / 1000.0, 2),
                "estimated_adb_runtime_s": round(total_duration / 1000.0 + len(all_motion_strokes) * 0.045 + len(plan_layers) * 1.0, 2)
            }
        )
