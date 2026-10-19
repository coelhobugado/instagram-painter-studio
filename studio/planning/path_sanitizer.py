"""
Sanitizador de Caminhos e Validador Estrito de Limites de Segurança (Safe Bounds).
Implementa a Seção 5 do Plano de Correções (P0.2 — STRICT Sanitizer sem clamp silencioso).
"""
import math
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
from .. import config

class SanitizerMode(str, Enum):
    STRICT = "STRICT"
    DEBUG_CLAMP = "DEBUG_CLAMP"

class PathSanitizer:
    last_stats: Dict[str, int] = {
        "total_points": 0,
        "valid_points": 0,
        "rejected_points": 0,
        "clamped_points": 0
    }

    @classmethod
    def sanitize_stroke(cls, points: List[Tuple[int, int]],
                        x_min: int = config.CANVAS_X_MIN,
                        x_max: int = config.CANVAS_X_MAX,
                        y_min: int = config.CANVAS_Y_MIN,
                        y_max: int = config.CANVAS_Y_MAX,
                        max_step_distance: float = 1900.0,
                        mode: SanitizerMode = SanitizerMode.STRICT) -> List[Tuple[int, int]]:
        """
        Valida e sanitiza estritamente cada ponto de uma pincelada antes de ir para o Android.
        No modo STRICT (produção): unsafe point = reject (ZERO clamped points).
        No modo DEBUG_CLAMP: unsafe point = clamp (apenas para diagnósticos de limites).
        """
        if not points or len(points) < 2:
            return []

        segments = cls.sanitize_stroke_split(
            points, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            max_step_distance=max_step_distance, mode=mode
        )
        if not segments:
            return []
        # Retorna o segmento mais expressivo para compatibilidade de assinatura
        return max(segments, key=len)

    @classmethod
    def sanitize_stroke_split(cls, points: List[Tuple[int, int]],
                              x_min: int = config.CANVAS_X_MIN,
                              x_max: int = config.CANVAS_X_MAX,
                              y_min: int = config.CANVAS_Y_MIN,
                              y_max: int = config.CANVAS_Y_MAX,
                              max_step_distance: float = 1900.0,
                              mode: SanitizerMode = SanitizerMode.STRICT) -> List[List[Tuple[int, int]]]:
        """
        Valida e sanitiza cada ponto de uma pincelada com SPLIT estrito (Fase 1.4).
        Se um ponto intermediário for inválido (NaN, fora de limites, salto abrupto),
        divide a pincelada: stroke_1 termina antes da anomalia e stroke_2 inicia após,
        garantindo ZERO conexões artificiais geradas por sanitização.
        """
        if not points or len(points) < 2:
            return []

        segments: List[List[Tuple[int, int]]] = []
        current_segment: List[Tuple[int, int]] = []

        for i, (px, py) in enumerate(points):
            cls.last_stats["total_points"] += 1
            # 1. Validação de NaN e Inf
            if math.isnan(px) or math.isnan(py) or math.isinf(px) or math.isinf(py):
                cls.last_stats["rejected_points"] += 1
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = []
                continue

            # 2. Verificação de limites
            is_inside = (x_min <= px <= x_max) and (y_min <= py <= y_max)
            if not is_inside:
                if mode == SanitizerMode.STRICT:
                    cls.last_stats["rejected_points"] += 1
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = []
                    continue
                else:
                    sx = int(max(x_min, min(x_max, round(px))))
                    sy = int(max(y_min, min(y_max, round(py))))
                    cls.last_stats["clamped_points"] += 1
            else:
                sx = int(round(px))
                sy = int(round(py))

            # 3. Verificação de salto abrupto anômalo entre nós adjacentes de um mesmo traço
            if current_segment:
                prev_x, prev_y = current_segment[-1]
                dist = math.hypot(sx - prev_x, sy - prev_y)
                if dist > max_step_distance:
                    cls.last_stats["rejected_points"] += 1
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = []
                    continue

            current_segment.append((sx, sy))
            cls.last_stats["valid_points"] += 1

        if len(current_segment) >= 2:
            segments.append(current_segment)

        # Filtra toques pontuais muito curtos (< 2.5px)
        valid_segments: List[List[Tuple[int, int]]] = []
        for seg in segments:
            if len(seg) == 2:
                dist = math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1])
                if dist < 2.5:
                    cls.last_stats["rejected_points"] += 2
                    continue
            valid_segments.append(seg)

        return valid_segments

    @classmethod
    def sanitize_all_strokes(cls, strokes_list: List[List[Tuple[int, int]]],
                             x_min: int = config.CANVAS_X_MIN,
                             x_max: int = config.CANVAS_X_MAX,
                             y_min: int = config.CANVAS_Y_MIN,
                             y_max: int = config.CANVAS_Y_MAX,
                             mode: SanitizerMode = SanitizerMode.STRICT) -> List[List[Tuple[int, int]]]:
        """Aplica sanitização estrita a todas as pinceladas do plano com divisão de sub-traços."""
        clean_strokes: List[List[Tuple[int, int]]] = []
        for s in strokes_list:
            splits = cls.sanitize_stroke_split(s, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, mode=mode)
            for sub_s in splits:
                if len(sub_s) >= 2:
                    clean_strokes.append(sub_s)
        return clean_strokes

    @classmethod
    def sanitize_strokes_detailed(cls, strokes_list: List[List[Tuple[int, int]]],
                                 x_min: int = config.CANVAS_X_MIN,
                                 x_max: int = config.CANVAS_X_MAX,
                                 y_min: int = config.CANVAS_Y_MIN,
                                 y_max: int = config.CANVAS_Y_MAX,
                                 mode: SanitizerMode = SanitizerMode.STRICT) -> Tuple[List[List[Tuple[int, int]]], int, List[str]]:
        """
        Aplica sanitização estrita retornando traços aceitos, contagem de rejeitados e motivos detalhados (P1.4).
        """
        clean_strokes: List[List[Tuple[int, int]]] = []
        rejected_count = 0
        reasons: List[str] = []

        for idx, s in enumerate(strokes_list):
            if not s or len(s) < 2:
                rejected_count += 1
                reasons.append(f"Traço {idx}: menos de 2 pontos (vazio ou degenerado)")
                continue
            splits = cls.sanitize_stroke_split(s, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, mode=mode)
            valid_splits = [sub_s for sub_s in splits if len(sub_s) >= 2]
            if not valid_splits:
                rejected_count += 1
                reasons.append(f"Traço {idx}: todos os nós fora dos limites ou NaN")
            else:
                clean_strokes.extend(valid_splits)

        return clean_strokes, rejected_count, reasons

    @classmethod
    def reset_stats(cls) -> None:
        cls.last_stats = {
            "total_points": 0,
            "valid_points": 0,
            "rejected_points": 0,
            "clamped_points": 0
        }
