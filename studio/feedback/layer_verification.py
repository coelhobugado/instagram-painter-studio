# -*- coding: utf-8 -*-
"""
Conferência pós-camada compartilhada entre ExecutionWorker (GUI) e
ExecutionOrchestrator (MCP), para que os dois caminhos usem os mesmos limites.
"""
from typing import Any, Dict, Tuple

import cv2
import numpy as np

MAX_LAYER_DELTA_E = 35.0
MIN_COVERAGE_AGREEMENT = 0.30
MIN_EDGE_MATCH_RATE = 0.20
THIN_LAYER_MAX_WIDTH_PX = 28.0
# Camadas que preenchem áreas com pincel fino (a largura medida é a da área).
AREA_FILL_PURPOSES = {"line_art_fill"}


def verify_layer_fidelity(aligned_live: np.ndarray,
                          expected_img: np.ndarray,
                          layer_data: Dict[str, Any],
                          check_thickness: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Compara a captura alinhada com a prévia acumulada da camada:
    cor média (ΔE Lab), cobertura dos traços, alinhamento de bordas e,
    em camadas finas, a espessura física do pincel.
    Retorna (aprovado, motivo, métricas).
    """
    strokes = layer_data.get("strokes", [])
    if not strokes:
        return False, "Verificação indeterminada: camada sem traços", {"status": "indeterminate"}
    if aligned_live is None or expected_img is None:
        return False, "Verificação indeterminada: captura ao vivo ou imagem esperada ausente", {"status": "indeterminate"}

    h, w = expected_img.shape[:2]
    layer_mask = np.zeros((h, w), dtype=np.uint8)
    for s in strokes:
        if len(s) >= 2:
            pts = np.array(s, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(layer_mask, [pts], False, 255, 5, cv2.LINE_AA)

    if np.count_nonzero(layer_mask) < 25:
        return False, "Verificação indeterminada: máscara de traços pequena demais", {"status": "indeterminate"}

    # 1. Fotometria / cor média
    on_mask = layer_mask > 0
    mean_expected_bgr = np.mean(expected_img[on_mask], axis=0)
    mean_live_bgr = np.mean(aligned_live[on_mask], axis=0)
    exp_lab = cv2.cvtColor(np.uint8([[mean_expected_bgr]]), cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)
    live_lab = cv2.cvtColor(np.uint8([[mean_live_bgr]]), cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)
    delta_e = float(np.sqrt(np.sum((exp_lab - live_lab) ** 2)))

    exp_rgb = tuple(int(round(v)) for v in mean_expected_bgr[::-1])
    live_rgb = tuple(int(round(v)) for v in mean_live_bgr[::-1])
    metrics: Dict[str, Any] = {
        "mean_delta_e": round(delta_e, 2),
        "expected_rgb": exp_rgb,
        "observed_rgb": live_rgb,
    }
    if delta_e > MAX_LAYER_DELTA_E:
        return False, (f"Divergência de cor na camada (Esperado RGB {exp_rgb}, "
                       f"Lido RGB {live_rgb}, ΔE={delta_e:.1f})"), metrics

    # 2. Cobertura espacial e traços ausentes
    gray_diff = cv2.cvtColor(cv2.absdiff(aligned_live, expected_img), cv2.COLOR_BGR2GRAY)
    mask_pixels_count = int(np.count_nonzero(on_mask))
    coverage_agreement = np.count_nonzero((gray_diff < 55) & on_mask) / max(1, mask_pixels_count)
    metrics["coverage_agreement"] = round(float(coverage_agreement), 3)
    if coverage_agreement < MIN_COVERAGE_AGREEMENT and mask_pixels_count > 200:
        return False, (f"Falha de cobertura ou traços ausentes na tela (concordância espacial "
                       f"{coverage_agreement * 100:.1f}% abaixo do mínimo)"), metrics

    # 3. Alinhamento de bordas / deslocamento
    rx, ry, rw, rh = cv2.boundingRect(layer_mask)
    pad = 8
    x0, y0 = max(0, rx - pad), max(0, ry - pad)
    x1, y1 = min(w, rx + rw + pad), min(h, ry + rh + pad)
    exp_roi = expected_img[y0:y1, x0:x1]
    live_roi = aligned_live[y0:y1, x0:x1]
    edges_exp = cv2.Canny(exp_roi, 50, 150)
    edges_live = cv2.Canny(live_roi, 50, 150)
    num_exp_edges = np.count_nonzero(edges_exp)
    num_live_edges = np.count_nonzero(edges_live)

    if num_exp_edges > 40:
        edges_exp_tol = cv2.dilate(edges_exp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        edge_match_rate = np.count_nonzero((edges_live > 0) & (edges_exp_tol > 0)) / max(1, num_live_edges)
        metrics["edge_match_rate"] = round(float(edge_match_rate), 3)
        if edge_match_rate < MIN_EDGE_MATCH_RATE and num_live_edges > 30:
            return False, (f"Deslocamento ou deformação espacial detectada (alinhamento de bordas "
                           f"{edge_match_rate * 100:.1f}% abaixo do tolerável)"), metrics

    # 4. Espessura física do pincel em camadas finas. Camadas de preenchimento
    # (ex.: pupilas do line-art) usam o pincel fino para cobrir áreas largas de propósito.
    is_area_fill = layer_data.get("purpose") in AREA_FILL_PURPOSES
    if check_thickness and not is_area_fill and layer_data.get("thickness", "min") == "min" and num_live_edges > 40:
        mask_roi = layer_mask[y0:y1, x0:x1]
        dist_to_mean = np.linalg.norm(live_roi.astype(np.float32) - mean_live_bgr, axis=2)
        dilated_mask = cv2.dilate(mask_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        stroke_pixels = ((dist_to_mean < 45.0) & (dilated_mask > 0)).astype(np.uint8) * 255
        if np.any(stroke_pixels == 0) and np.any(stroke_pixels > 0):
            dist_map = cv2.distanceTransform(stroke_pixels, cv2.DIST_L2, 3)
            measured_w = float(np.max(dist_map)) * 2.0 if np.count_nonzero(dist_map) > 0 else 0.0
            metrics["measured_width_px"] = round(measured_w, 1)
            if THIN_LAYER_MAX_WIDTH_PX < measured_w < 500.0:
                return False, (f"Espessura física divergente: traço grosso ({measured_w:.1f}px) "
                               "detectado em camada fina"), metrics

    return True, "OK", metrics
