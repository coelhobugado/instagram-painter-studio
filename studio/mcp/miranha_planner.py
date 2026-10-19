# -*- coding: utf-8 -*-
"""
Gerador de Arte Especializado para o Homem-Aranha (personagem.jpg).
Aplica isolamento semântico de fundo externo, preenchimento de massas vivas
(vermelho, azul, olhos brancos) e vetorização da teia e contornos em preto puro.
"""
import cv2
import numpy as np
import os
from typing import Dict, Any, List, Tuple

from studio.planning.skeleton_graph import zhang_suen_thinning, SkeletonGraph
from studio.planning.fill_strategies import FillStrategyEngine
from studio.planning.path_sanitizer import PathSanitizer

def create_miranha_artwork_plan(image_path: str, canvas_w: int = 1080, canvas_h: int = 2400, target_size: int = 800) -> Dict[str, Any]:
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        raise FileNotFoundError(f"Imagem {image_path} não encontrada.")

    # 1. Redimensiona preservando proporção para o tamanho alvo no canvas
    orig_h, orig_w = raw_img.shape[:2]
    scale = target_size / max(orig_h, orig_w)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    fit_img = cv2.resize(raw_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centraliza na área útil segura da tela (Y entre 550 e 1850)
    off_x = (canvas_w - new_w) // 2
    off_y = 650 # Posição nobre centralizada

    # 2. Segmentação de Cores e Fundo Externo
    gray = cv2.cvtColor(fit_img, cv2.COLOR_BGR2GRAY)
    _, white_thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    
    # Flood-fill a partir dos cantos para isolar estritamente o fundo externo transparente
    flood_mask = np.zeros((new_h + 2, new_w + 2), dtype=np.uint8)
    cv2.floodFill(white_thresh.copy(), flood_mask, (0, 0), 128)
    cv2.floodFill(white_thresh.copy(), flood_mask, (new_w - 1, 0), 128)
    cv2.floodFill(white_thresh.copy(), flood_mask, (0, new_h - 1), 128)
    cv2.floodFill(white_thresh.copy(), flood_mask, (new_w - 1, new_h - 1), 128)
    outer_bg = (flood_mask[1:-1, 1:-1] == 1)

    # Máscaras das regiões anatômicas
    eyes_mask = ((white_thresh == 255) & (~outer_bg)).astype(np.uint8) * 255
    black_mask = ((gray < 75) & (~outer_bg)).astype(np.uint8) * 255

    # Regiões do corpo (excluindo fundo, olhos brancos e contornos pretos)
    body_mask = (~outer_bg & (black_mask == 0) & (eyes_mask == 0))

    # Vermelho vibrante: Canal Vermelho claramente dominante
    red_mask = (body_mask & (fit_img[:, :, 2] > fit_img[:, :, 0] + 30)).astype(np.uint8) * 255

    # Azul Índigo / Calça e detalhes escuros do traje
    blue_mask = (body_mask & (red_mask == 0)).astype(np.uint8) * 255

    # Refina bordas (pequena dilatação para fechar microburacos)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k3)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, k3)
    eyes_mask = cv2.morphologyEx(eyes_mask, cv2.MORPH_CLOSE, k3)

    # Remove áreas pretas das bases de cor para evitar pintura desnecessária sob o preto
    red_clean = cv2.bitwise_and(red_mask, cv2.bitwise_not(cv2.dilate(black_mask, k3)))
    blue_clean = cv2.bitwise_and(blue_mask, cv2.bitwise_not(cv2.dilate(black_mask, k3)))

    def mask_to_screen_strokes(mask, spacing=10):
        # Gera traços de preenchimento e mapeia para coordenadas de tela
        h_m, w_m = mask.shape
        bbox = (0, 0, w_m, h_m)
        local_strokes = FillStrategyEngine._horizontal_scanline(mask, bbox, spacing=spacing)
        scr_strokes = []
        for s in local_strokes:
            scr_s = [(pt[0] + off_x, pt[1] + off_y) for pt in s]
            scr_strokes.append(scr_s)
        return scr_strokes

    layers = []

    # =========================================================================
    # CAMADA 1: O VERMELHO DO TRAJE (CABEÇA, MÃOS, PEITO E PÉS)
    # =========================================================================
    red_strokes = mask_to_screen_strokes(red_clean, spacing=10)
    layers.append({
        "name": "1. Vermelho do Traje (Cabeça, Luvas e Botas)",
        "color_name": "vermelho",
        "color_info": {"name": "Vermelho Coral", "target_x": 786, "target_y": 2185, "is_long_press": False},
        "thickness": "med",
        "strokes": red_strokes
    })

    # =========================================================================
    # CAMADA 2: O AZUL DO CORPO E DAS PERNAS
    # =========================================================================
    blue_strokes = mask_to_screen_strokes(blue_clean, spacing=9)
    layers.append({
        "name": "2. Azul do Traje (Tronco e Calça)",
        "color_name": "azul",
        "color_info": {"name": "Azul Claro", "target_x": 486, "target_y": 2185, "is_long_press": False},
        "thickness": "med",
        "strokes": blue_strokes
    })

    # =========================================================================
    # CAMADA 3: AS LENTES BRANCAS DOS OLHOS
    # =========================================================================
    eyes_strokes = mask_to_screen_strokes(eyes_mask, spacing=8)
    layers.append({
        "name": "3. Lentes Brancas dos Olhos da Máscara",
        "color_name": "branco",
        "color_info": {"name": "Branco", "target_x": 411, "target_y": 2185, "is_long_press": False},
        "thickness": "med",
        "strokes": eyes_strokes
    })

    # =========================================================================
    # CAMADA 4: CONTORNOS PRETOS E TEIA DE ARANHA DETALHADA
    # =========================================================================
    # Esqueletização para traçar o centro perfeito de cada linha e teia
    skel = zhang_suen_thinning(black_mask)
    sg = SkeletonGraph(skel)
    segments = sg.extract_continuous_strokes(approx_epsilon=1.0, min_length=5.0)
    
    # Contorno externo das grandes bordas
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contour_strokes = []
    for cnt in contours:
        if cv2.arcLength(cnt, True) > 25:
            pts = [(int(p[0][0] + off_x), int(p[0][1] + off_y)) for p in cnt]
            if len(pts) >= 3:
                pts.append(pts[0]) # fecha anel
                contour_strokes.append(pts)

    skel_strokes = []
    for seg in segments:
        if len(seg) >= 2:
            s_scr = [(pt[0] + off_x, pt[1] + off_y) for pt in seg]
            skel_strokes.append(s_scr)

    all_black_strokes = contour_strokes + skel_strokes
    layers.append({
        "name": "4. Contornos Pretos e Teia de Aranha",
        "color_name": "preto",
        "color_info": {"name": "Preto Puro", "is_black": True},
        "thickness": "min",
        "strokes": all_black_strokes
    })

    # Renderiza a simulação gráfica completa
    sim_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240
    for l in layers:
        col_rgb = (0, 0, 0)
        cname = l["color_name"]
        if cname == "vermelho": col_rgb = (40, 40, 220)
        elif cname == "azul": col_rgb = (180, 80, 40)
        elif cname == "branco": col_rgb = (255, 255, 255)
        elif cname == "preto": col_rgb = (20, 20, 20)
        
        w_px = 6 if l["thickness"] == "med" else 3
        for s in l["strokes"]:
            if len(s) >= 2:
                pts = np.array(s, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(sim_canvas, [pts], False, col_rgb, w_px, cv2.LINE_AA)

    out_sim_path = os.path.join("outputs", "miranha_custom_preview.png")
    cv2.imwrite(out_sim_path, sim_canvas)

    total_strokes = sum(len(l["strokes"]) for l in layers)
    return {
        "success": True,
        "layers": layers,
        "total_strokes": total_strokes,
        "simulation_path": out_sim_path
    }
