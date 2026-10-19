"""
Motor de Cor Perceptual CIE Lab, CIEDE2000 e Busca Adaptativa na LUT.
Implementa a Seção 4 (Color Engine v2) e Seção 5 (LUT 2D Adaptativa com Refinamento Local).
"""
import os
import json
import cv2
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Any, List

class PerceptualColorEngine:
    @staticmethod
    def physical_touch_key(touch: Dict[str, Any]) -> Tuple[Any, ...]:
        """Identidade física completa; coordenadas iguais em páginas distintas não são a mesma tinta."""
        return (
            int(touch.get("palette_page", 1)),
            bool(touch.get("is_long_press", False)),
            touch.get("base_x"), touch.get("base_y"),
            int(touch.get("target_x", touch.get("touch_x", 0))),
            int(touch.get("target_y", touch.get("touch_y", 0))),
            str(touch.get("lut_id", touch.get("id", ""))),
        )

    def __init__(self, lut_path: str = None):
        if lut_path is None:
            lut_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "profiles", "SM-A736B", "color_lut.json"
            )
        self.lut_path = lut_path
        self.lut_data = self._load_lut(lut_path)
        self.entries = self.lut_data.get("entries", [])
        
        # Constrói KD-Tree espacial no espaço CIE Lab para busca de altíssima velocidade
        self.lab_coords = np.array([e["lab"] for e in self.entries], dtype=np.float32)
        self.kdtree = cKDTree(self.lab_coords)
        
        # Cache de consultas frequentes para acelerar K-Means e segmentação
        self._cache = {}

    def _load_lut(self, path: str) -> Dict[str, Any]:
        """Carrega a LUT calibrada no formato Schema v2."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"schema_version": 2, "sample_count": 0, "entries": []}

    @staticmethod
    def rgb_to_lab(r: int, g: int, b: int) -> Tuple[float, float, float]:
        """Converte RGB (0-255) para coordenadas CIE L*a*b* normalizadas com precisão float32."""
        bgr_f = np.float32([[[b / 255.0, g / 255.0, r / 255.0]]])
        lab_f = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2Lab)[0][0]
        return float(lab_f[0]), float(lab_f[1]), float(lab_f[2])

    @staticmethod
    def lab_to_rgb(L: float, a: float, b: float) -> Tuple[int, int, int]:
        """Converte CIE L*a*b* de volta para RGB com precisão float32."""
        lab_f = np.float32([[[L, a, b]]])
        bgr_f = cv2.cvtColor(lab_f, cv2.COLOR_Lab2BGR)[0][0]
        r = int(np.clip(round(bgr_f[2] * 255.0), 0, 255))
        g = int(np.clip(round(bgr_f[1] * 255.0), 0, 255))
        b_val = int(np.clip(round(bgr_f[0] * 255.0), 0, 255))
        return r, g, b_val

    @staticmethod
    def delta_e_ciede2000(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
        """
        Cálculo estrito da fórmula perceptual CIEDE2000 (ISO/CIE 11664-6:2014).
        Modela as não-linearidades do sistema visual humano, rotação no quadrante azul e pesos SL, SC, SH.
        """
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2

        # 1. Cálculo de C*ab e média de croma
        c1 = np.sqrt(a1**2 + b1**2)
        c2 = np.sqrt(a2**2 + b2**2)
        c_bar = (c1 + c2) / 2.0

        # Fator G de correção de croma
        g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25**7 + 1e-12)))

        a1_prime = (1.0 + g) * a1
        a2_prime = (1.0 + g) * a2

        c1_prime = np.sqrt(a1_prime**2 + b1**2)
        c2_prime = np.sqrt(a2_prime**2 + b2**2)

        # 2. Ângulos de matiz h'
        h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
        h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

        # 3. Diferenças delta
        delta_L_prime = L2 - L1
        delta_C_prime = c2_prime - c1_prime

        # Diferença de matiz angular
        h_diff = h2_prime - h1_prime
        if c1_prime * c2_prime == 0.0:
            delta_h_prime = 0.0
        elif np.abs(h_diff) <= 180.0:
            delta_h_prime = h_diff
        elif h_diff > 180.0:
            delta_h_prime = h_diff - 360.0
        else:
            delta_h_prime = h_diff + 360.0

        delta_H_prime = 2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(np.radians(delta_h_prime / 2.0))

        # 4. Médias para ponderações
        L_bar_prime = (L1 + L2) / 2.0
        C_bar_prime = (c1_prime + c2_prime) / 2.0

        if c1_prime * c2_prime == 0.0:
            h_bar_prime = h1_prime + h2_prime
        elif np.abs(h_diff) <= 180.0:
            h_bar_prime = (h1_prime + h2_prime) / 2.0
        elif h1_prime + h2_prime < 360.0:
            h_bar_prime = (h1_prime + h2_prime + 360.0) / 2.0
        else:
            h_bar_prime = (h1_prime + h2_prime - 360.0) / 2.0

        # 5. Funções de ponderação
        T = (1.0 - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
             + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0)))

        delta_theta = 30.0 * np.exp(-((h_bar_prime - 275.0) / 25.0)**2)
        r_c = 2.0 * np.sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7 + 1e-12))
        s_L = 1.0 + (0.015 * (L_bar_prime - 50.0)**2) / np.sqrt(20.0 + (L_bar_prime - 50.0)**2)
        s_C = 1.0 + 0.045 * C_bar_prime
        s_H = 1.0 + 0.015 * C_bar_prime * T
        r_T = -np.sin(np.radians(2.0 * delta_theta)) * r_c

        # 6. Distância final
        term_L = delta_L_prime / s_L
        term_C = delta_C_prime / s_C
        term_H = delta_H_prime / s_H

        delta_E00 = np.sqrt(term_L**2 + term_C**2 + term_H**2 + r_T * term_C * term_H)
        return float(delta_E00)

    @staticmethod
    def compute_hue_consistency_penalty(target_rgb: Tuple[int, int, int], candidate_rgb: Tuple[int, int, int]) -> float:
        """
        Penaliza severamente discrepâncias perceptuais de matiz para cores cromáticas (S > 0.18),
        impedindo que azuis se tornem roxos/violetas, vermelhos se tornem verdes/amarelos, etc.
        """
        r1, g1, b1 = target_rgb[0] / 255.0, target_rgb[1] / 255.0, target_rgb[2] / 255.0
        r2, g2, b2 = candidate_rgb[0] / 255.0, candidate_rgb[1] / 255.0, candidate_rgb[2] / 255.0

        max1, min1 = max(r1, g1, b1), min(r1, g1, b1)
        max2, min2 = max(r2, g2, b2), min(r2, g2, b2)

        s1 = 0.0 if max1 == 0 else (max1 - min1) / max1
        v1 = max1
        s2 = 0.0 if max2 == 0 else (max2 - min2) / max2

        # Se o tom original for acromático (cinza/branco/preto neutro), não há exigência estrita de matiz
        if s1 < 0.18 or v1 < 0.12:
            return 0.0

        def _calc_hue(r, g, b, mx, mn):
            diff = mx - mn
            if diff == 0:
                return 0.0
            if mx == r:
                h = ((g - b) / diff) % 6.0
            elif mx == g:
                h = ((b - r) / diff) + 2.0
            else:
                h = ((r - g) / diff) + 4.0
            return (h * 60.0) % 360.0

        h1 = _calc_hue(r1, g1, b1, max1, min1)
        h2 = _calc_hue(r2, g2, b2, max2, min2)

        h_diff = min(abs(h1 - h2), 360.0 - abs(h1 - h2))
        penalty = 0.0

        # Penalidade angular progressiva para desvios > 25 graus
        if h_diff > 25.0:
            penalty += ((h_diff - 25.0) / 10.0) ** 2 * 3.5

        # Penalidade categórica estrita para violação de quadrantes cromáticos:
        # 1. Família Azul (185° - 255°): NÃO pode virar Roxo/Violeta (> 265°) nem Verde (< 165°)
        if 185.0 <= h1 <= 255.0 and s1 > 0.25:
            if h2 > 265.0 or h2 < 165.0:
                penalty += 35.0

        # 2. Família Vermelha (345° - 360° ou 0° - 25°): NÃO pode virar Verde nem Azul
        is_target_red = (h1 >= 345.0 or h1 <= 25.0) and s1 > 0.30
        if is_target_red:
            if 80.0 <= h2 <= 270.0:
                penalty += 40.0

        # 3. Família Verde (75° - 165°): NÃO pode virar Vermelho/Roxo
        if 75.0 <= h1 <= 165.0 and s1 > 0.25:
            if h2 > 280.0 or h2 < 55.0:
                penalty += 35.0

        return penalty

    def match_color(self, target_rgb: Tuple[int, int, int], preferred_lut_id: Any = None) -> Dict[str, Any]:
        """
        Encontra a melhor cor executável na interface do Instagram minimizando CIEDE2000
        com preservação semântica e categórica da família de matiz (Hue Consistency).
        """
        key = (target_rgb[0], target_rgb[1], target_rgb[2], preferred_lut_id)
        if key in self._cache:
            return self._cache[key]

        target_lab = self.rgb_to_lab(*target_rgb)

        # 1. Avalia primeiro todas as cores base nativas da paleta inferior (toque simples direto)
        base_entries = [e for e in self.entries if not e.get("is_long_press", False)]
        best_base = None
        min_score_base = float("inf")
        min_de_base = float("inf")
        for b in base_entries:
            de = self.delta_e_ciede2000(target_lab, tuple(b["lab"]))
            cand_rgb = tuple(b.get("rgb", (0, 0, 0)))
            pen = self.compute_hue_consistency_penalty(target_rgb, cand_rgb)
            score = de + pen
            if score < min_score_base:
                min_score_base = score
                min_de_base = de
                best_base = b

        # 2. Pré-filtra os candidatos de espectro via cKDTree
        if len(self.entries) > 15 and hasattr(self, "kdtree") and self.kdtree is not None:
            _, candidate_indices = self.kdtree.query(target_lab, k=min(60, len(self.entries)))
            if np.isscalar(candidate_indices):
                candidate_indices = [int(candidate_indices)]
            candidates = [self.entries[int(i)] for i in candidate_indices]
        else:
            candidates = self.entries

        best_spectrum = None
        min_score_spectrum = float("inf")
        min_de_spectrum = float("inf")
        for c in candidates:
            if not c.get("is_long_press", False):
                continue
            de = self.delta_e_ciede2000(target_lab, tuple(c["lab"]))
            cand_rgb = tuple(c.get("rgb", (0, 0, 0)))
            pen = self.compute_hue_consistency_penalty(target_rgb, cand_rgb)
            score = de + pen
            if score < min_score_spectrum:
                min_score_spectrum = score
                min_de_spectrum = de
                best_spectrum = c

        # 3. Prefere a cor base se o ganho visual do espectro for pequeno.
        use_base = False
        if best_base is not None:
            if best_spectrum is None:
                use_base = True
            elif min_score_base <= min_score_spectrum + 2.5:
                use_base = True

        if use_base and best_base is not None:
            best_entry = best_base
            min_de = min_de_base
        elif best_spectrum is not None:
            best_entry = best_spectrum
            min_de = min_de_spectrum
        else:
            best_entry = best_base or (self.entries[0] if self.entries else None)
            min_de = min_de_base

        if preferred_lut_id is not None:
            chosen = next((e for e in self.entries if e.get("id") == preferred_lut_id), None)
            if chosen is None:
                raise ValueError(f"Tinta física {preferred_lut_id} não existe no perfil")
            best_entry = chosen
            min_de = self.delta_e_ciede2000(target_lab, tuple(chosen["lab"]))

        # Fallback de segurança se LUT estiver vazia
        if best_entry is None:
            best_entry = {
                "id": "fallback_white",
                "name": "Branco Puro (Fallback)",
                "rgb": [255, 255, 255],
                "hex": "#FFFFFF",
                "lab": [100.0, 0.0, 0.0],
                "touch_x": 411,
                "touch_y": 2184,
                "is_long_press": False
            }
            min_de = 0.0

        result = dict(best_entry)
        result["delta_e"] = round(min_de, 2)
        result["target_rgb"] = target_rgb
        result["target_hex"] = f"#{target_rgb[0]:02X}{target_rgb[1]:02X}{target_rgb[2]:02X}"
        result["matched_hex"] = best_entry.get("hex", "")
        phys_rgb = tuple(best_entry.get("rgb", target_rgb))
        result["matched_rgb"] = phys_rgb
        result["rgb"] = phys_rgb

        is_lp = bool(best_entry.get("is_long_press", False))
        base_t = best_entry.get("base_touch") or {}
        base_x = base_t.get("x") if isinstance(base_t, dict) and "x" in base_t else best_entry.get("base_x")
        base_y = base_t.get("y") if isinstance(base_t, dict) and "y" in base_t else best_entry.get("base_y")

        # Se for toque longo no espectro e base_x não estiver alinhado com o matiz da cor,
        # seleciona a bolinha base mais próxima para ancorar o arrasto de forma natural
        if preferred_lut_id is None and is_lp and result.get("palette_page", 1) == 1 and (base_x is None or base_x == 636):
            palette_anchors = [
                (411, 2184, (255, 255, 255)),
                (486, 2184, (56, 151, 240)),
                (561, 2184, (112, 192, 80)),
                (636, 2184, (255, 200, 64)),
                (711, 2184, (253, 141, 50)),
                (786, 2184, (237, 73, 86)),
                (861, 2184, (209, 8, 105)),
                (936, 2184, (163, 7, 186)),
            ]
            cand_bx, cand_by = 636, 2184
            cand_min_dist = float("inf")
            for ax, ay, argb in palette_anchors:
                dist = np.linalg.norm(np.array(phys_rgb) - np.array(argb))
                if dist < cand_min_dist:
                    cand_min_dist = dist
                    cand_bx, cand_by = ax, ay
            base_x = cand_bx
            base_y = cand_by

        tx = best_entry.get("touch_x", best_entry.get("target_x", best_entry.get("x", 0)))
        ty = best_entry.get("touch_y", best_entry.get("target_y", best_entry.get("y", 0)))

        result["is_long_press"] = is_lp
        result["palette_page"] = int(result.get("palette_page", 1))
        result["base_x"] = base_x
        result["base_y"] = base_y
        result["target_x"] = tx
        result["target_y"] = ty
        result["x"] = tx
        result["y"] = ty
        result["touch_x"] = tx
        result["touch_y"] = ty
        result["base_touch"] = {"x": base_x, "y": base_y} if base_x is not None else None

        self._cache[key] = result
        return result

