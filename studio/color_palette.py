"""
Gerenciamento de Cores, Calibração de Espectro Infinito e Busca Perceptual CIE Lab / CIEDE2000.
Suporta gama cromática completa: tons de pele humana (Fitzpatrick), sombras, luzes, saturações e degradês.
"""
import os
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
from . import config

class ColorPaletteManager:
    def __init__(self, spectrum_image_path=None):
        self.spectrum_path = spectrum_image_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "outputs", "spectrum_crop.png"
        )
        self.lut_entries = []
        self._build_palette_lut()

    def _rgb_to_lab(self, r, g, b):
        """Converte RGB (0-255) para CIE Lab usando OpenCV."""
        bgr_pixel = np.uint8([[[b, g, r]]])
        lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2Lab)[0][0]
        return float(lab_pixel[0]), float(lab_pixel[1]), float(lab_pixel[2])

    def _delta_e_ciede(self, lab1, lab2):
        """
        Cálculo perceptual aprimorado de distância de cor no espaço CIE Lab.
        Equilibra luminância e croma para reproduzir a percepção visual do olho humano.
        """
        dL = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        
        c1 = np.sqrt(lab1[1]**2 + lab1[2]**2)
        c2 = np.sqrt(lab2[1]**2 + lab2[2]**2)
        dC = c1 - c2
        
        dH_sq = max(0.0, da**2 + db**2 - dC**2)
        dH = np.sqrt(dH_sq)
        
        # Ponderação CIE94/CIEDE2000
        sL = 1.0
        sC = 1.0 + 0.045 * c1
        sH = 1.0 + 0.015 * c1
        
        return np.sqrt((dL / sL)**2 + (dC / sC)**2 + (dH / sH)**2)

    def _build_palette_lut(self):
        """Constrói a Look-Up Table (LUT) cobrindo todo o espaço de cores do Instagram."""
        self.lut_entries = []

        # A tabela física do aparelho também alimenta o planejador de gamut.
        # Assim ele não promete tons que o executor não sabe selecionar.
        from .vision.perceptual_color import PerceptualColorEngine
        measured = PerceptualColorEngine().entries
        if measured:
            for entry in measured:
                item = dict(entry)
                item["rgb"] = tuple(item["rgb"])
                item["x"] = item.get("touch_x", item.get("target_x"))
                item["y"] = item.get("touch_y", item.get("target_y"))
                item["lab"] = self._rgb_to_lab(*item["rgb"])
                self.lut_entries.append(item)
            return

        # 1. As 9 cores base de toque simples
        for base in config.PALETTE_BASE_COLORS:
            r, g, b = base["rgb"]
            lab = self._rgb_to_lab(r, g, b)
            self.lut_entries.append({
                "name": base["name"],
                "is_long_press": False,
                "x": base["x"],
                "y": base["y"],
                "rgb": (r, g, b),
                "lab": lab
            })

        # 2. Amostras do Espectro Contínuo a partir do spectrum_crop.png real
        if os.path.exists(self.spectrum_path):
            spec_img = cv2.imread(self.spectrum_path)
            if spec_img is not None:
                h, w = spec_img.shape[:2]
                screen_x_start = 180
                screen_x_end = 900
                
                # Faz amostragem em 3 alturas do gradiente:
                # Y topo: cores mais claras/iluminadas
                # Y centro: cores puras de saturação média
                # Y base: tons mais escuros/sombreados
                y_levels = [
                    (int(h * 0.2), config.UI_SPECTRUM_POPUP_Y_TOP, "Claro"),
                    (int(h * 0.5), config.UI_SPECTRUM_POPUP_Y, "Médio"),
                    (int(h * 0.8), config.UI_SPECTRUM_POPUP_Y_BOTTOM, "Escuro")
                ]
                
                step = max(1, w // 45)  # ~45 divisões horizontais
                for sample_x in range(0, w, step):
                    for img_y, screen_y, shade_name in y_levels:
                        b, g, r = spec_img[img_y, sample_x]
                        touch_x = int(screen_x_start + (sample_x / w) * (screen_x_end - screen_x_start))
                        lab = self._rgb_to_lab(int(r), int(g), int(b))
                        
                        self.lut_entries.append({
                            "name": f"Espectro ({int(r)},{int(g)},{int(b)}) [{shade_name}]",
                            "is_long_press": True,
                            "base_x": 487,                     # Long press no círculo amarelo/laranja
                            "base_y": config.UI_PALETTE_BASE_Y,
                            "target_x": touch_x,
                            "target_y": screen_y,
                            "rgb": (int(r), int(g), int(b)),
                            "lab": lab
                        })

        # 3. Biblioteca Especializada de Tons de Pele (Fitzpatrick Scale)
        # Tons calibrados com as coordenadas de transição do amarelo/laranja para o espectro
        skin_tones = [
            ("Pele Alva / Marfim", (255, 230, 215), 320, config.UI_SPECTRUM_POPUP_Y_TOP),
            ("Pele Clara / Bege", (240, 205, 180), 410, config.UI_SPECTRUM_POPUP_Y_TOP),
            ("Pele Pêssego Natural", (230, 185, 155), 480, config.UI_SPECTRUM_POPUP_Y),
            ("Pele Dourada / Morena Clara", (215, 165, 130), 520, config.UI_SPECTRUM_POPUP_Y),
            ("Pele Parda / Mel", (190, 140, 105), 550, config.UI_SPECTRUM_POPUP_Y),
            ("Pele Castanha / Bronze", (165, 115, 80), 580, config.UI_SPECTRUM_POPUP_Y_BOTTOM),
            ("Pele Negra / Chocolate", (125, 80, 55), 620, config.UI_SPECTRUM_POPUP_Y_BOTTOM),
            ("Pele Ébano / Café Profundo", (75, 45, 30), 660, config.UI_SPECTRUM_POPUP_Y_BOTTOM),
            ("Sombra Facial Quente", (140, 70, 60), 450, config.UI_SPECTRUM_POPUP_Y_BOTTOM),
            ("Realce Labial Rosado", (210, 85, 95), 380, config.UI_SPECTRUM_POPUP_Y),
            ("Realce Labial Carmim", (175, 40, 50), 360, config.UI_SPECTRUM_POPUP_Y_BOTTOM),
            ("Cabelo Loiro / Dourado", (230, 200, 120), 500, config.UI_SPECTRUM_POPUP_Y_TOP),
            ("Cabelo Castanho Médio", (100, 65, 45), 560, config.UI_SPECTRUM_POPUP_Y_BOTTOM),
            ("Cabelo Castanho Escuro", (50, 35, 28), 640, config.UI_SPECTRUM_POPUP_Y_BOTTOM),
        ]
        for name, rgb, tx, ty in skin_tones:
            r, g, b = rgb
            lab = self._rgb_to_lab(r, g, b)
            self.lut_entries.append({
                "name": name,
                "is_long_press": True,
                "base_x": 487,
                "base_y": config.UI_PALETTE_BASE_Y,
                "target_x": tx,
                "target_y": ty,
                "rgb": rgb,
                "lab": lab
            })

    def match_color(self, target_bgr):
        """
        Encontra a melhor cor do Instagram para a cor BGR desejada,
        retornando a entrada da LUT com o menor Delta E e nome descritivo.
        """
        b, g, r = [int(c) for c in target_bgr]
        target_lab = self._rgb_to_lab(r, g, b)
        
        best_entry = None
        min_dist = float("inf")
        
        for entry in self.lut_entries:
            dist = self._delta_e_ciede(target_lab, entry["lab"])
            if dist < min_dist:
                min_dist = dist
                best_entry = entry
                
        match_res = dict(best_entry)
        match_res["delta_e"] = round(min_dist, 2)
        match_res["original_rgb"] = (r, g, b)
        match_res["original_hex"] = f"#{r:02x}{g:02x}{b:02x}".upper()
        matched_r, matched_g, matched_b = best_entry["rgb"]
        match_res["matched_hex"] = f"#{matched_r:02x}{matched_g:02x}{matched_b:02x}".upper()
        return match_res


class PhysicalGamutMatcher:
    """
    Matcher Físico Restrito ao Gamut Real do Instagram (Fase 10).
    Planeja (tool, color, passes, velocity) e detecta cores inalcançáveis com aviso explícito.
    """

    def __init__(self, palette_manager: Optional[ColorPaletteManager] = None, unreproducible_threshold: float = 28.0):
        self.palette = palette_manager or ColorPaletteManager()
        self.unreproducible_threshold = unreproducible_threshold

    def match_target(
        self,
        target_bgr: Tuple[int, int, int],
        background_rgb: Tuple[int, int, int] = (255, 255, 255),
    ) -> Dict[str, Any]:
        """
        Avalia o espaço físico de combinações (tool, color, passes)
        para alcançar a cor desejada com o menor erro perceptual CIEDE2000.
        """
        b, g, r = [int(c) for c in target_bgr]
        target_lab = self.palette._rgb_to_lab(r, g, b)

        best_cand = None
        min_delta_e = float("inf")

        # 1. Avalia ferramentas sólidas (1 passada opaca)
        for entry in self.palette.lut_entries:
            dist = self.palette._delta_e_ciede(target_lab, entry["lab"])
            if dist < min_delta_e:
                min_delta_e = dist
                best_cand = {
                    "tool": "solid",
                    "color_entry": entry,
                    "passes": 1,
                    "predicted_rgb": entry["rgb"],
                    "delta_e": round(dist, 2),
                }

        # 2. Avalia marcador translúcido (1, 2 e 3 passadas sobre o fundo)
        for entry in self.palette.lut_entries:
            er, eg, eb = entry["rgb"]
            bg_r, bg_g, bg_b = background_rgb
            
            # Modelo de composição alfa de passadas do marcador (Fase 4.2)
            for p in (1, 2, 3):
                alpha = 1.0 - (1.0 - 0.45) ** p
                comp_r = int(round(er * alpha + bg_r * (1.0 - alpha)))
                comp_g = int(round(eg * alpha + bg_g * (1.0 - alpha)))
                comp_b = int(round(eb * alpha + bg_b * (1.0 - alpha)))
                comp_lab = self.palette._rgb_to_lab(comp_r, comp_g, comp_b)
                dist = self.palette._delta_e_ciede(target_lab, comp_lab)
                if dist < min_delta_e:
                    min_delta_e = dist
                    best_cand = {
                        "tool": "marker",
                        "color_entry": entry,
                        "passes": p,
                        "predicted_rgb": (comp_r, comp_g, comp_b),
                        "delta_e": round(dist, 2),
                    }

        is_unreproducible = min_delta_e > self.unreproducible_threshold
        warning = None
        if is_unreproducible:
            warning = (
                f"Aviso de Gamut: Cor desejada RGB {r, g, b} está fora do gamut reproduzível "
                f"(menor erro ΔE={min_delta_e:.1f} > limiar {self.unreproducible_threshold}). "
                f"Usando melhor aproximação via {best_cand['tool']} (ΔE={min_delta_e:.1f})."
            )

        res = dict(best_cand)
        res["target_rgb"] = (r, g, b)
        res["is_unreproducible"] = is_unreproducible
        res["warning"] = warning
        return res

    def quantize_image_direct(
        self, image_bgr: np.ndarray, max_colors: int = 12
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
        """
        Quantização direta no Gamut da LUT (Fase 10.1):
        Gera a imagem quantizada exclusivamente com as cores alcançáveis pelo Instagram.
        Retorna (imagem_quantizada_bgr, lista_de_matches_usados, total_inalcançáveis).
        """
        h, w = image_bgr.shape[:2]
        # Reduz temporariamente para amostragem rápida se imagem for grande
        sample = cv2.resize(image_bgr, (min(w, 160), min(h, 160)), interpolation=cv2.INTER_AREA)
        pixels = sample.reshape(-1, 3)

        # Agrupa pixels em amostras representativas
        unique_colors = np.unique(pixels // 16 * 16, axis=0)
        matches: List[Dict[str, Any]] = []
        unreproducible_count = 0

        for col in unique_colors:
            b, g, r = int(col[0]), int(col[1]), int(col[2])
            m = self.match_target((b, g, r))
            matches.append(m)
            if m["is_unreproducible"]:
                unreproducible_count += 1

        # Mapeia cada pixel da imagem original para a melhor cor encontrada
        # Seleciona os N matches mais representativos
        matches = sorted(matches, key=lambda x: x["delta_e"])[:max_colors]

        # Constrói imagem quantizada diretamente
        quantized_bgr = np.zeros_like(image_bgr)
        # Substituição vetorizada aproximada
        for m in matches:
            t_r, t_g, t_b = m["target_rgb"]
            p_r, p_g, p_b = m["predicted_rgb"]
            dist_sq = (
                (image_bgr[:, :, 0].astype(np.int32) - t_b) ** 2
                + (image_bgr[:, :, 1].astype(np.int32) - t_g) ** 2
                + (image_bgr[:, :, 2].astype(np.int32) - t_r) ** 2
            )
            mask = dist_sq < 1200
            quantized_bgr[mask] = (p_b, p_g, p_r)

        # Preenche pixels restantes com o match mais próximo global
        black_mask = np.all(quantized_bgr == 0, axis=2)
        if np.any(black_mask) and len(matches) > 0:
            best_fallback = matches[0]["predicted_rgb"]
            quantized_bgr[black_mask] = (best_fallback[2], best_fallback[1], best_fallback[0])

        return quantized_bgr, matches, unreproducible_count
