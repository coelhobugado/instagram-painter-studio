"""
Analisador Semântico de Cena e Decisor Autônomo de Estratégias (AUTO Mode).
Implementa a Seção 65 (AUTO Mode) e Seção 23 (Specialized Renderers) do Plano de Evolução Completo.
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, Union, Optional
from ..core.models import QualityPreset

class AutoModeDecision:
    """
    Representação tipada e compatível com dicionário da decisão autônoma do modo de pintura.
    Permite acesso via atributos (.mode, .recommended_mode) e via chaves (['mode']).
    """
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self.mode = data.get("recommended_mode", "layered_paint")
        self.recommended_mode = self.mode
        self.scene_type = data.get("scene_type", "STANDARD_GRAPHIC")
        self.recommended_k_colors = data.get("recommended_k_colors", 8)
        self.k_colors = self.recommended_k_colors
        self.brush_spacing_px = data.get("brush_spacing_px", 5)
        self.preset = data.get("preset", QualityPreset.BALANCED)
        self.metrics = data.get("metrics", {})
        self.description = data.get("description", "")

    def __getitem__(self, item: str) -> Any:
        if item == "mode":
            return self.mode
        return self._data[item]

    def __contains__(self, item: str) -> bool:
        return item in self._data or item == "mode"

    def get(self, item: str, default: Any = None) -> Any:
        if item == "mode":
            return self.mode
        return self._data.get(item, default)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self._data)
        d["mode"] = self.mode
        return d

    def __repr__(self) -> str:
        return f"<AutoModeDecision mode='{self.mode}' scene='{self.scene_type}' k_colors={self.k_colors}>"

class AutoModeDecisor:
    @classmethod
    def decide_mode(cls, image_input: Union[str, np.ndarray]) -> AutoModeDecision:
        """
        Analisa a imagem e infere a estratégia ideal de pintura, retornando um AutoModeDecision
        com atributos como .mode, .recommended_mode, .recommended_k_colors e dicionário.
        Aceita o caminho para o arquivo de imagem (str) ou imagem carregada em BGR (np.ndarray).
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Arquivo de imagem não encontrado para inferência: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Falha ao decodificar imagem para inferência: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError(f"Entrada inválida para decide_mode: esperado str ou np.ndarray, recebido {type(image_input)}")

        raw_decision = cls.analyze_scene_and_decide(img)
        return AutoModeDecision(raw_decision)

    @staticmethod
    def choose_drawing_style(img_bgr: np.ndarray) -> str:
        """Estilo do modo Automático: "LINHAS", "COLORIR" ou "CAMADAS".

        - fundo não uniforme e muitas cores (foto)             → Ilustração chapada
        - fundo uniforme com áreas coloridas largas (cartoon)   → Livro de colorir
        - fundo uniforme só com linhas, mesmo coloridas         → Traçado colorido
        """
        h, w = img_bgr.shape[:2]
        scale = 400.0 / max(h, w)
        small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
        lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        bg = np.median(border, axis=0)
        # Fundo uniforme = desenho sobre papel/tela lisa; foto e degradê não têm
        if float(np.median(np.linalg.norm(border - bg, axis=1))) >= 6.0:
            return "CAMADAS"
        delta = np.linalg.norm(lab - bg, axis=2)
        foreground = delta > 22
        if foreground.mean() < 0.02:
            return "CAMADAS"  # quase nada destoa do fundo: só tons sutis, sem linhas
        chroma = np.hypot(lab[..., 1], lab[..., 2])
        colored = (foreground & (chroma > 20)).astype(np.uint8) * 255
        # Áreas coloridas mais largas que ~4% da imagem (roupa, formas de logo); letras e linhas não passam
        k = max(5, int(round(max(small.shape[:2]) * 0.04)) | 1)
        wide = cv2.morphologyEx(colored, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if float(np.count_nonzero(wide)) / wide.size > 0.03:
            return "COLORIR"
        # Foto recortada sobre fundo liso: muitos degradês suaves no primeiro plano
        gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1)
        g = np.hypot(gx, gy)[foreground]
        if float(np.mean((g > 0.5) & (g < 6.0))) > 0.35:
            return "CAMADAS"
        return "LINHAS"

    @staticmethod
    def analyze_scene_and_decide(img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Analisa a imagem e infere automaticamente a melhor estratégia, número de cores K,
        espaçamento de pincel e orçamento de traços.
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

        # 1. Entropia de Cores (Cálculo de diversidade cromática)
        hist_b = cv2.calcHist([img_bgr], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_bgr], [1], None, [32], [0, 256]).flatten()
        hist_r = cv2.calcHist([img_bgr], [2], None, [32], [0, 256]).flatten()
        hist_total = hist_b + hist_g + hist_r
        p = hist_total / (np.sum(hist_total) + 1e-8)
        color_entropy = -float(np.sum(p * np.log2(p + 1e-8)))

        # 2. Densidade de Bordas Estruturais
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float(np.count_nonzero(edges)) / float(h * w)

        # 3. Detecção Robusta de Tons de Pele (Espaço CIE Lab Analítico em float32)
        # Tons de pele humana (Escala Fitzpatrick I-VI) situam-se tipicamente em L*: [30, 92], a*: [3, 36], b*: [8, 48]
        img_rgb_f = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab_f = cv2.cvtColor(img_rgb_f, cv2.COLOR_RGB2Lab)
        L_f = lab_f[:, :, 0]
        a_f = lab_f[:, :, 1]
        b_f = lab_f[:, :, 2]

        skin_mask = (
            (L_f >= 30.0) & (L_f <= 92.0) &
            (a_f >= 3.0) & (a_f <= 36.0) &
            (b_f >= 8.0) & (b_f <= 48.0)
        )
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(h * w)

        # 4. Classificação de Tipo de Cena e Decisão
        if (edge_density > 0.04 and color_entropy < 3.5) or (color_entropy < 2.2):
            scene_type = "LOGO_OR_LINEART"
            recommended_mode = "line_art"
            k_colors = 1
            brush_spacing = 2
            preset = QualityPreset.FAST
            description = "Desenho de Linhas (Line-Art) identificado (ex: tulipa, logo ou esboço). Traço central único sem contorno duplicado, curvas suaves e preservação da cor e fundo da referência."
        elif skin_ratio > 0.15:
            scene_type = "PORTRAIT"
            recommended_mode = "layered_paint"
            k_colors = 10 if color_entropy < 4.0 else 14
            brush_spacing = 5
            preset = QualityPreset.BALANCED
            description = f"Retrato identificado ({skin_ratio*100:.1f}% de pele). Priorizando fidelidade facial e sombras suaves."
        elif color_entropy > 4.2:
            scene_type = "COMPLEX_SCENE"
            recommended_mode = "layered_paint"
            k_colors = 18
            brush_spacing = 4
            preset = QualityPreset.HIGH
            description = "Cena complexa com alta diversidade cromática. Alocando paleta estendida e detalhes finos."
        else:
            scene_type = "STANDARD_GRAPHIC"
            recommended_mode = "layered_paint"
            k_colors = 8
            brush_spacing = 6
            preset = QualityPreset.BALANCED
            description = "Ilustração equilibrada. Otimizado para renderização rápida com boa fidelidade."

        return {
            "scene_type": scene_type,
            "recommended_mode": recommended_mode,
            "recommended_k_colors": k_colors,
            "brush_spacing_px": brush_spacing,
            "preset": preset,
            "metrics": {
                "color_entropy": round(color_entropy, 2),
                "edge_density": round(edge_density, 3),
                "skin_ratio": round(skin_ratio, 3)
            },
            "description": description
        }
