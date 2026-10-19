# -*- coding: utf-8 -*-
"""
Extrator de Feições Semânticas e Isolamento Anatômico de Alta Resolução.
Implementa o Pilar 1 do Plano Mestre de Maturidade 100%.

Detecta micro-reflexos especulares (catchlights em olhos), detalhes anatômicos críticos
e gera máscaras de proteção e camadas de acabamento com micro-pincel fino.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class SemanticFeatureExtractor:
    """
    Extrator de estruturas semânticas, destaques especulares e feições anatômicas prioritárias.
    """

    @classmethod
    def extract_internal_white_features(cls, image: np.ndarray,
                                         min_val: int = 215,
                                         min_area: int = 25,
                                         max_area_ratio: float = 0.25) -> np.ndarray:
        """
        Detecta regiões claras/brancas internas encapsuladas (olhos de personagens, dentes,
        símbolos de peito, reflexos e luzes especulares amplas) que NÃO fazem parte do fundo externo
        e NUNCA devem ser invadidas por traços de preenchimento corporal escuro/médio.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        total_pixels = h * w
        max_allowed_area = int(total_pixels * max_area_ratio)

        # 1. Binarização de alta intensidade
        _, bright_mask = cv2.threshold(gray, min_val, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(bright_mask) == 0:
            return np.zeros_like(gray, dtype=np.uint8)

        # 2. Desacopla o fundo externo conectado aos cantos da imagem
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        inner_mask = bright_mask.copy()
        for corner in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cx, cy = corner
            if inner_mask[cy, cx] > 0:
                cv2.floodFill(inner_mask, flood_mask, (cx, cy), 0)

        # 3. Filtra componentes com área compatível com feições internas
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inner_mask, connectivity=8)
        protected_mask = np.zeros_like(gray, dtype=np.uint8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_allowed_area:
                protected_mask[labels == i] = 255

        return protected_mask

    @classmethod
    def extract_specular_highlights(cls, image: np.ndarray,
                                    min_val: int = 215,
                                    max_island_area: int = 120) -> np.ndarray:
        """
        Detecta ilhas de brilho especular e cavidades claras internas protegidas (catchlights,
        olhos de personagens, dentes ou reflexos) que NUNCA devem ser pintadas por pincéis de fundo ou corpo.
        Retorna uma máscara binária (uint8) das regiões de luz protegidas.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # 1. Binarização de alta intensidade
        _, bright_mask = cv2.threshold(gray, min_val, 255, cv2.THRESH_BINARY)

        # 2. Análise de componentes conexas para micro-ilhas especulares
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

        protected_mask = np.zeros_like(gray, dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Micro-catchlights e brilhos típicos
            if 2 <= area <= max_island_area and max(w, h) <= 25:
                protected_mask[labels == i] = 255

        # 3. Une com as cavidades claras internas amplas (olhos de personagens, emblemas)
        internal_whites = cls.extract_internal_white_features(image, min_val=min_val)
        return cv2.bitwise_or(protected_mask, internal_whites)

    @classmethod
    def extract_micro_contrast_edges(cls, image: np.ndarray,
                                     low_thresh: int = 50,
                                     high_thresh: int = 150) -> np.ndarray:
        """
        Extrai bordas de micro-contraste para detalhes anatômicos (linhas de cílios, pálpebras,
        junção labial e detalhes de alta frequência).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Suavização bilateral preservando bordas vivas
        filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=5)
        edges = cv2.Canny(filtered, low_thresh, high_thresh)
        return edges

    @classmethod
    def create_specular_preservation_mask(cls, base_mask: np.ndarray,
                                          specular_mask: np.ndarray,
                                          dilation_radius: int = 1) -> np.ndarray:
        """
        Subtrai as ilhas de brilho especular da máscara base de uma camada de fundo/pele,
        garantindo que os traços largos não invadam o reflexo dos olhos.
        """
        if np.count_nonzero(specular_mask) == 0:
            return base_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
        dilated_specular = cv2.dilate(specular_mask, kernel)

        # Remove o brilho especular da máscara base
        safe_mask = cv2.bitwise_and(base_mask, cv2.bitwise_not(dilated_specular))
        return safe_mask

    @classmethod
    def generate_specular_highlight_strokes(cls, specular_mask: Optional[np.ndarray], brush_width: float = 8.0) -> List[List[Tuple[int, int]]]:
        """
        Gera toques e traços para reflexos claros e cavidades brancas protegidas (catchlights e olhos de personagens).
        Micro-pontos recebem toques centrados; regiões maiores (olhos de cartoons/anime) recebem preenchimento estruturado.
        """
        if specular_mask is None or np.count_nonzero(specular_mask) == 0:
            return []

        from ..planning.fill_strategies import FillStrategyEngine

        specular_strokes: List[List[Tuple[int, int]]] = []
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(specular_mask, connectivity=8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            rx = stats[i, cv2.CC_STAT_LEFT]
            ry = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = int(round(centroids[i][0])), int(round(centroids[i][1]))

            if area <= 60 and max(w, h) <= 20:
                # Micro-traço compacto centrado no brilho/catchlight
                half_w = max(1, w // 2)
                p1 = (int(cx - half_w), int(cy))
                p2 = (int(cx + half_w), int(cy))
                specular_strokes.append([p1, p2])
            else:
                # Cavidade maior (olho de cartoon/anime): preenche com scanlines respeitando insetting
                comp_mask = (labels == i).astype(np.uint8) * 255
                sub_strokes = FillStrategyEngine.generate_region_strokes(
                    comp_mask, (rx, ry, w, h),
                    strategy="horizontal_scanline",
                    brush_width=brush_width,
                    brush_spacing=max(2, int(round(brush_width * 0.75)))
                )
                for s in sub_strokes:
                    if len(s) >= 2:
                        specular_strokes.append([(int(pt[0]), int(pt[1])) for pt in s])

        return specular_strokes

    @classmethod
    def sample_stroke_color(cls, image: np.ndarray, stroke: List[Tuple[int, int]],
                            mean_img_lum: Optional[float] = None) -> Tuple[int, int, int]:
        """
        Amostra a cor RGB original da feição ao longo da trajetória de um traço.
        Examina a vizinhança imediata para capturar a cor real do pigmento/tinta,
        evitando contaminação com a cor do fundo.
        """
        if not stroke:
            return (0, 0, 0)
        h, w = image.shape[:2]
        if len(image.shape) != 3 or image.shape[2] != 3:
            samples = [image[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] for x, y in stroke]
            v = int(round(float(np.median(samples))))
            return (v, v, v)

        neighborhood_samples = []
        for x, y in stroke:
            ix = int(np.clip(x, 0, w - 1))
            iy = int(np.clip(y, 0, h - 1))
            patch = image[max(0, iy - 1):min(h, iy + 2), max(0, ix - 1):min(w, ix + 2)]
            neighborhood_samples.extend(patch.reshape(-1, 3))

        arr = np.array(neighborhood_samples, dtype=np.float32)
        lum = 0.299 * arr[:, 2] + 0.587 * arr[:, 1] + 0.114 * arr[:, 0]
        if mean_img_lum is None:
            mean_img_lum = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)))

        # Em fundo claro, o detalhe estrutural é o traço/pigmento mais escuro
        if mean_img_lum > 128.0:
            sorted_idx = np.argsort(lum)
            k = max(1, int(len(sorted_idx) * 0.35))
            chosen = arr[sorted_idx[:k]]
        else:
            sorted_idx = np.argsort(lum)
            k = max(1, int(len(sorted_idx) * 0.35))
            chosen = arr[sorted_idx[-k:]]

        mean_val = np.median(chosen, axis=0)
        b, g, r = mean_val
        return (int(round(r)), int(round(g)), int(round(b)))

    @classmethod
    def generate_colored_detail_strokes(cls, image: np.ndarray,
                                        min_stroke_length: float = 6.0,
                                        exclude_mask: Optional[np.ndarray] = None) -> List[Tuple[List[Tuple[int, int]], Tuple[int, int, int]]]:
        """
        Extrai traços de detalhes anatômicos/estruturais mantendo a cor original da imagem de referência.
        Bordas escuras mantêm cor escura; branco é estritamente evitado a menos que a referência seja clara.
        """
        edges = cls.extract_micro_contrast_edges(image)
        if exclude_mask is not None and np.count_nonzero(exclude_mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_exclude = cv2.dilate(exclude_mask, kernel)
            edges = cv2.bitwise_and(edges, cv2.bitwise_not(dilated_exclude))

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        # Pré-computa a luminância média da imagem uma única vez
        mean_img_lum = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))) if len(image.shape) == 3 and image.shape[2] == 3 else 128.0

        colored_strokes: List[Tuple[List[Tuple[int, int]], Tuple[int, int, int]]] = []
        for cnt in contours:
            length = cv2.arcLength(cnt, closed=False)
            if length >= min_stroke_length:
                approx = cv2.approxPolyDP(cnt, epsilon=0.8, closed=False)
                pts = [tuple(p[0]) for p in approx if len(p[0]) == 2]
                if len(pts) >= 2:
                    stroke_rgb = cls.sample_stroke_color(image, pts, mean_img_lum=mean_img_lum)
                    colored_strokes.append((pts, stroke_rgb))

        return colored_strokes

    @classmethod
    def generate_micro_detail_strokes(cls, image: np.ndarray,
                                      specular_mask: Optional[np.ndarray] = None,
                                      min_stroke_length: float = 6.0,
                                      only_specular: bool = False) -> List[List[Tuple[int, int]]]:
        """
        Gera pinceladas de micro-precisão para acabamento fino com o Pincel 1 (ultra-fino).
        Se only_specular=True, retorna apenas os toques em reflexos claros especulares.
        """
        if only_specular:
            return cls.generate_specular_highlight_strokes(specular_mask)

        edges = cls.extract_micro_contrast_edges(image)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        detail_strokes: List[List[Tuple[int, int]]] = []
        for cnt in contours:
            length = cv2.arcLength(cnt, closed=False)
            if length >= min_stroke_length:
                # Simplificação adaptativa mantendo curvas finas
                approx = cv2.approxPolyDP(cnt, epsilon=0.8, closed=False)
                pts = [tuple(p[0]) for p in approx if len(p[0]) == 2]
                if len(pts) >= 2:
                    detail_strokes.append(pts)

        # Adiciona micro-traços nos brilhos especulares (se fornecidos)
        spec_strokes = cls.generate_specular_highlight_strokes(specular_mask)
        detail_strokes.extend(spec_strokes)

        return detail_strokes
