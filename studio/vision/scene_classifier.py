"""
Classificador de Cena, Detecção de Feições Anatômicas e Mapa de Importância Multimodal.
Implementa a Fase 13 (13.1 Classificador de Imagem, 13.2 Faces e Landmarks, 13.3 Segmentação Figura-Fundo e 13.4 Mapa de Importância) do Roadmap.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional
import os
import cv2
import numpy as np


class ImageClass(str, Enum):
    PORTRAIT = "portrait"
    PHOTO_GENERAL = "photo_general"
    ANIME = "anime"
    LINE_ART = "line_art"
    LOGO = "logo"
    TEXT_HEAVY = "text_heavy"
    MIXED = "mixed"


@dataclass
class FacialLandmarks:
    """Coordenadas e regiões faciais estimadas (Fase 13.2)."""
    face_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    nose: Optional[Tuple[int, int]] = None
    mouth: Optional[Tuple[int, int]] = None
    face_contour: List[Tuple[int, int]] = field(default_factory=list)
    confidence: float = 0.0


class SceneUnderstandingEngine:
    """Motor de Compreensão de Cena e Mapa de Importância Semântica."""

    @classmethod
    def classify_image(cls, img_bgr: np.ndarray) -> Tuple[ImageClass, float, Dict[str, Any]]:
        """
        Classificador de imagem analítico com base em propriedades visuais intrínsecas (Fase 13.1):
        - line_art: dominância de traços finos sobre fundo uniforme (poucas cores, alto gradiente)
        - logo: poucas cores sólidas e formas geométricas vetoriais
        - portrait: presença de tons de pele humana e feições faciais
        - anime: paleta saturada com contornos pretos nítidos
        - photo_general: textura contínua e distribuição de cor complexa
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        # 1. Análise de bordas e gradiente
        edges = cv2.Canny(gray, 60, 180)
        edge_ratio = float(np.count_nonzero(edges) / (h * w))

        # 2. Número de cores únicas quantizadas
        small = cv2.resize(img_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        quant = (small // 32) * 32
        unique_colors_count = len(np.unique(quant.reshape(-1, 3), axis=0))

        # 3. Detecção de tons de pele no espaço YCrCb
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        # Faixa comum de pele em CrCb
        skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        skin_ratio = float(np.count_nonzero(skin_mask) / (h * w))

        # 4. Proporção do fundo dominante (Line-Art tem fundo ocupando > 75% da área)
        counts = np.bincount(gray.ravel())
        dominant_bg_ratio = float(np.max(counts) / (h * w))

        # 5. Detecção de faces via Haar Cascade do OpenCV
        face_detected = False
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(40, 40))
                face_detected = len(faces) > 0
        except Exception:
            pass

        metrics = {
            "edge_ratio": round(edge_ratio, 3),
            "unique_colors": unique_colors_count,
            "skin_ratio": round(skin_ratio, 3),
            "dominant_bg_ratio": round(dominant_bg_ratio, 3),
            "face_detected": face_detected,
        }

        # Decisão das regras
        # Line-Art: traços finos sobre fundo esmagadoramente dominante (> 75%)
        if dominant_bg_ratio >= 0.75 and edge_ratio > 0.003 and unique_colors_count <= 10 and skin_ratio < 0.05:
            return ImageClass.LINE_ART, 0.96, metrics
        elif unique_colors_count <= 4 and edge_ratio > 0.003 and skin_ratio < 0.05:
            return ImageClass.LINE_ART, 0.95, metrics
        elif edge_ratio > 0.02 and unique_colors_count <= 8 and skin_ratio < 0.05:
            return ImageClass.LINE_ART, 0.92, metrics
        elif unique_colors_count <= 14 and edge_ratio < 0.06 and not face_detected:
            return ImageClass.LOGO, 0.88, metrics
        elif face_detected or (skin_ratio > 0.15 and unique_colors_count > 15):
            return ImageClass.PORTRAIT, 0.92, metrics
        elif np.mean(sat) > 100 and edge_ratio > 0.05 and unique_colors_count < 35:
            return ImageClass.ANIME, 0.82, metrics
        elif edge_ratio < 0.02 and unique_colors_count <= 6:
            return ImageClass.TEXT_HEAVY, 0.75, metrics
        
        return ImageClass.PHOTO_GENERAL, 0.85, metrics

    @classmethod
    def detect_facial_landmarks(cls, img_bgr: np.ndarray) -> Optional[FacialLandmarks]:
        """Detecta face e estima pontos de olhos, nariz e boca (Fase 13.2)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        try:
            cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            cascade_eye = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            faces = cascade_face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))
            if len(faces) == 0:
                return None

            # Pega a maior face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi_gray = gray[y:y+h, x:x+w]
            eyes = cascade_eye.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))

            left_eye_pt, right_eye_pt = None, None
            if len(eyes) >= 2:
                sorted_eyes = sorted(eyes, key=lambda e: e[0])
                ex1, ey1, ew1, eh1 = sorted_eyes[0]
                ex2, ey2, ew2, eh2 = sorted_eyes[1]
                left_eye_pt = (x + ex1 + ew1 // 2, y + ey1 + eh1 // 2)
                right_eye_pt = (x + ex2 + ew2 // 2, y + ey2 + eh2 // 2)
            else:
                # Estimativa geométrica proporcional
                left_eye_pt = (x + int(w * 0.33), y + int(h * 0.38))
                right_eye_pt = (x + int(w * 0.67), y + int(h * 0.38))

            nose_pt = (x + int(w * 0.50), y + int(h * 0.58))
            mouth_pt = (x + int(w * 0.50), y + int(h * 0.78))

            contour = [
                (x + int(w * 0.1), y + int(h * 0.4)),
                (x + int(w * 0.2), y + int(h * 0.8)),
                (x + int(w * 0.5), y + int(h * 0.98)),
                (x + int(w * 0.8), y + int(h * 0.8)),
                (x + int(w * 0.9), y + int(h * 0.4)),
            ]

            return FacialLandmarks(
                face_bbox=(x, y, w, h),
                left_eye=left_eye_pt,
                right_eye=right_eye_pt,
                nose=nose_pt,
                mouth=mouth_pt,
                face_contour=contour,
                confidence=0.85,
            )
        except Exception:
            return None

    @classmethod
    def segment_subject_background(cls, img_bgr: np.ndarray) -> np.ndarray:
        """
        Segmentação leve figura-fundo (Fase 13.3).
        Retorna máscara binária uint8 onde subject = 255 e background = 0.
        """
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (min(w, 240), min(h, 240)), interpolation=cv2.INTER_AREA)
        sh, sw = small.shape[:2]

        # Saliency spectral residual do OpenCV ou gradiente central
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Cria prior central gaussiano (objetos tipicamente ficam no terço central)
        cy, cx = sh / 2.0, sw / 2.0
        y_grid, x_grid = np.ogrid[:sh, :sw]
        dist_center = np.sqrt(((x_grid - cx) / (sw / 2.0)) ** 2 + ((y_grid - cy) / (sh / 2.0)) ** 2)
        center_prior = np.clip(1.0 - dist_center * 0.7, 0.0, 1.0)

        # Gradientes locais
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        mag_norm = mag / (np.max(mag) + 1e-6)

        combined_score = center_prior * 0.55 + mag_norm * 0.45
        mask_small = np.where(combined_score > 0.35, 255, 0).astype(np.uint8)

        # Upscale para o tamanho original
        return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    @classmethod
    def build_importance_map(
        cls,
        img_bgr: np.ndarray,
        landmarks: Optional[FacialLandmarks] = None,
        subject_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Mapa de Importância Multimodal (Fase 13.4).
        Combina semântica anatômica, saliência, contraste de bordas e textura.
        Retorna matriz float32 normalizada em [0.0, 1.0].
        """
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Gradiente / Contraste de Bordas
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edges_smooth = cv2.GaussianBlur(edges, (9, 9), 3.0)

        # 2. Saliência de contraste local
        local_mean = cv2.blur(gray.astype(np.float32), (21, 21))
        contrast = np.abs(gray.astype(np.float32) - local_mean) / 255.0

        # Base de importância: 50% contraste + 50% bordas
        importance = contrast * 0.5 + edges_smooth * 0.5

        # 3. Incorpora Subject Mask se disponível (+30% de importância no primeiro plano)
        if subject_mask is not None:
            sub_norm = (subject_mask > 0).astype(np.float32)
            importance = importance * 0.7 + sub_norm * 0.3

        # 4. Elevação de Prioridade para Feições Anatômicas Críticas (Olhos e Boca)
        if landmarks is not None:
            anatomical_boost = np.zeros((h, w), dtype=np.float32)
            for pt in [landmarks.left_eye, landmarks.right_eye]:
                if pt:
                    cv2.circle(anatomical_boost, pt, max(12, int(w * 0.05)), 1.0, -1)
            if landmarks.mouth:
                cv2.circle(anatomical_boost, landmarks.mouth, max(14, int(w * 0.06)), 0.8, -1)
            
            anatomical_boost_smooth = cv2.GaussianBlur(anatomical_boost, (15, 15), 5.0)
            importance = np.clip(importance + anatomical_boost_smooth * 0.5, 0.0, 1.0)

        # Normaliza estritamente em [0.0, 1.0]
        max_v = float(np.max(importance))
        if max_v > 0.0:
            importance /= max_v

        return importance.astype(np.float32)
