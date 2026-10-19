"""
Segmentador de Regiões v2 com Superpixels Coerentes e Extração de Características.
Implementa a Seção 12 do Plano de Evolução Completo (Region Engine v2).
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
from ..core.models import RegionFeatures
from .saliency import SaliencyEngine
from .texture_tensor import TextureTensorEngine

class SuperpixelEngine:
    @staticmethod
    def segment_superpixels_lab(img_bgr: np.ndarray, num_superpixels: int = 150) -> Tuple[np.ndarray, int]:
        """
        Segmenta a imagem em superpixels coerentes no espaço perceptual CIE Lab.
        Usa o algoritmo SLIC se disponível no OpenCV contrib ou Watershed adaptativo em Lab.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        
        # Tenta usar o módulo SLIC nativo do OpenCV
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createSuperpixelSLIC"):
            slic = cv2.ximgproc.createSuperpixelSLIC(lab, algorithm=cv2.ximgproc.SLIC, region_size=int(np.sqrt((img_bgr.shape[0] * img_bgr.shape[1]) / num_superpixels)), ruler=10.0)
            slic.iterate(8)
            labels = slic.getLabels()
            num_labels = slic.getNumberOfSuperpixels()
            return labels, num_labels
            
        # Fallback robusto e universal: K-Means espacial + Watershed morfológico em Lab
        h, w = img_bgr.shape[:2]
        step_y = max(10, h // int(np.sqrt(num_superpixels)))
        step_x = max(10, w // int(np.sqrt(num_superpixels)))
        
        markers = np.zeros((h, w), dtype=np.int32)
        mark_id = 1
        for y in range(step_y // 2, h, step_y):
            for x in range(step_x // 2, w, step_x):
                markers[y, x] = mark_id
                mark_id += 1
                
        # Gradiente morfológico para fronteiras do Watershed
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        grad_3c = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        
        cv2.watershed(grad_3c, markers)
        # Em watershed, -1 representa linhas de fronteira entre bacias (Seção 11.2 da Auditoria Real).
        # Preenche fronteiras (-1) propagando o marcador vizinho mais próximo para evitar colisão com label 0:
        cleaned_markers = markers.copy()
        if np.any(cleaned_markers < 0):
            boundary_mask = (cleaned_markers < 0)
            cleaned_markers[boundary_mask] = 0
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            for _ in range(3):
                if not np.any(cleaned_markers == 0):
                    break
                dilated = cv2.dilate(cleaned_markers.astype(np.float32), kernel).astype(np.int32)
                cleaned_markers = np.where(cleaned_markers == 0, dilated, cleaned_markers)

        # Mapeia identificadores únicos de 0 a K-1 de forma contígua e sem saltos (Seção 11.3)
        unique_labels, inverse = np.unique(cleaned_markers, return_inverse=True)
        labels = inverse.reshape(markers.shape).astype(np.int32)
        num_labels = int(len(unique_labels))
        return labels, num_labels

    @classmethod
    def extract_features(cls, img_bgr: np.ndarray, labels: np.ndarray, num_labels: int,
                         saliency_map: np.ndarray = None, angles_deg: np.ndarray = None,
                         coherence: np.ndarray = None) -> List[RegionFeatures]:
        """Extrai o vetor de características completo RegionFeatures para cada superpixel em float32 Lab."""
        if saliency_map is None:
            saliency_map = SaliencyEngine.compute_saliency_map(img_bgr)
        if angles_deg is None or coherence is None:
            angles_deg, coherence = TextureTensorEngine.compute_orientation_field(img_bgr)
            
        # Converte para float32 Lab com L* em [0, 100] e a*, b* com sinais reais (Seção 11.1)
        img_rgb_float = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lab_float = cv2.cvtColor(img_rgb_float, cv2.COLOR_RGB2Lab)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        
        features_list = []
        # Recorta cada label pela sua bbox em vez de criar uma máscara do frame inteiro por superpixel
        from scipy import ndimage
        slices = ndimage.find_objects(labels.astype(np.int32) + 1, max_label=num_labels)
        for l_id in range(num_labels):
            sl = slices[l_id] if l_id < len(slices) else None
            if sl is None:
                continue
            sl_y, sl_x = sl
            mask = (labels[sl] == l_id).astype(np.uint8)
            area = int(np.count_nonzero(mask))
            if area < 10:
                continue
                
            pixels_lab = lab_float[sl][mask > 0]
            mean_l = float(np.mean(pixels_lab[:, 0]))
            mean_a = float(np.mean(pixels_lab[:, 1]))
            mean_b = float(np.mean(pixels_lab[:, 2]))
            variance = float(np.var(pixels_lab[:, 0]) + np.var(pixels_lab[:, 1]) + np.var(pixels_lab[:, 2]))
            
            # Centroide e Bounding Box
            pts = np.where(mask > 0)
            cy = int(np.mean(pts[0])) + sl_y.start
            cx = int(np.mean(pts[1])) + sl_x.start
            min_y, max_y = int(np.min(pts[0])) + sl_y.start, int(np.max(pts[0])) + sl_y.start
            min_x, max_x = int(np.min(pts[1])) + sl_x.start, int(np.max(pts[1])) + sl_x.start
            bbox = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            
            reg_saliency = SaliencyEngine.get_region_saliency(saliency_map[sl], mask)
            dom_angle = TextureTensorEngine.get_region_dominant_angle(angles_deg[sl], coherence[sl], mask)
            edge_density = float(np.count_nonzero(edges[sl][mask > 0])) / float(area)
            
            rf = RegionFeatures(
                region_id=l_id,
                id=l_id,
                mean_lab=(mean_l, mean_a, mean_b),
                lab_variance=variance,
                texture_angle=dom_angle,
                edge_density=edge_density,
                area=area,
                centroid=(cx, cy),
                bbox=bbox,
                neighbors=[],
                saliency=reg_saliency
            )
            features_list.append(rf)
            
        return features_list
