"""
Registro Geométrico e Alinhamento Sub-Pixel de Screenshots do Canvas.
Implementa a Fase 6 (6.2 Canvas Registration com RegistrationResult estrito e Fail-Closed) do Roadmap.
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict, Iterator
import cv2
import numpy as np


@dataclass
class RegistrationResult:
    """
    Resultado detalhado do registro de imagem do canvas.
    Permite desempacotamento como tupla (registered_image, success) para retrocompatibilidade.
    """
    method: str
    confidence: float
    correlation: float
    reprojection_error: float
    anchor_residual: float
    homography: Optional[np.ndarray] = None
    registered_image: Optional[np.ndarray] = None
    success: bool = False
    details: Optional[Dict[str, Any]] = None

    def is_reliable(self, min_confidence: float = 0.65) -> bool:
        """Verifica se o alinhamento é confiavel segundo o limiar fail-closed."""
        return self.success and (self.confidence >= min_confidence)

    def __iter__(self) -> Iterator[Any]:
        """Permite unpacking (aligned_img, success)."""
        yield self.registered_image
        yield self.success


class CanvasRegistrationEngine:
    """
    Engine de registro de imagem e alinhamento geométrico/óptico do canvas.
    Substitui fallbacks cegos por avaliação quantitativa de métricas e fail-closed.
    """

    @staticmethod
    def register(
        screenshot_bgr: np.ndarray,
        target_bgr: np.ndarray,
        canvas_bbox: Optional[Tuple[int, int, int, int]] = None,
        transform: Any = None,
        min_confidence: float = 0.65,
    ) -> RegistrationResult:
        """
        Executa registro estrito do screenshot em relação à imagem-alvo.
        Retorna RegistrationResult completo.
        """
        if screenshot_bgr is None or target_bgr is None:
            return RegistrationResult(
                method="none",
                confidence=0.0,
                correlation=0.0,
                reprojection_error=999.0,
                anchor_residual=999.0,
                success=False,
            )

        target_h, target_w = target_bgr.shape[:2]
        h_screen, w_screen = screenshot_bgr.shape[:2]

        # 1. Determina a Bounding Box Física Conhecida (Geometry-First Prior)
        if canvas_bbox is not None:
            sx, sy, sw, sh = canvas_bbox
        elif transform is not None:
            sx = getattr(transform, "offset_x", 80)
            sy = getattr(transform, "offset_y", 380)
            sw = getattr(transform, "width", target_w) or target_w
            sh = getattr(transform, "height", target_h) or target_h
        else:
            sx = int((w_screen - target_w) / 2)
            sy = int((h_screen - target_h) / 2)
            sw, sh = target_w, target_h

        # Garante limites válidos dentro do screenshot físico
        crop_y1 = max(0, min(h_screen - 1, sy))
        crop_y2 = max(0, min(h_screen, sy + sh))
        crop_x1 = max(0, min(w_screen - 1, sx))
        crop_x2 = max(0, min(w_screen, sx + sw))

        cropped = screenshot_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.shape[0] < 30 or cropped.shape[1] < 30:
            fallback_resized = cv2.resize(screenshot_bgr, (target_w, target_h))
            return RegistrationResult(
                method="failed_crop",
                confidence=0.0,
                correlation=0.0,
                reprojection_error=999.0,
                anchor_residual=999.0,
                registered_image=fallback_resized,
                success=False,
            )

        resized_crop = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)

        gray_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)

        # 2. Refinamento Sub-Pixel via ECC (Enhanced Correlation Coefficient)
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)

            cc, warp_matrix = cv2.findTransformECC(
                gray_target, gray_crop, warp_matrix, cv2.MOTION_TRANSLATION, criteria, inputMask=None, gaussFiltSize=5
            )

            tx = float(abs(warp_matrix[0, 2]))
            ty = float(abs(warp_matrix[1, 2]))
            reproj_err = float(np.sqrt(tx ** 2 + ty ** 2))

            # Correlação normalizada entre o resultado ajustado e o target
            if reproj_err < 30.0 and cc >= 0.50:
                aligned = cv2.warpAffine(
                    resized_crop, warp_matrix, (target_w, target_h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE
                )
                conf = float(np.clip(cc, 0.0, 1.0))
                is_succ = conf >= min_confidence
                return RegistrationResult(
                    method="ecc",
                    confidence=conf,
                    correlation=float(cc),
                    reprojection_error=reproj_err,
                    anchor_residual=reproj_err,
                    homography=warp_matrix,
                    registered_image=aligned,
                    success=is_succ,
                    details={"tx": tx, "ty": ty, "ecc_cc": float(cc)},
                )
        except Exception:
            pass

        # 3. Fallback Óptico: ORB Feature Matching
        try:
            orb = cv2.ORB_create(nfeatures=600)
            kp1, des1 = orb.detectAndCompute(gray_crop, None)
            kp2, des2 = orb.detectAndCompute(gray_target, None)

            if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda m: m.distance)
                good_matches = matches[:min(50, len(matches))]

                if len(good_matches) >= 6:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    warp_mat, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if warp_mat is not None and inliers is not None:
                        num_inliers = int(np.sum(inliers))
                        inlier_ratio = num_inliers / max(1, len(good_matches))
                        
                        # Calcula reprojection residual
                        transformed_src = cv2.transform(src_pts, warp_mat)
                        residuals = np.linalg.norm(transformed_src - dst_pts, axis=2)
                        mean_residual = float(np.mean(residuals))

                        if inlier_ratio >= 0.4 and mean_residual < 25.0:
                            aligned = cv2.warpAffine(
                                resized_crop, warp_mat, (target_w, target_h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
                            )
                            conf = float(np.clip(inlier_ratio * (1.0 - min(1.0, mean_residual / 50.0)), 0.0, 1.0))
                            is_succ = conf >= min_confidence
                            return RegistrationResult(
                                method="orb",
                                confidence=conf,
                                correlation=float(inlier_ratio),
                                reprojection_error=mean_residual,
                                anchor_residual=mean_residual,
                                homography=warp_mat,
                                registered_image=aligned,
                                success=is_succ,
                                details={"inliers": num_inliers, "total_matches": len(good_matches)},
                            )
        except Exception:
            pass

        # 4. Fallback Geométrico Sem Confirmação Óptica (Fail-Closed: Nunca retornar success=True cego)
        # Calcula correlação de intensidade direta entre o crop e o target
        norm_crop = gray_crop.astype(np.float32)
        norm_target = gray_target.astype(np.float32)
        norm_crop -= np.mean(norm_crop)
        norm_target -= np.mean(norm_target)
        denom = (np.linalg.norm(norm_crop) * np.linalg.norm(norm_target)) + 1e-6
        direct_corr = float(np.sum(norm_crop * norm_target) / denom)
        direct_conf = float(np.clip(max(0.0, direct_corr), 0.0, 0.55))

        return RegistrationResult(
            method="geometric_prior_unconfirmed",
            confidence=direct_conf,
            correlation=direct_corr,
            reprojection_error=15.0,
            anchor_residual=15.0,
            homography=np.eye(2, 3, dtype=np.float32),
            registered_image=resized_crop,
            success=False,  # FAIL-CLOSED: Não confiar cegamente em crop sem alinhamento óptico
            details={"warning": "Nenhum método óptico convergiu. Fallback geométrico não confirmado."},
        )

    @staticmethod
    def align_screenshot_to_target(
        screenshot_bgr: np.ndarray,
        target_bgr: np.ndarray,
        canvas_bbox: Optional[Tuple[int, int, int, int]] = None,
        transform: Any = None,
        min_confidence: float = 0.65,
    ) -> RegistrationResult:
        """
        Método unificado. Retorna RegistrationResult, que também se desempacota
        como (registered_image, success) para manter 100% de compatibilidade.
        """
        return CanvasRegistrationEngine.register(
            screenshot_bgr, target_bgr, canvas_bbox=canvas_bbox, transform=transform, min_confidence=min_confidence
        )
