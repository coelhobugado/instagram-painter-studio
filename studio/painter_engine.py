"""
Motor Central de Pintura Computacional Avançado (Painter Engine v3 Enterprise).
Integração completa dos módulos de Visão (CIE Lab, CIEDE2000, Saliência, Tensor de Estrutura, Superpixels & RAG),
Planejamento (Fill Strategies com Path Safety para buracos, Layer Graph calibrado, Scheduler v2 com 2-opt),
Edge Hierarchy harmônica, Sanitização de Caminhos e Simulador Calibrado em Coordenadas do Canvas Local.
"""
import os
import hashlib
import threading
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from . import config
from .vision.perceptual_color import PerceptualColorEngine
from .vision.saliency import SaliencyEngine
from .vision.texture_tensor import TextureTensorEngine
from .vision.stencil_extractor import AdaptiveStencilExtractor
from .vision.superpixels import SuperpixelEngine
from .vision.region_graph import RegionAdjacencyGraph
from .vision.semantic_features import SemanticFeatureExtractor
from .vision.structure_tensor import StructureTensorEngine
from .planning.fill_strategies import FillStrategyEngine
from .planning.layer_graph import LayerGraphEngine
from .planning.scheduler_v2 import StrokeSchedulerV2
from .planning.path_sanitizer import PathSanitizer
from .planning.line_art_engine import LineArtEngine
from .planning.text_engine import TextEngine
from .planning.perceptual_optimizer import PerceptualPlanOptimizer
from .planning.motion_compiler import MotionCompiler, CompiledMotionPlan
from .planning.painting_plan import PaintingPlan, PaintingRegion, StrokeStrategy, LODLevel
from .planning.inspector_model import PaintingInspectorManager
from .device.calibration import HardwareCalibrationEngine
from .device.quality_gate import QualityGate
from .simulation.simulator_v2 import CalibratedSimulatorV2
from .core.models import QualityPreset, StrokePlan, CanvasTransform
from .vision.scene_classifier import SceneUnderstandingEngine, ImageClass
from .vision.memory_optimizer import ROIMask, RegionDict

_CV_KMEANS_LOCK = threading.Lock()

class PainterEngine:
    @staticmethod
    def _seeded_kmeans(samples, k, criteria, attempts, seed):
        # O estado aleatório do OpenCV é global; a seção crítica mantém as
        # prévias simultâneas reproduzíveis.
        with _CV_KMEANS_LOCK:
            cv2.setRNGSeed(int(seed))
            return cv2.kmeans(samples, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    def __init__(self, lut_path: str = None, profile_dir: str = None):
        self.color_engine = PerceptualColorEngine(lut_path)
        self.simulator = CalibratedSimulatorV2(profile_dir)
        self.calibrator = HardwareCalibrationEngine(profile_dir)

    @staticmethod
    def _attach_editable_regions(plan: Dict[str, Any], grouped_regions: List[Dict[str, Any]]) -> None:
        """Preserva componentes geométricos e vincula traços às áreas clicáveis."""
        fit_img = plan["fit_image"]
        h, w = fit_img.shape[:2]
        label_map = np.zeros((h, w), dtype=np.int32)
        physical_target = np.full_like(fit_img, 22)
        regions = []
        for group in grouped_regions:
            color = group["color_info"]
            palette_index = int(color["id"])
            phys_rgb = tuple(color.get("render_rgb") or color["touch"].get("matched_rgb") or color["rgb"])
            phys_bgr = tuple(reversed(phys_rgb))
            for reg in group["regions"]:
                roi = reg["roi_mask"]
                x, y, rw, rh = roi.bbox
                active = roi.local_mask > 0
                if not np.any(active):
                    continue
                region_id = len(regions) + 1
                label_map[y:y + rh, x:x + rw][active] = region_id
                physical_target[y:y + rh, x:x + rw][active] = phys_bgr
                regions.append({
                    "id": region_id,
                    "palette_index": palette_index,
                    "bbox": roi.bbox,
                    "area": int(np.count_nonzero(active)),
                    "name": f"Área {region_id}",
                })

        for layer in plan["layers"]:
            linked = []
            for stroke in layer.get("strokes", []):
                region_id = 0
                if stroke:
                    for point in (stroke[len(stroke) // 2], stroke[0], stroke[-1]):
                        px, py = int(point[0]), int(point[1])
                        if 0 <= px < w and 0 <= py < h:
                            region_id = int(label_map[py, px])
                            if region_id and "palette_index" in layer:
                                if regions[region_id - 1]["palette_index"] != layer["palette_index"]:
                                    region_id = 0
                            if region_id:
                                break
                linked.append(region_id)
            layer["stroke_region_ids"] = linked
        used_ids = {region_id for layer in plan["layers"]
                    for region_id in layer["stroke_region_ids"] if region_id}
        if regions:
            visible = np.zeros(len(regions) + 1, dtype=np.bool_)
            visible[list(used_ids)] = True
            label_map[~visible[label_map]] = 0
            regions = [region for region in regions if region["id"] in used_ids]
        plan["region_map"] = label_map
        plan["editable_regions"] = regions
        plan["target_executable"] = physical_target
        plan["ideal_quantized_image"] = plan.get("quantized_image")
        plan["quantized_image"] = physical_target
        plan["comparison_image"] = np.hstack([fit_img, physical_target, plan["simulation_image"]])

    @staticmethod
    def _attach_stroke_editable_regions(plan: Dict[str, Any]) -> None:
        """Permite selecionar e recolorir traços nos modos de linha e stencil."""
        h, w = plan["fit_image"].shape[:2]
        label_map = np.zeros((h, w), dtype=np.int32)
        regions = []
        palette = plan.get("palette_colors", [])
        next_id = 1
        for layer in plan.get("layers", []):
            is_background = layer.get("purpose") == "background"
            palette_index = int(layer.get("palette_index", 0 if is_background else max(0, len(palette) - 1)))
            layer["palette_index"] = palette_index
            linked = []
            if is_background and layer.get("strokes"):
                region_id = next_id
                next_id += 1
                label_map[:] = region_id
                regions.append({"id": region_id, "name": "Fundo", "palette_index": palette_index,
                                "bbox": (0, 0, w, h), "area": w * h,
                                "selection_role": "background"})
                linked = [region_id] * len(layer["strokes"])
            else:
                thickness = layer.get("thickness", "min")
                from .device.brush_width import parse_width_level
                exact = parse_width_level(thickness)
                hit_width = (max(7, int(exact * 0.6)) if exact is not None
                             else {"min": 7, "med": 9, "thick": 13}.get(thickness, 7))
                pad = hit_width // 2 + 2
                for stroke in layer.get("strokes", []):
                    if len(stroke) < 2:
                        linked.append(0)
                        continue
                    pts = np.asarray(stroke, dtype=np.int32)
                    x0 = max(0, int(pts[:, 0].min()) - pad)
                    y0 = max(0, int(pts[:, 1].min()) - pad)
                    x1 = min(w, int(pts[:, 0].max()) + pad + 1)
                    y1 = min(h, int(pts[:, 1].max()) + pad + 1)
                    if x0 >= x1 or y0 >= y1:
                        linked.append(0)
                        continue
                    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
                    local = pts - np.array([x0, y0], dtype=np.int32)
                    cv2.polylines(mask, [local.reshape((-1, 1, 2))], False, 255, hit_width)
                    active = mask > 0
                    region_id = next_id
                    next_id += 1
                    label_map[y0:y1, x0:x1][active] = region_id
                    linked.append(region_id)
                    regions.append({"id": region_id, "name": f"Traço {region_id}",
                                    "palette_index": palette_index, "bbox": (x0, y0, x1 - x0, y1 - y0),
                                    "area": int(np.count_nonzero(active))})
            layer["stroke_region_ids"] = linked
        plan["region_map"] = label_map
        plan["editable_regions"] = regions

    def load_and_normalize_image(self, image_path: str) -> np.ndarray:
        """Carrega a imagem tratando canais alfa (PNG transparente) e espaços de cor."""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Não foi possível abrir a imagem: {image_path}")

        # Se tiver canal alfa (RGBA), mescla sobre fundo escuro padrão de DM
        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3] / 255.0
            bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 22
            bgr = img[:, :, :3]
            for c in range(3):
                bg[:, :, c] = (alpha * bgr[:, :, c] + (1.0 - alpha) * bg[:, :, c]).astype(np.uint8)
            img = bg
        elif len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    def preprocess_fit(self, img: np.ndarray,
                       max_w: int = config.CANVAS_SAFE_MAX_W,
                       max_h: int = config.CANVAS_SAFE_MAX_H,
                       target_cx: int = config.CANVAS_SAFE_CX,
                       target_cy: int = config.CANVAS_SAFE_CY,
                       custom_bounds: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
        """Redimensiona com aspect-ratio locking mantendo a integridade na safe zone do canvas ou em limites customizados."""
        if custom_bounds is not None:
            max_w = custom_bounds.get("max_width", custom_bounds.get("right", 1000) - custom_bounds.get("left", 80))
            max_h = custom_bounds.get("max_height", custom_bounds.get("bottom", 1950) - custom_bounds.get("top", 350))
            target_cx = custom_bounds.get("center_x", int((custom_bounds.get("left", 80) + custom_bounds.get("right", 1000)) / 2))
            target_cy = custom_bounds.get("center_y", int((custom_bounds.get("top", 350) + custom_bounds.get("bottom", 1950)) / 2))

        h, w = img.shape[:2]
        scale = min(max_w / float(w), max_h / float(h))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        offset_x = int(round(target_cx - new_w / 2.0))
        offset_y = int(round(target_cy - new_h / 2.0))

        # Suavização bilateral para remover ruídos de alta frequência preservando bordas
        smoothed = cv2.bilateralFilter(resized, 7, 45, 45)
        return smoothed, scale, (offset_x, offset_y), (new_w, new_h)

    def _enrich_plan_with_apice_features(
        self,
        plan: Dict[str, Any],
        raw_img: np.ndarray,
        fit_img: np.ndarray,
        transform: CanvasTransform,
        mode: str,
        seed: int = 42,
        time_budget_s: float = 180.0,
    ) -> Dict[str, Any]:
        """
        Enriquece o plano gerado integrando 100% dos módulos do Roadmap Ápice:
        - Classificação e Compreensão Semântica (Fase 13)
        - Artefato Central PaintingPlan com Vocabulário e LOD (Fase 7 & 8)
        - Compilação de Movimento Determinística (Fase 2)
        - Tríade de Erros e Diagnóstico (Fase 5.3)
        - Modelo de Inspeção Interativa de Regiões (Fase 19)
        - Validação Pré-Execução QualityGate (Fase 20)
        """
        plan.setdefault("seed", seed)
        plan.setdefault("source_image_hash", hashlib.sha256(raw_img.tobytes()).hexdigest()[:12])
        preset_key = "LAYERED" if mode == "layered_paint" else mode.upper()
        plan["plan_hash"] = StrokePlan.compute_plan_hash(
            seed, preset_key, plan.get("layers", []),
            image_hash=plan["source_image_hash"], transform=transform,
        )
        # 1. Classificação Semântica
        if mode in ("line_art", "text"):
            img_class = ImageClass.LINE_ART
            conf = 0.98
            sem_metrics = {"dominant_bg_ratio": 0.95}
        else:
            img_class, conf, sem_metrics = SceneUnderstandingEngine.classify_image(fit_img)

        importance_map = SceneUnderstandingEngine.build_importance_map(fit_img)
        plan["image_class"] = img_class.value
        plan["classification_confidence"] = conf
        plan["semantic_metrics"] = sem_metrics
        plan["importance_map"] = importance_map

        # 2. PaintingPlan Formal (Fases 7 e 8)
        painting_plan = PaintingPlan(image_class=img_class.value, time_budget_s=time_budget_s)
        layers = plan.get("layers", [])

        for idx, layer in enumerate(layers):
            purpose = layer.get("purpose", "")
            if "contour" in purpose or "edge" in purpose:
                strategy = StrokeStrategy.CONTOUR
                lod_min = LODLevel.LOD_2_CONTOURS_FEATURES
                importance = 0.85
            elif "detail" in purpose or "micro" in purpose:
                strategy = StrokeStrategy.DETAIL
                lod_min = LODLevel.LOD_3_MICRO_DETAILS
                importance = 0.90
            elif "specular" in purpose or "highlight" in purpose:
                strategy = StrokeStrategy.POINT_HIGHLIGHT
                lod_min = LODLevel.LOD_3_MICRO_DETAILS
                importance = 0.95
            elif "negative" in purpose or layer.get("brush") == "eraser":
                strategy = StrokeStrategy.NEGATIVE_STROKE
                lod_min = LODLevel.LOD_3_MICRO_DETAILS
                importance = 0.90
            elif "centerline" in purpose or mode == "line_art":
                strategy = StrokeStrategy.CENTERLINE
                lod_min = LODLevel.LOD_0_SILHOUETTE_MASS
                importance = 0.95
            elif "flow" in purpose or "streamline" in purpose:
                strategy = StrokeStrategy.FLOW_FILL
                lod_min = LODLevel.LOD_1_CORE_STRUCTURE
                importance = 0.80
            elif "background" in purpose:
                strategy = StrokeStrategy.MASS_FILL
                lod_min = LODLevel.LOD_0_SILHOUETTE_MASS
                importance = 0.60
            else:
                strategy = StrokeStrategy.MASS_FILL
                lod_min = LODLevel.LOD_0_SILHOUETTE_MASS
                importance = 0.75

            s_count = layer.get("stroke_count", len(layer.get("strokes", [])))
            cost_s = s_count * 0.35
            target_rgb = tuple(layer.get("color_rgb", (0, 0, 0)))

            reg = PaintingRegion(
                id=f"layer_{idx + 1}",
                semantic_type=layer.get("purpose", "general"),
                importance=importance,
                strategy=strategy,
                lod_min=int(lod_min),
                time_cost_s=cost_s,
                stroke_count=s_count,
                target_color_rgb=target_rgb,
                selected_tool=layer.get("brush", "solid"),
                passes=1,
                predicted_delta_e=float(layer.get("touch_info", {}).get("delta_e", 0.0)),
                rationale=f"Renderização da camada {layer.get('name', '')} via estratégia {strategy.value}"
            )
            painting_plan.add_region(reg)

        plan["painting_plan"] = painting_plan

        # 3. Compilação de Movimento Determinística (Fase 2)
        try:
            motion_plan = MotionCompiler.compile_plan(layers, transform=transform, target_hz=120)
            plan["compiled_motion_plan"] = motion_plan
        except Exception as e:
            motion_plan = None
            plan["motion_compiler_error"] = str(e)

        # 4. Regiões espaciais clicáveis, com identidade própria.
        from collections import Counter
        region_stroke_counts = Counter(region_id for layer in layers
                                       for region_id in layer.get("stroke_region_ids", []) if region_id)
        for region in plan.get("editable_regions", []):
            palette_index = region.get("palette_index", -1)
            palette = plan.get("palette_colors", [])
            rgb = tuple(region.get("override_rgb") or (
                palette[palette_index].get("rgb", (255, 255, 255))
                if 0 <= palette_index < len(palette) else (255, 255, 255)
            ))
            painting_plan.add_region(PaintingRegion(
                id=f"region_{region['id']}", semantic_type="editable_area",
                importance=0.75, strategy=StrokeStrategy.MASS_FILL,
                lod_min=int(LODLevel.LOD_0_SILHOUETTE_MASS),
                time_cost_s=0.0, stroke_count=region_stroke_counts[region["id"]],
                target_color_rgb=rgb, selected_tool="solid", passes=1,
                predicted_delta_e=0.0, rationale="Área selecionável da imagem",
            ))
        plan["inspector"] = PaintingInspectorManager(painting_plan, label_map=plan.get("region_map"))

        # 5. Avaliação do QualityGate Pré-Execução (Fase 20: Fail-Closed P0.2)
        sim_img = plan.get("simulation_image")
        delta_e = float("inf")
        edge_f1 = 0.0
        preview_ssim = 0.0
        cached_error = None  # (heatmap, metrics) de target_executable x simulação, reutilizado abaixo
        if sim_img is not None and fit_img is not None:
            try:
                from .feedback.error_map import ErrorMapEngine
                ref_target = plan.get("target_executable", fit_img)
                if mode == "line_art" and plan.get("target_executable") is None:
                    h_t, w_t = sim_img.shape[:2]
                    ref_target = np.full((h_t, w_t, 3), 22, dtype=np.uint8)
                    plan["target_executable"] = ref_target

                cached_error = ErrorMapEngine.compute_error_map(ref_target, sim_img)
                metrics = cached_error[1]
                delta_e = float(metrics.get("mean_delta_e", float("inf")))
                preview_ssim = float(metrics.get("ssim", 0.0))

                # Cálculo honesto e desinflado de Edge F1 estrutural (bordas Canny)
                gray_target = cv2.cvtColor(ref_target, cv2.COLOR_BGR2GRAY)
                gray_sim = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY)
                e_target = cv2.Canny(gray_target, 50, 150)
                e_sim = cv2.Canny(gray_sim, 50, 150)
                t_cnt = np.count_nonzero(e_target)
                s_cnt = np.count_nonzero(e_sim)
                if t_cnt > 0 and s_cnt > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    e_target_dilated = cv2.dilate(e_target, kernel)
                    e_sim_dilated = cv2.dilate(e_sim, kernel)
                    prec = np.count_nonzero(cv2.bitwise_and(e_sim, e_target_dilated)) / max(1, s_cnt)
                    rec = np.count_nonzero(cv2.bitwise_and(e_target, e_sim_dilated)) / max(1, t_cnt)
                    edge_f1 = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
                elif t_cnt == 0 and s_cnt == 0:
                    edge_f1 = 1.0
                else:
                    edge_f1 = 0.0
            except Exception:
                # Falhas no cálculo não recebem métricas aprovadoras
                delta_e = float("inf")
                edge_f1 = 0.0
                preview_ssim = 0.0

        time_est = float(plan.get("estimated_time_s", 0.0))
        effective_budget = float(time_budget_s)
        plan["time_budget_s"] = effective_budget
        plan["preview_delta_e"] = delta_e
        plan["preview_edge_f1"] = edge_f1
        plan["preview_ssim"] = preview_ssim

        profile_bounds = {"safe_bounds": {"left": 80, "top": 350, "right": 1000, "bottom": 1950}}
        qgate_res = QualityGate.evaluate(
            geometric_layers=layers,
            motion_plan=motion_plan,
            ui_validation_res=None,
            device_profile=profile_bounds,
            time_budget_s=effective_budget,
            preview_delta_e=delta_e,
            preview_edge_f1=edge_f1,
            preview_ssim=preview_ssim,
            require_live_ui=False,
        )
        plan["quality_gate"] = qgate_res

        # 6. Tríplice Erro do Simulador (Fase 5.3)
        if hasattr(self.simulator, "compute_triplet_errors") and sim_img is not None:
            target_exec = plan.get("target_executable", fit_img)
            triad_diag = self.simulator.compute_triplet_errors(
                original_img=fit_img, target_executable=target_exec, predicted_device=sim_img,
                target_prediction_error=cached_error[1] if cached_error is not None else None
            )
            plan["triad_diagnostics"] = triad_diag

        # Fidelidade ao original: o alvo executável já embute as limitações do
        # plano, então comparar só com ele esconde o que a conversão perdeu.
        if sim_img is not None and fit_img is not None and sim_img.shape == fit_img.shape:
            try:
                from .feedback.error_map import ErrorMapEngine
                _, original_metrics = ErrorMapEngine.compute_error_map(fit_img, sim_img)
                plan["fidelity_metrics"] = original_metrics
            except Exception:
                plan["fidelity_metrics"] = None

        if cached_error is not None:
            plan["error_heatmap"], plan["error_metrics"] = cached_error
        elif sim_img is not None:
            from .feedback.error_map import ErrorMapEngine
            reference = plan.get("target_executable", fit_img)
            plan["error_heatmap"], plan["error_metrics"] = ErrorMapEngine.compute_error_map(reference, sim_img)
        return plan

    def quantize_instagram_constrained(self, fit_img: np.ndarray, num_colors: int = 8,
                                       saliency_map: Optional[np.ndarray] = None,
                                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Instagram-Constrained Quantization (Seção 10 e 32 do Plano de Fidelidade Máxima).
        Pipeline:
            1. Amostragem representativa rápida (70% uniforme, 20% saliência, 10% extremos de cor).
            2. K-Means inicial no espaço Perceptual Lab.
            3. Snap exaustivo dos centros contínuos para a LUT física real via CIEDE2000.
            4. Deduplicação de centros que convergem para a mesma coordenada de toque física.
            5. Reatribuição exata de todos os pixels para os clusters físicos finais.
            6. Construção do Target Executável estritamente reprodutível pelo Instagram.
        Retorna:
            (target_executable_bgr, target_ideal_bgr, physical_labels_img, physical_palette)
        """
        h, w = fit_img.shape[:2]
        # OpenCV uint8 Lab usa L 0..255 e a/b deslocados; a LUT usa CIE Lab físico.
        lab = cv2.cvtColor(fit_img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        flat_lab = lab.reshape((-1, 3))
        total_pixels = flat_lab.shape[0]

        # 1. Amostragem Rápida Inteligente se imagem for grande (Seção 32 do Plano)
        if total_pixels > 100000:
            rng = np.random.default_rng(seed)
            sample_size = 100000
            n_rand = int(sample_size * 0.70)
            n_sal = int(sample_size * 0.20)
            n_ext = sample_size - n_rand - n_sal

            rand_indices = rng.choice(total_pixels, size=n_rand, replace=False)

            if saliency_map is not None and np.sum(saliency_map) > 0:
                flat_sal = (saliency_map.flatten().astype(np.float32) + 0.01)
                sal_prob = flat_sal / np.sum(flat_sal)
                sal_indices = rng.choice(total_pixels, size=n_sal, replace=True, p=sal_prob)
            else:
                sal_indices = rng.choice(total_pixels, size=n_sal, replace=False)

            flat_L = flat_lab[:, 0]
            sorted_L_idx = np.argsort(flat_L)
            half_ext = max(1, n_ext // 2)
            ext_indices = np.concatenate([sorted_L_idx[:half_ext], sorted_L_idx[-half_ext:]])

            sampled_indices = np.unique(np.concatenate([rand_indices, sal_indices, ext_indices]))
            training_pixels = flat_lab[sampled_indices]
        else:
            training_pixels = flat_lab

        # 2. K-Means inicial no espaço Lab
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
        actual_k = min(num_colors, len(training_pixels))
        _, _, initial_centers = self._seeded_kmeans(training_pixels, actual_k, criteria, 10, seed)

        centers_bgr = np.clip(
            cv2.cvtColor(np.float32([initial_centers]), cv2.COLOR_Lab2BGR)[0] * 255.0,
            0, 255,
        ).astype(np.uint8)

        # 3. Snap para a LUT física e 4. Deduplicação por coordenada de toque (Seção 10)
        unique_physical_map: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for i in range(len(initial_centers)):
            b, g, r = [int(c) for c in centers_bgr[i]]
            touch_match = self.color_engine.match_color((r, g, b))
            touch_key = self.color_engine.physical_touch_key(touch_match)

            if touch_key not in unique_physical_map:
                p_rgb = tuple(touch_match.get("matched_rgb", touch_match.get("rgb", (r, g, b))))
                p_bgr = (int(p_rgb[2]), int(p_rgb[1]), int(p_rgb[0]))
                p_lab = self.color_engine.rgb_to_lab(p_rgb[0], p_rgb[1], p_rgb[2])
                unique_physical_map[touch_key] = {
                    "touch": touch_match,
                    "render_rgb": p_rgb,
                    "render_bgr": p_bgr,
                    "render_lab": p_lab,
                    "ideal_rgbs": [(r, g, b)],
                    "touch_key": touch_key
                }
            else:
                unique_physical_map[touch_key]["ideal_rgbs"].append((r, g, b))

        physical_clusters = list(unique_physical_map.values())
        k_phys = len(physical_clusters)

        phys_centers_lab = np.array([c["render_lab"] for c in physical_clusters], dtype=np.float32)

        # 5. Reatribuição exata de todos os pixels para os centros físicos únicos
        weights = np.array([1.0, 1.2, 1.2], dtype=np.float32)
        best_phys_indices = np.empty(total_pixels, dtype=np.int32)
        for start in range(0, total_pixels, 50000):
            stop = min(start + 50000, total_pixels)
            diffs = flat_lab[start:stop, np.newaxis, :] - phys_centers_lab[np.newaxis, :, :]
            weighted_sq_dist = np.sum((diffs * weights) ** 2, axis=2)
            best_phys_indices[start:stop] = np.argmin(weighted_sq_dist, axis=1)

        physical_labels_img = best_phys_indices.reshape((h, w))

        # 6. Construção do Target Executável e Paleta
        target_executable_bgr = np.zeros_like(fit_img)
        palette_meta = []
        for p_idx, cluster in enumerate(physical_clusters):
            p_mask = (physical_labels_img == p_idx)
            pixel_count = int(np.count_nonzero(p_mask))
            cluster["id"] = p_idx
            cluster["count"] = pixel_count
            cluster["bgr"] = cluster["render_bgr"]
            cluster["rgb"] = cluster["render_rgb"]
            cluster["hex"] = f"#{cluster['render_rgb'][0]:02X}{cluster['render_rgb'][1]:02X}{cluster['render_rgb'][2]:02X}"
            cluster["L"] = float(cluster["render_lab"][0])

            target_executable_bgr[p_mask] = cluster["render_bgr"]
            palette_meta.append(cluster)

        # Target Ideal para fins de benchmark e diagnóstico comparativo
        ideal_labels = np.empty(total_pixels, dtype=np.int32)
        for start in range(0, total_pixels, 50000):
            stop = min(start + 50000, total_pixels)
            ideal_dists = np.sum(
                (flat_lab[start:stop, np.newaxis, :] - initial_centers[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            ideal_labels[start:stop] = np.argmin(ideal_dists, axis=1)
        target_ideal_bgr = centers_bgr[ideal_labels].reshape(fit_img.shape)

        return target_executable_bgr, target_ideal_bgr, physical_labels_img, palette_meta

    # --------------------------------------------------------------------------
    # PIPELINE 1: MODO STENCIL RÁPIDO (ALTO CONTRASTE / LINE-ART)
    # --------------------------------------------------------------------------
    def generate_stencil_plan(self, image_path: str, seed: int = 42,
                              custom_bounds: Optional[Dict[str, int]] = None,
                              override_ink_rgb: Optional[Tuple[int, int, int]] = None,
                              time_budget_s: float = 600.0) -> Dict[str, Any]:
        """Gera plano de desenho em coordenadas Canvas-Local com RDP adaptativo e sanitização estrita."""
        raw_img = self.load_and_normalize_image(image_path)
        fit_img, scale, (off_x, off_y), (w, h) = self.preprocess_fit(raw_img, custom_bounds=custom_bounds)
        transform = CanvasTransform(offset_x=off_x, offset_y=off_y, scale=scale)

        # 1. Mapa de Saliência
        saliency_map = SaliencyEngine.compute_saliency_map(fit_img)

        # 2. Extração de Contornos com RDP Adaptativo (Seção 25/26)
        contour_tuples = AdaptiveStencilExtractor.extract_contours(fit_img, saliency_map)
        raw_paths = [t[0] for t in contour_tuples]

        # 3. Otimização de Trajetória TSP com 2-Opt
        ordered_paths = StrokeSchedulerV2.optimize_layer_strokes(raw_paths, enable_2opt=True)

        # 4. Coordenadas estritamente CANVAS-LOCAL (Seção 4 da Auditoria Real)
        clean_local_strokes = PathSanitizer.sanitize_all_strokes(
            ordered_paths, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
        )
        total_pts = sum(len(s) for s in clean_local_strokes)

        target_ink = override_ink_rgb if override_ink_rgb is not None else (240, 240, 240)
        touch_match = self.color_engine.match_color(target_ink)
        matched_rgb = tuple(touch_match.get("matched_rgb") or touch_match.get("rgb") or target_ink)
        hex_c = f"#{target_ink[0]:02X}{target_ink[1]:02X}{target_ink[2]:02X}"

        layer_info = {
            "name": f"Camada Única: Stencil ({touch_match['name']})",
            "color_rgb": target_ink,
            "render_color_rgb": matched_rgb,
            "target_color_rgb": target_ink,
            "color_hex": hex_c,
            "brush": "solid",
            "touch_info": touch_match,
            "strokes": clean_local_strokes,
            "stroke_count": len(clean_local_strokes),
            "point_count": total_pts,
            "thickness": "min",
            "palette_index": 0
        }

        # 5. Renderização Simulada no Canvas Local (Seção 4 e 10)
        sim_canvas, cum_buffers = self.simulator.render_simulation(
            (h, w, 3), [layer_info], seed=seed
        )

        time_est_s = len(clean_local_strokes) * 0.35 + total_pts * 0.003
        img_hash = hashlib.sha256(raw_img.tobytes()).hexdigest()[:12]
        plan_hash = StrokePlan.compute_plan_hash(
            seed, "STENCIL", [layer_info], image_hash=img_hash, transform=transform
        )

        base_plan = {
            "mode": "stencil",
            "plan_hash": plan_hash,
            "transform": transform,
            "transform_dict": {"offset_x": off_x, "offset_y": off_y, "scale": scale},
            "layers": [layer_info],
            "total_strokes": len(clean_local_strokes),
            "total_points": total_pts,
            "estimated_time_s": round(time_est_s, 1),
            "simulation_image": sim_canvas,
            "target_executable": sim_canvas,
            "cumulative_layer_images": cum_buffers,
            "comparison_image": np.hstack([fit_img, sim_canvas]),
            "fit_image": fit_img,
            "custom_bounds": custom_bounds,
            "palette_colors": [{
                "name": touch_match["name"],
                "hex": hex_c,
                "rgb": target_ink,
                "matched_hex": touch_match.get("matched_hex", hex_c),
                "delta_e": touch_match.get("delta_e", 0.0),
                "palette_page": touch_match.get("palette_page", 1),
                "is_long_press": touch_match.get("is_long_press", False)
            }]
        }
        self._attach_stroke_editable_regions(base_plan)
        return self._enrich_plan_with_apice_features(base_plan, raw_img, fit_img, transform, mode="stencil", seed=seed, time_budget_s=time_budget_s)

    # --------------------------------------------------------------------------
    # PIPELINE 1B: MODO DESENHO DE LINHAS / LINE-ART (TULIPAS / VETORIAL)
    # --------------------------------------------------------------------------
    def generate_line_art_plan(self, image_path: str, seed: int = 42,
                               custom_bounds: Optional[Dict[str, int]] = None,
                               paint_background: Optional[bool] = None,
                               override_ink_rgb: Optional[Tuple[int, int, int]] = None,
                               override_bg_rgb: Optional[Tuple[int, int, int]] = None,
                               time_budget_s: float = 600.0,
                               colorize: bool = False,
                               fill_regions: bool = False) -> Dict[str, Any]:
        """
        Gera plano dedicado para Desenhos de Linhas (Requisito 3):
        - Segue o centro exato de cada linha via Esqueletização (evita contorno duplicado).
        - Preserva curvas suaves e contínuas.
        - Identifica e mantém o fundo e a cor original dos traços da referência.
        - Suporta customização de cores via override_ink_rgb e override_bg_rgb.
        - Se paint_background=True, inclui camada executável de fundo no plano.
        - Calibra a espessura física do pincel.
        """
        raw_img = self.load_and_normalize_image(image_path)
        fit_img, scale, (off_x, off_y), (w, h) = self.preprocess_fit(raw_img, custom_bounds=custom_bounds)
        transform = CanvasTransform(offset_x=off_x, offset_y=off_y, scale=scale)

        base_plan = LineArtEngine.generate_plan(
            fit_img=fit_img,
            color_engine=self.color_engine,
            simulator=self.simulator,
            transform=transform,
            seed=seed,
            custom_bounds=custom_bounds,
            paint_background=paint_background,
            override_ink_rgb=override_ink_rgb,
            override_bg_rgb=override_bg_rgb,
            calibrator=self.calibrator,
            colorize=colorize,
            fill_regions=fill_regions
        )
        self._attach_stroke_editable_regions(base_plan)
        return self._enrich_plan_with_apice_features(base_plan, raw_img, fit_img, transform, mode="line_art", seed=seed, time_budget_s=time_budget_s)

    # --------------------------------------------------------------------------
    # PIPELINE 1C: MODO TEXTO (FONTES TRUETYPE AJUSTADAS À ÁREA DO CANVAS)
    # --------------------------------------------------------------------------
    def generate_text_plan(self, text: str, font: str = "Arial", seed: int = 42,
                           custom_bounds: Optional[Dict[str, int]] = None,
                           ink_rgb: Tuple[int, int, int] = (255, 255, 255),
                           brush_px: Optional[float] = None,
                           max_cap_px: Optional[float] = None,
                           align: str = "center",
                           time_budget_s: float = 600.0,
                           stroke: str = "fine") -> Dict[str, Any]:
        """Escreve o texto com a fonte pedida no maior tamanho que cabe na área.

        Levanta TextFitError quando o texto só caberia abaixo do tamanho legível.
        """
        bounds = custom_bounds or {
            "left": config.CANVAS_X_MIN, "top": config.CANVAS_Y_MIN,
            "right": config.CANVAS_X_MAX, "bottom": config.CANVAS_Y_MAX,
        }
        base_plan = TextEngine.generate_plan(
            text, font, self.color_engine, self.simulator, bounds, ink_rgb=ink_rgb,
            brush_px=brush_px, max_cap_px=max_cap_px, align=align, seed=seed, calibrator=self.calibrator,
            stroke=stroke)
        # A GUI lê custom_bounds como "área escolhida pelo usuário"; o padrão não conta
        base_plan["custom_bounds"] = custom_bounds
        self._attach_stroke_editable_regions(base_plan)
        fit_img = base_plan["fit_image"]
        return self._enrich_plan_with_apice_features(base_plan, fit_img, fit_img, base_plan["transform"],
                                                     mode="text", seed=seed, time_budget_s=time_budget_s)

    def generate_flat_illustration_plan(self, image_path: str, seed: int = 42, num_colors: int = 10,
                                        include_outline: bool = False,
                                        custom_bounds: Optional[Dict[str, int]] = None,
                                        time_budget_s: float = 900.0,
                                        include_details: bool = True) -> Dict[str, Any]:
        """Ilustração chapada: poucas cores planas pintadas em anéis (substitui a pintura em camadas)."""
        from .planning.flat_illustration import FlatIllustrationEngine
        raw_img = self.load_and_normalize_image(image_path)
        fit_img, scale, (off_x, off_y), (w, h) = self.preprocess_fit(raw_img, custom_bounds=custom_bounds)
        transform = CanvasTransform(offset_x=off_x, offset_y=off_y, scale=scale)
        base_plan = FlatIllustrationEngine.generate_plan(
            fit_img, self.color_engine, self.simulator, transform, self.calibrator, seed=seed,
            num_colors=num_colors, include_outline=include_outline, include_details=include_details,
            custom_bounds=custom_bounds)
        self._attach_stroke_editable_regions(base_plan)
        return self._enrich_plan_with_apice_features(base_plan, raw_img, fit_img, transform, mode="flat_illustration",
                                                     seed=seed, time_budget_s=time_budget_s)

    # --------------------------------------------------------------------------
    # PIPELINE 2: MODO PINTURA ARTÍSTICA EM CAMADAS (LAYERED OIL PAINTER v3)
    # --------------------------------------------------------------------------
    def generate_layered_paint_plan(self, image_path: str,
                                    num_colors: int = 8,
                                    brush_spacing: int = 4,
                                    min_area: int = 30,
                                    seed: int = 42,
                                    use_rag: bool = False,
                                    custom_bounds: Optional[Dict[str, int]] = None,
                                    include_contours: bool = False,
                                    enable_semantic_highlights: bool = True,
                                    enable_colored_details: bool = True,
                                    enable_streamlines: bool = False,
                                    enable_chiaroscuro: bool = False,
                                    enable_spline: bool = True,
                                    fidelity_mode: bool = True,
                                    enable_artistic_effects: bool = False,
                                    optimize_in_simulation: bool = True,
                                    time_budget_s: float = 600.0) -> Dict[str, Any]:
        """
        Gera plano de pintura por camadas com quantização perceptual Lab,
        respeito estrito às bordas, fechamento de frestas, detalhes coloridos e Painter's Algorithm.
        No modo fidelidade (Requisito 4), desliga efeitos artísticos como hachuras cruzadas e curvas decorativas.
        """
        raw_img = self.load_and_normalize_image(image_path)
        fit_img, scale, (off_x, off_y), (w, h) = self.preprocess_fit(raw_img, custom_bounds=custom_bounds)
        transform = CanvasTransform(offset_x=off_x, offset_y=off_y, scale=scale)

        # Se modo de fidelidade ativo e sem efeitos artísticos explícitos, desativa hachuras e streamlines
        if fidelity_mode and not enable_artistic_effects:
            enable_chiaroscuro = False
            enable_streamlines = False

        # Extração de Brilhos Especulares e Proteção Anatômica (Pilar 1)
        specular_mask = None
        if enable_semantic_highlights:
            specular_mask = SemanticFeatureExtractor.extract_specular_highlights(fit_img)

        # 1. Mapas de Saliência e Tensor de Estrutura
        saliency_map = SaliencyEngine.compute_saliency_map(fit_img)
        if fidelity_mode and not enable_artistic_effects and not use_rag:
            angles_deg, coherence = None, None
        else:
            angles_deg, coherence = TextureTensorEngine.compute_orientation_field(fit_img)

        # Pré-computa o campo de fluxo tangencial 3D uma única vez para toda a imagem
        flow_field = None
        if enable_streamlines:
            flow_dx, flow_dy, _ = StructureTensorEngine.compute_orientation_field(fit_img)
            flow_field = (flow_dx, flow_dy)

        # 2. Segmentação Regional e Quantização Perceptual Lab
        grouped_color_regions = []
        palette_meta = []
        palette_summary = []

        if use_rag:
            # Segmentação Avançada Superpixels + RAG Integrado (Seção 10 e 11 da Auditoria Real)
            sp_labels, num_sp = SuperpixelEngine.segment_superpixels_lab(fit_img, num_superpixels=max(60, num_colors * 12))
            features = SuperpixelEngine.extract_features(fit_img, sp_labels, num_sp, saliency_map, angles_deg, coherence)
            rag = RegionAdjacencyGraph(features, sp_labels)
            merged_labels, merged_features = rag.adaptive_merge(max_delta_e=5.0)

            # Agrupa as regiões fundidas em K clusters cromáticos
            feature_labs = np.array([f.mean_lab for f in merged_features], dtype=np.float32)
            if len(feature_labs) > num_colors:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
                _, cluster_labels, cluster_centers = self._seeded_kmeans(feature_labs, num_colors, criteria, 10, seed)
                for f_idx, feat in enumerate(merged_features):
                    feat.cluster_id = int(cluster_labels[f_idx][0])
            else:
                cluster_centers = feature_labs
                for f_idx, feat in enumerate(merged_features):
                    feat.cluster_id = f_idx

            # Reconstrói imagem quantizada a partir dos centros dos clusters (LUT região -> cor do cluster)
            center_bgr = np.array([
                self.color_engine.lab_to_rgb(float(c[0]), float(c[1]), float(c[2]))[::-1]
                for c in cluster_centers
            ], dtype=np.uint8)
            region_lut = np.zeros((int(merged_labels.max()) + 1, 3), dtype=np.uint8)
            region_known = np.zeros(int(merged_labels.max()) + 1, dtype=bool)
            for feat in merged_features:
                region_lut[feat.region_id] = center_bgr[feat.cluster_id]
                region_known[feat.region_id] = True
            quantized_bgr = np.where(region_known[merged_labels][..., None], region_lut[merged_labels], fit_img)

            for c_id in range(len(cluster_centers)):
                center_lab = cluster_centers[c_id]
                # Converte Lab float32 para RGB analítico
                c_rgb = self.color_engine.lab_to_rgb(float(center_lab[0]), float(center_lab[1]), float(center_lab[2]))
                touch_match = self.color_engine.match_color(c_rgb)

                member_features = [f for f in merged_features if f.cluster_id == c_id]
                if not member_features:
                    continue

                reg_list = []
                total_color_area = 0
                for feat in member_features:
                    area = feat.area
                    rx, ry, rw_reg, rh_reg = feat.bbox
                    sub_lbl = merged_labels[ry:ry + rh_reg, rx:rx + rw_reg]
                    local_binary = (sub_lbl == feat.region_id).astype(np.uint8) * 255

                    # Requisito 5: Preservar detalhes importantes mesmo pequenos se tiverem alta saliência
                    if area < min_area:
                        is_important = (feat.saliency > 0.35)
                        if not is_important and specular_mask is not None:
                            sub_spec = specular_mask[ry:ry + rh_reg, rx:rx + rw_reg]
                            is_important = np.any(sub_spec[local_binary > 0])
                        if not is_important:
                            continue

                    L_val = float(center_lab[0])
                    if fidelity_mode and not enable_artistic_effects:
                        strategy = "horizontal_scanline"
                    elif enable_chiaroscuro and L_val < 38.0:
                        strategy = "chiaroscuro_cross"
                    elif enable_streamlines and (abs(feat.texture_angle) > 6.0 or area > 350):
                        strategy = "streamline_tensor"
                    elif abs(feat.texture_angle) > 10.0:
                        strategy = "angled_hatching"
                    elif area > 10000:
                        strategy = "edge_aware_fill"
                    else:
                        strategy = "horizontal_scanline"

                    roi = ROIMask(bbox=(rx, ry, rw_reg, rh_reg), local_mask=local_binary)
                    reg_list.append(RegionDict({
                        "roi_mask": roi,
                        "bbox": (rx, ry, rw_reg, rh_reg),
                        "area": area,
                        "angle": feat.texture_angle,
                        "strategy": strategy,
                        "saliency": feat.saliency
                    }, full_shape=(h, w)))
                    total_color_area += area

                if reg_list:
                    p_data = {
                        "id": c_id,
                        "rgb": c_rgb,
                        "hex": f"#{c_rgb[0]:02X}{c_rgb[1]:02X}{c_rgb[2]:02X}",
                        "L": float(center_lab[0]),
                        "count": total_color_area,
                        "touch": touch_match
                    }
                    palette_meta.append(p_data)
                    palette_summary.append({
                        "name": touch_match["name"],
                        "hex": p_data["hex"],
                        "rgb": c_rgb,
                        "matched_hex": touch_match["matched_hex"],
                        "delta_e": touch_match["delta_e"],
                        "palette_page": touch_match.get("palette_page", 1),
                        "is_long_press": touch_match.get("is_long_press", False)
                    })
                    grouped_color_regions.append({
                        "color_info": p_data,
                        "mean_L": p_data["L"],
                        "total_area": total_color_area,
                        "saliency": np.mean([r["saliency"] for r in reg_list]),
                        "regions": reg_list
                    })

        else:
            # Segmentação Perceptual Lab Preservando as Cores da Imagem Original
            # Mantém todos os K clusters cromáticos da foto (tons de pele, luzes, sombras)
            lab = cv2.cvtColor(fit_img, cv2.COLOR_BGR2Lab)
            pixels = lab.reshape((-1, 3)).astype(np.float32)

            # Otimização de Alta Performance: amostragem de até 40.000 pixels para encontrar os centros
            if len(pixels) > 40000:
                sample_step = max(1, len(pixels) // 40000)
                sample_pixels = pixels[::sample_step]
                criteria_sample = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
                _, _, centers = self._seeded_kmeans(sample_pixels, num_colors, criteria_sample, 3, seed)

                # Rotulação ultrarrápida em chunks para economizar memória e tempo
                chunk_size = 200000
                labels_chunks = []
                for ch_i in range(0, len(pixels), chunk_size):
                    p_chunk = pixels[ch_i:ch_i + chunk_size]
                    dists = np.sum((p_chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                    labels_chunks.append(np.argmin(dists, axis=1))
                labels = np.concatenate(labels_chunks).reshape((-1, 1))
            else:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
                _, labels, centers = self._seeded_kmeans(pixels, num_colors, criteria, 3, seed)

            labels_img = labels.reshape(fit_img.shape[:2])
            quantized_lab = centers[labels.flatten()].reshape(lab.shape).astype(np.uint8)
            quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_Lab2BGR)
            centers_bgr = cv2.cvtColor(np.uint8([centers]), cv2.COLOR_Lab2BGR)[0]

            for i in range(num_colors):
                b, g, r = [int(c) for c in centers_bgr[i]]
                L_star = float(centers[i][0]) * (100.0 / 255.0)
                pixel_count = int(np.count_nonzero(labels_img == i))
                touch_match = self.color_engine.match_color((r, g, b))

                p_data = {
                    "id": i,
                    "bgr": (b, g, r),
                    "rgb": (r, g, b),
                    "render_rgb": tuple(touch_match.get("matched_rgb", touch_match.get("rgb", (r, g, b)))),
                    "hex": f"#{r:02X}{g:02X}{b:02X}",
                    "L": L_star,
                    "count": pixel_count,
                    "touch": touch_match
                }
                palette_meta.append(p_data)
                palette_summary.append({
                    "name": touch_match["name"],
                    "hex": p_data["hex"],
                    "rgb": (r, g, b),
                    "matched_hex": touch_match.get("matched_hex", p_data["hex"]),
                    "delta_e": touch_match.get("delta_e", 0.0),
                    "palette_page": touch_match.get("palette_page", 1),
                    "is_long_press": touch_match.get("is_long_press", False)
                })

            # Requisito 5: Preservar detalhes anatômicos finos e bordas ANTES da limpeza morfológica
            edge_guide = cv2.Canny(cv2.cvtColor(fit_img, cv2.COLOR_BGR2GRAY), 50, 150)
            detail_protect_mask = (saliency_map > 0.35) | (edge_guide > 0)
            if specular_mask is not None:
                detail_protect_mask = detail_protect_mask | (specular_mask > 0)
            detail_protect_u8 = detail_protect_mask.astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            for p_info in palette_meta:
                cid = p_info["id"]
                mask = (labels_img == cid).astype(np.uint8) * 255

                # Protege detalhes finos/salientes deste tom antes que a morfologia os remova
                protected_pixels = cv2.bitwise_and(mask, detail_protect_u8)
                cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
                # Restaura integralmente os traços finos essenciais
                cleaned_mask = cv2.bitwise_or(cleaned_mask, protected_pixels)

                num_labels_cc, comp_labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
                reg_list = []
                total_color_area = 0

                for comp_id in range(1, num_labels_cc):
                    area = int(stats[comp_id, cv2.CC_STAT_AREA])
                    rx = int(stats[comp_id, cv2.CC_STAT_LEFT])
                    ry = int(stats[comp_id, cv2.CC_STAT_TOP])
                    rw_reg = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                    rh_reg = int(stats[comp_id, cv2.CC_STAT_HEIGHT])

                    sub_comp = comp_labels[ry:ry + rh_reg, rx:rx + rw_reg]
                    local_binary = (sub_comp == comp_id).astype(np.uint8) * 255

                    reg_sal = SaliencyEngine.get_region_saliency(saliency_map, local_binary, bbox=(rx, ry, rw_reg, rh_reg))

                    # Requisito 5: Preservar detalhes importantes durante a simplificação
                    if area < min_area:
                        is_important = (reg_sal > 0.35)
                        if not is_important and specular_mask is not None:
                            sub_spec = specular_mask[ry:ry + rh_reg, rx:rx + rw_reg]
                            is_important = np.any(sub_spec[local_binary > 0])
                        if not is_important:
                            continue

                    L_val = p_info["L"]
                    if fidelity_mode and not enable_artistic_effects:
                        dom_angle = 0.0
                        strategy = "horizontal_scanline"
                    else:
                        dom_angle = TextureTensorEngine.get_region_dominant_angle(angles_deg, coherence, local_binary, bbox=(rx, ry, rw_reg, rh_reg)) if angles_deg is not None else 0.0
                        if enable_chiaroscuro and L_val < 38.0:
                            strategy = "chiaroscuro_cross"
                        elif enable_streamlines and (abs(dom_angle) > 8.0 or area > 1200):
                            strategy = "streamline_tensor"
                        elif abs(dom_angle) > 10.0:
                            strategy = "angled_hatching"
                        elif area > 10000:
                            strategy = "edge_aware_fill"
                        else:
                            strategy = "horizontal_scanline"

                    roi = ROIMask(bbox=(rx, ry, rw_reg, rh_reg), local_mask=local_binary)
                    reg_list.append(RegionDict({
                        "roi_mask": roi,
                        "bbox": (rx, ry, rw_reg, rh_reg),
                        "area": area,
                        "angle": dom_angle,
                        "strategy": strategy,
                        "saliency": reg_sal
                    }, full_shape=(h, w)))
                    total_color_area += area

                if reg_list:
                    grouped_color_regions.append({
                        "color_info": p_info,
                        "mean_L": p_info["L"],
                        "total_area": total_color_area,
                        "saliency": np.mean([r["saliency"] for r in reg_list]),
                        "regions": reg_list
                    })

        # 3. Painter's Algorithm via LayerGraphEngine (Seção 14, 15 e 18 do Plano)
        ordered_color_layers = LayerGraphEngine.build_layer_order(grouped_color_regions)

        # 4. Síntese de Pinceladas em Coordenadas CANVAS-LOCAL com Path Safety, Brush-Aware Spacing e 2-Opt
        layers_plan = []
        total_strokes = 0
        total_pts = 0
        total_canvas_area = w * h

        for idx, layer_data in enumerate(ordered_color_layers):
            c_info = layer_data["color_info"]
            layer_raw_strokes = []

            purpose = layer_data.get("purpose", "color_mass")
            # Para máxima finura e fidelidade:
            # Fundo amplo (> 30% da tela) usa espessura 'med'.
            # Todas as demais camadas (tons de pele, roupas, sombras, detalhes) usam 'min' (pincel fino)!
            if purpose == "background" and layer_data.get("total_area", 0) > (total_canvas_area * 0.30):
                thickness = "med"
            else:
                thickness = "min"

            # Requisito 6: Consumir largura de pincel calibrada do perfil de hardware real
            b_width = self.calibrator.get_effective_brush_width(thickness, "solid")
            actual_spacing = max(2, int(brush_spacing))
            # Corredor da camada (união das regiões): encadeamento só pode atravessar tinta da mesma cor
            layer_corridor = np.zeros((h, w), dtype=np.uint8)

            for reg in layer_data["regions"]:
                reg_mask = reg["mask"]
                # Proteção Semântica: impede que camadas de fundo ou pele cubram o brilho dos olhos
                if enable_semantic_highlights and specular_mask is not None and layer_data.get("mean_L", 50.0) < 85.0:
                    rx, ry, rw_reg, rh_reg = reg["bbox"]
                    if np.any(specular_mask[ry:ry + rh_reg, rx:rx + rw_reg] > 0):
                        reg_mask = SemanticFeatureExtractor.create_specular_preservation_mask(
                            reg_mask, specular_mask, dilation_radius=2
                        )
                if np.count_nonzero(reg_mask) == 0:
                    continue
                cv2.bitwise_or(layer_corridor, reg_mask, dst=layer_corridor)

                strokes = FillStrategyEngine.generate_region_strokes(
                    reg_mask, reg["bbox"],
                    strategy=reg["strategy"],
                    brush_spacing=actual_spacing,
                    texture_angle=reg["angle"],
                    brush_width=b_width,
                    image=fit_img,
                    flow_field=flow_field,
                    enable_spline=enable_spline
                )
                layer_raw_strokes.extend(strokes)

            # Otimização hierárquica TSP via 2-Opt
            opt_layer_strokes = StrokeSchedulerV2.optimize_layer_strokes(
                layer_raw_strokes, enable_2opt=True, allowed_mask=layer_corridor
            )

            # Sanitização estrita em coordenadas CANVAS-LOCAL (0 <= x < w, 0 <= y < h)
            clean_local_layer_strokes = PathSanitizer.sanitize_all_strokes(
                opt_layer_strokes, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
            )
            if not clean_local_layer_strokes:
                continue

            # Otimização e Poda de Segurança Perceptual (P5)
            layer_dict_temp = {"purpose": purpose, "strokes": clean_local_layer_strokes}
            PerceptualPlanOptimizer.filter_layer_strokes_for_safety(
                layer_dict_temp,
                protected_mask=specular_mask,
                brush_width=b_width,
                canvas_shape=(h, w)
            )
            clean_local_layer_strokes = layer_dict_temp["strokes"]
            if not clean_local_layer_strokes:
                continue

            total_strokes += len(clean_local_layer_strokes)
            pts_count = sum(len(s) for s in clean_local_layer_strokes)
            total_pts += pts_count

            # Preserva a cor original exata da foto do usuário para máxima fidelidade visual
            orig_rgb = c_info["rgb"]
            orig_hex = c_info.get("hex", f"#{orig_rgb[0]:02X}{orig_rgb[1]:02X}{orig_rgb[2]:02X}")
            exec_rgb = c_info.get("render_rgb", orig_rgb)

            layers_plan.append({
                "name": f"Camada {len(layers_plan) + 1}: {c_info['touch']['name']} ({purpose})",
                "color_rgb": orig_rgb,
                "render_color_rgb": exec_rgb,
                "target_color_rgb": orig_rgb,
                "color_hex": orig_hex,
                "brush": "solid",
                "purpose": purpose,
                "touch_info": c_info["touch"],
                "strokes": clean_local_layer_strokes,
                "stroke_count": len(clean_local_layer_strokes),
                "point_count": pts_count,
                "thickness": thickness,
                "palette_index": c_info.get("id", idx)
            })

        # 5. Camadas Finais: Edge Hierarchy com Agrupamento por Touch Físico (Seções 19, 20 e 21 do Plano)
        if include_contours:
            contour_tuples = AdaptiveStencilExtractor.extract_contours(fit_img, saliency_map)
            if contour_tuples:
                # Agrupa contornos diretamente por coordenada física de toque da LUT real (Seção 21 do Plano)
                # Evita dezenas de trocas de paleta para cores teóricas que mapeiam para o mesmo toque físico
                physical_contour_groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
                for path, c_rgb in contour_tuples:
                    if len(path) < 3:
                        continue
                    touch_c = self.color_engine.match_color(c_rgb)
                    t_key = self.color_engine.physical_touch_key(touch_c)
                    if t_key not in physical_contour_groups:
                        physical_contour_groups[t_key] = {
                            "touch": touch_c,
                            "color_rgb": touch_c.get("matched_rgb", c_rgb),
                            "paths": [path]
                        }
                    else:
                        physical_contour_groups[t_key]["paths"].append(path)

                for t_key, grp_data in physical_contour_groups.items():
                    paths = grp_data["paths"]
                    touch_c = grp_data["touch"]
                    c_rgb = grp_data["color_rgb"]
                    opt_contours = StrokeSchedulerV2.optimize_layer_strokes(paths, enable_2opt=True)
                    clean_contours = PathSanitizer.sanitize_all_strokes(
                        opt_contours, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
                    )
                    if not clean_contours:
                        continue

                    c_pts = sum(len(s) for s in clean_contours)
                    total_strokes += len(clean_contours)
                    total_pts += c_pts

                    layers_plan.append({
                        "name": f"Camada {len(layers_plan) + 1}: Contornos {touch_c['name']} (contour)",
                        "color_rgb": c_rgb,
                        "render_color_rgb": c_rgb,
                        "color_hex": f"#{c_rgb[0]:02X}{c_rgb[1]:02X}{c_rgb[2]:02X}",
                        "brush": "solid",
                        "purpose": "contour",
                        "touch_info": touch_c,
                        "strokes": clean_contours,
                        "stroke_count": len(clean_contours),
                        "point_count": c_pts,
                        "thickness": "min"
                    })

        # 5b. Camada de Reflexos Claros Especulares (Requisito 1: Branco estritamente restrito aos reflexos claros)
        if enable_semantic_highlights and specular_mask is not None and np.count_nonzero(specular_mask) > 0:
            b_width_hl = self.calibrator.get_effective_brush_width("min", "solid")
            spec_strokes = SemanticFeatureExtractor.generate_specular_highlight_strokes(specular_mask, brush_width=b_width_hl)
            if spec_strokes:
                opt_spec = StrokeSchedulerV2.optimize_layer_strokes(spec_strokes, enable_2opt=True)
                clean_spec = PathSanitizer.sanitize_all_strokes(
                    opt_spec, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
                )
                if clean_spec:
                    hl_rgb = (255, 255, 255)
                    hl_touch = self.color_engine.match_color(hl_rgb)
                    hl_pts = sum(len(s) for s in clean_spec)
                    total_strokes += len(clean_spec)
                    total_pts += hl_pts
                    layers_plan.append({
                        "name": f"Camada {len(layers_plan) + 1}: Reflexos Claros Especulares (catchlight)",
                        "color_rgb": hl_rgb,
                        "render_color_rgb": hl_rgb,
                        "color_hex": "#FFFFFF",
                        "brush": "solid",
                        "purpose": "highlight",
                        "touch_info": hl_touch,
                        "strokes": clean_spec,
                        "stroke_count": len(clean_spec),
                        "point_count": hl_pts,
                        "thickness": "min"
                    })

        # 5c. Camada de Detalhes Anatômicos/Estruturais com Cores Originais (Requisito 1: Cada detalhe usa sua cor original)
        if enable_colored_details:
            detail_tuples = SemanticFeatureExtractor.generate_colored_detail_strokes(
                fit_img, exclude_mask=specular_mask
            )
            if detail_tuples:
                # Agrupa os detalhes por toque da paleta real para não trocar de cor a cada micro-traço
                color_groups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
                for path, stroke_rgb in detail_tuples:
                    if len(path) < 2:
                        continue
                    touch_m = self.color_engine.match_color(stroke_rgb)
                    t_key = self.color_engine.physical_touch_key(touch_m)
                    if t_key not in color_groups:
                        color_groups[t_key] = {
                            "touch": touch_m,
                            "rgb": stroke_rgb,
                            "render_rgb": tuple(touch_m.get("matched_rgb", stroke_rgb)),
                            "paths": [path]
                        }
                    else:
                        color_groups[t_key]["paths"].append(path)

                for t_key, grp in color_groups.items():
                    opt_paths = StrokeSchedulerV2.optimize_layer_strokes(grp["paths"], enable_2opt=True)
                    clean_paths = PathSanitizer.sanitize_all_strokes(
                        opt_paths, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
                    )
                    if not clean_paths:
                        continue

                    c_pts = sum(len(s) for s in clean_paths)
                    total_strokes += len(clean_paths)
                    total_pts += c_pts

                    layers_plan.append({
                        "name": f"Camada {len(layers_plan) + 1}: Detalhes {grp['touch']['name']} (detail)",
                        "color_rgb": grp["rgb"],
                        "render_color_rgb": grp["render_rgb"],
                        "color_hex": f"#{grp['rgb'][0]:02X}{grp['rgb'][1]:02X}{grp['rgb'][2]:02X}",
                        "brush": "solid",
                        "purpose": "colored_detail",
                        "touch_info": grp["touch"],
                        "strokes": clean_paths,
                        "stroke_count": len(clean_paths),
                        "point_count": c_pts,
                        "thickness": "min"
                    })

        # 6. Renderização Simulada Calibrada no Canvas Local
        sim_canvas, cum_buffers = self.simulator.render_simulation(
            (h, w, 3), layers_plan, seed=seed
        )

        time_est_s = total_strokes * PerceptualPlanOptimizer.SECONDS_PER_STROKE + total_pts * PerceptualPlanOptimizer.SECONDS_PER_POINT
        comparison = np.hstack([fit_img, quantized_bgr, sim_canvas])
        img_hash = hashlib.sha256(raw_img.tobytes()).hexdigest()[:12]
        plan_hash = StrokePlan.compute_plan_hash(
            seed, "LAYERED", layers_plan, image_hash=img_hash, transform=transform
        )

        plan = {
            "mode": "layered_paint",
            "plan_hash": plan_hash,
            "transform": transform,
            "transform_dict": {"offset_x": off_x, "offset_y": off_y, "scale": scale},
            "layers": layers_plan,
            "total_strokes": total_strokes,
            "total_points": total_pts,
            "estimated_time_s": round(time_est_s, 1),
            "simulation_image": sim_canvas,
            "cumulative_layer_images": cum_buffers,
            "target_executable": quantized_bgr,
            "quantized_image": quantized_bgr,
            "comparison_image": comparison,
            "fit_image": fit_img,
            "custom_bounds": custom_bounds,
            "palette_colors": palette_summary
        }

        # 7. Otimização do Desenho na Simulação antes de Executar (Requisito 7)
        if optimize_in_simulation:
            plan = self.optimize_plan_in_simulation(plan, fit_img, specular_mask=specular_mask)

        # 8. Otimização Global de Orçamento por Ganho por Segundo (P5)
        budget_pruned = PerceptualPlanOptimizer.optimize_plan_budget_and_gain(
            plan, time_budget_s=time_budget_s, image_class="layered_paint"
        )
        if budget_pruned:
            # Camadas removidas: a pré-visualização e o QualityGate precisam refletir o plano real
            sim_canvas, cum_buffers = self.simulator.render_simulation((h, w, 3), plan["layers"], seed=seed)
            plan["simulation_image"] = sim_canvas
            plan["cumulative_layer_images"] = cum_buffers
            plan["comparison_image"] = np.hstack([fit_img, quantized_bgr, sim_canvas])
            plan["plan_hash"] = StrokePlan.compute_plan_hash(
                seed, "LAYERED", plan["layers"], image_hash=img_hash, transform=transform
            )

        self._attach_editable_regions(plan, grouped_color_regions)
        return self._enrich_plan_with_apice_features(plan, raw_img, fit_img, transform, mode="layered_paint", seed=seed, time_budget_s=time_budget_s)

    def optimize_plan_in_simulation(self, plan: Dict[str, Any], fit_img: np.ndarray,
                                   max_passes: int = 1,
                                   specular_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Otimiza o desenho na simulação antes de executar (Requisito 7 e Correção de Bordas):
        1. Compara a primeira versão simulada com o original.
        2. Protege bordas estruturais e detalhes salientes contra preenchimento cego de média.
        3. Sub-segmenta regiões críticas com alta variância cromática para não apagar detalhes.
        4. Avalia e aceita correções INDIVIDUALMENTE por região, rejeitando qualquer degradação local.
        """
        sim_img = plan.get("simulation_image")
        if sim_img is None or fit_img is None:
            return plan

        h, w = fit_img.shape[:2]
        target_lab = cv2.cvtColor(fit_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        current_sim = sim_img.copy()
        current_sim_lab = cv2.cvtColor(current_sim, cv2.COLOR_BGR2Lab).astype(np.float32)

        # Mapa de erro perceptual Delta E
        error_map = np.sqrt(np.sum((target_lab - current_sim_lab) ** 2, axis=2))
        base_error = float(np.mean(error_map))

        # Requisito 4: Proteção estrita de bordas estruturais para não apagar detalhes anatômicos/pétalas/olhos
        gray = cv2.cvtColor(fit_img, cv2.COLOR_BGR2GRAY)
        edge_mask = cv2.Canny(gray, 50, 150)
        edge_protect = cv2.dilate(edge_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # Identifica áreas críticas onde a diferença é alta (> 14.0)
        critical_error_mask = (error_map > 14.0).astype(np.uint8) * 255
        # Subtrai a proteção de borda para que a otimização não tente sobrescrever arestas finas
        critical_error_mask = cv2.bitwise_and(critical_error_mask, cv2.bitwise_not(edge_protect))
        # Protege olhos e cavidades brancas contra sobreposição destrutiva
        if specular_mask is not None and np.count_nonzero(specular_mask) > 0:
            spec_protect = cv2.dilate(specular_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            critical_error_mask = cv2.bitwise_and(critical_error_mask, cv2.bitwise_not(spec_protect))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        critical_cleaned = cv2.morphologyEx(critical_error_mask, cv2.MORPH_OPEN, kernel)

        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(critical_cleaned, connectivity=8)
        if num_labels <= 1:
            return plan

        calib_w = self.calibrator.get_effective_brush_width("min", "solid")
        accepted_refinement_layers = []
        current_layers = list(plan["layers"])

        for comp_id in range(1, min(num_labels, 10)):
            if len(accepted_refinement_layers) >= 2:
                break  # Limite de 2 camadas de refinamento atingido: não avalia mais candidatos
            area = stats[comp_id, cv2.CC_STAT_AREA]
            if area < 20:
                continue

            comp_mask = (comp_labels == comp_id).astype(np.uint8) * 255
            target_patch = fit_img[comp_mask > 0]
            if len(target_patch) == 0:
                continue

            # Sub-segmentação por cor e estrutura: se a variância interna for alta, não faz média cega
            std_color = np.std(target_patch, axis=0)
            sub_regions = []
            if np.max(std_color) > 22.0 and area >= 60:
                # Divide em 2 sub-regiões cromáticas distintas via K-Means local
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                _, sub_labels, sub_centers = cv2.kmeans(target_patch.astype(np.float32), 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                coords = np.argwhere(comp_mask > 0)
                for sub_k in range(2):
                    sub_m = np.zeros((h, w), dtype=np.uint8)
                    pts = coords[sub_labels.flatten() == sub_k]
                    if len(pts) >= 15:
                        sub_m[pts[:, 0], pts[:, 1]] = 255
                        sub_regions.append((sub_m, sub_centers[sub_k]))
            else:
                mean_col = np.mean(target_patch, axis=0)
                sub_regions.append((comp_mask, mean_col))

            for sub_mask, center_bgr in sub_regions:
                if len(accepted_refinement_layers) >= 2:
                    break
                sb, sg, sr = center_bgr
                target_rgb = (int(round(sr)), int(round(sg)), int(round(sb)))
                touch_match = self.color_engine.match_color(target_rgb)
                # Requisito 2: matched_rgb garantido pela cor física executável
                matched_rgb = tuple(touch_match.get("matched_rgb") or touch_match.get("rgb") or target_rgb)

                rx, ry, rw, rh = cv2.boundingRect(sub_mask)
                local_strokes = FillStrategyEngine.generate_region_strokes(
                    sub_mask, (rx, ry, rw, rh),
                    strategy="horizontal_scanline",
                    brush_spacing=2,
                    brush_width=calib_w,
                    ensure_full_coverage=False
                )
                clean_strokes = PathSanitizer.sanitize_all_strokes(
                    local_strokes, x_min=0, x_max=w - 1, y_min=0, y_max=h - 1
                )
                if not clean_strokes:
                    continue

                cand_layer = {
                    "name": f"Otimização Local: {touch_match['name']} ({np.count_nonzero(sub_mask)}px)",
                    "color_rgb": target_rgb,
                    "render_color_rgb": matched_rgb,
                    "target_color_rgb": target_rgb,
                    "color_hex": f"#{target_rgb[0]:02X}{target_rgb[1]:02X}{target_rgb[2]:02X}",
                    "brush": "solid",
                    "purpose": "simulation_optimization",
                    "touch_info": touch_match,
                    "strokes": clean_strokes,
                    "stroke_count": len(clean_strokes),
                    "point_count": sum(len(s) for s in clean_strokes),
                    "thickness": "min"
                }

                # Avaliação Individual Local (Requisito 4):
                # Mede o erro local antes e depois da adição EXCLUSIVA deste candidato
                mask_pixels = sub_mask > 0
                local_err_before = float(np.mean(np.sqrt(np.sum((target_lab[mask_pixels] - current_sim_lab[mask_pixels]) ** 2, axis=1))))

                # Renderização incremental: só o candidato sobre a simulação atual (o simulador é determinístico)
                test_sim, _ = self.simulator.render_simulation((h, w, 3), [cand_layer], initial_canvas=current_sim)
                test_sim_lab = cv2.cvtColor(test_sim, cv2.COLOR_BGR2Lab).astype(np.float32)
                local_err_after = float(np.mean(np.sqrt(np.sum((target_lab[mask_pixels] - test_sim_lab[mask_pixels]) ** 2, axis=1))))

                # Só aceita a correção se ela melhorar significativamente o erro local desta região
                # e limita estritamente o número de camadas adicionais de refinamento a no máximo 2 (P5)
                if local_err_after < local_err_before - 1.2 and len(accepted_refinement_layers) < 2:
                    accepted_refinement_layers.append(cand_layer)
                    current_layers.append(cand_layer)
                    current_sim = test_sim
                    current_sim_lab = test_sim_lab

        if not accepted_refinement_layers:
            return plan

        # Atualiza a simulação final consolidada com apenas as correções aceitas
        final_sim, final_cum = self.simulator.render_simulation((h, w, 3), current_layers)
        final_lab = cv2.cvtColor(final_sim, cv2.COLOR_BGR2Lab).astype(np.float32)
        final_error = float(np.mean(np.sqrt(np.sum((target_lab - final_lab) ** 2, axis=2))))

        if final_error < base_error:
            plan["layers"] = current_layers
            plan["total_strokes"] += sum(l["stroke_count"] for l in accepted_refinement_layers)
            plan["total_points"] += sum(l["point_count"] for l in accepted_refinement_layers)
            plan["simulation_image"] = final_sim
            plan["cumulative_layer_images"] = final_cum
            plan["comparison_image"] = np.hstack([fit_img, final_sim])
            plan["optimized_in_simulation"] = True
            plan["optimization_delta_gain"] = round(base_error - final_error, 2)

        return plan

    @staticmethod
    def _set_layer_color(layer: Dict[str, Any], rgb: Tuple[int, int, int], touch: Dict[str, Any]) -> None:
        layer["color_rgb"] = rgb
        layer["target_color_rgb"] = rgb
        layer["render_color_rgb"] = tuple(touch["matched_rgb"])
        layer["color_hex"] = f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
        layer["touch_info"] = dict(touch)

    def _refresh_edited_plan(self, plan: Dict[str, Any], previous: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reconstrói todos os dados derivados antes de publicar uma edição."""
        fit_img = plan["fit_image"]
        h, w = fit_img.shape[:2]
        layers = plan["layers"]
        mode = plan.get("mode", "layered_paint")
        if mode in ("line_art", "stencil", "flat_illustration") and not plan.get("editable_regions"):
            self._attach_stroke_editable_regions(plan)
        bg_bgr = (22, 22, 22)
        prefix = 0
        prior_buffers = previous.get("cumulative_layer_images", []) if previous else []
        if previous and len(prior_buffers) == len(previous.get("layers", [])):
            for old, new in zip(previous["layers"], layers):
                keys = ("strokes", "render_color_rgb", "touch_info", "brush", "thickness")
                if any(old.get(key) != new.get(key) for key in keys):
                    break
                prefix += 1
        if prefix and prefix < len(layers):
            sim, tail_buffers = self.simulator.render_simulation(
                (h, w, 3), layers[prefix:], background_bgr=bg_bgr,
                seed=plan.get("seed", 42), initial_canvas=prior_buffers[prefix - 1],
            )
            cumulative = prior_buffers[:prefix] + tail_buffers
        elif prefix == len(layers) and prior_buffers:
            sim = prior_buffers[-1]
            cumulative = list(prior_buffers)
        else:
            sim, cumulative = self.simulator.render_simulation(
                (h, w, 3), layers, background_bgr=bg_bgr, seed=plan.get("seed", 42)
            )
        plan["simulation_image"] = sim
        plan["cumulative_layer_images"] = cumulative
        if mode == "layered_paint":
            plan["quantized_image"] = plan["target_executable"]
            plan["comparison_image"] = np.hstack([fit_img, plan["target_executable"], sim])
        else:
            plan["target_executable"] = sim
            plan["comparison_image"] = np.hstack([fit_img, sim])
        plan["total_strokes"] = sum(len(layer.get("strokes", [])) for layer in layers)
        plan["total_points"] = sum(len(s) for layer in layers for s in layer.get("strokes", []))
        plan["estimated_time_s"] = round(plan["total_strokes"] * PerceptualPlanOptimizer.SECONDS_PER_STROKE
                                         + plan["total_points"] * PerceptualPlanOptimizer.SECONDS_PER_POINT, 1)
        seed = plan.get("seed", 42)
        transform = plan.get("transform")
        preset_key = "LAYERED" if mode == "layered_paint" else mode.upper()
        plan["plan_hash"] = StrokePlan.compute_plan_hash(
            seed, preset_key, layers,
            image_hash=plan.get("source_image_hash", ""), transform=transform,
        )
        plan["revision"] = int(plan.get("revision", 0)) + 1
        return self._enrich_plan_with_apice_features(
            plan, fit_img, fit_img, transform, mode, seed=seed,
            time_budget_s=plan.get("time_budget_s", 600.0),
        )

    def recolor_plan(self, plan: Dict[str, Any], color_index: int, new_rgb: Tuple[int, int, int],
                     preferred_lut_id: Any = None) -> Dict[str, Any]:
        """Troca uma entrada da paleta em nova revisão, sem confundir cores iguais de áreas distintas."""
        if not plan or not 0 <= color_index < len(plan.get("palette_colors", [])):
            return plan
        rgb = tuple(int(np.clip(value, 0, 255)) for value in new_rgb)
        touch = self.color_engine.match_color(rgb, preferred_lut_id=preferred_lut_id)
        revised = dict(plan)
        revised["palette_colors"] = [dict(entry) for entry in plan["palette_colors"]]
        revised["layers"] = [dict(layer) for layer in plan["layers"]]
        entry = revised["palette_colors"][color_index]
        entry.update(name=touch["name"], rgb=rgb,
                     hex=f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
                     matched_hex=touch["matched_hex"], delta_e=touch["delta_e"],
                     palette_page=touch.get("palette_page", 1),
                     is_long_press=touch.get("is_long_press", False))
        if plan.get("mode") in ("line_art", "stencil", "flat_illustration"):
            for layer in revised["layers"]:
                if layer.get("palette_index") == color_index and not layer.get("region_override"):
                    self._set_layer_color(layer, rgb, touch)
        else:
            for layer in revised["layers"]:
                if layer.get("palette_index") == color_index and not layer.get("region_override"):
                    self._set_layer_color(layer, rgb, touch)
            region_map = plan.get("region_map")
            if region_map is not None:
                ids = [r["id"] for r in plan.get("editable_regions", [])
                       if r.get("palette_index") == color_index and "override_rgb" not in r]
                if ids:
                    target = plan["target_executable"].copy()
                    target[np.isin(region_map, ids)] = tuple(reversed(touch["matched_rgb"]))
                    revised["target_executable"] = target
        return self._refresh_edited_plan(revised, previous=plan)

    def recolor_region(self, plan: Dict[str, Any], region_id: int, new_rgb: Tuple[int, int, int],
                       preferred_lut_id: Any = None) -> Dict[str, Any]:
        """Compatibilidade com a edição de uma única área."""
        return self.recolor_regions(plan, [region_id], new_rgb, preferred_lut_id)

    def recolor_regions(self, plan: Dict[str, Any], region_ids, new_rgb: Tuple[int, int, int],
                        preferred_lut_id: Any = None) -> Dict[str, Any]:
        """Recolore uma seleção inteira em uma revisão e uma etapa de desfazer."""
        selected_ids = {int(region_id) for region_id in region_ids}
        if not selected_ids:
            raise ValueError("Selecione pelo menos uma área para alterar a cor")
        regions = plan.get("editable_regions", [])
        known_ids = {int(region["id"]) for region in regions}
        if not selected_ids.issubset(known_ids):
            raise ValueError("A seleção contém uma área que não existe no plano")
        linked_ids = {int(region_id) for layer in plan["layers"]
                      for region_id in layer.get("stroke_region_ids", [])}
        if not selected_ids.issubset(linked_ids):
            raise ValueError("A seleção contém uma área sem traços próprios para recolorir")
        rgb = tuple(int(np.clip(value, 0, 255)) for value in new_rgb)
        touch = self.color_engine.match_color(rgb, preferred_lut_id=preferred_lut_id)
        revised = dict(plan)
        revised["editable_regions"] = [dict(r) for r in regions]
        for region in revised["editable_regions"]:
            if region["id"] in selected_ids:
                region["override_rgb"] = rgb
                region["physical_rgb"] = tuple(touch["matched_rgb"])
        split_layers = []
        for layer in plan["layers"]:
            strokes = layer.get("strokes", [])
            ids = layer.get("stroke_region_ids", [])
            if len(ids) != len(strokes) or not selected_ids.intersection(ids):
                split_layers.append(dict(layer))
                continue
            start = 0
            while start < len(strokes):
                selected = ids[start] in selected_ids
                end = start + 1
                while end < len(strokes) and (ids[end] in selected_ids) == selected:
                    end += 1
                part = dict(layer)
                part["strokes"] = strokes[start:end]
                part["stroke_region_ids"] = ids[start:end]
                part["stroke_count"] = end - start
                part["point_count"] = sum(len(s) for s in part["strokes"])
                if selected:
                    self._set_layer_color(part, rgb, touch)
                    part["region_override"] = True
                    part.pop("region_id", None)
                split_layers.append(part)
                start = end
        revised["layers"] = split_layers
        target = plan["target_executable"].copy()
        target[np.isin(plan["region_map"], list(selected_ids))] = tuple(reversed(touch["matched_rgb"]))
        revised["target_executable"] = target
        return self._refresh_edited_plan(revised, previous=plan)
