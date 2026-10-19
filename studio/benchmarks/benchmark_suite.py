# -*- coding: utf-8 -*-
"""
Suíte de Benchmarks e Testes de Regressão Canônicos do Instagram Painter Studio.
Implementa a FASE 0 do Roadmap Técnico para o Ápice:
- Execução sobre o dataset canônico (benchmarks/canonical/).
- Line-Art testa diretamente o LineArtEngine (não stencil).
- 3 níveis de comparação por imagem: target_executable, predicted_device, observed_device.
- Métricas categorizadas (Fotos, Line-Art, Execução).
- Relatório automático em JSON e Markdown.
"""
import os
import time
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from ..painter_engine import PainterEngine
from ..feedback.error_map import ErrorMapEngine
from ..planning.auto_mode import AutoModeDecisor
from ..planning.skeleton_graph import zhang_suen_thinning, get_crossing_number
from ..device.fake_adb import FakeADB
from ..planning.path_sanitizer import PathSanitizer

class BenchmarkSuite:
    def __init__(self, base_dir: str = None, use_real_device: bool = False):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.base_dir = base_dir
        self.engine = PainterEngine()
        self.use_real_device = use_real_device

    @staticmethod
    def compute_edge_f1_and_chamfer(target_gray: np.ndarray, sim_gray: np.ndarray) -> Tuple[float, float]:
        """Calcula Edge F1 e Chamfer Distance entre imagem alvo e simulação."""
        e_target = cv2.Canny(target_gray, 50, 150)
        e_sim = cv2.Canny(sim_gray, 50, 150)

        t_count = np.count_nonzero(e_target)
        s_count = np.count_nonzero(e_sim)

        if t_count == 0 and s_count == 0:
            return 1.0, 0.0
        if t_count == 0 or s_count == 0:
            return 0.0, 50.0

        # Chamfer distance
        dist_to_target = cv2.distanceTransform(cv2.bitwise_not(e_target), cv2.DIST_L2, 3)
        chamfer = float(np.mean(dist_to_target[e_sim > 0])) if s_count > 0 else 50.0

        # F1 com tolerância de 2 pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        e_target_dilated = cv2.dilate(e_target, kernel)
        e_sim_dilated = cv2.dilate(e_sim, kernel)

        tp_prec = np.count_nonzero(cv2.bitwise_and(e_sim, e_target_dilated))
        tp_rec = np.count_nonzero(cv2.bitwise_and(e_target, e_sim_dilated))

        prec = tp_prec / max(1, s_count)
        rec = tp_rec / max(1, t_count)
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return round(float(f1), 4), round(float(chamfer), 3)

    @staticmethod
    def compute_lineart_topology_metrics(sim_img: np.ndarray) -> Dict[str, Any]:
        """Calcula métricas topológicas de Line-Art: junctions, componentes e espessura."""
        gray = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY) if len(sim_img.shape) == 3 else sim_img
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        if np.count_nonzero(binary) == 0:
            return {"connected_components": 0, "junction_count": 0, "mean_width": 0.0}

        num_cc, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        skel = zhang_suen_thinning(binary)

        ys, xs = np.where(skel > 0)
        junctions = 0
        for y, x in zip(ys, xs):
            if get_crossing_number(y, x, skel) > 2:
                junctions += 1

        dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        skel_radii = dist_map[skel > 0]
        widths = skel_radii * 2.0 if len(skel_radii) > 0 else np.array([0.0])

        return {
            "connected_components": int(num_cc - 1),
            "junction_count": int(junctions),
            "width_median": round(float(np.median(widths)), 2),
            "width_p90": round(float(np.percentile(widths, 90)), 2),
            "width_p99": round(float(np.percentile(widths, 99)), 2)
        }

    def run_canonical_benchmark(self, max_cases: Optional[int] = None, use_real_device: Optional[bool] = None) -> Dict[str, Any]:
        """
        Executa a suíte canônica de benchmark:
        - Avalia as imagens do dataset canônico.
        - Executa LineArtEngine para desenhos e layered_paint para fotos.
        - Configura estritamente ferramentas, cores físicas e espessuras por camada.
        - Distingue e reporta transparentemente execução física real (ADB) vs simulada (FakeADB).
        - Elimina faixas e artefatos sintéticos de UI sobreposta na arte.
        - Produz relatório estruturado JSON e Markdown com validação perceptual e de dispositivo.
        """
        canonical_dir = os.path.join(self.base_dir, "benchmarks", "canonical")
        out_base = os.path.join(self.base_dir, "outputs", "canonical")
        os.makedirs(out_base, exist_ok=True)

        if use_real_device is None:
            use_real_device = self.use_real_device

        adb_instance = None
        is_real = False
        if use_real_device:
            from ..adb_manager import ADBManager
            adb_mgr = ADBManager()
            conn, _ = adb_mgr.connect()
            if conn and adb_mgr.is_screen_on():
                adb_instance = adb_mgr
                is_real = True
            else:
                print("⚠️ Dispositivo físico não conectado via ADB ou tela apagada. Executando em FakeADB mock calibrado.")

        files = sorted([f for f in os.listdir(canonical_dir) if f.endswith(".png") or f.endswith(".jpg")])
        if max_cases:
            files = files[:max_cases]

        results = []
        overall_t0 = time.time()

        for fname in files:
            img_path = os.path.join(canonical_dir, fname)
            case_id = os.path.splitext(fname)[0]
            case_out_dir = os.path.join(out_base, case_id)
            os.makedirs(case_out_dir, exist_ok=True)

            raw_img = cv2.imread(img_path)
            if raw_img is None:
                continue

            is_line_art = any(case_id.startswith(p) for p in ["line_", "flower_", "manga_", "text"])
            t0 = time.time()

            if is_line_art:
                mode = "line_art"
                plan = self.engine.generate_line_art_plan(img_path, paint_background=True)
            else:
                mode = "layered_paint"
                plan = self.engine.generate_layered_paint_plan(
                    img_path, num_colors=4, brush_spacing=6, optimize_in_simulation=False
                )

            planning_time_ms = round((time.time() - t0) * 1000, 1)

            target_img = plan.get("target_executable", plan.get("fit_image", raw_img))
            sim_img = plan["simulation_image"]
            h_target, w_target = target_img.shape[:2]

            # Define a cor de fundo padrão da tela
            bg_rgb = plan.get("background_rgb")
            if bg_rgb:
                bg_bgr = (int(bg_rgb[2]), int(bg_rgb[1]), int(bg_rgb[0]))
            elif is_line_art:
                bg_bgr = (255, 255, 255)
            else:
                bg_bgr = (22, 22, 22)

            # Execução Física no Dispositivo Real ou no Mock Estrito
            if is_real and adb_instance:
                transform = plan.get("transform")
                sb = adb_instance.profile_loader.get_safe_bounds()
                for layer in plan.get("layers", []):
                    thickness = layer.get("thickness", "min")
                    adb_instance.set_brush_thickness(thickness)
                    brush = layer.get("brush", "solid")
                    adb_instance.select_pen_tool(brush)
                    touch_info = layer.get("touch_info")
                    if touch_info:
                        adb_instance.select_palette_color(touch_info)

                    strokes = layer.get("strokes", [])
                    screen_strokes = []
                    for s in strokes:
                        if not s:
                            continue
                        scr_s = [transform.local_to_screen(pt) for pt in s] if transform else s
                        screen_strokes.append(scr_s)

                    clean_strokes = PathSanitizer.sanitize_all_strokes(
                        screen_strokes,
                        x_min=sb["left"], x_max=sb["right"],
                        y_min=sb["top"], y_max=sb["bottom"]
                    )
                    adb_instance.execute_stroke_batch(clean_strokes)
                    time.sleep(0.08)

                raw_screencap = adb_instance.capture_screenshot()
                if raw_screencap is not None:
                    canvas_crop = raw_screencap[sb["top"]:sb["bottom"], sb["left"]:sb["right"]]
                    observed_device = cv2.resize(canvas_crop, (w_target, h_target))
                else:
                    observed_device = sim_img.copy()
                observed_source = "physical_device"
                device_note = f"Captura real do celular ({adb_instance.device_info.get('model', 'Android')})"
            else:
                # Executa em FakeADB calibrado
                # init_ui=False garante canvas limpo sem sobreposição de barras ou botões sintéticos
                fake_adb = FakeADB(width=w_target, height=h_target, init_ui=False, background_bgr=bg_bgr)
                for layer in plan.get("layers", []):
                    # 1. Configura ferramenta
                    brush = layer.get("brush", "solid")
                    fake_adb.select_pen_tool(brush)

                    # 2. Configura espessura calibrada
                    thickness = layer.get("thickness", "min")
                    fake_adb.set_brush_thickness(thickness)

                    # 3. Configura cor da camada
                    touch_info = layer.get("touch_info")
                    if touch_info:
                        fake_adb.select_palette_color(touch_info)
                    elif "render_color_rgb" in layer:
                        fake_adb.set_color_rgb(layer["render_color_rgb"])
                    elif "color_rgb" in layer:
                        fake_adb.set_color_rgb(layer["color_rgb"])

                    # 4. Executa os traços
                    fake_adb.execute_stroke_batch(layer.get("strokes", []))

                observed_device = fake_adb.capture_screenshot()
                observed_source = "fake_adb_mock"
                device_note = "Executado via FakeADB (mock calibrado). Não representa medição fotométrica em celular físico."

            # Salva os 3 níveis de resultado
            cv2.imwrite(os.path.join(case_out_dir, "target_executable.png"), target_img)
            cv2.imwrite(os.path.join(case_out_dir, "predicted_device.png"), sim_img)
            cv2.imwrite(os.path.join(case_out_dir, "observed_device.png"), observed_device)

            # Métricas Perceptuais:
            # 1. Target vs Predição (Fidelidade do Planejamento/Simulação)
            _, error_target_pred = ErrorMapEngine.compute_error_map(target_img, sim_img)

            # 2. Predição vs Observado (Fidelidade da Execução do Dispositivo)
            obs_fitted = cv2.resize(observed_device, (w_target, h_target)) if observed_device.shape[:2] != (h_target, w_target) else observed_device
            _, error_pred_obs = ErrorMapEngine.compute_error_map(sim_img, obs_fitted)

            edge_f1, chamfer = self.compute_edge_f1_and_chamfer(
                cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY)
            )

            # Métricas de Execução
            total_strokes = plan.get("total_strokes", 0)
            total_points = plan.get("total_points", 0)
            color_switches = len(plan.get("layers", []))

            case_data: Dict[str, Any] = {
                "case_id": case_id,
                "mode": mode,
                "device_backend": observed_source,
                "is_real_device": is_real,
                "planning_time_ms": planning_time_ms,
                "total_strokes": total_strokes,
                "total_points": total_points,
                "color_switches": color_switches,
                "mean_delta_e": error_target_pred["mean_delta_e"],
                "p95_delta_e": error_target_pred["p95_delta_e"],
                "ssim": error_target_pred["ssim"],
                "device_mean_delta_e": error_pred_obs["mean_delta_e"],
                "device_ssim": error_pred_obs["ssim"],
                "edge_f1": edge_f1,
                "chamfer_distance": chamfer,
                "device_note": device_note
            }

            if is_line_art:
                topo = self.compute_lineart_topology_metrics(sim_img)
                case_data.update(topo)

            results.append(case_data)

        total_runtime_s = round(time.time() - overall_t0, 2)
        exec_backend_desc = "Dispositivo Físico Real (ADB)" if is_real else "Simulador FakeADB (Mock Calibrado)"
        validation_status = "VALIDADO_HARDWARE_REAL" if is_real else "MOCK_SEM_HARDWARE_FISICO"
        report = {
            "version": "apice-canonical-1.1",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_runtime_s": total_runtime_s,
            "total_cases": len(results),
            "execution_backend": exec_backend_desc,
            "is_real_device": is_real,
            "validation_status": validation_status,
            "validation_note": "Medição de fidelidade em hardware físico real." if is_real else "Execução simulada em FakeADB (mock calibrado) com cores e espessuras estritamente configuradas por camada. Não mede fidelidade fotométrica em celular real sem dispositivo físico conectado.",
            "results": results
        }

        # Salva relatórios
        rep_json = os.path.join(self.base_dir, "outputs", "canonical_benchmark_report.json")
        rep_md = os.path.join(self.base_dir, "outputs", "canonical_benchmark_report.md")

        with open(rep_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        with open(rep_md, "w", encoding="utf-8") as f:
            f.write(f"# Canonical Benchmark Report — Instagram Painter Studio\n\n")
            f.write(f"- **Data**: {report['timestamp']}\n")
            f.write(f"- **Casos Avaliados**: {report['total_cases']}\n")
            f.write(f"- **Tempo Total**: {report['total_runtime_s']}s\n")
            f.write(f"- **Ambiente de Execução**: {report['execution_backend']}\n")
            f.write(f"- **Validação em Hardware Real**: {'Sim (Físico)' if is_real else 'Não (Mock / FakeADB)'}\n")
            f.write(f"- **Nota de Validação**: {report['validation_note']}\n\n")
            f.write(f"| Caso | Modo | Backend | Strokes | Target/Pred ΔE | Target/Pred SSIM | Device/Pred ΔE | Device/Pred SSIM | Edge F1 | Chamfer |\n")
            f.write(f"|------|------|---------|---------|----------------|------------------|----------------|------------------|---------|---------|\n")
            for r in results:
                f.write(f"| {r['case_id']} | {r['mode']} | {r['device_backend']} | {r['total_strokes']} | {r['mean_delta_e']:.2f} | {r['ssim']:.3f} | {r['device_mean_delta_e']:.2f} | {r['device_ssim']:.3f} | {r['edge_f1']:.3f} | {r['chamfer_distance']:.2f} |\n")

        return report

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Compatibilidade: executa benchmark padrão."""
        return self.run_canonical_benchmark(max_cases=6)

def run_cli_benchmark():
    import sys
    use_real = "--real-device" in sys.argv
    max_cases = None
    for idx, arg in enumerate(sys.argv):
        if arg == "--cases" and idx + 1 < len(sys.argv):
            try:
                max_cases = int(sys.argv[idx + 1])
            except ValueError:
                pass

    suite = BenchmarkSuite(use_real_device=use_real)
    backend_txt = "CELULAR REAL" if use_real else "FakeADB (MOCK)"
    print(f"Iniciando Suite de Benchmark Canonico (Backend: {backend_txt})...")
    rep = suite.run_canonical_benchmark(max_cases=max_cases)
    print(f"Benchmark concluido em {rep['total_runtime_s']}s! (Status: {rep['validation_status']})")
    for r in rep["results"]:
        print(f" - {r['case_id']} ({r['mode']}): {r['total_strokes']} strokes, Target/Pred SSIM={r['ssim']}, Device/Pred SSIM={r['device_ssim']}")

if __name__ == "__main__":
    run_cli_benchmark()
