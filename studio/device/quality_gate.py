"""
Gate Final de Qualidade e Trava de Segurança Fail-Closed Pré-Execução.
Implementa a Fase 20 (Gate Obrigatório antes de qualquer Injeção Física de Toques) do Roadmap.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import math
import numpy as np


@dataclass
class QualityGateResult:
    """Resultado auditável da verificação de integridade pré-execução física."""
    passed: bool
    summary: str
    failed_checks: List[str] = field(default_factory=list)
    check_details: Dict[str, bool] = field(default_factory=dict)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)


class QualityGate:
    """
    Gate Final de Qualidade (Fase 20).
    Antes de iniciar qualquer execução no aparelho real, verifica:
    - geometry_valid
    - motion_valid
    - preview_quality_pass
    - safe_ui_valid
    - device_profile_valid
    - time_budget_valid

    Princípio Fail-Closed: Se qualquer item falhar, bloqueia estritamente a execução.
    """

    @classmethod
    def evaluate(
        cls,
        geometric_layers: List[Dict[str, Any]],
        motion_plan: Optional[Any],
        ui_validation_res: Optional[Any],
        device_profile: Optional[Dict[str, Any]],
        time_budget_s: float,
        preview_delta_e: float = float("inf"),
        preview_edge_f1: float = 0.0,
        preview_ssim: Optional[float] = None,
        require_live_ui: bool = True,
    ) -> QualityGateResult:
        check_details = {}
        failed_checks = []

        # 1. Validação Geométrica Estrita (geometry_valid)
        geometry_ok = isinstance(geometric_layers, list) and bool(geometric_layers)
        total_strokes = 0
        for l in geometric_layers or []:
            if not isinstance(l, dict):
                geometry_ok = False
                break
            strokes = l.get("strokes", [])
            if not isinstance(strokes, list) or not strokes:
                geometry_ok = False
                break
            total_strokes += len(strokes)
            for s in strokes:
                if not isinstance(s, (list, tuple)) or len(s) < 2:
                    geometry_ok = False
                    break
                for pt in s:
                    if (not isinstance(pt, (list, tuple, np.ndarray)) or len(pt) < 2 or
                            not all(isinstance(value, (int, float, np.number)) and math.isfinite(value)
                                    for value in pt[:2])):
                        geometry_ok = False
                        break
                if geometry_ok:
                    length = sum(
                        math.hypot(b[0] - a[0], b[1] - a[1])
                        for a, b in zip(s, s[1:])
                    )
                    if length < 2.5:
                        geometry_ok = False
                if not geometry_ok:
                    break
            if not geometry_ok:
                break
        geometry_ok = geometry_ok and (total_strokes > 0)
        check_details["geometry_valid"] = geometry_ok
        if not geometry_ok:
            failed_checks.append("geometry_valid: Camadas sem traços, com traços curtos demais ou coordenadas não finitas/NaN")

        # 2. Validação do Motion Plan (motion_valid)
        motion_ok = False
        if motion_plan is not None:
            # Checa se possui traços compilados e hash íntegro
            st_count = getattr(motion_plan, "total_strokes", 0)
            p_hash = getattr(motion_plan, "plan_hash", "")
            motion_ok = (st_count == total_strokes) and bool(p_hash)
        check_details["motion_valid"] = motion_ok
        if not motion_ok:
            failed_checks.append("motion_valid: Motion Plan ausente ou inválido")

        # 3. Validação de Qualidade de Prévia (preview_quality_pass)
        quality_ok = (
            math.isfinite(preview_delta_e) and 
            (preview_delta_e <= 35.0) and 
            math.isfinite(preview_edge_f1) and 
            (preview_edge_f1 >= 0.20 or (preview_ssim is not None and math.isfinite(preview_ssim) and preview_ssim >= 0.50))
        )
        if preview_ssim is not None:
            quality_ok = quality_ok and math.isfinite(preview_ssim) and (preview_ssim >= 0.50)
        check_details["preview_quality_pass"] = quality_ok
        if not quality_ok:
            failed_checks.append(f"preview_quality_pass: Desvio perceptual elevado ou métricas indeterminadas (ΔE={preview_delta_e:.1f}, Edge F1={preview_edge_f1:.2f}, SSIM={preview_ssim})")

        # 4. Validação de Segurança da UI (safe_ui_valid)
        ui_ok = False
        if not require_live_ui:
            ui_ok = True if ui_validation_res is None else getattr(ui_validation_res, "is_valid", False)
        elif ui_validation_res is not None:
            ui_ok = getattr(ui_validation_res, "is_valid", False)
        check_details["safe_ui_valid"] = ui_ok
        if not ui_ok:
            failed_checks.append("safe_ui_valid: Interface física do Instagram divergente ou não validada")

        # 5. Validação do Perfil do Dispositivo (device_profile_valid)
        profile_ok = False
        if isinstance(device_profile, dict):
            safe_bounds = device_profile.get("safe_bounds", {})
            if isinstance(safe_bounds, dict) and all(k in safe_bounds for k in ("left", "top", "right", "bottom")):
                l_b = safe_bounds["left"]
                t_b = safe_bounds["top"]
                r_b = safe_bounds["right"]
                b_b = safe_bounds["bottom"]
                if (isinstance(l_b, (int, float)) and isinstance(t_b, (int, float)) and
                    isinstance(r_b, (int, float)) and isinstance(b_b, (int, float)) and
                    math.isfinite(l_b) and math.isfinite(t_b) and 
                    math.isfinite(r_b) and math.isfinite(b_b) and 
                    l_b < r_b and t_b < b_b):
                    profile_ok = True
        check_details["device_profile_valid"] = profile_ok
        if not profile_ok:
            failed_checks.append("device_profile_valid: Perfil de geometria física ausente, incompleto ou limites invertidos")

        # 6. Validação Realista do Orçamento de Tempo Completo de Execução ADB (P1.6)
        pure_motion_time_s = 0.0
        if motion_plan is not None:
            if hasattr(motion_plan, "total_duration_ms"):
                pure_motion_time_s = float(motion_plan.total_duration_ms) / 1000.0
            elif hasattr(motion_plan, "estimated_duration_s"):
                pure_motion_time_s = float(motion_plan.estimated_duration_s)
        
        num_layers = len(geometric_layers) if geometric_layers else 1
        # Custos físicos calibrados de hardware (P1.6):
        layer_prep_overhead_s = num_layers * 0.60      # Troca de ferramenta, espessura e seleção de cor
        screencap_overhead_s = num_layers * 0.40       # Captura de tela ao final de cada camada e settle time
        adb_stroke_overhead_s = total_strokes * 0.045   # Overhead de comunicação e confirmação por traço

        # Tempo total previsto para o ciclo de execução real via ADB
        est_time_s = pure_motion_time_s + layer_prep_overhead_s + screencap_overhead_s + adb_stroke_overhead_s
        if est_time_s == 0.0 and geometric_layers:
            est_time_s = total_strokes * 0.20 + num_layers * 1.0

        # Margem de tolerância de 15% sobre o budget configurado
        time_budget_ok = (
            math.isfinite(time_budget_s) and time_budget_s > 0
            and math.isfinite(est_time_s) and est_time_s > 0
            and est_time_s <= (time_budget_s * 1.15)
        )
        check_details["time_budget_valid"] = time_budget_ok
        check_details["estimated_total_adb_time_s"] = round(est_time_s, 2)
        check_details["pure_motion_time_s"] = round(pure_motion_time_s, 2)
        if not time_budget_ok:
            failed_checks.append(f"time_budget_valid: Tempo estimado realista ({est_time_s:.1f}s) excede o orçamento ({time_budget_s:.1f}s)")

        # Decisão Final Fail-Closed: Todos os 6 critérios devem passar
        all_passed = len(failed_checks) == 0
        summary = "Quality Gate Aprovado: O plano cumpre todos os requisitos físicos e pode ser executado." if all_passed else \
            f"Quality Gate REPROVADO: Execução bloqueada ({len(failed_checks)} falhas de integridade detectadas)."

        return QualityGateResult(
            passed=all_passed,
            summary=summary,
            failed_checks=failed_checks,
            check_details=check_details,
            metrics_snapshot={
                "total_strokes": total_strokes,
                "preview_delta_e": preview_delta_e,
                "preview_edge_f1": preview_edge_f1,
                "estimated_time_s": est_time_s,
                "time_budget_s": time_budget_s,
            },
        )


def plan_content_matches_hash(plan: Dict[str, Any]) -> bool:
    """Confere que os traços e ajustes ainda correspondem à revisão aprovada."""
    stored_hash = plan.get("plan_hash")
    if not stored_hash:
        return True  # Planos sintéticos/legados ainda dependem do QualityGate.
    try:
        from ..core.models import StrokePlan
        mode = plan.get("mode", "")
        preset = "LAYERED" if mode == "layered_paint" else mode.upper()
        actual_hash = StrokePlan.compute_plan_hash(
            plan.get("seed", 42), preset, plan.get("layers", []),
            image_hash=plan.get("source_image_hash", ""),
            transform=plan.get("transform"),
        )
        return stored_hash == actual_hash
    except (TypeError, ValueError, KeyError, AttributeError):
        return False
