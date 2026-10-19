"""
Aprendizado Interpretável Sim-to-Real e Predição de Risco Físico.
Implementa a Fase 18 (Residual Model, Planner Ranking e Risk Prediction sem Black-Box) do Roadmap.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class ResidualCalibration:
    """Parâmetros interpretáveis de compensação de resíduo físico."""
    dx_bias: float = 0.0
    dy_bias: float = 0.0
    width_scale: float = 1.0
    sample_count: int = 0


class ResidualCorrectionModel:
    """
    Modelo de Compensação Residual Interpretável (Fase 18.1).
    Ajusta desvios sistemáticos de toque e espalhamento de tinta observados no device real.
    """

    def __init__(self):
        self.calib = ResidualCalibration()

    def update_from_observation(
        self,
        predicted_points: List[Tuple[int, int]],
        actual_observed_points: List[Tuple[int, int]],
        predicted_width: float,
        actual_observed_width: float,
    ) -> None:
        """Atualiza a calibração incremental a partir de pontos registrados."""
        if not predicted_points or not actual_observed_points:
            return

        min_len = min(len(predicted_points), len(actual_observed_points))
        p_arr = np.array(predicted_points[:min_len], dtype=np.float32)
        a_arr = np.array(actual_observed_points[:min_len], dtype=np.float32)

        diffs = a_arr - p_arr
        mean_dx = float(np.mean(diffs[:, 0]))
        mean_dy = float(np.mean(diffs[:, 1]))

        # Atualização por média móvel exponencial
        alpha = 0.2
        self.calib.dx_bias = (1.0 - alpha) * self.calib.dx_bias + alpha * mean_dx
        self.calib.dy_bias = (1.0 - alpha) * self.calib.dy_bias + alpha * mean_dy

        if predicted_width > 0:
            scale_obs = actual_observed_width / predicted_width
            self.calib.width_scale = (1.0 - alpha) * self.calib.width_scale + alpha * scale_obs

        self.calib.sample_count += 1

    def apply_compensation(
        self, stroke_points: List[Tuple[int, int]], desired_width: float
    ) -> Tuple[List[Tuple[int, int]], float]:
        """Aplica a correção sistemática aos pontos antes do envio ao dispositivo."""
        comp_pts = [
            (int(round(x - self.calib.dx_bias)), int(round(y - self.calib.dy_bias)))
            for x, y in stroke_points
        ]
        comp_width = desired_width / max(0.5, self.calib.width_scale)
        return comp_pts, comp_width


class RiskPredictor:
    """
    Preditor de Risco Físico de Traço (Fase 18.3).
    Estima a probabilidade de falha (evento perdido pelo touchscreen, perda de traço ou overshoot).
    """

    @classmethod
    def evaluate_stroke_risk(
        cls,
        points: List[Tuple[int, int]],
        velocity_px_s: float,
        canvas_bounds: Tuple[int, int, int, int],  # (x_min, y_min, x_max, y_max)
    ) -> Tuple[float, List[str]]:
        """
        Retorna score de risco no intervalo [0.0, 1.0] e fatores causais auditáveis.
        """
        if len(points) < 2:
            return 0.1, ["Traço pontual de baixo risco"]

        risk_score = 0.0
        factors = []
        x_min, y_min, x_max, y_max = canvas_bounds

        # 1. Risco de Velocidade Excessiva (Android input pode perder eventos acima de 1200 px/s)
        if velocity_px_s > 1200.0:
            risk_score += 0.45
            factors.append(f"Velocidade excessiva ({velocity_px_s:.0f} px/s > 1200 px/s)")
        elif velocity_px_s > 800.0:
            risk_score += 0.20
            factors.append(f"Velocidade alta ({velocity_px_s:.0f} px/s)")

        # 2. Proximidade de Borda da Tela (risco de interceptação por gesto de navegação)
        border_margin = 25
        near_border = any(
            x <= x_min + border_margin or x >= x_max - border_margin
            or y <= y_min + border_margin or y >= y_max - border_margin
            for x, y in points
        )
        if near_border:
            risk_score += 0.30
            factors.append("Proximidade crítica com limites do canvas/gestos do sistema")

        # 3. Curvatura Fechada / Ângulos Agudos
        pts = np.array(points, dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        lengths = np.linalg.norm(diffs, axis=1) + 1e-6
        tangents = diffs / lengths[:, np.newaxis]
        
        sharp_turns = 0
        for i in range(len(tangents) - 1):
            cos_angle = np.dot(tangents[i], tangents[i + 1])
            if cos_angle < -0.2:  # Ângulo agudo superior a ~100 graus
                sharp_turns += 1

        if sharp_turns > 0:
            risk_score += min(0.35, sharp_turns * 0.12)
            factors.append(f"Presença de {sharp_turns} curvas muito fechadas")

        total_risk = float(np.clip(risk_score, 0.0, 1.0))
        if not factors:
            factors.append("Traço estável dentro dos limites seguros")

        return total_risk, factors


class PlanRanker:
    """Ranking e Seleção de Planos de Pintura (Fase 18.2)."""

    @staticmethod
    def score_plan(
        predicted_delta_e: float,
        time_seconds: float,
        time_budget_s: float,
        risk_mean: float,
    ) -> float:
        """
        Calcula pontuação de qualidade global do plano.
        Quanto maior o score, melhor o compromisso entre fidelidade, tempo e segurança física.
        """
        # Fidelidade cromática (erro menor -> score maior)
        fidelity_score = max(0.0, 100.0 - predicted_delta_e * 2.5)

        # Penalidade por ultrapassar o budget
        budget_penalty = max(0.0, (time_seconds - time_budget_s) * 2.0) if time_seconds > time_budget_s else 0.0

        # Penalidade de risco
        risk_penalty = risk_mean * 40.0

        return float(fidelity_score - budget_penalty - risk_penalty)
