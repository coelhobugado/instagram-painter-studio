"""
Função Objetivo Global e Otimizador de Ganho por Segundo.
Implementa a Fase 9 (9.1 Ganho por Segundo e Pesos Variáveis por Tipo de Imagem) do Roadmap.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class LossWeights:
    """Pesos da função objetivo global por modalidade visual."""
    w_structure: float = 0.25
    w_color: float = 0.35
    w_edge: float = 0.20
    w_feature: float = 0.10
    w_time: float = 0.05
    w_risk: float = 0.05
    w_negative_space: float = 0.0
    w_overlap: float = 0.0


class GlobalObjectiveLoss:
    """Calculador de Loss Global e Ganho por Segundo."""

    @staticmethod
    def get_weights_for_class(image_class: str) -> LossWeights:
        """
        Retorna ponderação adaptativa por classe de imagem (Fase 9 e Seção 5 do Plano):
        - Retrato: features faciais têm prioridade altíssima
        - Cartoon / Anime: silhueta, cor e divisão de planos dominam
        - Logo / Lettering: geometria precisa, legibilidade e bordas
        - Flor / Botânica: separação de pétalas e caules (bordas e estrutura)
        - Line-Art: topologia e espessura de borda dominam
        - Geral: equilíbrio estético
        """
        cls_lower = (image_class or "").lower()
        if "portrait" in cls_lower or "face" in cls_lower:
            return LossWeights(
                w_structure=0.20,
                w_color=0.20,
                w_edge=0.15,
                w_feature=0.35,  # Olhos e boca dominam
                w_time=0.05,
                w_risk=0.05,
                w_negative_space=0.15,
            )
        elif "cartoon" in cls_lower or "anime" in cls_lower:
            return LossWeights(
                w_structure=0.35,  # Silhueta e divisão de planos
                w_color=0.35,      # Relação de cores puras
                w_edge=0.15,
                w_feature=0.10,
                w_time=0.03,
                w_risk=0.02,
                w_negative_space=0.10,
            )
        elif "flower" in cls_lower or "botanical" in cls_lower or "nature" in cls_lower:
            return LossWeights(
                w_structure=0.30,
                w_color=0.25,
                w_edge=0.30,  # Separação de pétalas e caules
                w_feature=0.10,
                w_time=0.03,
                w_risk=0.02,
                w_negative_space=0.10,
            )
        elif "logo" in cls_lower or "icon" in cls_lower or "lettering" in cls_lower:
            return LossWeights(
                w_structure=0.35,
                w_color=0.35,
                w_edge=0.20,
                w_feature=0.04,
                w_time=0.03,
                w_risk=0.03,
                w_negative_space=0.15,
            )
        elif "line" in cls_lower or "sketch" in cls_lower:
            return LossWeights(
                w_structure=0.30,
                w_color=0.05,
                w_edge=0.50,  # Bordas e topologia pura
                w_feature=0.05,
                w_time=0.05,
                w_risk=0.05,
                w_negative_space=0.20,
                w_overlap=0.15,
            )
        return LossWeights()

    @staticmethod
    def compute_loss(
        e_structure: float,
        e_color: float,
        e_edge: float,
        e_feature: float,
        time_cost_s: float,
        risk: float,
        weights: LossWeights,
        e_negative_space: float = 0.0,
        e_overlap: float = 0.0,
    ) -> float:
        """Calcula o valor escalar da perda global combinada J (Seção 5 do Plano)."""
        return (
            weights.w_structure * e_structure
            + weights.w_color * e_color
            + weights.w_edge * e_edge
            + weights.w_feature * e_feature
            + weights.w_time * (time_cost_s / 60.0)
            + weights.w_risk * risk
            + weights.w_negative_space * e_negative_space
            + weights.w_overlap * e_overlap
        )

    @staticmethod
    def compute_execution_cost(
        stroke_duration_s: float,
        is_color_switch: bool = False,
        is_tool_switch: bool = False,
        risk_factor: float = 0.0,
    ) -> float:
        """
        Custo total de execução física:
        Tempo de traço + penalidade de troca de cor (UI long-press) + penalidade de troca de ferramenta + risco.
        """
        cost = stroke_duration_s
        if is_color_switch:
            cost += 1.8  # Custo aproximado do toque e arraste do color picker
        if is_tool_switch:
            cost += 0.8  # Custo do toque na toolbar superior
        cost += risk_factor * 2.0
        return max(0.05, cost)

    @staticmethod
    def compute_gain_per_second(predicted_loss_reduction: float, execution_cost_s: float) -> float:
        """
        Calcula o ganho por segundo de um traço candidato (Fase 9.1):
        score = predicted_loss_reduction / execution_cost
        """
        return float(predicted_loss_reduction / max(0.05, execution_cost_s))

    @staticmethod
    def explain_stroke_choice(
        selected_stroke: Dict[str, Any], candidate_strokes: List[Dict[str, Any]]
    ) -> str:
        """
        Gera explicação auditável do critério de escolha (Critério de Aceite da Fase 9):
        'por que este stroke foi escolhido antes daquele?'
        """
        s_id = selected_stroke.get("id", "stroke_selected")
        s_score = selected_stroke.get("gain_per_second", 0.0)
        s_red = selected_stroke.get("predicted_loss_reduction", 0.0)
        s_cost = selected_stroke.get("execution_cost", 0.0)
        
        comp_text = []
        for other in candidate_strokes[:3]:
            if other.get("id") != s_id:
                o_id = other.get("id")
                o_score = other.get("gain_per_second", 0.0)
                comp_text.append(f"supera {o_id} (score {o_score:.3f})")

        joined = ", ".join(comp_text) if comp_text else "melhor opção única no lote"
        return (
            f"Stroke '{s_id}' selecionado: Ganho/s = {s_score:.3f} "
            f"(Redução de Perda = {s_red:.3f} / Custo = {s_cost:.2f}s). Motivo: {joined}."
        )
