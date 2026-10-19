"""
Telemetria Completa de Sessão de Pintura.
Implementa a Fase 17 (Registro Rigoroso de 17 Métricas por Sessão e Relatórios) do Roadmap.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
import time


@dataclass
class SessionTelemetry:
    """Registro estruturado das 17 métricas auditáveis por sessão de pintura (Fase 17)."""
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    planning_time_s: float = 0.0
    simulation_time_s: float = 0.0
    execution_time_s: float = 0.0
    events_per_s: float = 0.0
    strokes_per_s: float = 0.0
    color_switches: int = 0
    tool_switches: int = 0
    long_presses: int = 0
    undo_count: int = 0
    verification_failures: int = 0
    corrective_strokes: int = 0
    mean_delta_e: float = 0.0
    p95_delta_e: float = 0.0
    edge_f1: float = 0.0
    preview_real_residual: float = 0.0
    ram_peak_mb: float = 0.0
    device_temperature_c: float = 32.0

    def record_switch(self, is_color: bool = False, is_tool: bool = False, is_long_press: bool = False) -> None:
        if is_color:
            self.color_switches += 1
        if is_tool:
            self.tool_switches += 1
        if is_long_press:
            self.long_presses += 1

    def record_verification(self, passed: bool, corrective_count: int = 0) -> None:
        if not passed:
            self.verification_failures += 1
        self.corrective_strokes += corrective_count

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for k, v in data.items():
            if isinstance(v, float):
                data[k] = round(v, 3)
        return data

    def export_json(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_summary_markdown(self) -> str:
        d = self.to_dict()
        return (
            f"### Relatório de Telemetria de Sessão ({d['session_id']})\n"
            f"- **Tempo de Planejamento**: {d['planning_time_s']}s | **Simulação**: {d['simulation_time_s']}s | **Execução**: {d['execution_time_s']}s\n"
            f"- **Throughput**: {d['events_per_s']} eventos/s | {d['strokes_per_s']} traços/s\n"
            f"- **Operações de UI**: {d['color_switches']} trocas de cor | {d['tool_switches']} trocas de ferramenta | {d['long_presses']} long presses\n"
            f"- **Closed-Loop**: {d['verification_failures']} falhas de verificação | {d['undo_count']} undos | {d['corrective_strokes']} traços corretivos\n"
            f"- **Qualidade Óptica**: Média ΔE {d['mean_delta_e']} | P95 ΔE {d['p95_delta_e']} | Edge F1 {d['edge_f1']:.2f}\n"
            f"- **Sim-to-Real Residual**: {d['preview_real_residual']} px\n"
            f"- **Recursos**: RAM Pico {d['ram_peak_mb']} MB | Temp. Dispositivo {d['device_temperature_c']}°C\n"
        )
