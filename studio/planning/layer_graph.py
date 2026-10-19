"""
Grafo de Camadas e Ordenação Semântica do Algoritmo do Pintor (Painter's Algorithm).
Implementa a Seção 16 e Seção 17 do Plano de Evolução Completo (Layer Graph & Multi-resolution).
"""
from typing import List, Dict, Any
import numpy as np
from ..core.models import StrokePurpose

class LayerGraphEngine:
    @staticmethod
    def build_layer_order(regions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organiza as regiões em camadas estritas com base em semântica, luminância e saliência,
        impedindo que preenchimentos posteriores cubram detalhes finos ou destaques especulares.
        """
        # Categoriza cada grupo de cor por função pictórica (Seções 14 e 15 do Plano de Fidelidade Máxima)
        background_layers = []
        dark_base_layers = []
        base_color_layers = []
        midtone_layers = []
        shadow_layers = []
        highlight_layers = []

        total_all_area = sum(rg.get("total_area", 0) for rg in regions_data) or 1

        for reg_group in regions_data:
            raw_L = float(reg_group.get("mean_L", 50.0))
            # Normalização de escala para L* em [0, 100] (Seção 13 da Auditoria Real)
            L_star = raw_L * (100.0 / 255.0) if raw_L > 100.0 else raw_L
            area = reg_group.get("total_area", 100)
            saliency = reg_group.get("saliency", 0.5)

            # Classificação pictórica calibrada (Seções 14 e 15 do Plano):
            # Apenas regiões verdadeiramente dominantes de baixa saliência são fundo
            if area > (total_all_area * 0.28) and saliency < 0.22:
                reg_group["purpose"] = StrokePurpose.BACKGROUND.value
                background_layers.append(reg_group)
            elif L_star > 85.0 and area < (total_all_area * 0.12):
                # Luzes altas / reflexos especulares
                reg_group["purpose"] = StrokePurpose.HIGHLIGHT.value
                highlight_layers.append(reg_group)
            elif L_star < 28.0:
                # Seção 15 do Plano: Grandes massas escuras (cabelo preto, terno, etc.)
                # são DARK_BASE (cobertura base com brush largo), não micro-sombras
                if area >= (total_all_area * 0.08) or area > 5000:
                    reg_group["purpose"] = StrokePurpose.DARK_BASE.value
                    dark_base_layers.append(reg_group)
                else:
                    reg_group["purpose"] = StrokePurpose.SHADOW.value
                    shadow_layers.append(reg_group)
            elif 28.0 <= L_star <= 65.0:
                # Meios-tons e modulação
                reg_group["purpose"] = StrokePurpose.MIDTONE.value
                midtone_layers.append(reg_group)
            else:
                # Massas base
                reg_group["purpose"] = StrokePurpose.COLOR_MASS.value
                base_color_layers.append(reg_group)

        # Ordenação interna de cada categoria:
        # Fundo: maiores áreas primeiro
        background_layers.sort(key=lambda l: l["total_area"], reverse=True)
        # Massas escuras base (terno, cabelo preto): maiores áreas primeiro
        dark_base_layers.sort(key=lambda l: l["total_area"], reverse=True)
        # Cores base: maiores áreas primeiro
        base_color_layers.sort(key=lambda l: l["total_area"], reverse=True)
        # Meios-tons: maior área para menor
        midtone_layers.sort(key=lambda l: l["total_area"], reverse=True)
        # Sombras: aplicadas sobre as massas base
        shadow_layers.sort(key=lambda l: l["total_area"], reverse=True)
        # Highlights: sobrepostas por último para garantir brilho especular intacto
        highlight_layers.sort(key=lambda l: l["total_area"], reverse=False)

        # Ordem topológica imutável da pintura digital (Seções 14, 15 e 18 do Plano):
        # 1. Fundo -> 2. Massas Base (Dark Base e Color Mass) -> 3. Meios-Tons -> 4. Sombras -> 5. Luzes Altas
        ordered_layers = []
        ordered_layers.extend(background_layers)
        ordered_layers.extend(dark_base_layers)
        ordered_layers.extend(base_color_layers)
        ordered_layers.extend(midtone_layers)
        ordered_layers.extend(shadow_layers)
        ordered_layers.extend(highlight_layers)

        return ordered_layers
