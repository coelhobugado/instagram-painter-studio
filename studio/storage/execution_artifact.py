# -*- coding: utf-8 -*-
"""
Módulo de Armazenamento e Caching de Artefatos Executáveis (ExecutionArtifact).
Permite salvar e carregar planos de pintura completos e pré-compilados em disco,
eliminando a regeneração redundante e garantindo paridade 100% estrita entre Preview e Execução Física.
"""
import os
import sys
import json
import time
import hashlib
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

from ..core.models import CanvasTransform
from ..device.quality_gate import QualityGateResult
from .atomic import atomic_open


class LayerRiskLevel(str, Enum):
    """Nível de criticidade visual de uma camada para controle adaptativo de batch e closed-loop."""
    HIGH = "HIGH"        # Contornos faciais, olhos, mãos, linhas finas críticas
    MEDIUM = "MEDIUM"    # Sombras, transições de gradiente, detalhes secundários
    LOW = "LOW"          # Grandes blocos de preenchimento e massas de cor de fundo


def compute_plan_fingerprint(
    image_path: str,
    config_dict: Dict[str, Any],
    engine_version: str = "3.1",
    device_profile: Optional[Dict[str, Any]] = None,
    lut_data: Optional[Any] = None
) -> str:
    """Calcula o hash criptográfico determinístico SHA-256 da imagem, configurações, perfil e calibração/LUT (P1.2)."""
    hasher = hashlib.sha256()
    hasher.update(engine_version.encode("utf-8"))

    # Hash dos bytes da imagem
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
    else:
        hasher.update(image_path.encode("utf-8"))

    # Hash determinístico da configuração ordenada
    conf_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    hasher.update(conf_str.encode("utf-8"))

    # Inclui perfil de dispositivo físico se fornecido (P1.2)
    dev_prof = device_profile or config_dict.get("device_profile")
    if dev_prof:
        dev_str = json.dumps(dev_prof, sort_keys=True, ensure_ascii=False)
        hasher.update(dev_str.encode("utf-8"))

    # Inclui dados de LUT / calibração se fornecidos (P1.2)
    lut = lut_data or config_dict.get("lut_data")
    if lut:
        lut_str = json.dumps(lut, sort_keys=True, ensure_ascii=False) if isinstance(lut, (dict, list)) else str(lut)
        hasher.update(lut_str.encode("utf-8"))

    return hasher.hexdigest()


class ExecutionArtifact:
    """
    Artefato Executável Completo de Pintura.
    Contém todos os traços brutos, lotes compilados a 120 Hz, configurações de hardware,
    transformações de tela e a assinatura do QualityGate.
    """
    def __init__(
        self,
        plan_id: str,
        fingerprint: str,
        mode: str,
        image_class: str,
        transform: CanvasTransform,
        layers: List[Dict[str, Any]],
        quality_gate: Dict[str, Any],
        profile_metrics: Optional[Dict[str, float]] = None,
        source_image: str = "",
        estimated_time_s: float = 0.0,
        total_strokes: int = 0
    ):
        self.plan_id = plan_id
        self.fingerprint = fingerprint
        self.mode = mode
        self.image_class = image_class
        self.transform = transform
        self.layers = layers
        self.quality_gate = asdict(quality_gate) if isinstance(quality_gate, QualityGateResult) else quality_gate
        self.profile_metrics = profile_metrics or {}
        self.source_image = source_image
        self.estimated_time_s = estimated_time_s
        self.total_strokes = total_strokes or sum(len(l.get("strokes", [])) for l in layers)
        self.created_at = time.time()

    @property
    def is_quality_gate_approved(self) -> bool:
        """Garante validação estritamente fail-closed (default False)."""
        if isinstance(self.quality_gate, QualityGateResult):
            return self.quality_gate.passed is True
        if isinstance(self.quality_gate, dict):
            return self.quality_gate.get("passed") is True
        return False

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Gera o manifesto serializável legível para o usuário e MCP."""
        layers_summary = []
        for l in self.layers:
            layers_summary.append({
                "index": l.get("index", 0),
                "name": l.get("name", "Camada"),
                "color_rgb": list(l.get("color_rgb", [0, 0, 0])),
                "stroke_count": len(l.get("strokes", [])),
                "thickness": l.get("thickness", "min"),
                "brush": l.get("brush", "solid"),
                "purpose": l.get("purpose", "fill"),
                "risk_level": l.get("risk_level", LayerRiskLevel.MEDIUM.value),
                "touch_info": l.get("touch_info")
            })

        qg_passed = self.is_quality_gate_approved
        return {
            "version": "3.1-enterprise",
            "status": "READY_FOR_EXECUTION" if qg_passed else "REJECTED_BY_QUALITY_GATE",
            "ready_to_send_instagram": qg_passed,
            "sent_to_device": False,
            "plan_id": self.plan_id,
            "fingerprint": self.fingerprint,
            "source_image": self.source_image,
            "mode": self.mode,
            "image_class": self.image_class,
            "total_layers": len(self.layers),
            "total_strokes": self.total_strokes,
            "estimated_time_s": round(self.estimated_time_s, 1),
            "quality_gate": self.quality_gate,
            "profile_metrics": {k: round(v, 3) for k, v in self.profile_metrics.items()},
            "transform": {
                "offset_x": self.transform.offset_x,
                "offset_y": self.transform.offset_y,
                "scale": self.transform.scale
            },
            "layers": layers_summary
        }


    def _transform_values(self) -> List[float]:
        values = [self.transform.offset_x, self.transform.offset_y, self.transform.scale]
        # width/height só entram quando conhecidos, mantendo o hash de artefatos antigos válido
        if self.transform.width or self.transform.height:
            values += [int(self.transform.width), int(self.transform.height)]
        return values

    def compute_content_hash(self) -> str:
        """Calcula hash criptográfico determinístico SHA-256 do conteúdo real executável (P1.2)."""
        payload = {
            "plan_id": self.plan_id,
            "fingerprint": self.fingerprint,
            "mode": self.mode,
            "image_class": self.image_class,
            "transform": self._transform_values(),
            "layers": self.layers,
            "quality_gate": self.quality_gate,
            "profile_metrics": self.profile_metrics,
            "source_image": self.source_image,
            "estimated_time_s": self.estimated_time_s,
            "total_strokes": self.total_strokes,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, base_path: str) -> Tuple[str, str]:
        """
        Salva o artefato executável completo no disco:
        1. base_path + '_manifest.json': Manifesto legível de resumo.
        2. base_path + '_plan.npz': Pacote executável binário seguro sem pickle com assinatura de integridade.
        """
        dir_name = os.path.dirname(base_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        manifest_file = base_path + "_manifest.json" if not base_path.endswith(".json") else base_path
        npz_file = base_path + "_plan.npz" if not base_path.endswith(".npz") else base_path

        # 1. Salva Manifesto JSON (apenas metadados/resumo)
        manifest_data = self.to_manifest_dict()
        with atomic_open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        # 2. Prepara estruturas para o NPZ com hash de integridade (P1.2 e P1.3)
        content_hash = self.compute_content_hash()
        npz_payload = {
            "plan_id": np.array(self.plan_id),
            "fingerprint": np.array(self.fingerprint),
            "content_hash": np.array(content_hash),
            "mode": np.array(self.mode),
            "image_class": np.array(self.image_class),
            "transform": np.array(self._transform_values()),
            "layers_json": np.array(json.dumps(self.layers, ensure_ascii=False)),
            "quality_gate_json": np.array(json.dumps(self.quality_gate, ensure_ascii=False)),
            "profile_metrics_json": np.array(json.dumps(self.profile_metrics, ensure_ascii=False)),
            "source_image": np.array(self.source_image),
            "estimated_time_s": np.array(self.estimated_time_s),
            "total_strokes": np.array(self.total_strokes)
        }

        with atomic_open(npz_file, "wb") as f:
            np.savez_compressed(f, **npz_payload)
        return manifest_file, npz_file

    @classmethod
    def load(cls, base_path: str) -> "ExecutionArtifact":
        """Carrega o artefato executável completo a partir do .npz seguro sem pickle com verificação de integridade (P1.2, P1.3)."""
        npz_file = base_path if base_path.endswith(".npz") else base_path + "_plan.npz"
        manifest_file = base_path if base_path.endswith(".json") else base_path + "_manifest.json"

        # Se base_path for um .json ou o npz_file padrão não existir, tenta resolver o npz irmão
        if not os.path.exists(npz_file):
            alt_npz = base_path.replace("_manifest.json", "_plan.npz")
            if os.path.exists(alt_npz):
                npz_file = alt_npz

        if os.path.exists(npz_file):
            with np.load(npz_file, allow_pickle=False) as data:
                t_vals = data["transform"]
                width, height = (int(t_vals[3]), int(t_vals[4])) if len(t_vals) >= 5 else (0, 0)
                transform = CanvasTransform(offset_x=int(t_vals[0]), offset_y=int(t_vals[1]),
                                            width=width, height=height, scale=float(t_vals[2]))
                loaded_artifact = cls(
                    plan_id=str(data["plan_id"]),
                    fingerprint=str(data["fingerprint"]),
                    mode=str(data["mode"]),
                    image_class=str(data["image_class"]),
                    transform=transform,
                    layers=json.loads(str(data["layers_json"])),
                    quality_gate=json.loads(str(data["quality_gate_json"])),
                    profile_metrics=json.loads(str(data["profile_metrics_json"])),
                    source_image=str(data["source_image"]),
                    estimated_time_s=float(data["estimated_time_s"]),
                    total_strokes=int(data["total_strokes"])
                )
                expected_hash = str(data["content_hash"]) if "content_hash" in data else None

            # Um arquivo sem assinatura não pode ser tratado como plano executável.
            # Observação: SHA-256 sem chave detecta corrupção, não adulteração deliberada.
            if expected_hash is None:
                raise ValueError(f"Falha de integridade: artefato '{npz_file}' sem hash de conteúdo.")
            actual_hash = loaded_artifact.compute_content_hash()
            if expected_hash != actual_hash:
                raise ValueError(f"Falha de integridade (P1.2): Artefato binário '{npz_file}' adulterado ou corrompido.")

            return loaded_artifact

        elif os.path.exists(manifest_file):
            # O manifesto JSON contém apenas metadados e contagem de traços; não é executável sem os traços reais do .npz (P1.3)
            raise ValueError(
                f"Manifesto JSON '{manifest_file}' contém apenas metadados resumidos (sem traços brutos). "
                f"Para execução física é obrigatório o artefato binário .npz correspondente (P1.3)."
            )
        else:
            raise FileNotFoundError(f"Nenhum artefato encontrado em '{npz_file}' ou '{manifest_file}'")


class ArtifactCacheManager:
    """Gerenciador de cache persistente em disco para artefatos executáveis."""
    CACHE_DIR = "cache"

    @classmethod
    def get_cache_path(cls, fingerprint: str) -> str:
        return os.path.join(cls.CACHE_DIR, fingerprint)

    @classmethod
    def lookup(cls, fingerprint: str) -> Optional[ExecutionArtifact]:
        """Procura o artefato compilado no cache persistente pelo hash SHA-256."""
        c_path = cls.get_cache_path(fingerprint)
        npz_file = c_path + "_plan.npz"
        if os.path.exists(npz_file):
            try:
                artifact = ExecutionArtifact.load(c_path)
                return artifact if artifact.fingerprint == fingerprint else None
            except Exception:
                return None
        return None

    @classmethod
    def put(cls, artifact: ExecutionArtifact) -> Tuple[str, str]:
        """Armazena o artefato compilado no cache persistente."""
        c_path = cls.get_cache_path(artifact.fingerprint)
        return artifact.save(c_path)
