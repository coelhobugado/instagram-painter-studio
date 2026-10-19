"""Formato local de projeto editável, sem pickle nem objetos executáveis."""
import json
import math
from pathlib import Path

import numpy as np

from ..core.models import CanvasTransform
from .atomic import atomic_open


SCHEMA_VERSION = 1


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Tipo não serializável: {type(value).__name__}")


def save_project(path, plan):
    """Guarda apenas entradas editáveis; artefatos derivados são reconstruídos na abertura."""
    if not plan or plan.get("fit_image") is None or plan.get("target_executable") is None:
        raise ValueError("Plano incompleto para salvar")
    transform = plan.get("transform")
    transform_data = (
        {"offset_x": transform.offset_x, "offset_y": transform.offset_y,
         "width": transform.width, "height": transform.height, "scale": transform.scale}
        if transform is not None else None
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "mode": plan.get("mode", "layered_paint"),
        "seed": int(plan.get("seed", 42)),
        "revision": int(plan.get("revision", 0)),
        "time_budget_s": float(plan.get("time_budget_s", 600.0)),
        "source_image_hash": plan.get("source_image_hash", ""),
        "transform": transform_data,
        "custom_bounds": plan.get("custom_bounds"),
        "layers": plan["layers"],
        "palette_colors": plan.get("palette_colors", []),
        "editable_regions": plan.get("editable_regions", []),
    }
    data = json.dumps(metadata, ensure_ascii=False, default=_json_default).encode("utf-8")
    target_path = Path(path)
    with atomic_open(target_path, "wb") as output:
        np.savez_compressed(
            output,
            metadata=np.frombuffer(data, dtype=np.uint8),
            fit_image=np.asarray(plan["fit_image"], dtype=np.uint8),
            target_executable=np.asarray(plan["target_executable"], dtype=np.uint8),
            region_map=np.asarray(plan.get("region_map", np.zeros(plan["fit_image"].shape[:2])), dtype=np.int32),
        )


def load_project(path, engine):
    """Abre um projeto e recalcula prévia, movimentos, métricas e gate de qualidade."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Versão do projeto incompatível")
        fit = archive["fit_image"].copy()
        target = archive["target_executable"].copy()
        region_map = archive["region_map"].copy()
    if fit.ndim != 3 or fit.shape[2] != 3 or target.shape != fit.shape or region_map.shape != fit.shape[:2]:
        raise ValueError("Dimensões de projeto inválidas")
    if fit.shape[0] > 4096 or fit.shape[1] > 4096:
        raise ValueError("Projeto excede o limite de tamanho seguro")
    transform_data = metadata.get("transform")
    transform = CanvasTransform(**transform_data) if transform_data else None
    custom_bounds = metadata.get("custom_bounds")
    if custom_bounds is not None:
        if not isinstance(custom_bounds, dict) or not all(
            key in custom_bounds for key in ("left", "top", "right", "bottom")
        ):
            raise ValueError("Área personalizada inválida no projeto")
        if not all(
            isinstance(custom_bounds[key], (int, float)) and math.isfinite(custom_bounds[key])
            for key in ("left", "top", "right", "bottom")
        ):
            raise ValueError("Coordenadas inválidas na área personalizada")
        if custom_bounds["left"] >= custom_bounds["right"] or custom_bounds["top"] >= custom_bounds["bottom"]:
            raise ValueError("Área personalizada invertida no projeto")
    layers = metadata["layers"]
    if not isinstance(layers, list) or len(layers) > 10000:
        raise ValueError("Camadas inválidas")
    for layer in layers:
        strokes = layer.get("strokes", [])
        if not isinstance(strokes, list) or len(strokes) > 100000:
            raise ValueError("Traços inválidos")
        if len(layer.get("stroke_region_ids", [])) not in (0, len(strokes)):
            raise ValueError("Associação de áreas e traços inválida")
    plan = {
        "mode": metadata["mode"], "seed": metadata["seed"],
        "revision": metadata["revision"] - 1,
        "time_budget_s": metadata["time_budget_s"],
        "source_image_hash": metadata.get("source_image_hash", ""),
        "transform": transform, "layers": layers,
        "custom_bounds": custom_bounds,
        "palette_colors": metadata["palette_colors"],
        "editable_regions": metadata["editable_regions"],
        "fit_image": fit, "target_executable": target, "region_map": region_map,
    }
    return engine._refresh_edited_plan(plan)
