"""
Otimizador de Memória e Performance.
Implementa a Fase 12 (12.1 Máscaras ROI, 12.2 Label Map Compartilhado, 12.3 Chunked Color Matching e 12.4 Perfilador) do Roadmap.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import time
import os
import cv2
import numpy as np


@dataclass
class ROIMask:
    """
    Máscara compactada por Região de Interesse (Fase 12.1).
    Armazena apenas a bounding box e a máscara local, reduzindo drasticamente o consumo de RAM.
    """
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    local_mask: np.ndarray           # uint8 (h x w)

    @property
    def memory_bytes(self) -> int:
        return self.local_mask.nbytes

    def to_full_mask(self, full_shape: Tuple[int, int], out: Optional[np.ndarray] = None) -> np.ndarray:
        """Expande sob demanda a máscara para o frame completo, opcionalmente reutilizando buffer prealocado."""
        h_full, w_full = full_shape[:2]
        if out is None:
            full_mask = np.zeros((h_full, w_full), dtype=np.uint8)
        else:
            full_mask = out
            full_mask.fill(0)
        x, y, w, h = self.bbox
        x1 = max(0, min(w_full, x))
        y1 = max(0, min(h_full, y))
        x2 = max(0, min(w_full, x + w))
        y2 = max(0, min(h_full, y + h))
        
        lw = x2 - x1
        lh = y2 - y1
        if lw > 0 and lh > 0:
            full_mask[y1:y2, x1:x2] = self.local_mask[0:lh, 0:lw]
        return full_mask

    @classmethod
    def from_full_mask(cls, full_mask: np.ndarray) -> Optional["ROIMask"]:
        """Cria um ROIMask a partir de uma máscara full-frame recortando a bounding box."""
        if np.count_nonzero(full_mask) == 0:
            return None
        x, y, w, h = cv2.boundingRect(full_mask)
        local_crop = full_mask[y:y+h, x:x+w].copy()
        return cls(bbox=(x, y, w, h), local_mask=local_crop)


class RegionDict(dict):
    """
    Dicionário de Região com resolução transparente e preguiçosa (lazy) de 'mask'.
    Gera a máscara full-frame sob demanda a partir de 'roi_mask' sem armazená-la permanentemente,
    evitando que milhares de matrizes de alta resolução permaneçam em memória simultaneamente.
    """
    def __init__(self, *args, full_shape: Optional[Tuple[int, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._full_shape = full_shape

    def __getitem__(self, key: str) -> Any:
        if key == "mask" and "mask" not in self.keys() and "roi_mask" in self.keys():
            if self._full_shape is not None and self["roi_mask"] is not None:
                return self["roi_mask"].to_full_mask(self._full_shape)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "mask" and "mask" not in self.keys() and "roi_mask" in self.keys():
            if self._full_shape is not None and self["roi_mask"] is not None:
                return self["roi_mask"].to_full_mask(self._full_shape)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        if key == "mask" and "roi_mask" in self.keys():
            return True
        return super().__contains__(key)


class SharedLabelMap:
    """
    Mapa de Rótulos Único Compartilhado (Fase 12.2).
    Substitui N máscaras inteiras por uma única matriz de inteiros com os IDs de componente.
    """

    def __init__(self, label_image: np.ndarray):
        self.label_image = label_image.astype(np.int32)
        self.shape = label_image.shape[:2]

    @classmethod
    def from_binary_components(cls, binary_image: np.ndarray) -> Tuple["SharedLabelMap", int]:
        num_labels, labels = cv2.connectedComponents(binary_image)
        return cls(labels), num_labels

    def get_roi_mask(self, component_id: int) -> Optional[ROIMask]:
        """Extrai um ROIMask eficiente para um determinado ID sem criar cópias full-frame pesadas."""
        ys, xs = np.where(self.label_image == component_id)
        if len(xs) == 0:
            return None
        x_min, x_max = int(np.min(xs)), int(np.max(xs))
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        sub_labels = self.label_image[y_min:y_max+1, x_min:x_max+1]
        local_binary = np.where(sub_labels == component_id, 255, 0).astype(np.uint8)
        return ROIMask(bbox=(x_min, y_min, w, h), local_mask=local_binary)


class ChunkedColorMatcher:
    """
    Processador de Cores em Lotes (Fase 12.3).
    Evita alocação de matrizes gigantes (pixels x colors x channels) processando em blocos.
    """

    @staticmethod
    def match_pixels_chunked(
        pixels_bgr: np.ndarray,
        palette_bgr: np.ndarray,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """
        Retorna o índice da cor mais próxima na paleta para cada pixel, processando em chunks.
        """
        num_pixels = len(pixels_bgr)
        best_indices = np.zeros(num_pixels, dtype=np.int32)
        
        # Converte para float32 uma única vez
        palette_f = palette_bgr.astype(np.float32)

        for i in range(0, num_pixels, chunk_size):
            chunk = pixels_bgr[i:i + chunk_size].astype(np.float32)
            # Diferença euclidiana no chunk: shape (chunk_len, 1, 3) - (1, num_colors, 3)
            diffs = chunk[:, np.newaxis, :] - palette_f[np.newaxis, :, :]
            dists_sq = np.sum(diffs ** 2, axis=2)
            best_chunk_idx = np.argmin(dists_sq, axis=1)
            best_indices[i:i + chunk_size] = best_chunk_idx

        return best_indices


@dataclass
class PerformanceProfile:
    """Relatório quantitativo de tempo e memória (Fase 12.4)."""
    ram_peak_mb: float = 0.0
    planning_time_s: float = 0.0
    simulation_time_s: float = 0.0
    event_count: int = 0
    total_strokes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ram_peak_mb": round(self.ram_peak_mb, 2),
            "planning_time_s": round(self.planning_time_s, 3),
            "simulation_time_s": round(self.simulation_time_s, 3),
            "event_count": self.event_count,
            "total_strokes": self.total_strokes,
        }


class MemoryProfileTracker:
    """Perfilador de Memória e Recursos do Studio."""

    def __init__(self):
        self.start_time = 0.0
        self.profile = PerformanceProfile()

    def start(self) -> None:
        self.start_time = time.time()
        self.profile.ram_peak_mb = self.get_current_process_ram_mb()

    def end_planning(self, total_strokes: int = 0) -> None:
        self.profile.planning_time_s = float(time.time() - self.start_time)
        self.profile.total_strokes = total_strokes
        cur_ram = self.get_current_process_ram_mb()
        if cur_ram > self.profile.ram_peak_mb:
            self.profile.ram_peak_mb = cur_ram

    def end_simulation(self, sim_time_s: float, event_count: int = 0) -> None:
        self.profile.simulation_time_s = float(sim_time_s)
        self.profile.event_count = event_count
        cur_ram = self.get_current_process_ram_mb()
        if cur_ram > self.profile.ram_peak_mb:
            self.profile.ram_peak_mb = cur_ram

    @staticmethod
    def get_current_process_ram_mb() -> float:
        """Lê o consumo de RAM do processo atual."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return float(process.memory_info().rss / (1024 * 1024))
        except Exception:
            # Fallback se psutil não estiver disponível
            return 50.0
