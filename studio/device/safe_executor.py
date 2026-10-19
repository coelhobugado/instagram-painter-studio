"""
Executor de Micro-Lotes com Tokens de Identificação de Processo e Checkpoint Persistente em Disco.
Implementa a Seção 44 (Process-Specific Kill), Seção 45 (Execution Token) e Seção 50/51 (Checkpoint em Disco e Resume Seguro).
"""
import os
import json
import uuid
import time
from typing import Dict, Any, Optional

class PersistentCheckpointManager:
    CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "checkpoint.json")

    @classmethod
    def save_checkpoint(cls, plan_hash: str, layer_idx: int, stroke_idx: int,
                        device_id: str, metadata: Dict[str, Any] = None):
        """Salva o progresso atômico em disco de forma persistente."""
        data = {
            "plan_hash": plan_hash,
            "layer_index": layer_idx,
            "stroke_index": stroke_idx,
            "device_id": device_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metadata": metadata or {}
        }
        tmp_path = cls.CHECKPOINT_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cls.CHECKPOINT_FILE)

    @classmethod
    def load_checkpoint(cls) -> Optional[Dict[str, Any]]:
        """Lê o último checkpoint gravado se existir."""
        if os.path.exists(cls.CHECKPOINT_FILE):
            try:
                with open(cls.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    @classmethod
    def clear_checkpoint(cls):
        """Limpa o arquivo de checkpoint após finalização com sucesso."""
        if os.path.exists(cls.CHECKPOINT_FILE):
            try:
                os.remove(cls.CHECKPOINT_FILE)
            except Exception:
                pass

class ExecutionTokenManager:
    @staticmethod
    def generate_token() -> str:
        """Gera identificador único para o processo de desenho (Seção 45)."""
        return f"ig_painter_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def build_tracked_script(commands: list, token: str) -> str:
        """Envolve o script de micro-lotes em um identificador rastreável com gravação de PID."""
        lines = [
            "#!/bin/sh\n",
            f"# TOKEN: {token}\n",
            f"echo $$ > /data/local/tmp/{token}.pid\n"
        ]
        lines.extend(commands)
        lines.append(f"rm -f /data/local/tmp/{token}.pid\n")
        return "".join(lines)
