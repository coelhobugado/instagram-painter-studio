# -*- coding: utf-8 -*-
"""
Utilitário de Limpeza e Manutenção de Artefatos (P2.4).
Remove caches órfãos, screenshots temporários e arquivos descartáveis sem afetar insumos ou benchmarks.
"""
import os
import glob
import time


def cleanup_directory(directory_path: str, extensions: list = None, dry_run: bool = False):
    """Varre e remove arquivos descartáveis de um diretório específico com base em extensões ou todos."""
    removed = []
    if not os.path.exists(directory_path):
        return removed
    for root, dirs, files in os.walk(directory_path):
        for f in files:
            if extensions:
                if not any(f.endswith(ext) for ext in extensions):
                    continue
            fp = os.path.join(root, f)
            if not dry_run:
                try:
                    os.remove(fp)
                except Exception:
                    continue
            removed.append(fp)
    return removed


def cleanup_project_artifacts(keep_recent_minutes: int = 60, dry_run: bool = False):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    targets = [
        os.path.join(project_root, "cache"),
        os.path.join(project_root, "outputs", "temp"),
        os.path.join(project_root, "outputs", "miranha_previews")
    ]

    total_removed = 0
    total_bytes = 0
    # Arquivos modificados dentro da janela recente são preservados (planos em uso, sessões abertas)
    cutoff = time.time() - max(0, keep_recent_minutes) * 60

    print(f"[*] Iniciando varredura de limpeza de artefatos em: {project_root}")
    for target in targets:
        if os.path.exists(target):
            for root, dirs, files in os.walk(target):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        if os.path.getmtime(fp) > cutoff:
                            continue
                        sz = os.path.getsize(fp)
                        if not dry_run:
                            os.remove(fp)
                        total_removed += 1
                        total_bytes += sz
                    except Exception as e:
                        print(f"    - Erro ao remover {fp}: {e}")

    # Remove arquivos .tmp ou checkpoints obsoletos na raiz
    for tmp_file in glob.glob(os.path.join(project_root, "*.tmp")):
        try:
            if os.path.getmtime(tmp_file) > cutoff:
                continue
            sz = os.path.getsize(tmp_file)
            if not dry_run:
                os.remove(tmp_file)
            total_removed += 1
            total_bytes += sz
        except Exception:
            pass

    mb = total_bytes / (1024 * 1024)
    mode_str = "[SIMULAÇÃO]" if dry_run else "[CONCLUÍDO]"
    print(f"{mode_str} Total de arquivos removidos: {total_removed} ({mb:.2f} MB liberados)")


if __name__ == "__main__":
    cleanup_project_artifacts(dry_run=False)
