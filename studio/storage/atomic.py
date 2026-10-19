# -*- coding: utf-8 -*-
"""Escrita atômica: grava em arquivo temporário no mesmo diretório e troca com os.replace."""
import os
import tempfile
from contextlib import contextmanager


@contextmanager
def atomic_open(path, mode="wb", **kwargs):
    """Abre um temporário irmão de ``path``; só substitui o destino se o bloco terminar sem erro."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=os.path.basename(path), dir=directory)
    try:
        with os.fdopen(fd, mode, **kwargs) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
