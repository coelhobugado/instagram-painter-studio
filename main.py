"""Inicia a interface do Instagram Painter Studio com ``python main.py``."""
from pathlib import Path
import os


def main() -> int:
    # Planos, imagens e relatórios usam caminhos relativos à raiz do projeto.
    os.chdir(Path(__file__).resolve().parent)
    # A interface só é carregada quando o usuário realmente abre o programa.
    # Importar este módulo para testes não inicia janela nem busca dispositivos.
    from studio.app_gui import launch_app

    launch_app()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
