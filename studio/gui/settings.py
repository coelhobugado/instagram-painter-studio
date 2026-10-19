"""Preferências simples do usuário entre sessões (ex.: último celular conectado)."""
import json
import os

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".instagram_painter_studio.json")


def load_settings(path: str = SETTINGS_PATH) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_setting(key: str, value, path: str = SETTINGS_PATH) -> None:
    data = load_settings(path)
    data[key] = value
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
