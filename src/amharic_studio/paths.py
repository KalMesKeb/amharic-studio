"""Filesystem locations. Everything stays on this machine; nothing is fetched at runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "AmharicStudio"


def user_data_dir() -> Path:
    """Per-user writable directory for the lexicon, models and learned confusion data."""
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
        return Path(root) / APP_DIR_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    root = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(root) / APP_DIR_NAME.lower()


def user_config_dir() -> Path:
    if sys.platform == "win32":
        return user_data_dir()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Preferences" / APP_DIR_NAME
    root = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(root) / APP_DIR_NAME.lower()


def cache_dir() -> Path:
    return user_data_dir() / "cache"


def models_dir() -> Path:
    """OCR models, fine-tuned traineddata and any neural correction weights."""
    return user_data_dir() / "models"


def package_root() -> Path:
    return Path(__file__).resolve().parent


def resources_dir() -> Path:
    return package_root() / "resources"


def bundled_fonts_dir() -> Path:
    return resources_dir() / "fonts"


def ensure_dirs() -> None:
    for path in (user_data_dir(), cache_dir(), models_dir(), user_data_dir() / "lexicon"):
        path.mkdir(parents=True, exist_ok=True)
