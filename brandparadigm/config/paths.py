"""Canonical filesystem locations shared across the package."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = RAW_DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DOCS_DIR = PROJECT_ROOT / "docs"
