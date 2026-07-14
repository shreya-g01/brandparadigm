"""Small filesystem helpers used across scripts and package modules."""

import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write `data` as JSON, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def read_json(path: str | Path) -> Any:
    """Read a JSON file into a Python object."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create `path` (and parents) if it doesn't exist, return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
