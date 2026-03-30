from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.models import PipelineConfig
from app.core.exceptions import ConfigurationError


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    raw_text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        payload = _load_json(raw_text, config_path)
    elif suffix in {".yaml", ".yml"}:
        payload = _load_yaml(raw_text, config_path)
    else:
        raise ConfigurationError(f"Unsupported config file type: {config_path.suffix}")

    if not isinstance(payload, dict):
        raise ConfigurationError("Config root must be a JSON/YAML object")

    try:
        return PipelineConfig.from_dict(payload)
    except Exception as exc:
        raise ConfigurationError(f"Invalid config structure in: {config_path}") from exc


def _load_json(raw_text: str, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON config: {path}") from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(f"JSON config must be an object: {path}")
    return payload


def _load_yaml(raw_text: str, path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigurationError(
            "PyYAML is required to load YAML config files"
        ) from exc

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML config: {path}") from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(f"YAML config must be an object: {path}")
    return payload