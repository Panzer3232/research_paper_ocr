from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any] | list[Any], *, indent: int = 2) -> Path:
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def write_text(path: Path, text: str) -> Path:
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return path


def write_bytes(path: Path, data: bytes) -> Path:
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)
    return path