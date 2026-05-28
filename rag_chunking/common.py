from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

CHARS_PER_TOKEN = 4
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(r"[A-Za-z][^.!?]{20,}[.!?]")


def setup_logger(name: str, log_file: Path | None = None, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def iter_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("*.json") if not p.name.startswith("_"))


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = CONTROL_CHARS_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return WHITESPACE_RE.sub(" ", text).strip()


def stable_id(*parts: object, length: int = 16) -> str:
    payload = "::".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def estimate_tokens(text: str | None) -> int:
    text = text or ""
    return max(1, len(text) // CHARS_PER_TOKEN) if text else 0


def first_non_null(values: Iterable[Any]) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


def page_span_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, int | None]:
    pages = [b.get("page_idx") for b in blocks if isinstance(b.get("page_idx"), int)]
    if not pages:
        return {"start": None, "end": None}
    return {"start": min(pages), "end": max(pages)}


def sentence_like(text: str) -> bool:
    if not text:
        return False
    if SENTENCE_RE.search(text):
        return True
    words = re.findall(r"[A-Za-z]{2,}", text)
    return len(words) >= 8


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 640
    overlap: int = 64
    table_chunk_size: int = 768
    min_prose_tokens: int = 80
    max_context_blocks: int = 2
    split_long_tables: bool = True
