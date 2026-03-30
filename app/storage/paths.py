from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config.models import OutputConfig


_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_path_component(value: str, *, max_length: int = 180) -> str:
    cleaned = _SAFE_COMPONENT_RE.sub("_", value.strip()).strip("._")
    if not cleaned:
        cleaned = "unknown"
    return cleaned[:max_length]


@dataclass(slots=True)
class PathResolver:
    output: OutputConfig

    @property
    def root_dir(self) -> Path:
        return Path(self.output.root_dir).resolve()

    @property
    def manifests_dir(self) -> Path:
        return self.root_dir / self.output.manifests_dir_name

    @property
    def mineru_raw_dir(self) -> Path:
        return self.root_dir / self.output.mineru_raw_dir_name

    @property
    def markdown_dir(self) -> Path:
        return self.root_dir / self.output.markdown_dir_name

    @property
    def captioned_markdown_dir(self) -> Path:
        return self.root_dir / self.output.captioned_markdown_dir_name

    def ensure_base_dirs(self) -> None:
        for path in (
            self.root_dir,
            self.manifests_dir,
            self.mineru_raw_dir,
            self.markdown_dir,
            self.captioned_markdown_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def normalized_key(self, paper_key: str) -> str:
        return sanitize_path_component(paper_key)

    def manifest_path(self, paper_key: str) -> Path:
        return self.manifests_dir / f"{self.normalized_key(paper_key)}.json"

    def mineru_dir(self, paper_key: str) -> Path:
        return self.mineru_raw_dir / self.normalized_key(paper_key)

    def markdown_path(self, paper_key: str, extension: str = ".md") -> Path:
        ext = extension if extension.startswith(".") else f".{extension}"
        return self.markdown_dir / f"{self.normalized_key(paper_key)}{ext}"

    def captioned_markdown_path(self, paper_key: str, extension: str = ".md") -> Path:
        ext = extension if extension.startswith(".") else f".{extension}"
        return self.captioned_markdown_dir / f"{self.normalized_key(paper_key)}{ext}"