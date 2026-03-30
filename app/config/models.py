from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MinerUConfig:
    command: str = "mineru"
    backend: str = "pipeline"
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CaptioningConfig:
    enabled: bool = False
    max_retries: int = 2
    timeout_seconds: int = 180
    caption_prompt: str = (
        "Analyze this research paper figure and provide a detailed description "
        "under 300 characters. Focus on: 1. Chart/Diagram Type "
        "2. Main trend or component demonstrated "
        "3. Key data points or structural elements. "
        "Keep it concise but detailed enough for understanding without the image."
    )


@dataclass(slots=True)
class ApiConfig:
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_api_version: str = "2023-05-15"
    openai_model: str = "gpt-5-2025-08-07"

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        if not self.openai_base_url:
            self.openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")


@dataclass(slots=True)
class ResumeConfig:
    enabled: bool = True
    skip_completed_stages: bool = True


@dataclass(slots=True)
class OutputConfig:
    root_dir: str = "data_ocr"
    input_dir_name: str = "input"
    manifests_dir_name: str = "manifests"
    mineru_raw_dir_name: str = "mineru_raw"
    markdown_dir_name: str = "markdown"
    captioned_markdown_dir_name: str = "captioned_markdown"


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: str | None = None


@dataclass(slots=True)
class PipelineConfig:
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    apis: ApiConfig = field(default_factory=ApiConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            mineru=MinerUConfig(**dict(data.get("mineru") or {})),
            captioning=CaptioningConfig(**dict(data.get("captioning") or {})),
            apis=ApiConfig(**dict(data.get("apis") or {})),
            resume=ResumeConfig(**dict(data.get("resume") or {})),
            output=OutputConfig(**dict(data.get("output") or {})),
            logging=LoggingConfig(**dict(data.get("logging") or {})),
        )