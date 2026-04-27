from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MinerUConfig:
    enabled: bool = True
    command: str = "mineru"
    backend: str = "pipeline"
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CaptioningConfig:
    enabled: bool = False
    output_suffix: str = "_captioned"
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
class StructuredJsonConfig:
    enabled: bool = False


@dataclass(slots=True)
class ApiConfig:
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_api_version: str = "2023-05-15"
    openai_model: str = "gpt-5-2025-08-07"


@dataclass(slots=True)
class ResumeConfig:
    enabled: bool = True
    skip_completed_stages: bool = True
    verify_existing_files: bool = True
    retry_failed_stage_only: bool = True


@dataclass(slots=True)
class OutputConfig:
    root_dir: str = "data"
    input_dir_name: str = "input"
    manifests_dir_name: str = "manifests"
    mineru_raw_dir_name: str = "mineru_raw"
    markdown_dir_name: str = "markdown"
    captioned_markdown_dir_name: str = "captioned_markdown"
    structured_json_dir_name: str = "structured_json"
    reports_dir_name: str = "reports"


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: str | None = None


@dataclass(slots=True)
class PipelineConfig:
    mineru: MinerUConfig = field(default_factory=MinerUConfig)
    captioning: CaptioningConfig = field(default_factory=CaptioningConfig)
    structured_json: StructuredJsonConfig = field(default_factory=StructuredJsonConfig)
    apis: ApiConfig = field(default_factory=ApiConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            mineru=MinerUConfig(**dict(data.get("mineru") or {})),
            captioning=CaptioningConfig(**dict(data.get("captioning") or {})),
            structured_json=StructuredJsonConfig(**dict(data.get("structured_json") or {})),
            apis=ApiConfig(**dict(data.get("apis") or {})),
            resume=ResumeConfig(**dict(data.get("resume") or {})),
            output=OutputConfig(**dict(data.get("output") or {})),
            logging=LoggingConfig(**dict(data.get("logging") or {})),
        )