from __future__ import annotations

from app.config.models import PipelineConfig
from app.core.exceptions import ConfigurationError


def validate_config(config: PipelineConfig) -> None:
    _validate_mineru(config)
    _validate_captioning(config)
    _validate_output(config)


def _validate_mineru(config: PipelineConfig) -> None:
    if not config.mineru.command.strip():
        raise ConfigurationError("mineru.command must not be empty")
    if not config.mineru.backend.strip():
        raise ConfigurationError("mineru.backend must not be empty")


def _validate_captioning(config: PipelineConfig) -> None:
    if config.captioning.max_retries < 0:
        raise ConfigurationError("captioning.max_retries must be >= 0")
    if config.captioning.timeout_seconds <= 0:
        raise ConfigurationError("captioning.timeout_seconds must be > 0")
    if config.captioning.enabled:
        if not (config.apis.openai_api_key or "").strip():
            raise ConfigurationError(
                "apis.openai_api_key is required when captioning is enabled"
            )
        if not (config.apis.openai_base_url or "").strip():
            raise ConfigurationError(
                "apis.openai_base_url is required when captioning is enabled"
            )


def _validate_output(config: PipelineConfig) -> None:
    if not config.output.root_dir.strip():
        raise ConfigurationError("output.root_dir must not be empty")