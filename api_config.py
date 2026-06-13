from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class APIConfig:
    max_concurrent_processes: int
    base_config_path: Path
    default_output_dir: str | None
    max_wait_timeout_seconds: int
    default_wait_timeout_seconds: int

    @classmethod
    def from_env(cls) -> "APIConfig":
        raw_max = os.environ.get("MAX_CONCURRENT_PROCESSES", "2").strip()
        try:
            max_concurrent = int(raw_max)
            if max_concurrent < 1:
                raise ValueError
        except ValueError:
            raise RuntimeError(
                f"MAX_CONCURRENT_PROCESSES must be a positive integer, got: {raw_max!r}"
            )

        raw_config = os.environ.get("PIPELINE_CONFIG_PATH", "").strip()
        if raw_config:
            base_config_path = Path(raw_config).resolve()
            if not base_config_path.exists():
                raise RuntimeError(
                    f"PIPELINE_CONFIG_PATH does not exist: {base_config_path}"
                )
        else:
            base_config_path = Path(__file__).parent / "config.json"
            if not base_config_path.exists():
                raise RuntimeError(
                    f"Default config.json not found at: {base_config_path}"
                )

        default_output_dir = os.environ.get("DEFAULT_OUTPUT_DIR", "").strip() or None

        raw_max_wait = os.environ.get("MAX_WAIT_TIMEOUT_SECONDS", "600").strip()
        try:
            max_wait_timeout_seconds = int(raw_max_wait)
            if max_wait_timeout_seconds < 10:
                raise ValueError
        except ValueError:
            raise RuntimeError(
                f"MAX_WAIT_TIMEOUT_SECONDS must be an integer >= 10, got: {raw_max_wait!r}"
            )

        raw_default_wait = os.environ.get("DEFAULT_WAIT_TIMEOUT_SECONDS", "540").strip()
        try:
            default_wait_timeout_seconds = int(raw_default_wait)
            if default_wait_timeout_seconds < 10:
                raise ValueError
        except ValueError:
            raise RuntimeError(
                f"DEFAULT_WAIT_TIMEOUT_SECONDS must be an integer >= 10, got: {raw_default_wait!r}"
            )

        if default_wait_timeout_seconds > max_wait_timeout_seconds:
            raise RuntimeError(
                f"DEFAULT_WAIT_TIMEOUT_SECONDS ({default_wait_timeout_seconds}) "
                f"must not exceed MAX_WAIT_TIMEOUT_SECONDS ({max_wait_timeout_seconds})"
            )

        return cls(
            max_concurrent_processes=max_concurrent,
            base_config_path=base_config_path,
            default_output_dir=default_output_dir,
            max_wait_timeout_seconds=max_wait_timeout_seconds,
            default_wait_timeout_seconds=default_wait_timeout_seconds,
        )