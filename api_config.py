from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class APIConfig:
    max_concurrent_processes: int
    base_config_path: Path
    default_output_dir: str | None

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

        return cls(
            max_concurrent_processes=max_concurrent,
            base_config_path=base_config_path,
            default_output_dir=default_output_dir,
        )