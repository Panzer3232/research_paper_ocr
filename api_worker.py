from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).parent.resolve()
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def run_ocr_job(
    job_id: str,
    input_path: str,
    result_path: str,
    caption: bool,
    export_json: bool,
    output_dir: str | None,
    config_overrides: dict,
) -> None:
    
    from ocr import enable_logging, ocr
    from app.config.loader import load_config
    from app.config.models import PipelineConfig

    enable_logging(level="INFO")

    config_obj: PipelineConfig | None = None

    if config_overrides:
        base_config_path = config_overrides.pop("_base_config_path", None)
        try:
            if base_config_path:
                config_obj = load_config(base_config_path)
            else:
                config_obj = load_config(Path(__file__).parent / "config.json")

            _apply_overrides(config_obj, config_overrides)
        except Exception:
            _write_failure(result_path, job_id, traceback.format_exc())
            sys.exit(1)

    try:
        results = ocr(
            input_path,
            caption=caption,
            export_json=export_json,
            config=config_obj,
            output_dir=output_dir,
        )

        payload = {
            "job_id": job_id,
            "success": all(r.success for r in results),
            "total": len(results),
            "succeeded": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [
                {
                    "label": r.label,
                    "pdf_path": r.pdf_path,
                    "success": r.success,
                    "status": r.status,
                    "markdown_path": r.markdown_path,
                    "captioned_path": r.captioned_path,
                    "structured_json_path": r.structured_json_path,
                    "error": r.error,
                }
                for r in results
            ],
        }

        Path(result_path).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        sys.exit(0)

    except Exception:
        _write_failure(result_path, job_id, traceback.format_exc())
        sys.exit(1)


def _apply_overrides(config, overrides: dict) -> None:
 
    section_map = {
        "captioning": config.captioning,
        "mineru": config.mineru,
        "logging": config.logging,
        "resume": config.resume,
        "structured_json": config.structured_json,
    }

    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".", 1)
        if len(parts) != 2:
            continue
        section_name, field_name = parts
        section_obj = section_map.get(section_name)
        if section_obj is None:
            continue
        if not hasattr(section_obj, field_name):
            continue
        setattr(section_obj, field_name, value)


def _write_failure(result_path: str, job_id: str, tb: str) -> None:
    try:
        Path(result_path).write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "success": False,
                    "total": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "results": [],
                    "worker_traceback": tb,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass