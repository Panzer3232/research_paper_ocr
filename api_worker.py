from __future__ import annotations

import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any


_PACKAGE_ROOT = Path(__file__).parent.resolve()
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def run_ocr_job(
    job_id: str,
    input_path: str,
    result_path: str,
    caption: bool,
    export_json: bool,
    chunk: bool,
    evaluate: bool,
    output_dir: str | None,
    config_overrides: dict,
) -> None:
    from ocr import chunk as run_chunk
    from ocr import enable_logging, ocr
    from app.config.loader import load_config
    from app.config.models import PipelineConfig

    enable_logging(level="INFO")
    logger = logging.getLogger("paper_ocr")

    # When chunking is requested, structured JSON is a hard requirement.
    # Force it regardless of what the caller specified.
    if chunk:
        export_json = True

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

    #Stage 1: OCR
    try:
        ocr_results = ocr(
            input_path,
            caption=caption,
            export_json=export_json,
            config=config_obj,
            output_dir=output_dir,
        )
    except Exception:
        _write_failure(result_path, job_id, traceback.format_exc())
        sys.exit(1)

    ocr_succeeded = [r for r in ocr_results if r.success]

    payload: dict[str, Any] = {
        "job_id": job_id,
        "success": len(ocr_succeeded) == len(ocr_results) and bool(ocr_results),
        "total": len(ocr_results),
        "succeeded": len(ocr_succeeded),
        "failed": len(ocr_results) - len(ocr_succeeded),
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
            for r in ocr_results
        ],
        "chunking": {"ran": False},
    }

    # Stage 2: RAG Chunking 
    if chunk:
        if not ocr_succeeded:
            logger.warning(
                "job=%s: chunking requested but no papers succeeded OCR — skipping.",
                job_id,
            )
            payload["chunking"] = {
                "ran": False,
                "skipped_reason": "no_ocr_succeeded",
            }
        else:
            structured_json_dir = Path(ocr_succeeded[0].structured_json_path).parent
            chunk_payload = _run_chunk_stage(
                job_id=job_id,
                structured_json_dir=structured_json_dir,
                evaluate=evaluate,
                output_dir=output_dir,
                logger=logger,
            )
            payload["chunking"] = chunk_payload

            # Overall job success requires both stages to be fully clean.
            if not chunk_payload.get("success"):
                payload["success"] = False

    _write_result(result_path, payload)
    sys.exit(0)


def _run_chunk_stage(
    job_id: str,
    structured_json_dir: Path,
    evaluate: bool,
    output_dir: str | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Run the RAG chunking pipeline on the structured JSON directory produced
    by the OCR stage. Returns a dict suitable for payload["chunking"].
    Never raises — exceptions are caught, logged, and reflected in the return value.
    """
    from ocr import chunk as run_chunk

    try:
        chunk_results = run_chunk(
            structured_json_dir=structured_json_dir,
            evaluate=evaluate,
            output_dir=output_dir,
        )
    except Exception:
        tb = traceback.format_exc()
        logger.error("job=%s: chunking stage raised an exception:\n%s", job_id, tb)
        return {
            "ran": True,
            "success": False,
            "succeeded": 0,
            "failed": 0,
            "worker_traceback": tb,
            "results": [],
        }

    chunk_succeeded = [r for r in chunk_results if r.success]

    return {
        "ran": True,
        "success": len(chunk_succeeded) == len(chunk_results) and bool(chunk_results),
        "succeeded": len(chunk_succeeded),
        "failed": len(chunk_results) - len(chunk_succeeded),
        "results": [
            {
                "paper_key": r.paper_key,
                "success": r.success,
                "status": r.status,
                "normalized_json_path": r.normalized_json_path,
                "chunk_json_path": r.chunk_json_path,
                "evaluation_report_path": r.evaluation_report_path,
                "error": r.error,
            }
            for r in chunk_results
        ],
    }


def _apply_overrides(config: Any, overrides: dict) -> None:
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
        if section_obj is None or not hasattr(section_obj, field_name):
            continue
        setattr(section_obj, field_name, value)


def _write_result(result_path: str, payload: dict) -> None:
    Path(result_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_failure(result_path: str, job_id: str, tb: str) -> None:
    try:
        _write_result(
            result_path,
            {
                "job_id": job_id,
                "success": False,
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "results": [],
                "chunking": {"ran": False},
                "worker_traceback": tb,
            },
        )
    except Exception:
        pass