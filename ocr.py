"""
This module is the single entry point when using paper_ocr as a library
imported by another pipeline or script.

Usage examples
--------------
Minimal (extraction only):

    from paper_ocr.ocr import ocr

    results = ocr("/path/to/paper.pdf")
    for r in results:
        print(r.markdown_path)   # absolute path to .md file

With captioning:

    results = ocr("/path/to/pdfs/", caption=True)
    for r in results:
        print(r.markdown_path)          # plain MinerU .md
        print(r.captioned_path)         # GPT-5 captioned .md (None if captioning failed)

With explicit config and output directory:

    results = ocr(
        "/path/to/pdfs/",
        config_path="/abs/path/config.json",
        output_dir="/abs/path/output/",
        caption=True,
    )

Passing a pre-built PipelineConfig (advanced):

    from paper_ocr.ocr import ocr
    from paper_ocr.app.config.loader import load_config

    config = load_config("/abs/path/config.json")
    config.captioning.enabled = True
    results = ocr("/path/to/pdfs/", config=config)

Return value
------------
A list of :class:`OCRPipelineResult`, one per PDF, in input order.
Iterate and check ``result.success`` before using paths — a failed PDF
will have ``success=False`` and a non-None ``error`` string.

API credentials
---------------
Set in a ``.env`` file at the caller's working directory, or as environment
variables before importing this module:

    OPENAI_API_KEY=...          (or AZURE_OPENAI_API_KEY)
    OPENAI_BASE_URL=...         (or AZURE_OPENAI_ENDPOINT)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.loader import load_config
from app.config.models import PipelineConfig
from app.config.validator import validate_config
from app.pipeline.input_resolver import resolve_ocr_inputs
from app.pipeline.orchestrator import OCROrchestrator, OCRResult

logging.getLogger("paper_ocr").addHandler(logging.NullHandler())

_BUNDLED_CONFIG = Path(__file__).parent / "config.json"


@dataclass(frozen=True)
class OCRPipelineResult:
   
    pdf_path: str
    success: bool
    markdown_path: str | None
    captioned_path: str | None
    error: str | None
    status: str
    label: str

    def effective_markdown(self) -> str | None:
        
        return self.captioned_path or self.markdown_path



def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
        else:
            load_dotenv(override=False)
    except ImportError:
        pass


def _resolve_config(
    config: PipelineConfig | None,
    config_path: str | Path | None,
    output_dir: str | Path | None,
) -> PipelineConfig:
    if config is not None:
        resolved = config
    else:
        if config_path is None:
            cwd_config = Path.cwd() / "config.json"
            config_path = cwd_config if cwd_config.exists() else _BUNDLED_CONFIG
        resolved = load_config(config_path)

    if output_dir is not None:
        resolved.output.root_dir = str(Path(output_dir).resolve())

    return resolved


def _build_captioner(config: PipelineConfig) -> Any | None:
    if not config.captioning.enabled:
        return None

    logger = logging.getLogger("paper_ocr")

    try:
        from app.extract.captioner import MarkdownCaptioner
        from app.storage.paths import PathResolver
    except ImportError:
        logger.warning("captioner module unavailable — captioning will be skipped.")
        return None

    if not (config.apis.openai_api_key or "").strip():
        logger.warning(
            "captioning enabled but OPENAI_API_KEY / AZURE_OPENAI_API_KEY is not set"
            " — captioning will be skipped."
        )
        return None

    if not (config.apis.openai_base_url or "").strip():
        logger.warning(
            "captioning enabled but OPENAI_BASE_URL / AZURE_OPENAI_ENDPOINT is not set"
            " — captioning will be skipped."
        )
        return None

    paths = PathResolver(config.output)
    return MarkdownCaptioner(config.apis, config.captioning, paths)


def _to_public_result(r: OCRResult) -> OCRPipelineResult:
    return OCRPipelineResult(
        pdf_path=r.pdf_path,
        success=r.success,
        markdown_path=r.markdown_path,
        captioned_path=r.captioned_path,
        error=r.error,
        status=r.status,
        label=r.label,
    )


def ocr(
    input_path: str | Path,
    *,
    caption: bool = False,
    config: PipelineConfig | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[OCRPipelineResult]:
    
    _load_dotenv()

    config_obj = _resolve_config(config, config_path, output_dir)

    if caption:
        config_obj.captioning.enabled = True

    validate_config(config_obj)

    jobs = resolve_ocr_inputs(input_path)

    captioner = _build_captioner(config_obj)
    orchestrator = OCROrchestrator(config_obj, captioner=captioner)
    raw_results = orchestrator.run(jobs)

    return [_to_public_result(r) for r in raw_results]