from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_PACKAGE_ROOT = Path(__file__).parent.resolve()
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from app.config.loader import load_config
from app.config.models import PipelineConfig
from app.config.validator import validate_config
from app.pipeline.input_resolver import resolve_ocr_inputs
from app.pipeline.orchestrator import OCROrchestrator, OCRResult

logging.getLogger("paper_ocr").addHandler(logging.NullHandler())
logging.getLogger("rag_chunking").addHandler(logging.NullHandler())

_BUNDLED_CONFIG = Path(__file__).parent / "config.json"

def enable_logging(level: str = "INFO", log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    for name in ("paper_ocr", "rag_chunking", "app.extract.captioner", "httpx"):
        lg = logging.getLogger(name)
        lg.setLevel(getattr(logging, level.upper(), logging.INFO))
        lg.handlers = []
        for h in handlers:
            h.setFormatter(logging.Formatter(fmt))
            lg.addHandler(h)
        lg.propagate = False



@dataclass(frozen=True)
class OCRPipelineResult:
    pdf_path: str
    success: bool
    markdown_path: str | None
    captioned_path: str | None
    structured_json_path: str | None
    error: str | None
    status: str
    label: str

    def effective_markdown(self) -> str | None:
        return self.captioned_path or self.markdown_path



@dataclass(frozen=True)
class RAGChunkResult:
    """Per-paper result returned by chunk()."""

    paper_key: str
    source_json_path: str
    success: bool
    normalized_json_path: str | None
    chunk_json_path: str | None
    # None when evaluate=False or when a prior stage failed
    evaluation_report_path: str | None
    error: str | None
    # "completed" | "failed_prepare" | "failed_chunk" | "failed_evaluate"
    status: str



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


def _to_public_ocr_result(r: OCRResult) -> OCRPipelineResult:
    return OCRPipelineResult(
        pdf_path=r.pdf_path,
        success=r.success,
        markdown_path=r.markdown_path,
        captioned_path=r.captioned_path,
        structured_json_path=r.structured_json_path,
        error=r.error,
        status=r.status,
        label=r.label,
    )



def _load_raw_config(config_path: str | Path | None) -> dict[str, Any]:
    """
    Load config.json as a plain dict.  Does not go through PipelineConfig so
    that we can read the rag_chunking block without touching OCR models.
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        cwd_config = Path.cwd() / "config.json"
        path = cwd_config if cwd_config.exists() else _BUNDLED_CONFIG

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_chunk_input(
    structured_json_dir: str | Path | None,
    raw_config: dict[str, Any],
) -> Path:
    """
    Resolve the structured JSON input directory.

    Resolution order:
      1. Explicit structured_json_dir argument.
      2. config.json → rag_chunking.structured_json_input_dir.
      3. ValueError — user must supply the path.
    """
    if structured_json_dir is not None:
        path = Path(structured_json_dir).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"structured_json_dir does not exist: {path}"
            )
        return path

    config_value = (raw_config.get("rag_chunking") or {}).get(
        "structured_json_input_dir"
    )
    if config_value:
        path = Path(config_value).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"rag_chunking.structured_json_input_dir from config does not exist: {path}"
            )
        return path

    raise ValueError(
        "No structured JSON input directory provided. "
        "Supply it via the structured_json_dir argument or set "
        "rag_chunking.structured_json_input_dir in config.json."
    )


def _resolve_chunk_output(
    output_dir: str | Path | None,
    raw_config: dict[str, Any],
) -> Path:
    """
    Resolve the rag_chunks output root.

    Resolution order:
      1. Explicit output_dir argument → <output_dir>/rag_chunks/
      2. config.json → output.root_dir  → <root_dir>/rag_chunks/
      3. Falls back to data_ocr/rag_chunks/ relative to cwd.
    """
    if output_dir is not None:
        return Path(output_dir).resolve() / "rag_chunks"

    ocr_root = (raw_config.get("output") or {}).get("root_dir")
    if ocr_root:
        return Path(ocr_root).resolve() / "rag_chunks"

    return Path.cwd() / "data_ocr" / "rag_chunks"


def _build_chunk_config(raw_config: dict[str, Any]) -> Any:
    """Build a ChunkConfig from the rag_chunking block in config.json."""
    from rag_chunking.common import ChunkConfig

    rc = raw_config.get("rag_chunking") or {}
    return ChunkConfig(
        chunk_size=int(rc.get("chunk_size", 640)),
        overlap=int(rc.get("overlap", 64)),
        table_chunk_size=int(rc.get("table_chunk_size", 768)),
        min_prose_tokens=int(rc.get("min_prose_tokens", 80)),
        max_context_blocks=int(rc.get("max_context_blocks", 2)),
        split_long_tables=bool(rc.get("split_long_tables", True)),
    )


def _to_public_rag_result(r: Any) -> RAGChunkResult:
    """Convert internal RAGChunkResult (from orchestrator) to the public type."""
    return RAGChunkResult(
        paper_key=r.paper_key,
        source_json_path=r.source_json_path,
        success=r.success,
        normalized_json_path=r.normalized_json_path,
        chunk_json_path=r.chunk_json_path,
        evaluation_report_path=r.evaluation_report_path,
        error=r.error,
        status=r.status,
    )



def ocr(
    input_path: str | Path,
    *,
    caption: bool = False,
    export_json: bool = False,
    config: PipelineConfig | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[OCRPipelineResult]:
    """
    Run MinerU OCR on a PDF or a folder of PDFs.

    Parameters
    ----------
    input_path:
        Path to a single PDF file or a directory containing PDF files.
    caption:
        When True, run GPT captioning on every figure after OCR.
    export_json:
        When True, export a structured RAG-ready JSON file per paper.
    config:
        Pre-built PipelineConfig.  When supplied, config_path is ignored.
    config_path:
        Path to a config.json file.  Defaults to config.json in the working
        directory, then to the bundled default.
    output_dir:
        Override the output root directory (config.output.root_dir).

    Returns
    -------
    list[OCRPipelineResult]
        One result per PDF, regardless of success or failure.
    """
    _load_dotenv()

    config_obj = _resolve_config(config, config_path, output_dir)

    if caption:
        config_obj.captioning.enabled = True

    if export_json:
        config_obj.structured_json.enabled = True

    validate_config(config_obj)

    jobs = resolve_ocr_inputs(input_path)

    captioner = _build_captioner(config_obj)
    orchestrator = OCROrchestrator(config_obj, captioner=captioner)
    raw_results = orchestrator.run(jobs)

    return [_to_public_ocr_result(r) for r in raw_results]


def chunk(
    structured_json_dir: str | Path | None = None,
    *,
    evaluate: bool = False,
    chunk_config: Any | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[RAGChunkResult]:
    """
    Run the RAG chunking pipeline on a folder of structured JSON files
    produced by the OCR pipeline (export_json=True).

    Input resolution (first wins):
      1. structured_json_dir argument.
      2. rag_chunking.structured_json_input_dir in config.json.
      3. ValueError is raised — the path must be supplied one of these ways.

    Output is written to:
      <output_dir>/rag_chunks/       (if output_dir is given)
      <config.output.root_dir>/rag_chunks/   (if set in config.json)
      data_ocr/rag_chunks/           (fallback)

    Sub-directories created automatically:
      normalized/    — prepare step output (one JSON per paper)
      chunks/        — chunk step output   (one JSON per paper)
      evaluation/    — evaluation output   (only when evaluate=True)
      logs/          — one log file per stage, always written

    Parameters
    ----------
    structured_json_dir:
        Path to the folder containing structured JSON files from the OCR
        pipeline.  Overrides config.json.
    evaluate:
        When True, run the quality evaluation step after chunking and write
        per-paper evaluation JSON and a summary report.  Off by default.
    chunk_config:
        A ChunkConfig instance.  When None, values are read from the
        rag_chunking block in config.json, with built-in defaults applied
        for any missing keys.
    config_path:
        Path to a config.json file.  Defaults to config.json in the working
        directory, then to the bundled default.
    output_dir:
        Override the output root.  rag_chunks/ is appended automatically.

    Returns
    -------
    list[RAGChunkResult]
        One result per JSON file, regardless of success or failure.
    """
    from rag_chunking.common import iter_json_files
    from rag_chunking.orchestrator import RAGChunkingOrchestrator

    raw_config = _load_raw_config(config_path)

    input_path = _resolve_chunk_input(structured_json_dir, raw_config)
    rag_output_root = _resolve_chunk_output(output_dir, raw_config)

    resolved_config = chunk_config if chunk_config is not None else _build_chunk_config(raw_config)

    input_files = iter_json_files(input_path)
    if not input_files:
        raise ValueError(
            f"No JSON files found in structured_json_dir: {input_path}"
        )

    logger = logging.getLogger("rag_chunking")
    logger.info(
        "chunk() called | input=%s | files=%d | output=%s | evaluate=%s",
        input_path,
        len(input_files),
        rag_output_root,
        evaluate,
    )

    orchestrator = RAGChunkingOrchestrator(
        rag_chunks_root=rag_output_root,
        config=resolved_config,
        evaluate=evaluate,
    )
    raw_results = orchestrator.run(input_files)

    return [_to_public_rag_result(r) for r in raw_results]