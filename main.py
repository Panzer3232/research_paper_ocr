"""
paper_ocr — main entry point.

Runs MinerU OCR extraction and optional GPT-5 image captioning on one or
more PDF files. No downloading, no Semantic Scholar, no network calls
(unless captioning is enabled).

CLI usage:
    # Single PDF, no captioning, no JSON export
    python main.py --input /path/to/paper.pdf

    # Folder of PDFs, captioning enabled via flag
    python main.py --input /path/to/pdfs/ --caption

    # Export structured RAG-ready JSON alongside markdown
    python main.py --input /path/to/pdfs/ --export-json

    # Both together
    python main.py --input /path/to/pdfs/ --caption --export-json

    # Custom config
    python main.py --input /path/to/pdfs/ --config config.json --caption

Programmatic usage (library-compatible):
    from pathlib import Path
    from app.config.loader import load_config
    from app.config.validator import validate_config
    from app.extract.captioner import MarkdownCaptioner
    from app.pipeline.input_resolver import resolve_ocr_inputs
    from app.pipeline.orchestrator import OCROrchestrator
    from app.storage.paths import PathResolver

    config = load_config("config.json")
    config.captioning.enabled = True          # toggle at runtime if desired
    config.structured_json.enabled = True     # toggle at runtime if desired
    validate_config(config)

    paths = PathResolver(config.output)
    captioner = MarkdownCaptioner(config.apis, config.captioning, paths)

    orchestrator = OCROrchestrator(config, paths=paths, captioner=captioner)
    jobs = resolve_ocr_inputs("/path/to/pdfs/")
    results = orchestrator.run(jobs)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from app.config.loader import load_config
from app.config.models import PipelineConfig
from app.config.validator import validate_config
from app.pipeline.input_resolver import resolve_ocr_inputs
from app.pipeline.orchestrator import OCROrchestrator

_DEFAULT_CONFIG = Path(__file__).parent / "config.json"


def _setup_logging(level: str, log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_captioner(config: PipelineConfig) -> Any | None:
    """
    Construct a MarkdownCaptioner only when captioning is enabled and
    all required credentials are present. Logs a warning and returns None
    otherwise — the orchestrator treats None as "captioning disabled".
    """
    if not config.captioning.enabled:
        return None

    try:
        from app.extract.captioner import MarkdownCaptioner
        from app.storage.paths import PathResolver
    except ImportError:
        logging.getLogger("paper_ocr").warning(
            "captioner module unavailable — captioning will be skipped."
        )
        return None

    if not (config.apis.openai_api_key or "").strip():
        logging.getLogger("paper_ocr").warning(
            "captioning enabled but apis.openai_api_key is missing — skipping captioning."
        )
        return None

    if not (config.apis.openai_base_url or "").strip():
        logging.getLogger("paper_ocr").warning(
            "captioning enabled but apis.openai_base_url is missing — skipping captioning."
        )
        return None

    paths = PathResolver(config.output)
    return MarkdownCaptioner(config.apis, config.captioning, paths)


def run(
    input_path: str | Path,
    *,
    config_path: str | Path = _DEFAULT_CONFIG,
    caption: bool = False,
    export_json: bool = False,
) -> int:
    """
    Run MinerU extraction (and optionally captioning / structured JSON export)
    on PDFs at ``input_path``.

    Parameters
    ----------
    input_path:
        Path to a single PDF file or a directory of PDF files.
    config_path:
        Path to the pipeline JSON config. Defaults to config.json next to main.py.
    caption:
        When ``True``, forces ``config.captioning.enabled = True`` regardless
        of what the config file says. The API key and endpoint must still be
        configured.
    export_json:
        When ``True``, forces ``config.structured_json.enabled = True``.
        Reads MinerU's ``content_list.json`` and writes a section-grouped,
        RAG-ready JSON file to ``data/structured_json/``.

    Returns
    -------
    int
        0 if every PDF was processed successfully, 1 otherwise.
    """
    config = load_config(config_path)

    if caption:
        config.captioning.enabled = True

    if export_json:
        config.structured_json.enabled = True

    validate_config(config)
    _setup_logging(config.logging.level, config.logging.log_file)

    logger = logging.getLogger("paper_ocr")

    jobs = resolve_ocr_inputs(input_path)
    logger.info(
        "paper_ocr starting | pdfs=%d | captioning=%s | export_json=%s",
        len(jobs),
        config.captioning.enabled,
        config.structured_json.enabled,
    )

    captioner = _build_captioner(config)
    orchestrator = OCROrchestrator(config, captioner=captioner)
    results = orchestrator.run(jobs)

    succeeded = sum(1 for r in results if r.success)
    total = len(results)

    for r in results:
        if not r.success:
            logger.error("FAILED | %s | %s | %s", r.label, r.status, r.error)

    logger.info("paper_ocr finished | %d/%d succeeded", succeeded, total)

    return 0 if succeeded == total else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MinerU OCR extraction and optional GPT-5 captioning "
            "on one or more PDF files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a single PDF file or a directory of PDF files.",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="Path to config.json.",
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        default=False,
        help=(
            "Enable GPT-5 image captioning. Overrides config.captioning.enabled. "
            "Requires apis.openai_api_key and apis.openai_base_url to be set."
        ),
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        default=False,
        dest="export_json",
        help=(
            "Export a structured RAG-ready JSON file alongside the markdown output. "
            "Reads MinerU's content_list.json and writes section-grouped chunks to "
            "data/structured_json/. No network calls required."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sys.exit(
        run(
            args.input,
            config_path=args.config,
            caption=args.caption,
            export_json=args.export_json,
        )
    )


if __name__ == "__main__":
    main()