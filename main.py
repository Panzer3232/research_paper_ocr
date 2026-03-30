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

_BUNDLED_CONFIG = Path(__file__).parent / "config.json"


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
            "captioning enabled but OPENAI_API_KEY / AZURE_OPENAI_API_KEY is not set — skipping captioning."
        )
        return None

    if not (config.apis.openai_base_url or "").strip():
        logging.getLogger("paper_ocr").warning(
            "captioning enabled but OPENAI_BASE_URL / AZURE_OPENAI_ENDPOINT is not set — skipping captioning."
        )
        return None

    paths = PathResolver(config.output)
    return MarkdownCaptioner(config.apis, config.captioning, paths)


def run(
    input_path: str | Path,
    *,
    config_path: str | Path | None = None,
    caption: bool = False,
) -> int:
    
    _load_dotenv()

    if config_path is None:
        cwd_config = Path.cwd() / "config.json"
        config_path = cwd_config if cwd_config.exists() else _BUNDLED_CONFIG

    config = load_config(config_path)

    if caption:
        config.captioning.enabled = True

    validate_config(config)
    _setup_logging(config.logging.level, config.logging.log_file)

    logger = logging.getLogger("paper_ocr")

    jobs = resolve_ocr_inputs(input_path)
    logger.info(
        "paper_ocr starting | pdfs=%d | captioning=%s",
        len(jobs),
        config.captioning.enabled,
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
        default=None,
        help=(
            "Path to config.json. Defaults to config.json in the current directory, "
            "falling back to the bundled config."
        ),
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        default=False,
        help=(
            "Enable GPT-5 image captioning. Overrides config.captioning.enabled. "
            "Requires OPENAI_API_KEY and OPENAI_BASE_URL in .env or environment."
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
        )
    )


if __name__ == "__main__":
    main()