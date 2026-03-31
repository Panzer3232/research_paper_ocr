from __future__ import annotations

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

_BUNDLED_CONFIG = Path(__file__).parent / "config.json"


def enable_logging(level: str = "INFO", log_file: str | None = None) -> None:
   
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    for name in ("paper_ocr", "app.extract.captioner", "httpx"):
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
    error: str | None
    status: str
    label: str

    def effective_markdown(self) -> str | None:
        """
        Return the best available markdown path for this result.
        """
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