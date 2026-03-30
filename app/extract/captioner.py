from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.models import ApiConfig, CaptioningConfig
from app.core.exceptions import CaptioningError, ConfigurationError
from app.models.manifest import PipelineManifest
from app.models.paper import PaperRecord
from app.storage.paths import PathResolver
from app.storage.writers import write_text

logger = logging.getLogger(__name__)

try:
    from openai import AzureOpenAI as _AzureOpenAI
except ImportError:
    _AzureOpenAI = None

_IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|webp))\)",
    re.IGNORECASE,
)


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _image_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    _map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return _map.get(suffix, "image/jpeg")


@dataclass(slots=True)
class CaptioningSummary:
    captioned_markdown_path: str
    images_found: int
    images_captioned: int
    images_skipped: int
    elapsed_seconds: float
    reused_existing: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "captioned_markdown_path": self.captioned_markdown_path,
            "images_found": self.images_found,
            "images_captioned": self.images_captioned,
            "images_skipped": self.images_skipped,
            "elapsed_seconds": self.elapsed_seconds,
            "reused_existing": self.reused_existing,
        }


class _ImageCaptioner:

    def __init__(self, api_cfg: ApiConfig, captioning_cfg: CaptioningConfig) -> None:
        if _AzureOpenAI is None:
            raise ConfigurationError(
                "The 'openai' package is required for image captioning. "
            )
        if not api_cfg.openai_api_key:
            raise ConfigurationError(
                "apis.openai_api_key is required when captioning is enabled."
            )
        if not api_cfg.openai_base_url:
            raise ConfigurationError(
                "apis.openai_base_url (Azure endpoint) is required when captioning is enabled."
            )

        self._client = _AzureOpenAI(
            api_key=api_cfg.openai_api_key,
            azure_endpoint=api_cfg.openai_base_url,
            api_version=api_cfg.openai_api_version,
        )
        self._model = api_cfg.openai_model
        self._prompt = captioning_cfg.caption_prompt
        self._max_retries = captioning_cfg.max_retries
        self._timeout = captioning_cfg.timeout_seconds

    def caption(self, image_path: Path) -> str | None:
        b64 = _encode_image(image_path)
        media_type = _image_media_type(image_path)

        for attempt in range(1, self._max_retries + 2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self._prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    timeout=self._timeout,
                )
                return response.choices[0].message.content.strip()
            except Exception as exc:
                if attempt <= self._max_retries:
                    logger.warning(
                        "Caption attempt %d/%d failed for %s: %s — retrying",
                        attempt,
                        self._max_retries + 1,
                        image_path.name,
                        exc,
                    )
                else:
                    logger.warning(
                        "Caption generation failed for %s after %d attempts: %s",
                        image_path.name,
                        self._max_retries + 1,
                        exc,
                    )
        return None


class MarkdownCaptioner:

    def __init__(
        self,
        api_cfg: ApiConfig,
        captioning_cfg: CaptioningConfig,
        paths: PathResolver,
    ) -> None:
        self._captioner = _ImageCaptioner(api_cfg, captioning_cfg)
        self._paths = paths

    def __call__(
        self,
        paper: PaperRecord,
        manifest: PipelineManifest,
    ) -> dict[str, Any]:
        markdown_path_str = manifest.output_paths.get("markdown")
        if not markdown_path_str:
            raise CaptioningError(
                f"No markdown path in manifest for {paper.paper_key}; "
                "captioning requires a completed extraction stage."
            )

        markdown_path = Path(markdown_path_str)
        if not markdown_path.exists():
            raise CaptioningError(
                f"Markdown file not found at {markdown_path}; "
                "captioning requires a completed extraction stage."
            )

        output_path = self._paths.captioned_markdown_path(paper.paper_key)

        if output_path.exists():
            logger.debug(
                "Captioned markdown already exists for %s, reusing.", paper.paper_key
            )
            return CaptioningSummary(
                captioned_markdown_path=str(output_path),
                images_found=0,
                images_captioned=0,
                images_skipped=0,
                elapsed_seconds=0.0,
                reused_existing=True,
            ).to_dict()

        content = markdown_path.read_text(encoding="utf-8")
        matches = list(_IMAGE_PATTERN.finditer(content))

        if not matches:
            logger.info(
                "No images found in markdown for %s; writing unchanged copy.",
                paper.paper_key,
            )
            write_text(output_path, content)
            return CaptioningSummary(
                captioned_markdown_path=str(output_path),
                images_found=0,
                images_captioned=0,
                images_skipped=0,
                elapsed_seconds=0.0,
                reused_existing=False,
            ).to_dict()

        image_base_dir = self._resolve_image_base_dir(paper, manifest)
        logger.debug("Resolved image base dir: %s", image_base_dir)

        images_captioned = 0
        images_skipped = 0
        start = time.monotonic()

        for idx, match in enumerate(matches, 1):
            image_rel = match.group(2)
            image_full = image_base_dir / image_rel

            if not image_full.exists():
                logger.warning(
                    "[%d/%d] Image not found, skipping: %s (resolved to %s)",
                    idx,
                    len(matches),
                    image_rel,
                    image_full,
                )
                images_skipped += 1
                continue

            logger.debug("[%d/%d] Captioning: %s", idx, len(matches), image_rel)
            caption = self._captioner.caption(image_full)

            if caption:
                old_tag = match.group(0)
                new_tag = f"![{caption}]({image_rel})"
                content = content.replace(old_tag, new_tag, 1)
                images_captioned += 1
            else:
                images_skipped += 1

        elapsed = time.monotonic() - start
        write_text(output_path, content)

        logger.info(
            "Captioning complete for %s: %d/%d captioned in %.2fs",
            paper.paper_key,
            images_captioned,
            len(matches),
            elapsed,
        )

        return CaptioningSummary(
            captioned_markdown_path=str(output_path),
            images_found=len(matches),
            images_captioned=images_captioned,
            images_skipped=images_skipped,
            elapsed_seconds=round(elapsed, 3),
            reused_existing=False,
        ).to_dict()

    def _resolve_image_base_dir(
        self,
        paper: PaperRecord,
        manifest: PipelineManifest,
    ) -> Path:
       
        mineru_doc_dir_str = (manifest.stats.get("extraction") or {}).get("mineru_dir")
        if mineru_doc_dir_str:
            candidate = Path(mineru_doc_dir_str)
            if candidate.exists():
                return candidate

        mineru_output_root = self._paths.mineru_dir(paper.paper_key)
        for images_dir in sorted(mineru_output_root.rglob("images")):
            if images_dir.is_dir():
                return images_dir.parent

        return mineru_output_root