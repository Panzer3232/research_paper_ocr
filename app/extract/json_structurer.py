from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.exceptions import ExtractionError


_CONTENT_TYPES: frozenset[str] = frozenset({"text", "image", "table", "equation"})
_DISCARD_TYPES: frozenset[str] = frozenset({"discarded"})

# Matches markdown image tags: ![alt text](path/to/image.ext)
_MD_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)]+\.(?:png|jpg|jpeg|gif|webp))\)",
    re.IGNORECASE,
)


def build_structured_json(
    content_list_path: str | Path,
    *,
    paper_key: str,
    source_pdf: str,
    captioned_markdown_path: str | Path | None = None,
) -> dict[str, Any]:
   
    path = Path(content_list_path)
    if not path.exists():
        raise ExtractionError(
            f"content_list.json not found for structured JSON export: {path}"
        )

    try:
        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ExtractionError(
            f"Failed to parse content_list.json at {path}: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise ExtractionError(
            f"content_list.json must be a JSON array, got {type(raw).__name__}: {path}"
        )

    sections = _group_into_sections(raw)
    total_pages = _infer_total_pages(raw)

    if captioned_markdown_path is not None:
        llm_captions = _extract_llm_captions(captioned_markdown_path)
        if llm_captions:
            _apply_llm_captions(sections, llm_captions)

    return {
        "metadata": {
            "paper_key": paper_key,
            "source_pdf": source_pdf,
            "total_pages": total_pages,
            "total_sections": len(sections),
            "llm_captioned": captioned_markdown_path is not None,
        },
        "sections": sections,
    }


def _extract_llm_captions(captioned_markdown_path: str | Path) -> dict[str, str]:
    """
    Parse the captioned markdown file and return a mapping of
    ``img_path → llm_caption`` for every image that has a non-empty alt-text.

    The captioner writes ``![GPT caption](relative/path/to/image.jpg)`` so the
    alt-text is the LLM-generated caption.
    """
    md_path = Path(captioned_markdown_path)
    if not md_path.exists():
        return {}

    captions: dict[str, str] = {}
    content = md_path.read_text(encoding="utf-8")

    for match in _MD_IMAGE_RE.finditer(content):
        alt_text = match.group(1).strip()
        img_path = match.group(2).strip()
        if alt_text and img_path:
            # Normalise path separators so keys match regardless of OS
            captions[img_path.replace("\\", "/")] = alt_text

    return captions


def _apply_llm_captions(
    sections: list[dict[str, Any]],
    llm_captions: dict[str, str],
) -> None:
    """
    Walk all image blocks in the section list and attach ``caption_llm``
    where a matching LLM caption exists.  Mutates sections in-place.
    """
    for section in sections:
        for block in section.get("content", []):
            if block.get("type") != "image":
                continue
            img_path = (block.get("img_path") or "").replace("\\", "/")
            if not img_path:
                continue
            llm_caption = llm_captions.get(img_path)
            if llm_caption:
                block["caption_llm"] = llm_caption


def _group_into_sections(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None

    for block in blocks:
        block_type = block.get("type", "")

        if block_type in _DISCARD_TYPES:
            continue

        if _is_heading(block):
            if current_section is not None:
                sections.append(current_section)
            current_section = _new_section(
                title=block.get("text", "").strip(),
                level=int(block.get("text_level", 1)),
                page_start=block.get("page_idx", 0),
            )
            continue

        if block_type not in _CONTENT_TYPES:
            continue

        if current_section is None:
            current_section = _new_section(
                title="[Preamble]",
                level=0,
                page_start=block.get("page_idx", 0),
            )

        content_block = _normalise_block(block)
        if content_block is not None:
            current_section["content"].append(content_block)
            current_section["page_end"] = block.get("page_idx", current_section["page_end"])

    if current_section is not None:
        sections.append(current_section)

    return sections


def _is_heading(block: dict[str, Any]) -> bool:
    return block.get("type") == "text" and "text_level" in block


def _new_section(*, title: str, level: int, page_start: int) -> dict[str, Any]:
    return {
        "section_title": title,
        "section_level": level,
        "page_start": page_start,
        "page_end": page_start,
        "content": [],
    }


def _normalise_block(block: dict[str, Any]) -> dict[str, Any] | None:
    block_type = block.get("type", "")

    if block_type == "text":
        text = (block.get("text") or "").strip()
        if not text:
            return None
        return {
            "type": "text",
            "text": text,
            "page_idx": block.get("page_idx"),
        }

    if block_type == "image":
        captions = block.get("image_caption") or []
        caption_text = " ".join(c.strip() for c in captions if isinstance(c, str)).strip()
        return {
            "type": "image",
            "img_path": block.get("img_path"),
            "caption": caption_text or None,
            "page_idx": block.get("page_idx"),
        }

    if block_type == "table":
        captions = block.get("table_caption") or []
        caption_text = " ".join(c.strip() for c in captions if isinstance(c, str)).strip()
        table_body = (block.get("table_body") or "").strip()
        return {
            "type": "table",
            "caption": caption_text or None,
            "table_body": table_body or None,
            "page_idx": block.get("page_idx"),
        }

    if block_type == "equation":
        return {
            "type": "equation",
            "text": (block.get("text") or "").strip() or None,
            "page_idx": block.get("page_idx"),
        }

    return None


def _infer_total_pages(blocks: list[dict[str, Any]]) -> int:
    pages = [b.get("page_idx", 0) for b in blocks if isinstance(b.get("page_idx"), int)]
    return (max(pages) + 1) if pages else 0