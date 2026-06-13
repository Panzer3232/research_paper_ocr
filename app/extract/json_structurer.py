from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.exceptions import ExtractionError


_CONTENT_TYPES: frozenset[str] = frozenset({
    "text",
    "image",
    "table",
    "equation",
    "code",    # algorithms and code blocks (sub_type: "algorithm" | "code")
    "chart",   # data-visualization figures (has img_path + chart_caption)
    "list",    # reference lists and enumerated items (sub_type: "ref_text" | "ol" | "ul")
})
_DISCARD_TYPES: frozenset[str] = frozenset({
    "discarded",
    # Page auxiliary block types — official MinerU docs, content_list.json spec:
    "header",        # running page headers (journal name, section title in margin)
    "footer",        # running page footers
    "page_number",   # literal page numbers
    "aside_text",    # margin notes, arXiv watermarks, rotated sidebar text
    "page_footnote", # per-page footnotes (author affiliations, copyright lines)
})

# re.DOTALL so multiline GPT captions (containing \n) are captured in full.
# .*? instead of [^\]]* so ] characters inside GPT captions do not break the match.
_MD_IMAGE_RE = re.compile(
    r"!\[(.*?)\]\(([^)]+\.(?:png|jpg|jpeg|gif|webp))\)",
    re.IGNORECASE | re.DOTALL,
)

# Matches a "Figure N:" line that optionally follows an image tag in the markdown.
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*\n?Figure\s+\d+[.:][^\n]+",
    re.IGNORECASE,
)

# Matches MinerU post-element text captions emitted as a plain text block.
_FLOAT_CAPTION_RE = re.compile(
    r"^(Table|Figure|Algorithm|Listing|Appendix)\s+\d+[\.:].{0,400}$",
    re.IGNORECASE | re.DOTALL,
)

# Matches markdown section headings at any level.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


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

    annotated = _annotate_blocks(raw)
    sections = _group_into_sections(annotated)
    total_pages = _infer_total_pages(raw)

    # Extract LLM captions from the captioned markdown (alt-text written by your captioner).
    llm_captions: dict[str, str] = {}
    if captioned_markdown_path is not None:
        llm_captions = _extract_llm_captions(captioned_markdown_path)

    # MinerU sometimes omits image blocks from content_list.json entirely while
    # still writing the image tags into the markdown. Recover these missing images
    # by parsing the markdown and injecting them into the matching sections.
    # This uses the captioned markdown when available so that caption_llm is
    # populated in the same pass; falls back to the plain markdown path derived
    # from the content_list location when captioned markdown is not provided.
    md_path = captioned_markdown_path
    if md_path is not None:
        _inject_markdown_images(sections, md_path, llm_captions)
    
    # Apply LLM captions to image blocks that already came from content_list.json.
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


# ---------------------------------------------------------------------------
# Markdown image extraction helpers
# ---------------------------------------------------------------------------

def _parse_markdown_sections(md_text: str) -> dict[str, list[dict[str, Any]]]:
    """
    Parse the markdown into sections using heading boundaries and return a dict
    mapping section_title -> list of image dicts found within that section.

    Each image dict has keys: img_path, caption (figure label if present),
    caption_llm (alt-text written by captioner, empty string if none).
    """
    headings = list(_MD_HEADING_RE.finditer(md_text))
    result: dict[str, list[dict[str, Any]]] = {}

    for i, h in enumerate(headings):
        title = h.group(2).strip()
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(md_text)
        section_text = md_text[start:end]

        images: list[dict[str, Any]] = []
        for m in _MD_IMAGE_RE.finditer(section_text):
            alt = m.group(1).strip()
            img_path = m.group(2).strip().replace("\\", "/")

            # Look for a "Figure N:" line immediately after the image tag.
            after = section_text[m.end():m.end() + 400]
            fig_match = _FIGURE_CAPTION_RE.match(after)
            fig_caption = fig_match.group().strip() if fig_match else None

            images.append({
                "img_path": img_path,
                "caption": fig_caption,
                "caption_llm": alt if alt else None,
            })

        if images:
            result[title] = images

    return result


def _inject_markdown_images(
    sections: list[dict[str, Any]],
    markdown_path: str | Path,
    llm_captions: dict[str, str],
) -> None:
    """
    For each section that has no image blocks, check the markdown for images
    belonging to that section and append them. Mutates sections in-place.

    Already-present image img_paths are tracked to avoid duplicates.
    """
    md_path = Path(markdown_path)
    if not md_path.exists():
        return

    md_text = md_path.read_text(encoding="utf-8")
    md_sections = _parse_markdown_sections(md_text)

    if not md_sections:
        return

    for section in sections:
        title = section["section_title"]
        md_images = md_sections.get(title)
        if not md_images:
            continue

        # Collect img_paths already present in this section from content_list.json.
        existing_paths: set[str] = {
            (b.get("img_path") or "").replace("\\", "/")
            for b in section["content"]
            if b.get("type") == "image"
        }

        for img in md_images:
            img_path = img["img_path"]
            if img_path in existing_paths:
                continue

            block: dict[str, Any] = {
                "type": "image",
                "img_path": img_path,
                "caption": img["caption"],
                "page_idx": None,
            }

            # caption_llm: use llm_captions dict (keyed by img_path) as the
            # authoritative source; fall back to alt-text parsed from the markdown.
            llm_cap = llm_captions.get(img_path) or img.get("caption_llm")
            if llm_cap:
                block["caption_llm"] = llm_cap

            section["content"].append(block)
            existing_paths.add(img_path)


# ---------------------------------------------------------------------------
# Block annotation helpers (content_list.json path)
# ---------------------------------------------------------------------------

def _join_string_list(items: list[Any]) -> str:
    return " ".join(str(c).strip() for c in items if isinstance(c, str)).strip()


def _annotate_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Shallow-copy all blocks and attach resolved caption / fallback fields before
    grouping into sections.

    Caption resolution order for tables:
      1. table_caption list embedded in the block.
      2. table_footnote list embedded in the block.
      3. Immediately preceding text block matching _FLOAT_CAPTION_RE (lookbehind).
         Handles the case where MinerU places the caption text between a preceding
         image and the table (e.g. "Table 6: ...").
      4. First following text block matching _FLOAT_CAPTION_RE (lookahead).
         Consumed so it is not also emitted as a body text block.

    Caption resolution order for images:
      1. image_caption list embedded in the block.
      2. image_footnote list embedded in the block.

    Caption resolution for charts:
      1. chart_caption list embedded in the block.
      2. chart_footnote list embedded in the block.

    Caption resolution for code/algorithm blocks:
      1. code_caption list embedded in the block.
      2. code_footnote list embedded in the block.

    Additional annotations:
      _img_path    — on table, chart, and equation blocks (fallback rendered image).
      _text_format — on equation blocks (e.g. "latex").
      _sub_type    — on code and list blocks (e.g. "algorithm", "ref_text").
    """
    result: list[dict[str, Any]] = [dict(b) for b in blocks]
    consumed: set[int] = set()

    for i, block in enumerate(result):
        block_type = block.get("type", "")

        if block_type == "table":
            cap = _join_string_list(block.get("table_caption") or [])
            fn = _join_string_list(block.get("table_footnote") or [])
            resolved = cap or fn

            if not resolved:
                resolved, consumed_idx = _lookbehind_caption(result, i, consumed)
                if consumed_idx is not None:
                    consumed.add(consumed_idx)
            if not resolved:
                resolved, consumed_idx = _lookahead_caption(result, i, consumed)
                if consumed_idx is not None:
                    consumed.add(consumed_idx)
            block["_resolved_caption"] = resolved
            block["_img_path"] = block.get("img_path") or None

        elif block_type == "image":
            cap = _join_string_list(block.get("image_caption") or [])
            fn = _join_string_list(block.get("image_footnote") or [])
            block["_resolved_caption"] = cap or fn

        elif block_type == "chart":
            cap = _join_string_list(block.get("chart_caption") or [])
            fn = _join_string_list(block.get("chart_footnote") or [])
            block["_resolved_caption"] = cap or fn
            block["_img_path"] = block.get("img_path") or None

        elif block_type == "code":
            cap = _join_string_list(block.get("code_caption") or [])
            fn = _join_string_list(block.get("code_footnote") or [])
            block["_resolved_caption"] = cap or fn
            block["_sub_type"] = block.get("sub_type") or None

        elif block_type == "list":
            block["_sub_type"] = block.get("sub_type") or None

        elif block_type == "equation":
            block["_img_path"] = block.get("img_path") or None
            block["_text_format"] = block.get("text_format") or None

    for idx in consumed:
        result[idx]["_consumed_as_caption"] = True

    return result


def _lookbehind_caption(
    blocks: list[dict[str, Any]],
    element_idx: int,
    already_consumed: set[int],
) -> tuple[str, int | None]:
    """
    Scan backward from element_idx skipping discarded blocks and return the
    (text, index) of the immediately preceding text block if it matches
    _FLOAT_CAPTION_RE, has no text_level key, and has not already been consumed.

    Only inspects one non-discarded text block — if the first non-discarded block
    before the element is not a matching caption, returns ("", None).
    This prevents over-reaching into unrelated body text.
    """
    for j in range(element_idx - 1, -1, -1):
        if j in already_consumed:
            continue
        candidate = blocks[j]
        ctype = candidate.get("type", "")
        if ctype in _DISCARD_TYPES:
            continue
        if ctype != "text":
            break
        text = (candidate.get("text") or "").strip()
        if _FLOAT_CAPTION_RE.match(text) and "text_level" not in candidate:
            return text, j
        break
    return "", None


def _lookahead_caption(
    blocks: list[dict[str, Any]],
    element_idx: int,
    already_consumed: set[int],
) -> tuple[str, int | None]:
    """
    Scan forward skipping discarded blocks and return the (text, index) of the
    first text block matching _FLOAT_CAPTION_RE that has no text_level key and
    has not already been consumed. Returns ("", None) if not found within two
    non-discarded text blocks.
    """
    checked = 0
    for j in range(element_idx + 1, len(blocks)):
        if j in already_consumed:
            continue
        candidate = blocks[j]
        ctype = candidate.get("type", "")
        if ctype in _DISCARD_TYPES:
            continue
        if ctype != "text":
            break
        checked += 1
        text = (candidate.get("text") or "").strip()
        if _FLOAT_CAPTION_RE.match(text) and "text_level" not in candidate:
            return text, j
        if checked >= 2:
            break
    return "", None


# ---------------------------------------------------------------------------
# LLM caption helpers (captioned markdown path)
# ---------------------------------------------------------------------------

def _extract_llm_captions(captioned_markdown_path: str | Path) -> dict[str, str]:
    """
    Return img_path -> llm_caption for every image tag with non-empty alt-text
    in the captioned markdown file written by your GPT captioner.
    """
    md_path = Path(captioned_markdown_path)
    if not md_path.exists():
        return {}

    captions: dict[str, str] = {}
    content = md_path.read_text(encoding="utf-8")

    for match in _MD_IMAGE_RE.finditer(content):
        alt_text = match.group(1).strip()
        img_path = match.group(2).strip().replace("\\", "/")
        if alt_text and img_path:
            captions[img_path] = alt_text

    return captions


def _apply_llm_captions(
    sections: list[dict[str, Any]],
    llm_captions: dict[str, str],
) -> None:
    """
    Attach caption_llm to image blocks that came from content_list.json and
    did not go through _inject_markdown_images. Mutates sections in-place.
    """
    for section in sections:
        for block in section.get("content", []):
            if block.get("type") != "image":
                continue
            if "caption_llm" in block:
                continue
            img_path = (block.get("img_path") or "").replace("\\", "/")
            if not img_path:
                continue
            llm_caption = llm_captions.get(img_path)
            if llm_caption:
                block["caption_llm"] = llm_caption


# ---------------------------------------------------------------------------
# Section grouping
# ---------------------------------------------------------------------------

def _group_into_sections(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None

    for block in blocks:
        block_type = block.get("type", "")

        if block_type in _DISCARD_TYPES:
            continue

        if block.get("_consumed_as_caption"):
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
        caption_text = block.get("_resolved_caption") or ""
        return {
            "type": "image",
            "img_path": block.get("img_path"),
            "caption": caption_text or None,
            "page_idx": block.get("page_idx"),
        }

    if block_type == "table":
        caption_text = block.get("_resolved_caption") or ""
        table_body = (block.get("table_body") or "").strip()
        result: dict[str, Any] = {
            "type": "table",
            "caption": caption_text or None,
            "table_body": table_body or None,
            "page_idx": block.get("page_idx"),
        }
        if block.get("_img_path"):
            result["img_path"] = block["_img_path"]
        return result

    if block_type == "equation":
        text = (block.get("text") or "").strip()
        result = {
            "type": "equation",
            "text": text or None,
            "page_idx": block.get("page_idx"),
        }
        if block.get("_text_format"):
            result["text_format"] = block["_text_format"]
        if block.get("_img_path"):
            result["img_path"] = block["_img_path"]
        return result

    if block_type == "chart":
        caption_text = block.get("_resolved_caption") or ""
        result = {
            "type": "image",          # charts are visual figures; treated as image for RAG
            "img_path": block.get("_img_path") or block.get("img_path"),
            "caption": caption_text or None,
            "page_idx": block.get("page_idx"),
        }
        return result

    if block_type == "code":
        body = (block.get("code_body") or "").strip()
        if not body:
            return None
        result = {
            "type": "algorithm",
            "code_body": body,
            "page_idx": block.get("page_idx"),
        }
        caption_text = block.get("_resolved_caption") or ""
        if caption_text:
            result["caption"] = caption_text
        if block.get("_sub_type"):
            result["sub_type"] = block["_sub_type"]
        return result

    if block_type == "list":
        items: list[str] = [
            str(it).strip() for it in (block.get("list_items") or [])
            if str(it).strip()
        ]
        if not items:
            return None
        result = {
            "type": "list",
            "items": items,
            "page_idx": block.get("page_idx"),
        }
        if block.get("_sub_type"):
            result["sub_type"] = block["_sub_type"]
        return result

    return None



def _infer_total_pages(blocks: list[dict[str, Any]]) -> int:
    pages = [b.get("page_idx", 0) for b in blocks if isinstance(b.get("page_idx"), int)]
    return (max(pages) + 1) if pages else 0