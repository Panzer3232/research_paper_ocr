from __future__ import annotations

import re
from typing import Any

from .common import normalize_text

EQUATION_INTRO_RE = re.compile(
    r"\b(is|are|as|given by|defined as|computed as|calculated as|objective is|loss is|formally|written as|represented as|we define|we obtain|is then)\s*[:：]?\s*$",
    re.IGNORECASE,
)
EQUATION_EXPLAIN_RE = re.compile(
    r"^\s*(where|which|then|therefore|here|in which|with|the resulting|this|we use|we set|we define)\b",
    re.IGNORECASE,
)
TABLE_REF_RE = re.compile(r"\bTable\s*\d+[A-Za-z]?\b", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\b(Figure|Fig\.)\s*\d+[A-Za-z]?\b", re.IGNORECASE)


def nearby_text_blocks(
    content: list[dict[str, Any]],
    index: int,
    direction: int,
    max_text_blocks: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    cursor = index + direction
    while 0 <= cursor < len(content) and len(result) < max_text_blocks:
        block = content[cursor]
        if block.get("type") == "text" and normalize_text(block.get("text")):
            result.append(block)
        cursor += direction
    if direction < 0:
        result.reverse()
    return result


def select_equation_context(
    content: list[dict[str, Any]],
    equation_index: int,
    max_context_blocks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    previous = nearby_text_blocks(content, equation_index, -1, max_context_blocks)
    following = nearby_text_blocks(content, equation_index, 1, max_context_blocks)

    previous_cues = [bool(EQUATION_INTRO_RE.search(normalize_text(b.get("text")))) for b in previous]
    following_cues = [bool(EQUATION_EXPLAIN_RE.search(normalize_text(b.get("text")))) for b in following]

    flags = {
        "previous_intro_cue": any(previous_cues),
        "following_explanation_cue": any(following_cues),
        "one_sided_context_is_acceptable": any(previous_cues) or any(following_cues),
    }
    return previous, following, flags


def find_table_reference_text(
    content: list[dict[str, Any]],
    table_index: int,
    window: int = 2,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for idx in range(max(0, table_index - window), min(len(content), table_index + window + 1)):
        if idx == table_index:
            continue
        block = content[idx]
        if block.get("type") == "text" and TABLE_REF_RE.search(normalize_text(block.get("text"))):
            refs.append(block)
    return refs


def find_figure_reference_text(
    content: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    window: int = 2,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    left = max(0, start_index - window)
    right = min(len(content), end_index + window + 1)
    for idx in range(left, right):
        if start_index <= idx <= end_index:
            continue
        block = content[idx]
        text = normalize_text(block.get("text")) if block.get("type") == "text" else ""
        if text and FIGURE_REF_RE.search(text):
            refs.append(block)
    return refs


def recover_table_caption(
    table_block: dict[str, Any],
    reference_blocks: list[dict[str, Any]],
    section_title: str,
) -> tuple[str, str]:
    explicit = normalize_text(table_block.get("caption"))
    if explicit:
        return explicit, "explicit"

    for block in reference_blocks:
        text = normalize_text(block.get("text"))
        if TABLE_REF_RE.search(text):
            return text, "recovered_from_reference_text"

    return f"Table from section: {section_title}", "inferred_from_section"


def recover_figure_caption(
    image_blocks: list[dict[str, Any]],
    reference_blocks: list[dict[str, Any]],
    section_title: str,
) -> tuple[str, str]:
    captions = [normalize_text(b.get("caption")) for b in image_blocks if normalize_text(b.get("caption"))]
    if captions:
        return "\n".join(dict.fromkeys(captions)), "explicit"

    for block in reference_blocks:
        text = normalize_text(block.get("text"))
        if FIGURE_REF_RE.search(text):
            return text, "recovered_from_reference_text"

    return f"Figure group from section: {section_title}", "inferred_from_section"
