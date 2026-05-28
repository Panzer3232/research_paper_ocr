from __future__ import annotations

import re
from typing import Any

from .common import normalize_text, sentence_like, stable_id
from .sectioning import build_section_records


CHECKLIST_PATTERNS = [
    re.compile(r"\bDid you\b", re.IGNORECASE),
    re.compile(r"\bLeft blank\b", re.IGNORECASE),
    re.compile(r"\bnot applicable\b", re.IGNORECASE),
    re.compile(r"\bresponsible nlp checklist\b", re.IGNORECASE),
    re.compile(r"\bsubmission checklist\b", re.IGNORECASE),
]

NOISE_ONLY_RE = re.compile(r"^[\W\d_\s]{1,40}$")
CHECKLIST_TAIL_RE = re.compile(r"(?:\s|^)[✓✗]\s*[A-Z]\d+\.")


def strip_checklist_tail(text: str) -> tuple[str, bool]:
    text = normalize_text(text)
    match = CHECKLIST_TAIL_RE.search(text)
    if not match:
        return text, False
    return text[: match.start()].strip(), True


def is_checklist_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False

    hits = sum(1 for pattern in CHECKLIST_PATTERNS if pattern.search(text))
    return hits >= 1 and ("Did you" in text or hits >= 2)


def is_noise_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True
    if NOISE_ONLY_RE.match(text):
        return True
    if len(text) < 8 and not sentence_like(text):
        return True
    return False


def extract_front_matter(
    paper_key: str,
    section_records: list[dict[str, Any]],
) -> dict[str, Any]:
    for section in section_records:
        if section.get("section_role") != "title_authors":
            continue

        text_blocks: list[dict[str, Any]] = []
        text_values: list[str] = []

        for block_index, block in enumerate(section.get("content") or []):
            if block.get("type") != "text":
                continue

            text = normalize_text(block.get("text"))
            if not text:
                continue

            block_id = stable_id(
                paper_key,
                section["section_id"],
                block_index,
                block.get("type"),
            )

            text_blocks.append(
                {
                    "block_id": block_id,
                    "block_index": block_index,
                    "text": text,
                    "page_idx": block.get("page_idx"),
                }
            )
            text_values.append(text)

        return {
            "title": normalize_text(section.get("section_title")) or None,
            "authors_affiliations_text": " ".join(text_values) or None,
            "source_section_id": section.get("section_id"),
            "source_section_index": section.get("section_index"),
            "source_section_title": section.get("section_title"),
            "source_page_start": section.get("page_start"),
            "source_page_end": section.get("page_end"),
            "source_blocks": text_blocks,
        }

    return {
        "title": None,
        "authors_affiliations_text": None,
        "source_section_id": None,
        "source_section_index": None,
        "source_section_title": None,
        "source_page_start": None,
        "source_page_end": None,
        "source_blocks": [],
    }


def normalize_blocks(
    paper_key: str,
    section: dict[str, Any],
    drop_invalid_blocks: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    drops: dict[str, int] = {}
    normalized_blocks: list[dict[str, Any]] = []

    for block_index, raw_block in enumerate(section.get("content") or []):
        btype = raw_block.get("type")
        block = dict(raw_block)
        block_id = stable_id(paper_key, section["section_id"], block_index, btype)

        block.update(
            {
                "block_id": block_id,
                "block_index": block_index,
                "section_id": section["section_id"],
                "section_index": section["section_index"],
            }
        )

        if btype == "text":
            text, trimmed_checklist_tail = strip_checklist_tail(block.get("text"))

            if trimmed_checklist_tail:
                drops["trimmed_checklist_tail"] = drops.get("trimmed_checklist_tail", 0) + 1

            if is_checklist_text(text):
                drops["checklist_text"] = drops.get("checklist_text", 0) + 1
                continue

            if drop_invalid_blocks and is_noise_text(text):
                drops["noise_text"] = drops.get("noise_text", 0) + 1
                continue

            block["text"] = text

        elif btype == "equation":
            text = (block.get("text") or "").strip()

            if drop_invalid_blocks and not text:
                drops["empty_equation"] = drops.get("empty_equation", 0) + 1
                continue

            block["text"] = text
            block["text_format"] = block.get("text_format") or "latex"

        elif btype == "table":
            table_body = (block.get("table_body") or "").strip()

            if drop_invalid_blocks and not table_body:
                drops["empty_table"] = drops.get("empty_table", 0) + 1
                continue

            block["caption"] = normalize_text(block.get("caption")) or None
            block["table_body"] = table_body

        elif btype == "image":
            block["caption"] = normalize_text(block.get("caption")) or None
            block["caption_llm"] = normalize_text(block.get("caption_llm")) or None

            has_image_content = (
                block.get("caption")
                or block.get("caption_llm")
                or block.get("img_path")
            )

            if drop_invalid_blocks and not has_image_content:
                drops["empty_image"] = drops.get("empty_image", 0) + 1
                continue

        else:
            drops[f"unknown_{btype}"] = drops.get(f"unknown_{btype}", 0) + 1
            continue

        normalized_blocks.append(block)

    return normalized_blocks, drops


def prepare_document(data: dict[str, Any], file_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(data.get("metadata") or {})
    sections = data.get("sections") or []

    paper_key = metadata.get("paper_key") or file_name.rsplit(".", 1)[0]
    metadata["paper_key"] = paper_key
    metadata.setdefault("source_pdf", "unknown")

    section_records = build_section_records(sections)
    front_matter = extract_front_matter(paper_key, section_records)

    paper_title = (
        metadata.get("paper_title")
        or front_matter.get("title")
        or (section_records[0]["section_title"] if section_records else None)
        or metadata.get("paper_key", "unknown")
    )

    metadata["paper_title"] = paper_title
    metadata["front_matter"] = front_matter

    prepared_sections: list[dict[str, Any]] = []

    report: dict[str, Any] = {
        "file": file_name,
        "paper_key": paper_key,
        "paper_title": metadata["paper_title"],
        "front_matter_detected": bool(front_matter.get("authors_affiliations_text")),
        "input_sections": len(sections),
        "kept_sections": 0,
        "dropped_sections": {},
        "block_drops": {},
        "input_blocks": 0,
        "kept_blocks": 0,
    }

    for section in section_records:
        role = section.get("section_role")
        content = section.get("content") or []
        report["input_blocks"] += len(content)

        if role in {"references", "acknowledgements", "checklist"}:
            report["dropped_sections"][role] = report["dropped_sections"].get(role, 0) + 1
            continue

        normalized_blocks, block_drops = normalize_blocks(paper_key, section)

        for key, value in block_drops.items():
            report["block_drops"][key] = report["block_drops"].get(key, 0) + value

        section_out = dict(section)
        section_out["content"] = normalized_blocks

        prepared_sections.append(section_out)
        report["kept_sections"] += 1
        report["kept_blocks"] += len(normalized_blocks)

    prepared = {
        "metadata": metadata,
        "sections": prepared_sections,
    }

    return prepared, report