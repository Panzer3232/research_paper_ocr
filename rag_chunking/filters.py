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

        if btype in ("text", "page_footnote"):
            # page_footnote is treated identically to text here, and downstream in
            text, trimmed_checklist_tail = strip_checklist_tail(block.get("text"))
            if trimmed_checklist_tail:
                drops["trimmed_checklist_tail"] = drops.get("trimmed_checklist_tail", 0) + 1
            if is_checklist_text(text):
                drops["checklist_text"] = drops.get("checklist_text", 0) + 1
                continue
            if drop_invalid_blocks and is_noise_text(text):
                noise_key = "noise_text" if btype == "text" else "noise_page_footnote"
                drops[noise_key] = drops.get(noise_key, 0) + 1
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
            block["footnote"] = normalize_text(block.get("footnote")) or None

        elif btype == "image":
            block["caption"] = normalize_text(block.get("caption")) or None
            block["caption_llm"] = normalize_text(block.get("caption_llm")) or None
            block["footnote"] = normalize_text(block.get("footnote")) or None
            if drop_invalid_blocks and not block.get("caption") and not block.get("caption_llm") and not block.get("img_path"):
                drops["empty_image"] = drops.get("empty_image", 0) + 1
                continue

        elif btype in ("algorithm", "code"):
            code_body = (block.get("code_body") or "").strip()
            if drop_invalid_blocks and not code_body:
                drops["empty_algorithm"] = drops.get("empty_algorithm", 0) + 1
                continue
            block["code_body"] = code_body
            block["caption"] = normalize_text(block.get("caption")) or None
            block["footnote"] = normalize_text(block.get("footnote")) or None
            # sub_type distinguishes pseudocode ("algorithm") from verbatim code ("code").
            # Preserve it as-is; downstream chunker uses it for chunk text labelling.
            block["sub_type"] = block.get("sub_type") or "algorithm"

        elif btype == "list":
            # Belt-and-suspenders: ref_text lists that leak past section-level dropping
            # (e.g. a references list inside a non-references section title) are discarded.
            if block.get("sub_type") == "ref_text":
                drops["list_ref_text_dropped"] = drops.get("list_ref_text_dropped", 0) + 1
                continue
            raw_items: list[Any] = block.get("items") or []
            clean_items = [normalize_text(str(it)) for it in raw_items if normalize_text(str(it))]
            if drop_invalid_blocks and not clean_items:
                drops["empty_list"] = drops.get("empty_list", 0) + 1
                continue
            block["items"] = clean_items
            block["sub_type"] = block.get("sub_type") or None

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
    paper_title = section_records[0]["section_title"] if section_records else metadata.get("paper_key", "unknown")
    metadata["paper_title"] = metadata.get("paper_title") or paper_title

    prepared_sections: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "file": file_name,
        "paper_key": paper_key,
        "paper_title": metadata["paper_title"],
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

    prepared = {"metadata": metadata, "sections": prepared_sections}
    return prepared, report