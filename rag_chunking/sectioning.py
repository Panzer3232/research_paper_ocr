from __future__ import annotations

import re
from typing import Any

from .common import normalize_text, stable_id


NUMERIC_SECTION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)(?:[.)])?\s+(.+?)\s*$"
)

COMPACT_NUMERIC_SECTION_RE = re.compile(
    r"^\s*(\d+)([A-Z][A-Za-z].*)$"
)

APPENDIX_TITLE_RE = re.compile(
    r"^\s*Appendix\s+([A-Z](?:\.\d+)*)(?:[.)])?\s+(.+?)\s*$",
    re.IGNORECASE,
)

APPENDIX_LETTER_SECTION_RE = re.compile(
    r"^\s*([A-Z])(?:[.)])?\s+(.+?)\s*$"
)

APPENDIX_SUBSECTION_RE = re.compile(
    r"^\s*([A-Z](?:\.\d+)+)(?:[.)])?\s+(.+?)\s*$"
)

SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:Appendix\s+)?(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)(?:[.)])?\s+(.+?)\s*$",
    re.IGNORECASE,
)

TOP_LEVEL_UNNUMBERED = {
    "abstract",
    "introduction",
    "related work",
    "method",
    "methodology",
    "experiments",
    "experimental setup",
    "results",
    "analysis",
    "discussion",
    "limitations",
    "ethics statement",
    "conclusion",
    "references",
    "acknowledgements",
    "acknowledgments",
}


def infer_section_number(title: str) -> str | None:
    title = normalize_text(title)

    appendix_title = APPENDIX_TITLE_RE.match(title)
    if appendix_title:
        return appendix_title.group(1).rstrip(".")

    numeric = NUMERIC_SECTION_RE.match(title)
    if numeric:
        return numeric.group(1).rstrip(".")

    compact_numeric = COMPACT_NUMERIC_SECTION_RE.match(title)
    if compact_numeric:
        return compact_numeric.group(1)

    appendix_subsection = APPENDIX_SUBSECTION_RE.match(title)
    if appendix_subsection:
        return appendix_subsection.group(1).rstrip(".")

    appendix_letter = APPENDIX_LETTER_SECTION_RE.match(title)
    if appendix_letter:
        return appendix_letter.group(1).rstrip(".")

    return None


def strip_section_prefix(title: str) -> str:
    title = normalize_text(title)
    match = SECTION_PREFIX_RE.match(title)
    if not match:
        return title
    return normalize_text(match.group(1))


def infer_section_depth(title: str, section_level: int | None) -> int:
    number = infer_section_number(title)
    if number:
        return number.count(".") + 1

    normalized = normalize_text(title).lower()
    if normalized in TOP_LEVEL_UNNUMBERED:
        return 1

    if section_level and section_level > 0:
        return max(1, min(int(section_level), 6))

    return 1


def infer_section_role(title: str, index: int, content: list[dict[str, Any]]) -> str:
    normalized = normalize_text(title).lower()
    normalized_without_prefix = strip_section_prefix(title).lower()

    if normalized_without_prefix == "references" or normalized_without_prefix.startswith("references"):
        return "references"

    if normalized_without_prefix in {"acknowledgements", "acknowledgments"}:
        return "acknowledgements"

    if "did you" in normalized or "left blank" in normalized:
        return "checklist"

    if "checklist" in normalized and (
        "responsible" in normalized
        or "acl" in normalized
        or "nlp" in normalized
    ):
        return "checklist"

    if index == 0:
        text = " ".join(
            normalize_text(block.get("text"))
            for block in content
            if block.get("type") == "text"
        )
        email_like = "@" in text
        shortish = len(text.split()) <= 80
        if email_like or shortish:
            return "title_authors"

    return "content"


def build_section_records(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stack: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []

    for idx, section in enumerate(sections):
        title = normalize_text(section.get("section_title")) or f"Untitled Section {idx + 1}"
        level = section.get("section_level")
        content = section.get("content") or []

        number = infer_section_number(title)
        depth = infer_section_depth(title, level)
        section_id = stable_id(
            "section",
            idx,
            title,
            section.get("page_start"),
            section.get("page_end"),
        )
        role = infer_section_role(title, idx, content)

        while len(stack) >= depth:
            stack.pop()

        stack.append(
            {
                "section_id": section_id,
                "section_title": title,
            }
        )

        copied = dict(section)
        copied.update(
            {
                "section_id": section_id,
                "section_index": idx,
                "section_title": title,
                "section_number": number,
                "section_depth": depth,
                "section_path": [item["section_title"] for item in stack],
                "section_path_ids": [item["section_id"] for item in stack],
                "section_role": role,
            }
        )
        records.append(copied)

    return records