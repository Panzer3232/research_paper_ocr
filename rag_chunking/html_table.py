from __future__ import annotations

import html
import re
from typing import Any

from .common import estimate_tokens, normalize_text

TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def extract_rows(table_html: str) -> list[str]:
    return [row.group(0) for row in TR_RE.finditer(table_html or "")]


def strip_tags(value: str) -> str:
    return normalize_text(html.unescape(TAG_RE.sub(" ", value or "")))


def row_to_text(row_html: str) -> str:
    cells = [strip_tags(cell.group(1)) for cell in CELL_RE.finditer(row_html or "")]
    cells = [c for c in cells if c]
    if not cells:
        text = strip_tags(row_html)
        return text
    return " | ".join(cells)


def table_html_to_text(table_html: str) -> str:
    rows = extract_rows(table_html)
    if not rows:
        return strip_tags(table_html)
    return "\n".join(row_to_text(row) for row in rows if row_to_text(row))


def table_parts(
    table_html: str,
    caption: str | None,
    max_tokens: int,
    split_long_tables: bool = True,
) -> list[dict[str, Any]]:
    rows = extract_rows(table_html)
    full_text = table_html_to_text(table_html)
    if not split_long_tables or estimate_tokens(full_text) <= max_tokens or len(rows) <= 3:
        return [
            {
                "part_index": 0,
                "part_count": 1,
                "row_start": 0 if rows else None,
                "row_end": len(rows) - 1 if rows else None,
                "header_rows": rows[:1],
                "rows": rows,
                "text": full_text,
                "html": table_html,
            }
        ]

    header_rows = rows[:1]
    body_rows = rows[1:]
    parts: list[dict[str, Any]] = []
    current_rows: list[str] = []
    current_start = 1

    def emit(end_index: int) -> None:
        nonlocal current_rows, current_start
        if not current_rows:
            return
        part_rows = header_rows + current_rows
        html_part = "<table>" + "".join(part_rows) + "</table>"
        parts.append(
            {
                "part_index": len(parts),
                "part_count": -1,
                "row_start": current_start,
                "row_end": end_index,
                "header_rows": header_rows,
                "rows": part_rows,
                "text": table_html_to_text(html_part),
                "html": html_part,
            }
        )
        current_rows = []
        current_start = end_index + 1

    for offset, row in enumerate(body_rows, start=1):
        candidate_rows = header_rows + current_rows + [row]
        candidate_html = "<table>" + "".join(candidate_rows) + "</table>"
        candidate_text = "\n".join([
            caption or "",
            table_html_to_text(candidate_html),
        ]).strip()
        if current_rows and estimate_tokens(candidate_text) > max_tokens:
            emit(offset - 1)
        current_rows.append(row)

    emit(len(rows) - 1)

    total = len(parts)
    for part in parts:
        part["part_count"] = total
    return parts
