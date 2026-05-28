from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .common import estimate_tokens, normalize_text, sentence_like

REFERENCE_TERMS = ("references", "bibliography")
CHECKLIST_TERMS = ("did you", "left blank", "responsible nlp checklist", "submission checklist")
LATEX_MARKERS = ("$$", "\\begin", "\\end", "\\frac", "\\sum", "\\mathcal", "\\mathbf")


def _status(count: int, fail: bool = False) -> str:
    if count == 0:
        return "PASS"
    return "FAIL" if fail else "WARN"


def _preview(text: str, n: int = 160) -> str:
    return normalize_text(text)[:n]


def _has_natural_context(text: str) -> bool:
    cleaned = text.replace("Equation LaTeX:", "")
    return sentence_like(cleaned)


def evaluate_chunks(
    chunks: list[dict[str, Any]],
    chunk_size: int,
    table_chunk_size: int,
    min_prose_tokens: int,
) -> dict[str, Any]:
    by_type = Counter(chunk.get("type") for chunk in chunks)
    token_stats: dict[str, dict[str, Any]] = {}
    for chunk_type in sorted(by_type):
        values = [estimate_tokens(chunk.get("text", "")) for chunk in chunks if chunk.get("type") == chunk_type]
        token_stats[chunk_type] = {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": round(sum(values) / len(values), 1) if values else 0,
            "total_chunks": len(values),
        }

    empty = [c for c in chunks if not normalize_text(c.get("text"))]
    missing_section_path = [c for c in chunks if not c.get("metadata", {}).get("section_path")]
    missing_source_blocks = [c for c in chunks if not c.get("metadata", {}).get("source_block_ids")]
    missing_relations = [c for c in chunks if not c.get("metadata", {}).get("relations")]

    reference_leaks = []
    checklist_leaks = []
    for c in chunks:
        text = normalize_text(c.get("text")).lower()
        section_title = normalize_text(c.get("metadata", {}).get("section_title")).lower()
        if section_title in REFERENCE_TERMS or section_title.startswith("references"):
            reference_leaks.append(c)
        if any(term in text for term in CHECKLIST_TERMS):
            checklist_leaks.append(c)

    seen: dict[str, str] = {}
    duplicates = []
    for c in chunks:
        key = normalize_text(c.get("text")).lower()
        if not key:
            continue
        if key in seen:
            duplicates.append({"chunk_id": c.get("chunk_id"), "duplicate_of": seen[key], "preview": _preview(c.get("text", ""))})
        else:
            seen[key] = c.get("chunk_id")

    prose_short = []
    prose_oversized = []
    prose_noise = []
    for c in chunks:
        if c.get("type") != "prose":
            continue
        tokens = estimate_tokens(c.get("text"))
        if tokens < min_prose_tokens and "short_structural_prose_kept" not in c.get("metadata", {}).get("quality_flags", []):
            prose_short.append({"chunk_id": c.get("chunk_id"), "tokens": tokens, "preview": _preview(c.get("text", ""))})
        if tokens > int(chunk_size * 1.15):
            prose_oversized.append({"chunk_id": c.get("chunk_id"), "tokens": tokens, "preview": _preview(c.get("text", ""))})
        if not sentence_like(c.get("text", "")):
            prose_noise.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})

    equation_chunks = [c for c in chunks if c.get("type") == "equation"]
    equation_without_context = []
    equation_one_sided_needs_review = []
    equation_missing_latex = []
    equation_only = []
    for c in equation_chunks:
        meta = c.get("metadata", {})
        context = meta.get("equation_context", {})
        prev_ids = context.get("previous_text_block_ids") or []
        next_ids = context.get("next_text_block_ids") or []
        latex = meta.get("equation_latex") or ""
        if not latex:
            equation_missing_latex.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})
        if not prev_ids and not next_ids:
            equation_without_context.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})
        if (not prev_ids or not next_ids) and not context.get("one_sided_context_is_acceptable"):
            equation_one_sided_needs_review.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})
        if latex and normalize_text(c.get("text", "")).replace(normalize_text(latex), "").strip() in {"", "Equation LaTeX:"}:
            equation_only.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})
        elif not _has_natural_context(c.get("text", "")):
            equation_only.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})

    table_chunks = [c for c in chunks if c.get("type") == "table"]
    tables_missing_caption = []
    table_oversized = []
    table_split_parts = defaultdict(int)
    for c in table_chunks:
        meta = c.get("metadata", {})
        caption_source = meta.get("table_caption_source")
        if caption_source in {None, "inferred_from_section"}:
            tables_missing_caption.append({"chunk_id": c.get("chunk_id"), "caption_source": caption_source, "preview": _preview(c.get("text", ""))})
        tokens = estimate_tokens(c.get("text"))
        if tokens > int(table_chunk_size * 1.15):
            table_oversized.append({"chunk_id": c.get("chunk_id"), "tokens": tokens, "preview": _preview(c.get("text", ""))})
        if meta.get("table_part_count", 1) > 1:
            source_ids = tuple(meta.get("relations", {}).get("related_table_block_ids") or [])
            table_split_parts[source_ids] += 1

    figure_chunks = [c for c in chunks if c.get("type") == "figure"]
    figures_missing_caption = []
    figures_missing_llm = []
    for c in figure_chunks:
        meta = c.get("metadata", {})
        if meta.get("figure_caption_source") in {None, "inferred_from_section"}:
            figures_missing_caption.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})
        if not meta.get("llm_caption_count"):
            figures_missing_llm.append({"chunk_id": c.get("chunk_id"), "preview": _preview(c.get("text", ""))})

    relation_neighbor_gaps = []
    for idx, c in enumerate(chunks):
        relations = c.get("metadata", {}).get("relations", {})
        if idx > 0 and not relations.get("previous_chunk_id"):
            relation_neighbor_gaps.append({"chunk_id": c.get("chunk_id"), "missing": "previous_chunk_id"})
        if idx + 1 < len(chunks) and not relations.get("next_chunk_id"):
            relation_neighbor_gaps.append({"chunk_id": c.get("chunk_id"), "missing": "next_chunk_id"})

    quality_flags = Counter(flag for c in chunks for flag in c.get("metadata", {}).get("quality_flags", []))

    hard_fail_count = sum(
        len(x)
        for x in [
            empty,
            missing_section_path,
            missing_source_blocks,
            missing_relations,
            reference_leaks,
            checklist_leaks,
            equation_without_context,
            equation_missing_latex,
            equation_only,
            relation_neighbor_gaps,
        ]
    )
    warn_count = sum(
        len(x)
        for x in [
            duplicates,
            prose_short,
            prose_oversized,
            prose_noise,
            equation_one_sided_needs_review,
            tables_missing_caption,
            table_oversized,
            figures_missing_caption,
            figures_missing_llm,
        ]
    )

    return {
        "overall_status": "FAIL" if hard_fail_count else "WARN" if warn_count else "PASS",
        "total_chunks": len(chunks),
        "chunk_count_by_type": dict(by_type),
        "token_stats": token_stats,
        "quality_flags": dict(quality_flags),
        "empty_chunks": {"count": len(empty), "status": _status(len(empty), fail=True), "examples": [c.get("chunk_id") for c in empty[:10]]},
        "section_path_metadata": {"missing_count": len(missing_section_path), "status": _status(len(missing_section_path), fail=True)},
        "source_block_metadata": {"missing_count": len(missing_source_blocks), "status": _status(len(missing_source_blocks), fail=True)},
        "relationship_metadata": {"missing_count": len(missing_relations), "neighbor_gap_count": len(relation_neighbor_gaps), "status": _status(len(missing_relations) + len(relation_neighbor_gaps), fail=True), "neighbor_gap_examples": relation_neighbor_gaps[:10]},
        "references_leak": {"count": len(reference_leaks), "status": _status(len(reference_leaks), fail=True), "examples": [c.get("chunk_id") for c in reference_leaks[:10]]},
        "checklist_leak": {"count": len(checklist_leaks), "status": _status(len(checklist_leaks), fail=True), "examples": [c.get("chunk_id") for c in checklist_leaks[:10]]},
        "duplicates": {"count": len(duplicates), "status": _status(len(duplicates)), "examples": duplicates[:10]},
        "prose_short_chunks": {"count": len(prose_short), "status": _status(len(prose_short)), "examples": prose_short[:10]},
        "prose_oversized_chunks": {"count": len(prose_oversized), "status": _status(len(prose_oversized)), "examples": prose_oversized[:10]},
        "prose_noise_chunks": {"count": len(prose_noise), "status": _status(len(prose_noise)), "examples": prose_noise[:10]},
        "equation_context": {
            "total": len(equation_chunks),
            "without_context": len(equation_without_context),
            "one_sided_needs_review": len(equation_one_sided_needs_review),
            "missing_latex": len(equation_missing_latex),
            "equation_only": len(equation_only),
            "status": _status(len(equation_without_context) + len(equation_missing_latex) + len(equation_only), fail=True),
            "without_context_examples": equation_without_context[:10],
            "one_sided_review_examples": equation_one_sided_needs_review[:10],
            "equation_only_examples": equation_only[:10],
        },
        "table_quality": {
            "total": len(table_chunks),
            "missing_or_inferred_caption": len(tables_missing_caption),
            "oversized": len(table_oversized),
            "split_table_groups": len(table_split_parts),
            "status": _status(len(tables_missing_caption) + len(table_oversized)),
            "caption_examples": tables_missing_caption[:10],
            "oversized_examples": table_oversized[:10],
        },
        "figure_quality": {
            "total": len(figure_chunks),
            "missing_or_inferred_caption": len(figures_missing_caption),
            "missing_llm_caption": len(figures_missing_llm),
            "status": _status(len(figures_missing_caption) + len(figures_missing_llm)),
            "caption_examples": figures_missing_caption[:10],
            "llm_caption_examples": figures_missing_llm[:10],
        },
    }
