from __future__ import annotations

from collections import Counter
from typing import Any

from .chonkie_adapters import RecursiveTextSplitter
from .common import ChunkConfig, estimate_tokens, normalize_text, page_span_from_blocks, stable_id
from .context import (
    find_figure_reference_text,
    find_table_reference_text,
    recover_figure_caption,
    recover_table_caption,
    select_equation_context,
)
from .html_table import table_parts


def base_metadata(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    blocks: list[dict[str, Any]],
    chunk_type: str,
) -> dict[str, Any]:
    return {
        "paper_key": paper_meta.get("paper_key"),
        "source_pdf": paper_meta.get("source_pdf"),
        "paper_title": paper_meta.get("paper_title"),
        "paper_front_matter": paper_meta.get("front_matter"),
        "chunk_type": chunk_type,
        "section_id": section.get("section_id"),
        "section_index": section.get("section_index"),
        "section_title": section.get("section_title"),
        "section_number": section.get("section_number"),
        "section_depth": section.get("section_depth"),
        "section_path": section.get("section_path") or [section.get("section_title")],
        "section_path_ids": section.get("section_path_ids") or [section.get("section_id")],
        "page_span": page_span_from_blocks(blocks),
        "source_block_ids": [b.get("block_id") for b in blocks if b.get("block_id")],
        "source_block_indices": [b.get("block_index") for b in blocks if b.get("block_index") is not None],
        "source_block_types": [b.get("type") for b in blocks],
        "relations": {
            "parent_section_id": section.get("section_id"),
            "ancestor_section_ids": section.get("section_path_ids") or [],
            "previous_chunk_id": None,
            "next_chunk_id": None,
            "previous_same_section_chunk_id": None,
            "next_same_section_chunk_id": None,
            "previous_source_block_id": None,
            "next_source_block_id": None,
            "related_text_block_ids": [],
            "related_equation_block_ids": [],
            "related_table_block_ids": [],
            "related_figure_block_ids": [],
        },
        "quality_flags": [],
    }


def make_chunk(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    blocks: list[dict[str, Any]],
    chunk_type: str,
    text: str,
    ordinal: int,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chunk_id = f"{paper_meta.get('paper_key', 'paper')}_{ordinal:05d}_{chunk_type}"
    metadata = base_metadata(paper_meta, section, blocks, chunk_type)
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata["estimated_tokens"] = estimate_tokens(text)
    return {
        "chunk_id": chunk_id,
        "type": chunk_type,
        "text": text.strip(),
        "metadata": metadata,
    }


def chunk_prose_blocks(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    text_blocks: list[dict[str, Any]],
    splitter: RecursiveTextSplitter,
    next_ordinal: int,
    min_prose_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    clean_blocks = [
        block
        for block in text_blocks
        if normalize_text(block.get("text"))
    ]

    if not clean_blocks:
        return [], next_ordinal

    chunk_units: list[dict[str, Any]] = []
    current_blocks: list[dict[str, Any]] = []

    def blocks_text(blocks: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            normalize_text(block.get("text"))
            for block in blocks
            if normalize_text(block.get("text"))
        ).strip()

    def emit_current() -> None:
        nonlocal current_blocks
        if not current_blocks:
            return

        text = blocks_text(current_blocks)
        if text:
            chunk_units.append(
                {
                    "text": text,
                    "blocks": current_blocks,
                    "quality_flags": [],
                }
            )

        current_blocks = []

    def emit_split_single_block(block: dict[str, Any]) -> None:
        text = normalize_text(block.get("text"))
        if not text:
            return

        raw_parts = splitter.split(text)
        parts = [normalize_text(part.text) for part in raw_parts if normalize_text(part.text)]

        if not parts:
            return

        search_start = 0
        for part_text in parts:
            start = text.find(part_text, search_start)
            if start < 0:
                start = None
                end = None
            else:
                end = start + len(part_text)
                search_start = end

            chunk_units.append(
                {
                    "text": part_text,
                    "blocks": [block],
                    "quality_flags": ["single_source_block_split"],
                    "source_block_char_spans": [
                        {
                            "block_id": block.get("block_id"),
                            "block_index": block.get("block_index"),
                            "char_start": start,
                            "char_end": end,
                            "offset_basis": "normalized_text",
                        }
                    ],
                }
            )

    for block in clean_blocks:
        block_text = normalize_text(block.get("text"))
        block_tokens = estimate_tokens(block_text)

        if block_tokens > splitter.chunk_size_chars // 4:
            emit_current()
            emit_split_single_block(block)
            continue

        candidate_blocks = current_blocks + [block]
        candidate_text = blocks_text(candidate_blocks)

        if current_blocks and estimate_tokens(candidate_text) > splitter.chunk_size_chars // 4:
            emit_current()

        current_blocks.append(block)

    emit_current()

    chunks: list[dict[str, Any]] = []
    part_count = len(chunk_units)

    for part_index, unit in enumerate(chunk_units):
        part_text = unit["text"]
        blocks = unit["blocks"]

        next_ordinal += 1

        extra = {
            "prose_part_index": part_index,
            "prose_part_count": part_count,
            "splitter_backend": splitter.backend,
        }

        if "source_block_char_spans" in unit:
            extra["source_block_char_spans"] = unit["source_block_char_spans"]

        chunk = make_chunk(
            paper_meta,
            section,
            blocks,
            "prose",
            part_text,
            next_ordinal,
            extra,
        )

        for flag in unit.get("quality_flags", []):
            chunk["metadata"]["quality_flags"].append(flag)

        if estimate_tokens(part_text) < min_prose_tokens:
            chunk["metadata"]["quality_flags"].append("short_prose_candidate")

        chunks.append(chunk)

    return chunks, next_ordinal


def chunk_equation_block(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    content: list[dict[str, Any]],
    equation_index: int,
    next_ordinal: int,
    config: ChunkConfig,
) -> tuple[dict[str, Any], int]:
    eq_block = content[equation_index]
    previous_ctx, following_ctx, cue_flags = select_equation_context(
        content, equation_index, config.max_context_blocks
    )
    eq_text = (eq_block.get("text") or "").strip()

    before_text = "\n\n".join(normalize_text(b.get("text")) for b in previous_ctx if normalize_text(b.get("text")))
    after_text = "\n\n".join(normalize_text(b.get("text")) for b in following_ctx if normalize_text(b.get("text")))

    parts = [
        f"Section path: {' > '.join(section.get('section_path') or [section.get('section_title')])}",
    ]
    if before_text:
        parts.append(f"Context before equation:\n{before_text}")
    parts.append(f"Equation LaTeX:\n{eq_text}")
    if after_text:
        parts.append(f"Context after equation:\n{after_text}")

    blocks = previous_ctx + [eq_block] + following_ctx
    next_ordinal += 1
    extra = {
        "equation_latex": eq_text,
        "equation_text_format": eq_block.get("text_format") or "latex",
        "equation_img_path": eq_block.get("img_path"),
        "equation_context": {
            "previous_text_block_ids": [b.get("block_id") for b in previous_ctx],
            "next_text_block_ids": [b.get("block_id") for b in following_ctx],
            "previous_intro_cue": cue_flags["previous_intro_cue"],
            "following_explanation_cue": cue_flags["following_explanation_cue"],
            "one_sided_context_is_acceptable": cue_flags["one_sided_context_is_acceptable"],
        },
    }
    chunk = make_chunk(paper_meta, section, blocks, "equation", "\n\n".join(parts), next_ordinal, extra)
    relations = chunk["metadata"]["relations"]
    relations["related_text_block_ids"] = [b.get("block_id") for b in previous_ctx + following_ctx]
    relations["related_equation_block_ids"] = [eq_block.get("block_id")]

    if not before_text and not after_text:
        chunk["metadata"]["quality_flags"].append("equation_without_text_context")
    elif not before_text or not after_text:
        chunk["metadata"]["quality_flags"].append("equation_one_sided_context")

    return chunk, next_ordinal


def chunk_table_block(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    content: list[dict[str, Any]],
    table_index: int,
    next_ordinal: int,
    config: ChunkConfig,
) -> tuple[list[dict[str, Any]], int]:
    table_block = content[table_index]
    refs = find_table_reference_text(content, table_index)
    caption, caption_source = recover_table_caption(table_block, refs, section.get("section_title") or "")
    parts = table_parts(
        table_block.get("table_body") or "",
        caption,
        config.table_chunk_size,
        split_long_tables=config.split_long_tables,
    )

    chunks: list[dict[str, Any]] = []
    for part in parts:
        text_parts = [
            f"Section path: {' > '.join(section.get('section_path') or [section.get('section_title')])}",
            f"Table caption ({caption_source}): {caption}",
        ]
        if refs:
            ref_text = "\n".join(normalize_text(b.get("text")) for b in refs)
            text_parts.append(f"Nearby table reference text:\n{ref_text}")
        text_parts.append(f"Table rows:\n{part['text']}")
        text = "\n\n".join(p for p in text_parts if p)

        next_ordinal += 1
        extra = {
            "table_caption": caption,
            "table_caption_source": caption_source,
            "table_img_path": table_block.get("img_path"),
            "table_part_index": part["part_index"],
            "table_part_count": part["part_count"],
            "table_row_start": part["row_start"],
            "table_row_end": part["row_end"],
            "table_html": part["html"],
            "table_reference_text_block_ids": [b.get("block_id") for b in refs],
        }
        blocks = refs + [table_block]
        chunk = make_chunk(paper_meta, section, blocks, "table", text, next_ordinal, extra)
        relations = chunk["metadata"]["relations"]
        relations["related_text_block_ids"] = [b.get("block_id") for b in refs]
        relations["related_table_block_ids"] = [table_block.get("block_id")]
        if caption_source == "inferred_from_section":
            chunk["metadata"]["quality_flags"].append("table_caption_inferred")
        if part["part_count"] > 1:
            chunk["metadata"]["quality_flags"].append("table_row_split_with_header_repeated")
        chunks.append(chunk)

    return chunks, next_ordinal


def chunk_image_group(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    content: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    next_ordinal: int,
) -> tuple[dict[str, Any], int]:
    image_blocks = content[start_index : end_index + 1]
    refs = find_figure_reference_text(content, start_index, end_index)
    caption, caption_source = recover_figure_caption(image_blocks, refs, section.get("section_title") or "")
    llm_captions = [normalize_text(b.get("caption_llm")) for b in image_blocks if normalize_text(b.get("caption_llm"))]
    image_paths = [b.get("img_path") for b in image_blocks if b.get("img_path")]

    text_parts = [
        f"Section path: {' > '.join(section.get('section_path') or [section.get('section_title')])}",
        f"Figure caption ({caption_source}): {caption}",
    ]
    if refs:
        text_parts.append("Nearby figure reference text:\n" + "\n".join(normalize_text(b.get("text")) for b in refs))
    if llm_captions:
        text_parts.append("Image summaries:\n" + "\n".join(f"- {caption_text}" for caption_text in llm_captions))

    next_ordinal += 1
    extra = {
        "figure_caption": caption,
        "figure_caption_source": caption_source,
        "figure_group_size": len(image_blocks),
        "image_paths": image_paths,
        "llm_caption_count": len(llm_captions),
        "figure_reference_text_block_ids": [b.get("block_id") for b in refs],
    }
    blocks = refs + image_blocks
    chunk = make_chunk(paper_meta, section, blocks, "figure", "\n\n".join(text_parts), next_ordinal, extra)
    relations = chunk["metadata"]["relations"]
    relations["related_text_block_ids"] = [b.get("block_id") for b in refs]
    relations["related_figure_block_ids"] = [b.get("block_id") for b in image_blocks]
    if caption_source == "inferred_from_section":
        chunk["metadata"]["quality_flags"].append("figure_caption_inferred")
    if not llm_captions:
        chunk["metadata"]["quality_flags"].append("figure_without_llm_caption")
    return chunk, next_ordinal


def merge_short_prose_chunks(chunks: list[dict[str, Any]], min_tokens: int) -> list[dict[str, Any]]:
    if not chunks:
        return chunks
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(chunks):
        current = chunks[i]
        if current.get("type") != "prose" or estimate_tokens(current.get("text")) >= min_tokens:
            merged.append(current)
            i += 1
            continue

        section_id = current.get("metadata", {}).get("section_id")
        can_merge_next = (
            i + 1 < len(chunks)
            and chunks[i + 1].get("type") == "prose"
            and chunks[i + 1].get("metadata", {}).get("section_id") == section_id
        )
        can_merge_prev = (
            merged
            and merged[-1].get("type") == "prose"
            and merged[-1].get("metadata", {}).get("section_id") == section_id
        )

        if can_merge_next:
            nxt = chunks[i + 1]
            current["text"] = f"{current['text']}\n\n{nxt['text']}".strip()
            current["metadata"]["source_block_ids"] = list(dict.fromkeys(current["metadata"].get("source_block_ids", []) + nxt["metadata"].get("source_block_ids", [])))
            current["metadata"]["source_block_indices"] = list(dict.fromkeys(current["metadata"].get("source_block_indices", []) + nxt["metadata"].get("source_block_indices", [])))
            current["metadata"]["estimated_tokens"] = estimate_tokens(current["text"])
            current["metadata"]["quality_flags"].append("short_prose_merged_with_next")
            merged.append(current)
            i += 2
        elif can_merge_prev:
            merged[-1]["text"] = f"{merged[-1]['text']}\n\n{current['text']}".strip()
            merged[-1]["metadata"]["source_block_ids"] = list(dict.fromkeys(merged[-1]["metadata"].get("source_block_ids", []) + current["metadata"].get("source_block_ids", [])))
            merged[-1]["metadata"]["source_block_indices"] = list(dict.fromkeys(merged[-1]["metadata"].get("source_block_indices", []) + current["metadata"].get("source_block_indices", [])))
            merged[-1]["metadata"]["estimated_tokens"] = estimate_tokens(merged[-1]["text"])
            merged[-1]["metadata"]["quality_flags"].append("short_prose_merged_with_previous")
            i += 1
        else:
            current["metadata"]["quality_flags"].append("short_structural_prose_kept")
            merged.append(current)
            i += 1
    return merged


def attach_chunk_neighbors(chunks: list[dict[str, Any]]) -> None:
    last_by_section: dict[str, dict[str, Any]] = {}
    for idx, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        relations = metadata.setdefault("relations", {})
        relations["previous_chunk_id"] = chunks[idx - 1]["chunk_id"] if idx > 0 else None
        relations["next_chunk_id"] = chunks[idx + 1]["chunk_id"] if idx + 1 < len(chunks) else None

        section_id = metadata.get("section_id")
        previous_same = last_by_section.get(section_id) if section_id else None
        relations["previous_same_section_chunk_id"] = previous_same["chunk_id"] if previous_same else None
        if previous_same:
            previous_same["metadata"]["relations"]["next_same_section_chunk_id"] = chunk["chunk_id"]
        if section_id:
            last_by_section[section_id] = chunk


def chunk_document(data: dict[str, Any], config: ChunkConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = data.get("metadata") or {}
    paper_meta = {
        "paper_key": metadata.get("paper_key", "unknown"),
        "source_pdf": metadata.get("source_pdf", "unknown"),
        "paper_title": metadata.get("paper_title", metadata.get("paper_key", "unknown")),
        "front_matter": metadata.get("front_matter"),
    }
    splitter = RecursiveTextSplitter(config.chunk_size, config.overlap)
    chunks: list[dict[str, Any]] = []
    ordinal = 0
    report = {
        "paper_key": paper_meta["paper_key"],
        "splitter_backend": splitter.backend,
        "sections_processed": 0,
        "sections_skipped_title_authors": 0,
        "chunks_by_type": {},
        "quality_flags": Counter(),
    }

    for section in data.get("sections") or []:
        report["sections_processed"] += 1
        if section.get("section_role") == "title_authors":
            report["sections_skipped_title_authors"] += 1
            continue

        content = section.get("content") or []
        prose_accumulator: list[dict[str, Any]] = []
        index = 0

        def flush_prose() -> None:
            nonlocal chunks, ordinal, prose_accumulator
            created, ordinal = chunk_prose_blocks(
                paper_meta,
                section,
                prose_accumulator,
                splitter,
                ordinal,
                config.min_prose_tokens,
            )
            chunks.extend(created)
            prose_accumulator = []

        while index < len(content):
            block = content[index]
            btype = block.get("type")

            if btype == "text":
                prose_accumulator.append(block)
                index += 1
                continue

            flush_prose()

            if btype == "equation":
                chunk, ordinal = chunk_equation_block(paper_meta, section, content, index, ordinal, config)
                chunks.append(chunk)
                index += 1
                continue

            if btype == "table":
                created, ordinal = chunk_table_block(paper_meta, section, content, index, ordinal, config)
                chunks.extend(created)
                index += 1
                continue

            if btype == "image":
                start = index
                end = index
                while end + 1 < len(content) and content[end + 1].get("type") == "image":
                    end += 1
                chunk, ordinal = chunk_image_group(paper_meta, section, content, start, end, ordinal)
                chunks.append(chunk)
                index = end + 1
                continue

            index += 1

        flush_prose()

    chunks = merge_short_prose_chunks(chunks, config.min_prose_tokens)
    attach_chunk_neighbors(chunks)

    by_type = Counter(chunk.get("type") for chunk in chunks)
    report["chunks_by_type"] = dict(by_type)
    flags = Counter(flag for chunk in chunks for flag in chunk.get("metadata", {}).get("quality_flags", []))
    report["quality_flags"] = dict(flags)
    report["total_chunks"] = len(chunks)
    return chunks, report
