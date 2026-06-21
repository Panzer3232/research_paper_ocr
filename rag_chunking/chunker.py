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
            "related_algorithm_block_ids": [],
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
    if not text_blocks:
        return [], next_ordinal

    text = "\n\n".join(normalize_text(b.get("text")) for b in text_blocks if normalize_text(b.get("text"))).strip()
    if not text:
        return [], next_ordinal

    raw_chunks = splitter.split(text)
    chunks: list[dict[str, Any]] = []
    for part_index, part in enumerate(raw_chunks):
        part_text = normalize_text(part.text)
        if not part_text:
            continue
        next_ordinal += 1
        extra = {
            "prose_part_index": part_index,
            "prose_part_count": len(raw_chunks),
            "splitter_backend": splitter.backend,
        }
        chunk = make_chunk(paper_meta, section, text_blocks, "prose", part_text, next_ordinal, extra)
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


def chunk_algorithm_block(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    content: list[dict[str, Any]],
    algorithm_index: int,
    next_ordinal: int,
    config: ChunkConfig,
) -> tuple[dict[str, Any], int]:
    alg_block = content[algorithm_index]
    code_body = (alg_block.get("code_body") or "").strip()
    caption = (alg_block.get("caption") or "").strip()
    footnote = normalize_text(alg_block.get("footnote")) or None
    sub_type = alg_block.get("sub_type") or "algorithm"

    # Reuse equation context machinery: prose immediately before/after grounds the
    # algorithm in its narrative, identical to how equations are contextualised.
    previous_ctx, following_ctx, cue_flags = select_equation_context(
        content, algorithm_index, config.max_context_blocks
    )
    before_text = "\n\n".join(
        normalize_text(b.get("text")) for b in previous_ctx if normalize_text(b.get("text"))
    )
    after_text = "\n\n".join(
        normalize_text(b.get("text")) for b in following_ctx if normalize_text(b.get("text"))
    )

    label = "Algorithm" if sub_type == "algorithm" else "Code"
    parts = [
        f"Section path: {' > '.join(section.get('section_path') or [section.get('section_title')])}",
    ]
    if caption:
        parts.append(f"{label} caption: {caption}")
    if before_text:
        parts.append(f"Context before {label.lower()}:\n{before_text}")
    parts.append(f"{label}:\n{code_body}")
    if after_text:
        parts.append(f"Context after {label.lower()}:\n{after_text}")
    if footnote:
        parts.append(f"{label} footnote: {footnote}")

    blocks = previous_ctx + [alg_block] + following_ctx
    next_ordinal += 1
    extra = {
        "algorithm_body": code_body,
        "algorithm_caption": caption or None,
        "algorithm_footnote": footnote,
        "algorithm_sub_type": sub_type,
        "algorithm_context": {
            "previous_text_block_ids": [b.get("block_id") for b in previous_ctx],
            "next_text_block_ids": [b.get("block_id") for b in following_ctx],
            "previous_intro_cue": cue_flags["previous_intro_cue"],
            "following_explanation_cue": cue_flags["following_explanation_cue"],
            "one_sided_context_is_acceptable": cue_flags["one_sided_context_is_acceptable"],
        },
    }
    chunk = make_chunk(paper_meta, section, blocks, "algorithm", "\n\n".join(parts), next_ordinal, extra)
    relations = chunk["metadata"]["relations"]
    relations["related_text_block_ids"] = [b.get("block_id") for b in previous_ctx + following_ctx]
    relations["related_algorithm_block_ids"] = [alg_block.get("block_id")]

    if not before_text and not after_text:
        chunk["metadata"]["quality_flags"].append("algorithm_without_text_context")
    elif not before_text or not after_text:
        chunk["metadata"]["quality_flags"].append("algorithm_one_sided_context")

    return chunk, next_ordinal


def chunk_page_footnote_block(
    paper_meta: dict[str, Any],
    section: dict[str, Any],
    footnote_block: dict[str, Any],
    next_ordinal: int,
) -> tuple[dict[str, Any], int]:
    footnote_text = normalize_text(footnote_block.get("text"))
    next_ordinal += 1
    extra = {
        "is_page_footnote": True,
        "prose_part_index": 0,
        "prose_part_count": 1,
        "splitter_backend": "page_footnote_passthrough",
    }
    chunk = make_chunk(paper_meta, section, [footnote_block], "prose", footnote_text, next_ordinal, extra)
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
    footnote = normalize_text(table_block.get("footnote")) or None
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
        if footnote:
            text_parts.append(f"Table footnote: {footnote}")
        text = "\n\n".join(p for p in text_parts if p)

        next_ordinal += 1
        extra = {
            "table_caption": caption,
            "table_caption_source": caption_source,
            "table_img_path": table_block.get("img_path"),
            "table_footnote": footnote,
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
    image_footnotes = [normalize_text(b.get("footnote")) for b in image_blocks if normalize_text(b.get("footnote"))]
    image_paths = [b.get("img_path") for b in image_blocks if b.get("img_path")]

    text_parts = [
        f"Section path: {' > '.join(section.get('section_path') or [section.get('section_title')])}",
        f"Figure caption ({caption_source}): {caption}",
    ]
    if refs:
        text_parts.append("Nearby figure reference text:\n" + "\n".join(normalize_text(b.get("text")) for b in refs))
    if llm_captions:
        text_parts.append("Image summaries:\n" + "\n".join(f"- {caption_text}" for caption_text in llm_captions))
    if image_footnotes:
        text_parts.append("Image footnotes:\n" + "\n".join(f"- {fn}" for fn in image_footnotes))

    next_ordinal += 1
    extra = {
        "figure_caption": caption,
        "figure_caption_source": caption_source,
        "figure_group_size": len(image_blocks),
        "image_paths": image_paths,
        "image_footnotes": image_footnotes,
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
        # page_footnote chunks are prose-typed by design (chunk_page_footnote_block) but
        # must stay standalone: merging would silently drop is_page_footnote, since this
        # function only carries source_block_ids/source_block_indices/estimated_tokens
        # across a merge, not arbitrary extra metadata.
        if current.get("metadata", {}).get("is_page_footnote"):
            merged.append(current)
            i += 1
            continue
        if current.get("type") != "prose" or estimate_tokens(current.get("text")) >= min_tokens:
            merged.append(current)
            i += 1
            continue

        section_id = current.get("metadata", {}).get("section_id")
        can_merge_next = (
            i + 1 < len(chunks)
            and chunks[i + 1].get("type") == "prose"
            and not chunks[i + 1].get("metadata", {}).get("is_page_footnote")
            and chunks[i + 1].get("metadata", {}).get("section_id") == section_id
        )
        can_merge_prev = (
            merged
            and merged[-1].get("type") == "prose"
            and not merged[-1].get("metadata", {}).get("is_page_footnote")
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

            if btype == "page_footnote":
                chunk, ordinal = chunk_page_footnote_block(paper_meta, section, block, ordinal)
                chunks.append(chunk)
                index += 1
                continue

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

            if btype in ("algorithm", "code"):
                chunk, ordinal = chunk_algorithm_block(paper_meta, section, content, index, ordinal, config)
                chunks.append(chunk)
                index += 1
                continue

            if btype == "list":
                # Flatten list items into the prose accumulator as a single synthetic
                # text block. Lists in content sections are enumerated prose (e.g.
                # numbered hyperparameter sets, appendix table-of-contents entries);
                # they carry no retrieval value distinct from surrounding prose.
                items = block.get("items") or []
                joined = "\n".join(f"- {it}" for it in items if it)
                if joined:
                    synthetic: dict[str, Any] = {
                        "type": "text",
                        "text": joined,
                        "block_id": block.get("block_id"),
                        "block_index": block.get("block_index"),
                        "section_id": block.get("section_id"),
                        "section_index": block.get("section_index"),
                        "page_idx": block.get("page_idx"),
                    }
                    prose_accumulator.append(synthetic)
                index += 1
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