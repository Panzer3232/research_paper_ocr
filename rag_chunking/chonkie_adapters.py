from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import CHARS_PER_TOKEN


@dataclass
class TextChunk:
    text: str


@dataclass(frozen=True)
class TextChunk:
    text: str


class RecursiveTextSplitter:
    def __init__(self, chunk_size_tokens: int, overlap_tokens: int):
        self.chunk_size_chars = max(1, chunk_size_tokens * CHARS_PER_TOKEN)
        self.overlap_chars = max(0, overlap_tokens * CHARS_PER_TOKEN)
        self._chonkie = None
        self._overlap = None

        try:
            from chonkie import RecursiveChunker

            self._chonkie = RecursiveChunker(
                tokenizer="character",
                chunk_size=self.chunk_size_chars,
            )

            if self.overlap_chars > 0:
                from chonkie.refinery import OverlapRefinery

                self._overlap = OverlapRefinery(
                    tokenizer="character",
                    context_size=self.overlap_chars,
                )
        except Exception:
            self._chonkie = None
            self._overlap = None

    @property
    def backend(self) -> str:
        if self._chonkie is None:
            return "local_recursive_fallback"
        if self.overlap_chars == 0:
            return "chonkie_recursive_no_overlap"
        return "chonkie_recursive_with_overlap"

    def split(self, text: str) -> list[TextChunk]:
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size_chars:
            return [TextChunk(text)]

        if self._chonkie is not None:
            chunks = self._split_with_chonkie(text)
            if chunks:
                return chunks

        return self._fallback_split(text)

    def _split_with_chonkie(self, text: str) -> list[TextChunk]:
        raw_chunks = self._chonkie.chunk(text)

        if self._overlap is not None and self.overlap_chars > 0:
            raw_chunks = self._overlap.refine(raw_chunks)

        chunks: list[TextChunk] = []
        for raw_chunk in raw_chunks:
            chunk_text = self._extract_chunk_text(raw_chunk)
            if chunk_text:
                chunks.append(TextChunk(chunk_text))

        return chunks

    @staticmethod
    def _extract_chunk_text(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk.strip()

        if isinstance(chunk, dict):
            for key in ("text", "content", "chunk"):
                value = chunk.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        for attr in ("text", "content", "chunk"):
            value = getattr(chunk, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""

    def _fallback_split(self, text: str) -> list[TextChunk]:
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

            if len(candidate) <= self.chunk_size_chars:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(paragraph) <= self.chunk_size_chars:
                current = paragraph
                continue

            step = max(1, self.chunk_size_chars - self.overlap_chars)
            for start in range(0, len(paragraph), step):
                part = paragraph[start : start + self.chunk_size_chars].strip()
                if part:
                    chunks.append(part)

            current = ""

        if current:
            chunks.append(current)

        return [TextChunk(chunk) for chunk in chunks if chunk.strip()]
