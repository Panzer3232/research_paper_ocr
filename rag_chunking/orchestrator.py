from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import ChunkConfig, iter_json_files, read_json, setup_logger, write_json
from .chunker import chunk_document
from .evaluator import evaluate_chunks
from .filters import prepare_document

_NORMALIZED_DIR = "normalized"
_CHUNKS_DIR = "chunks"
_EVALUATION_DIR = "evaluation"
_LOGS_DIR = "logs"

_PREPARE_REPORT_NAME = "_prepare_report.json"
_CHUNK_REPORT_NAME = "_chunk_report.json"
_EVALUATION_REPORT_NAME = "_evaluation_report.json"


@dataclass(frozen=True)
class RAGChunkResult:
    """Per-paper result returned by RAGChunkingOrchestrator.run()."""

    paper_key: str
    source_json_path: str
    success: bool
    normalized_json_path: str | None
    chunk_json_path: str | None
    evaluation_report_path: str | None  # None when evaluate=False or when a prior stage failed
    error: str | None
    # "completed" | "failed_prepare" | "failed_chunk" | "failed_evaluate"
    status: str


class RAGChunkingOrchestrator:
    """
    Orchestrates the three-stage RAG chunking pipeline for a set of structured
    JSON files produced by the OCR pipeline.

    Stage 1 — Prepare : normalize blocks, drop noise/references/checklists.
    Stage 2 — Chunk   : produce relationship-aware chunk JSON per paper.
    Stage 3 — Evaluate: optional quality report per paper (only when evaluate=True).

    Directory layout created under rag_chunks_root/:
        normalized/          prepare output
        chunks/              chunk output
        evaluation/          evaluation output (only when evaluate=True)
        logs/                one log file per stage, always written
    """

    def __init__(
        self,
        rag_chunks_root: Path,
        config: ChunkConfig,
        evaluate: bool = False,
    ) -> None:
        self._root = rag_chunks_root
        self._config = config
        self._evaluate = evaluate
        self._logger = logging.getLogger("rag_chunking.orchestrator")

        self._normalized_dir = self._root / _NORMALIZED_DIR
        self._chunks_dir = self._root / _CHUNKS_DIR
        self._logs_dir = self._root / _LOGS_DIR

        self._normalized_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)

        if self._evaluate:
            self._evaluation_dir = self._root / _EVALUATION_DIR
            self._evaluation_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._evaluation_dir = None

        # Per-stage loggers write to rag_chunks/logs/
        self._prepare_logger = setup_logger(
            "rag_chunking.prepare",
            self._logs_dir / "01_prepare.log",
        )
        self._chunk_logger = setup_logger(
            "rag_chunking.chunk",
            self._logs_dir / "02_chunk.log",
        )
        self._eval_logger = (
            setup_logger(
                "rag_chunking.evaluate",
                self._logs_dir / "03_evaluate.log",
            )
            if self._evaluate
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, input_paths: list[Path]) -> list[RAGChunkResult]:
        """
        Process each path through prepare → chunk → (optional) evaluate.
        Returns one RAGChunkResult per input file.
        Aggregate reports (_prepare_report.json, _chunk_report.json,
        _evaluation_report.json) are written to their respective directories.
        """
        prepare_reports: list[dict[str, Any]] = []
        chunk_reports: list[dict[str, Any]] = []
        eval_reports: list[dict[str, Any]] = []
        results: list[RAGChunkResult] = []

        self._logger.info(
            "RAG chunking pipeline started | files=%d | evaluate=%s | config=%s",
            len(input_paths),
            self._evaluate,
            self._config,
        )

        for path in input_paths:
            result, prep_report, chunk_report, eval_report = self._process_one(path)
            results.append(result)
            if prep_report:
                prepare_reports.append(prep_report)
            if chunk_report:
                chunk_reports.append(chunk_report)
            if eval_report:
                eval_reports.append(eval_report)

        # Write aggregate reports
        write_json(self._normalized_dir / _PREPARE_REPORT_NAME, prepare_reports)
        write_json(self._chunks_dir / _CHUNK_REPORT_NAME, chunk_reports)

        if self._evaluate and self._evaluation_dir is not None:
            status_counts: dict[str, int] = {}
            for r in eval_reports:
                s = r.get("overall_status", r.get("status", "UNKNOWN"))
                status_counts[s] = status_counts.get(s, 0) + 1
            eval_summary = {
                "total_files": len(eval_reports),
                "status_counts": status_counts,
                "reports": eval_reports,
            }
            eval_report_path = self._evaluation_dir / _EVALUATION_REPORT_NAME
            write_json(eval_report_path, eval_summary)
            if self._eval_logger:
                self._eval_logger.info(
                    "Evaluation report written: %s | status_counts=%s",
                    eval_report_path,
                    status_counts,
                )

        succeeded = sum(1 for r in results if r.success)
        self._logger.info(
            "RAG chunking pipeline complete | succeeded=%d | failed=%d",
            succeeded,
            len(results) - succeeded,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_one(
        self,
        path: Path,
    ) -> tuple[RAGChunkResult, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """
        Run all stages for a single file.
        Returns (RAGChunkResult, prepare_report, chunk_report, eval_report).
        Any report may be None if its stage was not reached.
        """
        paper_key = path.stem

        # ── Stage 1: Prepare ────────────────────────────────────────────
        normalized_path: Path | None = None
        prep_report: dict[str, Any] | None = None
        try:
            raw_data = read_json(path)
            prepared_data, prep_report = prepare_document(raw_data, path.name)
            normalized_path = self._normalized_dir / path.name
            write_json(normalized_path, prepared_data)
            self._prepare_logger.info(
                "Prepared %s | sections %d→%d | blocks %d→%d | drops=%s",
                path.name,
                prep_report["input_sections"],
                prep_report["kept_sections"],
                prep_report["input_blocks"],
                prep_report["kept_blocks"],
                prep_report["block_drops"],
            )
            paper_key = prep_report.get("paper_key", paper_key)
        except Exception as exc:
            self._prepare_logger.exception("Failed preparing %s: %s", path.name, exc)
            prep_report = {"file": path.name, "status": "ERROR", "error": str(exc)}
            return (
                RAGChunkResult(
                    paper_key=paper_key,
                    source_json_path=str(path.resolve()),
                    success=False,
                    normalized_json_path=None,
                    chunk_json_path=None,
                    evaluation_report_path=None,
                    error=str(exc),
                    status="failed_prepare",
                ),
                prep_report,
                None,
                None,
            )

        # ── Stage 2: Chunk ──────────────────────────────────────────────
        chunk_path: Path | None = None
        chunk_report: dict[str, Any] | None = None
        chunks: list[dict[str, Any]] = []
        try:
            chunks, chunk_report = chunk_document(prepared_data, self._config)
            chunk_report["file"] = path.name
            chunk_path = self._chunks_dir / path.name
            write_json(chunk_path, chunks)
            self._chunk_logger.info(
                "Chunked %s | total=%d | by_type=%s | flags=%s | backend=%s",
                path.name,
                chunk_report["total_chunks"],
                chunk_report["chunks_by_type"],
                chunk_report["quality_flags"],
                chunk_report["splitter_backend"],
            )
        except Exception as exc:
            self._chunk_logger.exception("Failed chunking %s: %s", path.name, exc)
            chunk_report = {"file": path.name, "status": "ERROR", "error": str(exc)}
            return (
                RAGChunkResult(
                    paper_key=paper_key,
                    source_json_path=str(path.resolve()),
                    success=False,
                    normalized_json_path=str(normalized_path.resolve()),
                    chunk_json_path=None,
                    evaluation_report_path=None,
                    error=str(exc),
                    status="failed_chunk",
                ),
                prep_report,
                chunk_report,
                None,
            )

        # ── Stage 3: Evaluate (optional) ────────────────────────────────
        eval_report_path: Path | None = None
        eval_report: dict[str, Any] | None = None
        if self._evaluate and self._evaluation_dir is not None and self._eval_logger is not None:
            try:
                eval_result = evaluate_chunks(
                    chunks,
                    self._config.chunk_size,
                    self._config.table_chunk_size,
                    self._config.min_prose_tokens,
                )
                eval_result["file"] = path.name
                eval_report = eval_result
                eval_report_path = self._evaluation_dir / path.name
                write_json(eval_report_path, eval_result)
                self._eval_logger.info(
                    "Evaluated %s | status=%s | chunks=%d | by_type=%s",
                    path.name,
                    eval_result["overall_status"],
                    eval_result["total_chunks"],
                    eval_result["chunk_count_by_type"],
                )
            except Exception as exc:
                self._eval_logger.exception("Failed evaluating %s: %s", path.name, exc)
                eval_report = {"file": path.name, "status": "ERROR", "error": str(exc)}
                return (
                    RAGChunkResult(
                        paper_key=paper_key,
                        source_json_path=str(path.resolve()),
                        success=False,
                        normalized_json_path=str(normalized_path.resolve()),
                        chunk_json_path=str(chunk_path.resolve()),
                        evaluation_report_path=None,
                        error=str(exc),
                        status="failed_evaluate",
                    ),
                    prep_report,
                    chunk_report,
                    eval_report,
                )

        return (
            RAGChunkResult(
                paper_key=paper_key,
                source_json_path=str(path.resolve()),
                success=True,
                normalized_json_path=str(normalized_path.resolve()),
                chunk_json_path=str(chunk_path.resolve()),
                evaluation_report_path=str(eval_report_path.resolve()) if eval_report_path else None,
                error=None,
                status="completed",
            ),
            prep_report,
            chunk_report,
            eval_report,
        )