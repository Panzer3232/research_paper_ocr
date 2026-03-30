from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.exceptions import ExtractionError
from app.extract.mineru_runner import MinerURunner
from app.models.manifest import PipelineManifest
from app.models.paper import PaperRecord
from app.storage.paths import PathResolver
from app.storage.writers import write_json, write_text


@dataclass(slots=True)
class ExtractionSummary:
    markdown_path: str
    mineru_markdown_path: str
    mineru_dir: str
    images_dir: str | None
    middle_json_path: str | None
    content_list_path: str | None
    layout_pdf_path: str | None
    span_pdf_path: str | None
    num_images: int
    num_tables: int
    num_equations: int
    num_headings: int
    reused_existing: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown_path": self.markdown_path,
            "mineru_markdown_path": self.mineru_markdown_path,
            "mineru_dir": self.mineru_dir,
            "images_dir": self.images_dir,
            "middle_json_path": self.middle_json_path,
            "content_list_path": self.content_list_path,
            "layout_pdf_path": self.layout_pdf_path,
            "span_pdf_path": self.span_pdf_path,
            "num_images": self.num_images,
            "num_tables": self.num_tables,
            "num_equations": self.num_equations,
            "num_headings": self.num_headings,
            "reused_existing": self.reused_existing,
        }


class MarkdownExtractor:
    def __init__(
        self,
        mineru_runner: MinerURunner,
        paths: PathResolver,
    ) -> None:
        self.mineru_runner = mineru_runner
        self.paths = paths

    def __call__(
        self,
        paper: PaperRecord,
        manifest: PipelineManifest,
    ) -> dict[str, Any]:
        pdf_path = manifest.output_paths.get("pdf")
        if not pdf_path:
            raise ExtractionError(f"No PDF path recorded in manifest for {paper.paper_key}")

        mineru_dir = self.paths.mineru_dir(paper.paper_key)
        final_markdown_path = self.paths.markdown_path(paper.paper_key)

        if final_markdown_path.exists():
            text = final_markdown_path.read_text(encoding="utf-8")
            summary = self._build_summary(
                markdown_path=final_markdown_path,
                mineru_markdown_path=manifest.stats.get("extraction", {}).get("mineru_markdown_path"),
                mineru_dir=manifest.stats.get("extraction", {}).get("mineru_dir") or str(mineru_dir),
                images_dir=manifest.stats.get("extraction", {}).get("images_dir"),
                middle_json_path=manifest.stats.get("extraction", {}).get("middle_json_path"),
                content_list_path=manifest.stats.get("extraction", {}).get("content_list_path"),
                layout_pdf_path=manifest.stats.get("extraction", {}).get("layout_pdf_path"),
                span_pdf_path=manifest.stats.get("extraction", {}).get("span_pdf_path"),
                markdown_text=text,
                reused_existing=True,
            )
            self._write_summary_json(paper.paper_key, summary)
            return summary.to_dict()

        run_result = self.mineru_runner.run(pdf_path, mineru_dir)

        mineru_markdown_path = Path(run_result.markdown_path)
        if not mineru_markdown_path.exists():
            raise ExtractionError(
                f"MinerU output markdown not found after run: {mineru_markdown_path}"
            )

        markdown_text = mineru_markdown_path.read_text(encoding="utf-8")
        write_text(final_markdown_path, markdown_text)

        summary = self._build_summary(
            markdown_path=final_markdown_path,
            mineru_markdown_path=run_result.markdown_path,
            mineru_dir=run_result.document_dir,
            images_dir=run_result.images_dir,
            middle_json_path=run_result.middle_json_path,
            content_list_path=run_result.content_list_path,
            layout_pdf_path=run_result.layout_pdf_path,
            span_pdf_path=run_result.span_pdf_path,
            markdown_text=markdown_text,
            reused_existing=run_result.reused_existing,
        )
        self._write_summary_json(paper.paper_key, summary)
        return summary.to_dict()

    def _build_summary(
        self,
        *,
        markdown_path: Path,
        mineru_markdown_path: str | None,
        mineru_dir: str,
        images_dir: str | None,
        middle_json_path: str | None,
        content_list_path: str | None,
        layout_pdf_path: str | None,
        span_pdf_path: str | None,
        markdown_text: str,
        reused_existing: bool,
    ) -> ExtractionSummary:
        num_images = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", markdown_text))
        num_tables = len(re.findall(r"(?m)^\|.+\|\s*\n\|[-:| ]+\|", markdown_text))
        num_equations = markdown_text.count("$$") // 2
        num_headings = len(re.findall(r"(?m)^#{1,6}\s+", markdown_text))

        return ExtractionSummary(
            markdown_path=str(markdown_path),
            mineru_markdown_path=mineru_markdown_path or str(markdown_path),
            mineru_dir=mineru_dir,
            images_dir=images_dir,
            middle_json_path=middle_json_path,
            content_list_path=content_list_path,
            layout_pdf_path=layout_pdf_path,
            span_pdf_path=span_pdf_path,
            num_images=num_images,
            num_tables=num_tables,
            num_equations=num_equations,
            num_headings=num_headings,
            reused_existing=reused_existing,
        )

    def _write_summary_json(self, paper_key: str, summary: ExtractionSummary) -> None:
        summary_path = self.paths.mineru_dir(paper_key) / "extraction_summary.json"
        write_json(summary_path, summary.to_dict())