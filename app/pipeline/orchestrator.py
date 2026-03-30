from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.config.models import PipelineConfig
from app.core.exceptions import CaptioningError, ExtractionError
from app.core.stages import PipelineStage
from app.extract.extractor import MarkdownExtractor
from app.extract.mineru_runner import MinerURunner
from app.models.manifest import PipelineManifest
from app.models.paper import PaperRecord
from app.pipeline.input_resolver import OCRJob
from app.state.manifest_store import ManifestStore
from app.state.status import StageStatus
from app.storage.paths import PathResolver


@dataclass(slots=True)
class OCRResult:
    
    paper_key: str
    label: str
    pdf_path: str
    success: bool
    status: str
    markdown_path: str | None = None
    captioned_path: str | None = None
    manifest_path: str | None = None
    extraction_stats: dict[str, Any] = field(default_factory=dict)
    captioning_stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_key": self.paper_key,
            "label": self.label,
            "pdf_path": self.pdf_path,
            "success": self.success,
            "status": self.status,
            "markdown_path": self.markdown_path,
            "captioned_path": self.captioned_path,
            "manifest_path": self.manifest_path,
            "extraction_stats": self.extraction_stats,
            "captioning_stats": self.captioning_stats,
            "error": self.error,
        }


class OCROrchestrator:
    

    def __init__(
        self,
        config: PipelineConfig,
        *,
        paths: PathResolver | None = None,
        manifest_store: ManifestStore | None = None,
        extractor: MarkdownExtractor | None = None,
        captioner: Any | None = None,
    ) -> None:
        self.config = config
        self.paths = paths or PathResolver(config.output)
        self.paths.ensure_base_dirs()
        self.manifest_store = manifest_store or ManifestStore(self.paths)
        self.logger = logging.getLogger("paper_ocr")

        if extractor is None:
            runner = MinerURunner(config.mineru)
            extractor = MarkdownExtractor(runner, self.paths)
        self.extractor = extractor
        self.captioner = captioner

    def run(self, jobs: list[OCRJob]) -> list[OCRResult]:
       
        total = len(jobs)
        results: list[OCRResult] = []

        for index, job in enumerate(jobs, start=1):
            self._log(index, total, job.label, "starting")
            try:
                results.append(self._process(job, index=index, total=total))
            except Exception as exc:
                self._log(index, total, job.label, f"unhandled error: {exc}", level="error")
                results.append(
                    OCRResult(
                        paper_key=job.paper_key,
                        label=job.label,
                        pdf_path=str(job.pdf_path),
                        success=False,
                        status="failed_pipeline_error",
                        error=str(exc),
                    )
                )

        return results

    def _process(self, job: OCRJob, *, index: int, total: int) -> OCRResult:
        paper = self._paper_from_job(job)
        manifest = self._prepare_manifest(paper, job)

        try:
            extract_result = self._handle_extraction(manifest, paper, job, index=index, total=total)
        except ExtractionError as exc:
            return OCRResult(
                paper_key=job.paper_key,
                label=job.label,
                pdf_path=str(job.pdf_path),
                success=False,
                status="failed_extraction",
                manifest_path=str(self.manifest_store.path_for(job.paper_key)),
                error=str(exc),
            )

        markdown_path = self._resolve_markdown_path(extract_result, manifest)

        if not self.config.captioning.enabled or self.captioner is None:
            self.manifest_store.update_stage(
                manifest,
                PipelineStage.CAPTION_MARKDOWN,
                StageStatus.SKIPPED,
                message="captioning disabled or not configured",
            )
            self.manifest_store.mark_completed(manifest)
            self._log(index, total, job.label, f"completed | markdown={markdown_path}")

            return OCRResult(
                paper_key=job.paper_key,
                label=job.label,
                pdf_path=str(job.pdf_path),
                success=True,
                status="completed",
                markdown_path=markdown_path,
                manifest_path=str(self.manifest_store.path_for(job.paper_key)),
                extraction_stats=extract_result,
            )

        try:
            caption_result = self._handle_captioning(manifest, paper, job, index=index, total=total)
        except CaptioningError as exc:
            return OCRResult(
                paper_key=job.paper_key,
                label=job.label,
                pdf_path=str(job.pdf_path),
                success=False,
                status="failed_captioning",
                markdown_path=markdown_path,
                manifest_path=str(self.manifest_store.path_for(job.paper_key)),
                extraction_stats=extract_result,
                error=str(exc),
            )

        captioned_path = self._resolve_captioned_path(caption_result, manifest)
        self.manifest_store.mark_completed(manifest)
        self._log(index, total, job.label, f"completed | captioned={captioned_path}")

        return OCRResult(
            paper_key=job.paper_key,
            label=job.label,
            pdf_path=str(job.pdf_path),
            success=True,
            status="completed",
            markdown_path=markdown_path,
            captioned_path=captioned_path,
            manifest_path=str(self.manifest_store.path_for(job.paper_key)),
            extraction_stats=extract_result,
            captioning_stats=caption_result,
        )

    def _paper_from_job(self, job: OCRJob) -> PaperRecord:
        return PaperRecord.from_local_pdf(
            paper_key=job.paper_key,
            pdf_path=job.pdf_path,
            input_value=str(job.pdf_path),
        )

    def _prepare_manifest(self, paper: PaperRecord, job: OCRJob) -> PipelineManifest:
        manifest = self.manifest_store.get_or_create(paper)
        manifest.output_paths["pdf"] = str(job.pdf_path)
        manifest.selected_source = {
            "source_name": "local_pdf",
            "pdf_url": str(job.pdf_path),
            "version_type": "local_file",
            "host_type": "local_file",
        }
        self.manifest_store.save(manifest)
        self.manifest_store.update_paper_snapshot(manifest, paper)

        for stage in (
            PipelineStage.PARSE_INPUT,
            PipelineStage.FETCH_METADATA,
            PipelineStage.RECOVER_IDENTIFIERS,
            PipelineStage.RESOLVE_SOURCE,
            PipelineStage.DOWNLOAD_PDF,
        ):
            self.manifest_store.update_stage(
                manifest, stage, StageStatus.SKIPPED,
                message="ocr_pipeline: not applicable",
            )

        return manifest

    def _handle_extraction(
        self,
        manifest: PipelineManifest,
        paper: PaperRecord,
        job: OCRJob,
        *,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        if self._is_stage_done(manifest, PipelineStage.EXTRACT_MARKDOWN):
            self._log(index, total, job.label, "reusing previous MinerU output")
            return dict(manifest.stats.get("extraction") or {})

        self.manifest_store.update_stage(
            manifest, PipelineStage.EXTRACT_MARKDOWN, StageStatus.IN_PROGRESS,
            message="extracting markdown", increment_attempt=True,
        )
        self._log(index, total, job.label, "MinerU started")

        try:
            result = self.extractor(paper, manifest) or {}
            if not isinstance(result, dict):
                raise ExtractionError("extractor must return a dict or None")

            markdown_path = result.get("markdown_path")
            if isinstance(markdown_path, str) and markdown_path:
                manifest.output_paths["markdown"] = markdown_path

            manifest.stats["extraction"] = dict(result)
            self.manifest_store.save(manifest)

            self.manifest_store.update_stage(
                manifest, PipelineStage.EXTRACT_MARKDOWN, StageStatus.SUCCEEDED,
                message="markdown extracted", details=result,
            )

            suffix = (
                f"reusing previous MinerU output | markdown={manifest.output_paths.get('markdown')}"
                if result.get("reused_existing")
                else f"MinerU completed | markdown={manifest.output_paths.get('markdown')}"
            )
            self._log(index, total, job.label, suffix)
            return result

        except Exception as exc:
            self.manifest_store.update_stage(
                manifest, PipelineStage.EXTRACT_MARKDOWN, StageStatus.FAILED, error=str(exc),
            )
            raise ExtractionError(str(exc)) from exc

    def _handle_captioning(
        self,
        manifest: PipelineManifest,
        paper: PaperRecord,
        job: OCRJob,
        *,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        if self._is_stage_done(manifest, PipelineStage.CAPTION_MARKDOWN):
            self._log(index, total, job.label, "reusing previous captioned markdown")
            return dict(manifest.stats.get("captioning") or {})

        self.manifest_store.update_stage(
            manifest, PipelineStage.CAPTION_MARKDOWN, StageStatus.IN_PROGRESS,
            message="captioning markdown", increment_attempt=True,
        )
        self._log(index, total, job.label, "captioning started")

        try:
            result = self.captioner(paper, manifest) or {}
            if not isinstance(result, dict):
                raise CaptioningError("captioner must return a dict or None")

            captioned_path = result.get("captioned_markdown_path") or result.get("markdown_path")
            if isinstance(captioned_path, str) and captioned_path:
                manifest.output_paths["captioned_markdown"] = captioned_path

            manifest.stats["captioning"] = dict(result)
            self.manifest_store.save(manifest)

            self.manifest_store.update_stage(
                manifest, PipelineStage.CAPTION_MARKDOWN, StageStatus.SUCCEEDED,
                message="captioning completed", details=result,
            )
            self._log(index, total, job.label, "captioning completed")
            return result

        except Exception as exc:
            self.manifest_store.update_stage(
                manifest, PipelineStage.CAPTION_MARKDOWN, StageStatus.FAILED, error=str(exc),
            )
            raise CaptioningError(str(exc)) from exc

   
    def _is_stage_done(self, manifest: PipelineManifest, stage: PipelineStage) -> bool:
        if not self.config.resume.enabled or not self.config.resume.skip_completed_stages:
            return False
        return manifest.get_stage_state(stage).status == StageStatus.SUCCEEDED

    def _resolve_markdown_path(
        self, extract_result: dict[str, Any], manifest: PipelineManifest
    ) -> str | None:
        value = extract_result.get("markdown_path")
        if isinstance(value, str) and value:
            return value
        return manifest.output_paths.get("markdown")

    def _resolve_captioned_path(
        self, caption_result: dict[str, Any], manifest: PipelineManifest
    ) -> str | None:
        value = (
            caption_result.get("captioned_markdown_path")
            or caption_result.get("markdown_path")
        )
        if isinstance(value, str) and value:
            return value
        return manifest.output_paths.get("captioned_markdown")

    def _log(
        self, index: int, total: int, label: str, message: str, *, level: str = "info"
    ) -> None:
        text = f"{index}/{total} | {label} | {message}"
        if level == "error":
            self.logger.error(text)
        elif level == "warning":
            self.logger.warning(text)
        else:
            self.logger.info(text)