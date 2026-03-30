from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.exceptions import ManifestStoreError
from app.core.stages import PipelineStage
from app.models.manifest import PipelineManifest
from app.models.paper import PaperRecord
from app.state.status import BatchStatus, StageStatus
from app.storage.paths import PathResolver


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ManifestStore:
    def __init__(self, paths: PathResolver) -> None:
        self.paths = paths
        self.paths.ensure_base_dirs()

    def path_for(self, paper_key: str) -> Path:
        return self.paths.manifest_path(paper_key)

    def exists(self, paper_key: str) -> bool:
        return self.path_for(paper_key).exists()

    def load(self, paper_key: str) -> PipelineManifest | None:
        path = self.path_for(paper_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return PipelineManifest.from_dict(data)
        except Exception as exc:
            raise ManifestStoreError(f"Failed to load manifest: {path}") from exc

    def save(self, manifest: PipelineManifest) -> Path:
        manifest.touch()
        path = self.path_for(manifest.paper_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp_path.write_text(
                json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(path)
            return path
        except Exception as exc:
            raise ManifestStoreError(f"Failed to save manifest: {path}") from exc

    def create(self, paper: PaperRecord) -> PipelineManifest:
        manifest = PipelineManifest.new(
            paper_key=paper.paper_key,
            input_snapshot={
                "input_type": paper.input_type,
                "input_value": paper.input_value,
            },
            paper_snapshot=paper.to_dict(),
        )
        manifest.output_paths = {
            "manifest": str(self.paths.manifest_path(paper.paper_key)),
            "mineru_dir": str(self.paths.mineru_dir(paper.paper_key)),
            "markdown": str(self.paths.markdown_path(paper.paper_key)),
            "captioned_markdown": str(self.paths.captioned_markdown_path(paper.paper_key)),
        }
        self.save(manifest)
        return manifest

    def get_or_create(self, paper: PaperRecord) -> PipelineManifest:
        manifest = self.load(paper.paper_key)
        if manifest is not None:
            return manifest
        return self.create(paper)

    def update_stage(
        self,
        manifest: PipelineManifest,
        stage: PipelineStage,
        status: StageStatus,
        *,
        message: str | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
        increment_attempt: bool = False,
    ) -> PipelineManifest:
        stage_state = manifest.get_stage_state(stage)

        if increment_attempt:
            stage_state.attempts += 1

        if status == StageStatus.IN_PROGRESS and stage_state.started_at is None:
            stage_state.started_at = _utc_now_iso()

        if status in {StageStatus.SUCCEEDED, StageStatus.FAILED, StageStatus.SKIPPED}:
            if stage_state.started_at is None:
                stage_state.started_at = _utc_now_iso()
            stage_state.completed_at = _utc_now_iso()

        stage_state.status = status
        stage_state.message = message
        stage_state.error = error

        if details:
            stage_state.details.update(details)

        manifest.current_stage = stage.value
        manifest.updated_at = _utc_now_iso()

        if status == StageStatus.FAILED:
            manifest.failed_stage = stage.value
            manifest.final_error = error or message
            manifest.batch_status = BatchStatus.FAILED
        elif status == StageStatus.IN_PROGRESS:
            manifest.batch_status = BatchStatus.RUNNING

        self.save(manifest)
        return manifest

    def update_paper_snapshot(
        self,
        manifest: PipelineManifest,
        paper: PaperRecord,
    ) -> PipelineManifest:
        manifest.paper_snapshot = paper.to_dict()
        manifest.updated_at = _utc_now_iso()
        self.save(manifest)
        return manifest

    def mark_completed(self, manifest: PipelineManifest) -> PipelineManifest:
        manifest.batch_status = BatchStatus.SUCCEEDED
        manifest.failed_stage = None
        manifest.final_error = None
        manifest.updated_at = _utc_now_iso()
        self.save(manifest)
        return manifest