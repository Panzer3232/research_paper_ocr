from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.stages import PIPELINE_STAGE_ORDER, PipelineStage
from app.state.status import BatchStatus, StageStatus


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class StageState:
    stage: PipelineStage
    status: StageStatus = StageStatus.NOT_STARTED
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    message: str | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageState":
        return cls(
            stage=PipelineStage(data["stage"]),
            status=StageStatus(data["status"]),
            attempts=int(data.get("attempts", 0)),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            message=data.get("message"),
            error=data.get("error"),
            details=dict(data.get("details") or {}),
        )


@dataclass(slots=True)
class PipelineManifest:
    manifest_version: int
    paper_key: str
    created_at: str
    updated_at: str

    batch_status: BatchStatus = BatchStatus.NOT_STARTED
    current_stage: str | None = None
    failed_stage: str | None = None
    final_error: str | None = None

    input_snapshot: dict[str, Any] = field(default_factory=dict)
    paper_snapshot: dict[str, Any] = field(default_factory=dict)
    selected_source: dict[str, Any] = field(default_factory=dict)
    output_paths: dict[str, str] = field(default_factory=dict)
    stage_states: dict[str, StageState] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(
        cls,
        *,
        paper_key: str,
        input_snapshot: dict[str, Any] | None = None,
        paper_snapshot: dict[str, Any] | None = None,
    ) -> "PipelineManifest":
        now = utc_now_iso()
        stage_states = {
            stage.value: StageState(stage=stage) for stage in PIPELINE_STAGE_ORDER
        }
        return cls(
            manifest_version=1,
            paper_key=paper_key,
            created_at=now,
            updated_at=now,
            input_snapshot=dict(input_snapshot or {}),
            paper_snapshot=dict(paper_snapshot or {}),
            stage_states=stage_states,
        )

    def ensure_all_stage_entries(self) -> None:
        for stage in PIPELINE_STAGE_ORDER:
            self.stage_states.setdefault(stage.value, StageState(stage=stage))

    def get_stage_state(self, stage: PipelineStage) -> StageState:
        self.ensure_all_stage_entries()
        return self.stage_states[stage.value]

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        self.ensure_all_stage_entries()
        return {
            "manifest_version": self.manifest_version,
            "paper_key": self.paper_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "batch_status": self.batch_status.value,
            "current_stage": self.current_stage,
            "failed_stage": self.failed_stage,
            "final_error": self.final_error,
            "input_snapshot": self.input_snapshot,
            "paper_snapshot": self.paper_snapshot,
            "selected_source": self.selected_source,
            "output_paths": self.output_paths,
            "stage_states": {
                key: value.to_dict() for key, value in self.stage_states.items()
            },
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineManifest":
        stage_states = {
            key: StageState.from_dict(value)
            for key, value in (data.get("stage_states") or {}).items()
        }
        manifest = cls(
            manifest_version=int(data.get("manifest_version", 1)),
            paper_key=data["paper_key"],
            created_at=data["created_at"],
            updated_at=data.get("updated_at") or data["created_at"],
            batch_status=BatchStatus(data.get("batch_status", BatchStatus.NOT_STARTED.value)),
            current_stage=data.get("current_stage"),
            failed_stage=data.get("failed_stage"),
            final_error=data.get("final_error"),
            input_snapshot=dict(data.get("input_snapshot") or {}),
            paper_snapshot=dict(data.get("paper_snapshot") or {}),
            selected_source=dict(data.get("selected_source") or {}),
            output_paths=dict(data.get("output_paths") or {}),
            stage_states=stage_states,
            stats=dict(data.get("stats") or {}),
        )
        manifest.ensure_all_stage_entries()
        return manifest