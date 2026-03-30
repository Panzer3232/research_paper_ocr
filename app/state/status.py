from __future__ import annotations

from enum import Enum


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.SKIPPED}


class BatchStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PARTIAL_SUCCESS = "partial_success"
    SUCCEEDED = "succeeded"
    FAILED = "failed"