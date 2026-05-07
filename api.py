from __future__ import annotations

import json
import multiprocessing
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api_config import APIConfig



@dataclass
class JobEntry:
    job_id: str
    input_path: str
    output_dir: str | None
    caption: bool
    export_json: bool
    config_overrides: dict
    submitted_at: str
    process: multiprocessing.Process
    result_path: str
    status: str = "running"
    finished_at: str | None = None
    result: dict | None = None



class AppState:
    def __init__(self, api_config: APIConfig) -> None:
        self.api_config = api_config
        self.max_concurrent: int = api_config.max_concurrent_processes
        self._jobs: dict[str, JobEntry] = {}
        self._lock = threading.Lock()

    def active_count(self) -> int:
        with self._lock:
            return sum(
                1 for entry in self._jobs.values()
                if entry.status == "running" and entry.process.is_alive()
            )

    def register(self, entry: JobEntry) -> None:
        with self._lock:
            self._jobs[entry.job_id] = entry

    def get(self, job_id: str) -> JobEntry | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[dict]:
        with self._lock:
            return [_entry_to_dict(e) for e in self._jobs.values()]

    def refresh_status(self, entry: JobEntry) -> None:
        """
        Sync the entry's status from the child process state and result file.
        Mutates entry in-place. Must be called under no lock since it does I/O.
        """
        if entry.status != "running":
            return

        if entry.process.is_alive():
            return

        finished_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        result_file = Path(entry.result_path)

        if result_file.exists():
            try:
                result = json.loads(result_file.read_text(encoding="utf-8"))
            except Exception:
                result = None
        else:
            result = None

        with self._lock:
            if entry.process.exitcode == 0 and result and result.get("success"):
                entry.status = "succeeded"
            else:
                entry.status = "failed"
            entry.finished_at = finished_at
            entry.result = result


_app_state: AppState | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_state
    api_config = APIConfig.from_env()
    multiprocessing.set_start_method("fork", force=True)
    _app_state = AppState(api_config)
    yield



app = FastAPI(
    title="research_paper_ocr API",
    version="1.0.0",
    description=(
        "Submit OCR pipeline jobs for research paper PDFs. "
        "Each job runs as an independent OS process. "
        "If the server is at capacity, resubmit when GPU is free."
    ),
    lifespan=lifespan,
)


def _state() -> AppState:
    if _app_state is None:
        raise RuntimeError("AppState not initialised")
    return _app_state



class JobSubmitRequest(BaseModel):
    input_path: str = Field(..., description="Absolute path to a PDF file or directory of PDFs.")
    output_dir: str | None = Field(None, description="Override output root directory for this job.")
    caption: bool = Field(False, description="Enable GPT image captioning.")
    export_json: bool = Field(False, description="Export structured RAG-ready JSON.")
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-job hyperparameter overrides applied on top of the base config.json. "
            "Keys use dotted notation, e.g. 'captioning.timeout_seconds', 'mineru.backend'. "
            "config.json on disk is never modified."
        ),
    )


class CapacityPatchRequest(BaseModel):
    max_concurrent_processes: int = Field(..., ge=1, le=32)



def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _entry_to_dict(entry: JobEntry) -> dict:
    return {
        "job_id": entry.job_id,
        "status": entry.status,
        "input_path": entry.input_path,
        "output_dir": entry.output_dir,
        "caption": entry.caption,
        "export_json": entry.export_json,
        "config_overrides": entry.config_overrides,
        "submitted_at": entry.submitted_at,
        "finished_at": entry.finished_at,
        "pid": entry.process.pid,
        "result": entry.result,
    }


def _result_file_path(job_id: str) -> str:
    results_dir = Path(__file__).parent / "data_ocr" / "job_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir / f"{job_id}.json")


@app.get("/health", tags=["meta"])
def health():
    state = _state()
    active = state.active_count()
    return {
        "status": "ok",
        "active_processes": active,
        "max_concurrent_processes": state.max_concurrent,
        "capacity_available": active < state.max_concurrent,
    }


@app.post("/jobs", status_code=202, tags=["jobs"])
def submit_job(body: JobSubmitRequest):
    state = _state()

    input_path = Path(body.input_path)
    if not input_path.exists():
        raise HTTPException(status_code=422, detail=f"input_path does not exist: {input_path}")

    active = state.active_count()
    if active >= state.max_concurrent:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    f"Pipeline at capacity: {active}/{state.max_concurrent} processes running. "
                    "Come back when GPU is free."
                ),
                "active_processes": active,
                "max_concurrent_processes": state.max_concurrent,
            },
        )

    job_id = str(uuid.uuid4())
    result_path = _result_file_path(job_id)

    overrides = dict(body.overrides)
    overrides["_base_config_path"] = str(state.api_config.base_config_path)

    output_dir = body.output_dir or state.api_config.default_output_dir

    process = multiprocessing.Process(
        target=_worker_target,
        args=(job_id, str(body.input_path), result_path, body.caption, body.export_json, output_dir, overrides),
        name=f"ocr-job-{job_id[:8]}",
        daemon=True,
    )
    process.start()

    entry = JobEntry(
        job_id=job_id,
        input_path=str(body.input_path),
        output_dir=output_dir,
        caption=body.caption,
        export_json=body.export_json,
        config_overrides=body.overrides,
        submitted_at=_utc_now(),
        process=process,
        result_path=result_path,
    )
    state.register(entry)

    return {
        "job_id": job_id,
        "status": "running",
        "pid": process.pid,
        "submitted_at": entry.submitted_at,
        "message": "Job accepted. Poll GET /jobs/{job_id} for status.",
    }


@app.get("/jobs/{job_id}", tags=["jobs"])
def get_job(job_id: str):
    state = _state()
    entry = state.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    state.refresh_status(entry)
    return _entry_to_dict(entry)


@app.get("/jobs", tags=["jobs"])
def list_jobs():
    state = _state()
    jobs = state.all_jobs()

    for entry in list(state._jobs.values()):
        state.refresh_status(entry)

    return {
        "total": len(jobs),
        "active": state.active_count(),
        "max_concurrent_processes": state.max_concurrent,
        "jobs": state.all_jobs(),
    }


@app.patch("/config", tags=["meta"])
def update_capacity(body: CapacityPatchRequest):
    """
    Update max_concurrent_processes at runtime without restarting uvicorn.
    Change is in-memory only — reverts to the environment variable value on restart.
    """
    state = _state()
    old = state.max_concurrent
    state.max_concurrent = body.max_concurrent_processes
    return {
        "max_concurrent_processes": state.max_concurrent,
        "previous": old,
        "note": "In-memory change only. Restart uvicorn to revert to environment default.",
    }



def _worker_target(
    job_id: str,
    input_path: str,
    result_path: str,
    caption: bool,
    export_json: bool,
    output_dir: str | None,
    config_overrides: dict,
) -> None:
    from api_worker import run_ocr_job
    run_ocr_job(
        job_id=job_id,
        input_path=input_path,
        result_path=result_path,
        caption=caption,
        export_json=export_json,
        output_dir=output_dir,
        config_overrides=config_overrides,
    )