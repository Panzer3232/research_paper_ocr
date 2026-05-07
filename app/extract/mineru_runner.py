from __future__ import annotations

import fcntl
import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
import subprocess

from app.config.models import MinerUConfig
from app.core.exceptions import ExtractionError

logger = logging.getLogger(__name__)

_LOCKS_DIR_NAME = ".mineru_locks"


def _lock_path(pdf_path: Path) -> Path:
    digest = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()
    locks_dir = pdf_path.parent / _LOCKS_DIR_NAME
    locks_dir.mkdir(parents=True, exist_ok=True)
    return locks_dir / f"{digest}.lock"


@dataclass(slots=True)
class MinerURunResult:
    pdf_path: str
    output_root: str
    document_dir: str
    markdown_path: str
    images_dir: str | None = None
    middle_json_path: str | None = None
    content_list_path: str | None = None
    layout_pdf_path: str | None = None
    span_pdf_path: str | None = None
    reused_existing: bool = False

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "pdf_path": self.pdf_path,
            "output_root": self.output_root,
            "document_dir": self.document_dir,
            "markdown_path": self.markdown_path,
            "images_dir": self.images_dir,
            "middle_json_path": self.middle_json_path,
            "content_list_path": self.content_list_path,
            "layout_pdf_path": self.layout_pdf_path,
            "span_pdf_path": self.span_pdf_path,
            "reused_existing": self.reused_existing,
        }


class MinerURunner:
    def __init__(self, config: MinerUConfig) -> None:
        self.config = config

    def run(
        self,
        pdf_path: str | Path,
        output_root: str | Path,
        *,
        skip_if_markdown_exists: bool = True,
    ) -> MinerURunResult:
        pdf_path = Path(pdf_path).resolve()
        output_root = Path(output_root).resolve()

        if not pdf_path.exists():
            raise ExtractionError(f"PDF does not exist: {pdf_path}")

        output_root.mkdir(parents=True, exist_ok=True)

        
        existing = self._find_existing_result(pdf_path, output_root)
        if existing is not None and skip_if_markdown_exists:
            existing.reused_existing = True
            return existing

       
        lock_file = _lock_path(pdf_path)
        lock_fd = open(lock_file, "w")
        try:
            logger.debug("Acquiring MinerU lock for %s", pdf_path.name)
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            logger.debug("Lock acquired for %s", pdf_path.name)

            # Re-check after acquiring lock — another process may have
            # completed extraction into this same output_root while we waited.
            existing = self._find_existing_result(pdf_path, output_root)
            if existing is not None and skip_if_markdown_exists:
                existing.reused_existing = True
                return existing

            # Detect and clean up a stale output_root — one that exists but
            # contains no markdown. This happens when a previous MinerU run
            # was killed mid-execution (GPU OOM, process kill, power loss).
            # Leaving a stale directory causes _locate_result to find nothing
            # and raise ExtractionError even though MinerU would succeed if
            # given a clean directory.
            if self._is_stale_output(pdf_path, output_root):
                logger.warning(
                    "Stale MinerU output detected for %s at %s — cleaning up and retrying.",
                    pdf_path.name,
                    output_root,
                )
                self._cleanup_stale_output(output_root)
                output_root.mkdir(parents=True, exist_ok=True)

            result = self._run_mineru(pdf_path, output_root)
            return result

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def _is_stale_output(self, pdf_path: Path, output_root: Path) -> bool:
        if not output_root.exists():
            return False
        existing = self._locate_result(pdf_path, output_root)
        has_any_files = any(output_root.rglob("*"))
        return has_any_files and existing is None

    def _cleanup_stale_output(self, output_root: Path) -> None:
        for child in output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    def _run_mineru(
        self,
        pdf_path: Path,
        output_root: Path,
    ) -> MinerURunResult:
        command = [
            self.config.command,
            "-p",
            str(pdf_path),
            "-o",
            str(output_root),
            "-b",
            self.config.backend,
        ]
        if self.config.extra_args:
            command.extend(self.config.extra_args)

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ExtractionError(
                f"MinerU command not found: {self.config.command}"
            ) from exc
        except Exception as exc:
            raise ExtractionError(
                f"Failed to execute MinerU for {pdf_path}"
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            message = stderr or stdout or f"MinerU exited with code {completed.returncode}"
            raise ExtractionError(message)

        result = self._locate_result(pdf_path, output_root)
        if result is None:
            raise ExtractionError(
                f"MinerU completed but no markdown output was found for: {pdf_path}"
            )
        result.reused_existing = False
        return result

    def _find_existing_result(
        self,
        pdf_path: Path,
        output_root: Path,
    ) -> MinerURunResult | None:
        return self._locate_result(pdf_path, output_root)

    def _locate_result(
        self,
        pdf_path: Path,
        output_root: Path,
    ) -> MinerURunResult | None:
        pdf_stem = pdf_path.stem

        preferred_dirs = [
            output_root / pdf_stem,
            output_root / pdf_stem / "auto",
            output_root / pdf_stem / "hybrid_auto",
            output_root / "auto" / pdf_stem,
            output_root / "auto",
            output_root,
        ]

        for directory in preferred_dirs:
            result = self._result_from_directory(pdf_path, output_root, directory, pdf_stem)
            if result is not None:
                return result

        for markdown_path in output_root.rglob(f"{pdf_stem}.md"):
            directory = markdown_path.parent
            result = self._result_from_directory(pdf_path, output_root, directory, pdf_stem)
            if result is not None:
                return result

        return None

    def _result_from_directory(
        self,
        pdf_path: Path,
        output_root: Path,
        directory: Path,
        pdf_stem: str,
    ) -> MinerURunResult | None:
        if not directory.exists():
            return None

        markdown_candidates = [
            directory / f"{pdf_stem}.md",
            directory / "output.md",
        ]
        markdown_path = next((path for path in markdown_candidates if path.exists()), None)

        if markdown_path is None:
            md_files = sorted(directory.glob("*.md"))
            if len(md_files) == 1:
                markdown_path = md_files[0]

        if markdown_path is None or not markdown_path.exists():
            return None

        images_dir = directory / "images"
        middle_json_path = self._first_existing(
            directory / f"{pdf_stem}_middle.json",
            directory / "middle.json",
        )
        content_list_path = self._first_existing(
            directory / f"{pdf_stem}_content_list.json",
            directory / "content_list.json",
        )
        layout_pdf_path = self._first_existing(
            directory / f"{pdf_stem}_layout.pdf",
            directory / "layout.pdf",
        )
        span_pdf_path = self._first_existing(
            directory / f"{pdf_stem}_span.pdf",
            directory / "span.pdf",
        )

        return MinerURunResult(
            pdf_path=str(pdf_path),
            output_root=str(output_root),
            document_dir=str(directory),
            markdown_path=str(markdown_path),
            images_dir=str(images_dir) if images_dir.exists() else None,
            middle_json_path=str(middle_json_path) if middle_json_path else None,
            content_list_path=str(content_list_path) if content_list_path else None,
            layout_pdf_path=str(layout_pdf_path) if layout_pdf_path else None,
            span_pdf_path=str(span_pdf_path) if span_pdf_path else None,
            reused_existing=False,
        )

    def _first_existing(self, *paths: Path) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None