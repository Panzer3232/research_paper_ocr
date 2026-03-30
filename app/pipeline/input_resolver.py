from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import InputParseError


@dataclass(slots=True)
class OCRJob:
 

    paper_key: str
    pdf_path: Path
    label: str


def _key_from_path(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    return f"ocr__{path.stem}__{digest}"


def _job_from_pdf(path: Path) -> OCRJob:
    resolved = path.resolve()
    if not resolved.exists():
        raise InputParseError(f"PDF file not found: {resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise InputParseError(f"Expected a .pdf file, got: {resolved}")
    return OCRJob(
        paper_key=_key_from_path(resolved),
        pdf_path=resolved,
        label=resolved.stem,
    )


def resolve_ocr_inputs(input_path: str | Path) -> list[OCRJob]:
   
    path = Path(input_path).resolve()

    if not path.exists():
        raise InputParseError(f"Input path does not exist: {path}")

    if path.is_file():
        return [_job_from_pdf(path)]

    if path.is_dir():
        pdf_files = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        )
        if not pdf_files:
            raise InputParseError(f"No PDF files found in directory: {path}")
        return [_job_from_pdf(p) for p in pdf_files]

    raise InputParseError(f"Input path is neither a file nor a directory: {path}")