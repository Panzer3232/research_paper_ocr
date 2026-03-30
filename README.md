# research_paper_ocr

A pipeline for extracting structured Markdown from research paper PDFs using [MinerU](https://github.com/opendatalab/MinerU), with optional GPT image captioning via Azure OpenAI.

**Repository:** https://github.com/Panzer3232/research_paper_ocr

---

## What it does

1. Accepts a single PDF file or a folder of PDF files as input.
2. Runs MinerU OCR on each PDF and produces a clean `.md` file per paper.
3. Optionally captions every figure/image in the markdown using GPT , producing a separate captioned `.md` file.
4. Saves all outputs under `data_ocr/` with a structured layout.
5. Writes a per-paper JSON manifest for resume support — interrupted runs pick up from where they stopped.

---

## Requirements

- Python 3.10 or later
- MinerU system dependencies (CUDA recommended for GPU acceleration — see [MinerU installation guide](https://github.com/opendatalab/MinerU))

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Panzer3232/research_paper_ocr.git
cd research_paper_ocr
```


### 2. Install dependencies

```bash
pip install -U "mineru[all]"
pip install openai python-dotenv
```

Or install everything at once from the requirements file:

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials (for captioning only)

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_azure_openai_key_here
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
```

These are only required when captioning is enabled. Extraction works without them.

---

## Output structure

All outputs are written under `data_ocr/` relative to the working directory (or the path you specify via `output_dir`):

```
data_ocr/
├── markdown/               # Plain MinerU-extracted .md files (one per PDF)
├── captioned_markdown/     # GPT captioned .md files (one per PDF, if captioning enabled)
├── mineru_raw/             # Raw MinerU output: images, middle JSON, layout PDF, span PDF
├── manifests/              # Per-paper JSON manifests (resume + audit trail)
├── input/
└── reports/
```

Each output file is named using a stable key derived from the PDF path, e.g.:
```
ocr__attention__a3f2b1c4d5e6f7a8.md
```

---

## Usage

### As a CLI tool

```bash
# Single PDF, extraction only
python main.py --input /path/to/paper.pdf

# Folder of PDFs, extraction only
python main.py --input /path/to/pdfs/

# Single PDF with captioning enabled
python main.py --input /path/to/paper.pdf --caption

# Folder of PDFs with captioning, custom config
python main.py --input /path/to/pdfs/ --caption --config /path/to/config.json
```

---

### As a Python library

Import from `ocr.py`. The main function is `ocr()`.

#### Extraction only — single PDF

```python
from ocr import ocr

results = ocr("/path/to/paper.pdf")

for r in results:
    if r.success:
        print(r.markdown_path)      # absolute path to .md file
    else:
        print(r.label, r.error)
```

#### Extraction only — folder of PDFs

```python
from ocr import ocr

results = ocr("/path/to/pdfs/")

for r in results:
    if r.success:
        print(r.label, "→", r.markdown_path)
```

#### Extraction + GPT captioning

```python
from ocr import ocr

results = ocr("/path/to/pdfs/", caption=True)

for r in results:
    if r.success:
        print(r.markdown_path)      # plain MinerU .md
        print(r.captioned_path)     # captioned .md (None if captioning was skipped)
```

#### Using `effective_markdown()` — get the best available output

```python
from ocr import ocr

results = ocr("/path/to/pdfs/", caption=True)

for r in results:
    if r.success:
        md = r.effective_markdown()
        # Returns captioned_path if captioning ran, otherwise markdown_path
        print(md)
```

#### With explicit output directory

```python
from ocr import ocr

results = ocr(
    "/path/to/pdfs/",
    caption=True,
    output_dir="/absolute/path/to/output/",
)
```

#### With explicit config file

```python
from ocr import ocr

results = ocr(
    "/path/to/pdfs/",
    config_path="/absolute/path/to/config.json",
    caption=True,
)
```

#### With a pre-built PipelineConfig (advanced)

```python
from ocr import ocr
from app.config.loader import load_config

config = load_config("/absolute/path/to/config.json")
config.captioning.enabled = True
config.output.root_dir = "/absolute/path/to/output/"

results = ocr("/path/to/pdfs/", config=config)
```

#### Handling failures

```python
from ocr import ocr

results = ocr("/path/to/pdfs/")

succeeded = [r for r in results if r.success]
failed    = [r for r in results if not r.success]

for r in failed:
    print(f"FAILED: {r.label} | status={r.status} | error={r.error}")
```

---

## `OCRPipelineResult` reference

Every call to `ocr()` returns a `list[OCRPipelineResult]`, one item per PDF.

| Field | Type | Description |
|---|---|---|
| `pdf_path` | `str` | Absolute path to the source PDF |
| `success` | `bool` | `True` when all enabled stages completed without error |
| `markdown_path` | `str \| None` | Absolute path to the MinerU-extracted `.md` file. `None` only if extraction itself failed |
| `captioned_path` | `str \| None` | Absolute path to the GPT-5 captioned `.md` file. `None` if captioning was disabled, skipped, or failed |
| `error` | `str \| None` | Error message when `success=False`, otherwise `None` |
| `status` | `str` | `"completed"`, `"failed_extraction"`, `"failed_captioning"`, or `"failed_pipeline_error"` |
| `label` | `str` | Human-readable PDF filename stem |

**Method:**

| Method | Returns | Description |
|---|---|---|
| `effective_markdown()` | `str \| None` | Returns `captioned_path` if available, else `markdown_path` |

---

## Configuration

The pipeline is configured via `config.json`. All fields have defaults — you only need to override what you want to change.

---

## Resume support

The pipeline writes a JSON manifest per PDF under `data_ocr/manifests/`. If a run is interrupted:

- Re-running with the same input automatically skips already-completed stages.
- Only the failed or incomplete stage is retried.
- Set `resume.enabled: false` in config to force a full re-run.

---
