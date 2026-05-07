# research_paper_ocr

A pipeline for extracting structured Markdown and RAG-ready JSON from research paper PDFs using [MinerU](https://github.com/opendatalab/MinerU), with optional GPT image captioning via Azure OpenAI.

**Repository:** https://github.com/Panzer3232/research_paper_ocr

---

## What it does

1. Accepts a single PDF file or a folder of PDF files as input.
2. Runs MinerU OCR on each PDF and produces a clean `.md` file per paper.
3. Optionally captions every figure/image in the markdown using GPT, producing a separate captioned `.md` file.
4. Optionally exports a structured  `.json` file per paper, grouped by section with page metadata. If captioning is also enabled, GPT-generated captions are embedded into image blocks as `caption_llm`.
5. Saves all outputs under `data_ocr/` with a structured layout.
6. Writes a per-paper JSON manifest for resume support — interrupted runs pick up from where they stopped.
7. Exposes a FastAPI backend for submitting multiple independent batch jobs simultaneously on a shared GPU server.

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

Create a `.env` file in the same directory as your `config.json`:

```
OPENAI_API_KEY=your_azure_openai_key_here
OPENAI_BASE_URL=https://your-resource.openai.azure.com/
```

These are only required when `--caption` is enabled. Extraction and JSON export work without them.

---

## Output structure

All outputs are written under `data_ocr/` relative to the working directory (or the path you specify via `output_dir`):

```
data_ocr/
├── markdown/               # Plain MinerU-extracted .md files (one per PDF)
├── captioned_markdown/     # GPT-captioned .md files (one per PDF, if --caption enabled)
├── structured_json/        # structured .json files (one per PDF, if --export-json enabled)
├── mineru_raw/             # Raw MinerU output: images, content_list.json, middle.json, layout PDF
├── manifests/              # Per-paper JSON manifests (resume + audit trail)
├── input/
└── reports/
```

Each output file is named using a stable key derived from the PDF path, e.g.:
```
ocr__attention__a3f2b1c4d5e6f7a8.md
ocr__attention__a3f2b1c4d5e6f7a8.json
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

# Export structured RAG-ready JSON alongside markdown
python main.py --input /path/to/pdfs/ --export-json

# Captioning and JSON export together
python main.py --input /path/to/pdfs/ --caption --export-json

# Custom config file
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
        print(r.markdown_path)
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
        print(r.captioned_path)     # GPT-captioned .md (None if captioning was skipped)
```

#### Extraction + structured JSON export

```python
from ocr import ocr

results = ocr("/path/to/pdfs/", export_json=True)

for r in results:
    if r.success:
        print(r.markdown_path)
        print(r.structured_json_path)   
```

#### Captioning + structured JSON export together

When both are enabled, GPT captions are embedded into image blocks in the JSON as `caption_llm` in addition to the paper's original figure caption.

```python
from ocr import ocr

results = ocr("/path/to/pdfs/", caption=True, export_json=True)

for r in results:
    if r.success:
        print(r.markdown_path)
        print(r.captioned_path)
        print(r.structured_json_path)
```

#### Using `effective_markdown()` — get the best available output

```python
from ocr import ocr

results = ocr("/path/to/pdfs/", caption=True)

for r in results:
    if r.success:
        md = r.effective_markdown()
        
        print(md)
```

#### With explicit output directory

```python
from ocr import ocr

results = ocr(
    "/path/to/pdfs/",
    caption=True,
    export_json=True,
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
    export_json=True,
)
```

#### With a pre-built PipelineConfig (advanced)

```python
from ocr import ocr
from app.config.loader import load_config

config = load_config("/absolute/path/to/config.json")
config.captioning.enabled = True
config.structured_json.enabled = True
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

#### With logging enabled

```python
from ocr import ocr, enable_logging

enable_logging()

results = ocr("/path/to/pdfs/", caption=True, export_json=True)

for r in results:
    if r.success:
        print(r.markdown_path)
    else:
        print(r.label, r.error)
```

---

## `OCRPipelineResult` reference

Every call to `ocr()` returns a `list[OCRPipelineResult]`, one item per PDF.

| Field | Type | Description |
|---|---|---|
| `pdf_path` | `str` | Absolute path to the source PDF |
| `success` | `bool` | `True` when all enabled stages completed without error |
| `markdown_path` | `str \| None` | Absolute path to the MinerU-extracted `.md` file. `None` only if extraction itself failed |
| `captioned_path` | `str \| None` | Absolute path to the GPT-captioned `.md` file. `None` if captioning was disabled, skipped, or failed |
| `structured_json_path` | `str \| None` | Absolute path to the RAG-ready structured `.json` file. `None` if `export_json` was not requested or `content_list.json` was not produced by MinerU |
| `error` | `str \| None` | Error message when `success=False`, otherwise `None` |
| `status` | `str` | `"completed"`, `"failed_extraction"`, `"failed_captioning"`, or `"failed_pipeline_error"` |
| `label` | `str` | Human-readable PDF filename stem |

**Method:**

| Method | Returns | Description |
|---|---|---|
| `effective_markdown()` | `str \| None` | Returns `captioned_path` if available, else `markdown_path` |

---

## Structured JSON format

When `--export-json` is used, each PDF produces a `.json` file under `data_ocr/structured_json/`. It is built from MinerU's native `content_list.json` — no custom Markdown parsing is needed for the primary content. The structurer groups MinerU's flat block list into logical sections based on heading blocks, then enriches image blocks with captions from both MinerU and GPT.

### How sections are built
 
A new section opens whenever MinerU emits a `text` block with a `text_level` field (a heading). All content blocks that follow are accumulated into that section until the next heading. Content that appears before the first heading is placed into a `[Preamble]` section at level 0. `discarded` blocks from MinerU are dropped entirely.
 
### How captions are resolved
 
**Images** — caption resolved from `image_caption` or `image_footnote` fields in the MinerU block. If captioning was enabled, `caption_llm` is added from the GPT alt-text written into the captioned markdown file.
 
**Tables** — caption resolved in priority order: `table_caption` field → `table_footnote` field → immediately preceding text block matching `"Table N: ..."` pattern (lookbehind) → immediately following text block matching the same pattern (lookahead). The consumed caption text block is removed from the section body so it does not appear twice.
 
**Equations** — emitted with `text` (LaTeX string if available), optional `text_format` (e.g. `"latex"`), and optional `img_path` (rendered fallback image from MinerU).
 
### Missing image recovery
 
MinerU occasionally omits image blocks from `content_list.json` while still writing the image tags into the markdown. The structurer detects this by parsing the captioned markdown (or plain markdown if captioning is disabled), matching images by section heading, and injecting any missing image blocks into the correct section. This ensures no figures are silently lost from the JSON output.

```json
{
  "metadata": {
    "paper_key": "ocr__attention__a3f2b1c4d5e6f7a8",
    "source_pdf": "attention_is_all_you_need",
    "total_pages": 15,
    "total_sections": 12,
    "llm_captioned": false
  },
  "sections": [
    {
      "section_title": "Abstract",
      "section_level": 1,
      "page_start": 0,
      "page_end": 0,
      "content": [
        {
          "type": "text",
          "text": "We propose a new simple network architecture...",
          "page_idx": 0
        },
        {
          "type": "image",
          "img_path": "images/abc123.jpg",
          "caption": "Figure 1: The Transformer model architecture.",
          "caption_llm": "Encoder-decoder diagram with multi-head attention blocks connected by residual layers.",
          "page_idx": 0
        }
      ]
    }
  ]
}
```

`caption_llm` is only present on image blocks when `--caption` and `--export-json` are both enabled. `caption` is always the original figure caption from the paper.

---

## Configuration

The pipeline is configured via `config.json`. All fields have defaults — you only need to override what you want to change.

Key sections:

```json
{
  "captioning": {
    "enabled": false
  },
  "structured_json": {
    "enabled": false
  },
  "output": {
    "root_dir": "data_ocr"
  }
}
```

Both `captioning.enabled` and `structured_json.enabled` are overridden by their respective CLI flags (`--caption`, `--export-json`) or `ocr()` parameters. You do not need to edit `config.json` to use these features.

---

## Resume support

The pipeline writes a JSON manifest per PDF under `data_ocr/manifests/`. If a run is interrupted:

- Re-running with the same input automatically skips already-completed stages.
- Only the failed or incomplete stage is retried.
- Set `resume.enabled: false` in `config.json` to force a full re-run.

---

## FastAPI Backend

The FastAPI backend allows multiple independent batch jobs to run simultaneously on a shared GPU server. Each job runs as a completely separate OS process with its own Python interpreter, memory, and MinerU instance. The server manages process count — GPU scheduling is left entirely to the server.
 
### Additional requirements

If installed through requirements.txt then okay otherwise install these.

```bash
pip install fastapi "uvicorn[standard]"
```
 
### Start the server
 
```bash
MAX_CONCURRENT_PROCESSES=2 uvicorn api:app --host 0.0.0.0 --port 8002
```
 
With captioning credentials:
 
```bash
OPENAI_API_KEY=your_key \
OPENAI_BASE_URL=https://your-resource.openai.azure.com/ \
MAX_CONCURRENT_PROCESSES=2 \
uvicorn api:app --host 0.0.0.0 --port 8002
```
 
`MAX_CONCURRENT_PROCESSES` controls how many jobs run simultaneously. Default is `2`. Each job runs MinerU papers sequentially within its own process, so at most `N` MinerU instances run on the GPU at any given time.
 
### Endpoints
 
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Active process count and capacity status |
| `POST` | `/jobs` | Submit a new batch job |
| `GET` | `/jobs/{job_id}` | Poll status of a specific job |
| `GET` | `/jobs` | List all jobs in the current session |
| `PATCH` | `/config` | Update `max_concurrent_processes` at runtime |
 
### Submit a job
 
```bash
# Extraction only
curl -X POST http://localhost:8002/jobs \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/absolute/path/to/pdfs/"}'
 
# With captioning and JSON export
curl -X POST http://localhost:8002/jobs \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/absolute/path/to/pdfs/", "caption": true, "export_json": true}'
 
# With per-job hyperparameter overrides (config.json is never modified)
curl -X POST http://localhost:8002/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "input_path": "/absolute/path/to/pdfs/",
    "caption": true,
    "export_json": true,
    "overrides": {
      "captioning.timeout_seconds": 300,
      "captioning.max_retries": 3
    }
  }'
```
 
### Poll status
 
```bash
curl http://localhost:8002/jobs/{job_id}
curl http://localhost:8002/jobs
curl http://localhost:8002/health
```
 
### Checking job results and conversion counts
 
After a job completes, always check `data_ocr/job_results/{job_id}.json` for the full per-paper breakdown. This file is written by the worker process on completion and contains succeeded/failed counts, output paths, and error messages for every PDF in the batch. Key fields to check: The `total`, `succeeded`, and `failed` fields immediately tell you how many PDFs were converted. Individual `error` fields show what went wrong for any failed paper.
 
### Duplicate PDFs across concurrent jobs
 
> **Important:** If the same PDF filename exists in two different folders submitted as separate concurrent jobs, both processes will process it independently because the `paper_key` is derived from the full resolved path — different folders produce different keys.
 
To avoid redundant processing and GPU contention, **do not include the same PDF filename in two folders submitted simultaneously**. Each folder submitted to a concurrent job should contain a unique set of PDFs. The `fcntl` lock in `mineru_runner.py` serializes MinerU runs on the same filename stem across processes (preventing GPU crash), but it does not skip the second job's processing entirely — it will still run MinerU and produce a separate output under a different `paper_key`.
 
If a paper is accidentally processed twice, its two outputs will have different `paper_key` hashes and both will exist on disk. To clean up, identify the duplicate manifests:

Keep the one from the intended run and delete the other along with its corresponding output files.
 
### Recovering failed papers
 
If a paper fails due to GPU interruption, its manifest records the failure and resume logic will skip it on resubmission. To force a retry, delete its manifest first:
 
```bash
ls data_ocr/manifests/ | grep "paper_filename_stem"
rm data_ocr/manifests/{paper_key}.json
```
 
Then resubmit the job.
 
---