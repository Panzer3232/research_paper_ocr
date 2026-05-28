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
8. Provides chunks for vector db specially designed for research paper.

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

## RAG Chunking Pipeline

The RAG chunking pipeline converts the structured JSON produced by the OCR stage into retrieval-ready chunks. It runs as a separate, independent stage after OCR completes. Each chunk is self-contained and carries full provenance metadata, making it suitable for direct ingestion into any vector store.

The pipeline has three sequential steps: **prepare** (normalize and filter), **chunk** (split by content type), and **evaluate** (optional quality audit).

---

### Output structure

When chunking runs, all output is written under `data_ocr/rag_chunks/`:

```
data_ocr/
└── rag_chunks/
    ├── normalized/          # Filtered, normalized JSON — one file per paper
    │   └── _prepare_report.json
    ├── chunks/              # Final chunk JSON — one file per paper
    │   └── _chunk_report.json
    ├── evaluation/          # Per-paper quality reports (only when evaluate=True)
    │   └── _evaluation_report.json
    └── logs/
        ├── 01_prepare.log
        ├── 02_chunk.log
        └── 03_evaluate.log  # Only when evaluate=True
```

---

### Chunk types

The pipeline treats prose, equations, tables, and figures as structurally distinct and applies separate assembly logic to each. They are never mixed within a single chunk.

**Prose** — consecutive text blocks within a section are joined and split by a recursive character splitter (Chonkie `RecursiveChunker` with `OverlapRefinery` if available, deterministic paragraph-based fallback otherwise). Short prose fragments below `min_prose_tokens` are merged with an adjacent prose chunk in the same section rather than emitted as standalone noise.

**Equation** — each equation becomes its own chunk. The chunk text includes the raw LaTeX string, the section path for context, and up to `max_context_blocks` surrounding prose blocks on each side. The surrounding prose is included because equations without any textual context are retrievable in name only — the context explains what the equation computes and why. Quality flags distinguish between equations with full bilateral context, one-sided context, and no context at all.

**Table** — each table becomes one or more chunks depending on size. If the rendered HTML exceeds `table_chunk_size` tokens, the table is split row-wise with the caption and header row repeated in every part so each chunk is self-contained. Caption recovery searches the `table_caption` field, then `table_footnote`, then nearby text blocks matching a `"Table N: ..."` pattern in both directions. Nearby in-text table references are included in the chunk text.

**Figure** — consecutive image blocks in the same section are grouped into a single figure chunk. Caption recovery follows the same multi-source strategy as tables. If GPT captioning was enabled during OCR, the `caption_llm` field from each image block is included in the chunk text as an image summary, giving the vector store a natural-language description of visual content that would otherwise be unretrievable.

---

### Metadata and chunk IDs

Every chunk carries a deterministic `chunk_id` of the form `{paper_key}_{ordinal:05d}_{chunk_type}` — stable across re-runs on the same input, which means re-indexing a corrected paper does not produce phantom duplicates in a vector store that deduplicates by ID.

Each chunk's metadata includes:

- `paper_key`, `source_pdf`, `paper_title`, `venue`, `venue_year`, `authors_raw` — paper-level provenance for filtered retrieval without a separate metadata lookup.
- `section_path` — full hierarchy from root to the containing section, e.g. `["3 Methodology", "3.2 Training Objective"]`. Enables section-scoped retrieval and reranking by position in the paper.
- `page_span` — `{start, end}` page indices of the source blocks, useful for returning page-anchored citations.
- `source_block_ids` and `source_block_indices` — trace every chunk back to the exact MinerU blocks it was assembled from.
- `relations` — explicit links to the previous and next chunks globally, previous and next chunks within the same section, and cross-type block references (e.g. which text blocks surround an equation, which text blocks reference a table). These relations allow a retrieval system to fetch neighboring context at query time without re-embedding.
- `quality_flags` — a list of string tags written during chunking that describe structural anomalies (e.g. `equation_one_sided_context`, `table_caption_inferred`, `figure_without_llm_caption`, `short_prose_merged_with_next`). These are non-blocking — the chunk is still emitted — but they are surfaced in the evaluation report so problematic chunks can be identified and filtered downstream.

---

### Hyperparameters

All hyperparameters are set under the `rag_chunking` key in `config.json` and apply to every paper in the run. They can be tuned without touching any code.

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | `640` | Target token size for prose chunks. Prose is split at paragraph boundaries up to this limit. |
| `overlap` | `64` | Token overlap between adjacent prose chunks from the same block group, implemented via Chonkie `OverlapRefinery`. |
| `table_chunk_size` | `768` | Token threshold above which a table is split row-wise. Tables at or below this size are emitted as a single chunk. |
| `min_prose_tokens` | `80` | Prose chunks below this size are merged with an adjacent same-section chunk rather than emitted standalone. |
| `max_context_blocks` | `2` | Maximum number of surrounding prose blocks included in each equation chunk on each side. |
| `split_long_tables` | `true` | When `false`, oversized tables are emitted as a single chunk regardless of size. Set to `false` only if your vector store has a hard token limit that exceeds `table_chunk_size`. |

```json
"rag_chunking": {
  "structured_json_input_dir": null,
  "chunk_size": 640,
  "overlap": 64,
  "table_chunk_size": 768,
  "min_prose_tokens": 80,
  "max_context_blocks": 1,
  "split_long_tables": true
}
```

Set `structured_json_input_dir` to a path if you want `chunk()` to resolve the input directory from config rather than passing it as an argument.

---

### Quality evaluation

When `evaluate=True`, the pipeline runs a structural audit on every chunk file after chunking completes. Each paper receives an `overall_status` of `PASS`, `WARN`, or `FAIL`.

**FAIL conditions** (hard problems that indicate data loss or structural corruption): empty chunks, missing `section_path`, missing `source_block_ids`, missing `relations`, reference section leaks into chunk content, checklist text leaks, equations with no LaTeX string, equations with no surrounding context at all.

**WARN conditions** (soft problems worth reviewing but not blocking): duplicate chunk text, prose chunks below `min_prose_tokens`, prose chunks exceeding `chunk_size` by more than 15%, prose that does not read as natural sentences, tables with inferred or missing captions, figures without a GPT caption when captioning was expected.

The aggregate evaluation report at `rag_chunks/evaluation/_evaluation_report.json` contains status counts across all papers and per-paper breakdowns with example chunk IDs for every flagged condition.

---

### Running chunking via Python

Chunking is exposed through `chunk()` in `ocr.py`. It is fully independent of `ocr()` — call it on any folder of structured JSON files, whether produced by this pipeline or another source.

```python
from ocr import ocr, chunk

# Step 1: OCR — export_json=True is required for chunking to have input
ocr_results = ocr("/path/to/pdfs/", export_json=True)

# Step 2: chunk — reads from the structured_json directory produced above
chunk_results = chunk(structured_json_dir="data_ocr/structured_json/")

for r in chunk_results:
    if r.success:
        print(r.paper_key, "→", r.chunk_json_path)
    else:
        print(r.paper_key, "FAILED:", r.error)
```

With evaluation and a custom output directory:

```python
chunk_results = chunk(
    structured_json_dir="data_ocr/structured_json/",
    evaluate=True,
    output_dir="/absolute/path/to/output/",
)
```

With a custom `ChunkConfig`:

```python
from ocr import chunk
from rag_chunking.common import ChunkConfig

chunk_results = chunk(
    structured_json_dir="data_ocr/structured_json/",
    chunk_config=ChunkConfig(chunk_size=512, overlap=32, min_prose_tokens=60),
)
```

Input path can also be set in `config.json` under `rag_chunking.structured_json_input_dir` and omitted from the call entirely:

```python
chunk_results = chunk(config_path="/path/to/config.json")
```

`chunk()` raises `ValueError` with a descriptive message if no input path is resolvable from either source.

#### `RAGChunkResult` reference

Every call to `chunk()` returns a `list[RAGChunkResult]`, one item per JSON file.

| Field | Type | Description |
|---|---|---|
| `paper_key` | `str` | Filename stem of the source JSON |
| `source_json_path` | `str` | Absolute path to the input structured JSON |
| `success` | `bool` | `True` when all enabled stages completed without error |
| `normalized_json_path` | `str \| None` | Prepare step output. `None` if prepare failed |
| `chunk_json_path` | `str \| None` | Chunk step output. `None` if chunk failed or was not reached |
| `evaluation_report_path` | `str \| None` | Evaluation output. `None` if `evaluate=False` or a prior stage failed |
| `error` | `str \| None` | Error message when `success=False`, otherwise `None` |
| `status` | `str` | `"completed"`, `"failed_prepare"`, `"failed_chunk"`, or `"failed_evaluate"` |

---

### Running chunking via the FastAPI backend

The API accepts `chunk` and `evaluate` as independent boolean fields alongside the existing OCR fields. All flags default to `False`.

```bash
# OCR only — existing behaviour, unchanged
curl -X POST http://localhost:8002/jobs \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/absolute/path/to/pdfs/"}'

# OCR + caption (independent of chunking)
curl -X POST http://localhost:8002/jobs \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/absolute/path/to/pdfs/", "caption": true}'

# OCR + chunking (export_json is forced true internally)
curl -X POST http://localhost:8002/jobs \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/absolute/path/to/pdfs/", "chunk": true}'

# OCR + caption + chunking + evaluation
curl -X POST http://localhost:8002/jobs \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/absolute/path/to/pdfs/", "caption": true, "chunk": true, "evaluate": true}'
```

`caption` and `chunk` are independent flags. Setting `caption=true` without `chunk=true` runs GPT captioning and stops — no chunking occurs. Setting `chunk=true` without `caption=true` runs OCR and chunking without captioning. Setting both runs all stages sequentially in a single process, and figure chunks will contain GPT image summaries (`caption_llm`) in their text.

`evaluate` has no effect when `chunk=false`.

The job result file at `data_ocr/job_results/{job_id}.json` contains a top-level `chunking` key alongside the existing OCR `results` key:

```json
{
  "succeeded": 3,
  "failed": 0,
  "results": [ ... ],
  "chunking": {
    "ran": true,
    "succeeded": 3,
    "failed": 0,
    "results": [
      {
        "paper_key": "...",
        "success": true,
        "status": "completed",
        "normalized_json_path": "...",
        "chunk_json_path": "...",
        "evaluation_report_path": null
      }
    ]
  }
}
```

When `chunk=false`, `chunking` is `{"ran": false}` — present but clearly skipped. Existing consumers of the result file that do not inspect the `chunking` key are unaffected.