# Receipt, Invoice, and Business Card Extraction

An image extraction service built with shared RapidOCR, document classification, separate Receipt/Invoice and Business Card pipelines, Ollama-compatible LLM extraction, Pydantic validation, and an ARQ worker.

## Current workflow

```text
Image upload
    |
    v
FastAPI upload validation and temporary file
    |
    v
Redis / ARQ process_document job
    |
    v
app.common.document_pipeline
    |
    +-- app.common.ocr.read_ocr_rows (one full-page OCR pass)
    |
    +-- normalize OCR text and run receipt currency recovery
    |
    +-- app.classifier.document_classifier (once)
            |
            +-- business_card --> app.business_card.pipeline
            |
            +-- receipt_invoice --> app.main receipt processing
            |
            +-- unknown --> rejected; no extraction pipeline runs
```

The shared orchestrator passes the OCR rows and text to the selected pipeline. The Business Card pipeline does not OCR again. Receipt processing receives the prepared text, rows, currency recovery result, classification, and same LLM client, preserving its extraction, validation, and category stages.

The full-page RapidOCR engine is cached in `app/common/ocr.py` and reused within a process. Receipt currency recovery retains its existing separate crop-recognition engine because its recognition mode affects engine state. The ARQ worker keeps `max_jobs = 1` to limit model and image memory use.

## Extraction pipelines

### Business Card

`app/business_card/pipeline.py` performs deterministic contact extraction, LLM extraction of semantic fields, merging, and Pydantic validation. Its output fields are:

- name
- designation
- company_name
- phone
- fax
- email
- website
- address

Unknown fields remain `null`.

### Receipt and Invoice

The established Receipt/Invoice flow performs OCR text normalization, currency recovery and symbol restoration, document gating, LLM extraction, Pydantic validation, total reconciliation, and expense category selection. Receipt extraction behavior remains separate from the Business Card pipeline.

The category model is TF-IDF plus Logistic Regression. Train it with:

```bash
python train.py
```

Training reads `data/receipt/train/categories.csv` and writes `models/receipt/category_model.joblib`.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure the Ollama-compatible endpoint, API key, and model:

```env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama
OLLAMA_MODEL=qwen3:8b
REDIS_URL=redis://localhost:6379/0
```

For a hosted Ollama-compatible endpoint, set `OLLAMA_BASE_URL`, `OLLAMA_API_KEY`, and `OLLAMA_MODEL` to the provider values. Never commit `.env` to Git.

## Run the backend

Start Redis, then run the API and ARQ worker in separate terminals:

```bash
uvicorn app.main:app --reload
```

```bash
arq app.workers.worker.WorkerSettings
```

The API listens at `http://127.0.0.1:8000` by default. Interactive API docs are at `/docs`; the OpenAPI schema is at `/openapi.json`.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `GET` | `/ready` | Checks required model, category model, and Redis readiness |
| `POST` | `/api/v1/documents` | Validates an image and queues unified document processing |
| `POST` | `/api/v1/receipts` | Backward-compatible receipt-only upload and job |
| `GET` | `/api/v1/jobs/{job_id}` | Returns status, type, result, or error |

Both upload endpoints accept multipart form data with the field name `file`. Supported formats are JPEG, PNG, and WEBP, up to 10 MB and 40 million pixels. Upload returns a job ID immediately:

```json
{
  "job_id": "<job-id>",
  "status": "queued"
}
```

Poll `/api/v1/jobs/{job_id}` for `QUEUED`, `PROCESSING`, `COMPLETED`, `REJECTED`, or `FAILED`. Generalized document job results include the detected type and typed result. For example:

```json
{
  "job_id": "<job-id>",
  "status": "COMPLETED",
  "document_type": "business_card",
  "result": {
    "document_type": "business_card",
    "result": {
      "name": "Ada Lovelace",
      "designation": "Engineer",
      "company_name": "Analytical Engines",
      "phone": null,
      "fax": null,
      "email": "ada@example.com",
      "website": null,
      "address": null
    }
  }
}
```

For a receipt or invoice, `document_type` is `receipt_invoice`, and the nested result uses the existing Receipt schema. Unknown documents are rejected and do not enter either extraction pipeline. Upload/API errors use an `error_code` and human-readable `message`.

The legacy `/api/v1/receipts` endpoint continues to enqueue `process_receipt` and keeps its receipt-specific result format. Use `/api/v1/documents` for new integrations.

## Run the React frontend

```bash
cd frontend
npm install
npm run dev
```

Use Node.js 20.19 or newer, then open `http://localhost:5173`. The frontend uploads to `/api/v1/documents`, polls the shared job-status endpoint, and displays Business Card or Receipt/Invoice results according to `document_type`. Configure the backend address with `VITE_API_BASE_URL` in `frontend/.env.local`; `frontend/.env.example` contains the local default.

## Run the legacy Receipt/Invoice CLI

```bash
python -m app.main data/receipt/test/image6.jpeg
```

The CLI accepts an image or OCR text file and runs the existing Receipt/Invoice flow. Unified Business Card routing is exposed through `/api/v1/documents`.

## Tests

Run the full backend test suite with:

```bash
pytest -v
```
