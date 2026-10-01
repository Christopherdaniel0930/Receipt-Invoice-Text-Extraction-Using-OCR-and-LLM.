import argparse
from contextlib import asynccontextmanager
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from arq import create_pool
from arq.connections import RedisSettings

from app.category_model import predict_category
from app.currency_recovery import analyse_text, recover_currency, restore_symbols
from app.doc_classifier import classify_document #is_receipt_or_invoice
from app.extractor import extract_receipt
from app.llm_validator import OllamaGptClient
from app.ocr import read_ocr_rows
from app.text_normalizer import normalize_ocr_text
from app.validator import CATEGORIES, deterministic_validate, normalize_currency, normalize_data, reconcile_totals
from app.api.routes_receipts import router as receipts_router
from app.api.routes_jobs import router as jobs_router
from app.core.config import get_settings
from app.core.errors import install_error_handlers
from app.core.logging_config import configure_logging
from app.core.upload_limits import UploadBodyLimitMiddleware


def load_input(path):
    """Return (ocr_text, rows). rows is None for a .txt input.

    One OCR pass produces both, so the currency pixel layer can crop the
    image without the engine being run a second time.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".txt":
        return source.read_text(encoding="utf-8"), None
    rows = read_ocr_rows(str(source))
    return "\n".join(row["text"] for row in rows if row["text"]), rows


def recover_stage(path, ocr_text, rows):
    """Recover the currency the OCR text lost.

    Returns the recovery report, or None when the layer could not run.
    A currency guess is never worth failing an extraction over.
    """
    try:
        if rows is None:
            return analyse_text(ocr_text)
        return recover_currency(path, ocr_text=ocr_text, rows=rows)
    except Exception as error:
        print(
            f"currency recovery unavailable: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None


MIN_FORWARDED_CONFIDENCE = 0.5


def gate_document(ocr_text):
    """Reject the documents that can never be a receipt or an invoice.

    Only the classifier's hard counters veto. A receipt that merely
    scores too low is left alone: UPI payment screens, bus tickets,
    bank screenshots and even plain letterheads are real money
    documents or at least worth reading, they just do not look like a
    till receipt, and rejecting them would throw away data the
    extractor handles fine.

    Call this on the patched text, not the raw OCR: a payment screen
    whose only figure is a bare "190" reads as a document with no money
    in it until the recovered symbol is written back next to it.

    Anything that survives is reported at no less than
    MIN_FORWARDED_CONFIDENCE, so a downstream consumer that filters on
    confidence keeps the same set of documents that this gate keeps.
    The raw classifier verdict is left intact alongside it.

    Returns the classifier report.
    """

    document = classify_document(ocr_text)

    
    if document["hard_counters"]:
        raise RuntimeError(
            "the image is not a receipt or invoice ("
            + ", ".join(document["hard_counters"])
            + ")"
        )

    document["confidence"] = max(
        document["confidence"], MIN_FORWARDED_CONFIDENCE
    )

    # if is_receipt_or_invoice:
    #     raise RuntimeError("The image is not a reciept or invoice \n"+"Reason:\n"+"\tDocument confidence : "+str(document["confidence"]))


    return document


def build_classifier_text(data):
    parts = []
    if data.vendor_name:
        parts.append(data.vendor_name)
    for item in data.line_items:
        if item.description:
            parts.append(item.description)
    return " ".join(parts).strip()


def process(path):
    raw_text, rows = load_input(path)
    ocr_text = normalize_ocr_text(raw_text)
    if not ocr_text:
        raise RuntimeError("OCR returned no text")

    recovery = recover_stage(path, ocr_text, rows)

    if recovery and recovery["currency"]:
        ocr_text, _ = restore_symbols(
            ocr_text,
            recovery["currency"],
            recovery["symbol"],
        )

    document = gate_document(ocr_text)

    Gpt = OllamaGptClient()
    data = extract_receipt(ocr_text, Gpt)
    data = normalize_data(data)

    # A ticket prints its fare once with no "Total" label, so the model
    # leaves total_amount null. Reconcile against the currency-anchored
    # amounts on the page before anything downstream reads the total.
    data.reconciliation_notes.extend(reconcile_totals(data, ocr_text))

    data.document_type = document["document_type"]
    data.document_confidence = document["confidence"]

    # Gpt reads the patched text above, so it usually names the currency
    # itself. Recovery only speaks up when the LLM could not.
    if not data.currency and recovery and recovery["currency"]:
        data.currency = normalize_currency(recovery["currency"])



    errors = deterministic_validate(data)
    if errors:
        if any("tax" in error for error in errors):
            data.tax_amount = None
        if any("total" in error for error in errors):
            data.total_amount = None

    try:
        category_text = build_classifier_text(data) or ocr_text
        category, confidence = predict_category(category_text)
        data.expense_category = category if category in CATEGORIES else "Other"
        data.classification_confidence = confidence
    except FileNotFoundError:
        data.expense_category = None
        data.classification_confidence = None

    return data


def main():
    parser = argparse.ArgumentParser(description="Receipt/invoice information extraction")
    parser.add_argument("input", help="Image or .txt OCR input")
    args = parser.parse_args()
    result = process(args.input)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


configure_logging()
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Connect the API process to Redis; OCR engines live in the worker."""
    redis = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    try:
        await redis.ping()
        api.state.redis = redis
        yield
    finally:
        await redis.aclose()


api = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
api.add_middleware(
    UploadBodyLimitMiddleware,
    max_body_bytes=settings.max_upload_bytes + 64 * 1024,
    path="/api/v1/receipts",
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
install_error_handlers(api)
api.include_router(receipts_router)
api.include_router(jobs_router)


@api.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@api.get("/ready", tags=["health"])
async def ready():
    configured = bool(settings.ollama_model)
    model_available = settings.category_model_path.is_file()
    redis = getattr(api.state, "redis", None)
    try:
        queue_available = redis is not None and bool(await redis.ping())
    except Exception:
        queue_available = False
    if not configured or not model_available or not queue_available:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "status": 503,
                "error_code": "NOT_READY",
                "message": "Required extraction models or queue service are unavailable.",
            },
        )
    return {"status": "ready"}


app = api


if __name__ == "__main__":
    main()
