import argparse
from contextlib import asynccontextmanager
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from arq import create_pool
from arq.connections import RedisSettings

from app.receipt.models.category_model import predict_category
from app.receipt.modules.currency_recovery import analyse_text, recover_currency, restore_symbols
from app.classifier.document_classifier import classify_document #is_receipt_or_invoice
from app.receipt.modules.extractor import extract_receipt
from app.receipt.modules.llm_validator import OllamaGptClient
from app.common.ocr import read_ocr_rows
from app.receipt.modules.text_normalizer import normalize_ocr_text
from app.receipt.modules.validator import deterministic_validate, normalize_currency, normalize_data, reconcile_totals
from app.api.routes import router as receipts_router
from app.api.routes_jobs import router as jobs_router
from app.common.config import get_settings
from app.common.errors import install_error_handlers
from app.common.logging import configure_logging
from app.common.upload_limits import UploadBodyLimitMiddleware
from app.common.processing_timing import timed_stage
from app.receipt.modules.category_evidence import evaluate_category_evidence


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

    for block in rows:
        text = block["text"]
        conf = block["score"]
        bbox = block["box"]
        print(f"Text: {text}")
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
MIN_ML_CATEGORY_CONFIDENCE = 0
MIN_STRONG_EVIDENCE_CONFIDENCE = 0
logger = logging.getLogger(__name__)


def gate_document(ocr_text, gpt_client=None):
    """Use the LLM document label to admit receipts/invoices only.

    Call this on the patched text, not the raw OCR: a payment screen
    whose only figure is a bare "190" reads as a document with no money
    in it until the recovered symbol is written back next to it.

    Returns the classifier report.
    """

    document = classify_document(ocr_text, gpt_client)

    
    if not document["is_receipt_invoice"]:
        raise RuntimeError(
            "unsupported document type: " + document["document_type"]
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


def select_expense_category(ocr_text, classifier_text, gpt_client):
    """Keep ML when its confident prediction agrees with strong OCR evidence.

    Otherwise ask the configured gpt-oss/Ollama model to evaluate the full OCR
    text. The evidence rules intentionally do not classify from vendor names.
    """
    with timed_stage("category_evidence"):
        evidence = evaluate_category_evidence(ocr_text)
    predicted_category = None
    ml_confidence = None
    try:
        with timed_stage("tfidf_logistic_regression"):
            predicted_category, ml_confidence = predict_category(classifier_text)
    except Exception as exc:
        logger.warning("Expense category ML prediction unavailable (%s)", type(exc).__name__)

    strong_evidence = (
        evidence.strong_rule
        and evidence.confidence >= MIN_STRONG_EVIDENCE_CONFIDENCE
    )
    ml_is_confident = (
        ml_confidence is not None
        and ml_confidence >= MIN_ML_CATEGORY_CONFIDENCE
    )
    if strong_evidence:
        if predicted_category == evidence.category and ml_is_confident:
            return (
                predicted_category, ml_confidence, "ML", ml_confidence,
                evidence.confidence,
            )
        return (
            evidence.category, evidence.confidence, "Evidence engine",
            ml_confidence, evidence.confidence,
        )

    try:
        with timed_stage("gpt_oss_category_fallback"):
            category, confidence = gpt_client.evaluate_expense_category(ocr_text)
    except Exception as exc:
        logger.warning("Expense category LLM evaluation unavailable (%s)", type(exc).__name__)
        return None, None, None, ml_confidence, evidence.confidence
    if not category or confidence is None:
        return None, None, None, ml_confidence, evidence.confidence
    return category, confidence, "LLM", ml_confidence, evidence.confidence


def process(path):
    with timed_stage("input_load_and_rapidocr"):
        raw_text, rows = load_input(path)
    with timed_stage("ocr_text_normalization"):
        ocr_text = normalize_ocr_text(raw_text)
    if not ocr_text:
        raise RuntimeError("OCR returned no text")

    with timed_stage("currency_recovery"):
        recovery = recover_stage(path, ocr_text, rows)

    if recovery and recovery["currency"]:
        with timed_stage("currency_symbol_restore"):
            ocr_text, _ = restore_symbols(
                ocr_text,
                recovery["currency"],
                recovery["symbol"],
            )

    with timed_stage("qwen_client_setup"):
        Gpt = OllamaGptClient()
    with timed_stage("document_classification"):
        document = gate_document(ocr_text, Gpt)
    data = extract_receipt(ocr_text, Gpt)
    with timed_stage("receipt_data_normalization"):
        data = normalize_data(data)

    # A ticket prints its fare once with no "Total" label, so the model
    # leaves total_amount null. Reconcile against the currency-anchored
    # amounts on the page before anything downstream reads the total.
    with timed_stage("receipt_reconciliation"):
        data.reconciliation_notes.extend(reconcile_totals(data, ocr_text))

    data.document_type = document["document_type"]
    data.document_confidence = document["confidence"]

    # Gpt reads the patched text above, so it usually names the currency
    # itself. Recovery only speaks up when the LLM could not.
    if not data.currency and recovery and recovery["currency"]:
        data.currency = normalize_currency(recovery["currency"])



    with timed_stage("deterministic_validation"):
        errors = deterministic_validate(data)
    if errors:
        if any("tax" in error for error in errors):
            data.tax_amount = None
        if any("total" in error for error in errors):
            data.total_amount = None

    category_text = build_classifier_text(data) or ocr_text
    (
        category,
        confidence,
        category_source,
        ml_confidence,
        evidence_confidence,
    ) = select_expense_category(
        ocr_text, category_text, Gpt
    )
    data.expense_category = category
    data.classification_confidence = confidence
    data.ml_confidence = ml_confidence
    data.strong_evidence_confidence = evidence_confidence
    data.category_source = category_source

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
