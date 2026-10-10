"""Shared OCR, classification, and dispatch for supported documents."""

from app.common.ocr import read_ocr_rows
from app.classifier.document_classifier import classify_document


def _text_from_rows(rows):
    return "\n".join(
        str(row.get("text", "")).strip()
        for row in rows
        if row.get("text")
    )


def _get_llm_client(llm_client):
    if llm_client is not None:
        return llm_client
    from app.receipt.modules.llm_validator import OllamaGptClient

    return OllamaGptClient()


def process_document(image_path, llm_client=None):
    """OCR once, classify once, then run the matching extraction pipeline.

    Returns a dictionary containing the classifier's ``document_type`` and
    the selected pipeline's Pydantic result in ``data``. Unknown documents
    return ``data=None`` and are not sent to either extractor.
    """
    rows = read_ocr_rows(str(image_path))
    raw_text = _text_from_rows(rows)
    if not raw_text.strip():
        raise RuntimeError("OCR returned no text")

    # Receipt classification historically ran after currency recovery. Keep
    # that ordering while sharing the single OCR result with both branches.
    from app.main import prepare_document_input, process as process_receipt

    ocr_text, recovery = prepare_document_input(image_path, raw_text, rows)
    client = _get_llm_client(llm_client)
    document = classify_document(ocr_text, client)
    document_type = document.get("document_type", "unknown")

    if document_type == "business_card":
        from app.business_card.pipeline import process_business_card

        data = process_business_card(
            image_path,
            ocr_text=ocr_text,
            ocr_rows=rows,
            llm_client=client,
        )
    elif document.get("is_receipt_invoice") or document_type in {
        "receipt", "invoice", "receipt_or_invoice", "receipt_invoice"
    }:
        data = process_receipt(
            image_path,
            ocr_text=ocr_text,
            ocr_rows=rows,
            llm_client=client,
            document=document,
            recovery=recovery,
            preprocessed=True,
        )
    else:
        return {"document_type": document_type, "data": None}

    return {"document_type": document_type, "data": data}
