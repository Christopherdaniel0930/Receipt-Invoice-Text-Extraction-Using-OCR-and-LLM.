"""Independent business card processing pipeline."""

from app.business_card.modules.deterministic_extractor import extract_deterministic
from app.business_card.modules.llm_extractor import extract_business_card_semantics
from app.business_card.modules.merger import merge_business_card
from app.business_card.modules.validator import validate_business_card


def process_business_card(
    image_path,
    ocr_text=None,
    ocr_rows=None,
    llm_client=None,
    ocr_function=None,
):
    """Extract card data, reusing OCR text/rows when supplied."""
    if ocr_text is None and ocr_rows is not None:
        ocr_text = "\n".join(
            str(row.get("text", "")).strip()
            for row in ocr_rows
            if row.get("text")
        )
    if ocr_text is None:
        if ocr_function is None:
            from app.common.ocr import run_ocr

            ocr_function = run_ocr
        ocr_text = ocr_function(str(image_path))
    if not isinstance(ocr_text, str) or not ocr_text.strip():
        raise RuntimeError("OCR returned no text")
    deterministic = extract_deterministic(ocr_text)
    semantic = extract_business_card_semantics(ocr_text, llm_client)
    return validate_business_card(merge_business_card(semantic, deterministic))
