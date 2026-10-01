import json
import re

from app.schema import ReceiptData
from app.text_normalizer import normalize_ocr_text
from app.llm_validator import build_extraction_prompt

INVOICE_LABEL_RE = re.compile(
    r"\b(?:invoice\s*(?:no\.?|number|#)?|inv\s*(?:no\.?|number|#)|"
    r"bill\s*(?:no\.?|number|#)|receipt\s*(?:no\.?|number|#)|order\s*(?:id|no\.?|number|#))\s*[:#=-]?\s*"
    r"([A-Z0-9][A-Z0-9./_-]{1,})",
    re.IGNORECASE,
)


def _ocr_supported_invoice_number(value, ocr_text):
    if not value or not ocr_text:
        return None
    candidate = str(value).strip()
    candidate = re.sub(
        r"^(?:invoice|inv|bill|receipt)\s*(?:no\.?|number|#)?\s*[:#=-]*\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"[^a-z0-9]", "", candidate.lower())
    if not normalized:
        return None
    for match in INVOICE_LABEL_RE.finditer(ocr_text):
        supported = re.sub(r"[^a-z0-9]", "", match.group(1).lower())
        if normalized == supported:
            return match.group(1)
    return None


def normalize_llm_response(data: dict) -> dict:
    """Normalize common LLM field-name variations."""

    line_items = data.get("line_items", [])

    normalized_items = []

    for item in line_items:
        if not isinstance(item, dict):
            continue

        normalized_items.append({
            "description": (
                item.get("description")
                or item.get("item")
                or item.get("product")
                or item.get("name")
            ),
            "quantity": item.get("quantity"),
            "unit_price": item.get("unit_price"),
            "amount": (
                item.get("amount")
                if item.get("amount") is not None
                else item.get("total_price")
            ),
        })

    data["line_items"] = normalized_items

    return data


def parse_gpt_response(response: str, ocr_text: str | None = None) -> ReceiptData:
    data = json.loads(response)

    data = normalize_llm_response(data)
    data["invoice_number"] = _ocr_supported_invoice_number(
        data.get("invoice_number"), ocr_text
    )

    return ReceiptData.model_validate(data)




def extract_receipt(ocr_text: str, gpt_client) -> ReceiptData:
    clean_text = normalize_ocr_text(ocr_text)

    prompt = build_extraction_prompt(clean_text)

    response = gpt_client.extract(prompt)

    return parse_gpt_response(response, clean_text)