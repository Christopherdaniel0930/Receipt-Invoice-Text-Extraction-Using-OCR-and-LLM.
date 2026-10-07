"""Semantic name, title, and organization extraction."""

import json


FIELDS = ("name", "designation", "company_name")


def build_business_card_prompt(ocr_text: str) -> str:
    return '''Extract business card fields from the OCR text below. Use only information present in the OCR text. Never invent values. Return null when information is missing. Do not infer a company name without evidence. Do not confuse designation with company name. Return strict JSON matching exactly this schema: {"name": null, "designation": null, "company_name": null}. Do not return any other fields or explanation.\n\nOCR TEXT:\n''' + str(ocr_text or "")


def extract_business_card_semantics(ocr_text: str, llm_client=None) -> dict:
    """Call the configured Ollama-compatible client and parse its JSON output."""
    if llm_client is None:
        from app.receipt.modules.llm_validator import OllamaGptClient

        llm_client = OllamaGptClient()
    raw = llm_client.extract(build_business_card_prompt(ocr_text))
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("Business card LLM response must be a JSON object")
    return {
        field: parsed.get(field) if isinstance(parsed.get(field), str) else None
        for field in FIELDS
    }
