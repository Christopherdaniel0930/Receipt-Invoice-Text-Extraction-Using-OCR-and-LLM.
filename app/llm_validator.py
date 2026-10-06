import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


EXTRACTION_PROMPT = """
You are an invoice and receipt information extraction system.

Extract ONLY information that is directly supported by the OCR text.

Return ONLY valid JSON using exactly this structure:

{
    "vendor_name": null,
    "invoice_number": null,
    "invoice_date": null,
    "tax_amount": null,
    "currency": null,
    "line_items": [
        {
            "description": null,
            "quantity": null,
            "unit_price": null,
            "amount": null
        }
    ],
    "total_amount": null,
}

IMPORTANT:
The line item field names MUST be exactly:

"description"
"quantity"
"unit_price"
"amount"

NEVER use:
"item"
"total_price"
"price"
"product"
"name"

for line-item fields.

Rules:

1. NEVER invent, guess, or assume information.

2. VENDOR:
Identify the merchant, hotel, restaurant, company, store,
or organization that issued the invoice or receipt.

3. INVOICE NUMBER:
Only extract an invoice number when the OCR explicitly
associates a value with labels such as:

- Invoice No
- Invoice Number
- Invoice #
- Inv No
- Inv Number
- Bill No
- Bill Number
- Receipt No
- Receipt Number
- Order ID

NEVER treat a standalone numeric value as an invoice number.

4. DATE:
Search the ENTIRE OCR text for dates and timestamps.

If a date is clearly present, convert it to:

YYYY-MM-DD

Ignore the time portion.

5. TAX:
Extract tax only when the OCR explicitly indicates:

- Tax
- GST
- CGST
- SGST
- IGST
- VAT
- Sales Tax

6. TOTAL:
Extract the final payable, charged, amount paid,
grand total, or total amount when clearly supported
by the OCR.

Do NOT treat transaction IDs, UPI IDs, phone numbers,
timestamps, or arbitrary numbers as total_amount.

7. CURRENCY:
Extract currency only when supported by the OCR.

Examples:

₹ -> INR
Rs -> INR
Rs. -> INR
INR -> INR
$ -> USD
USD -> USD
€ -> EUR
EUR -> EUR

If currency cannot be established, return null.

8. LINE ITEMS:
Extract ONLY actual purchased goods or services.

For every line item:

"description" = item/service name
"quantity" = quantity if explicitly available, otherwise null
"unit_price" = price per unit if explicitly available, otherwise null
"amount" = total price for that line item if explicitly available, otherwise null

Example:

{
    "description": "Sambar Sadam",
    "quantity": 2,
    "unit_price": 93,
    "amount": 186
}

Do NOT use "item" instead of "description".
Do NOT use "total_price" instead of "amount".

Do NOT treat UPI/payment text such as:

- Pay again
- Completed
- Google Pay
- UPI
- Transaction ID
- Payment successful

as line items.

9. PAYMENT INFORMATION:
Do not confuse payment metadata with invoice information.

10. MISSING INFORMATION:
If a field cannot be reliably determined from the OCR,
return null.

If there are no actual line items, return:

"line_items": []

11. OUTPUT:
Return ONLY the JSON object.
Do not include Markdown.
Do not include explanations.
Do not include ```json.

OCR TEXT:
----------------
"""

class OllamaGptClient:

    def __init__(self):
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434/v1",
        )

        api_key = os.getenv(
            "OLLAMA_API_KEY",
            "ollama",
        )

        model = os.getenv("OLLAMA_MODEL")

        if not model:
            raise RuntimeError(
                "OLLAMA_MODEL is not configured"
            )

        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def extract(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured invoice and "
                        "receipt data. Never invent information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "Ollama/GPT returned an empty response"
            )

        return content

    def evaluate_expense_category(self, ocr_text: str) -> tuple[str | None, float | None]:
        """Classify from OCR, allowing a new label when the known labels do not fit."""
        from app.validator import CATEGORIES

        allowed = sorted(CATEGORIES)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify a receipt expense using only its OCR text. "
                        "Do not infer a category from the merchant name alone. "
                        "Use an existing category when it accurately describes the "
                        "expense. If none fits, return a concise new category label. "
                        "Return JSON with expense_category and confidence from 0 to 1."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Existing categories: {json.dumps(allowed)}\n"
                        "Return only {\"expense_category\": string, \"confidence\": number}. "
                        "The category may be a concise new label only if none of the "
                        "existing categories fits.\n"
                        f"OCR text:\n{ocr_text}"
                    ),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            return None, None
        try:
            result = json.loads(content)
            category = result.get("expense_category")
            confidence = float(result.get("confidence"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None, None
        if (
            not isinstance(category, str)
            or not category.strip()
            or len(category.strip()) > 60
            or not 0.0 <= confidence <= 1.0
        ):
            return None, None
        return category.strip(), confidence


def build_extraction_prompt(ocr: str) -> str:
    """Build the extraction prompt from OCR text."""

    return (
        EXTRACTION_PROMPT
        + ocr
        + "\n----------------\n"
    )


def build_prompt(ocr_text, data):
    """Backward-compatible validation prompt helper."""

    return (
        "Validate this receipt extraction using ONLY "
        "the OCR-supported information.\n\n"
        "Return ONLY valid JSON.\n\n"
        "OCR:\n"
        + ocr_text
        + "\n\nCANDIDATE:\n"
        + json.dumps(
            data.model_dump(),
            ensure_ascii=False,
        )
    )


def validate_with_llm(ocr_text, data, llm_call):
    result = llm_call(
        build_prompt(
            ocr_text,
            data,
        )
    )

    if isinstance(result, str):
        result = json.loads(result)

    from app.schema import ValidationResult

    return ValidationResult.model_validate(result)
