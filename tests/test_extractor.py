import json

from app.extractor import extract_receipt, parse_gpt_response


def test_parse_gpt_response():
    response = json.dumps({
        "vendor_name": "Fresh Mart",
        "invoice_number": "INV-1001",
        "invoice_date": "2026-09-12",
        "tax_amount": 12.6,
        "total_amount": 152.6,
        "currency": "INR",
        "line_items": [],
    })
    data = parse_gpt_response(response)
    assert data.vendor_name == "Fresh Mart"
    assert data.total_amount == 152.6


def test_extract_uses_gpt_client():
    class Fakegpt:
        def extract(self, prompt):
            return json.dumps({
                "vendor_name": "Fresh Mart",
                "invoice_number": "INV-1",
                "invoice_date": "2026-09-12",
                "tax_amount": 10,
                "total_amount": 110,
                "currency": "INR",
                "line_items": [],
            })

    data = extract_receipt("Fresh Mart\nTotal 110", Fakegpt())
    assert data.vendor_name == "Fresh Mart"
    assert data.total_amount == 110
