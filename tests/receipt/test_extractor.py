import json

from app.receipt.modules.extractor import extract_receipt, parse_gpt_response, parse_gpt_response
from app.receipt.schemas.schema import ReceiptData
from app.receipt.modules.validator import CATEGORIES, normalize_data, reconcile_totals


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


def test_gpt_line_item_amount_survives_parsing():
    response = json.dumps({
        "line_items": [{
            "description": "CHILLY PAROTTA", "quantity": 1.0,
            "unit_price": 110.0, "amount": 110.0,
        }],
    })
    item = parse_gpt_response(response).line_items[0]
    assert item.amount == 110.0


def test_line_item_amounts_survive_receipt_normalization_and_reconciliation():
    response = json.dumps({"line_items": [
        {"description": "CHILLY PAROTTA", "quantity": 1, "unit_price": 110, "amount": 110},
        {"description": "TEA", "quantity": 1, "unit_price": 60, "amount": 60},
        {"description": "WATER", "quantity": 1, "unit_price": 90, "amount": 90},
    ]})
    receipt = normalize_data(parse_gpt_response(response))
    reconcile_totals(receipt, "CHILLY PAROTTA 110\nTEA 60\nWATER 90")
    assert [item.amount for item in receipt.line_items] == [110.0, 60.0, 90.0]


def test_explicit_invoice_number_is_preserved_from_ocr():
    response = json.dumps({"invoice_number": "12345"})
    parsed = parse_gpt_response(response, "Fresh Mart\nInvoice No: 12345")
    assert parsed.invoice_number == "12345"


def test_standalone_numeric_value_is_rejected_as_invoice_number():
    response = json.dumps({"invoice_number": "12345"})
    parsed = parse_gpt_response(response, "Fresh Mart\n12345\nTotal 110.00")
    assert parsed.invoice_number is None


def test_only_exact_expense_categories_are_accepted():
    assert CATEGORIES == {
        "Food", "Travel", "Fuel", "Electronics", "Office Supplies",
        "Accommodation", "Healthcare", "Utilities", "Other",
    }
    for category in CATEGORIES:
        assert normalize_data(ReceiptData(expense_category=category)).expense_category == category
    assert normalize_data(ReceiptData(expense_category="Food & Dining")).expense_category is None
