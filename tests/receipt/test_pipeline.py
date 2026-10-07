from app.receipt.schemas.schema import ReceiptData
from app.receipt.modules.validator import normalize_data


def test_normalize_receipt():
    data = ReceiptData(
        invoice_date="12/09/2026",
        total_amount="152.60",
        currency="₹",
    )
    data = normalize_data(data)
    assert data.invoice_date == "2026-09-12"
    assert data.total_amount == 152.60
    assert data.currency == "INR"
