from app.schema import LineItem, ReceiptData
from app.validator import (
    is_garbled_description,
    money_anchors,
    normalize_amount,
    normalize_currency,
    normalize_date,
    normalize_data,
    reconcile_totals,
)


def test_amount():
    assert normalize_amount("₹1,234.50") == 1234.50


def test_currency():
    assert normalize_currency("₹") == "INR"


def test_date():
    assert normalize_date("12/09/2026") == "2026-09-12"


def test_data():
    x = normalize_data(ReceiptData(invoice_date="12/09/2026", total_amount="100.00"))
    assert x.invoice_date == "2026-09-12" and x.total_amount == 100.0


TICKET_OCR = (
    "Pollachi 3 DEPOT\n"
    "28-09-2026\n"
    "07:29:21\n"
    "UPI\n"
    "ADU4T (S)343)-R ₹5729.00\n"
    "Rs:129.00\n"
)


def test_garbled_descriptions_are_told_from_real_ones():
    assert is_garbled_description("ADU4T (S)343)-R") is True
    assert is_garbled_description("MALE(S)") is False
    assert is_garbled_description("ADULT (S)") is False
    assert is_garbled_description("Sambar Sadam") is False


def test_money_anchors_flag_the_garbled_line():
    anchors = money_anchors(TICKET_OCR)

    trusted = [amount for amount, _, ok in anchors if ok]
    untrusted = [amount for amount, _, ok in anchors if not ok]

    assert trusted == [129.0]
    assert untrusted == [5729.0]


def test_a_missing_total_is_taken_from_the_single_line_item():
    ocr = "Kangeyam DEPOT\nMALE(S):1 = 11.0 = Rs 11.00\nRs:11.00\n"
    data = ReceiptData(
        currency="INR",
        line_items=[LineItem(description="MALE(S)", quantity=1, amount=11.0)],
    )

    notes = reconcile_totals(data, ocr)

    assert data.total_amount == 11.0
    assert "total taken from the single line item" in notes


def test_a_total_taken_from_a_garbled_line_is_rejected():
    """img5: the model priced a mangled line at 5729 and called it the total."""

    data = ReceiptData(
        currency="INR",
        line_items=[LineItem(description="ADU4T (S)343)-R", amount=5729.0)],
        total_amount=5729.0,
    )

    notes = reconcile_totals(data, TICKET_OCR)

    assert data.line_items == []
    assert data.total_amount == 129.0
    assert any("garbled line item" in note for note in notes)
    assert any("cleared total" in note for note in notes)


def test_a_good_total_is_left_alone():
    ocr = "SHOP\nTOTAL Rs 189.00\n"
    data = ReceiptData(currency="INR", total_amount=189.0)

    reconcile_totals(data, ocr)

    assert data.total_amount == 189.0


def test_a_page_with_no_trustworthy_amount_keeps_a_null_total():
    ocr = "ADU4T (S)343)-R ₹5729.00\n"
    data = ReceiptData(currency="INR")

    notes = reconcile_totals(data, ocr)

    assert data.total_amount is None
    assert any("no trustworthy amount" in note for note in notes)
