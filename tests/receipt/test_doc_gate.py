import pytest

from app.receipt.modules.currency_recovery import restore_symbols
from app.classifier.document_classifier import classify_document
from app.main import MIN_FORWARDED_CONFIDENCE, gate_document
from app.receipt.schemas.schema import ReceiptData

UPI_SCREENSHOT = """UPI transaction ID
566869484790
To: OCEANA HOTEL AND RESIDENCY
paytm.s1eOazm@pty
From: Mr Sankar Selvaraj (Indian Bank)
Google Pay
"""

BUS_TICKET = """Pollachi 3 DEPOT
28-09-2026
07:29:21
WB18051
UPI
R5729.00
Rs:129.00
"""

CARD_PAYMENT_SCREEN = """x
Coffee House Family Restaurant
176
SUCCESSFUL
12:42pm, 25th nov'25
Paid via CRED
Kotak Mahindra Bank
payment initiated from your Kotak Mahindra Bank account
"""

RETAIL_RECEIPT = """THE CORNER CAFE
GSTIN: 29ABCDE1234F1Z5
2 x Sambar Sadam           120.00
Sub Total                  180.00
GST 5%                       9.00
TOTAL                      189.00
Thank you for your visit
"""


def test_a_retail_receipt_passes_the_gate():
    report = gate_document(RETAIL_RECEIPT)

    assert report["is_receipt_invoice"] is True
    assert report["document_type"] in ("receipt", "receipt_or_invoice")
    assert report["hard_counters"] == []


@pytest.mark.parametrize("text", [UPI_SCREENSHOT, BUS_TICKET, CARD_PAYMENT_SCREEN])
def test_money_screens_and_tickets_still_reach_the_llm(text):
    report = gate_document(text)

    assert report["hard_counters"] == []
    assert report["is_receipt_invoice"] is False, (
        "these score too low but are still real money documents"
    )


def test_a_low_score_receipt_is_not_rejected():
    # The classifier is not confident, but nothing vetoes it.
    report = gate_document("Total 12.00\n")

    assert report["hard_counters"] == []
    assert report["is_receipt_invoice"] is False
    assert report["blocked_by"]


@pytest.mark.parametrize(
    "text,name",
    [
        (
            "John Smith\nDate of Birth: 04/07/1990\nPassport No: X1234567\n"
            "Nationality: Indian\nSex: M\n",
            "identity_document",
        ),
        (
            "Dear Sir,\nPlease find attached our quotation.\n"
            "Yours sincerely,\nRamesh Iyer\n",
            "personal_letter",
        ),
        (
            "CURRICULUM VITAE\nRamesh Iyer\nWork Experience\n"
            "Skills Section\nEducational Qualification\n",
            "resume_or_cv",
        ),
        (
            "This Agreement is made on 01/03/2024 between the parties.\n"
            "Whereas the parties hereby agree to Clause 1.\n"
            "Governing Law: India. Termination of this agreement applies.\n",
            "legal_contract",
        ),
        (
            "Published on March 4, by A Reporter for Reuters\n"
            "Copyright all rights reserved\nRead more\nClick here\n",
            "news_or_article",
        ),
    ],
)
def test_hard_counter_documents_are_rejected(text, name):
    with pytest.raises(RuntimeError) as error:
        gate_document(text)

    assert name in str(error.value)


def test_gate_rejects_even_when_a_resume_also_looks_invoice_like():
    # A CV that happens to mention an invoice must still be rejected.
    text = (
        "CURRICULUM VITAE\n"
        "Ramesh Iyer\n"
        "Work Experience\n"
        "References available on request.\n"
        "Invoice No: INV-9  Total 4500.00 GST 500.00 Due Date 30 days\n"
    )

    with pytest.raises(RuntimeError) as error:
        gate_document(text)

    assert "resume_or_cv" in str(error.value)


def test_a_bank_statement_is_not_vetoed():
    # bank_statement is a soft counter: a statement that is really a
    # payment confirmation must still be extracted.
    report = gate_document(
        "AXIS BANK\nStatement of Account\n"
        "Date Description Withdrawal Deposit Balance\n"
        "01/03/2024 ATM WITHDRAWAL 5,000.00 45,000.00\n"
        "Total 189.00\n"
    )

    assert report["counters"] == ["bank_statement"]
    assert report["hard_counters"] == []


def test_document_fields_default_to_none():
    data = ReceiptData()

    assert data.document_type is None
    assert data.document_confidence is None


def test_document_fields_are_serialised():
    data = ReceiptData(vendor_name="Fresh Mart", document_type="receipt")
    payload = data.model_dump()

    assert payload["document_type"] == "receipt"
    assert payload["document_confidence"] is None


def test_classifier_report_is_passed_through_unchanged():
    report = classify_document(RETAIL_RECEIPT)
    gated = gate_document(RETAIL_RECEIPT)

    assert gated["is_receipt_invoice"] == report["is_receipt_invoice"]
    assert gated["score"] == report["score"]
    assert gated["document_type"] == report["document_type"]
    assert gated["hard_counters"] == report["hard_counters"]


def test_a_weak_but_forwardable_document_is_reported_at_the_floor():
    """A low-scoring money document still reaches the LLM, so report 0.50."""

    weak = (
        "UPI\n"
        "POWEHED HY\n"
        "G Pay\n"
        "123456789012\n"
    )

    raw = classify_document(weak)
    assert raw["confidence"] < MIN_FORWARDED_CONFIDENCE
    assert raw["hard_counters"] == []

    report = gate_document(weak)

    assert report["confidence"] == MIN_FORWARDED_CONFIDENCE


def test_business_cards_are_rejected():
    """Contact details and no money is a card, not a receipt."""

    cards = [
        "Ph:412-621-1356\nFax:412-621-2410\nMARK D. MALVIN, D.M.D.\n"
        "Comprehensive Dental Care\n410 S. Craig St. Suite 102\n"
        "Pittsburgh, PA 15213\nBy Appointment",
        "AVTEC SYSTEMS\n14432 ALBEMARLE POINT PLACE\nCHANTILLY, VIRGINIA 20151\n"
        "(703) 488-2500\nFax (703) 488-2555\nmpoole@avtec.com\n"
        "http://www.avtec.com",
        "JULIANA SILVA\nMANAGER\n123 Anywhere St., Any City\n"
        "+123-456-7890\nhello@reallygreatsite.com\nwww.reallygreatsite.com",
    ]

    for card in cards:
        report = classify_document(card)

        assert "business_card" in report["hard_counters"]
        assert report["is_receipt_invoice"] is False

        with pytest.raises(RuntimeError, match="business_card"):
            gate_document(card)


def test_a_receipt_with_a_phone_number_is_not_a_business_card():
    """A contact block does not veto a document that has money on it."""

    receipt = (
        "FRESH MART\n"
        "Phone 9887766554\n"
        "www.freshmart.example\n"
        "TOTAL 189.00\n"
        "Thank you for your visit\n"
    )

    report = gate_document(receipt)

    assert "business_card" not in report["hard_counters"]
    assert report["is_receipt_invoice"] is True


def test_a_strong_document_keeps_its_own_confidence():
    raw = classify_document(RETAIL_RECEIPT)

    assert gate_document(RETAIL_RECEIPT)["confidence"] == raw["confidence"]


def test_a_bare_total_reads_as_a_receipt_once_patched():
    """The image1 shape: money only survives after the symbol is written back."""

    raw = (
        "To OCEANA HOTEL AND RESIDENCY\n"
        "190\n"
        "Pay again\n"
        "Completed\n"
        "29 Oct 2025, 5:20 pm\n"
        "Indian Bank 6521\n"
        "UPI transaction ID\n"
        "566869484790\n"
    )

    assert classify_document(raw)["is_receipt_invoice"] is False

    patched, count = restore_symbols(raw, "INR", "₹")
    report = classify_document(patched)

    assert count == 1
    assert report["is_receipt_invoice"] is True
    assert report["document_type"] == "receipt"
    assert report["hard_counters"] == []
