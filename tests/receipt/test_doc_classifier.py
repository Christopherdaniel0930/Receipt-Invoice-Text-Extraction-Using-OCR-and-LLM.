"""Regression tests for app.classifier.document_classifier.

The corpus below mirrors the SAMPLES dict in the module's ``__main__`` block,
copied here because that block is not importable. These tests pin the exact
score, confidence and document_type for every sample so that any change to a
rule weight, a threshold or a pattern shows up as a diff in review.

Tests whose name starts with ``test_known_defect_`` assert the behaviour the
classifier *should* have and are marked ``xfail(strict=True)``. When one of
them starts passing, pytest reports XPASS and fails, which forces the marker
to be deleted at the same time as the fix lands.
"""

import textwrap

import pytest

from app.classifier import document_classifier as dc
from app.classifier.document_classifier import classify_document, is_receipt_or_invoice

CORPUS = {
    "restaurant_receipt": """
        THE CORNER CAFE
        14 MG ROAD, BENGALURU
        GSTIN: 29ABCDE1234F1Z5
        --------------------------------
        Order No : 4821        Table 7
        2 x Sambar Sadam           120.00
        1 x Filter Coffee           60.00
        --------------------------------
        Sub Total                  180.00
        GST 5%                       9.00
        TOTAL                      189.00
        UPI / GPay                189.00
        Thank you for your visit
        """,
    "b2b_invoice": """
        TAX INVOICE
        Invoice No: INV-2024-0091
        Invoice Date: 12/03/2024
        Due Date: 11/04/2024
        Bill To: Northwind Traders Ltd
        GSTIN: 27AAECN1234C1ZV
        Particulars      Qty    Unit Price    Amount
        Cloud Hosting      12        450.00     5400.00
        Support Retained    1       1200.00     1200.00
        Sub Total                            6600.00
        CGST 9%                              297.00
        Total Due                          6897.00
        Payment Terms: Net 30 days
        Remit To: HDFC Bank A/c 50200012345678 IFSC HDFC0000123
        """,
    "bank_statement": """
        HDFC BANK
        Statement of Account
        Account Number: 50100234567890
        Statement Period: 01/03/2024 - 31/03/2024
        Date   Description        Withdrawal    Deposit    Balance
        01/03/2024 ATM WITHDRAWAL      5,000.00        -    45,000.00
        05/03/2024 SALARY CREDIT            -   85,000.00   130,000.00
        Opening Balance 50,000.00
        Closing Balance 128,000.00
        """,
    "cover_letter": """
        Dear Sir,
        Please find attached our quotation for the upcoming project.
        We trust this arrangement will be mutually beneficial.
        Yours sincerely,
        Ramesh Iyer
        Accounts Manager
        """,
    "text_article": """
        India's retail inflation cooled to 4.2 percent in March, official
        data showed on Monday. The central bank has kept rates steady for
        six consecutive meetings. Analysts expect a cut later this year.
        """,
    "pharmacy_receipt": """
        APEX MEDICAL CENTRE
        22 Station Road, Pune
        Bill No: 7734
        Paracetamol 500mg Tablets  10   85.00   850.00
        ORS Sachet                    5   20.00   100.00
        Sub Total                             950.00
        TOTAL                              950.00
        Cash Tendered                    1000.00
        Change                              50.00
        """,
    "restaurant_menu": """
        MENU
        Starters
        Paneer Tikka  240.00
        Chicken 65     280.00
        Main Course
        Butter Chicken 380.00
        Dal Makhani     260.00
        Desserts
        Gulab Jamun     120.00
        All taxes excluded. Subject to availability.
        """,
    "thermal_minimal": """
        BLUE BOTTLE COFFEE
        2 x 180.00
        TOTAL  360.00
        CASH 400.00
        CHANGE 40.00
        """,
    "gibberish_ocr": """
        ||| ### ~~~ ###
        ^^^ &&& ((( )))
        """,
    "letter_mentioning_invoice": """
        Dear Sir,
        I enclose our invoice for the work completed in March.
        The total due is 4500.00 and is payable now.
        Yours sincerely,
        Ramesh Iyer
        """,
}

EXTRA_CASES = {
    "one_line_receipt": "TOTAL 45.50 CASH 50.00 CHANGE 4.50",
    "integer_amounts_with_labels": "BLUE BOTTLE\nTOTAL 450\nCASH 500\nCHANGE 50\nTHANK YOU",
    "integer_amounts_no_labels": "BLUE BOTTLE\n2 X 180\n360",
    "short_fragment": "TOTAL 45.50",
    "long_single_line_gibberish": "x" * 500,
    "plain_paragraph": "hello world this is plain text",
    "invoice_dear_customer": """
        INVOICE
        Invoice No: INV-9
        Invoice Date: 01/03/2024
        Bill To: ACME LTD
        Sub Total 1000.00
        Total Due 1180.00
        Thank you for your business
        Dear Customer, please contact us for any queries.
        """,
    "invoice_references": """
        TAX INVOICE
        Invoice No: INV-77
        Invoice Date: 01/03/2024
        Due Date: 30/03/2024
        Bill To: ACME LTD
        Particulars Qty Unit Price
        Widget 2 500.00
        Sub Total 1000.00
        Total Due 1180.00
        References: PO-45
        """,
    "email_about_invoice": """
        Hi team,
        Please find attached the invoice for March. The total due is 4500.00 and
        is payable within 30 days. The balance change is minimal. Our cash flow
        improved thanks to net banking and upi. Card and cheque are accepted.
        Regards
        """,
    "invoice_with_deposit": """
        ACME PVT LTD
        TAX INVOICE
        Bill To: XYZ Ltd
        Invoice No: 88
        Item Description Qty Rate Amount
        Consulting 10 1000 10000
        Deposit received 3000
        Balance due 7000
        """,
}

EXPECTED = {
    "restaurant_receipt": (True, 18, 0.97, "receipt_or_invoice"),
    "b2b_invoice": (True, 33, 0.97, "receipt_or_invoice"),
    "bank_statement": (False, 0, 0.33, "unknown"),
    "cover_letter": (False, -4, 0.2, "unknown"),
    "text_article": (False, 0, 0.33, "unknown"),
    "pharmacy_receipt": (True, 15, 0.97, "receipt_or_invoice"),
    "restaurant_menu": (False, 0, 0.33, "unknown"),
    "thermal_minimal": (True, 11, 0.88, "receipt"),
    "gibberish_ocr": (False, 0, 0.33, "unknown"),
    "letter_mentioning_invoice": (False, 8, 0.5, "unknown"),
    "one_line_receipt": (True, 8, 0.79, "receipt"),
    "integer_amounts_with_labels": (True, 7, 0.76, "receipt"),
    "integer_amounts_no_labels": (False, 3, 0.46, "unknown"),
    "short_fragment": (False, 0, 0.2, "unknown"),
    "long_single_line_gibberish": (False, 0, 0.33, "unknown"),
    "plain_paragraph": (False, 0, 0.33, "unknown"),
    "invoice_dear_customer": (False, 17, 0.5, "unknown"),
    "invoice_references": (False, 21, 0.5, "unknown"),
    "email_about_invoice": (True, 20, 0.97, "receipt_or_invoice"),
    "invoice_with_deposit": (True, 15, 0.97, "receipt_or_invoice"),
}

ALL_CASES = {**CORPUS, **EXTRA_CASES}


def text_of(name):
    raw = ALL_CASES[name]
    return textwrap.dedent(raw).strip() if "\n" in raw else raw


def test_corpus_and_expectations_are_in_sync():
    assert set(ALL_CASES) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_golden_verdict(name):
    verdict, score, confidence, doc_type = EXPECTED[name]
    result = classify_document(text_of(name))
    assert (
        result["is_receipt_invoice"],
        result["score"],
        result["confidence"],
        result["document_type"],
    ) == (verdict, score, confidence, doc_type), f"{name} changed: {result}"


@pytest.mark.parametrize(
    "name, expected_reasons",
    [
        ("thermal_minimal", ["payment_method", "decimal_amount", "label_then_amount",
                             "quantity_x_price", "amount_on_many_lines"]),
        ("b2b_invoice", ["bill_word", "invoice_word", "invoice_party_block", "due_date",
                         "gst_fields", "totals_block", "item_table_header",
                         "quantity_column", "bank_remittance", "payment_terms"]),
        ("cover_letter", []),
    ],
)
def test_reasons_are_reported(name, expected_reasons):
    result = classify_document(text_of(name))
    for rule_name in expected_reasons:
        assert rule_name in result["reasons"], f"{name}: lost {rule_name}"
    if not expected_reasons:
        assert result["reasons"] == []


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_blocked_by_is_always_populated_when_rejected(name):
    result = classify_document(text_of(name))
    if not result["is_receipt_invoice"]:
        assert result["blocked_by"], f"{name}: rejected with no explanation"
    else:
        assert result["blocked_by"] == []


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_confidence_stays_in_unit_range(name):
    confidence = classify_document(text_of(name))["confidence"]
    assert 0.0 <= confidence <= 1.0


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_is_receipt_or_invoice_agrees_with_classify_document(name):
    text = text_of(name)
    assert is_receipt_or_invoice(text) == classify_document(text)["is_receipt_invoice"]


def test_short_text_is_not_scored():
    result = classify_document("TOTAL 45.50")
    assert result["score"] == 0
    assert result["reasons"] == []
    assert result["counters"] == []
    assert result["blocked_by"] == ["text too short to classify"]
    assert result["document_type"] == "unknown"


@pytest.mark.parametrize("value", [None, "", 12345, [], {}])
def test_empty_or_scalarlike_input_is_unknown(value):
    result = classify_document(value)
    assert result["is_receipt_invoice"] is False
    assert result["document_type"] == "unknown"
    assert result["blocked_by"] == ["text too short to classify"]


def test_counter_rules_reject_non_receipts():
    for name in ("bank_statement", "cover_letter", "text_article", "restaurant_menu", "gibberish_ocr"):
        result = classify_document(text_of(name))
        assert result["is_receipt_invoice"] is False, name
        assert result["document_type"] == "unknown", name


def test_hard_counter_vetoes_the_document():
    result = classify_document(text_of("cover_letter"))
    assert result["hard_counters"] == ["personal_letter"]
    assert result["is_receipt_invoice"] is False
    assert result["blocked_by"] == [
        "vetoed by hard counter: personal_letter",
        "no monetary amount detected, so this has no extractable total",
        "score -4 below threshold 5",
    ]


def test_hard_counter_caps_confidence_at_one_half():
    result = classify_document(text_of("invoice_references"))
    assert result["score"] == 21
    assert result["hard_counters"] == ["resume_or_cv"]
    assert result["confidence"] == 0.5


def test_soft_counter_penalises_without_vetoing():
    result = classify_document(text_of("pharmacy_receipt"))
    assert result["counters"] == ["prescription"]
    assert result["hard_counters"] == []
    assert result["is_receipt_invoice"] is True


def test_score_is_clamped_but_reasons_survive_the_clamp():
    result = classify_document(text_of("bank_statement"))
    assert result["counters"] == ["bank_statement"]
    assert result["score"] == 0
    assert "label_then_amount" in result["reasons"]


def test_confidence_saturates_above_max_score():
    assert dc.RECEIPT_MAX_SCORE < EXPECTED["b2b_invoice"][1]
    assert classify_document(text_of("b2b_invoice"))["confidence"] == 0.97


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A\nA\n\n\n  B  \n\nC", "a\na\n\n\n b \n\nc"),
        ("Invoice No: \uff11.\uff15\uff10", "invoice no: 1.50"),
        ("THANK\u2019S 1,2\u20ac", "thank's 1,2\u20ac"),
        ("  Mixed   CASE \t Text  ", "mixed case text"),
        (None, ""),
    ],
)
def test_normalise(text, expected):
    assert dc._normalise(text) == expected


@pytest.mark.parametrize("char", ["\u200b", "\ufeff"])
def test_normalise_keeps_zero_width_characters(char):
    assert char in dc._normalise(f"Total{char} 10.00")


def test_rule_tables_are_well_formed():
    for table, needs_kind in ((dc.RULES, True), (dc.COUNTER_RULES, False), (dc.STRUCTURE_RULES, False)):
        names = [rule["name"] for rule in table]
        assert len(names) == len(set(names))
        for rule in table:
            assert rule["weight"] > 0
            if needs_kind:
                assert rule["kind"] in {"receipt", "invoice"}
            if table is dc.COUNTER_RULES:
                assert isinstance(rule["hard"], bool)
            assert dc.re.compile(rule["pattern"])


def test_overlapping_structure_rules_double_count_the_same_amount():
    result = classify_document(text_of("one_line_receipt"))
    assert "decimal_amount" in result["reasons"]
    assert "amount_on_many_lines" in result["reasons"]


def test_known_defect_email_mentioning_invoice_is_accepted(name="email_about_invoice"):
    result = classify_document(text_of(name))
    assert result["is_receipt_invoice"] is False, (
        "prose about an invoice is classified as a document with confidence "
        f"{result['confidence']}; payment_method matches bare 'cash'/'card'/'cheque'"
    )


def test_known_defect_dear_customer_vetoes_a_real_invoice():
    result = classify_document(text_of("invoice_dear_customer"))
    assert result["is_receipt_invoice"] is True, (
        f"hard counter {result['hard_counters']} vetoed a complete invoice "
        f"holding score {EXPECTED['invoice_dear_customer'][1]}"
    )


def test_known_defect_references_vetoes_a_real_invoice():
    result = classify_document(text_of("invoice_references"))
    assert result["is_receipt_invoice"] is True, (
        f"hard counter {result['hard_counters']} vetoed a complete invoice "
        f"holding score {EXPECTED['invoice_references'][1]}"
    )


def test_known_defect_whole_number_receipt_is_rejected():
    result = classify_document(text_of("integer_amounts_no_labels"))
    assert result["is_receipt_invoice"] is True, (
        f"receipt in whole units rejected, blocked_by={result['blocked_by']}"
    )


def test_known_defect_bytes_input_is_silently_coerced():
    result = classify_document(b"TOTAL 45.50 CASH 50.00")
    assert result["is_receipt_invoice"] is False, (
        "bytes are stringified to \"b'...'\" and still classify as a receipt"
    )


def test_known_defect_clamp_hides_the_counter_in_blocked_by():
    result = classify_document(text_of("bank_statement"))
    assert any("bank_statement" in reason for reason in result["blocked_by"]), (
        f"blocked_by={result['blocked_by']} blames the score, not the counter"
    )


def test_known_defect_plain_receipt_is_typed_as_receipt_or_invoice():
    result = classify_document(text_of("restaurant_receipt"))
    assert result["document_type"] == "receipt", (
        "gst_fields/totals_block are tagged kind='invoice' but appear on retail receipts"
    )


KNOWN_DEFECT = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="known defect recorded during review of doc_classifier; "
           "delete this marker when the fix lands",
)

for _name, _test in list(globals().items()):
    if _name.startswith("test_known_defect_"):
        globals()[_name] = KNOWN_DEFECT(_test)
