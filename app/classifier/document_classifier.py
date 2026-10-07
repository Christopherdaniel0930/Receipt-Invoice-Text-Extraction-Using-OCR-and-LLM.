"""Standalone rule-based receipt/invoice detector.

Decides whether a block of OCR text looks like a receipt or an invoice
without calling any LLM.

The module is self-contained: it imports only the standard library and
nothing from the rest of the package, so it can be used on its own as
well as from the pipeline, where app.main gates on its hard counters.

Usage:
    from app.classifier.document_classifier import classify_document, is_receipt_or_invoice

    if is_receipt_or_invoice(ocr_text):
        ...

    result = classify_document(ocr_text)
    result["is_receipt_invoice"], result["score"], result["reasons"]
"""

import re
import unicodedata

RECEIPT_MIN_SCORE = 5
RECEIPT_MAX_SCORE = 14

STRUCTURE_ONLY_MIN_SCORE = RECEIPT_MIN_SCORE + 2

RULES = (
    {
        "name": "receipt_word",
        "kind": "receipt",
        "weight": 3,
        "pattern": r"\breceipts?\b",
    },
    {
        "name": "change_due",
        "kind": "receipt",
        "weight": 3,
        "pattern": r"\b(change\s*due|cash\s*tendered|amount\s*returned|balance\s*change)\b",
    },
    {
        "name": "cashier",
        "kind": "receipt",
        "weight": 2,
        "pattern": r"\bcashier\b|\b(cash\s*counter|register\s*(no|id)|till\s*(no|id)|counter\s*(no|id))\b",
    },
    {
        "name": "payment_method",
        "kind": "receipt",
        "weight": 2,
        "pattern": (
            r"\b(visa|mastercard|maestro|amex|discover|rupay|debit|credit)\b"
            r"|\b(card|cash|upi|net\s*banking|imps|neft|rtgs|cheque)\b"
            r"|\bcheck\s*(no|#|number)\b"
        ),
    },
    {
        "name": "pos_terminal_ref",
        "kind": "receipt",
        "weight": 3,
        "pattern": (
            r"\b(terminal\s*id|mid|merchant\s*id|auth(orization)?\s*code"
            r"|approval\s*(code|no)|txn\s*(ref|id)|transaction\s*(id|no)"
            r"|ref(no)?\s*(no|#|id|number))\b"
        ),
    },
    {
        "name": "retail_footer",
        "kind": "receipt",
        "weight": 2,
        "pattern": r"\bthank\s*(you|u)\b|\bplease\s*(come\s*again|retain)\b",
    },
    {
        "name": "bill_word",
        "kind": "receipt",
        "weight": 2,
        "pattern": r"\bbill\b|\bbill\s*(no|number|#)\b",
    },
    {
        "name": "invoice_word",
        "kind": "invoice",
        "weight": 3,
        "pattern": r"\binvoices?\b",
    },
    {
        "name": "invoice_party_block",
        "kind": "invoice",
        "weight": 2,
        "pattern": (
            r"\b(bill(?:ing)?\s*(to|address|from)|ship\s*to|invoice\s*to"
            r"|supplier|customer\s*(details|no|name|address))\b"
        ),
    },
    {
        "name": "due_date",
        "kind": "invoice",
        "weight": 3,
        "pattern": (
            r"\b(due\s*date|payment\s*due|due\s*on|due\s*by|net\s*\d+\s*days"
            r"|payable\s*(by|within|on)|within\s*\d+\s*days)\b"
        ),
    },
    {
        "name": "gst_fields",
        "kind": "invoice",
        "weight": 3,
        "pattern": (
            r"\b(gstin|gst\s*(no|number|in|amt|amount|rate)?|cgst|sgst|igst"
            r"|hsn\s*code|hsn|place\s*of\s*supply)\b"
        ),
    },
    {
        "name": "tax_registration",
        "kind": "invoice",
        "weight": 2,
        "pattern": r"\b(vat|vat\s*(no|number|reg)|tax\s*(id|no|number)|pan|tin\s*(no|number)|abn|acn|ein|siren|tva)\b",
    },
    {
        "name": "totals_block",
        "kind": "invoice",
        "weight": 3,
        "pattern": (
            r"\b(sub\s*-?\s*total|grand\s*total|net\s*amount|net\s*payable"
            r"|total\s*tax|total\s*due|total\s*paid|amount\s*due"
            r"|amount\s*payable|balance\s*due|amount\s*chargeable)\b"
        ),
    },
    {
        "name": "item_table_header",
        "kind": "invoice",
        "weight": 2,
        "pattern": (
            r"\b(description|particulars|item\s*(name|code|desc)"
            r"|unit\s*price|unit\s*cost|hsn\s*code)\b"
        ),
    },
    {
        "name": "quantity_column",
        "kind": "invoice",
        "weight": 2,
        "pattern": r"\b(qty|quantity|qty\.?|nos|units)\b",
    },
    {
        "name": "purchase_order_ref",
        "kind": "invoice",
        "weight": 1,
        "pattern": r"\b(purchase\s*order|p\.?\s?o\.?\s*(no|number|#)|so\s*(no|number))\b",
    },
    {
        "name": "bank_remittance",
        "kind": "invoice",
        "weight": 2,
        "pattern": (
            r"\b(remit\s*to|beneficiary|account\s*(name|no|number)"
            r"|ifsc|swift|sort\s*code|iban)\b"
        ),
    },
    {
        "name": "payment_terms",
        "kind": "invoice",
        "weight": 2,
        "pattern": (
            r"\b(payment\s*terms|terms\s*(and|&)\s*conditions"
            r"|late\s*payment|interest\s*(at|per)|overdue)\b"
        ),
    },
    {
        "name": "order_confirmation",
        "kind": "receipt",
        "weight": 1,
        "pattern": r"\border\s*(confirmed|confirmation|status)\b|\bthank\s*you\s*for\s*(your\s*)?(order|purchase)\b",
    },
)

COUNTER_RULES = (
    {
        "name": "bank_statement",
        "kind": "counter",
        "weight": 4,
        "hard": False,
        "pattern": (
            r"\b(opening\s*balance|closing\s*balance|available\s*balance"
            r"|statement\s*of\s*account|account\s*summary"
            r"|withdrawal|deposit|transaction\s*date|value\s*date)\b"
        ),
    },
    {
        "name": "identity_document",
        "kind": "counter",
        "weight": 4,
        "hard": True,
        "pattern": (
            r"\b(date\s*of\s*birth|place\s*of\s*birth|nationality"
            r"|passport\s*no|driver'?s?\s*licen[cs]e|id\s*card"
            r"|government\s*id|identification\s*number|sex\s*:\s*(m|f|male|female))\b"
        ),
    },
    {
        "name": "personal_letter",
        "kind": "counter",
        "weight": 4,
        "hard": True,
        "pattern": (
            r"\bdear\s+(sir|madam|all|[a-z]+)\b|\b(sincerely|kindly\s*regards|"
            r"yours\s*sincerely|best\s*regards|warm\s*regards)\b"
        ),
    },
    {
        "name": "resume_or_cv",
        "kind": "counter",
        "weight": 5,
        "hard": True,
        "pattern": (
            r"\b(curriculum\s*vitae|work\s*experience|educational\s*qualification"
            r"|employment\s*history|skills\s*section|career\s*objective|references)\b"
        ),
    },
    {
        "name": "menu_or_catalogue",
        "kind": "counter",
        "weight": 3,
        "hard": False,
        "pattern": (
            r"\b(appetizer|starter|main\s*course|dessert|beverages?"
            r"|our\s*menu|price\s*list|menu\s*:\s*$|all\s*taxes\s*excluded"
            r"|subject\s*to\s*availability)\b"
        ),
    },
    {
        "name": "legal_contract",
        "kind": "counter",
        "weight": 4,
        "hard": True,
        "pattern": (
            r"\b(this\s+agreement|hereby|whereas|hereinafter|hereinafter\s+referred"
            r"|party\s+of\s+the\s+first\s+part|clause\s+\d|termination\s+of\s+this"
            r"|governing\s*law)\b"
        ),
    },
    {
        "name": "news_or_article",
        "kind": "counter",
        "weight": 3,
        "hard": True,
        "pattern": (
            r"\b(published\s+on|by\s+[a-z]+\s+reporter|reuters|associated\s+press"
            r"|copyright\s+all\s+rights\s+reserved|read\s+more|click\s+here)\b"
        ),
    },
    {
        "name": "prescription",
        "kind": "counter",
        "weight": 3,
        "hard": False,
        "pattern": (
            r"\b(rx|prescribed\s+by|dosage|tablets?|capsules?\b"
            r"|take\s+\d+\s*tablets?\s+after)\b"
        ),
    },
)

CONTACT_BLOCK_RE = re.compile(
    r"(?:\be-?mail\b|https?://|\bwww\.|\bfax\b|\btel\b|\bphone\b"
    r"|\bph\.?\s*:|\bcell\b|\bmobile\b|\bext\.?\s*\d)"
    r"|[\w.+-]+@[\w-]+\.[\w.]+"
    r"|\b(?:inc|llc|ltd|pvt|plc|corp)\b",
    re.IGNORECASE,
)
CONTACT_BLOCK_MIN_MARKERS = 2


def _looks_like_contact_card(normalised, signals, has_money):
    """A letterhead or business card: contact details, and no money.

    A phone number alone proves nothing, because plenty of real receipts
    print one. What separates a contact card is a block of contact
    details with no money on the page at all, so the money test is part
    of the rule rather than a detail.

    has_money is preferred over the raw amount count because an invoice
    can print its figures as bare integers ("10 1000 10000") that carry
    no currency symbol or decimal point and so never reach AMOUNT_RE.
    """

    if has_money or signals["amounts"] or signals["amount_lines"]:
        return False

    return len(CONTACT_BLOCK_RE.findall(normalised)) >= CONTACT_BLOCK_MIN_MARKERS


STRUCTURE_RULES = (
    {
        "name": "currency_amount",
        "weight": 2,
        "pattern": (
            r"(?:[\u20b9$€£¥₽₩]|\b(?:inr|rs\.?|usd|eur|gbp|jpy|cad|aud|myr|idr|php|sgd)\b)\s*-?\d"
        ),
    },
    {
        "name": "decimal_amount",
        "weight": 1,
        "pattern": r"\b\d+(?:,\d{3})*\.\d{2}\b",
    },
    {
        "name": "label_then_amount",
        "weight": 3,
        "pattern": (
            r"\b(total|amount|grand\s*total|sub\s*-?\s*total|net\s*(amount|payable)"
            r"|balance|tax|gst|vat|price|cost|rate|paid|due)\b[^\n]{0,40}?"
            r"(?:[\u20b9$€£¥]|\b(?:inr|usd|eur|gbp)\b)?\s*-?\d"
        ),
    },
    {
        "name": "quantity_x_price",
        "weight": 3,
        "pattern": r"\b\d+(?:\.\d+)?\s*(?:x|\u00d7|@|\*)\s*[\d,.]+|\b\d+\s*@\s*[\d,.]+",
    },
    {
        "name": "order_or_receipt_id",
        "weight": 2,
        "pattern": r"\b(invoice|bill|receipt|order|txn|transaction|doc(ument)?|ref)\s*(no|number|num|#|id)\b[^\n]{0,20}\w",
    },
    {
        "name": "date_present",
        "weight": 1,
        "pattern": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4}\b",
    },
    {
        "name": "amount_on_many_lines",
        "weight": 2,
        "pattern": r"(?:\d[\d,]*\.\d{2})(?:\s|$)",
    },
)

CHARACTER_FLOOR = 24
MIN_DOCUMENT_LINES = 2

DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"
)
AMOUNT_RE = re.compile(
    r"[\u20b9$€£¥₽₩]\s*-?\d+(?:,\d{3})*(?:\.\d{2})?"
    r"|\b(?:inr|rs\.?|usd|eur|gbp|jpy)\s*-?\d+(?:,\d{3})*(?:\.\d{2})?"
    r"|\b\d+(?:,\d{3})*\.\d{2}\b"
)


def _normalise(text):
    """Lowercase, strip unicode oddities and collapse whitespace."""

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _match(rules, text):
    """Return [(name, weight), ...] for every rule whose pattern hits."""

    hits = []

    for rule in rules:
        if re.search(rule["pattern"], text, flags=re.IGNORECASE | re.MULTILINE):
            hits.append((rule["name"], rule["weight"]))

    return hits


def _confidence(score, is_receipt_invoice):
    """Map a raw score onto a monotonic 0-1 confidence."""

    if is_receipt_invoice:
        span = max(1, RECEIPT_MAX_SCORE - RECEIPT_MIN_SCORE)
        ratio = min(1.0, max(0.0, (score - RECEIPT_MIN_SCORE) / span))
        return round(0.7 + 0.27 * ratio, 2)

    span = max(1, RECEIPT_MIN_SCORE + 3)
    ratio = min(1.0, max(0.0, (RECEIPT_MIN_SCORE - score) / span))

    return round(0.55 - 0.35 * ratio, 2)


def classify_document(text, gpt_client=None):
    """Classify OCR text, optionally using the configured LLM for the final type.

    Returns a dict with:
        is_receipt_invoice : bool
        score              : int, weighted evidence minus penalties
        confidence         : float, 0-1
        document_type      : receipt/invoice, business_card, or unsupported_document
        llm_document_type   : LLM label when a client is supplied
        reasons            : list[str], rule names that fired
        counters           : list[str], competing-document rules that fired
        hard_counters      : list[str], counters that veto the decision
        signals            : dict of structural observations
    """

    normalised = _normalise(text)
    lines = [line for line in normalised.split("\n") if line.strip()]

    if len(normalised) < CHARACTER_FLOOR and len(lines) < MIN_DOCUMENT_LINES:
        result = {
            "is_receipt_invoice": False,
            "score": 0,
            "confidence": 0.2,
            "document_type": "unknown",
            "reasons": [],
            "counters": [],
            "signals": {
                "chars": len(normalised),
                "lines": len(lines),
                "amounts": len(AMOUNT_RE.findall(normalised)),
                "dates": len(DATE_RE.findall(normalised)),
                "amount_lines": 0,
            },
            "hard_counters": [],
            "blocked_by": ["text too short to classify"],
        }
        return _apply_llm_classification(result, text, gpt_client)

    positive_hits = _match(RULES, normalised)
    counter_hits = _match(COUNTER_RULES, normalised)
    structure_hits = _match(STRUCTURE_RULES, normalised)

    score = sum(weight for _, weight in positive_hits)
    score += sum(weight for _, weight in structure_hits)
    score -= sum(weight for _, weight in counter_hits)

    reasons = [name for name, _ in positive_hits + structure_hits]

    amounts = AMOUNT_RE.findall(normalised)
    amount_lines = sum(1 for line in lines if AMOUNT_RE.search(line))
    signals = {
        "chars": len(normalised),
        "lines": len(lines),
        "amounts": len(amounts),
        "dates": len(DATE_RE.findall(normalised)),
        "amount_lines": amount_lines,
    }

    names = {name for name, _ in positive_hits}
    has_identity = bool(
        {
            "receipt_word",
            "change_due",
            "pos_terminal_ref",
            "invoice_word",
            "due_date",
            "gst_fields",
            "totals_block",
        }
        & names
    )
    structure_names = {name for name, _ in structure_hits}
    has_money = bool(
        structure_names
        & {
            "currency_amount",
            "decimal_amount",
            "label_then_amount",
            "quantity_x_price",
        }
    )

    hard_counters = [
        rule["name"] for rule in COUNTER_RULES if rule.get("hard") and rule["name"] in {n for n, _ in counter_hits}
    ]

    if _looks_like_contact_card(normalised, signals, has_money):
        counter_hits = counter_hits + [("business_card", 4)]
        reasons.append("business_card")
        hard_counters.append("business_card")

    if counter_hits and not has_identity:
        score = min(score, 0)

    is_receipt_invoice = (
        not hard_counters
        and score >= RECEIPT_MIN_SCORE
        and has_money
        and (has_identity or score >= STRUCTURE_ONLY_MIN_SCORE)
    )

    blocked_by = []

    if is_receipt_invoice:
        pass
    else:
        if hard_counters:
            blocked_by.append(
                "vetoed by hard counter: " + ", ".join(hard_counters)
            )
        if not has_money:
            blocked_by.append(
                "no monetary amount detected, so this has no extractable total"
            )
        if score < RECEIPT_MIN_SCORE:
            blocked_by.append(
                f"score {score} below threshold {RECEIPT_MIN_SCORE}"
            )
        elif not has_identity and score < STRUCTURE_ONLY_MIN_SCORE:
            blocked_by.append(
                f"score {score} below {STRUCTURE_ONLY_MIN_SCORE} "
                "and no strong document marker"
            )

    if is_receipt_invoice:
        has_receipt = any(
            rule["name"] in names and rule["kind"] == "receipt"
            for rule in RULES
        )
        has_invoice = any(
            rule["name"] in names and rule["kind"] == "invoice"
            for rule in RULES
        )
        if has_receipt and has_invoice:
            document_type = "receipt_or_invoice"
        elif has_invoice:
            document_type = "invoice"
        else:
            document_type = "receipt"
    else:
        document_type = "unknown"

    if counter_hits and not is_receipt_invoice:
        document_type = "unknown"

    confidence = _confidence(score, is_receipt_invoice)

    if hard_counters:
        confidence = min(confidence, 0.5)

    result = {
        "is_receipt_invoice": is_receipt_invoice,
        "score": score,
        "confidence": confidence,
        "document_type": document_type,
        "reasons": reasons,
        "counters": [name for name, _ in counter_hits],
        "hard_counters": hard_counters,
        "blocked_by": blocked_by,
        "signals": signals,
    }
    return _apply_llm_classification(result, text, gpt_client)


def _apply_llm_classification(result, text, gpt_client):
    if gpt_client is None:
        return result
    classification = gpt_client.classify_document(str(text or ""))
    kind = classification["document_type"]
    result["document_type"] = {
        "receipt_invoice": "receipt_or_invoice",
        "business_card": "business_card",
        "unsupported_document": "unsupported_document",
    }[kind]
    result["is_receipt_invoice"] = kind == "receipt_invoice"
    result["confidence"] = classification["confidence"]
    result["llm_document_type"] = kind
    if kind != "receipt_invoice":
        result["blocked_by"] = [kind]
    return result


def is_receipt_or_invoice(text, gpt_client=None):
    """Convenience boolean wrapper around classify_document."""

    return classify_document(text, gpt_client)["is_receipt_invoice"]


if __name__ == "__main__":
    SAMPLES = {
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

    for label, sample in SAMPLES.items():
        result = classify_document(sample)
        print(f"{label:20} -> {str(result['is_receipt_invoice']):5} "
              f"score={result['score']:>4} conf={result['confidence']:.2f} "
              f"type={result['document_type']}")
        if result["reasons"]:
            print(f"{'':22}why: {', '.join(result['reasons'])}")
        if result["counters"]:
            print(f"{'':22}not: {', '.join(result['counters'])}"
                  f"{'  [VETO]' if result['hard_counters'] else ''}")
        print()
