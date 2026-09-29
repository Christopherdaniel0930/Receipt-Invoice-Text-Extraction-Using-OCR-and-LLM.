import re
from datetime import datetime
from app.schema import ReceiptData

CATEGORIES = {
    "Food", "Travel", "Fuel", "Electronics", "Office Supplies",
    "Accommodation", "Healthcare", "Utilities", "Other"
}
CURRENCY_MAP = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR",
    "$": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP",
}

CURRENCY_TOKEN = (
    r"[\u20b9$€£¥₽₩]|\b(?:inr|rs\.?|usd|eur|gbp|jpy|cad|aud|myr)\b"
)
MONEY_ANCHOR_RE = re.compile(
    r"(?:" + CURRENCY_TOKEN + r")\s*[:=]?\s*(\d[\d,]*(?:\.\d{2})?)",
    re.IGNORECASE,
)
GARBLED_LETTER_RATIO = 0.6


def normalize_currency(value):
    if not value:
        return None
    v = str(value).strip().lower()
    return CURRENCY_MAP.get(v, v.upper() if len(v) == 3 else None)


def normalize_amount(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    text = re.sub(r"[^\d,.\-]", "", str(value))
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        text = "".join(parts[:-1]) + "." + parts[-1] if len(parts[-1]) == 2 else text.replace(",", "")
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def normalize_date(value):
    if not value:
        return None
    for fmt in (
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    ):
        try:
            return datetime.strptime(str(value).strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def clean_invoice_number(value):
    if not value:
        return None
    value = re.sub(r"^(invoice|inv|bill|ref(?:erence)?)\s*[#: -]*", "", str(value), flags=re.I).strip()
    return value or None


def normalize_data(data):
    data.invoice_date = normalize_date(data.invoice_date)
    data.currency = normalize_currency(data.currency)
    data.tax_amount = normalize_amount(data.tax_amount)
    data.total_amount = normalize_amount(data.total_amount)
    data.invoice_number = clean_invoice_number(data.invoice_number)
    for item in data.line_items:
        item.quantity = normalize_amount(item.quantity)
        item.unit_price = normalize_amount(item.unit_price)
        item.amount = normalize_amount(item.amount)
    if data.expense_category not in CATEGORIES:
        data.expense_category = None
    return data


def deterministic_validate(data):
    errors = []
    if data.tax_amount is not None and data.tax_amount < 0:
        errors.append("tax_amount negative")
    if data.total_amount is not None and data.total_amount < 0:
        errors.append("total_amount negative")
    if data.tax_amount is not None and data.total_amount is not None and data.tax_amount > data.total_amount:
        errors.append("tax exceeds total")
    return errors


def _letter_ratio(text):
    kept = [ch for ch in str(text) if not ch.isspace()]
    if not kept:
        return 0.0
    letters = sum(1 for ch in kept if ch.isalpha())
    return letters / len(kept)


def is_garbled_description(text):
    """True when a description is mostly OCR damage rather than words.

    A ticket line can come back as "ADU4T (S)343)-R", where the letters
    are few and swamped by stray digits and brackets. Such a string is
    never a real product name, and the amount sitting next to it is
    usually part of the same damage.
    """

    return _letter_ratio(text) < GARBLED_LETTER_RATIO


def _line_is_garbled(line, match):
    """True when everything on the line apart from the amount is damage."""

    remainder = (line[: match.start()] + line[match.end() :]).strip()
    return bool(remainder) and is_garbled_description(remainder)


def money_anchors(text):
    """Every amount on the page that a currency symbol sits next to.

    A symbol in front of a number is the one amount marker on these
    documents that OCR cannot invent, because the symbol has to survive
    for the pair to be found at all.

    Each entry is (amount, line, trustworthy), where trustworthy is False
    when the text around the amount is garbled, as in
    "ADU4T (S)343)-R ₹5729.00" where 5729.00 is part of the damage
    rather than a real price.
    """

    found = []
    for line in str(text or "").split("\n"):
        for match in MONEY_ANCHOR_RE.finditer(line):
            amount = normalize_amount(match.group(1))
            if amount is not None:
                found.append(
                    (amount, line, not _line_is_garbled(line, match))
                )
    return found


def reconcile_totals(data, ocr_text):
    """Fill a missing total and drop amounts that are OCR damage.

    Two defects are corrected here, both seen on toll tickets where the
    fare is printed once with no "Total" label:

    * The LLM puts the fare in line_items and leaves total_amount null,
      because the prompt only asks for an explicitly labelled total.
    * A mangled line yields a confident but wrong amount, as in
      "ADU4T (S)343)-R" priced at 5729.00 when the real fare is 129.00.

    Returns a list of notes describing what was changed, so a caller can
    tell a filled-in total from one the model stated.
    """

    notes = []

    all_anchors = money_anchors(ocr_text)
    trusted = [amount for amount, _, ok in all_anchors if ok]
    untrusted = [amount for amount, _, ok in all_anchors if not ok]

    kept = []
    dropped_amounts = []
    for item in data.line_items:
        if is_garbled_description(item.description):
            notes.append(
                "dropped garbled line item %r" % (item.description,)
            )
            if item.amount is not None:
                dropped_amounts.append(item.amount)
            continue
        kept.append(item)

    data.line_items = kept

    if data.total_amount is not None:
        # A total that matches a price we just rejected as OCR damage is
        # the same damage read twice, so it is cleared and re-derived.
        if data.total_amount in dropped_amounts and data.total_amount not in trusted:
            notes.append(
                "cleared total %r taken from a garbled line" % data.total_amount
            )
            data.total_amount = None
        else:
            return notes

    item_amounts = [
        item.amount for item in data.line_items if item.amount is not None
    ]

    if len(item_amounts) == 1 and item_amounts[0] in trusted:
        data.total_amount = item_amounts[0]
        notes.append("total taken from the single line item")
        return notes

    if item_amounts and sum(item_amounts) in trusted:
        data.total_amount = round(sum(item_amounts), 2)
        notes.append("total taken from the line item sum")
        return notes

    # An amount that only survived because a confusable symbol was patched
    # in front of a garbled line is not evidence of a total, so it is never
    # used on its own. A total the model stated is left alone above.
    if len(trusted) == 1:
        data.total_amount = trusted[0]
        notes.append("total taken from the only currency-anchored amount")
        return notes

    if not trusted and not item_amounts:
        notes.append(
            "no trustworthy amount on the page; total left empty"
        )

    return notes

