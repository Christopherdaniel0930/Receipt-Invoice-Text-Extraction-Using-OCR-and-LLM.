"""Explainable regex extraction for contact details."""

import re


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\]\[(){}]+", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s().-]*)?(?:\(?\d{2,5}\)?[\s.-]*)?\d{3,5}[\s.-]*\d{3,5}(?:\s*(?:ext\.?|x)\s*\d{1,6})?(?!\w)", re.I)
FAX_LABEL_RE = re.compile(r"\bfax\b\s*[:#-]?\s*(.*)", re.I)
PHONE_LABEL_RE = re.compile(r"\b(?:phone|mobile|cell|tel|telephone)\b\s*[:#-]?\s*(.*)", re.I)
ADDRESS_HINT_RE = re.compile(r"\b(?:street|st\.?|road|rd\.?|avenue|ave\.?|suite|ste\.?|floor|building|blvd\.?|lane|ln\.?|drive|dr\.?|parkway|pkwy|po box|postal|zip|\d{5,6})\b", re.I)


def _clean(value):
    return re.sub(r"\s+", " ", value or "").strip(" ,;\t\r\n") or None


def _valid_phone(value):
    digits = re.sub(r"\D", "", value)
    return len(digits) >= 7 and len(digits) <= 15


def _find_phone(text, label_re):
    for line in text.splitlines():
        label = label_re.search(line)
        if not label:
            continue
        candidate = PHONE_RE.search(label.group(1))
        if candidate and _valid_phone(candidate.group()):
            return _clean(candidate.group())
    return None


def extract_deterministic(text: str) -> dict:
    """Return contact fields supported by the supplied OCR text."""
    text = str(text or "")
    emails = EMAIL_RE.findall(text)
    urls = URL_RE.findall(text)
    urls = [url.rstrip(".,;:") for url in urls]

    fax = _find_phone(text, FAX_LABEL_RE)
    phone = _find_phone(text, PHONE_LABEL_RE)
    if not phone:
        for match in PHONE_RE.finditer(text):
            value = match.group().strip()
            if _valid_phone(value) and not any(
                FAX_LABEL_RE.search(line) and value in line
                for line in text.splitlines()
            ):
                phone = _clean(value)
                break

    address_lines = []
    for line in text.splitlines():
        line = _clean(line)
        if not line:
            continue
        if ADDRESS_HINT_RE.search(line) and not EMAIL_RE.search(line) and not URL_RE.search(line):
            address_lines.append(line)

    return {
        "email": _clean(emails[0]) if emails else None,
        "phone": phone,
        "fax": fax,
        "website": _clean(urls[0]) if urls else None,
        "address": ", ".join(address_lines) if address_lines else None,
    }
