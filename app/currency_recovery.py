"""Standalone currency recovery layer.

OCR drops currency symbols constantly: a glyph is clipped by the text
detector, a rupee sign is read as "R8", a dollar sign comes back as "S",
or the line is discarded outright because its score fell under the
threshold. This module re-derives the currency that the OCR text is
missing, and can write the symbol back into the amounts that lost it.

Three evidence layers are combined, strongest first:

1. glyph_crop  - crop the ink immediately left of a recognised amount and
                 re-run the PP-OCR recogniser on that crop alone. The
                 recogniser sees one large glyph instead of a whole
                 receipt line, which is where the symbol usually comes
                 back. This is the only layer that looks at pixels.
2. ocr_symbol  - a currency symbol or ISO code that did survive in the
                 OCR text, used to settle the rest of the document.
3. confusable  - a letter that is a common OCR stand-in for a symbol
                 ("R8 17.00", "S 12.00"), accepted only when it is glued
                 to a number.
4. locale      - regional markers such as GSTIN or UPI. Capped low
                 enough that it can never decide alone, only break ties.

The extraction pipeline calls recover_currency before the LLM, writes the
recovered symbol back into the OCR text, and falls back to the recovered
ISO code when the LLM could not name a currency. The CLI below stays
available for inspecting a single receipt:

    python -m app.currency_recovery data/test/image6.jpeg
    python -m app.currency_recovery receipt.jpg --patch
    python -m app.currency_recovery receipt.jpg --json
    python -m app.currency_recovery receipt.jpg --save-crops crops/
    python -m app.currency_recovery receipt.jpg --text-only --min-score 0.3

A .txt input skips the pixel layer and scores the text alone.
"""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPEATED_DECAY = 0.6

SYMBOL_WEIGHT = 1.0
CODE_WEIGHT = 0.9
PREFIX_WEIGHT = 0.9
CONFUSABLE_WEIGHT = 0.35
LOCALE_CAP = 0.5

MIN_SCORE = 0.45
DIRECT_CONFIDENCE_CAP = 0.95
INFERRED_CONFIDENCE_CAP = 0.7

GLYPH_SCALES = (2, 4, 6)
GLYPH_MIN_SCORE = 0.35
GLYPH_MIN_HITS = 2
MAX_CROPS = 24

INK_LEVEL = 200
INK_MIN_PIXELS = 6

LINE_RULE = "-" * 72

DIRECT_SOURCES = ("ocr_symbol", "ocr_code", "glyph_crop")

SYMBOL_TO_CODE = {
    "\u20b9": "INR",
    "$": "USD",
    "US$": "USD",
    "S$": "SGD",
    "SGD": "SGD",
    "A$": "AUD",
    "AU$": "AUD",
    "C$": "CAD",
    "CA$": "CAD",
    "R$": "BRL",
    "NZ$": "NZD",
    "HK$": "HKD",
    "\u20a6": "NGN",
    "\u20b1": "PHP",
    "\u20b4": "UAH",
    "\u20a9": "KRW",
    "\u20aa": "ILS",
    "\u20ab": "ILS",
    "\u20ac": "EUR",
    "\u20ad": "TRY",
    "\u20b8": "THB",
    "\u20ba": "TRY",
    "\u20bc": "GBP",
    "\u20bd": "RUB",
    "\u20bf": "BTC",
    "\u20c6": "INR",
    "\u20a1": "MXN",
    "\u20b2": "PHP",
    "\u0e3f": "THB",
    "\u0e3b": "THB",
    "\u20eb": "VND",
    "\u20ab": "ILS",
    "\u20a3": "GBP",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "元": "CNY",
    "\u20a5": "KRW",
    "\u20b7": "RUB",
    "z\u0142": "PLN",
}

CODE_TO_SYMBOL = {
    "AED": "AED",
    "AUD": "A$",
    "BRL": "R$",
    "CAD": "C$",
    "CHF": "CHF",
    "CNY": "\u00a5",
    "EUR": "\u20ac",
    "GBP": "\u00a3",
    "HKD": "HK$",
    "IDR": "Rp",
    "ILS": "\u20aa",
    "INR": "\u20b9",
    "JPY": "\u00a5",
    "KRW": "\u20a9",
    "MYR": "RM",
    "NGN": "\u20a6",
    "NOK": "kr",
    "NZD": "NZ$",
    "PHP": "\u20b1",
    "PLN": "z\u0142",
    "RUB": "\u20bd",
    "SAR": "SAR",
    "SEK": "kr",
    "SGD": "S$",
    "THB": "\u0e3f",
    "TRY": "\u20ba",
    "TWD": "NT$",
    "USD": "$",
    "VND": "\u20ab",
    "ZAR": "R",
}

PREFIX_TO_CODE = {
    "rs": "INR",
    "rs.": "INR",
    "rp": "IDR",
    "rm": "MYR",
    "nt$": "TWD",
    "a$": "AUD",
    "c$": "CAD",
    "s$": "SGD",
    "hk$": "HKD",
    "nz$": "NZD",
    "r$": "BRL",
    "us$": "USD",
    "z\u0142": "PLN",
    "ft": "HUF",
    "kr": "SEK",
    "\u0e3f": "THB",
}

ISO_CODES = (
    "AED", "AUD", "BRL", "CAD", "CHF", "CNY", "EUR", "GBP", "HKD", "IDR",
    "ILS", "INR", "JPY", "KRW", "MYR", "NGN", "NOK", "NZD", "PHP", "PLN",
    "RUB", "SAR", "SEK", "SGD", "THB", "TRY", "TWD", "USD", "VND", "ZAR",
)

CURRENCY_CHARS = "".join(sorted(set(SYMBOL_TO_CODE)))
CURRENCY_CLASS = re.escape(CURRENCY_CHARS)

SYMBOL_RE = re.compile(f"[{CURRENCY_CLASS}]")
ISO_RE = re.compile(r"\b(" + "|".join(ISO_CODES) + r")\b", re.IGNORECASE)
MONEY_RE = re.compile(
    r"(?<![\d.,/])-?\d{1,3}(?:,\d{3})+(?:\.\d{2})?(?![\d.,/])"
    r"|(?<![\d.,/])-?\d+\.\d{2}(?![\d.,/])"
)
STANDALONE_AMOUNT_RE = re.compile(r"^\s*(?:\d{1,3}(?:,\d{3})+|\d{1,8})\s*$")
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
IDENTITY_RE = re.compile(
    r"\b(?:invoice|inv|bill|bill\s*no|gstin|gst\s*no|pan|a/?c|account|ifsc"
    r"|ph|tel|fax|mobile|order\s*(?:no|id)|txn|utr|rrn|ref|receipt\s*no"
    r"|lic|licence|license|permit|reg|code)\b"
    r"|@|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    re.IGNORECASE,
)

CONFUSABLE_RE = re.compile(
    r"(?<![\w.])(rs\.?|r8|rp|rm|nt\$|s|s5|us\$|a\$|c\$|c|eu|eur|"
    r"u\$|f|\u00a3|y|yen|jpy|w|\u20a9|p|py|rb|rub|brl|r|thb|b|\u0e3f|"
    r"rmb|cny|sgd|myr|idr|php|aud|cad|chf|aed|sar|zar|nzd|hkd|twd)\b"
    r"\s*[:.]?\s*-?\d",
    re.IGNORECASE,
)

SINGLE_LETTER_CONFUSABLES = {
    "s": "USD",
    "s5": "USD",
    "c": "EUR",
    "eu": "EUR",
    "f": "GBP",
    "y": "JPY",
    "yen": "JPY",
    "w": "KRW",
    "p": "IDR",
    "py": "INR",
    "b": "THB",
}

MULTI_LETTER_CONFUSABLES = {
    "rs": "INR",
    "rs.": "INR",
    "r8": "INR",
    "rp": "IDR",
    "rm": "MYR",
    "r": "RUB",
    "rb": "BRL",
    "rub": "RUB",
    "brl": "BRL",
    "thb": "THB",
    "jpy": "JPY",
    "rmb": "CNY",
    "cny": "CNY",
    "sgd": "SGD",
    "myr": "MYR",
    "idr": "IDR",
    "php": "PHP",
    "aud": "AUD",
    "cad": "CAD",
    "chf": "CHF",
    "aed": "AED",
    "sar": "SAR",
    "zar": "ZAR",
    "nzd": "NZD",
    "hkd": "HKD",
    "twd": "TWD",
}

LOCALE_RULES = (
    ("india_gst", r"\b(?:cgst|sgst|igst|gstin|gst)\b", "INR"),
    ("india_upi", r"\b(?:upi|paytm|phone\s*pe|phonepe|gpay|neft|imps)\b", "INR"),
    ("india_words", r"\b(?:rupees?|paise|inr)\b", "INR"),
    ("india_bank", r"\b(?:hdfc|icici|indusind|state\s*bank|sbi|kotak|"
                   r"indian\s*bank|punjab\s*national)\b", "INR"),
    ("uk_vat", r"\b(?:vat\s*(?:no|reg|number)?|hmrc|companies\s*house)\b", "GBP"),
    ("uk_words", r"\b(?:pounds?|pence|sterling|gbp)\b", "GBP"),
    ("euro_vat", r"\b(?:ust\.?id|tva|more\s*than\s*twenty|umsatzsteuer)\b", "EUR"),
    ("euro_words", r"\b(?:euros?|eur)\b", "EUR"),
    ("us_words", r"\b(?:dollars?|cents?|usd)\b", "USD"),
    ("jp_words", r"\b(?:yen|jpy|\u5186)\b", "JPY"),
    ("ae_words", r"\b(?:dirhams?|aed|dubai|abu\s*dhabi)\b", "AED"),
    ("sa_words", r"\b(?:riyals?|sar|saudi\s*riyal)\b", "SAR"),
    ("sg_words", r"\b(?:sgd|singapore\s*dollar|gst\s*reg)\b", "SGD"),
    ("my_words", r"\b(?:ringgit|myr|malaysian\s*ringgit)\b", "MYR"),
    ("br_words", r"\b(?:reais?|real|brazil|nota\s*fiscal)\b", "BRL"),
    ("kr_words", r"\b(?:won|krw)\b", "KRW"),
)

_ENGINE = None


_ENGINE = None


def get_engine():
    """Return the engine used for crop recognition, created once per process.

    This is deliberately NOT the pipeline's engine. Calling a shared
    RapidOCR with use_det=False poisons it: every later full-image run
    comes back with zero rows, so the next receipt in a batch reads as
    empty. Keep the crop recogniser on its own instance.
    """

    global _ENGINE

    if _ENGINE is None:
        from rapidocr import RapidOCR

        _ENGINE = RapidOCR()

    return _ENGINE


def read_rows(image_path):
    """Run OCR and return [{text, score, box}, ...] with a 4-number box.

    Delegates to app.ocr so the pipeline and this module never disagree
    about what the OCR engine produced. The pipeline passes the rows it
    already has instead of calling this, which keeps it to one pass.
    """

    from app.ocr import read_ocr_rows

    source = Path(image_path)

    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source.resolve()}")

    return read_ocr_rows(str(source.resolve()))


def _as_float(value):
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _normalise(text):
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2010-\u2015]", "-", text)

    return text


def symbol_to_code(token):
    """Map one OCR token to an ISO code, or None if it is not currency."""

    if not token:
        return None

    text = _normalise(token).strip()
    upper = text.upper()

    if text in SYMBOL_TO_CODE:
        return SYMBOL_TO_CODE[text]

    if upper in SYMBOL_TO_CODE:
        return SYMBOL_TO_CODE[upper]

    lowered = text.lower()

    if lowered in PREFIX_TO_CODE:
        return PREFIX_TO_CODE[lowered]

    if upper in ISO_CODES:
        return upper

    for symbol in SYMBOL_TO_CODE:
        if symbol.isascii() and symbol in text:
            return SYMBOL_TO_CODE[symbol]

    return None


def _repeated(total, base):
    """Accumulate evidence with each further sighting worth less."""

    if total <= 0:
        return round(base, 3)

    return round(total + base * REPEATED_DECAY ** total, 3)


def read_symbols(text):
    """Every currency symbol or ISO code the OCR text still carries."""

    found = []

    for line in _normalise(text).split("\n"):
        if not line.strip():
            continue

        for match in SYMBOL_RE.finditer(line):
            code = SYMBOL_TO_CODE.get(match.group(0))
            if code:
                found.append(
                    {
                        "code": code,
                        "symbol": match.group(0),
                        "source": "ocr_symbol",
                        "detail": match.group(0),
                        "line": line.strip(),
                    }
                )

        for match in ISO_RE.finditer(line):
            found.append(
                {
                    "code": match.group(1).upper(),
                    "symbol": match.group(1).upper(),
                    "source": "ocr_code",
                    "detail": match.group(1),
                    "line": line.strip(),
                }
            )

        for match in re.finditer(r"(?<![\w.])(rs\.?|rp|rm|nt\$)\s*[:.]?\s*-?\d",
                                 line, re.IGNORECASE):
            code = PREFIX_TO_CODE.get(match.group(1).lower().rstrip("."))
            if code:
                found.append(
                    {
                        "code": code,
                        "symbol": match.group(1),
                        "source": "ocr_symbol",
                        "detail": match.group(1),
                        "line": line.strip(),
                    }
                )

    return found


def read_confusables(text):
    """Letters glued to a number that OCR may have flattened a symbol into.

    A single letter is only trusted when it repeats on a second line,
    because "S 12.00" is a dollar sign but "1 x 2" is not.
    """

    found = []
    seen_single = {}

    for line in _normalise(text).split("\n"):
        if not line.strip() or DATE_RE.search(line):
            continue

        for match in CONFUSABLE_RE.finditer(line):
            token = match.group(1)
            lowered = token.lower().rstrip(".")

            if lowered in MULTI_LETTER_CONFUSABLES:
                code = MULTI_LETTER_CONFUSABLES[lowered]
                detail = match.group(0).strip()
            elif lowered in SINGLE_LETTER_CONFUSABLES:
                code = SINGLE_LETTER_CONFUSABLES[lowered]
                detail = match.group(0).strip()
                seen_single[code] = seen_single.get(code, 0) + 1
            else:
                continue

            found.append(
                {
                    "code": code,
                    "symbol": token,
                    "source": "confusable",
                    "detail": detail,
                    "line": line.strip(),
                }
            )

    trusted = []

    for item in found:
        token = item["detail"].rstrip("0123456789 ,.:-").lower().rstrip(".")
        single = token in SINGLE_LETTER_CONFUSABLES and len(token) == 1

        if single and seen_single.get(item["code"], 0) < 2:
            continue

        trusted.append(item)

    return trusted


def read_locale(text):
    """Regional markers, weakest layer, capped so it never decides alone."""

    found = []
    normalised = _normalise(text)
    tally = {}

    for name, pattern, code in LOCALE_RULES:
        if not re.search(pattern, normalised, re.IGNORECASE):
            continue

        tally[code] = tally.get(code, 0) + 1
        found.append(
            {
                "code": code,
                "symbol": CODE_TO_SYMBOL.get(code, code),
                "source": "locale",
                "detail": name,
                "line": "",
            }
        )

    counts = {}

    for item in found:
        counts[item["code"]] = counts.get(item["code"], 0) + 1

    return [item for item in found if counts[item["code"]] < LOCALE_HITS_CUTOFF(
        item["code"], tally)]


def LOCALE_HITS_CUTOFF(code, tally):
    """Locale evidence is only returned while a code keeps adding new rules."""

    return tally.get(code, 0) + 1


def score_evidence(evidence):
    """Fold evidence items into per-code scores.

    Returns (scores, best_source) where scores maps an ISO code to its
    accumulated weight and best_source names the source of the single
    strongest item for the winning code.
    """

    scores = {}
    best = {}
    locale_count = {}

    for item in evidence:
        code = item["code"]
        source = item["source"]

        if source == "ocr_symbol":
            weight = SYMBOL_WEIGHT
        elif source == "ocr_code":
            weight = CODE_WEIGHT
        elif source == "confusable":
            weight = CONFUSABLE_WEIGHT
        elif source == "glyph_crop":
            weight = SYMBOL_WEIGHT
        else:
            locale_count[code] = locale_count.get(code, 0) + 1
            weight = min(LOCALE_CAP, 0.25 * locale_count[code])

        scores[code] = _repeated(scores.get(code, 0.0), weight)

        if scores[code] > best.get(code, (0.0, ""))[0]:
            best[code] = (scores[code], source)

    return scores, {code: source for code, (_, source) in best.items()}


def decide(scores, sources, min_score=MIN_SCORE):
    """Pick a currency from scored evidence.

    Returns (code, confidence, source, candidates). Confidence is the
    winner's share of the top two scores, capped harder when nothing was
    actually read off the page.
    """

    if not scores:
        return None, 0.0, "none", {}

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    top_code, top_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else 0.0

    candidates = {code: round(score, 3) for code, score in ranked}
    source = sources.get(top_code, "none")

    if top_score < min_score:
        return None, 0.0, "none", candidates

    share = top_score / (top_score + runner_score) if (top_score + runner_score) else 0.0
    cap = DIRECT_CONFIDENCE_CAP if source in DIRECT_SOURCES \
        else INFERRED_CONFIDENCE_CAP

    return top_code, round(share * cap, 2), source, candidates


def text_evidence(text):
    """Every evidence item the text layers can offer, cheapest first."""

    return read_symbols(text) + read_confusables(text) + read_locale(text)


def needs_pixels(text, min_score=MIN_SCORE):
    """True unless a symbol or ISO code that survived the OCR already wins.

    A currency the OCR text still spells out has nothing to recover, so
    the crop pass is skipped and the image is not re-read at all.

    The check is on the winning code rather than on decide's source,
    which names whichever item last moved the running total and so
    reports a document full of rupee signs as locale-decided.
    """

    evidence = text_evidence(text)
    scores, sources = score_evidence(evidence)
    code, _, _, _ = decide(scores, sources, min_score=min_score)

    if not code:
        return True

    return not any(
        item["source"] in DIRECT_SOURCES and item["code"] == code
        for item in evidence
    )


def leading_currency(text):
    """The currency symbol at the front of a recogniser's reading, if any.

    The recogniser glues the glyph to whatever sits beside it, so a read
    of "₹1" is still a rupee sign found. Requiring the whole string to
    be a currency token throws away the one character the crop aimed at.
    The shortest matching prefix wins, so the token returned is the
    symbol rather than the digits it was glued to.
    """

    text = _normalise(text).strip()

    for size in (1, 2, 3):
        head = text[:size]

        if head and symbol_to_code(head):
            return head

    return None


def _read_glyph(crop, scales=GLYPH_SCALES, min_score=GLYPH_MIN_SCORE):
    """Re-run the recogniser on one crop at several scales.

    Returns the best (symbol_text, score, scale) that names a currency,
    or None.
    """

    from PIL import Image

    engine = get_engine()
    best = None

    for scale in scales:
        upscaled = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        result = engine(upscaled, use_det=False, use_cls=False, use_rec=True)

        texts = getattr(result, "txts", None) or ()
        scores = getattr(result, "scores", None) or ()
        text = str(texts[0]).strip() if texts else ""
        score = _as_float(scores[0]) if scores else None

        if not text or score is None or score < min_score:
            continue

        token = leading_currency(text)

        if token is None:
            continue

        if best is None or score > best[1]:
            best = (token, score, scale)

    return best


def _has_ink(crop):
    """True when the crop holds enough dark pixels to be worth reading."""

    histogram = crop.convert("L").histogram()
    ink = sum(histogram[:INK_LEVEL + 1])

    return ink >= INK_MIN_PIXELS


def _glyph_width(box, text):
    """Width of one character in a row, from the text the OCR read.

    line_height is useless for this: a skewed or rotated box is far
    taller than its glyphs are wide, so a window sized off it swallows
    the neighbouring column instead of the symbol.
    """

    x0, y0, x1, y1 = box
    text_width = max(1.0, x1 - x0)
    line_height = max(1.0, y1 - y0)
    count = max(1, len(_normalise(text).strip()))

    return max(2.0, min(text_width / count, line_height * 0.8))


def _symbol_windows(box, text, image_size):
    """Candidate crops around one amount.

    "pre" catches a symbol the text detector clipped off entirely, just
    left of the digits. "lead" catches one that is still in the box but
    that OCR failed to transcribe. Both are one glyph wide.
    """

    x0, y0, x1, y1 = box
    width, height = image_size
    glyph = _glyph_width(box, text)

    windows = [
        (
            "pre",
            (x0 - glyph * 1.5, y0 - glyph * 0.2,
             x0 + glyph * 0.1, y1 + glyph * 0.2),
        ),
        (
            "lead",
            (x0 - glyph * 0.4, y0 - glyph * 0.2,
             x0 + glyph * 1.1, y1 + glyph * 0.2),
        ),
    ]

    out = []

    for name, (ax, ay, bx, by) in windows:
        cropped = (
            max(0, int(ax)),
            max(0, int(ay)),
            min(width, int(bx)),
            min(height, int(by)),
        )

        if cropped[2] - cropped[0] < 4 or cropped[3] - cropped[1] < 4:
            continue

        out.append((name, cropped))

    return out


def read_glyphs(image_path, rows, scales=GLYPH_SCALES, max_crops=MAX_CROPS):
    """Crop the ink around every amount and re-read it as a symbol.

    Returns [{"code", "symbol", "text", "score", "scale", "window",
    "line", "box"}, ...] sorted by score.
    """

    from PIL import Image

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

    image_size = image.size
    hits = []
    seen = set()

    for row in rows:
        if len(hits) >= max_crops:
            break

        if not row.get("box") or not MONEY_RE.search(row["text"]):
            continue

        for name, window in _symbol_windows(row["box"], row["text"], image_size):
            if window in seen or len(hits) >= max_crops:
                continue

            seen.add(window)
            crop = image.crop(window)

            if not _has_ink(crop):
                continue

            read = _read_glyph(crop, scales=scales)

            if read is None:
                continue

            text, score, scale = read
            code = symbol_to_code(text)

            hits.append(
                {
                    "code": code,
                    "symbol": CODE_TO_SYMBOL.get(code, code),
                    "text": text,
                    "score": score,
                    "scale": scale,
                    "window": name,
                    "line": row["text"],
                    "box": window,
                }
            )

    return sorted(hits, key=lambda hit: -hit["score"])


def analyse_text(text):
    """Score currency evidence that is already in OCR text.

    Returns the same report shape as recover_currency, minus the pixel
    layer, so the two can be compared on the same document.
    """

    symbols = read_symbols(text)
    evidence = symbols + read_confusables(text) + read_locale(text)
    scores, sources = score_evidence(evidence)
    code, confidence, source, candidates = decide(scores, sources)

    return {
        "image": None,
        "currency": code,
        "symbol": CODE_TO_SYMBOL.get(code) if code else None,
        "confidence": confidence,
        "source": source,
        "recovered": not symbols and code is not None,
        "symbol_in_ocr_text": bool(symbols),
        "candidates": candidates,
        "evidence": evidence,
        "glyph_hits": [],
        "notes": _notes(code, source, evidence),
    }


def recover_currency(image_path, ocr_text=None, use_pixels=True,
                     scales=GLYPH_SCALES, min_score=MIN_SCORE,
                     max_crops=MAX_CROPS, save_crops=None, rows=None):
    """Recover the currency for one receipt image.

    ocr_text reuses text the caller already has instead of re-reading it
    from the OCR result. rows does the same for the boxes the pixel layer
    needs, so a caller that already ran OCR does not pay for it twice.
    set use_pixels to False for a text-only run.

    The pixel layer only runs when the OCR text did not already name a
    currency, since a symbol that survived has nothing to recover.

    Returns a dict with:
        image              : str, the path that was read
        currency           : ISO code or None
        symbol             : display symbol or None
        confidence         : float, 0-1
        source             : winning evidence source
        recovered          : True when no symbol survived in the OCR text
        symbol_in_ocr_text : True when a symbol was already readable
        candidates         : {code: score} for every code considered
        evidence           : every item that contributed
        glyph_hits         : symbols read off the image itself
        notes              : human-readable caveats
    """

    notes = []
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")

    if rows is None:
        rows = read_rows(path) if use_pixels or ocr_text is None else []

    if ocr_text is None:
        ocr_text = "\n".join(row["text"] for row in rows)

    symbols = read_symbols(ocr_text)
    evidence = symbols + read_confusables(ocr_text) + read_locale(ocr_text)

    glyph_hits = []

    if use_pixels and needs_pixels(ocr_text, min_score=min_score):
        try:
            glyph_hits = read_glyphs(path, rows, scales=scales, max_crops=max_crops)
        except Exception as error:
            notes.append(f"glyph layer unavailable: {type(error).__name__}: {error}")

        evidence += [
            {
                "code": hit["code"],
                "symbol": hit["symbol"],
                "source": "glyph_crop",
                "detail": f"{hit['text']} @ {hit['window']} x{hit['scale']}",
                "line": hit["line"],
            }
            for hit in glyph_hits
        ]
    elif use_pixels:
        notes.append("a symbol survived the OCR text; pixel layer skipped")

    if save_crops and glyph_hits:
        notes.append(f"glyph crops listed under glyph_hits: {len(glyph_hits)}")

    scores, sources = score_evidence(evidence)
    code, confidence, source, candidates = decide(scores, sources, min_score=min_score)

    return {
        "image": str(path),
        "currency": code,
        "symbol": CODE_TO_SYMBOL.get(code) if code else None,
        "confidence": confidence,
        "source": source,
        "recovered": not symbols and code is not None,
        "symbol_in_ocr_text": bool(symbols),
        "candidates": candidates,
        "evidence": evidence,
        "glyph_hits": glyph_hits,
        "notes": _notes(code, source, evidence) + notes,
    }


def restore_symbols(text, code, symbol=None):
    """Write a currency symbol back in front of the amounts that lost it.

    Money-shaped numbers on lines that are not invoice numbers, dates,
    phone numbers or licence plates are patched, and a number that
    already carries a symbol is left alone.

    A page with no money-shaped number anywhere is also treated: a
    standalone figure that could be the total is marked, because a
    receipt written as a bare "190" otherwise reads downstream as a
    document with no money in it at all.

    Returns (patched_text, patched_count).
    """

    if not code:
        return text or "", 0

    if symbol is None:
        symbol = CODE_TO_SYMBOL.get(code, code)

    if not text:
        return "", 0

    lines = _normalise(text).split("\n")
    bare_page = not MONEY_RE.search(text)
    patched = 0
    out = []

    for line in lines:
        if IDENTITY_RE.search(line) or DATE_RE.search(line):
            out.append(line)
            continue

        line, count = _patch_line(line, symbol)

        if bare_page:
            line, extra = _mark_standalone_amount(line, symbol, count)
            count += extra

        patched += count
        out.append(line)

    return "\n".join(out), patched


def _mark_standalone_amount(line, symbol, already):
    """Mark a line that is nothing but a short figure.

    Deliberately strict: a 12-digit UPI transaction id or a bank account
    is just as standalone as a total, and marking one of those would be
    worse than leaving a real total unmarked.

    Returns (line, number of insertions).
    """

    if already or not STANDALONE_AMOUNT_RE.match(line):
        return line, 0

    return f"{symbol}{line.strip()}", 1


def _patch_line(line, symbol):
    """Insert symbol before every bare money number on one line.

    Returns (line, number of insertions).
    """

    matches = list(MONEY_RE.finditer(line))

    if not matches:
        return line, 0

    out = []
    cursor = 0
    patched = 0

    for match in matches:
        start, end = match.span()
        before = line[:start]

        if _already_marked(before):
            continue

        out.append(line[cursor:start])
        out.append(symbol if not before or before[-1].isspace() or before[-1] in "([" else f" {symbol}")
        cursor = start
        patched += 1

    out.append(line[cursor:])

    return "".join(out), patched


def _already_marked(before):
    """True when a currency symbol or letter code already sits before the number."""

    tail = before[-4:].strip().lower().rstrip(":. ")

    if not tail:
        return False

    if tail[-1:] in SYMBOL_TO_CODE or tail in PREFIX_TO_CODE:
        return True

    return False


def _notes(code, source, evidence):
    notes = []

    if code is None:
        notes.append("no currency cleared the evidence bar")
        return notes

    if source == "locale":
        notes.append("decided by regional markers only; treat as a hint")

    if source == "confusable":
        notes.append("decided by OCR lookalikes; the glyph was not confirmed")

    if not any(item["source"] in ("ocr_symbol", "ocr_code", "glyph_crop") for item in evidence):
        notes.append("no symbol or code was read directly from the page")

    return notes


def use_utf8_output():
    """Force UTF-8 on stdout/stderr so currency symbols survive the console."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_parser():
    parser = argparse.ArgumentParser(
        description="Recover the currency symbol that OCR dropped")
    parser.add_argument("image", nargs="+", help="Receipt image, or a .txt of OCR text")
    parser.add_argument("--text-only", action="store_true",
                        help="Skip the pixel layer and score the OCR text alone")
    parser.add_argument("--text", help="OCR text to score instead of re-running OCR")
    parser.add_argument("--patch", action="store_true",
                        help="Print the OCR text with the recovered symbol restored")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Print the full report as JSON")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE,
                        help=f"Evidence weight needed to name a currency (default {MIN_SCORE})")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Exit non-zero when the recovered confidence is below this")
    parser.add_argument("--scales", default=",".join(str(s) for s in GLYPH_SCALES),
                        help="Upscale factors for the glyph layer")
    parser.add_argument("--max-crops", type=int, default=MAX_CROPS,
                        help=f"Cap on glyph crops per image (default {MAX_CROPS})")
    parser.add_argument("--save-crops", help="Directory to write the glyph crops into")
    parser.add_argument("--evidence", action="store_true",
                        help="Print every evidence item, not just the winner")
    return parser


def _scales(raw):
    try:
        scales = tuple(int(part) for part in str(raw).split(",") if int(part) > 0)
    except (TypeError, ValueError):
        return GLYPH_SCALES

    return scales or GLYPH_SCALES


def print_report(report, show_evidence=False):
    print(LINE_RULE)
    print(f"IMAGE      {report['image'] or '(ocr text)'}")
    print(f"CURRENCY   {report['currency'] or '-'}  symbol={report['symbol'] or '-'}")
    print(f"CONFIDENCE {report['confidence']:.2f}   source={report['source']}")
    print(f"RECOVERED  {report['recovered']}   symbol_in_ocr_text={report['symbol_in_ocr_text']}")

    if report["candidates"]:
        ranked = "  ".join(f"{code}={score:.2f}" for code, score in report["candidates"].items())
        print(f"CANDIDATES {ranked}")

    if show_evidence:
        for item in report["evidence"]:
            line = f"  {item['source']:11} {item['code']:4} {item['symbol']:6} {item['detail']}"
            print(line)
            if item["line"]:
                print(f"  {'':11} on {item['line'][:56]}")

    if report["glyph_hits"]:
        print(LINE_RULE)
        print("GLYPH CROPS")

        for hit in report["glyph_hits"]:
            print(f"  {hit['symbol']:6} {hit['code']:4} {hit['score']:.2f} "
                  f"x{hit['scale']:<2} {hit['window']:5} {hit['box']}  read {hit['text']!r}")
            print(f"  {'':6} line {hit['line'][:56]}")

    for note in report["notes"]:
        print(f"NOTE       {note}")

    print(LINE_RULE)


def save_crops(image_path, report, directory):
    """Write every glyph crop the image layer read, for eyeballing."""

    from PIL import Image

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    saved = []

    with Image.open(image_path) as handle:
        image = handle.convert("RGB")

        for index, hit in enumerate(report["glyph_hits"], start=1):
            name = f"{stem}.{index:02d}.{hit['code']}.{hit['window']}.png"
            path = target / name
            image.crop(tuple(hit["box"])).resize((160, 160), Image.LANCZOS).save(path)
            saved.append(str(path))

    return saved


def run(argv=None):
    """Entry point. Returns a process exit code."""

    use_utf8_output()

    args = build_parser().parse_args(argv)
    scales = _scales(args.scales)
    exit_code = 0

    for source in args.image:
        path = Path(source)
        is_text = path.suffix.lower() == ".txt"

        if not path.exists():
            print(f"error: {source} not found", file=sys.stderr)
            exit_code = 1
            continue

        try:
            if is_text or args.text_only:
                text = args.text if args.text else path.read_text(encoding="utf-8")
                report = analyse_text(text)
                report["min_score"] = args.min_score
                report = _with_min_score(report, args.min_score)
            else:
                report = recover_currency(
                    path,
                    ocr_text=args.text,
                    use_pixels=not args.text_only,
                    scales=scales,
                    min_score=args.min_score,
                    max_crops=args.max_crops,
                )
        except FileNotFoundError as error:
            print(f"error: {error}", file=sys.stderr)
            exit_code = 1
            continue
        except Exception as error:
            print(f"error: {source}: {type(error).__name__}: {error}", file=sys.stderr)
            exit_code = 1
            continue

        if args.save_crops and report["glyph_hits"]:
            for saved in save_crops(source, report, args.save_crops):
                print(f"saved {saved}")

        if args.as_json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print_report(report, show_evidence=args.evidence)

        if args.patch and report["currency"]:
            from app.text_normalizer import normalize_ocr_text

            text = args.text or (
                "\n".join(row["text"] for row in read_rows(path))
                if not is_text
                else path.read_text(encoding="utf-8")
            )
            patched, count = restore_symbols(normalize_ocr_text(text), report["currency"])
            print(LINE_RULE)
            print(f"PATCHED {count} line(s) with {report['symbol']}")
            print(LINE_RULE)
            print(patched)
            print(LINE_RULE)

        if report["confidence"] < args.min_confidence:
            exit_code = max(exit_code, 3)

    return exit_code


def _with_min_score(report, min_score):
    """Re-run the decision so a .txt input honours --min-score too."""

    scores, sources = score_evidence(report["evidence"])
    code, confidence, source, candidates = decide(scores, sources, min_score=min_score)
    report["currency"] = code
    report["symbol"] = CODE_TO_SYMBOL.get(code) if code else None
    report["confidence"] = confidence
    report["source"] = source
    report["candidates"] = candidates
    report["notes"] = _notes(code, source, report["evidence"])
    return report


if __name__ == "__main__":
    raise SystemExit(run())
