"""High precision OCR evidence used to decide when the ML category is trusted."""

from dataclasses import dataclass
import re

from app.validator import CATEGORIES


STRONG_RULES = {
    "Food": (
        "food", "meal", "breakfast", "lunch", "dinner", "sambar", "parotta",
        "chapathi", "dosa", "idli", "pizza", "burger", "rice", "coffee", "tea",
        "chicken", "fish", "vegetarian",
    ),
    "Travel": (
        "flight ticket", "train ticket", "bus ticket", "boarding pass", "fare",
        "railway", "airline", "taxi fare", "travel ticket",
    ),
    "Fuel": ("petrol", "diesel", "fuel", "gasoline", "litre", "liter"),
    "Electronics": (
        "electronics", "laptop", "mobile phone", "smartphone", "headphones",
        "television", "monitor", "keyboard", "computer",
    ),
    "Office Supplies": (
        "office supplies", "stationery", "printer paper", "toner cartridge",
        "notebook", "ballpoint pen", "file folder",
    ),
    "Accommodation": (
        "hotel room", "room charge", "room nights", "check in", "check-out",
        "lodging", "accommodation",
    ),
    "Healthcare": (
        "pharmacy", "prescription", "medicine", "medical", "clinic", "hospital",
        "healthcare",
    ),
    "Utilities": (
        "electricity bill", "water bill", "utility bill", "internet bill",
        "broadband bill", "mobile bill", "gas bill",
    ),
    
}

_PATTERNS = {
    category: tuple(
        (term, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE))
        for term in terms
    )
    for category, terms in STRONG_RULES.items()
}


@dataclass(frozen=True)
class CategoryEvidence:
    category: str | None
    confidence: float
    strong_rule: bool
    matched_terms: tuple[str, ...] = ()

def evaluate_category_evidence(ocr_text: str) -> CategoryEvidence:
    """Return a category only when OCR has unambiguous, explicit evidence.

    Merchant names alone do not trigger a category. This intentionally avoids
    assigning Food just because a vendor name contains "restaurant".
    """
    matches = {
        category: tuple(term for term, pattern in patterns if pattern.search(ocr_text or ""))
        for category, patterns in _PATTERNS.items()
    }
    matched = {category: terms for category, terms in matches.items() if terms}
    if len(matched) != 1:
        return CategoryEvidence(category=None, confidence=0.0, strong_rule=False)

    category, terms = next(iter(matched.items()))
    if category not in CATEGORIES:
        return CategoryEvidence(category=None, confidence=0.0, strong_rule=False)
    confidence = min(0.99, 0.85 + 0.05 * (len(terms) - 1))
    return CategoryEvidence(
        category=category,
        confidence=confidence,
        strong_rule=True,
        matched_terms=terms,
    )
