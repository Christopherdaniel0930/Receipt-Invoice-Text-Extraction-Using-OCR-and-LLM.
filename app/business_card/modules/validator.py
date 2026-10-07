"""Pydantic validation and light normalization for extracted cards."""

import re

from pydantic import ValidationError

from app.business_card.models.business_card_model import BusinessCardData

EMAIL_RE = re.compile(r"^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$", re.I)
WEBSITE_RE = re.compile(r"^(?:https?://)?(?:www\.)?[A-Z0-9.-]+\.[A-Z]{2,}(?:[/:?#][^\s]*)?$", re.I)


def validate_business_card(data) -> BusinessCardData:
    if isinstance(data, BusinessCardData):
        values = data.model_dump()
    else:
        values = dict(data or {})
    for field in values:
        if isinstance(values[field], str):
            values[field] = re.sub(r"\s+", " ", values[field]).strip() or None
    email = values.get("email")
    if email and not EMAIL_RE.fullmatch(email):
        values["email"] = None
    website = values.get("website")
    if website and not WEBSITE_RE.fullmatch(website):
        values["website"] = None
    for field in ("phone", "fax"):
        number = values.get(field)
        if number:
            digits = re.sub(r"\D", "", number)
            if not 7 <= len(digits) <= 15:
                values[field] = None
            else:
                values[field] = re.sub(r"\s+", " ", number).strip()
    try:
        return BusinessCardData.model_validate(values)
    except ValidationError:
        raise
