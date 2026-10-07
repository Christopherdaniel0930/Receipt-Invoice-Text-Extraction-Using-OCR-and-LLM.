from app.business_card.modules.validator import validate_business_card


def test_validates_and_normalizes_optional_fields():
    result = validate_business_card({"email": " ada@example.com ", "website": "www.example.com", "phone": "+1 555 123 4567"})
    assert result.email == "ada@example.com"
    assert result.website == "www.example.com"
    assert result.phone == "+1 555 123 4567"
    assert result.name is None


def test_invalid_contact_formats_are_cleared_without_rejecting_card():
    result = validate_business_card({"email": "bad", "website": "not a site", "phone": "123"})
    assert result.email is result.website is result.phone is None
