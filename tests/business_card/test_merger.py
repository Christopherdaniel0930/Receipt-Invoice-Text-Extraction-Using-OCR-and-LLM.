from app.business_card.modules.merger import merge_business_card


def test_merges_distinct_field_sources_without_overwriting_deterministic_values():
    result = merge_business_card(
        {"name": "Ada", "phone": "guessed"},
        {"phone": "+1 555 123 4567", "email": "ada@example.com"},
    )
    assert result["name"] == "Ada"
    assert result["phone"] == "+1 555 123 4567"
    assert result["email"] == "ada@example.com"
    assert result["fax"] is None
