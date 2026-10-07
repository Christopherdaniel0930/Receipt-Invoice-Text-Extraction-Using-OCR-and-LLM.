from app.business_card.modules.deterministic_extractor import extract_deterministic


def test_extracts_email_website_phone_fax_and_address():
    result = extract_deterministic(
        "Tel: +91 9876543210\nFax: +91 44 12345678\n"
        "john@example.com\nhttps://example.com\n12 MG Road, Bengaluru 560001"
    )
    assert result == {
        "email": "john@example.com",
        "phone": "+91 9876543210",
        "fax": "+91 44 12345678",
        "website": "https://example.com",
        "address": "12 MG Road, Bengaluru 560001",
    }


def test_unlabelled_phone_is_not_reused_as_fax():
    result = extract_deterministic("9876543210\nFax: +91-44-12345678")
    assert result["phone"] == "9876543210"
    assert result["fax"] == "+91-44-12345678"
