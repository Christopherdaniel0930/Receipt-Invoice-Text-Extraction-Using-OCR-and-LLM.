from app.business_card.modules.llm_extractor import (
    build_business_card_prompt,
    extract_business_card_semantics,
)


class FakeClient:
    def extract(self, prompt):
        assert "Never invent values" in prompt
        return '{"name":"Ada Lovelace","designation":"Engineer","company_name":null}'


def test_semantic_extraction_parses_only_expected_fields():
    result = extract_business_card_semantics("Ada Lovelace, Engineer", FakeClient())
    assert result == {"name": "Ada Lovelace", "designation": "Engineer", "company_name": None}


def test_prompt_requests_strict_schema():
    prompt = build_business_card_prompt("sample")
    assert '"company_name": null' in prompt
    assert "Do not infer a company name" in prompt
