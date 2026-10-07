from app.business_card.pipeline import process_business_card


class FakeClient:
    def extract(self, prompt):
        return '{"name":"Ada Lovelace","designation":"Engineer","company_name":"Analytical Engines"}'


def test_full_pipeline_uses_injected_ocr_and_llm():
    text = "Ada Lovelace\nEngineer\nAnalytical Engines\nTel: +44 20 1234 5678\nada@example.com"
    result = process_business_card("unused.jpg", llm_client=FakeClient(), ocr_function=lambda _: text)
    assert result.name == "Ada Lovelace"
    assert result.company_name == "Analytical Engines"
    assert result.phone == "+44 20 1234 5678"
    assert result.email == "ada@example.com"
