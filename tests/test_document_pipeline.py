import pytest

from app.business_card.models.business_card_model import BusinessCardData
from app.common import document_pipeline


class FakeClient:
    def __init__(self):
        self.classify_calls = 0
        self.extract_calls = 0

    def classify_document(self, text):
        self.classify_calls += 1
        return {"document_type": "business_card", "confidence": 0.98}

    def extract(self, prompt):
        self.extract_calls += 1
        return '{"name":"Ada Lovelace","designation":"Engineer","company_name":"Analytical Engines"}'


def _setup_ocr(monkeypatch, rows):
    calls = []

    def read(path):
        calls.append(path)
        return rows

    monkeypatch.setattr(document_pipeline, "read_ocr_rows", read)
    return calls


def test_routes_business_card_with_one_ocr_and_one_classification(monkeypatch):
    import app.main as main

    rows = [{"text": "Ada Lovelace"}, {"text": "Engineer"}]
    ocr_calls = _setup_ocr(monkeypatch, rows)
    client = FakeClient()
    classifier_calls = []

    def classify(text, passed_client):
        classifier_calls.append((text, passed_client))
        result = passed_client.classify_document(text)
        return {**result, "is_receipt_invoice": False}

    monkeypatch.setattr(document_pipeline, "classify_document", classify)
    monkeypatch.setattr(main, "prepare_document_input", lambda path, text, ocr_rows: (text, None))
    from app.business_card import pipeline as card_pipeline
    original = card_pipeline.process_business_card
    received = {}

    def process_card(*args, **kwargs):
        received.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(card_pipeline, "process_business_card", process_card)

    result = document_pipeline.process_document("card.png", llm_client=client)

    assert len(ocr_calls) == 1
    assert len(classifier_calls) == 1
    assert client.classify_calls == 1
    assert client.extract_calls == 1
    assert received["ocr_text"] == "Ada Lovelace\nEngineer"
    assert received["ocr_rows"] is rows
    assert received["llm_client"] is client
    assert result["document_type"] == "business_card"
    assert isinstance(result["data"], BusinessCardData)


def test_routes_receipt_with_same_client_and_precomputed_ocr(monkeypatch):
    import app.main as main

    rows = [{"text": "Cafe"}, {"text": "Total 12.00"}]
    ocr_calls = _setup_ocr(monkeypatch, rows)
    client = FakeClient()
    classifier_calls = []

    def classify(text, passed_client):
        classifier_calls.append((text, passed_client))
        result = passed_client.classify_document(text)
        return {**result, "document_type": "receipt_or_invoice", "is_receipt_invoice": True}

    monkeypatch.setattr(document_pipeline, "classify_document", classify)
    monkeypatch.setattr(main, "prepare_document_input", lambda path, text, ocr_rows: (text, {"currency": None}))
    received = {}

    def process_receipt(path, **kwargs):
        received.update(kwargs)
        return "receipt-result"

    monkeypatch.setattr(main, "process", process_receipt)
    result = document_pipeline.process_document("receipt.png", llm_client=client)

    assert len(ocr_calls) == 1
    assert len(classifier_calls) == client.classify_calls == 1
    assert received["ocr_text"] == "Cafe\nTotal 12.00"
    assert received["ocr_rows"] is rows
    assert received["llm_client"] is client
    assert received["document"]["is_receipt_invoice"] is True
    assert received["preprocessed"] is True
    assert result == {"document_type": "receipt_or_invoice", "data": "receipt-result"}


def test_unknown_document_is_not_dispatched(monkeypatch):
    import app.main as main

    _setup_ocr(monkeypatch, [{"text": "unrecognized content"}])
    client = FakeClient()
    monkeypatch.setattr(
        document_pipeline,
        "classify_document",
        lambda text, passed: {"document_type": "unknown", "is_receipt_invoice": False},
    )
    monkeypatch.setattr(main, "prepare_document_input", lambda path, text, rows: (text, None))
    result = document_pipeline.process_document("unknown.png", llm_client=client)
    assert result == {"document_type": "unknown", "data": None}
    assert client.classify_calls == 0


def test_classifier_receives_the_supplied_client(monkeypatch):
    import app.main as main

    _setup_ocr(monkeypatch, [{"text": "not enough to classify"}])
    monkeypatch.setattr(main, "prepare_document_input", lambda path, text, rows: (text, None))
    client = FakeClient()
    seen = []

    def classify(text, passed_client):
        seen.append(passed_client)
        return {"document_type": "unknown", "is_receipt_invoice": False}

    monkeypatch.setattr(document_pipeline, "classify_document", classify)
    document_pipeline.process_document("unknown.png", llm_client=client)
    assert seen == [client]
