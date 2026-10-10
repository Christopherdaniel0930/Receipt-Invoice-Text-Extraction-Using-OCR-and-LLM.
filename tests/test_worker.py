import asyncio
from pathlib import Path

import httpx
import pytest
from arq.worker import Retry
from openai import APIConnectionError

from app.business_card.models.business_card_model import BusinessCardData
from app.receipt.schemas.schema import ReceiptData
from app.workers import jobs
from app.workers.jobs import process_document, process_receipt


class FakeRedis:
    def __init__(self):
        self.hashes = {}

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    async def expire(self, key, seconds):
        return True


def run_job(monkeypatch, tmp_path, result_or_error, *, job_try=1, max_tries=3):
    import app.main

    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"image")
    redis = FakeRedis()

    def fake_process(path):
        assert path == str(image_path)
        if isinstance(result_or_error, Exception):
            raise result_or_error
        return result_or_error

    monkeypatch.setattr(app.main, "process", fake_process)
    ctx = {"redis": redis, "job_try": job_try, "max_tries": max_tries}
    return image_path, redis, ctx


def test_worker_success(monkeypatch, tmp_path):
    image_path, redis, ctx = run_job(
        monkeypatch, tmp_path, ReceiptData(
            vendor_name="Shop", total_amount=12.5,
            line_items=[
                {"description": "CHILLY PAROTTA", "quantity": 1, "unit_price": 110, "amount": 110},
                {"description": "TEA", "quantity": 1, "unit_price": 60, "amount": 60},
                {"description": "WATER", "quantity": 1, "unit_price": 90, "amount": 90},
            ],
        )
    )
    asyncio.run(process_receipt(ctx, "receipt-1", str(image_path)))
    state = redis.hashes["receipt-job:receipt-1"]
    assert state["status"] == "COMPLETED"
    assert '"vendor_name": "Shop"' in state["result"]
    import json
    assert [item["amount"] for item in json.loads(state["result"])["line_items"]] == [110.0, 60.0, 90.0]
    assert not image_path.exists()


def test_worker_marks_terminal_error_failed(monkeypatch, tmp_path):
    image_path, redis, ctx = run_job(
        monkeypatch, tmp_path, RuntimeError("OCR returned no text")
    )
    asyncio.run(process_receipt(ctx, "receipt-2", str(image_path)))
    state = redis.hashes["receipt-job:receipt-2"]
    assert state["status"] == "FAILED"
    assert state["error_code"] == "PROCESSING_FAILED"
    assert not image_path.exists()


def test_worker_rejects_non_receipt(monkeypatch, tmp_path):
    image_path, redis, ctx = run_job(
        monkeypatch,
        tmp_path,
        RuntimeError("the image is not a receipt or invoice (identity_document)"),
    )
    asyncio.run(process_receipt(ctx, "receipt-3", str(image_path)))
    state = redis.hashes["receipt-job:receipt-3"]
    assert state["status"] == "REJECTED"
    assert state["error_code"] == "NOT_A_RECEIPT"
    assert not image_path.exists()


def test_worker_retries_temporary_ollama_failure(monkeypatch, tmp_path):
    error = APIConnectionError(
        message="connection unavailable",
        request=httpx.Request("POST", "http://ollama.test/v1/chat/completions"),
    )
    image_path, redis, ctx = run_job(monkeypatch, tmp_path, error)
    with pytest.raises(Retry):
        asyncio.run(process_receipt(ctx, "receipt-4", str(image_path)))
    assert redis.hashes["receipt-job:receipt-4"]["status"] == "QUEUED"
    assert image_path.exists()


def test_worker_marks_exhausted_temporary_failure_failed(monkeypatch, tmp_path):
    error = APIConnectionError(
        message="connection unavailable",
        request=httpx.Request("POST", "http://ollama.test/v1/chat/completions"),
    )
    image_path, redis, ctx = run_job(
        monkeypatch, tmp_path, error, job_try=3, max_tries=3
    )
    asyncio.run(process_receipt(ctx, "receipt-5", str(image_path)))
    assert redis.hashes["receipt-job:receipt-5"]["status"] == "FAILED"
    assert not image_path.exists()


def run_document_job(monkeypatch, tmp_path, result_or_error, *, job_try=1, max_tries=3):
    image_path = tmp_path / "document.png"
    image_path.write_bytes(b"image")
    redis = FakeRedis()

    async def fake_pipeline(path, document_id):
        assert path == image_path
        if isinstance(result_or_error, Exception):
            raise result_or_error
        return result_or_error

    monkeypatch.setattr(jobs, "_run_document_pipeline", fake_pipeline)
    ctx = {"redis": redis, "job_try": job_try, "max_tries": max_tries}
    return image_path, redis, ctx


@pytest.mark.parametrize(
    ("document_type", "data"),
    [
        ("business_card", BusinessCardData(name="Ada", email="ada@example.com")),
        ("receipt_invoice", ReceiptData(vendor_name="Shop", total_amount=12.5)),
    ],
)
def test_document_worker_serializes_typed_results(
    monkeypatch, tmp_path, document_type, data
):
    image_path, redis, ctx = run_document_job(
        monkeypatch,
        tmp_path,
        {"document_type": document_type, "data": data},
    )
    asyncio.run(process_document(ctx, "doc-success", str(image_path)))
    state = redis.hashes["document-job:doc-success"]
    assert state["status"] == "COMPLETED"
    assert state["document_type"] == document_type
    import json
    result = json.loads(state["result"])
    assert result == data.model_dump(mode="json")
    assert not image_path.exists()


def test_document_worker_rejects_unknown_and_cleans_up(monkeypatch, tmp_path):
    image_path, redis, ctx = run_document_job(
        monkeypatch,
        tmp_path,
        {"document_type": "unknown", "data": None},
    )
    asyncio.run(process_document(ctx, "doc-unknown", str(image_path)))
    state = redis.hashes["document-job:doc-unknown"]
    assert state["status"] == "REJECTED"
    assert state["document_type"] == "unknown"
    assert state["error"] == "Unsupported document type"
    assert not image_path.exists()


def test_document_worker_retries_temporary_upstream_failure(monkeypatch, tmp_path):
    error = APIConnectionError(
        message="connection unavailable",
        request=httpx.Request("POST", "http://ollama.test/v1/chat/completions"),
    )
    image_path, redis, ctx = run_document_job(monkeypatch, tmp_path, error)
    with pytest.raises(Retry):
        asyncio.run(process_document(ctx, "doc-retry", str(image_path)))
    assert redis.hashes["document-job:doc-retry"]["status"] == "QUEUED"
    assert image_path.exists()


def test_document_worker_marks_nonretryable_error_failed_and_cleans_up(
    monkeypatch, tmp_path
):
    image_path, redis, ctx = run_document_job(
        monkeypatch, tmp_path, RuntimeError("OCR returned no text")
    )
    asyncio.run(process_document(ctx, "doc-failed", str(image_path)))
    state = redis.hashes["document-job:doc-failed"]
    assert state["status"] == "FAILED"
    assert state["error_code"] == "PROCESSING_FAILED"
    assert not image_path.exists()
