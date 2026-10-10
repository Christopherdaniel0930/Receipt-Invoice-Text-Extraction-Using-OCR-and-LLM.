from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.enqueued = []

    async def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    async def expire(self, key, seconds):
        return True

    async def ping(self):
        return True

    async def enqueue_job(self, function, *args, **kwargs):
        self.enqueued.append((function, args, kwargs))
        return object()

    async def hgetall(self, key):
        return self.hashes.get(key, {})

    async def delete(self, key):
        self.hashes.pop(key, None)


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(app.state, "redis", redis, raising=False)
    yield redis
    for _, args, _ in redis.enqueued:
        Path(args[1]).unlink(missing_ok=True)


def make_image(format_name="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format=format_name)
    return buffer.getvalue()


def test_health_and_ready(fake_redis):
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_receipt_upload_enqueues_job_and_returns_immediately(fake_redis):
    response = client.post(
        "/api/v1/receipts",
        files={"file": ("receipt.png", make_image(), "image/png")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert fake_redis.enqueued[0][0] == "process_receipt"
    assert fake_redis.enqueued[0][1][0] == body["job_id"]
    assert Path(fake_redis.enqueued[0][1][1]).is_file()


def test_document_upload_enqueues_generalized_job(fake_redis):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("document.png", make_image(), "image/png")},
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert fake_redis.enqueued[0][0] == "process_document"
    assert fake_redis.enqueued[0][1][0] == body["job_id"]
    assert Path(fake_redis.enqueued[0][1][1]).is_file()
    assert fake_redis.hashes[f"document-job:{body['job_id']}"]["job_kind"] == "document"


@pytest.mark.parametrize("filename", ["business-card.png", "receipt.png"])
def test_document_upload_accepts_both_supported_document_classes(fake_redis, filename):
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, make_image(), "image/png")},
    )
    assert response.status_code == 202
    assert fake_redis.enqueued[0][0] == "process_document"


def test_document_upload_requires_file(fake_redis):
    response = client.post("/api/v1/documents")
    assert response.status_code == 422


def test_document_upload_rejects_invalid_file_type(fake_redis):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("document.gif", make_image("GIF"), "image/gif")},
    )
    assert response.status_code == 415
    assert response.json()["error_code"] == "UNSUPPORTED_IMAGE_TYPE"


def test_document_upload_rejects_oversized_image(fake_redis):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("large.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "FILE_TOO_LARGE"


def test_job_status(fake_redis):
    fake_redis.hashes["receipt-job:job-123"] = {
        "status": "COMPLETED",
        "result": '{"vendor_name":"Shop"}',
        "error_code": "",
        "message": "",
    }
    response = client.get("/api/v1/jobs/job-123")
    assert response.status_code == 200
    assert response.json() == {
        "job_id": "job-123",
        "status": "COMPLETED",
        "result": {"vendor_name": "Shop"},
    }


@pytest.mark.parametrize("status", ["QUEUED", "PROCESSING"])
def test_document_job_status_reports_active_state(fake_redis, status):
    fake_redis.hashes[f"document-job:doc-{status.lower()}"] = {
        "status": status,
        "document_type": "",
        "error_code": "",
        "message": "",
    }
    response = client.get(f"/api/v1/jobs/doc-{status.lower()}")
    assert response.status_code == 200
    assert response.json()["status"] == status
    assert response.json()["document_type"] is None


@pytest.mark.parametrize("document_type", ["business_card", "receipt_invoice"])
def test_document_job_status_returns_typed_result(fake_redis, document_type):
    fake_redis.hashes[f"document-job:{document_type}"] = {
        "status": "COMPLETED",
        "document_type": document_type,
        "result": '{"name":"Ada"}' if document_type == "business_card" else '{"vendor_name":"Shop"}',
        "error_code": "",
        "message": "",
    }
    response = client.get(f"/api/v1/jobs/{document_type}")
    assert response.status_code == 200
    assert response.json() == {
        "job_id": document_type,
        "status": "COMPLETED",
        "document_type": document_type,
        "result": {
            "document_type": document_type,
            "result": {"name": "Ada"} if document_type == "business_card" else {"vendor_name": "Shop"},
        },
    }


def test_document_job_status_reports_rejection(fake_redis):
    fake_redis.hashes["document-job:unknown-doc"] = {
        "status": "REJECTED",
        "document_type": "unknown",
        "error": "Unsupported document type",
        "error_code": "UNSUPPORTED_DOCUMENT_TYPE",
        "message": "Unsupported document type",
    }
    response = client.get("/api/v1/jobs/unknown-doc")
    assert response.status_code == 200
    assert response.json()["status"] == "REJECTED"
    assert response.json()["document_type"] == "unknown"
    assert response.json()["error"]["message"] == "Unsupported document type"


def test_document_job_status_reports_failure(fake_redis):
    fake_redis.hashes["document-job:failed-doc"] = {
        "status": "FAILED",
        "document_type": "",
        "error_code": "PROCESSING_FAILED",
        "message": "Document processing failed.",
    }
    response = client.get("/api/v1/jobs/failed-doc")
    assert response.status_code == 200
    assert response.json()["status"] == "FAILED"
    assert response.json()["error"]["message"] == "Document processing failed."


def test_job_status_preserves_serialized_line_item_amounts(fake_redis):
    fake_redis.hashes["receipt-job:amounts"] = {
        "status": "COMPLETED",
        "result": '{"line_items":[{"description":"CHILLY PAROTTA","quantity":1,"unit_price":110,"amount":110},{"description":"TEA","quantity":1,"unit_price":60,"amount":60},{"description":"WATER","quantity":1,"unit_price":90,"amount":90}]}',
        "error_code": "",
        "message": "",
    }
    response = client.get("/api/v1/jobs/amounts")
    assert response.status_code == 200
    assert [item["amount"] for item in response.json()["result"]["line_items"]] == [110, 60, 90]


def test_unknown_job_status_is_404(fake_redis):
    response = client.get("/api/v1/jobs/missing")
    assert response.status_code == 404
    assert response.json()["error_code"] == "JOB_NOT_FOUND"


def test_invalid_image_type(fake_redis):
    response = client.post(
        "/api/v1/receipts",
        files={"file": ("receipt.gif", make_image("GIF"), "image/gif")},
    )
    assert response.status_code == 415
    assert response.json()["error_code"] == "UNSUPPORTED_IMAGE_TYPE"


def test_oversized_image(fake_redis):
    contents = b"x" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/v1/receipts",
        files={"file": ("large.png", contents, "image/png")},
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "FILE_TOO_LARGE"


def test_corrupted_image(fake_redis):
    response = client.post(
        "/api/v1/receipts",
        files={"file": ("broken.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["error_code"] == "INVALID_IMAGE"
