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
