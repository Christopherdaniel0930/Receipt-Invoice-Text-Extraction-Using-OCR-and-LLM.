"""Load the receipt upload queue and job-status endpoint with Locust.

Run from the repository root:
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000

Set RECEIPT_IMAGE_PATH to a valid receipt image for the target environment.
"""

import mimetypes
import os
from collections import deque
from pathlib import Path

from locust import HttpUser, between, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = PROJECT_ROOT / "data" / "test" / "image5.jpeg"
IMAGE_PATH = Path(os.getenv("data/test/", str(DEFAULT_IMAGE)))
MAX_PENDING_JOBS = int(os.getenv("MAX_PENDING_JOBS", "500"))


class ReceiptUser(HttpUser):
    """Submit receipts continuously and poll previously submitted jobs."""

    wait_time = between(1, 3)

    def on_start(self):
        self.pending_jobs = deque()
        self.image_bytes = IMAGE_PATH.read_bytes()
        self.image_mime = mimetypes.guess_type(IMAGE_PATH.name)[0] or "image/jpeg"

    @task(1)
    def submit_receipt(self):
        if len(self.pending_jobs) >= MAX_PENDING_JOBS:
            return

        with self.client.post(
            "/api/v1/receipts",
            files={"file": (IMAGE_PATH.name, self.image_bytes, self.image_mime)},
            name="POST /api/v1/receipts",
            catch_response=True,
        ) as response:
            if response.status_code != 202:
                response.failure(f"Upload returned HTTP {response.status_code}")
                return
            try:
                job_id = response.json().get("job_id")
            except ValueError:
                response.failure("Upload response was not valid JSON")
                return
            if not job_id:
                response.failure("Upload response did not include job_id")
                return
            response.success()
            self.pending_jobs.append(job_id)

    @task(4)
    def poll_receipt_job(self):
        if not self.pending_jobs:
            return

        job_id = self.pending_jobs.popleft()
        with self.client.get(
            f"/api/v1/jobs/{job_id}",
            name="GET /api/v1/jobs/[job_id]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Job status returned HTTP {response.status_code}")
                return
            try:
                job_status = response.json().get("status")
            except ValueError:
                response.failure("Job status response was not valid JSON")
                return

            if job_status in {"QUEUED", "PROCESSING"}:
                response.success()
                self.pending_jobs.append(job_id)
            elif job_status == "COMPLETED":
                response.success()
            elif job_status in {"FAILED", "REJECTED"}:
                response.failure(f"Receipt job ended as {job_status}")
            else:
                response.failure(f"Unexpected job status: {job_status!r}")
