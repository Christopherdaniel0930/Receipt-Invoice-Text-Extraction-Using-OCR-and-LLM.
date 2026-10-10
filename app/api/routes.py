from io import BytesIO
import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.common.config import get_settings
from app.common.errors import ApiError
from app.workers.jobs import JOB_TTL_SECONDS, document_job_key, job_key

router = APIRouter(prefix="/api/v1/receipts", tags=["receipts"])
documents_router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
logger = logging.getLogger(__name__)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_IMAGE_PIXELS = 40_000_000


async def _submit_image(request, file, *, document=False):
    settings = get_settings()
    contents = await file.read(settings.max_upload_bytes + 1)
    if len(contents) > settings.max_upload_bytes:
        raise ApiError(413, "FILE_TOO_LARGE", "Image must be 10 MB or smaller.")
    try:
        with Image.open(BytesIO(contents)) as image:
            image_format = image.format
            if image_format not in ALLOWED_FORMATS:
                raise ApiError(415, "UNSUPPORTED_IMAGE_TYPE", "Upload a JPEG, PNG, or WEBP image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ApiError(413, "IMAGE_DIMENSIONS_TOO_LARGE", "Image dimensions are too large to process.")
            image.verify()
    except ApiError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise ApiError(400, "INVALID_IMAGE", "The uploaded image is corrupted or invalid.")

    suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[image_format]
    temp_path = None
    job_id = str(uuid4())
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        name = "Document" if document else "Receipt"
        raise ApiError(503, "QUEUE_UNAVAILABLE", f"{name} processing queue is unavailable.")

    key = document_job_key(job_id) if document else job_key(job_id)
    worker_function = "process_document" if document else "process_receipt"
    prefix = "document" if document else "receipt"
    label = "Document" if document else "Receipt"

    try:
        with tempfile.NamedTemporaryFile(prefix=f"{prefix}-", suffix=suffix, delete=False) as image_file:
            image_file.write(contents)
            temp_path = image_file.name

        await redis.hset(
            key,
            mapping={
                "job_id": job_id,
                "job_kind": "document" if document else "receipt",
                "status": "QUEUED",
                "error_code": "",
                "message": "",
            },
        )
        await redis.expire(key, JOB_TTL_SECONDS)
        queued = await redis.enqueue_job(worker_function, job_id, temp_path, _job_id=job_id)
        if queued is None:
            raise RuntimeError(f"ARQ did not enqueue the {label.lower()} job")
        logger.info("%s job %s queued", label, job_id)
        return {"job_id": job_id, "status": "queued"}
    except Exception as exc:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        try:
            await redis.delete(key)
        except Exception:
            pass
        logger.error("%s job enqueue failed (%s)", label, type(exc).__name__)
        raise ApiError(503, "QUEUE_UNAVAILABLE", f"{label} processing queue is unavailable.")


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_receipt(request: Request, file: UploadFile = File(...)):
    return await _submit_image(request, file, document=False)


@documents_router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_document(request: Request, file: UploadFile = File(...)):
    return await _submit_image(request, file, document=True)
