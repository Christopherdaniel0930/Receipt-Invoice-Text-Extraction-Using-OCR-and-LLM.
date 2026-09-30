from io import BytesIO
import logging
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.core.errors import ApiError
from app.workers.jobs import JOB_TTL_SECONDS, job_key

router = APIRouter(prefix="/api/v1/receipts", tags=["receipts"])
logger = logging.getLogger(__name__)
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_receipt(request: Request, file: UploadFile = File(...)):
    settings = get_settings()
    contents = await file.read(settings.max_upload_bytes + 1)
    if len(contents) > settings.max_upload_bytes:
        raise ApiError(413, "FILE_TOO_LARGE", "Image must be 10 MB or smaller.")
    try:
        with Image.open(BytesIO(contents)) as image:
            image_format = image.format
            if image_format not in ALLOWED_FORMATS:
                raise ApiError(415, "UNSUPPORTED_IMAGE_TYPE", "Upload a JPEG, PNG, or WEBP image.")
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
        raise ApiError(503, "QUEUE_UNAVAILABLE", "Receipt processing queue is unavailable.")

    try:
        with tempfile.NamedTemporaryFile(prefix="receipt-", suffix=suffix, delete=False) as image_file:
            image_file.write(contents)
            temp_path = image_file.name

        await redis.hset(
            job_key(job_id),
            mapping={"job_id": job_id, "status": "QUEUED", "error_code": "", "message": ""},
        )
        await redis.expire(job_key(job_id), JOB_TTL_SECONDS)
        queued = await redis.enqueue_job(
            "process_receipt", job_id, temp_path, _job_id=job_id
        )
        if queued is None:
            raise RuntimeError("ARQ did not enqueue the receipt job")
        logger.info("Receipt job %s queued", job_id)
        return {"job_id": job_id, "status": "queued"}
    except Exception as exc:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
        try:
            await redis.delete(job_key(job_id))
        except Exception:
            pass
        logger.error("Receipt job enqueue failed (%s)", type(exc).__name__)
        raise ApiError(503, "QUEUE_UNAVAILABLE", "Receipt processing queue is unavailable.")
