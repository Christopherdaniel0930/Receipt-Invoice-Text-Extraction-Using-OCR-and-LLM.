import asyncio
import json
import logging
from pathlib import Path

from arq.worker import Retry
from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError

logger = logging.getLogger(__name__)
JOB_TTL_SECONDS = 24 * 60 * 60
RETRY_DELAY_SECONDS = 5


def job_key(receipt_id: str) -> str:
    return f"receipt-job:{receipt_id}"


def _is_retryable_upstream_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    return isinstance(error, APIStatusError) and (
        error.status_code in {408, 409, 429} or error.status_code >= 500
    )


async def _save_job(redis, receipt_id: str, **fields) -> None:
    key = job_key(receipt_id)
    await redis.hset(key, mapping=fields)
    await redis.expire(key, JOB_TTL_SECONDS)


async def startup(ctx) -> None:
    """Warm the worker's two process-local OCR engines once."""
    from app.currency_recovery import get_engine as get_currency_engine
    from app.ocr import get_engine

    get_engine()
    get_currency_engine()
    logger.info("ARQ worker OCR engines initialized")


async def process_receipt(ctx, receipt_id: str, image_path: str) -> None:
    """Run the established synchronous receipt pipeline as an ARQ job."""
    redis = ctx["redis"]
    path = Path(image_path)
    terminal = True
    await _save_job(redis, receipt_id, status="PROCESSING")
    logger.info("Receipt job %s processing started", receipt_id)

    try:
        from app.main import process

        receipt = await asyncio.to_thread(process, str(path))
        result = receipt.model_dump(mode="json")
        await _save_job(
            redis,
            receipt_id,
            status="COMPLETED",
            result=json.dumps(result, ensure_ascii=False),
            error_code="",
            message="",
        )
        logger.info("Receipt job %s completed", receipt_id)
    except RuntimeError as exc:
        if "not a receipt or invoice" in str(exc).lower():
            await _save_job(
                redis,
                receipt_id,
                status="REJECTED",
                error_code="NOT_A_RECEIPT",
                message="Uploaded image is not a receipt or invoice.",
            )
            logger.info("Receipt job %s rejected", receipt_id)
        else:
            await _save_job(
                redis,
                receipt_id,
                status="FAILED",
                error_code="PROCESSING_FAILED",
                message="Receipt processing failed.",
            )
            logger.error("Receipt job %s failed (%s)", receipt_id, type(exc).__name__)
    except Exception as exc:
        if _is_retryable_upstream_error(exc):
            job_try = int(ctx.get("job_try", 1))
            max_tries = int(ctx.get("max_tries", 3))
            if job_try < max_tries:
                terminal = False
                await _save_job(redis, receipt_id, status="QUEUED", error_code="")
                logger.warning("Receipt job %s retry scheduled (%s)", receipt_id, type(exc).__name__)
                raise Retry(defer=RETRY_DELAY_SECONDS) from exc
            error_code = "UPSTREAM_UNAVAILABLE"
            message = "Receipt extraction service is temporarily unavailable."
        else:
            error_code = "PROCESSING_FAILED"
            message = "Receipt processing failed."
        await _save_job(
            redis,
            receipt_id,
            status="FAILED",
            error_code=error_code,
            message=message,
        )
        logger.error("Receipt job %s failed (%s)", receipt_id, type(exc).__name__)
    finally:
        if terminal:
            path.unlink(missing_ok=True)
