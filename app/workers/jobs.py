import asyncio
import json
import logging
from pathlib import Path
from time import perf_counter

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
    from app.receipt.modules.currency_recovery import get_engine as get_currency_engine
    from app.common.ocr import get_engine

    started = perf_counter()
    get_engine()
    logger.info(
        "ARQ worker stage=rapidocr_engine_warm elapsed_ms=%.2f",
        (perf_counter() - started) * 1000,
    )
    started = perf_counter()
    get_currency_engine()
    logger.info(
        "ARQ worker stage=currency_ocr_engine_warm elapsed_ms=%.2f",
        (perf_counter() - started) * 1000,
    )
    logger.info("ARQ worker OCR engines initialized")


async def _run_pipeline(path: Path, receipt_id: str):
    """Keep the input alive until the synchronous pipeline thread has stopped."""
    from app.main import process
    from app.common.processing_timing import bind_stage_callback, reset_stage_callback

    def run_with_timing():
        def log_stage(stage: str, elapsed_ms: float, succeeded: bool) -> None:
            result = "completed" if succeeded else "failed"
            logger.info(
                "Receipt job %s stage=%s result=%s elapsed_ms=%.2f",
                receipt_id,
                stage,
                result,
                elapsed_ms,
            )

        token = bind_stage_callback(log_stage)
        try:
            return process(str(path))
        finally:
            reset_stage_callback(token)

    task = asyncio.create_task(asyncio.to_thread(run_with_timing))
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Cancelling to_thread does not stop its thread. Wait for it before
            # letting ARQ retry or cleanup remove the image it is reading.
            if task.done():
                return task.result()
            continue


async def process_receipt(ctx, receipt_id: str, image_path: str) -> None:
    """Run the established synchronous receipt pipeline as an ARQ job."""
    redis = ctx["redis"]
    path = Path(image_path)
    terminal = True
    max_tries = int(ctx.get("max_tries", 3))
    processing_started = perf_counter()
    try:
        await _save_job(redis, receipt_id, status="PROCESSING")
        logger.info("Receipt job %s processing started", receipt_id)
        receipt = await _run_pipeline(path, receipt_id)
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
    except asyncio.CancelledError:
        # ARQ may cancel a timed out attempt. Keep the image for the retry and
        # avoid leaving a stale PROCESSING status behind.
        job_try = int(ctx.get("job_try", 1))
        if job_try < max_tries:
            terminal = False
            try:
                await asyncio.shield(_save_job(redis, receipt_id, status="QUEUED"))
            except Exception:
                logger.error("Receipt job %s retry status update failed", receipt_id)
        else:
            try:
                await asyncio.shield(_save_job(
                    redis, receipt_id, status="FAILED",
                    error_code="PROCESSING_TIMEOUT",
                    message="Receipt processing timed out.",
                ))
            except Exception:
                logger.error("Receipt job %s final status update failed", receipt_id)
        raise
    except Exception as exc:
        if _is_retryable_upstream_error(exc):
            job_try = int(ctx.get("job_try", 1))
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
        logger.info(
            "Receipt job %s stage=worker_processing_total elapsed_ms=%.2f",
            receipt_id,
            (perf_counter() - processing_started) * 1000,
        )
        if terminal:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Receipt job %s image cleanup failed (%s)", receipt_id, type(exc).__name__)
