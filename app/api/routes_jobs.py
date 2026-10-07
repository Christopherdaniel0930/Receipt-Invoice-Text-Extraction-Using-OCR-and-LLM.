import json

from fastapi import APIRouter, Request

from app.common.errors import ApiError
from app.workers.jobs import job_key

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


@router.get("/{job_id}")
async def get_job_status(job_id: str, request: Request):
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise ApiError(503, "QUEUE_UNAVAILABLE", "Receipt processing queue is unavailable.")
    values = await redis.hgetall(job_key(job_id))
    if not values:
        raise ApiError(404, "JOB_NOT_FOUND", "Receipt job was not found.")

    job = {_decode(key): _decode(value) for key, value in values.items()}
    response = {"job_id": job_id, "status": job.get("status", "QUEUED")}
    if job.get("result"):
        response["result"] = json.loads(job["result"])
    if job.get("error_code"):
        response["error"] = {
            "error_code": job["error_code"],
            "message": job.get("message", "Receipt processing failed."),
        }
    return response
