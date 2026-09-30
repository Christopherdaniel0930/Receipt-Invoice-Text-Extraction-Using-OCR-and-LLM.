from arq.connections import RedisSettings

from app.core.config import get_settings
from app.workers.jobs import process_receipt, startup

settings = get_settings()


class WorkerSettings:
    functions = [process_receipt]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 1
    max_tries = 3
    on_startup = startup
