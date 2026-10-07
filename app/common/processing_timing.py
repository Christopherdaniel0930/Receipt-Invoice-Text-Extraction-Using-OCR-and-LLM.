"""Optional per-stage timing hooks for worker pipeline runs."""

from contextlib import contextmanager
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Callable


StageCallback = Callable[[str, float, bool], None]
_stage_callback: ContextVar[StageCallback | None] = ContextVar(
    "receipt_stage_callback", default=None
)


def bind_stage_callback(callback: StageCallback) -> Token:
    return _stage_callback.set(callback)


def reset_stage_callback(token: Token) -> None:
    _stage_callback.reset(token)


@contextmanager
def timed_stage(name: str):
    callback = _stage_callback.get()
    started = perf_counter()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        if callback is not None:
            elapsed_ms = (perf_counter() - started) * 1000
            try:
                callback(name, elapsed_ms, succeeded)
            except Exception:
                # Timing instrumentation must never fail receipt processing.
                pass
