"""Small, dependency-free retry helpers for outbound cloud calls.

Retries are deliberately limited to transient failures.  Validation, auth and
dimension errors are returned immediately so a bad configuration does not turn
into a slow or expensive retry loop.
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar


T = TypeVar("T")


def is_transient_error(exc: Exception) -> bool:
    """Best-effort check for rate limits, server errors and network blips."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        if int(status) == 429 or int(status) >= 500:
            return True
    except (TypeError, ValueError):
        pass
    text = str(exc).lower()
    return any(token in text for token in (
        "timeout", "timed out", "temporar", "connection reset",
        "connection aborted", "rate limit", "http 429", "http 500",
        "http 502", "http 503", "http 504",
    ))


def retry_cloud_call(operation: Callable[[], T], attempts: int = 3,
                     base_delay: float = 0.25, max_delay: float = 2.0,
                     retry_if: Callable[[Exception], bool] = is_transient_error,
                     sleep: Callable[[float], None] = time.sleep) -> T:
    """Run a cloud operation with bounded exponential backoff + small jitter."""
    attempts = max(1, min(int(attempts), 5))
    last_error = None
    for number in range(attempts):
        try:
            return operation()
        except Exception as exc:  # The caller adds provider-specific context.
            last_error = exc
            if number == attempts - 1 or not retry_if(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** number))
            sleep(delay + random.uniform(0, min(0.1, delay / 4)))
    raise last_error  # pragma: no cover - loop always returns or raises
