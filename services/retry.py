"""Step 5: retries for transient DynamoDB / network failures.

Only transient errors are retried, and only a few times:
  * ProvisionedThroughputExceededException / ThrottlingException
  * RequestLimitExceeded, InternalServerError, ServiceUnavailable
  * network errors from boto3 / requests (ConnectionError, Timeout)
Permanent errors (ConditionalCheckFailed, ValidationError, 4xx auth,
access denied, not found) are NEVER retried — retrying them would
just burn capacity and hide a real bug.

Backoff is exponential with jitter: 0.1s, 0.2s, 0.4s by default
(max 3 attempts). A chat answer must never wait long for a retry,
so the defaults are deliberately short.
"""

from __future__ import annotations

import functools
import random
import time

TRANSIENT_NAMES = (
    "ProvisionedThroughputExceededException",
    "ThrottlingException",
    "RequestLimitExceeded",
    "InternalServerError",
    "ServiceUnavailable",
    "Throttling",
)


def _is_transient(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name in ("ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout"):
        return True
    text = str(exc)
    if any(n in text for n in TRANSIENT_NAMES):
        return True
    # boto3 ClientError: check the error code, not the message.
    code = ""
    try:
        code = exc.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
    except Exception:
        pass
    return code in TRANSIENT_NAMES


def retry_on_transient(max_attempts: int = 3, base_delay: float = 0.1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised below if permanent
                    last_exc = exc
                    if not _is_transient(exc) or attempt == max_attempts - 1:
                        raise
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.05)
                    time.sleep(delay)
            raise last_exc  # pragma: no cover - loop always raises first
        return wrapper
    return decorator
