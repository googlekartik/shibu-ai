"""A small in-process rate limiter for costly cloud endpoints.

It is deliberately a safety belt, not distributed billing enforcement.  For a
multi-instance Render deployment, set an edge/WAF limit too.  This prevents one
browser/session from rapidly consuming embedding or LLM quota on one process.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries = defaultdict(deque)

    def allow(self, key: str, limit: int, window_seconds: int) -> bool:
        now = time.monotonic()
        limit = max(1, min(int(limit), 1000))
        window_seconds = max(1, min(int(window_seconds), 3600))
        with self._lock:
            events = self._entries[key]
            cutoff = now - window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                return False
            events.append(now)
            # Avoid an unbounded map in a long-lived process.
            if len(self._entries) > 5000:
                stale = [k for k, values in self._entries.items() if not values or values[-1] <= cutoff]
                for stale_key in stale:
                    self._entries.pop(stale_key, None)
            return True


request_limiter = SlidingWindowLimiter()
