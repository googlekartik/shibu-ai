"""Process-local, privacy-safe retrieval observability.

This intentionally stores aggregate timing and result counts only.  Questions,
document text, IDs and provider credentials are never retained in metrics.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RetrievalMetrics:
    def __init__(self, window_size: int = 200):
        self._lock = threading.Lock()
        self._window_size = window_size
        self._events = defaultdict(lambda: deque(maxlen=window_size))
        self._counts = defaultdict(int)

    def record(self, operation: str, duration_ms: float, success: bool,
               result_count: int = 0) -> None:
        with self._lock:
            self._counts[f"{operation}.requests"] += 1
            if not success:
                self._counts[f"{operation}.errors"] += 1
            self._events[operation].append((max(0.0, float(duration_ms)), max(0, int(result_count))))

    def snapshot(self) -> dict:
        with self._lock:
            operations = {}
            for name, events in self._events.items():
                durations = sorted(item[0] for item in events)
                result_counts = [item[1] for item in events]
                if not durations:
                    continue
                p95_index = max(0, int(len(durations) * 0.95 + 0.999) - 1)
                requests = self._counts[f"{name}.requests"]
                errors = self._counts[f"{name}.errors"]
                operations[name] = {
                    "requests": requests,
                    "errors": errors,
                    "sample_size": len(durations),
                    "avg_latency_ms": round(sum(durations) / len(durations), 1),
                    "p95_latency_ms": round(durations[p95_index], 1),
                    "avg_results": round(sum(result_counts) / len(result_counts), 2),
                }
            return {"scope": "current_process", "window_size": self._window_size,
                    "generated_at": int(time.time()), "operations": operations}


retrieval_metrics = RetrievalMetrics()
