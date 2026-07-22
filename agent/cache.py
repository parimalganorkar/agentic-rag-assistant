"""Phase 10 — small in-process caches.

Deliberately minimal. One class, `Cache`: a dict with a size cap (LRU eviction)
and an optional TTL, plus hit/miss counters so the LangSmith trace can show
whether a cache is actually earning its place. No Redis, no external store —
this stays a locally-runnable project, and the working set (one user, repeated
queries in a session) fits in memory.

Three instances are created here and imported where they're needed:

    EMBED_CACHE   query text        -> embedding vector      (deterministic, no TTL)
    VERDICT_CACHE (guard,q,answer)  -> guard verdict          (LLM output, TTL)
    ANSWER_CACHE  (history,query)   -> final guarded answer   (only CLEAN answers, TTL)

Keys are hashed so a huge answer string never becomes a dict key by value.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

_SENTINEL = object()


def key_of(*parts: Any) -> str:
    """Stable short key from arbitrary parts (order matters)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


class Cache:
    """Thread-safe LRU cache with optional per-entry TTL."""

    def __init__(self, name: str, maxsize: int = 1024, ttl_seconds: float | None = None):
        self.name = name
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        """Return the cached value, or the module sentinel if absent/expired."""
        with self._lock:
            item = self._data.get(key, _SENTINEL)
            if item is _SENTINEL:
                self.misses += 1
                return _SENTINEL
            ts, value = item
            if self.ttl is not None and (time.time() - ts) > self.ttl:
                del self._data[key]           # expired
                self.misses += 1
                return _SENTINEL
            self._data.move_to_end(key)        # LRU touch
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)  # evict least-recently-used

    def get_or_compute(self, key: str, compute: Callable[[], Any]) -> Any:
        """Return cached value, or compute + store it. `compute` runs OUTSIDE the
        lock so a slow model call doesn't block other cache users."""
        hit = self.get(key)
        if hit is not _SENTINEL:
            return hit
        value = compute()
        self.set(key, value)
        return value

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "name": self.name,
            "size": len(self._data),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0


# The sentinel is exported so callers can distinguish "cached None" from "absent".
MISSING = _SENTINEL

# Deterministic → no TTL. Query embeddings never change for the same text.
EMBED_CACHE = Cache("embed", maxsize=4096, ttl_seconds=None)

# LLM guard verdicts. temperature=0, but models drift over time, so a 1h TTL
# bounds staleness while still collapsing repeats within a session/eval run.
VERDICT_CACHE = Cache("verdict", maxsize=2048, ttl_seconds=3600)

# Whole guarded answers. Short TTL because the live docs / PyPI can change under
# a tool answer; only CLEAN answers are ever stored (see graph wiring).
ANSWER_CACHE = Cache("answer", maxsize=512, ttl_seconds=900)


def all_stats() -> list[dict]:
    return [c.stats() for c in (EMBED_CACHE, VERDICT_CACHE, ANSWER_CACHE)]
