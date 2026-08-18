"""Minimal async Loki client with batching and back-pressure.

Design notes
------------
* Log records are appended to a bounded in-memory deque by a *synchronous*
  loguru sink, so logging never blocks the audio pipeline and never needs an
  event loop of its own.
* A background asyncio task drains the deque and POSTs to
  ``/loki/api/v1/push`` on a timer or once a batch fills up.
* Labels are deliberately low cardinality (service, env, level, component).
  High-cardinality fields — ``call_id`` above all — live *inside* the JSON log
  line and are queried with LogQL's ``| json`` parser. Putting a per-call id in
  a Loki label would create one stream per call and melt the ingester.
* If Loki is down, batches are dropped (oldest first) rather than growing
  without bound. Logs still reach stdout and the JSONL file.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any

import httpx


class LokiSink:
    """Batches log lines and ships them to Loki's push API."""

    def __init__(
        self,
        url: str,
        *,
        labels: dict[str, str],
        batch_secs: float = 1.0,
        batch_size: int = 200,
        timeout_secs: float = 5.0,
        max_queue: int = 10_000,
    ):
        """Initialize the sink.

        Args:
            url: Full push URL, e.g. ``http://localhost:3100/loki/api/v1/push``.
            labels: Base labels attached to every stream. Keep low cardinality.
            batch_secs: Maximum time a line waits before being flushed.
            batch_size: Flush as soon as this many lines are queued.
            timeout_secs: HTTP timeout for a push.
            max_queue: Hard cap on buffered lines; oldest are dropped past it.
        """
        self._url = url
        self._labels = dict(labels)
        self._batch_secs = batch_secs
        self._batch_size = batch_size
        self._timeout = timeout_secs
        self._queue: deque[tuple[int, str, dict[str, str]]] = deque(maxlen=max_queue)
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task | None = None
        self._wakeup: asyncio.Event | None = None
        self._closed = False
        self.dropped = 0
        self.pushed = 0
        self.push_errors = 0

    # -- producer side (sync, callable from any thread) ---------------------

    def emit(self, timestamp_ns: int, line: str, extra_labels: dict[str, str]) -> None:
        """Queue one log line. Never blocks, never raises."""
        if self._closed:
            return
        if len(self._queue) == self._queue.maxlen:
            self.dropped += 1
        self._queue.append((timestamp_ns, line, extra_labels))
        # Wake the flusher when a full batch is ready. call_soon_threadsafe keeps
        # this safe from loguru's writer thread.
        if self._wakeup is not None and len(self._queue) >= self._batch_size:
            loop = getattr(self, "_loop", None)
            if loop is not None and not loop.is_closed():
                try:
                    loop.call_soon_threadsafe(self._wakeup.set)
                except RuntimeError:
                    pass

    # -- consumer side ------------------------------------------------------

    async def start(self) -> None:
        """Start the background flush task. Idempotent."""
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._wakeup = asyncio.Event()
        self._client = httpx.AsyncClient(timeout=self._timeout)
        self._task = asyncio.create_task(self._run(), name="loki-flusher")

    async def stop(self) -> None:
        """Flush what is buffered and shut the client down."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._flush()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _run(self) -> None:
        assert self._wakeup is not None
        while True:
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=self._batch_secs)
            except asyncio.TimeoutError:
                pass
            self._wakeup.clear()
            await self._flush()

    async def _flush(self) -> None:
        if not self._queue or self._client is None:
            return

        # Drain everything currently queued.
        batch: list[tuple[int, str, dict[str, str]]] = []
        while self._queue and len(batch) < self._batch_size * 5:
            batch.append(self._queue.popleft())
        if not batch:
            return

        # Group by label set — Loki requires one entry per unique stream.
        streams: dict[tuple[tuple[str, str], ...], list[list[str]]] = {}
        for ts_ns, line, extra in batch:
            labels = {**self._labels, **extra}
            key = tuple(sorted(labels.items()))
            streams.setdefault(key, []).append([str(ts_ns), line])

        payload: dict[str, Any] = {
            "streams": [
                {"stream": dict(key), "values": sorted(values, key=lambda v: v[0])}
                for key, values in streams.items()
            ]
        }

        try:
            resp = await self._client.post(
                self._url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code >= 400:
                self.push_errors += 1
                # Deliberately printed, not logged: logging here would recurse.
                print(
                    f"[loki] push failed {resp.status_code}: {resp.text[:300]}",
                    flush=True,
                )
            else:
                self.pushed += len(batch)
        except Exception as exc:  # network hiccup, DNS, Loki restarting...
            self.push_errors += 1
            print(f"[loki] push error: {exc!r}", flush=True)


def format_loki_line(record: dict[str, Any]) -> str:
    """Render a loguru record as a compact JSON line for Loki/JSONL sinks."""
    extra = dict(record.get("extra") or {})
    payload: dict[str, Any] = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
    }
    # Flatten bound/patched fields (call_id, turn, event, metric values, ...).
    for key, value in extra.items():
        if key in payload:
            continue
        try:
            json.dumps(value)
            payload[key] = value
        except (TypeError, ValueError):
            payload[key] = repr(value)

    exception = record.get("exception")
    if exception is not None:
        payload["exception"] = str(exception)

    return json.dumps(payload, ensure_ascii=False, default=str)


def now_ns() -> int:
    """Current wall-clock time in nanoseconds (Loki's timestamp unit)."""
    return time.time_ns()
