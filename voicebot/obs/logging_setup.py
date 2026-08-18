"""Loguru configuration: human-readable stdout, JSONL on disk, and Loki push.

Pipecat logs through loguru natively, so configuring the root loguru logger
captures framework logs and our own with one setup call. Every record is
enriched with the ambient ``call_id``/``turn`` from :mod:`voicebot.obs.context`,
which is what makes per-call filtering possible in Grafana.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from voicebot.config import Settings
from voicebot.obs.context import current_call_id, current_turn
from voicebot.obs.loki import LokiSink, format_loki_line, now_ns

_STDOUT_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> "
    "<level>{level: <7}</level> "
    "<cyan>{extra[call_id]}</cyan>/<cyan>{extra[turn]}</cyan> "
    "<level>{message}</level>"
)

_loki_sink: LokiSink | None = None


def _patcher(record: dict[str, Any]) -> None:
    """Enrich every record with call context and a pre-rendered JSON line."""
    extra = record["extra"]
    extra.setdefault("call_id", current_call_id())
    extra.setdefault("turn", current_turn())
    # Pre-render once so the JSONL file sink and the Loki sink share the work.
    extra["json_line"] = format_loki_line(record)


def setup_logging(settings: Settings) -> LokiSink | None:
    """Configure logging sinks.

    Args:
        settings: Resolved application settings.

    Returns:
        The Loki sink when direct shipping is enabled, else ``None``. The caller
        must ``await sink.start()`` once an event loop is running.
    """
    global _loki_sink

    logger.remove()
    logger.configure(patcher=_patcher)

    level = settings.log_level.upper()

    # 1. Human-readable console output.
    logger.add(
        sys.stderr,
        level=level,
        format=_STDOUT_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # 2. Structured JSONL on disk. Also the fallback ingestion path: point
    #    Promtail/Grafana Alloy at this file if you'd rather not push directly.
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "voicebot.jsonl",
        level=level,
        # The patcher renders the JSON; braces inside the value are not
        # re-interpreted by loguru's formatter.
        format="{extra[json_line]}",
        rotation="50 MB",
        retention="7 days",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    # 3. Direct push to Loki.
    if settings.loki_url:
        _loki_sink = LokiSink(
            settings.loki_url,
            labels={
                "service": settings.service_name,
                "env": settings.env,
                "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
            },
            batch_secs=settings.loki_batch_secs,
            batch_size=settings.loki_batch_size,
            timeout_secs=settings.loki_timeout_secs,
        )
        sink = _loki_sink

        def _emit(message: Any) -> None:
            record = message.record
            sink.emit(
                now_ns(),
                record["extra"]["json_line"],
                {
                    "level": record["level"].name.lower(),
                    "component": str(record["extra"].get("component", "bot")),
                },
            )

        logger.add(_emit, level=level, format="{message}")

    logger.bind(component="startup").info(
        "logging configured",
        level=level,
        loki_enabled=bool(settings.loki_url),
        log_dir=str(log_dir),
    )
    return _loki_sink


async def shutdown_logging() -> None:
    """Flush and close the Loki sink, if one is running."""
    global _loki_sink
    if _loki_sink is not None:
        await _loki_sink.stop()
        _loki_sink = None
