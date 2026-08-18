"""Per-call ambient context.

`call_id` and `turn` are carried in context variables so every log line — including
ones emitted deep inside Pipecat itself — can be correlated to a call without
threading an argument through every call site.
"""

from __future__ import annotations

import contextlib
import uuid
from contextvars import ContextVar
from typing import Iterator

_call_id: ContextVar[str] = ContextVar("voicebot_call_id", default="-")
_turn: ContextVar[int] = ContextVar("voicebot_turn", default=0)


def current_call_id() -> str:
    """Return the call id bound to the current context ("-" outside a call)."""
    return _call_id.get()


def current_turn() -> int:
    """Return the conversation turn number bound to the current context."""
    return _turn.get()


def set_turn(turn: int) -> None:
    """Set the current turn number for subsequent log lines and metrics."""
    _turn.set(turn)


def new_call_id() -> str:
    """Generate a short, log-friendly call identifier."""
    return uuid.uuid4().hex[:12]


@contextlib.contextmanager
def call_context(call_id: str | None = None) -> Iterator[str]:
    """Bind a call id for the duration of the block.

    Args:
        call_id: Identifier to bind. Generated when omitted.

    Yields:
        The bound call id.
    """
    cid = call_id or new_call_id()
    call_token = _call_id.set(cid)
    turn_token = _turn.set(0)
    try:
        yield cid
    finally:
        _call_id.reset(call_token)
        _turn.reset(turn_token)
