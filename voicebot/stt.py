"""Soniox STT with a finalize watchdog.

The bug this exists for
-----------------------
Soniox streams tokens continuously but Pipecat only turns them into a
``TranscriptionFrame`` when an **end token** (``<end>``/``<fin>``) arrives —
until then finalized tokens sit in ``_final_transcription_buffer``
(``pipecat/services/soniox/stt.py``, ``send_endpoint_transcript``). The end
token is normally provoked by a ``{"type":"finalize"}`` message that the
service sends when it sees ``VADUserStoppedSpeakingFrame``.

If that finalize is missed — the websocket was reconnecting, the VAD frame did
not reach the service, or Soniox simply did not answer — nothing else ever asks
for one. The buffered text is then released only when the *next* utterance
produces an end token, so the caller's previous sentence and their new one
arrive glued together, one turn late. That is the "transcription gets stuck on
the last chunk until you speak again" symptom.

Pipecat 1.7.0 has the makings of a guard: it records ``_last_tokens_received``
on every token batch, and a comment about "auto finalize delay". But nothing
reads that attribute — grep the module and it is written once and never used.
There is no timer, so there is no recovery.

This subclass supplies the missing timer. It watches for the specific stalled
state — tokens buffered, nothing new arriving — and re-sends ``finalize``. On a
healthy call it never fires: the VAD-driven finalize gets there first and the
buffer is empty within a few hundred milliseconds.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, StartFrame
from pipecat.services.soniox.stt import FINALIZE_MESSAGE, SonioxSTTService

# How often the watchdog looks. Cheap: an attribute read and a clock compare.
_POLL_SECS = 0.2


class FinalizingSonioxSTTService(SonioxSTTService):
    """Soniox STT that re-asks for a transcript when one gets stranded."""

    def __init__(self, *, finalize_after: float = 1.5, **kwargs):
        """Initialize the service.

        Args:
            finalize_after: Seconds of silence from Soniox, with text still
                buffered, before re-sending ``finalize``. Wants to sit well
                clear of normal turn timing — the VAD-driven finalize lands
                within a few hundred ms of the caller stopping — so that this
                only ever fires on a genuine stall. 0 disables the watchdog.
            **kwargs: Passed to :class:`SonioxSTTService`.
        """
        super().__init__(**kwargs)
        self._finalize_after = finalize_after
        self._watchdog: asyncio.Task | None = None
        self._rescues = 0
        self._log = logger.bind(component="stt")

    @property
    def rescued_transcripts(self) -> int:
        """Transcripts this call that needed a re-sent finalize."""
        return self._rescues

    async def start(self, frame: StartFrame) -> None:
        """Start the service and arm the watchdog."""
        await super().start(frame)
        if self._finalize_after > 0 and self._watchdog is None:
            self._watchdog = self.create_task(self._watch())

    async def stop(self, frame: EndFrame) -> None:
        """Stop the watchdog, then the service."""
        await self._stop_watchdog()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the watchdog, then the service."""
        await self._stop_watchdog()
        await super().cancel(frame)

    async def _stop_watchdog(self) -> None:
        if self._watchdog is not None:
            await self.cancel_task(self._watchdog)
            self._watchdog = None

    async def _watch(self) -> None:
        """Re-send finalize when a transcript looks stranded."""
        while True:
            await asyncio.sleep(_POLL_SECS)
            try:
                # Created inside _receive_messages, so it may not exist yet.
                buffered = getattr(self, "_final_transcription_buffer", None)
                last = getattr(self, "_last_tokens_received", None)
                if not buffered or last is None:
                    continue
                if time.time() - last < self._finalize_after:
                    continue

                websocket = self._websocket
                if websocket is None or websocket.state.name != "OPEN":
                    continue

                text = "".join(token.get("text", "") for token in buffered)
                await websocket.send(FINALIZE_MESSAGE)
                # Reset the clock so a slow Soniox is asked once per stall
                # rather than every poll.
                self._last_tokens_received = time.time()
                self._rescues += 1
                self._log.warning(
                    "transcript stalled; re-sent finalize",
                    event="stt_finalize_rescue",
                    stalled_secs=round(self._finalize_after, 2),
                    text=text[:120],
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a watchdog must never kill the call
                self._log.warning(f"finalize watchdog error: {exc}")
