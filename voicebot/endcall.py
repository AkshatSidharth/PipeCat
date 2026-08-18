"""Hang up after the bot has finished saying goodbye.

The hard part is *when*, not *whether*. Ending on the text is too early — the
words have been generated but not spoken, and the caller hears the line die
mid-sentence. Ending on a timer is a guess. The correct signal is the transport
telling us the audio actually finished playing, which is
``BotStoppedSpeakingFrame``.

So this runs in two steps:

1. **Arm** when a bot response *ends with* a configured goodbye.
2. **Fire** on the next ``BotStoppedSpeakingFrame`` — the goodbye has now been
   heard — by pushing ``EndWorkerFrame`` downstream, which closes the pipeline
   gracefully and flushes whatever is still queued.

Why *ends with* rather than *contains*
--------------------------------------
A substring match hangs up on a mid-call "धन्यवाद". A goodbye is by definition
the last thing said, so the phrase is matched against the tail of the response
with trailing punctuation stripped. That turns a whole class of false hang-ups
into a non-issue, at the cost of missing a goodbye the model buries mid-reply —
which is the right way round: a call that fails to auto-end is an annoyance, one
that hangs up on a talking customer is a complaint.

Matching reuses :func:`voicebot.backchannel.normalize`, so it is punctuation-
and case-insensitive and does not mangle Devanagari (a ``\\w``-based strip
would — see that module).
"""

from __future__ import annotations

import asyncio

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voicebot.backchannel import normalize

# Full closing lines, not bare words. "धन्यवाद" alone appears mid-call all the
# time; "धन्यवाद, आपका दिन शुभ हो" does not.
DEFAULT_GOODBYES_HI = (
    "धन्यवाद आपका दिन शुभ हो",
    "कॉल करने के लिए धन्यवाद",
    "आपका दिन शुभ हो",
    "धन्यवाद नमस्कार",
)

DEFAULT_GOODBYES_EN = (
    "thank you for calling have a nice day",
    "thank you for calling",
    "have a great day goodbye",
    "goodbye have a nice day",
)


def parse_goodbyes(raw: str, language: str = "hi") -> list[str]:
    """Parse a newline-separated goodbye list, normalized for matching.

    Args:
        raw: One closing line per line. Blank uses the built-in set.
        language: Base language code, used only to pick the fallback set.

    Returns:
        Normalized phrases, never empty.
    """
    phrases = [normalize(line) for line in (raw or "").splitlines() if line.strip()]
    phrases = [p for p in phrases if p]
    if phrases:
        return phrases
    default = (
        DEFAULT_GOODBYES_HI
        if language.split("-")[0].lower() == "hi"
        else DEFAULT_GOODBYES_EN
    )
    return [normalize(p) for p in default]


class EndCallProcessor(FrameProcessor):
    """Ends the call once a goodbye has finished being spoken."""

    def __init__(
        self,
        *,
        goodbyes: list[str],
        linger_secs: float = 0.4,
        on_end=None,
        **kwargs,
    ):
        """Initialize.

        Args:
            goodbyes: Normalized closing phrases. A response ending with any of
                them arms the hang-up.
            linger_secs: Pause between the audio finishing and the pipeline
                closing. Covers the transport's own jitter buffer, which still
                holds a little audio when ``BotStoppedSpeakingFrame`` fires;
                without it the final syllable can be clipped on a PSTN leg.
            on_end: Optional callback invoked when the call is ended, for
                reporting.
            **kwargs: Passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._goodbyes = goodbyes
        self._linger = max(0.0, linger_secs)
        self._on_end = on_end
        self._buffer: list[str] = []
        self._armed = False
        self._ended = False
        self._matched = ""
        self._log = logger.bind(component="endcall")

    @property
    def ended_call(self) -> bool:
        """Whether this processor hung up."""
        return self._ended

    @property
    def matched_phrase(self) -> str:
        """The goodbye that triggered the hang-up, if any."""
        return self._matched

    def _is_goodbye(self, text: str) -> str | None:
        """Return the phrase this text ends with, or None."""
        cleaned = normalize(text)
        if not cleaned:
            return None
        for phrase in self._goodbyes:
            if cleaned.endswith(phrase):
                return phrase
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass everything through, watching for a spoken goodbye."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = []
        elif isinstance(frame, LLMTextFrame):
            self._buffer.append(frame.text or "")
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._arm("".join(self._buffer))
            self._buffer = []
        elif isinstance(frame, TTSSpeakFrame):
            # Fixed lines — the greeting, re-engagement prompts, or a scripted
            # sign-off — bypass the LLM entirely, so check them too.
            self._arm(frame.text or "")

        await self.push_frame(frame, direction)

        # After forwarding, so the frame that finished the audio is not held up
        # behind the shutdown we are about to start.
        if isinstance(frame, BotStoppedSpeakingFrame) and self._armed and not self._ended:
            await self._end()

    def _arm(self, text: str) -> None:
        phrase = self._is_goodbye(text)
        if phrase is None:
            return
        self._armed = True
        self._matched = phrase
        self._log.info(
            "goodbye detected; will end the call once it has been spoken",
            event="end_call_armed",
            matched=phrase,
            text=text[:160],
        )

    async def _end(self) -> None:
        self._ended = True
        if self._linger:
            await asyncio.sleep(self._linger)
        self._log.info(
            "ending call after goodbye",
            event="end_call",
            matched=self._matched,
        )
        if self._on_end is not None:
            self._on_end()
        # Downstream: EndWorkerFrame asks the worker to close gracefully,
        # flushing frames queued ahead of it rather than cutting them off.
        await self.push_frame(
            EndWorkerFrame(reason="bot said goodbye"), FrameDirection.DOWNSTREAM
        )


def build_end_call(settings, on_end=None) -> EndCallProcessor | None:
    """Build from settings, or None when disabled."""
    if not settings.end_call_enabled:
        return None
    return EndCallProcessor(
        goodbyes=parse_goodbyes(settings.end_call_phrases, settings.sarvam_language),
        linger_secs=settings.end_call_linger_secs,
        on_end=on_end,
    )
