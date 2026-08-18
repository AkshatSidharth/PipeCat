"""Bilingual support: follow the caller between Indian English and Hindi.

Soniox tags each transcript with the language it detected (when
``enable_language_identification`` is on). This processor watches that tag and
retunes Sarvam to match, so a caller who switches to Hindi mid-call gets a Hindi
voice back instead of Devanagari read by an English voice.

Sarvam applies the change on its open websocket — its ``_update_settings``
re-sends the config rather than reconnecting — so a switch costs no extra
latency on the audio path.

Code-mixing caveat
------------------
Indian callers routinely mix English and Hindi inside a single sentence
("mera order kab deliver hoga"). Soniox reports the *dominant* language of the
turn, so a Hinglish sentence lands on whichever side won. Two guards keep that
from thrashing the voice:

* only switch on languages that were explicitly configured, and
* require ``switch_after`` consecutive turns in the new language before acting.

With ``switch_after=2`` a single ambiguous Hinglish turn never flips the voice;
a caller who has genuinely changed language does, on their second turn.
"""

from __future__ import annotations

from loguru import logger
from pipecat.frames.frames import Frame, TranscriptionFrame, TTSUpdateSettingsFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.tts_service import TTSSettings
from pipecat.transcriptions.language import Language

# Soniox reports base codes ("en", "hi"); Sarvam wants regional ones.
_SARVAM_LANGUAGE = {
    Language.EN: Language.EN_IN,
    Language.EN_IN: Language.EN_IN,
    Language.HI: Language.HI_IN,
    Language.HI_IN: Language.HI_IN,
}


class LanguageFollower(FrameProcessor):
    """Retunes the TTS voice to the language the caller is speaking."""

    def __init__(
        self,
        *,
        initial: Language,
        allowed: list[Language],
        switch_after: int = 2,
        **kwargs,
    ):
        """Initialize the follower.

        Args:
            initial: Language the TTS starts in (matches the greeting).
            allowed: Languages that may be switched to. Anything Soniox reports
                outside this set is ignored, so a stray misdetection cannot
                strand the bot in a language you never configured.
            switch_after: Consecutive turns required in the new language before
                switching. 1 switches immediately.
            **kwargs: Passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._current = _SARVAM_LANGUAGE.get(initial, initial)
        self._allowed = {
            _SARVAM_LANGUAGE.get(lang, lang) for lang in allowed
        }
        self._switch_after = max(1, switch_after)
        self._pending: Language | None = None
        self._pending_count = 0
        self._log = logger.bind(component="language")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame through, switching TTS language when warranted."""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.language is not None:
            await self._maybe_switch(frame.language)

        await self.push_frame(frame, direction)

    async def _maybe_switch(self, detected: Language) -> None:
        target = _SARVAM_LANGUAGE.get(detected)
        if target is None or target not in self._allowed or target == self._current:
            # Same language, unconfigured language, or one Sarvam has no
            # mapping for — reset the streak so alternating turns don't
            # accumulate toward a switch.
            self._pending = None
            self._pending_count = 0
            return

        if target != self._pending:
            self._pending = target
            self._pending_count = 1
        else:
            self._pending_count += 1

        if self._pending_count < self._switch_after:
            return

        previous, self._current = self._current, target
        self._pending = None
        self._pending_count = 0

        self._log.info(
            "switching TTS language",
            event="tts_language_switch",
            from_language=str(previous),
            to_language=str(target),
        )
        # Downstream: the TTS service sits after this processor in the pipeline.
        # `delta` is the current API — the `settings` dict form is deprecated.
        # `language` lives on the base TTSSettings, so this stays provider
        # agnostic rather than importing Sarvam's subclass.
        await self.push_frame(
            TTSUpdateSettingsFrame(delta=TTSSettings(language=target)),
            FrameDirection.DOWNSTREAM,
        )

    @property
    def current_language(self) -> Language:
        """Language the TTS is currently set to."""
        return self._current


def parse_languages(raw: str) -> list[Language]:
    """Parse a comma-separated language list into ``Language`` values.

    Unknown codes are dropped with a warning rather than raising: a typo in an
    env var should not take an inbound line down.
    """
    languages: list[Language] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        try:
            languages.append(Language(token))
        except ValueError:
            logger.bind(component="language").warning(
                "ignoring unknown language code", code=token
            )
    return languages or [Language.EN]
