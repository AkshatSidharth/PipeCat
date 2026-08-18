"""TTS providers, with a speakability gate.

Two providers are selectable per agent:

* **Sarvam** (``bulbul:*``) — Indic-first, websocket streaming.
* **ElevenLabs** (``eleven_flash_v2_5``) — 32 languages including Hindi.

The gate
--------
Sarvam rejects text containing no letter or digit with

    400: Text must contain at least one character from the allowed languages.

Pipecat's ``TTSService`` drops empty and whitespace-only text, but not
punctuation-only text — and sentence aggregation regularly produces exactly
that. An LLM streaming ``"…done"`` then ``"."`` leaves a lone ``"."`` as its own
chunk, which becomes a failed turn.

Verified against Sarvam's live API — these are measured, not assumed:

======================  ========
``"."`` ``"?"`` ``"…"``  rejected
``" "``                  rejected
``"😀"`` (emoji only)     rejected
``"123"`` ``"42."``      accepted
``"A"`` ``"Rs. 500"``    accepted
======================  ========

The gate is applied to both providers. It costs nothing on a provider that
would have accepted the chunk — those characters carry no audio either way, and
the surrounding sentences have already been spoken.
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.transcriptions.language import Language

from voicebot.omnivoice import OmniVoiceTTSService

# One unicode letter or digit, underscore excluded. Matches Devanagari and Latin
# alike; does not match punctuation, whitespace or emoji.
_SPEAKABLE = re.compile(r"[^\W_]", re.UNICODE)


def is_speakable(text: str) -> bool:
    """True when the text contains something a TTS engine can actually voice."""
    return bool(text) and _SPEAKABLE.search(text) is not None


class _SpeakabilityGate:
    """Mixin that drops TTS chunks with nothing voiceable in them."""

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        """Synthesize ``text``, skipping anything with no letter or digit."""
        if not is_speakable(text):
            # debug, not warning: expected and handled. A warning per
            # punctuation fragment would bury real problems.
            logger.debug(f"{self}: skipping unspeakable TTS chunk {text!r}")
            return
        async for frame in super().run_tts(text, context_id):  # type: ignore[misc]
            yield frame


class GatedSarvamTTSService(_SpeakabilityGate, SarvamTTSService):
    """Sarvam websocket TTS that silently drops chunks it would reject."""


class GatedElevenLabsTTSService(_SpeakabilityGate, ElevenLabsTTSService):
    """ElevenLabs websocket TTS with the same gate."""


class GatedOmniVoiceTTSService(_SpeakabilityGate, OmniVoiceTTSService):
    """OmniVoice websocket TTS with the same gate."""


def build_tts(settings, call_id: str | None = None):
    """Build the configured TTS service.

    Args:
        settings: Resolved application settings.
        call_id: Correlation id, used by OmniVoice as its per-call socket id.

    Returns:
        A ready-to-use TTS service for the selected provider.
    """
    log = logger.bind(component="tts")

    aggregation = (
        TextAggregationMode.TOKEN
        if settings.tts_text_aggregation == "token"
        else TextAggregationMode.SENTENCE
    )

    if settings.tts_provider == "omnivoice":
        # Sentence aggregation is FORCED here, whatever the agent asked for.
        # TOKEN mode works for Sarvam because Sarvam buffers server-side
        # (min_buffer_size) and only synthesizes once it has enough text.
        # OmniVoice has no such gate: every chunk Pipecat hands over becomes a
        # full synthesis request on the one per-call websocket, each costing
        # ~300ms. A single 130-token reply in token mode fired 72 requests,
        # 36 of which hit the 30s read timeout — the bot managed one turn and
        # then stopped producing audio. Measured on a real call.
        if aggregation is TextAggregationMode.TOKEN:
            log.warning(
                "ignoring token aggregation for OmniVoice; forcing sentence",
                event="tts_aggregation_override",
            )
            aggregation = TextAggregationMode.SENTENCE
        # Language code is the bare base ("en", "hi") — OmniVoice does not take
        # regional variants.
        lang = settings.sarvam_language.split("-")[0]
        log.info(
            "using OmniVoice TTS",
            event="tts_provider",
            url=settings.omnivoice_url,
            voice=settings.omnivoice_voice_id,
            language=lang,
        )
        return GatedOmniVoiceTTSService(
            url=settings.omnivoice_url,
            voice_id=settings.omnivoice_voice_id,
            language=lang,
            speed=settings.omnivoice_speed,
            call_id=call_id,
            text_aggregation_mode=aggregation,
        )

    if settings.tts_provider == "elevenlabs":
        # eleven_flash_v2_5 is ElevenLabs' lowest-latency model and covers Hindi.
        # Pipecat derives auto_mode from the aggregation mode, so it is not set
        # here: TOKEN forces auto_mode off, which re-enables server-side chunk
        # scheduling.
        log.info(
            "using ElevenLabs TTS",
            event="tts_provider",
            model=settings.elevenlabs_model,
            voice=settings.elevenlabs_voice_id,
        )
        return GatedElevenLabsTTSService(
            api_key=settings.elevenlabs_api_key,
            sample_rate=settings.tts_sample_rate,
            text_aggregation_mode=aggregation,
            settings=GatedElevenLabsTTSService.Settings(
                model=settings.elevenlabs_model,
                voice=settings.elevenlabs_voice_id,
                language=Language(settings.sarvam_language),
            ),
        )

    log.info(
        "using Sarvam TTS",
        event="tts_provider",
        model=settings.tts_model,
        voice=settings.sarvam_speaker,
    )
    return GatedSarvamTTSService(
        api_key=settings.sarvam_api_key,
        # Websocket streaming service (SarvamHttpTTSService is the slower,
        # non-streaming sibling). Leaving sample_rate unset uses the model's
        # native rate and lets the output transport resample.
        sample_rate=settings.tts_sample_rate,
        text_aggregation_mode=aggregation,
        settings=GatedSarvamTTSService.Settings(
            model=settings.tts_model,
            voice=settings.sarvam_speaker,
            language=Language(settings.sarvam_language),
            min_buffer_size=settings.sarvam_min_buffer_size,
        ),
    )
