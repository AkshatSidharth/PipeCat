"""Bot-side backchannels: short acknowledgements while the caller talks on.

When a caller has been speaking for several seconds with nothing coming back,
the line feels dead and people start asking "hello? are you there?". A human
agent fills that with a quiet *hmm* or *ji ji* — it says "still here, still
listening" without taking the floor.

Why this does not go through the TTS service
--------------------------------------------
The obvious implementation is ``TTSSpeakFrame("hmm")``. It is wrong here, for
three separate reasons found in the Pipecat 1.7.0 source:

1. **It would delay the real answer.** ``BaseOutputTransport`` has exactly one
   ``_audio_queue`` per destination and drains it serially at wall-clock pace
   (``transports/base_output.py:820-865``). Any audio frame queued ahead of the
   response pushes the response back by its own duration. Frame injection is
   never parallel.
2. **It would look like the bot taking a turn.** ``TTSAudioRawFrame`` sets
   ``_tts_audio_received`` and triggers ``BotStartedSpeakingFrame``
   (``base_output.py:778-787``), which drives turn tracking, the idle
   controller, recording segmentation, and the barge-in word gate. Spurious
   bot-speaking events corrupt all of them.
3. It would cost synthesis latency on every occurrence.

The audio **mixer** is the one genuinely parallel path in the framework. With a
mixer installed the transport calls ``mixer.mix()`` on every outgoing chunk
*and*, when the queue is empty, synthesizes a frame from the mixer alone::

    frame = OutputAudioRawFrame(audio=await mixer.mix(silence), ...)

That frame is a plain ``OutputAudioRawFrame``, so it plays without any
bot-speaking bookkeeping; it never enters ``_audio_queue``, so it cannot delay
the response by even one chunk; and it is still pushed downstream after being
written, so :class:`AudioBufferProcessor` records it. This satisfies "must not
block actual TTS generation in any way" literally rather than probabilistically.

Pacing is safe because both transports in use block per chunk: SmallWebRTC
awaits a completion future resolved by its paced ``recv()`` loop, and the
FastAPI websocket sleeps a send interval.

On "emotion"
------------
The tone is picked from **speech energy relative to this caller's own running
baseline** — loud and sustained gets a firmer "ji ji", quiet and halting gets a
soft "hmm". That is prosody, not emotion recognition: three seconds of
streaming audio without a dedicated speech-emotion model does not support a
claim about how someone *feels*, and this module does not make one. It is
enough to keep the acknowledgement from sounding tone-deaf, which is the part
that matters on a call.

Echo
----
The filler plays while the caller's microphone is open, so it can come back
through their mic and reach the STT. Browsers apply echo cancellation and PSTN
carriers apply their own, but neither is guaranteed. What actually contains
this is :mod:`voicebot.backchannel`: an echoed "hmm" is itself a backchannel, so
the turn-start strategy ignores it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import hashlib
import json
import math
import re
import wave
from pathlib import Path
import httpx
import numpy as np
from loguru import logger
from pipecat.audio.mixers.base_audio_mixer import BaseAudioMixer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    MixerControlFrame,
    MixerEnableFrame,
    MixerUpdateSettingsFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

CACHE_DIR = Path(__file__).resolve().parent.parent / "assets" / "backchannels"

# Tone -> phrases, per language.
#
# "ह्म्म" is the correct Devanagari for *hmm* — ह् + म्म, no vowel. The obvious
# "हम्म" is not: it has a real vowel and is pronounced "hum", which is audible
# and wrong. (Round-tripping "हम्म्म" through Soniox confirms it: the transcript
# comes back "हम".)
#
# Every phrase is two words because single tokens frequently render as silence.
FILLER_PHRASES: dict[str, dict[str, list[str]]] = {
    "hi": {
        "neutral": ["ह्म्म ह्म्म", "जी जी"],
        "emphatic": ["जी जी", "हाँ जी"],
        "hesitant": ["ह्म्म ह्म्म", "जी हाँ"],
    },
    "en": {
        "neutral": ["hmm hmm", "mhm mhm"],
        "emphatic": ["yes yes", "right right"],
        "hesitant": ["hmm hmm", "I see"],
    },
}

# Engines fail on a phrase by returning *silence* rather than an error, so a
# correct phrase can vanish from the pool with no symptom.
#
# For "ह्म्म ह्म्म" on OmniVoice the failure is INTERMITTENT, not absolute:
# measured 2 successes in 5 attempts, same text, same voice. That is why
# render_filler retries the wanted phrase several times before reaching for a
# substitute — the goal is the configured spelling, and a clip is cached the
# first time it works, so a retry costs nothing after that.
#
# These are the substitutes for a phrase that will not render at all. Sarvam
# says "ह्म्म ह्म्म" cleanly; for OmniVoice the fallback is "हूँ हूँ", the only
# Devanagari spelling it renders reliably AND that Soniox transcribes back as
# "Hmm."
FILLER_FALLBACKS: dict[str, list[str]] = {
    "ह्म्म ह्म्म": ["हूँ हूँ", "जी जी"],
    "hmm hmm": ["mhm mhm", "yes yes"],
    "mhm mhm": ["hmm hmm"],
    "जी जी": ["जी हाँ"],
}

TONES = ("neutral", "emphatic", "hesitant")


def phrases_for(language: str, settings=None) -> dict[str, list[str]]:
    """Filler phrases for a language, with per-agent overrides applied.

    A tone configured with no phrases falls back to the built-in set for that
    tone, so clearing one box does not silently mute that tone.
    """
    built_in = FILLER_PHRASES.get(language.split("-")[0].lower(), FILLER_PHRASES["en"])
    if settings is None:
        return built_in

    resolved: dict[str, list[str]] = {}
    for tone in TONES:
        raw = getattr(settings, f"filler_phrases_{tone}", "") or ""
        custom = [line.strip() for line in raw.splitlines() if line.strip()]
        resolved[tone] = custom or list(built_in.get(tone, []))
    return resolved


# --------------------------------------------------------------------------- #
# Clip rendering
# --------------------------------------------------------------------------- #


def voice_key(settings) -> str:
    """Cache identity for a rendered clip: everything that changes how it SOUNDS.

    Cached audio is only reusable if it would be re-rendered identically.
    Provider and voice alone are not enough — Sarvam's ``bulbul:v2`` and
    ``bulbul:v3`` share speaker names but not their delivery, ElevenLabs'
    ``eleven_flash_v2_5`` and ``eleven_v3`` share voice ids, and OmniVoice's
    ``speed`` changes the reading. Keying on the narrower tuple would serve a
    clip rendered by one configuration to another, which defeats the point of
    caching in the bot's own voice.

    Sample rate is deliberately absent: clips are stored at whatever rate the
    engine produced and resampled on load, so one file serves every transport.
    """
    provider = settings.tts_provider
    language = settings.sarvam_language
    if provider == "sarvam":
        parts = (provider, settings.tts_model, settings.sarvam_speaker, language)
    elif provider == "elevenlabs":
        parts = (provider, settings.elevenlabs_model, settings.elevenlabs_voice_id, language)
    else:
        parts = (
            provider,
            settings.omnivoice_voice_id,
            language,
            f"speed{settings.omnivoice_speed}",
        )
    # Custom wording is part of the identity too: a clip rendered for one set
    # of phrases must not be served to an agent that configured different ones.
    custom = "|".join(
        (getattr(settings, f"filler_phrases_{tone}", "") or "").strip()
        for tone in TONES
    )
    if custom.strip("|"):
        parts = parts + (hashlib.sha1(custom.encode("utf-8")).hexdigest()[:8],)
    readable = "_".join(str(p) for p in parts if p)
    # Keep it a legal, bounded directory name while staying recognisable.
    return re.sub(r"[^A-Za-z0-9._-]", "-", readable)[:120]


def _slug(text: str) -> str:
    """Filesystem-safe, collision-free-enough name for a phrase."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(pcm) == 0:
        return pcm
    count = int(len(pcm) * dst_rate / src_rate)
    return np.interp(
        np.linspace(0, len(pcm) - 1, count), np.arange(len(pcm)), pcm
    ).astype(np.int16)


def _render_sarvam(settings, text: str) -> tuple[np.ndarray, int]:
    response = httpx.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": settings.sarvam_api_key},
        json={
            "text": text,
            "target_language_code": settings.sarvam_language,
            "speaker": settings.sarvam_speaker,
            "model": settings.tts_model,
        },
        timeout=30,
    )
    response.raise_for_status()
    raw = base64.b64decode(response.json()["audios"][0])
    with wave.open(io.BytesIO(raw)) as handle:
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        return pcm, handle.getframerate()


def _render_elevenlabs(settings, text: str) -> tuple[np.ndarray, int]:
    response = httpx.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}",
        headers={"xi-api-key": settings.elevenlabs_api_key},
        params={"output_format": "pcm_24000"},
        json={"text": text, "model_id": settings.elevenlabs_model},
        timeout=30,
    )
    response.raise_for_status()
    return np.frombuffer(response.content, dtype=np.int16), 24000


async def _render_omnivoice(settings, text: str) -> tuple[np.ndarray, int]:
    import websockets

    from voicebot.omnivoice import language_to_omnivoice, split_frame

    url = f"{settings.omnivoice_url.rstrip('/')}/ws/filler-{_slug(text)}"
    async with websockets.connect(
        url, max_size=100 * 1024 * 1024, open_timeout=15, ping_interval=None
    ) as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "synthesize",
                    "call_id": "filler",
                    "text_id": "f1",
                    "text": text,
                    "streaming": True,
                    "voice_id": settings.omnivoice_voice_id,
                    "language": language_to_omnivoice(settings.sarvam_language),
                    "speed": settings.omnivoice_speed,
                }
            )
        )
        pcm = bytearray()
        while True:
            raw = await asyncio.wait_for(socket.recv(), timeout=60)
            if isinstance(raw, (bytes, bytearray)) and (not raw or raw[0] != 0x7B):
                pcm += raw  # bare PCM continuation frame
                continue
            header, audio = split_frame(raw)
            pcm += audio
            if header.get("type") in ("audio_done", "error"):
                break
        return np.frombuffer(bytes(pcm), dtype=np.int16), 24000


async def render_clip(settings, text: str) -> tuple[np.ndarray, int]:
    """Synthesize one filler phrase with the agent's own configured voice.

    Voice consistency is the point: a filler in a different voice from the bot
    is more jarring than no filler at all.
    """
    if settings.tts_provider == "omnivoice":
        return await _render_omnivoice(settings, text)
    if settings.tts_provider == "elevenlabs":
        return await asyncio.to_thread(_render_elevenlabs, settings, text)
    return await asyncio.to_thread(_render_sarvam, settings, text)


async def render_filler(settings, text: str, log, attempts: int = 3):
    """Render one filler phrase, working hard to get the *configured* wording.

    Order matters, and follows what the failures actually are:

    1. Retry the phrase itself. Engines drop short Devanagari intermittently
       (OmniVoice: 2 successes in 5 identical attempts), so a single empty
       result says nothing about whether the phrase is sayable.
    2. Try it doubled — some engines need more to work with.
    3. Only then substitute, and say so loudly.

    Returns:
        ``(pcm, sample_rate)``; empty pcm if nothing worked.
    """
    for attempt in range(attempts):
        pcm, rate = await render_clip(settings, text)
        if len(pcm) > 0:
            return pcm, rate
        log.debug(f"empty audio for {text!r} (attempt {attempt + 1}/{attempts})")

    pcm, rate = await render_clip(settings, f"{text} {text}")
    if len(pcm) > 0:
        return pcm, rate

    for alternative in FILLER_FALLBACKS.get(text, []):
        pcm, rate = await render_clip(settings, alternative)
        if len(pcm) > 0:
            # Warning, not debug: the bot is about to say something other than
            # what is configured. Silent degradation is exactly how the earlier
            # empty clips went unnoticed.
            log.warning(
                f"{settings.tts_provider} would not say {text!r} after "
                f"{attempts} attempts; using {alternative!r}",
                event="filler_substituted",
                wanted=text,
                used=alternative,
                provider=settings.tts_provider,
            )
            return pcm, rate

    return pcm, rate


async def load_clips(settings, sample_rate: int) -> dict[str, list[np.ndarray]]:
    """Build the tone -> clips map, synthesizing only what is not cached.

    Clips are cached on disk keyed by provider, voice, language and phrase, so
    only the first call for a given voice pays synthesis cost. A phrase that
    fails to render is skipped rather than raising: fillers are a nicety and
    must never take a call down.

    Args:
        settings: Resolved application settings.
        sample_rate: Transport output rate; clips are resampled to it.

    Returns:
        Mapping of tone to ready-to-mix int16 clips.
    """
    log = logger.bind(component="filler")
    folder = CACHE_DIR / voice_key(settings)
    folder.mkdir(parents=True, exist_ok=True)

    gain = max(0.0, min(1.0, settings.filler_gain))
    clips: dict[str, list[np.ndarray]] = {tone: [] for tone in TONES}

    for tone, texts in phrases_for(settings.sarvam_language, settings).items():
        for text in texts:
            path = folder / f"{_slug(text)}.wav"
            try:
                if path.exists():
                    with wave.open(str(path)) as handle:
                        pcm = np.frombuffer(
                            handle.readframes(handle.getnframes()), dtype=np.int16
                        )
                        rate = handle.getframerate()
                else:
                    pcm, rate = await render_filler(settings, text, log)
                    if len(pcm) == 0:
                        raise ValueError("empty audio")
                    with wave.open(str(path), "wb") as handle:
                        handle.setnchannels(1)
                        handle.setsampwidth(2)
                        handle.setframerate(rate)
                        handle.writeframes(pcm.tobytes())
                    log.info(
                        "rendered filler clip",
                        event="filler_clip_rendered",
                        text=text,
                        tone=tone,
                        secs=round(len(pcm) / rate, 2),
                    )
            except Exception as exc:
                log.warning(f"could not render filler {text!r}: {exc}")
                continue

            resampled = _resample(pcm, rate, sample_rate)
            clips[tone].append((resampled.astype(np.float32) * gain).astype(np.int16))

    total = sum(len(v) for v in clips.values())
    log.info(
        "filler clips ready",
        event="filler_clips_ready",
        clips=total,
        sample_rate=sample_rate,
        cache=str(folder),
    )
    return clips


# --------------------------------------------------------------------------- #
# Mixer
# --------------------------------------------------------------------------- #


class FillerMixer(BaseAudioMixer):
    """Plays a filler clip once, mixed into whatever the transport is sending.

    Created before the agent's settings are known (the transport is built
    first), so it starts empty and plays nothing until :meth:`set_clips` is
    called. Everything here runs on the transport's audio path, so ``mix`` is
    written to be allocation-light and to never raise — a failure here would
    corrupt every outgoing chunk, not just the filler.
    """

    def __init__(self) -> None:
        """Initialize an empty mixer."""
        self._sample_rate = 0
        self._clips: dict[str, list[np.ndarray]] = {}
        self._playing: np.ndarray | None = None
        self._position = 0
        self._enabled = True
        self._played = 0
        self._log = logger.bind(component="filler")

    # -- lifecycle ---------------------------------------------------------

    async def start(self, sample_rate: int) -> None:
        """Record the transport's output rate."""
        self._sample_rate = sample_rate

    async def stop(self) -> None:
        """Stop any clip in flight."""
        self._playing = None
        self._position = 0

    async def process_frame(self, frame: MixerControlFrame) -> None:
        """Handle mixer control frames.

        Supported so the mixer can also be driven the framework way; the
        prompter calls :meth:`play` directly, which is one less hop.
        """
        if isinstance(frame, MixerEnableFrame):
            self._enabled = frame.enable
        elif isinstance(frame, MixerUpdateSettingsFrame):
            tone = frame.settings.get("tone")
            if isinstance(tone, str):
                self.play(tone)

    # -- control -----------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Transport output rate, or 0 before start."""
        return self._sample_rate

    @property
    def played(self) -> int:
        """Fillers played on this call."""
        return self._played

    @property
    def ready(self) -> bool:
        """True once clips are loaded."""
        return any(self._clips.values())

    @property
    def busy(self) -> bool:
        """True while a clip is still playing."""
        return self._playing is not None

    def set_clips(self, clips: dict[str, list[np.ndarray]]) -> None:
        """Install the rendered clips. Safe to call while a call is running."""
        self._clips = clips

    def play(self, tone: str) -> bool:
        """Start a clip for ``tone``. No-op if one is already playing.

        Returns:
            Whether a clip was started.
        """
        if not self._enabled or self._playing is not None:
            return False
        pool = self._clips.get(tone) or self._clips.get("neutral") or []
        if not pool:
            return False
        # Rotate rather than random: the same filler twice in a row is the one
        # thing that makes this sound synthetic, and rotation guarantees it
        # cannot happen with a pool of 2+.
        clip = pool[self._played % len(pool)]
        self._playing = clip
        self._position = 0
        self._played += 1
        return True

    # -- audio path --------------------------------------------------------

    async def mix(self, audio: bytes) -> bytes:
        """Add the playing clip onto ``audio``.

        Called for every outgoing chunk, including the silence the transport
        synthesizes while the queue is empty — which is exactly when a filler
        plays.
        """
        if self._playing is None:
            return audio
        try:
            chunk = np.frombuffer(audio, dtype=np.int16)
            remaining = self._playing[self._position : self._position + len(chunk)]
            if len(remaining) == 0:
                self._playing = None
                self._position = 0
                return audio
            self._position += len(remaining)
            if self._position >= len(self._playing):
                self._playing = None
                self._position = 0
            # int32 headroom, then clip: the filler is quiet and the bot is
            # usually silent underneath it, but a barge-in can overlap them.
            mixed = chunk.astype(np.int32)
            mixed[: len(remaining)] += remaining.astype(np.int32)
            return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()
        except Exception as exc:  # never break the audio path
            self._log.warning(f"filler mix failed, passing audio through: {exc}")
            self._playing = None
            self._position = 0
            return audio


# --------------------------------------------------------------------------- #
# Trigger
# --------------------------------------------------------------------------- #


class FillerPrompter(FrameProcessor):
    """Fires a filler once the caller has been talking for a while.

    Sits right after ``transport.input()`` so it sees the caller's audio and
    VAD frames first. It holds the mixer directly rather than pushing a
    ``MixerUpdateSettingsFrame`` the length of the pipeline: the frame would
    have to survive the STT, LLM and TTS services to reach the transport, and
    there is nothing to gain from the extra hops.
    """

    def __init__(
        self,
        *,
        mixer: FillerMixer,
        after_secs: float = 3.0,
        interval_secs: float = 4.5,
        max_per_turn: int = 3,
        **kwargs,
    ):
        """Initialize the prompter.

        Args:
            mixer: Mixer installed on the output transport.
            after_secs: Continuous speech before the first filler.
            interval_secs: Minimum gap between fillers within one utterance.
            max_per_turn: Cap per utterance, so a long monologue does not turn
                into a stream of hmms.
            **kwargs: Passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._mixer = mixer
        self._after = after_secs
        self._interval = interval_secs
        self._max_per_turn = max_per_turn

        self._speaking = False
        self._last_filler_secs = 0.0
        self._this_turn = 0
        self._bot_speaking = False
        self._rate = 0

        # Speech length is measured from the audio itself, not the wall clock:
        # the sample count is exactly how much the caller has said, and it stays
        # right when the event loop stalls under load.
        #
        # Energy, tracked two ways: within the current utterance, and as a
        # running baseline for this caller so "loud" means loud *for them*.
        self._sum_squares = 0.0
        self._samples = 0
        self._baseline: float | None = None

        self._log = logger.bind(component="filler")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Pass every frame through, timing the caller's speech."""
        await super().process_frame(frame, direction)

        if isinstance(frame, (StartFrame, EndFrame)):
            self._reset_utterance()
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._reset_utterance()
            self._speaking = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._settle_baseline()
            self._reset_utterance()
        elif isinstance(frame, InputAudioRawFrame):
            self._accumulate(frame)
            await self._maybe_play()

        await self.push_frame(frame, direction)

    # -- energy ------------------------------------------------------------

    def _accumulate(self, frame: InputAudioRawFrame) -> None:
        if not self._speaking or not frame.audio:
            return
        samples = np.frombuffer(frame.audio, dtype=np.int16)
        if samples.size:
            self._sum_squares += float(np.dot(samples.astype(np.float64), samples))
            self._samples += samples.size
            self._rate = frame.sample_rate or self._rate

    def _speaking_secs(self) -> float:
        """Seconds of audio received since the caller started this utterance."""
        return self._samples / self._rate if self._rate else 0.0

    def _rms(self) -> float:
        return math.sqrt(self._sum_squares / self._samples) if self._samples else 0.0

    def _settle_baseline(self) -> None:
        """Fold this utterance's loudness into the caller's running baseline."""
        rms = self._rms()
        if rms <= 0:
            return
        self._baseline = rms if self._baseline is None else 0.7 * self._baseline + 0.3 * rms

    def _reset_utterance(self) -> None:
        self._speaking = False
        self._this_turn = 0
        self._last_filler_secs = 0.0
        self._sum_squares = 0.0
        self._samples = 0

    def _tone(self) -> str:
        """Coarse prosodic bucket: loud, quiet, or neither, for this caller."""
        rms = self._rms()
        if not self._baseline or rms <= 0:
            return "neutral"
        ratio = rms / self._baseline
        if ratio >= 1.25:
            return "emphatic"
        if ratio <= 0.8:
            return "hesitant"
        return "neutral"

    # -- trigger -----------------------------------------------------------

    async def _maybe_play(self) -> None:
        if (
            not self._speaking
            or self._bot_speaking          # never talk over our own answer
            or self._mixer.busy
            or not self._mixer.ready
            or self._this_turn >= self._max_per_turn
        ):
            return

        elapsed = self._speaking_secs()
        if elapsed < self._after:
            return
        if self._this_turn and elapsed - self._last_filler_secs < self._interval:
            return

        tone = self._tone()
        if not self._mixer.play(tone):
            return

        self._last_filler_secs = elapsed
        self._this_turn += 1
        self._log.info(
            "played filler backchannel",
            event="filler_played",
            tone=tone,
            speaking_secs=round(elapsed, 2),
            index=self._this_turn,
        )


def build_filler(settings, mixer: FillerMixer) -> FillerPrompter:
    """Build the prompter from settings."""
    return FillerPrompter(
        mixer=mixer,
        after_secs=settings.filler_after_secs,
        interval_secs=settings.filler_interval_secs,
        max_per_turn=settings.filler_max_per_turn,
    )


async def warm_filler_clips(settings, mixer: FillerMixer) -> None:
    """Load clips into ``mixer`` once the transport rate is known.

    Runs as a background task so first-call synthesis never delays call setup;
    until it finishes, :meth:`FillerMixer.play` is a no-op.
    """
    try:
        rate = mixer.sample_rate or settings.audio_out_sample_rate
        mixer.set_clips(await load_clips(settings, rate))
    except Exception as exc:
        logger.bind(component="filler").warning(f"filler clips unavailable: {exc}")


__all__ = [
    "FILLER_PHRASES",
    "FillerMixer",
    "FillerPrompter",
    "build_filler",
    "load_clips",
    "phrases_for",
    "render_clip",
    "warm_filler_clips",
]
