"""OmniVoice (FlowTTS) streaming TTS service for Pipecat.

Protocol, as reverse-engineered from ``FlowTTS/sample_files/test_tts_ws.ipynb``
and verified against the live server:

**Connect** — one websocket per call, the call id in the path::

    ws://<host><prefix>/ws/<call_id>

**Request** — a JSON text frame::

    {"type": "synthesize", "call_id": ..., "text_id": ..., "text": ...,
     "streaming": true, "voice_id": ..., "language": ..., "speed": 1.0}

**Response** — binary frames of ``{json header}`` immediately followed by raw
PCM. The header has no length prefix, so the boundary is found by counting
brace depth. Occasionally the audio arrives as the *next* frame instead of
trailing the header, which is handled.

    ``audio_chunk``  chunk_index, sample_rate, encoding, is_final, cache_hit,
                     decoder_ttft_ms, llm_ttft_ms, rtf, text_id
    ``audio_done``   chunks, total_wav_bytes, avg_rtf, text_id
    ``error``        error

Audio is mono ``pcm_int16`` at 24kHz.

Two design choices that matter for latency
------------------------------------------
**The socket is held open for the whole call.** The notebook opens a fresh
connection per utterance; the first request on a cold connection measured
3.2s versus ~450ms warm. Since the call id is in the URL and a call maps 1:1
to a connection, reusing it is both natural and the single biggest win.

**Barge-in filters on ``text_id`` rather than reconnecting.** When Pipecat
cancels a turn mid-synthesis, the server keeps sending chunks for the old
request. Those would otherwise be read as the *next* turn's audio. Every frame
carries the ``text_id`` it belongs to, so stale audio is simply dropped — which
avoids paying reconnect latency on every interruption.

Measured TTFB (warm socket, this deployment)
--------------------------------------------
=========================  ======  =======
text                       chars    TTFB
=========================  ======  =======
"Sure."                         5   447ms
"One moment please."           18   449ms
"Your order will arrive…"      43   454ms
(149-char sentence)           149  1043ms
=========================  ======  =======

There is a **~450ms floor**, almost all of it the model's own
``decoder_ttft_ms`` (~420ms), and it is flat with text length until the text
gets long. Shortening the first sentence therefore does *not* buy anything
below that floor — see the README's latency section.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator

import websockets
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

OMNIVOICE_SAMPLE_RATE = 24000


def split_frame(raw: bytes | str) -> tuple[dict[str, Any], bytes]:
    """Split a combined ``{json header}`` + raw-PCM frame.

    The header carries no length prefix, so the end of the JSON object is found
    by counting brace depth — the same approach the reference notebook uses.

    Args:
        raw: A text or binary websocket frame.

    Returns:
        ``(header, pcm_bytes)``; ``pcm_bytes`` is empty when the frame is
        header-only.
    """
    if isinstance(raw, str):
        return json.loads(raw), b""
    depth = end = 0
    for i, byte in enumerate(raw):
        if byte == 0x7B:  # {
            depth += 1
        elif byte == 0x7D:  # }
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if not end:
        raise ValueError("no JSON header found in binary frame")
    return json.loads(raw[:end]), bytes(raw[end:])


def language_to_omnivoice(language: Language | str | None) -> str:
    """Map a Pipecat language to OmniVoice's bare code.

    OmniVoice takes base codes only (``hi``, not ``hi-IN``), so regional
    variants are truncated. Unknown values pass through unchanged.
    """
    if language is None:
        return "en"
    value = language.value if isinstance(language, Language) else str(language)
    return value.split("-")[0].lower()


class OmniVoiceTTSService(TTSService):
    """Streaming TTS over the OmniVoice websocket protocol."""

    # Voice and language live in settings so runtime updates work — the
    # bilingual LanguageFollower pushes TTSUpdateSettingsFrame(language=...),
    # and reading it per request is what makes that take effect.
    Settings = TTSSettings

    def __init__(
        self,
        *,
        url: str,
        voice_id: str,
        language: str = "en",
        speed: float = 1.0,
        call_id: str | None = None,
        connect_timeout: float = 10.0,
        first_chunk_timeout: float = 8.0,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            url: Base websocket URL, e.g. ``ws://172.16.1.4:80/omnivoice-tts``.
                ``/ws/<call_id>`` is appended.
            voice_id: Server-side voice, e.g. ``hausa-female-1``.
            language: Language code matching the text (``en``, ``hi``, ``ha``…).
            speed: Speaking rate multiplier.
            call_id: Correlation id; one is generated when omitted.
            connect_timeout: Websocket open timeout in seconds.
            first_chunk_timeout: How long to wait for the first audio chunk
                before giving up on the utterance. Deliberately short. A warm
                synthesis answers in ~300ms, so anything past a few seconds
                means the server is wedged, and on a phone call a long wait is
                strictly worse than dropping one sentence: the turn lock is
                held for the whole wait, so every following sentence queues
                behind it and the bot goes silent for the duration. Observed at
                the old 30s value: one wedged server turned into 30s+ of dead
                air and the caller hung up.
            **kwargs: Passed to :class:`TTSService`.
        """
        settings = kwargs.pop("settings", None) or TTSSettings(
            # OmniVoice has no separate model parameter — the voice selects it.
            model=voice_id,
            voice=voice_id,
            language=Language(language),
        )
        super().__init__(
            # The base class opens the audio context and emits TTSStartedFrame,
            # so run_tts only has to yield audio.
            push_start_frame=True,
            push_stop_frames=False,
            sample_rate=kwargs.pop("sample_rate", None) or OMNIVOICE_SAMPLE_RATE,
            settings=settings,
            **kwargs,
        )
        self._base_url = url.rstrip("/")
        self._speed = speed
        self._call_id = call_id or f"pc-{uuid.uuid4().hex[:8]}"
        self._connect_timeout = connect_timeout
        self._first_chunk_timeout = first_chunk_timeout

        self._websocket: Any = None
        self._lock = asyncio.Lock()
        # Separate from _lock (which guards connect): one socket carries every
        # utterance for the call, and two overlapping run_tts calls would race
        # on recv() — each consuming frames belonging to the other, then
        # blocking until the read timeout. Serialising the request/response
        # cycle is what makes a single shared socket safe.
        self._turn_lock = asyncio.Lock()
        self._log = logger.bind(component="omnivoice")

    # -- connection ---------------------------------------------------------

    @property
    def _url(self) -> str:
        return f"{self._base_url}/ws/{self._call_id}"

    async def _ensure_connected(self) -> Any:
        """Open the websocket if needed and return it. Safe to call per turn."""
        async with self._lock:
            if self._websocket is not None and self._websocket.state.name == "OPEN":
                return self._websocket
            self._log.debug("connecting to OmniVoice", url=self._url)
            self._websocket = await websockets.connect(
                self._url,
                max_size=100 * 1024 * 1024,  # a whole utterance can arrive in one frame
                open_timeout=self._connect_timeout,
                close_timeout=3,
                ping_interval=None,  # the server does not expect client pings
            )
            return self._websocket

    async def start(self, frame: StartFrame):
        """Connect up front so the first turn does not pay setup cost."""
        await super().start(frame)
        try:
            await self._ensure_connected()
        except Exception as exc:  # non-fatal: run_tts retries
            self._log.warning(f"OmniVoice pre-connect failed: {exc}")

    async def stop(self, frame):
        """Close the websocket."""
        await super().stop(frame)
        await self._close()

    async def cancel(self, frame):
        """Close the websocket on cancellation."""
        await super().cancel(frame)
        await self._close()

    async def _close(self) -> None:
        async with self._lock:
            if self._websocket is not None:
                try:
                    await self._websocket.close()
                except Exception:
                    pass
                self._websocket = None

    # -- synthesis ----------------------------------------------------------

    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame | None, None]:
        """Synthesize ``text`` and yield audio as the server produces it.

        Args:
            text: Text to speak.
            context_id: Pipecat audio-context id for this utterance.

        Yields:
            ``TTSAudioRawFrame`` per chunk, then ``TTSStoppedFrame``.
        """
        text_id = uuid.uuid4().hex[:8]
        try:
            ws = await self._ensure_connected()
        except Exception as exc:
            yield ErrorFrame(error=f"OmniVoice connect failed: {exc}")
            return

        # Read voice/language from settings each turn so a runtime
        # TTSUpdateSettingsFrame (e.g. the bilingual language follower) applies.
        request = {
            "type": "synthesize",
            "call_id": self._call_id,
            "text_id": text_id,
            "text": text,
            "streaming": True,
            "voice_id": self._settings.voice,
            "language": language_to_omnivoice(self._settings.language),
            "speed": self._speed,
        }

        # One utterance at a time on the shared socket. Acquired inside the
        # try below so that every exit path — error, timeout, or the
        # GeneratorExit an interruption raises — runs the finally and releases
        # it. Acquiring outside strands the lock on the error paths and every
        # later utterance blocks forever.
        first = True
        sample_rate = self.sample_rate
        try:
            await self._turn_lock.acquire()
            await self.start_ttfb_metrics()
            started = time.perf_counter()
            try:
                await ws.send(json.dumps(request))
                await self.start_tts_usage_metrics(text)
            except Exception as exc:
                await self._close()
                yield ErrorFrame(error=f"OmniVoice send failed: {exc}")
                return
            while True:
                # Later chunks get a longer budget than the first: the server
                # has already proven it is alive and is mid-utterance.
                timeout = self._first_chunk_timeout if first else 15.0
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                header, audio = split_frame(raw)

                # Audio from a turn the user interrupted is still in flight.
                # Drop it rather than reconnecting — every frame is tagged.
                if header.get("text_id") not in (None, text_id):
                    continue

                kind = header.get("type")
                if kind == "audio_chunk":
                    if not audio:
                        # Some builds send the PCM as the following frame.
                        nxt = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        audio = nxt.encode() if isinstance(nxt, str) else nxt
                    sample_rate = header.get("sample_rate", sample_rate)
                    if first:
                        first = False
                        await self.stop_ttfb_metrics()
                        self._log.debug(
                            "omnivoice first chunk",
                            event="omnivoice_ttfb",
                            ttfb_ms=round((time.perf_counter() - started) * 1000),
                            # The server reports its own breakdown; decoder_ttft
                            # is the floor no client-side change can move.
                            decoder_ttft_ms=header.get("decoder_ttft_ms"),
                            llm_ttft_ms=header.get("llm_ttft_ms"),
                            cache_hit=header.get("cache_hit"),
                        )
                    if audio:
                        yield TTSAudioRawFrame(audio, sample_rate, 1, context_id=context_id)

                elif kind == "audio_done":
                    self._log.debug(
                        "omnivoice done",
                        event="omnivoice_done",
                        chunks=header.get("chunks"),
                        rtf=header.get("avg_rtf") or header.get("rtf"),
                    )
                    break

                elif kind == "error":
                    yield ErrorFrame(error=f"OmniVoice error: {header.get('error')}")
                    break

        except asyncio.TimeoutError:
            yield ErrorFrame(error="OmniVoice timed out waiting for audio")
            await self._close()
        except websockets.exceptions.ConnectionClosed:
            # Reconnect on the next turn rather than failing this one loudly.
            await self._close()
            yield ErrorFrame(error="OmniVoice connection closed mid-synthesis")
        except Exception as exc:
            yield ErrorFrame(error=f"OmniVoice failure: {exc}")
        finally:
            await self.stop_ttfb_metrics()
            # Released here rather than via `async with`: this is an async
            # generator, and an interruption closes it mid-iteration — finally
            # still runs, so the lock cannot be stranded.
            if self._turn_lock.locked():
                self._turn_lock.release()

        yield TTSStoppedFrame(context_id=context_id)
