"""Call recording to WAV.

Wraps Pipecat's :class:`AudioBufferProcessor` and streams its periodic flushes
straight to disk, so memory stays flat regardless of call length (a naive
``buffer_size=0`` setup holds the entire call in RAM until it ends).

Files are laid out one directory per agent, named so they sort chronologically::

    recordings/
      pooja-kapture/
        20260815-014526-a1b2c3d4-mixed.wav   stereo: user left, bot right
        20260815-014526-a1b2c3d4-user.wav    mono, user only
        20260815-014526-a1b2c3d4-bot.wav     mono, bot only
      selfhosted-stack/
        ...

The timestamp is local wall-clock at the moment the call started, so `ls` in an
agent's directory is already a call log in order. The call id stays on the end
because two calls can begin inside the same second.

Stereo separation is what makes a recording useful after the fact: you can hear
exactly where the two talkers overlap, which is how you debug a barge-in that
fired too eagerly or not at all.
"""

from __future__ import annotations

import asyncio
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

from voicebot.obs.metrics import Metrics

_BYTES_PER_SAMPLE = 2  # 16-bit PCM


@dataclass
class _Track:
    """One WAV file being written incrementally."""

    path: Path
    channels: int
    sample_rate: int
    writer: wave.Wave_write | None = None
    bytes_written: int = 0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = wave.open(str(self.path), "wb")
        writer.setnchannels(self.channels)
        writer.setsampwidth(_BYTES_PER_SAMPLE)
        writer.setframerate(self.sample_rate)
        self.writer = writer

    def write(self, audio: bytes) -> None:
        if not audio:
            return
        if self.writer is None:
            self.open()
        assert self.writer is not None
        self.writer.writeframes(audio)
        self.bytes_written += len(audio)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    @property
    def duration_secs(self) -> float:
        divisor = self.sample_rate * _BYTES_PER_SAMPLE * self.channels
        return self.bytes_written / divisor if divisor else 0.0


@dataclass
class CallRecorder:
    """Streams the audio buffer processor's output to WAV files."""

    call_id: str
    output_dir: Path
    sample_rate: int
    metrics: Metrics
    flush_secs: float = 30.0
    agent: str = "default"
    # Called the instant recording starts, so the transcript can share the
    # audio's timeline rather than the pipeline's construction time.
    on_started: Any = None

    processor: AudioBufferProcessor = field(init=False)
    stem: str = field(init=False, default="")
    _tracks: dict[str, _Track] = field(init=False, default_factory=dict)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    _pending: int = field(init=False, default=0)
    _log: object = field(init=False)

    def __post_init__(self) -> None:
        self._log = logger.bind(component="recording")

        from voicebot.agents import slugify

        # One directory per agent, files prefixed with the call's start time so
        # a plain `ls` reads chronologically. slugify keeps a display name from
        # escaping the recordings directory.
        folder = self.output_dir / (slugify(self.agent) or "default")
        # Public: the post-call report reuses it, so the analysis file sits
        # beside its own audio. Generating a second timestamp at call end gave
        # the two a different stem and the UI could not pair them.
        self.stem = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.call_id}"
        stem = self.stem

        # buffer_size is measured in bytes of a single mono 16-bit track, so
        # convert the desired flush interval accordingly.
        buffer_size = int(self.sample_rate * _BYTES_PER_SAMPLE * self.flush_secs)

        self.processor = AudioBufferProcessor(
            sample_rate=self.sample_rate,
            num_channels=2,  # stereo: user left, bot right
            buffer_size=buffer_size,
            auto_start_recording=True,
        )

        self._tracks = {
            "mixed": _Track(
                path=folder / f"{stem}-mixed.wav",
                channels=2,
                sample_rate=self.sample_rate,
            ),
            "user": _Track(
                path=folder / f"{stem}-user.wav",
                channels=1,
                sample_rate=self.sample_rate,
            ),
            "bot": _Track(
                path=folder / f"{stem}-bot.wav",
                channels=1,
                sample_rate=self.sample_rate,
            ),
        }

        self._register_handlers()

    def _register_handlers(self) -> None:
        processor = self.processor

        @processor.event_handler("on_recording_started")
        async def _on_started(_buffer) -> None:
            if self.on_started is not None:
                self.on_started()
            self._log.info(
                "recording started",
                event="recording_started",
                call_id=self.call_id,
                output_dir=str(self.output_dir),
                sample_rate=self.sample_rate,
            )

        @processor.event_handler("on_audio_data")
        async def _on_audio_data(_buffer, audio: bytes, sample_rate: int, num_channels: int) -> None:
            # Merged stream. With num_channels=2 this is interleaved stereo.
            await self._append("mixed", audio)

        @processor.event_handler("on_track_audio_data")
        async def _on_track_audio_data(
            _buffer,
            user_audio: bytes,
            bot_audio: bytes,
            sample_rate: int,
            num_channels: int,
        ) -> None:
            # The individual tracks are mono regardless of the processor's
            # configured channel count — that count describes the merged mix.
            await self._append("user", user_audio)
            await self._append("bot", bot_audio)

        @processor.event_handler("on_recording_stopped")
        async def _on_stopped(_buffer) -> None:
            await self.close()

    async def _append(self, track_name: str, audio: bytes) -> None:
        if not audio:
            return
        track = self._tracks[track_name]
        # Incremented before the first await so close() can see the write is
        # in flight the moment this coroutine starts running.
        self._pending += 1
        try:
            async with self._lock:
                # Keep file I/O off the event loop so it never stalls audio.
                await asyncio.to_thread(track.write, audio)
        finally:
            self._pending -= 1

    async def close(self) -> None:
        """Flush anything still buffered, finalize the WAVs, record metrics.

        The explicit ``stop_recording`` matters. The buffer only emits audio
        when it fills (30s by default) or when the processor is stopped, and
        the processor is normally stopped by an ``EndFrame`` travelling the
        pipeline. If anything upstream is wedged — a TTS service sitting in a
        read timeout, say — that frame never arrives, and a whole call's audio
        is discarded with no error anywhere. Observed: three consecutive calls
        logged "recording started" and wrote nothing.
        """
        try:
            await self.processor.stop_recording()
        except Exception as exc:  # never let recording break call teardown
            self._log.warning(f"could not flush recording buffer: {exc}")

        # stop_recording dispatches the audio handlers as TASKS and returns
        # without awaiting them (BaseObject._call_event_handler uses
        # asyncio.create_task for async handlers). Closing the files here would
        # race those writes — which is exactly how calls ended up with empty
        # recordings. Let the tasks start, then wait for them to drain.
        await asyncio.sleep(0)
        for _ in range(500):  # bounded: 5s, then give up rather than hang
            if not self._pending:
                break
            await asyncio.sleep(0.01)
        if self._pending:
            self._log.warning(f"{self._pending} recording writes still pending")

        async with self._lock:
            for name, track in self._tracks.items():
                if track.writer is None and track.bytes_written == 0:
                    continue
                duration = track.duration_secs
                await asyncio.to_thread(track.close)
                self.metrics.recordings.labels(track=name).inc()
                self.metrics.recorded_seconds.labels(track=name).inc(duration)
                self._log.info(
                    "recording written",
                    event="recording_written",
                    call_id=self.call_id,
                    track=name,
                    path=str(track.path),
                    duration_secs=round(duration, 2),
                    bytes=track.bytes_written,
                )

    @property
    def paths(self) -> dict[str, str]:
        """Map of track name to output path."""
        return {name: str(track.path) for name, track in self._tracks.items()}
