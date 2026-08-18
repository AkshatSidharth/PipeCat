"""The frame observer that converts pipeline activity into metrics and logs.

A Pipecat observer sees every frame that moves between processors without
sitting in the pipeline itself, so it adds no latency to the audio path.

One subtlety drives the whole design: ``on_push_frame`` fires once per
processor *edge*, so a single frame travelling through N processors is
delivered N times. Every handler here is therefore gated on a bounded
frame-id dedup set — otherwise a single LLM response would be counted once per
hop downstream.
"""

from __future__ import annotations

import re
import time
from collections import deque
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsageMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from voicebot.obs.context import set_turn
from voicebot.obs.metrics import Metrics, token_kinds


# Splits a CamelCase processor name into tokens: "ElevenLabsTTSService#2" ->
# {"Eleven", "Labs", "TTS", "Service", "2"}.
_CAMEL_TOKEN = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[0-9]+")


def _service_type(processor: str) -> str:
    """Infer a coarse service type from a processor's class name.

    Token matching rather than substring matching, because uppercased
    substrings collide in practice: ``ElevenLabsTTSService`` upper-cases to
    ``...LABSTTSSERVICE``, which contains "STT" and would classify a TTS
    service as speech-to-text.
    """
    tokens = set(_CAMEL_TOKEN.findall(processor))
    if "STT" in tokens:
        return "stt"
    if "TTS" in tokens:
        return "tts"
    if "LLM" in tokens:
        return "llm"
    return "other"


class _Seen:
    """Bounded 'have I already handled this frame id' set."""

    def __init__(self, maxlen: int = 4096):
        self._set: set[int] = set()
        self._order: deque[int] = deque(maxlen=maxlen)

    def add(self, frame_id: int) -> bool:
        """Record a frame id. Returns True the first time it is seen."""
        if frame_id in self._set:
            return False
        if len(self._order) == self._order.maxlen:
            self._set.discard(self._order[0])
        self._order.append(frame_id)
        self._set.add(frame_id)
        return True


class TelemetryObserver(BaseObserver):
    """Records Prometheus metrics and structured logs for pipeline activity."""

    def __init__(
        self,
        *,
        metrics: Metrics,
        log_transcripts: bool = True,
        latency_budget_ms: int = 500,
        **kwargs,
    ):
        """Initialize the observer.

        Args:
            metrics: Collector container to record into.
            log_transcripts: Whether to include user transcripts and bot
                response text in the logs. Disable when transcripts are
                sensitive; latency and token metrics are unaffected.
            latency_budget_ms: Voice-to-voice target. Exceeding it increments a
                counter and logs at warning level; it does not alter behaviour.
            **kwargs: Passed through to :class:`BaseObserver`.
        """
        super().__init__(**kwargs)
        self._m = metrics
        self._log_transcripts = log_transcripts
        self._budget_secs = latency_budget_ms / 1000.0
        self._seen = _Seen()
        self._log = logger.bind(component="telemetry")

        # Latency budget tracking. Two marks, because the caller-perceived
        # number starts at raw VAD speech end, not at turn end:
        #   vad_stopped  -> bot_started : voice-to-voice (what the caller feels)
        #   vad_stopped  -> user_stopped: turn detection (VAD + smart turn + STT)
        #   user_stopped -> bot_started : response (LLM + TTS)
        self._vad_stopped_at: float | None = None
        self._user_stopped_at: float | None = None
        # Buffer of streamed LLM text, flushed as one log line per response.
        self._llm_text: list[str] = []
        self._llm_started_at: float | None = None

        # --- post-call analysis accumulator --------------------------------
        # Everything below is kept only so the end-of-call report can be built
        # from one call's own numbers. Prometheus aggregates across calls and
        # cannot answer "what happened on THIS call", which is the question a
        # disposition or a latency complaint actually asks.
        self._transcript: list[dict[str, Any]] = []
        self._exchanges: list[dict[str, float | None]] = []
        self._ttfb: dict[str, list[float]] = {}
        self._tokens: dict[str, int] = {"prompt": 0, "completion": 0}
        self._interruptions = 0
        self._backchannels = 0
        self._reengagements = 0
        self._ended_by_bot = False
        self._bot_stopped_at: float | None = None
        # Onset marks. A transcript arrives when the STT *finalizes* it, which
        # is after the speaker stopped — timestamping there puts every line
        # later than the audio it describes, and clicking it in review lands
        # you past the words. These record when each side actually began.
        self._user_onset: float | None = None
        self._bot_onset: float | None = None
        self._user_response_secs: list[float] = []
        self._started_at = time.monotonic()

    # -- lifecycle ----------------------------------------------------------

    def on_call_started(self) -> None:
        """Record the start of a call."""
        self._m.calls.inc()
        self._m.calls_active.inc()

    def on_call_ended(self, duration_secs: float) -> None:
        """Record the end of a call."""
        self._m.calls_active.dec()
        self._m.call_duration.observe(duration_secs)

    # -- turn tracking (wired to PipelineWorker.turn_tracking_observer) ------

    async def on_turn_started(self, turn_number: int) -> None:
        """Handle a conversation turn starting."""
        set_turn(turn_number)
        self._log.info("turn started", event="turn_started", turn_number=turn_number)

    async def on_turn_ended(
        self, turn_number: int, duration: float, was_interrupted: bool
    ) -> None:
        """Handle a conversation turn ending."""
        self._m.turns.labels(interrupted=str(bool(was_interrupted)).lower()).inc()
        self._m.turn_duration.observe(duration)
        self._log.info(
            "turn ended",
            event="turn_ended",
            turn_number=turn_number,
            duration_secs=round(duration, 3),
            was_interrupted=bool(was_interrupted),
        )

    # -- frame observation --------------------------------------------------

    async def on_push_frame(self, data: FramePushed) -> None:
        """Handle a frame moving between two processors."""
        frame = data.frame
        if not self._seen.add(frame.id):
            return
        try:
            await self._handle(frame)
        except Exception as exc:  # never let telemetry break the call
            self._log.opt(exception=exc).warning(
                "telemetry handler failed", frame_type=type(frame).__name__
            )

    async def _handle(self, frame: Frame) -> None:
        if isinstance(frame, MetricsFrame):
            self._handle_metrics(frame)

        elif isinstance(frame, UserStartedSpeakingFrame):
            # How long the caller took to answer the bot — the other half of
            # responsiveness, and the one that shows hesitation or confusion.
            if self._bot_stopped_at is not None:
                self._user_response_secs.append(time.monotonic() - self._bot_stopped_at)
                self._bot_stopped_at = None
            self._user_onset = time.monotonic()
            self._log.debug("user started speaking", event="user_started_speaking")

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Raw end of speech, before turn detection has decided anything.
            self._vad_stopped_at = time.monotonic()

        elif isinstance(frame, UserStoppedSpeakingFrame):
            now = time.monotonic()
            self._user_stopped_at = now
            if self._vad_stopped_at is not None:
                self._m.turn_detection_latency.observe(now - self._vad_stopped_at)
            self._log.debug("user stopped speaking", event="user_stopped_speaking")

        elif isinstance(frame, BotStartedSpeakingFrame):
            # TTS begins before the LLM has finished streaming, so this fires
            # first and is the right mark for the reply that follows.
            self._bot_onset = time.monotonic()
            self._record_latency()
            self._log.debug("bot started speaking", event="bot_started_speaking")

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_stopped_at = time.monotonic()
            self._log.debug("bot stopped speaking", event="bot_stopped_speaking")

        elif isinstance(frame, InterruptionFrame):
            self._interruptions += 1
            self._m.interruptions.inc()
            self._log.info("user interrupted the bot", event="interruption")

        elif isinstance(frame, TranscriptionFrame):
            onset = self._user_onset if self._user_onset is not None else time.monotonic()
            self._user_onset = None
            self._transcript.append({
                "role": "user",
                "text": frame.text or "",
                "at_secs": round(max(0.0, onset - self._started_at), 2),
                "language": str(getattr(frame, "language", None) or ""),
            })
            self._log.info(
                "user transcript",
                event="transcript",
                text=frame.text if self._log_transcripts else "<redacted>",
                chars=len(frame.text or ""),
                user_id=getattr(frame, "user_id", None),
                language=str(getattr(frame, "language", None) or ""),
            )

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._llm_text = []
            self._llm_started_at = time.monotonic()

        elif isinstance(frame, LLMTextFrame):
            self._llm_text.append(frame.text or "")

        elif isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._llm_text)
            elapsed = (
                time.monotonic() - self._llm_started_at
                if self._llm_started_at is not None
                else None
            )
            self._llm_text = []
            self._llm_started_at = None
            if text.strip():
                onset = self._bot_onset if self._bot_onset is not None else time.monotonic()
                self._bot_onset = None
                self._transcript.append({
                    "role": "bot",
                    "text": text,
                    "at_secs": round(max(0.0, onset - self._started_at), 2),
                    "elapsed_secs": round(elapsed, 3) if elapsed is not None else None,
                })
            self._log.info(
                "llm response",
                event="llm_response",
                text=text if self._log_transcripts else "<redacted>",
                chars=len(text),
                elapsed_secs=round(elapsed, 3) if elapsed is not None else None,
            )

        elif isinstance(frame, FunctionCallInProgressFrame):
            self._m.function_calls.labels(
                function=frame.function_name, status="started"
            ).inc()
            self._log.info(
                "function call started",
                event="function_call_started",
                function=frame.function_name,
                tool_call_id=frame.tool_call_id,
            )

        elif isinstance(frame, FunctionCallResultFrame):
            self._m.function_calls.labels(
                function=frame.function_name, status="completed"
            ).inc()
            self._log.info(
                "function call completed",
                event="function_call_completed",
                function=frame.function_name,
                tool_call_id=frame.tool_call_id,
            )

        elif isinstance(frame, ErrorFrame):
            processor = type(frame.processor).__name__ if frame.processor else "unknown"
            self._m.errors.labels(
                processor=processor, fatal=str(bool(frame.fatal)).lower()
            ).inc()
            self._log.error(
                "pipeline error",
                event="pipeline_error",
                error=frame.error,
                fatal=bool(frame.fatal),
                processor=processor,
            )

    def _record_latency(self) -> None:
        """Close out the latency budget for one exchange.

        Called when the bot starts speaking. Emits the two halves and the
        caller-perceived total, then clears the marks so a mid-turn barge-in
        cannot produce a second, bogus measurement from stale timestamps.
        """
        now = time.monotonic()
        vad_at, user_at = self._vad_stopped_at, self._user_stopped_at
        self._vad_stopped_at = self._user_stopped_at = None

        response = now - user_at if user_at is not None else None
        if response is not None:
            self._m.response_latency.observe(response)

        if vad_at is None:
            return

        voice_to_voice = now - vad_at
        turn_detection = (user_at - vad_at) if user_at is not None else None
        self._m.voice_to_voice_latency.observe(voice_to_voice)

        over_budget = voice_to_voice > self._budget_secs
        if over_budget:
            self._m.budget_exceeded.inc()

        self._exchanges.append({
            "voice_to_voice_secs": round(voice_to_voice, 4),
            "turn_detection_secs": round(turn_detection, 4) if turn_detection else None,
            "response_secs": round(response, 4) if response else None,
            "over_budget": over_budget,
        })

        log = self._log.warning if over_budget else self._log.info
        log(
            "voice-to-voice latency",
            event="voice_to_voice_latency",
            voice_to_voice_secs=round(voice_to_voice, 4),
            turn_detection_secs=round(turn_detection, 4) if turn_detection else None,
            response_secs=round(response, 4) if response else None,
            budget_secs=self._budget_secs,
            over_budget=over_budget,
        )

    def note_backchannel(self) -> None:
        """Count a suppressed caller backchannel (reported, not measured here)."""
        self._backchannels += 1

    def mark_timeline_start(self) -> None:
        """Rebase transcript timestamps onto the moment recording began.

        The observer is built during pipeline construction; the audio buffer
        starts recording later, when it sees the StartFrame. Timestamping
        transcripts from the observer's own birth therefore offsets every line
        against the audio by that gap — a constant, and constant misalignment
        is exactly what makes a transcript useless for review. Called from the
        recorder so both timelines share an origin.
        """
        self._started_at = time.monotonic()

    def note_ended_by_bot(self) -> None:
        """The bot hung up after saying goodbye (rather than the caller)."""
        self._ended_by_bot = True

    def note_reengagement(self) -> None:
        """Count a re-engagement prompt spoken to a silent caller."""
        self._reengagements += 1

    def snapshot(self) -> dict[str, Any]:
        """Everything this call produced, for the post-call report.

        Deliberately raw: percentiles and averages are computed by the analyser
        so this stays a record of what happened rather than a summary of it.
        """
        return {
            "transcript": list(self._transcript),
            "exchanges": list(self._exchanges),
            "ttfb_secs": {k: list(v) for k, v in self._ttfb.items()},
            "tokens": dict(self._tokens),
            "interruptions": self._interruptions,
            "backchannels_suppressed": self._backchannels,
            "reengagement_prompts": self._reengagements,
            "ended_by_bot": self._ended_by_bot,
            "user_response_secs": list(self._user_response_secs),
            "latency_budget_secs": self._budget_secs,
        }

    # -- metrics frames -----------------------------------------------------

    def _handle_metrics(self, frame: MetricsFrame) -> None:
        for item in frame.data:
            processor = item.processor
            model = item.model or "unknown"
            stype = _service_type(processor)

            if isinstance(item, TTFAMetricsData):
                # TTFA carries its own ttfb; that same value also arrives as a
                # separate TTFBMetricsData, so only the TTFA-specific parts are
                # recorded here to avoid double counting.
                self._m.ttfa.labels(processor=processor, model=model).observe(item.ttfa)
                self._m.tts_leading_silence.labels(
                    processor=processor, model=model
                ).observe(item.leading_silence)
                self._log.info(
                    "ttfa",
                    event="ttfa",
                    processor=processor,
                    model=model,
                    ttfa_secs=round(item.ttfa, 4),
                    ttfb_secs=round(item.ttfb, 4),
                    leading_silence_secs=round(item.leading_silence, 4),
                )

            elif isinstance(item, TTFBMetricsData):
                self._ttfb.setdefault(stype, []).append(item.value)
                self._m.ttfb.labels(
                    service_type=stype, processor=processor, model=model
                ).observe(item.value)
                self._m.requests.labels(
                    service_type=stype, processor=processor, model=model
                ).inc()
                self._log.info(
                    "ttfb",
                    event="ttfb",
                    service_type=stype,
                    processor=processor,
                    model=model,
                    ttfb_secs=round(item.value, 4),
                )

            elif isinstance(item, ProcessingMetricsData):
                self._m.processing.labels(
                    service_type=stype, processor=processor, model=model
                ).observe(item.value)
                self._log.debug(
                    "processing time",
                    event="processing",
                    service_type=stype,
                    processor=processor,
                    model=model,
                    processing_secs=round(item.value, 4),
                )

            elif isinstance(item, TextAggregationMetricsData):
                self._m.text_aggregation.labels(processor=processor).observe(item.value)

            elif isinstance(item, LLMUsageMetricsData):
                usage = item.value
                self._m.llm_calls.labels(model=model).inc()
                counted: dict[str, int] = {}
                for kind, value in token_kinds(usage):
                    self._m.llm_tokens.labels(model=model, kind=kind).inc(value)
                    if kind in self._tokens:
                        self._tokens[kind] += int(value)
                    counted[kind] = value
                self._log.info(
                    "llm token usage",
                    event="llm_tokens",
                    processor=processor,
                    model=model,
                    **{f"tokens_{k}": v for k, v in counted.items()},
                )

            elif isinstance(item, TTSUsageMetricsData):
                self._m.tts_characters.labels(processor=processor, model=model).inc(
                    item.value
                )
                self._log.debug(
                    "tts usage",
                    event="tts_usage",
                    processor=processor,
                    model=model,
                    characters=item.value,
                )

            elif isinstance(item, STTUsageMetricsData):
                self._m.stt_audio_seconds.labels(processor=processor, model=model).inc(
                    item.value.audio_seconds
                )
                self._log.debug(
                    "stt usage",
                    event="stt_usage",
                    processor=processor,
                    model=model,
                    audio_seconds=round(item.value.audio_seconds, 3),
                )

            elif isinstance(item, TurnMetricsData):
                self._m.smart_turn_predictions.labels(
                    is_complete=str(bool(item.is_complete)).lower()
                ).inc()
                self._m.smart_turn_probability.observe(item.probability)
                self._m.smart_turn_latency.observe(item.e2e_processing_time_ms / 1000.0)
                self._log.info(
                    "smart turn prediction",
                    event="smart_turn",
                    processor=processor,
                    is_complete=bool(item.is_complete),
                    probability=round(item.probability, 4),
                    e2e_ms=round(item.e2e_processing_time_ms, 2),
                )
