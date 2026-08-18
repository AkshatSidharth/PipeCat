"""Prometheus metrics for the voicebot.

Everything Grafana graphs comes from here. Metric names are prefixed
``voicebot_`` and follow Prometheus conventions (``_seconds`` /``_total``
suffixes) so `rate()` and histogram_quantile() work as expected.

Label cardinality is kept bounded: labels are service type, processor class
name, and model name — all drawn from a small fixed set. Per-call identifiers
are **not** labels; use the Loki logs to drill into an individual call.
"""

from __future__ import annotations

from typing import Iterable

from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

# Latency buckets tuned for conversational voice: sub-100ms matters, and
# anything past ~5s is already a failed interaction.
_LATENCY_BUCKETS: tuple[float, ...] = (
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
)

# Voice-to-voice buckets are deliberately dense around the 500ms target so the
# p95 line has real resolution where the SLO sits.
_V2V_BUCKETS: tuple[float, ...] = (
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.6,
    0.7,
    0.85,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    5.0,
)

_TURN_BUCKETS: tuple[float, ...] = (0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60, 120)
_CALL_BUCKETS: tuple[float, ...] = (5, 15, 30, 60, 120, 300, 600, 1800, 3600)


class Metrics:
    """Container for every Prometheus collector the bot exports.

    Instantiating against a custom registry keeps tests isolated; the default
    is the global registry that :func:`start_metrics_server` exposes.
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        """Create the collectors.

        Args:
            registry: Registry to register into. Defaults to the global one.
        """
        kw = {"registry": registry} if registry is not None else {}

        # --- latency -------------------------------------------------------
        self.ttfb = Histogram(
            "voicebot_ttfb_seconds",
            "Time to first byte for a service response.",
            ["service_type", "processor", "model"],
            buckets=_LATENCY_BUCKETS,
            **kw,
        )
        self.ttfa = Histogram(
            "voicebot_ttfa_seconds",
            "Time to first audible TTS sample (TTFB plus leading silence).",
            ["processor", "model"],
            buckets=_LATENCY_BUCKETS,
            **kw,
        )
        self.tts_leading_silence = Histogram(
            "voicebot_tts_leading_silence_seconds",
            "Silence padded onto the front of a TTS response.",
            ["processor", "model"],
            buckets=(0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8),
            **kw,
        )
        self.processing = Histogram(
            "voicebot_processing_seconds",
            "Wall-clock processing time for a service request.",
            ["service_type", "processor", "model"],
            buckets=_LATENCY_BUCKETS,
            **kw,
        )
        self.text_aggregation = Histogram(
            "voicebot_text_aggregation_seconds",
            "Time from first LLM token to first complete sentence handed to TTS.",
            ["processor"],
            buckets=_LATENCY_BUCKETS,
            **kw,
        )
        # --- the latency budget, split into its two additive halves ---------
        # voice_to_voice == turn_detection + response. Graph all three together
        # and the one that blew the budget is obvious at a glance.
        self.voice_to_voice_latency = Histogram(
            "voicebot_voice_to_voice_latency_seconds",
            "VAD speech end to bot audio start. The latency a caller actually "
            "experiences, including turn detection.",
            buckets=_V2V_BUCKETS,
            **kw,
        )
        self.turn_detection_latency = Histogram(
            "voicebot_turn_detection_latency_seconds",
            "VAD speech end to user turn end: VAD silence wait, smart-turn "
            "inference and STT finalization.",
            buckets=_V2V_BUCKETS,
            **kw,
        )
        self.response_latency = Histogram(
            "voicebot_response_latency_seconds",
            "User turn end to bot audio start: LLM plus TTS. Excludes turn "
            "detection — compare against voice_to_voice for the full picture.",
            buckets=_V2V_BUCKETS,
            **kw,
        )
        self.budget_exceeded = Counter(
            "voicebot_latency_budget_exceeded_total",
            "Responses whose voice-to-voice latency exceeded the configured budget.",
            **kw,
        )

        # --- request volume ------------------------------------------------
        self.requests = Counter(
            "voicebot_requests_total",
            "Service requests observed, counted from emitted TTFB metrics.",
            ["service_type", "processor", "model"],
            **kw,
        )

        # --- tokens & usage ------------------------------------------------
        self.llm_tokens = Counter(
            "voicebot_llm_tokens_total",
            "LLM tokens by kind.",
            ["model", "kind"],
            **kw,
        )
        self.llm_calls = Counter(
            "voicebot_llm_calls_total",
            "Completed LLM generations that reported token usage.",
            ["model"],
            **kw,
        )
        self.tts_characters = Counter(
            "voicebot_tts_characters_total",
            "Characters synthesized by TTS.",
            ["processor", "model"],
            **kw,
        )
        self.stt_audio_seconds = Counter(
            "voicebot_stt_audio_seconds_total",
            "Seconds of audio submitted to STT.",
            ["processor", "model"],
            **kw,
        )

        # --- turns, interruptions, smart turn -------------------------------
        self.turns = Counter(
            "voicebot_turns_total",
            "Conversation turns completed.",
            ["interrupted"],
            **kw,
        )
        self.turn_duration = Histogram(
            "voicebot_turn_duration_seconds",
            "Duration of a conversation turn.",
            buckets=_TURN_BUCKETS,
            **kw,
        )
        self.interruptions = Counter(
            "voicebot_interruptions_total",
            "Times the user barged in and interrupted the bot.",
            **kw,
        )
        self.smart_turn_predictions = Counter(
            "voicebot_smart_turn_predictions_total",
            "Smart turn end-of-turn predictions.",
            ["is_complete"],
            **kw,
        )
        self.smart_turn_probability = Histogram(
            "voicebot_smart_turn_probability",
            "Confidence of smart turn end-of-turn predictions.",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
            **kw,
        )
        self.smart_turn_latency = Histogram(
            "voicebot_smart_turn_inference_seconds",
            "VAD silence to smart-turn verdict, end to end.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5),
            **kw,
        )

        # --- calls ----------------------------------------------------------
        self.calls_active = Gauge(
            "voicebot_calls_active",
            "Calls currently in progress.",
            **kw,
        )
        self.calls = Counter(
            "voicebot_calls_total",
            "Calls started.",
            **kw,
        )
        self.call_duration = Histogram(
            "voicebot_call_duration_seconds",
            "Total call duration.",
            buckets=_CALL_BUCKETS,
            **kw,
        )

        # --- recording ------------------------------------------------------
        self.recordings = Counter(
            "voicebot_recordings_total",
            "Call recordings finalized on disk.",
            ["track"],
            **kw,
        )
        self.recorded_seconds = Counter(
            "voicebot_recorded_seconds_total",
            "Seconds of audio written to recordings.",
            ["track"],
            **kw,
        )

        # --- errors & tools ---------------------------------------------------
        self.errors = Counter(
            "voicebot_errors_total",
            "Error frames observed in the pipeline.",
            ["processor", "fatal"],
            **kw,
        )
        self.function_calls = Counter(
            "voicebot_function_calls_total",
            "LLM tool/function invocations.",
            ["function", "status"],
            **kw,
        )


def start_metrics_server(port: int, registry: CollectorRegistry | None = None) -> None:
    """Expose ``/metrics`` over HTTP for Prometheus to scrape.

    Failure to bind is logged, not raised: losing metrics should never take
    a live call down.

    Args:
        port: TCP port to listen on.
        registry: Registry to serve. Defaults to the global one.
    """
    try:
        if registry is not None:
            start_http_server(port, registry=registry)
        else:
            start_http_server(port)
        logger.bind(component="metrics").info(
            "prometheus metrics server listening", port=port
        )
    except OSError as exc:
        logger.bind(component="metrics").warning(
            "could not start metrics server; metrics will not be scrapable",
            port=port,
            error=str(exc),
        )


def token_kinds(usage: object) -> Iterable[tuple[str, int]]:
    """Yield ``(kind, count)`` pairs from a Pipecat ``LLMTokenUsage``.

    Optional fields that the provider did not report are skipped rather than
    recorded as zero, so an absent metric stays absent instead of looking like
    a real zero measurement.
    """
    mapping = {
        "prompt": getattr(usage, "prompt_tokens", None),
        "completion": getattr(usage, "completion_tokens", None),
        "total": getattr(usage, "total_tokens", None),
        "cache_read": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation": getattr(usage, "cache_creation_input_tokens", None),
        "reasoning": getattr(usage, "reasoning_tokens", None),
    }
    for kind, value in mapping.items():
        if value:
            yield kind, int(value)
