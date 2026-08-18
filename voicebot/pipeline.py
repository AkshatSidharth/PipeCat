"""Assembles the voicebot pipeline.

Cascaded architecture::

    transport.input()          caller audio in (mic or PSTN)
      -> SonioxSTTService      audio -> text   (stt-rt-v5, realtime websocket)
      -> user aggregator       VAD + turn strategies + context write
      -> OpenAILLMService      text -> text    (gpt-4.1-nano, streamed)
      -> SarvamTTSService      text -> audio   (bulbul:v2, websocket streaming)
      -> transport.output()    audio out to the caller
      -> AudioBufferProcessor  taps both directions for recording
      -> assistant aggregator  writes the bot turn back into context

The audio buffer processor sits *after* ``transport.output()`` deliberately:
that is the only point where both the inbound user audio (which passes through
the whole pipeline) and the outbound bot audio are visible.

Latency
-------
This stack is configured for a sub-500ms voice-to-voice budget. The choices
that actually buy that, in order of impact:

1. A non-reasoning LLM. ``gpt-4.1-nano`` starts emitting tokens almost
   immediately; any o-series or gpt-5.x reasoning model spends hundreds of ms
   thinking before the first token and cannot fit the budget.
2. ``vad_stop_secs=0.15``. Turn-detection silence is pure dead air on the wire
   and is usually the largest single term.
3. Sarvam's **websocket** service plus a lowered ``min_buffer_size``, so
   synthesis starts before a full sentence has accumulated.

What does *not* help: raising ``smart_turn_stop_secs``. That is only the
fallback ceiling for when the model keeps judging the turn incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from voicebot.backchannel import BackchannelAwareUserTurnStartStrategy
from voicebot.config import Settings
from voicebot.endcall import build_end_call
from voicebot.filler import FillerMixer, build_filler
from voicebot.language import LanguageFollower, parse_languages
from voicebot.llm import ContextAwareOpenAILLMService, context_limit
from voicebot.obs.metrics import Metrics
from voicebot.obs.observer import TelemetryObserver
from voicebot.recording import CallRecorder
from voicebot.reengage import build_reengagement
from voicebot.stt import FinalizingSonioxSTTService
from voicebot.tts import build_tts
from voicebot.turn import build_turn_analyzer


@dataclass
class Bot:
    """Everything the entrypoint needs to run and observe one call."""

    worker: PipelineWorker
    pipeline: Pipeline
    context: LLMContext
    observer: TelemetryObserver
    recorder: CallRecorder | None


def build_turn_strategies(settings: Settings) -> UserTurnStrategies:
    """Build the turn-taking strategy set.

    Two independent decisions live here.

    **When does a user turn *start*?** This is what interruption (barge-in)
    means in Pipecat 1.x: starting a user turn while the bot is speaking emits
    an interruption. Strategies are evaluated in order and the *first* one to
    fire wins, so mixing a VAD strategy with a word-count strategy would let
    VAD win every time and silently defeat the word gate. Therefore:

    * ``backchannel_suppression`` (default) → ``BackchannelAwareUserTurnStartStrategy``
      alone: the word gate *plus* a lexicon of Indian backchannels, both applied
      only while the bot is speaking. A word count cannot do this job on its own
      — "haan haan haan" is three words and means nothing, "ruko" is one word
      and means stop.
    * ``interrupt_min_words > 0`` → ``MinWordsUserTurnStartStrategy`` alone.
      It requires N words to interrupt while the bot is speaking, but only one
      word when the bot is idle. Word count only, so a three-word backchannel
      still interrupts.
    * ``interrupt_min_words == 0`` → the framework defaults (VAD +
      transcription), which interrupt on the first hint of speech. Lowest
      latency, most false triggers.

    **When does a user turn *stop*?** Smart turn v3: a local ONNX model that
    reads prosody to decide whether the user actually finished or is just
    pausing mid-thought. It ships inside ``pipecat-ai`` and runs on CPU, so
    there is no network hop and no extra dependency.
    """
    if settings.backchannel_suppression:
        # Supersedes MinWords: it keeps the word gate and adds a lexical one, so
        # "haan haan" no longer cuts the bot off while "haan lekin ek problem
        # hai" still does. Needs transcript text, hence it cannot be layered on
        # the VAD-only path below.
        start = [
            BackchannelAwareUserTurnStartStrategy(
                # Deliberately not settings.interrupt_min_words: that gate is a
                # crude stand-in for "ignore backchannels", and the lexicon does
                # that job precisely. Stacking them only swallows one-word
                # interruptions ("stop", "रुको") that must always get through.
                # interrupt_min_words still applies when suppression is off.
                min_words=1,
                max_tokens=settings.backchannel_max_tokens,
                max_distinct=settings.backchannel_max_distinct,
            )
        ]
    elif settings.interrupt_min_words > 0:
        start = [MinWordsUserTurnStartStrategy(min_words=settings.interrupt_min_words)]
    else:
        start = [VADUserTurnStartStrategy(), TranscriptionUserTurnStartStrategy()]

    smart_turn = build_turn_analyzer(
        model_path=settings.smart_turn_model_path,
        threshold=settings.smart_turn_threshold,
        params=SmartTurnParams(stop_secs=settings.smart_turn_stop_secs),
    )
    stop = [
        TurnAnalyzerUserTurnStopStrategy(
            turn_analyzer=smart_turn,
            wait_for_transcript=settings.wait_for_transcript,
        )
    ]
    return UserTurnStrategies(start=start, stop=stop)


def build_bot(
    *,
    transport: BaseTransport,
    settings: Settings,
    call_id: str,
    metrics: Metrics,
    filler_mixer: FillerMixer | None = None,
    agent_name: str = "default",
) -> Bot:
    """Construct the pipeline and worker for a single call.

    Args:
        transport: An already-created Pipecat transport.
        settings: Resolved application settings.
        call_id: Correlation id used for logs and recording filenames.
        metrics: Prometheus collectors to record into.
        filler_mixer: Mixer installed on the output transport, when filler
            backchannels are enabled. Built before the transport exists, so it
            is handed in rather than created here.
        agent_name: Saved agent driving this call. Recordings are filed under a
            directory of this name.

    Returns:
        A :class:`Bot` bundling the worker, context, observer and recorder.
    """
    settings.require_credentials()
    log = logger.bind(component="pipeline")

    # --- services ----------------------------------------------------------
    # vad_force_turn_endpoint=True (the default) keeps Soniox's own endpoint
    # detection OFF and lets local VAD + smart turn decide when the turn ends.
    # Flipping it to False hands turn-taking to Soniox and *bypasses smart turn
    # entirely* — so it stays on. The consequence: Soniox's endpoint_* settings
    # are inert here, and tuning them would be a no-op.
    languages = parse_languages(settings.stt_languages)
    # Only worth paying for when there is more than one language to tell apart;
    # it is what stamps TranscriptionFrame.language for the TTS follower.
    identify_language = len(languages) > 1

    stt = FinalizingSonioxSTTService(
        # Adds only a stalled-transcript watchdog; see voicebot/stt.py.
        finalize_after=settings.stt_finalize_after,
        api_key=settings.soniox_api_key,
        sample_rate=settings.audio_in_sample_rate,
        vad_force_turn_endpoint=True,
        # Sizes the turn-stop strategy's safety-net timeout. Soniox measures at
        # 0.35s P99, tied for the fastest of Pipecat's benchmarked STT services.
        ttfs_p99_latency=settings.stt_ttfs_p99,
        settings=FinalizingSonioxSTTService.Settings(
            model=settings.stt_model,
            language_hints=languages,
            language_hints_strict=settings.stt_languages_strict,
            enable_language_identification=identify_language,
        ),
    )

    # service_tier is only sent when set: "priority" is a paid OpenAI add-on and
    # an account without it rejects the request outright.
    llm_base_url, llm_api_key = settings.llm_endpoint()
    # Asked once per call: the reply budget is clamped against it so no
    # configured value can push a request past the window mid-conversation.
    window = context_limit(llm_base_url, llm_api_key, settings.llm_model)
    llm = ContextAwareOpenAILLMService(
        context_window=window,
        api_key=llm_api_key,
        # Any OpenAI-compatible server; None keeps the SDK's own default.
        base_url=llm_base_url,
        service_tier=settings.openai_service_tier or None,
        settings=ContextAwareOpenAILLMService.Settings(
            model=settings.llm_model,
            # max_completion_tokens, NOT max_tokens. Verified against the live
            # API: GPT-5 family models reject `max_tokens` outright
            # ("Unsupported parameter ... use 'max_completion_tokens' instead"),
            # while gpt-4.1 / gpt-4o accept both. This spelling is the only one
            # that works across every model the picker offers.
            max_completion_tokens=settings.llm_max_tokens,
        ),
    )
    if window:
        log.info(
            "llm context window",
            event="llm_context_window",
            model=settings.llm_model,
            context_tokens=window,
            reply_cap=settings.llm_max_tokens,
        )

    tts_language = Language(settings.sarvam_language)
    tts = build_tts(settings, call_id=call_id)

    # Retunes the voice when the caller changes language. Sits between STT and
    # the aggregator so it sees transcripts early, and pushes its update
    # downstream to the TTS.
    language_follower: LanguageFollower | None = None
    if settings.tts_follow_caller_language and identify_language:
        language_follower = LanguageFollower(
            initial=tts_language,
            allowed=languages,
            switch_after=settings.tts_language_switch_after,
        )

    # --- context + turn taking ---------------------------------------------
    context = LLMContext(
        messages=[{"role": "system", "content": settings.system_prompt}],
    )

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # VAD lives on the user aggregator in 1.x — it builds the
            # VADController internally, so no separate VADProcessor is needed.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=settings.vad_stop_secs)
            ),
            user_turn_strategies=build_turn_strategies(settings),
            # Arms on BotStoppedSpeakingFrame, cancels the moment anyone
            # speaks, and re-arms after each bot turn — so speaking a prompt
            # schedules the next check for free. 0 disables it entirely.
            user_idle_timeout=(
                settings.reengage_after_secs if settings.reengage_enabled else 0
            ),
        ),
    )

    # Built here rather than beside the worker below because the
    # re-engagement handlers report into it.
    observer = TelemetryObserver(
        metrics=metrics,
        log_transcripts=settings.log_transcripts,
        latency_budget_ms=settings.latency_budget_ms,
    )

    # --- re-engagement ------------------------------------------------------
    reengagement = build_reengagement(settings)
    if reengagement is not None:
        user_aggregator = aggregators.user()

        @user_aggregator.event_handler("on_user_turn_idle")
        async def _on_user_idle(aggregator) -> None:
            prompt = reengagement.next_prompt()
            if prompt is None:
                return
            observer.note_reengagement()
            # Spoken, not generated: these are fixed check-ins, and routing
            # them through the LLM would add its latency to a moment that is
            # already dead air. append_to_context=False keeps them out of the
            # transcript the model reasons over — the caller said nothing, so
            # there is nothing to remember.
            await aggregator.push_frame(
                TTSSpeakFrame(prompt, append_to_context=False),
                FrameDirection.DOWNSTREAM,
            )

        @user_aggregator.event_handler("on_user_turn_started")
        async def _on_user_spoke(aggregator, *args) -> None:
            reengagement.reset()

    end_call = build_end_call(settings, on_end=observer.note_ended_by_bot)

    # --- recording ----------------------------------------------------------
    recorder: CallRecorder | None = None
    if settings.recording_enabled:
        recorder = CallRecorder(
            call_id=call_id,
            output_dir=Path(settings.recordings_dir),
            sample_rate=settings.recording_sample_rate,
            metrics=metrics,
            flush_secs=settings.recording_flush_secs,
            agent=agent_name,
            on_started=observer.mark_timeline_start,
        )

    # --- pipeline -----------------------------------------------------------
    stages = [
        transport.input(),
    ]
    # Before the STT so it sees the caller's audio and VAD frames first; it only
    # observes, and talks to the mixer directly rather than pushing frames.
    if settings.filler_enabled and filler_mixer is not None:
        stages.append(build_filler(settings, filler_mixer))
    stages.append(stt)
    if language_follower is not None:
        stages.append(language_follower)
    stages += [
        aggregators.user(),
        llm,
    ]
    # Between LLM and TTS: it sees the response text going downstream and the
    # bot-stopped-speaking signal coming back upstream, which are the two
    # things it needs.
    if end_call is not None:
        stages.append(end_call)
    stages += [
        tts,
        transport.output(),
    ]
    if recorder is not None:
        stages.append(recorder.processor)
    stages.append(aggregators.assistant())

    pipeline = Pipeline(stages)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=settings.audio_in_sample_rate,
            audio_out_sample_rate=settings.audio_out_sample_rate,
            # Without these two the services emit no MetricsFrames at all, and
            # every latency/token panel in Grafana stays empty.
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        conversation_id=call_id,
        enable_turn_tracking=True,
        observers=[observer],
        idle_timeout_secs=300,
    )

    # Reuse the worker's own turn tracker rather than adding a second one.
    if worker.turn_tracking_observer is not None:
        # These must be `async def`, not lambdas: Pipecat dispatches with
        # `inspect.iscoroutinefunction(handler)`, which is False for a lambda
        # that merely returns a coroutine — it would be called synchronously
        # and the coroutine never awaited, silently dropping every turn metric.
        @worker.turn_tracking_observer.event_handler("on_turn_started")
        async def _on_turn_started(_obs, turn_number: int) -> None:
            await observer.on_turn_started(turn_number)

        @worker.turn_tracking_observer.event_handler("on_turn_ended")
        async def _on_turn_ended(
            _obs, turn_number: int, duration: float, was_interrupted: bool
        ) -> None:
            await observer.on_turn_ended(turn_number, duration, was_interrupted)

    log.info(
        "pipeline built",
        event="pipeline_built",
        call_id=call_id,
        llm_model=settings.llm_model,
        stt_model=settings.stt_model,
        tts_model=settings.tts_model,
        smart_turn=settings.smart_turn_model_path or "bundled v3.2",
        smart_turn_threshold=settings.smart_turn_threshold,
        smart_turn_stop_secs=settings.smart_turn_stop_secs,
        vad_stop_secs=settings.vad_stop_secs,
        interrupt_min_words=settings.interrupt_min_words,
        recording_enabled=settings.recording_enabled,
        stt_languages=[str(lang) for lang in languages],
        language_identification=identify_language,
        tts_language=str(tts_language),
        tts_follows_caller=language_follower is not None,
        openai_service_tier=settings.openai_service_tier or "account-default",
        tts_provider=settings.tts_provider,
        tts_text_aggregation=settings.tts_text_aggregation,
        sarvam_min_buffer_size=settings.sarvam_min_buffer_size,
        stt_ttfs_p99=settings.stt_ttfs_p99,
        latency_budget_ms=settings.latency_budget_ms,
        audio_in_sample_rate=settings.audio_in_sample_rate,
        audio_out_sample_rate=settings.audio_out_sample_rate,
    )

    return Bot(
        worker=worker,
        pipeline=pipeline,
        context=context,
        observer=observer,
        recorder=recorder,
    )
