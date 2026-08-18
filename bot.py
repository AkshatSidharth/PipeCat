"""Voicebot entrypoint.

Run it with the Pipecat development runner::

    python bot.py -t webrtc      # browser client at http://localhost:7860
    python bot.py -t twilio      # inbound PSTN calls via Twilio
    python bot.py -t exotel      # inbound PSTN calls via Exotel
    python bot.py -t daily       # Daily room (incl. Daily PSTN dial-in)

The runner discovers the module-level ``bot`` coroutine below and calls it once
per session — one call, inbound or otherwise.

Inbound telephony
-----------------
For the websocket providers (Twilio / Telnyx / Plivo / Exotel) the runner
detects the provider from the first message on the socket and builds the right
frame serializer itself; all this file supplies is the audio params. PSTN is
8kHz end to end, so those transports run the pipeline at the wire rate instead
of resampling up and back down for no benefit.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import EndFrame, LLMRunFrame, TTSSpeakFrame
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

from voicebot.agents import load_agent
from voicebot.analysis import analyze_call
from voicebot.config import get_settings
from voicebot.filler import FillerMixer, warm_filler_clips
from voicebot.obs import (
    Metrics,
    call_context,
    setup_logging,
    shutdown_logging,
    start_metrics_server,
)
from voicebot.pipeline import build_bot

load_dotenv(override=False)

_settings = get_settings()
_loki_sink = setup_logging(_settings)

# One Metrics instance per process: Prometheus collectors are process-global and
# aggregate across every call the process handles.
_metrics = Metrics()
if _settings.metrics_enabled:
    start_metrics_server(_settings.metrics_port)


# Websocket telephony providers. The runner auto-detects which one is calling
# and attaches the matching serializer; we only supply audio params.
TELEPHONY_TRANSPORTS = ("twilio", "telnyx", "plivo", "exotel")


def _transport_params(mixer=None) -> dict:
    """Per-transport audio parameters.

    Note there is no ``vad_analyzer`` or ``turn_analyzer`` here: in Pipecat 1.x
    both live on the user context aggregator, not the transport.

    Args:
        mixer: Output audio mixer, used for filler backchannels. It has to be
            attached here because the transport is built before the agent's
            settings are known, so the mixer starts empty and is loaded later.
    """
    from pipecat.transports.daily.transport import DailyParams
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

    common = dict(audio_in_enabled=True, audio_out_enabled=True)
    if mixer is not None:
        common["audio_out_mixer"] = mixer
    rate = _settings.telephony_sample_rate

    params = {
        "daily": lambda: DailyParams(**common),
        "webrtc": lambda: TransportParams(**common),
    }
    for provider in TELEPHONY_TRANSPORTS:
        # add_wav_header and serializer are set by the runner.
        params[provider] = lambda: FastAPIWebsocketParams(
            **common,
            audio_in_sample_rate=rate,
            audio_out_sample_rate=rate,
        )
    return params


def _is_telephony(runner_args: RunnerArguments) -> bool:
    """True when this session is an inbound PSTN call over a websocket."""
    return getattr(runner_args, "transport_type", None) in TELEPHONY_TRANSPORTS


def _requested_agent(runner_args: RunnerArguments) -> str | None:
    """Saved-agent id carried on the session, if any.

    Browser calls put it in the WebRTC offer's ``request_data``; telephony can
    carry the same key in the webhook body, so one lookup covers both.
    """
    body = getattr(runner_args, "body", None)
    if isinstance(body, dict):
        value = body.get("agent") or body.get("agent_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def bot(runner_args: RunnerArguments) -> None:
    """Run one voicebot session.

    Args:
        runner_args: Session arguments supplied by the Pipecat runner.
    """
    session_id = getattr(runner_args, "session_id", None)
    with call_context(session_id) as call_id:
        log = logger.bind(component="bot")
        started_at = time.monotonic()

        if _loki_sink is not None:
            await _loki_sink.start()

        # create_transport auto-detects the telephony provider from the first
        # websocket message and writes it back onto runner_args, so the
        # telephony check has to happen after this call, not before.
        # Built unconditionally: the transport is created before we know which
        # agent (and so whether fillers are on), and an unloaded mixer plays
        # nothing. Clips are loaded below only if the agent wants them.
        filler_mixer = FillerMixer()
        transport = await create_transport(
            runner_args, _transport_params(filler_mixer)
        )

        settings = _settings

        # A saved agent, if one was requested. The builder UI puts its id in the
        # offer's request_data, which the runner surfaces as runner_args.body.
        # Anything the agent leaves blank keeps its .env value.
        agent_id = _requested_agent(runner_args)
        agent = load_agent(agent_id) if agent_id else None
        if agent is not None:
            settings = agent.apply_to(settings)
        elif agent_id:
            log.warning("unknown agent requested; using .env defaults", agent_id=agent_id)

        if _is_telephony(runner_args):
            rate = settings.telephony_sample_rate
            settings = settings.model_copy(
                update={"audio_in_sample_rate": rate, "audio_out_sample_rate": rate}
            )

        # Populated by the runner for inbound PSTN calls; None otherwise.
        call_data = getattr(runner_args, "call_data", None)

        bot_ctx = build_bot(
            transport=transport,
            settings=settings,
            call_id=call_id,
            metrics=_metrics,
            filler_mixer=filler_mixer,
            # Recordings are filed per agent; fall back to a shared directory
            # when the call did not name one.
            agent_name=(agent.id if agent else None) or "default",
        )
        worker = bot_ctx.worker
        bot_ctx.observer.on_call_started()
        log.info(
            "call started",
            event="call_started",
            call_id=call_id,
            transport=type(transport).__name__,
            transport_type=getattr(runner_args, "transport_type", None),
            agent=agent.name if agent else None,
            agent_id=agent.id if agent else None,
            inbound=call_data is not None,
            from_number=getattr(call_data, "from_number", None),
            to_number=getattr(call_data, "to_number", None),
            provider_call_id=getattr(call_data, "call_id", None),
        )

        @transport.event_handler("on_client_connected")
        async def _on_client_connected(_transport, client) -> None:
            log.info("client connected", event="client_connected")
            if settings.filler_enabled:
                # Background: first call for a given voice synthesizes the
                # clips, and nothing should wait on that. Until it finishes the
                # mixer simply plays nothing.
                asyncio.create_task(warm_filler_clips(settings, filler_mixer))
            # The bot speaks first — essential for inbound calls, where silence
            # after pickup reads as a dead line.
            if settings.greeting_mode == "generate":
                # The greeting is an instruction for the LLM to open the call.
                # Note a strict system prompt may decline to improvise, which is
                # why this is not the default.
                bot_ctx.context.add_message(
                    {"role": "user", "content": settings.greeting}
                )
                await worker.queue_frames([LLMRunFrame()])
            else:
                # Spoken verbatim, every call, with no LLM round-trip — so the
                # intro cannot be reinterpreted, refused, or answered instead of
                # said. append_to_context keeps the LLM aware of what it opened
                # with, so it does not greet twice.
                await worker.queue_frames(
                    [TTSSpeakFrame(settings.greeting, append_to_context=True)]
                )

        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, client) -> None:
            log.info("client disconnected", event="client_disconnected")
            # EndFrame drains in-flight work; the runner exits when the
            # pipeline finishes.
            await worker.queue_frames([EndFrame()])

        runner = WorkerRunner(
            handle_sigint=runner_args.handle_sigint,
            handle_sigterm=getattr(runner_args, "handle_sigterm", False),
        )
        await runner.add_workers(worker)

        try:
            await runner.run()
        finally:
            duration = time.monotonic() - started_at
            if bot_ctx.recorder is not None:
                await bot_ctx.recorder.close()
            bot_ctx.observer.on_call_ended(duration)
            log.info(
                "call ended",
                event="call_ended",
                call_id=call_id,
                duration_secs=round(duration, 2),
                recordings=bot_ctx.recorder.paths if bot_ctx.recorder else None,
            )
            # Post-call analysis. Off the hot path by construction — the call
            # is already over — so it runs to completion rather than being
            # raced against teardown. Failures are contained inside.
            try:
                await asyncio.to_thread(
                    analyze_call,
                    settings=settings,
                    snapshot=bot_ctx.observer.snapshot(),
                    call_id=call_id,
                    agent=(agent.id if agent else None) or "default",
                    duration_secs=duration,
                    output_dir=Path(settings.analysis_dir),
                    recordings=bot_ctx.recorder.paths if bot_ctx.recorder else None,
                    stem=bot_ctx.recorder.stem if bot_ctx.recorder else None,
                )
            except Exception as exc:
                log.warning(f"post-call analysis failed: {exc}")

            # Give the Loki batcher a moment to ship the tail of the call.
            await asyncio.sleep(settings.loki_batch_secs)


if __name__ == "__main__":
    from pipecat.runner.run import main

    try:
        main()
    finally:
        try:
            asyncio.run(shutdown_logging())
        except RuntimeError:
            pass
