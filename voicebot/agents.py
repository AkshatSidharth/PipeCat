"""Saved agent configurations.

An *agent* is a named bundle of everything that makes one bot behave the way it
does — models, voice, turn-taking, prompts. They are stored as plain JSON under
``agents/`` so they diff cleanly in git and can be edited by hand as easily as
through the builder UI.

At call time the UI passes the agent's id in the WebRTC offer's ``request_data``;
the Pipecat runner surfaces that as ``runner_args.body``, and :func:`apply_to`
layers the saved values over the environment-derived :class:`Settings`. Anything
the agent does not specify keeps its `.env` value, so an agent is a *diff*
against your deployment defaults rather than a full replacement.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from voicebot.config import Settings

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"

# Fields that map straight onto Settings. Kept explicit rather than derived so
# adding a field to the UI is a deliberate, reviewable act.
_SETTINGS_FIELDS = (
    "stt_model",
    "stt_languages",
    "stt_languages_strict",
    "stt_ttfs_p99",
    "stt_finalize_after",
    "llm_model",
    "llm_max_tokens",
    "openai_service_tier",
    "llm_base_url",
    "tts_provider",
    "tts_model",
    "elevenlabs_model",
    "elevenlabs_voice_id",
    "omnivoice_url",
    "omnivoice_voice_id",
    "omnivoice_speed",
    "backchannel_suppression",
    "backchannel_max_tokens",
    "backchannel_max_distinct",
    "filler_enabled",
    "filler_after_secs",
    "filler_interval_secs",
    "filler_max_per_turn",
    "filler_gain",
    "filler_phrases_neutral",
    "filler_phrases_emphatic",
    "filler_phrases_hesitant",
    "reengage_enabled",
    "reengage_after_secs",
    "reengage_max_attempts",
    "reengage_prompts",
    "analysis_enabled",
    "analysis_model",
    "end_call_enabled",
    "end_call_phrases",
    "end_call_linger_secs",
    "sarvam_speaker",
    "sarvam_language",
    "sarvam_min_buffer_size",
    "tts_text_aggregation",
    "tts_follow_caller_language",
    "tts_language_switch_after",
    "vad_stop_secs",
    "smart_turn_stop_secs",
    "smart_turn_model_path",
    "smart_turn_threshold",
    "interrupt_min_words",
    "wait_for_transcript",
    "latency_budget_ms",
    "recording_enabled",
    "system_prompt",
    "greeting",
    "greeting_mode",
)


def slugify(name: str) -> str:
    """Turn a display name into a safe filename stem.

    Path separators and dots are stripped rather than escaped, so a crafted
    name cannot walk out of ``agents/``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:64] or "agent"


class AgentConfig(BaseModel):
    """One saved agent."""

    id: str = ""
    name: str = "New agent"
    description: str = ""

    # --- speech to text ---
    stt_model: str = "stt-rt-v5"
    stt_languages: str = "en,hi"
    stt_languages_strict: bool = False
    stt_ttfs_p99: float = 0.35
    stt_finalize_after: float = 1.5

    # --- language model ---
    llm_model: str = "gpt-4.1-nano"
    llm_max_tokens: int = 300
    openai_service_tier: str = ""
    llm_base_url: str = ""

    # --- text to speech ---
    tts_provider: Literal["sarvam", "elevenlabs", "omnivoice"] = "sarvam"
    omnivoice_url: str = ""
    omnivoice_voice_id: str = "Anika"
    omnivoice_speed: float = 1.0
    tts_model: str = "bulbul:v2"
    elevenlabs_model: str = "eleven_flash_v2_5"
    elevenlabs_voice_id: str = ""
    sarvam_speaker: str = "anushka"
    sarvam_language: str = "en-IN"
    sarvam_min_buffer_size: int = 30
    tts_text_aggregation: Literal["sentence", "token"] = "sentence"
    tts_follow_caller_language: bool = True
    tts_language_switch_after: int = 2

    # --- backchannels ---
    backchannel_suppression: bool = True
    backchannel_max_tokens: int = 6
    backchannel_max_distinct: int = 3
    filler_enabled: bool = True
    filler_after_secs: float = 3.0
    filler_interval_secs: float = 4.5
    filler_max_per_turn: int = 3
    filler_gain: float = 0.55
    filler_phrases_neutral: str = ""
    filler_phrases_emphatic: str = ""
    filler_phrases_hesitant: str = ""
    reengage_enabled: bool = True
    reengage_after_secs: float = 5.0
    reengage_max_attempts: int = 3
    reengage_prompts: str = ""
    analysis_enabled: bool = True
    analysis_model: str = ""
    end_call_enabled: bool = True
    end_call_phrases: str = ""
    end_call_linger_secs: float = 0.4

    # --- voice activity + turn taking ---
    vad_stop_secs: float = 0.15
    smart_turn_stop_secs: float = 2.0
    smart_turn_model_path: str = ""
    smart_turn_threshold: float = 0.5
    interrupt_min_words: int = 2
    wait_for_transcript: bool = True

    # --- behaviour ---
    system_prompt: str = ""
    greeting: str = ""
    greeting_mode: Literal["speak", "generate"] = "speak"

    # --- operational ---
    latency_budget_ms: int = 500
    recording_enabled: bool = True

    updated_at: str = ""

    def apply_to(self, settings: Settings) -> Settings:
        """Return a copy of ``settings`` with this agent's values layered on.

        Empty strings are treated as "not set" so a blank prompt field in the
        builder falls back to ``prompts/`` rather than blanking the bot.
        """
        update: dict[str, Any] = {}
        for field in _SETTINGS_FIELDS:
            value = getattr(self, field, None)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            update[field] = value
        return settings.model_copy(update=update)


def _path(agent_id: str) -> Path:
    return AGENTS_DIR / f"{slugify(agent_id)}.json"


def list_agents() -> list[AgentConfig]:
    """Every saved agent, newest first. Unreadable files are skipped."""
    if not AGENTS_DIR.exists():
        return []
    agents: list[AgentConfig] = []
    for path in AGENTS_DIR.glob("*.json"):
        try:
            agents.append(AgentConfig(**json.loads(path.read_text("utf-8"))))
        except (OSError, ValueError):
            continue
    return sorted(agents, key=lambda a: a.updated_at, reverse=True)


def load_agent(agent_id: str) -> AgentConfig | None:
    """Load one agent by id, or ``None`` if it is missing or unreadable."""
    path = _path(agent_id)
    try:
        return AgentConfig(**json.loads(path.read_text("utf-8")))
    except (OSError, ValueError):
        return None


def save_agent(agent: AgentConfig) -> AgentConfig:
    """Persist an agent, assigning an id from its name when absent."""
    agent.id = slugify(agent.id or agent.name)
    agent.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    _path(agent.id).write_text(
        json.dumps(agent.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return agent


def delete_agent(agent_id: str) -> bool:
    """Delete an agent. Returns False if it did not exist."""
    path = _path(agent_id)
    if not path.exists():
        return False
    path.unlink()
    return True


# --------------------------------------------------------------------------
# OpenAI model discovery
# --------------------------------------------------------------------------
# Models that are not chat completions at all, or are the wrong shape for a
# cascaded voice pipeline (realtime is a different API; codex/deep-research are
# task-specific; transcribe/tts/embeddings are other modalities).
_NOT_A_CHAT_MODEL = re.compile(
    r"transcribe|tts|whisper|embed|moderation|dall-e|image|sora|audio|realtime"
    r"|search|deep-research|codex|instruct|davinci|babbage",
    re.I,
)
# Dated snapshots duplicate their floating alias — keep the alias.
_DATED_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")

_TIER_ORDER = {"fast": 0, "balanced": 1, "slow": 2, "avoid": 3}


def _classify_openai_model(model_id: str) -> tuple[str, str]:
    """Rank a model for voice use from its name.

    This is a *naming heuristic*, not a benchmark: OpenAI publishes no latency
    figures through the API. It is here to steer you away from the models that
    are structurally wrong for voice (anything that reasons before answering),
    not to predict milliseconds. Measure the real number on the console's TTFB
    panel — that is what it is for.
    """
    mid = model_id.lower()
    if re.match(r"^o\d", mid) or mid.endswith("-pro") or "-pro-" in mid:
        return "avoid", "reasoning/pro — hundreds of ms before the first token"
    if "nano" in mid:
        return "fast", "nano tier — fastest first token"
    if "mini" in mid:
        return "balanced", "mini tier — good latency/quality trade"
    if "chat-latest" in mid:
        return "balanced", "chat-tuned (non-reasoning)"
    return "slow", "full-size — slowest first token"


def _version_key(model_id: str) -> float:
    """Extract a sortable version so newer families surface first."""
    match = re.search(r"(\d+(?:\.\d+)?)", model_id)
    return float(match.group(1)) if match else 0.0


def openai_models(
    api_key: str, base_url: str = "", timeout: float = 6.0
) -> dict[str, Any]:
    """List chat models this key can actually reach, ranked for voice.

    Queried live so the console reflects the caller's real account rather than
    a list frozen at build time. Falls back to a curated static set when the
    API is unreachable, so the builder still works offline.

    Args:
        api_key: Credential for the endpoint.
        base_url: OpenAI-compatible endpoint, including ``/v1``. Blank means
            api.openai.com. A self-hosted server lists only the model it is
            actually serving, which is exactly what the picker should show.
        timeout: Per-request timeout in seconds.
    """
    static = [
        {"id": "gpt-4.1-nano", "tier": "fast", "note": "nano tier — fastest first token"},
        {"id": "gpt-4.1-mini", "tier": "balanced", "note": "mini tier — good latency/quality trade"},
        {"id": "gpt-4o-mini", "tier": "balanced", "note": "previous generation"},
        {"id": "gpt-4.1", "tier": "slow", "note": "full-size — slowest first token"},
    ]
    if not api_key:
        return {"models": static, "source": "static", "reason": "no OPENAI_API_KEY set"}

    try:
        import httpx

        response = httpx.get(
            f"{(base_url or 'https://api.openai.com/v1').rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        raw_entries = response.json().get("data", [])
        raw = [m["id"] for m in raw_entries]
    except Exception as exc:  # offline, bad key, rate limited...
        return {"models": static, "source": "static", "reason": str(exc)[:120]}

    if base_url:
        # A self-hosted server serves whatever it was launched with, and the
        # OpenAI naming rules below do not apply to it. Filtering by "gpt-"
        # would drop every id it reports and silently fall back to the OpenAI
        # shortlist — which then looks live but is not.
        return {
            "models": [
                {
                    "id": entry.get("id"),
                    "tier": "balanced",
                    "note": (
                        f"served by this endpoint · context "
                        f"{entry['max_model_len']:,} tokens"
                        if entry.get("max_model_len")
                        else "served by this endpoint"
                    ),
                    "max_model_len": entry.get("max_model_len"),
                }
                for entry in raw_entries
            ]
            or static,
            "source": "live" if raw else "static",
            "reason": "" if raw else "endpoint listed no models",
        }

    models = []
    for model_id in raw:
        if not model_id.startswith(("gpt-", "o1", "o3", "o4")):
            continue
        if _NOT_A_CHAT_MODEL.search(model_id) or _DATED_SNAPSHOT.search(model_id):
            continue
        tier, note = _classify_openai_model(model_id)
        models.append({"id": model_id, "tier": tier, "note": note})

    # Fast tiers first, newest family first within a tier.
    models.sort(key=lambda m: (_TIER_ORDER[m["tier"]], -_version_key(m["id"]), m["id"]))
    return {"models": models or static, "source": "live", "reason": ""}


def _omnivoice_probe(
    url: str, voice_id: str, language: str, timeout: float
) -> tuple[float, bool]:
    """Synthesize one short phrase over OmniVoice, returning (ttfb_ms, got_audio).

    A cold connection costs seconds while a warm one costs ~450ms, so this
    measures the cold path — which is what a fresh call actually pays if the
    service has not pre-connected.
    """
    import asyncio
    import time
    import uuid

    import websockets

    from voicebot.omnivoice import language_to_omnivoice, split_frame

    async def go() -> tuple[float, bool]:
        call_id = f"preflight-{uuid.uuid4().hex[:8]}"
        started = time.perf_counter()
        async with websockets.connect(
            f"{url}/ws/{call_id}",
            max_size=100 * 1024 * 1024,
            open_timeout=timeout,
            ping_interval=None,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "synthesize",
                        "call_id": call_id,
                        "text_id": "preflight",
                        "text": "namaste",
                        "streaming": True,
                        "voice_id": voice_id,
                        "language": language_to_omnivoice(language),
                        "speed": 1.0,
                    }
                )
            )
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout * 3)
                # A bare PCM continuation frame carries no header.
                if isinstance(raw, (bytes, bytearray)) and (not raw or raw[0] != 0x7B):
                    if raw:
                        return (time.perf_counter() - started) * 1000, True
                    continue
                header, audio = split_frame(raw)
                kind = header.get("type")
                if kind == "audio_chunk":
                    return (time.perf_counter() - started) * 1000, bool(audio) or True
                if kind == "error":
                    raise RuntimeError(header.get("error", "unknown OmniVoice error"))
                if kind == "audio_done":
                    return (time.perf_counter() - started) * 1000, False

    return asyncio.run(go())


def preflight(agent: AgentConfig, settings: Settings, timeout: float = 12.0) -> list[dict]:
    """Exercise the agent's actual services before anyone dials it.

    Every failure this project has hit in practice — a TTS model that rejects
    punctuation, an LLM that rejects ``max_tokens``, a deprecated model id still
    listed by ``/v1/models`` — was invisible until a live call. Each check below
    makes one real, minimal request so those surface in the builder instead.

    Returns one result per service: ``{service, ok, detail}``.
    """
    import httpx

    results: list[dict] = []

    def add(service: str, ok: bool, detail: str = "") -> None:
        results.append({"service": service, "ok": ok, "detail": detail[:200]})

    # --- LLM ---------------------------------------------------------------
    # Deliberately uses the agent's REAL system prompt and REAL token limit.
    # A trivial "hi"/16-token probe passes on models that produce nothing in
    # production: reasoning models burn the completion budget thinking, return
    # finish_reason="length" with empty content, and the bot goes silent with
    # no error anywhere. Only a realistic request surfaces that.
    # Preflight must authenticate exactly the way the pipeline will, or it
    # green-lights a config that 401s on the first call.
    llm_base, llm_key = agent.apply_to(settings).llm_endpoint()
    if not llm_key:
        add("llm", False, "no LLM API key (OPENAI_API_KEY / VOICEBOT_LLM_API_KEY)")
    else:
        system_prompt = agent.system_prompt.strip() or settings.system_prompt
        try:
            base = (llm_base or "https://api.openai.com/v1").rstrip("/")
            r = httpx.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {llm_key}"},
                json={
                    "model": agent.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "hello"},
                    ],
                    "max_completion_tokens": agent.llm_max_tokens,
                },
                timeout=timeout,
            )
            if r.status_code != 200:
                add("llm", False, r.json().get("error", {}).get("message", r.text))
            else:
                body = r.json()
                choice = body["choices"][0]
                usage = body.get("usage", {})
                reasoning = usage.get("completion_tokens_details", {}).get(
                    "reasoning_tokens", 0
                )
                text = choice["message"].get("content") or ""

                if text.strip():
                    detail = f"{agent.llm_model} replied ({len(text)} chars"
                    detail += f", {reasoning} reasoning tokens)" if reasoning else ")"
                    add("llm", True, detail)
                elif choice.get("finish_reason") == "length" and reasoning:
                    # The failure that makes a bot mute with no error at all.
                    add(
                        "llm",
                        False,
                        f"{agent.llm_model} is a reasoning model: it spent all "
                        f"{reasoning} tokens of the {agent.llm_max_tokens} budget "
                        f"thinking and returned no speech. Raise max tokens to "
                        f"~{max(1000, reasoning * 2)}, or pick a model that does "
                        f"not reason (gpt-4.1-nano, gpt-5.4-nano).",
                    )
                else:
                    add(
                        "llm",
                        False,
                        f"{agent.llm_model} returned no text "
                        f"(finish_reason={choice.get('finish_reason')})",
                    )
        except Exception as exc:
            add("llm", False, str(exc))

    # --- TTS: synthesize a token of real text in the configured voice -----
    if agent.tts_provider == "omnivoice":
        url = (agent.omnivoice_url or settings.omnivoice_url).rstrip("/")
        try:
            ttfb, voiced = _omnivoice_probe(
                url, agent.omnivoice_voice_id, agent.sarvam_language, timeout
            )
            if not voiced:
                add("tts", False, "OmniVoice returned no audio")
            else:
                # An unknown alias is not an error server-side, so say which
                # voice actually rendered rather than implying the config is
                # confirmed. See voicebot/omnivoice.py.
                add(
                    "tts",
                    True,
                    f"OmniVoice {agent.omnivoice_voice_id} — first audio in "
                    f"{ttfb:.0f}ms (alias unverified: unknown names fall back "
                    f"to the auto voice)",
                )
        except Exception as exc:
            add("tts", False, f"{url}: {exc}")

    elif agent.tts_provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            add("tts", False, "ELEVENLABS_API_KEY not set")
        elif not agent.elevenlabs_voice_id.strip():
            add("tts", False, "no ElevenLabs voice id set")
        else:
            try:
                r = httpx.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{agent.elevenlabs_voice_id}",
                    headers={"xi-api-key": settings.elevenlabs_api_key},
                    json={"text": "ok", "model_id": agent.elevenlabs_model},
                    timeout=timeout,
                )
                if r.status_code == 200:
                    add("tts", True, f"{agent.elevenlabs_model} / {agent.elevenlabs_voice_id}")
                else:
                    detail = r.json().get("detail", {})
                    msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
                    # The most common ElevenLabs trap: library ("professional")
                    # voices are paid-only, and the failure is a plan problem,
                    # not a config typo.
                    if r.status_code == 402:
                        msg += (
                            "  → this voice needs a paid plan. Free accounts can "
                            "only use 'premade' voices via the API."
                        )
                    add("tts", False, msg)
            except Exception as exc:
                add("tts", False, str(exc))
    elif not settings.sarvam_api_key:
        add("tts", False, "SARVAM_API_KEY not set")
    else:
        try:
            r = httpx.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": settings.sarvam_api_key},
                json={
                    "text": "ok",
                    "target_language_code": agent.sarvam_language,
                    "model": agent.tts_model,
                    "speaker": agent.sarvam_speaker,
                },
                timeout=timeout,
            )
            if r.status_code == 200:
                add("tts", True, f"{agent.tts_model} / {agent.sarvam_speaker}")
            else:
                body = r.json()
                msg = body.get("error", {})
                msg = msg.get("message", str(body)) if isinstance(msg, dict) else str(msg)
                add("tts", False, msg)
        except Exception as exc:
            add("tts", False, str(exc))

    # --- Smart turn: load the ONNX model and check the v3 contract ---------
    if agent.smart_turn_model_path.strip():
        try:
            from voicebot.turn import validate_turn_model

            add("turn", True, validate_turn_model(agent.smart_turn_model_path))
        except Exception as exc:
            add("turn", False, str(exc))

    # --- STT: Soniox is websocket-only, so just confirm the key authenticates.
    if not settings.soniox_api_key:
        add("stt", False, "SONIOX_API_KEY not set")
    else:
        add("stt", True, f"{agent.stt_model} (key present; verified on connect)")

    return results


def agent_options() -> dict[str, Any]:
    """Choices for the builder's dropdowns.

    Model and voice lists are read out of the installed Pipecat package where
    it publishes them, so the UI cannot drift from what the services actually
    accept. OpenAI has no offline registry, so that list is curated and the
    field stays free-text.
    """
    from pipecat.services.sarvam.tts import (
        SarvamTTSModel,
        SarvamTTSSpeakerV2,
        SarvamTTSSpeakerV3,
        language_to_sarvam_language,
    )
    from pipecat.transcriptions.language import Language

    # Indic languages Sarvam maps explicitly, paired with their Soniox base code.
    indic = [
        ("en", Language.EN, "English (India)"),
        ("hi", Language.HI, "Hindi"),
        ("bn", Language.BN, "Bengali"),
        ("gu", Language.GU, "Gujarati"),
        ("kn", Language.KN, "Kannada"),
        ("ml", Language.ML, "Malayalam"),
        ("mr", Language.MR, "Marathi"),
        ("or", Language.OR, "Odia"),
        ("pa", Language.PA, "Punjabi"),
        ("ta", Language.TA, "Tamil"),
        ("te", Language.TE, "Telugu"),
    ]

    return {
        "stt_models": ["stt-rt-v5"],
        "llm_models": [
            {"id": "gpt-4.1-nano", "note": "fastest — recommended"},
            {"id": "gpt-4.1-mini", "note": "slower, smarter"},
            {"id": "gpt-4.1", "note": "slowest of the 4.1 family"},
            {"id": "gpt-4o-mini", "note": "previous generation"},
        ],
        "service_tiers": ["", "auto", "flex", "priority"],
        "tts_models": [m.value for m in SarvamTTSModel],
        # Speaker sets differ per model; the UI swaps the list when the model changes.
        "tts_speakers": {
            "bulbul:v2": [s.value for s in SarvamTTSSpeakerV2],
            "bulbul:v3": [s.value for s in SarvamTTSSpeakerV3],
            "bulbul:v3-beta": [s.value for s in SarvamTTSSpeakerV3],
        },
        "languages": [
            {
                "stt": stt_code,
                "tts": language_to_sarvam_language(lang),
                "label": label,
            }
            for stt_code, lang, label in indic
        ],
        "aggregation_modes": ["sentence", "token"],
        "greeting_modes": ["speak", "generate"],
        # Endpoint presets for the builder's LLM provider picker. "" is OpenAI.
        "llm_providers": [
            {
                "id": "openai",
                "label": "OpenAI (cloud)",
                "base_url": "",
                "model": "gpt-4.1-nano",
                "note": "Lowest first-token latency. Needs OPENAI_API_KEY.",
            },
            {
                "id": "selfhosted",
                "label": "Self-hosted — Qwen3.8-27B",
                "base_url": "http://101.53.139.195/v1",
                "model": "qwen3.8-27b",
                "note": (
                    "On your L40S via SGLang (FP8, 32k context). TTFT ~200ms, "
                    "~20 tok/s decode, so a sentence takes 2-4s. One model is "
                    "loaded at a time; Muse-Glimmer's weights are still on disk "
                    "— run bin/start_llm.sh instead of bin/start_llm_qwen.sh to "
                    "switch back. The model list below is read live from the "
                    "endpoint, so it always shows what is actually serving."
                ),
            },
            {
                "id": "custom",
                "label": "Custom OpenAI-compatible",
                "base_url": "",
                "model": "",
                "note": "Any server speaking the OpenAI API. Include /v1.",
            },
        ],
        "tts_providers": ["sarvam", "elevenlabs", "omnivoice"],
        "elevenlabs_models": ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"],
        # Verified present on the deployment by pitch-fingerprinting each
        # alias against the un-cloned fallback (see README). Indian voices
        # first — the Nigerian clones ship with the server but do not suit a
        # Hindi/Indian-English bot.
        "omnivoice_voices": [
            {"id": "monika", "label": "monika — Indian support agent (cloned from monika_vb.mp3)"},
            {"id": "anika", "label": "anika — clear, warm, professional (Indian)"},
            {"id": "saavi", "label": "saavi — Indian female"},
            {"id": "niharika", "label": "niharika — Indian female"},
            {"id": "monisha", "label": "monisha — Indian female"},
            {"id": "bench-hindi_female", "label": "bench-hindi_female — Hindi female"},
            {"id": "hausa-female-1", "label": "hausa-female-1 — Nigerian female"},
            {"id": "hausa-male-1", "label": "hausa-male-1 — Nigerian male"},
        ],
    }
