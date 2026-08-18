"""Environment-driven configuration for the voicebot.

Everything tunable lives here so the pipeline code stays declarative. Values are
read from the process environment (and `.env`, loaded by the Pipecat runner or
by :func:`get_settings`).

Defaults are tuned for a **sub-500ms voice-to-voice** budget on the
Soniox / OpenAI / Sarvam stack. See ``latency_budget_ms`` and
the README's latency section before changing them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _read_prompt(filename: str, fallback: str) -> str:
    """Load a prompt from ``prompts/``, falling back to a built-in default.

    Prompts live in files rather than environment variables because they are
    multi-line and get edited often — a long system prompt in a `.env` is
    miserable to maintain.
    """
    path = _PROMPT_DIR / filename
    try:
        text = path.read_text(encoding="utf-8").strip()
        return text or fallback
    except OSError:
        return fallback


class Settings(BaseSettings):
    """All runtime configuration, sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- credentials -------------------------------------------------------
    # Read without the VOICEBOT_ prefix: these are the names the vendor SDKs
    # and the Pipecat docs use.
    soniox_api_key: str = Field(default="", alias="SONIOX_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")

    # --- models ------------------------------------------------------------
    # Soniox realtime v5. This is the wire model id ("stt-rt-v5"), which is what
    # the Soniox websocket API expects.
    stt_model: str = Field(default="stt-rt-v5", alias="VOICEBOT_STT_MODEL")
    # Comma-separated Soniox language hints. Soniox has no regional variants —
    # it takes base codes, so "en" covers Indian English. Listing more than one
    # turns on language identification, which is what tags each transcript and
    # drives the TTS language follower.
    stt_languages: str = Field(default="en,hi", alias="VOICEBOT_STT_LANGUAGES")
    # Restrict recognition to the hinted languages. Leave False so a caller who
    # drops into a third language still gets transcribed rather than mangled
    # into one of the two.
    stt_languages_strict: bool = Field(default=False, alias="VOICEBOT_STT_LANGUAGES_STRICT")

    # gpt-4.1-nano is OpenAI's lowest-latency text model, and critically it does
    # no reasoning: reasoning-class models (o-series, gpt-5.x) spend hundreds of
    # ms before the first token, which a voice bot cannot absorb. Step up to
    # gpt-4.1-mini if answer quality falls short.
    llm_model: str = Field(default="gpt-4.1-nano", alias="VOICEBOT_LLM_MODEL")
    llm_max_tokens: int = Field(default=300, alias="VOICEBOT_LLM_MAX_TOKENS")
    # OpenAI processing tier: "" (account default), "auto", "flex", "priority".
    # "priority" lowers and tightens latency but is a paid add-on — sending it
    # from an account without it errors, so it stays unset by default.
    openai_service_tier: str = Field(default="", alias="VOICEBOT_OPENAI_SERVICE_TIER")
    # Point the OpenAI client at any OpenAI-compatible server (self-hosted
    # SGLang/vLLM, a gateway, …). Blank = api.openai.com. The URL must include
    # the /v1 suffix, matching what the OpenAI SDK expects.
    llm_base_url: str = Field(default="", alias="VOICEBOT_LLM_BASE_URL")
    # Credential for llm_base_url when it is not OpenAI. Blank falls back to
    # OPENAI_API_KEY, so pointing at a self-hosted server does not require
    # overwriting the OpenAI key you still use elsewhere.
    llm_api_key: str = Field(default="", alias="VOICEBOT_LLM_API_KEY")

    def llm_endpoint(self) -> tuple[str | None, str]:
        """The LLM endpoint and the credential that belongs to it.

        Credentials are per-endpoint. Handing OpenAI a self-hosted key is a
        401, and the failure is quiet — the model picker just drops back to its
        static shortlist and preflight reports an auth error that looks like a
        bad OpenAI key. So the base URL decides which key is sent.

        Returns:
            ``(base_url_or_None, api_key)``. ``None`` means api.openai.com.
        """
        if self.llm_base_url:
            return self.llm_base_url, (self.llm_api_key or self.openai_api_key)
        return None, self.openai_api_key

    # Which TTS engine speaks. Sarvam is Indic-first; ElevenLabs covers 32
    # languages including Hindi and is often smoother on English.
    tts_provider: Literal["sarvam", "elevenlabs", "omnivoice"] = Field(
        default="sarvam", alias="VOICEBOT_TTS_PROVIDER"
    )
    # OmniVoice / FlowTTS — self-hosted. Base websocket URL; "/ws/<call_id>"
    # is appended per call.
    omnivoice_url: str = Field(
        default="ws://172.16.1.4:80/omnivoice-tts", alias="VOICEBOT_OMNIVOICE_URL"
    )
    # Voice-clone alias. An unknown alias does NOT error — the server silently
    # falls back to OmniVoice's un-cloned auto voice, so a typo here is
    # inaudible as a failure and merely wrong. "Anika" is verified present on
    # the deployment and is the Indian female voice suited to this bot; the
    # notebook's "hausa-female-1" is a real *Nigerian* clone and was the wrong
    # default to inherit.
    omnivoice_voice_id: str = Field(
        default="Anika", alias="VOICEBOT_OMNIVOICE_VOICE_ID"
    )
    omnivoice_speed: float = Field(default=1.0, alias="VOICEBOT_OMNIVOICE_SPEED")
    # eleven_flash_v2_5 is ElevenLabs' lowest-latency model.
    elevenlabs_model: str = Field(
        default="eleven_flash_v2_5", alias="VOICEBOT_ELEVENLABS_MODEL"
    )
    elevenlabs_voice_id: str = Field(default="", alias="VOICEBOT_ELEVENLABS_VOICE_ID")

    # Sarvam bulbul:v2 is the standard model; v3 / v3-beta are heavier
    # "advanced" variants. The pipeline uses Sarvam's websocket service, not the
    # HTTP one — streaming is what makes first-audio fast.
    tts_model: str = Field(default="bulbul:v2", alias="VOICEBOT_TTS_MODEL")
    sarvam_speaker: str = Field(default="anushka", alias="VOICEBOT_SARVAM_SPEAKER")
    # Language the bot opens in — the greeting is spoken before the caller has
    # said anything, so this is the one language that cannot be auto-detected.
    sarvam_language: str = Field(default="en-IN", alias="VOICEBOT_SARVAM_LANGUAGE")
    # Follow the caller between the configured languages mid-call. Sarvam
    # applies this on its open socket, so a switch costs no extra latency.
    tts_follow_caller_language: bool = Field(
        default=True, alias="VOICEBOT_TTS_FOLLOW_CALLER_LANGUAGE"
    )
    # Consecutive turns in a new language before the voice switches. 2 stops a
    # single code-mixed "Hinglish" turn from flipping the voice back and forth.
    tts_language_switch_after: int = Field(
        default=2, alias="VOICEBOT_TTS_LANGUAGE_SWITCH_AFTER"
    )
    # Characters Sarvam buffers before it starts synthesizing (its own default
    # is 50). Lower starts audio sooner at some cost to prosody.
    sarvam_min_buffer_size: int = Field(
        default=30, alias="VOICEBOT_SARVAM_MIN_BUFFER_SIZE"
    )
    # Leave unset to use the model's native rate and let Pipecat resample on the
    # way out. Forcing a rate the model does not support fails the connection.
    tts_sample_rate: int | None = Field(default=None, alias="VOICEBOT_TTS_SAMPLE_RATE")

    # How text reaches TTS. "sentence" waits for a sentence boundary
    # (~200-300ms); "token" streams tokens as they arrive. Sarvam already gates
    # synthesis with min_buffer_size, so "token" pairs well here — measure both.
    tts_text_aggregation: Literal["sentence", "token"] = Field(
        default="sentence", alias="VOICEBOT_TTS_TEXT_AGGREGATION"
    )

    # --- prompts -----------------------------------------------------------
    # Edit prompts/system.txt and prompts/greeting.txt. The env vars below
    # override the files when set, which is handy for per-deployment tweaks.
    system_prompt: str = Field(
        default_factory=lambda: _read_prompt(
            "system.txt",
            "You are a helpful voice assistant. Reply in short, plain "
            "conversational prose with no markdown or special characters.",
        ),
        alias="VOICEBOT_SYSTEM_PROMPT",
    )
    greeting: str = Field(
        default_factory=lambda: _read_prompt(
            "greeting.txt",
            "Hello! I am an automated assistant. How can I help you today?",
        ),
        alias="VOICEBOT_GREETING",
    )
    # "speak"    -> the greeting text is spoken verbatim, every call, with no
    #               LLM round-trip. Deterministic and faster to first audio.
    # "generate" -> the greeting is an *instruction*; the LLM writes the opening
    #               line. More natural, but a strict system prompt can refuse it
    #               or answer it instead of saying it.
    greeting_mode: Literal["speak", "generate"] = Field(
        default="speak", alias="VOICEBOT_GREETING_MODE"
    )

    # --- turn taking / interruptions ---------------------------------------
    # Max silence the smart-turn model tolerates before ending the turn anyway.
    # This is the *fallback* ceiling, not the common path: when the model says
    # "complete" the turn ends immediately after vad_stop_secs.
    smart_turn_stop_secs: float = Field(default=2.0, alias="VOICEBOT_SMART_TURN_STOP_SECS")
    # Path to a fine-tuned smart-turn ONNX model. Blank uses Pipecat's bundled
    # English v3.2. A custom model must keep the v3 contract: input_features
    # (B, 80, 800) float32 in, sigmoid probabilities out.
    smart_turn_model_path: str = Field(default="", alias="VOICEBOT_SMART_TURN_MODEL_PATH")
    # Probability above which a turn counts as complete. Pipecat hardcodes 0.5;
    # for a voice agent, biasing higher is usually right because talking over
    # the caller costs far more than answering a beat late. Fine-tuned models
    # normally publish their own operating point — use it.
    smart_turn_threshold: float = Field(default=0.5, alias="VOICEBOT_SMART_TURN_THRESHOLD")
    # Pure dead air on the wire, and typically the largest single term in the
    # budget. Pipecat's published STT TTFS benchmarks were measured at 0.2; we
    # run at 0.15 to buy 50ms, which is safe here because smart turn v3 — not
    # this timer — makes the actual end-of-turn call. Going below ~0.1 starts
    # triggering the model on mid-sentence pauses.
    vad_stop_secs: float = Field(default=0.15, alias="VOICEBOT_VAD_STOP_SECS")
    interrupt_min_words: int = Field(default=2, alias="VOICEBOT_INTERRUPT_MIN_WORDS")
    # Ignore "hmm"/"haan haan"/"ji ji" as interruptions while the bot is
    # speaking. The same words still start a turn when the bot is idle, where
    # they are an answer rather than a listener signal — see voicebot/backchannel.py.
    backchannel_suppression: bool = Field(
        default=True, alias="VOICEBOT_BACKCHANNEL_SUPPRESSION"
    )
    # Length caps a *varied* run of acknowledgements only. Repetition is how
    # Indian callers backchannel — "हाँ हाँ हाँ हाँ हाँ" runs as long as you keep
    # talking — so a repetitive run is never capped, however long.
    backchannel_max_tokens: int = Field(
        default=6, alias="VOICEBOT_BACKCHANNEL_MAX_TOKENS"
    )
    # Distinct acknowledgements still counted as repetition. Repeats fold
    # together first, so "haan"/"haaan"/"haanhaan" count as one.
    backchannel_max_distinct: int = Field(
        default=3, alias="VOICEBOT_BACKCHANNEL_MAX_DISTINCT"
    )

    # --- bot-side filler backchannels --------------------------------------
    # A quiet "hmm"/"ji ji" while the caller keeps talking, so the line does
    # not feel dead. Played through the output mixer, which is the only path
    # that cannot delay the real response — see voicebot/filler.py.
    filler_enabled: bool = Field(default=True, alias="VOICEBOT_FILLER_ENABLED")
    filler_after_secs: float = Field(default=3.0, alias="VOICEBOT_FILLER_AFTER_SECS")
    filler_interval_secs: float = Field(
        default=4.5, alias="VOICEBOT_FILLER_INTERVAL_SECS"
    )
    filler_max_per_turn: int = Field(default=3, alias="VOICEBOT_FILLER_MAX_PER_TURN")
    # Fillers are mixed under an open microphone; full volume reads as the bot
    # interrupting rather than acknowledging.
    filler_gain: float = Field(default=0.55, alias="VOICEBOT_FILLER_GAIN")
    # Bot backchannel wording, one phrase per line. Blank uses the built-in set
    # for the agent's language. Changing these re-renders the clip cache.
    filler_phrases_neutral: str = Field(
        default="", alias="VOICEBOT_FILLER_PHRASES_NEUTRAL"
    )
    filler_phrases_emphatic: str = Field(
        default="", alias="VOICEBOT_FILLER_PHRASES_EMPHATIC"
    )
    filler_phrases_hesitant: str = Field(
        default="", alias="VOICEBOT_FILLER_PHRASES_HESITANT"
    )

    # --- re-engagement -----------------------------------------------------
    # Check the caller is still there when the line goes quiet after the bot
    # finishes a turn. See voicebot/reengage.py.
    reengage_enabled: bool = Field(default=True, alias="VOICEBOT_REENGAGE_ENABLED")
    reengage_after_secs: float = Field(
        default=5.0, alias="VOICEBOT_REENGAGE_AFTER_SECS"
    )
    reengage_max_attempts: int = Field(
        default=3, alias="VOICEBOT_REENGAGE_MAX_ATTEMPTS"
    )
    # One prompt per line, escalating. Blank uses the built-in Hindi set.
    reengage_prompts: str = Field(default="", alias="VOICEBOT_REENGAGE_PROMPTS")

    # --- post-call analysis ------------------------------------------------
    # Runs after the call ends, so its latency costs the caller nothing. The
    # measured half (latency, counts, tokens) is always written; this switch
    # only controls the LLM-inferred half.
    analysis_enabled: bool = Field(default=True, alias="VOICEBOT_ANALYSIS_ENABLED")
    # Blank reuses the call's own LLM. Point it at a bigger/cheaper model if
    # you want better notes than the low-latency voice model produces.
    analysis_model: str = Field(default="", alias="VOICEBOT_ANALYSIS_MODEL")
    analysis_dir: str = Field(default="./recordings", alias="VOICEBOT_ANALYSIS_DIR")

    # --- end call ----------------------------------------------------------
    # Hang up once the bot has FINISHED speaking a closing line. Matched on the
    # tail of a response, not anywhere in it — see voicebot/endcall.py.
    end_call_enabled: bool = Field(default=True, alias="VOICEBOT_END_CALL_ENABLED")
    # One closing line per line. Blank uses the built-in set for the language.
    end_call_phrases: str = Field(default="", alias="VOICEBOT_END_CALL_PHRASES")
    # Covers the transport's jitter buffer, which still holds audio when the
    # bot-stopped-speaking signal fires.
    end_call_linger_secs: float = Field(
        default=0.4, alias="VOICEBOT_END_CALL_LINGER_SECS"
    )
    wait_for_transcript: bool = Field(default=True, alias="VOICEBOT_WAIT_FOR_TRANSCRIPT")
    # P99 seconds from speech end to final transcript, used by the turn-stop
    # strategy to size its safety-net timeout. Pipecat's measured Soniox value
    # is 0.35; re-measure for your region with pipecat-ai/stt-benchmark.
    stt_ttfs_p99: float = Field(default=0.35, alias="VOICEBOT_STT_TTFS_P99")
    # Watchdog for a stranded transcript: Soniox only emits a TranscriptionFrame
    # on an end token, and if the finalize that provokes one is missed the text
    # waits for the caller's NEXT utterance and arrives glued to it. Seconds of
    # silence, with text buffered, before asking again. 0 disables.
    stt_finalize_after: float = Field(
        default=1.5, alias="VOICEBOT_STT_FINALIZE_AFTER"
    )

    # --- latency target ----------------------------------------------------
    # Used for logging/alerting only; it does not change pipeline behaviour.
    latency_budget_ms: int = Field(default=500, alias="VOICEBOT_LATENCY_BUDGET_MS")

    # --- audio -------------------------------------------------------------
    audio_in_sample_rate: int = Field(default=16000, alias="VOICEBOT_AUDIO_IN_SAMPLE_RATE")
    audio_out_sample_rate: int = Field(default=24000, alias="VOICEBOT_AUDIO_OUT_SAMPLE_RATE")
    # PSTN is 8kHz end to end. Running the pipeline at the wire rate avoids
    # pointless up/down-sampling on inbound calls; bot.py applies this
    # automatically when the transport is a telephony provider.
    telephony_sample_rate: int = Field(
        default=8000, alias="VOICEBOT_TELEPHONY_SAMPLE_RATE"
    )

    # --- recording ---------------------------------------------------------
    recording_enabled: bool = Field(default=True, alias="VOICEBOT_RECORDING_ENABLED")
    recordings_dir: str = Field(default="./recordings", alias="VOICEBOT_RECORDINGS_DIR")
    recording_sample_rate: int = Field(default=24000, alias="VOICEBOT_RECORDING_SAMPLE_RATE")
    recording_flush_secs: float = Field(default=30.0, alias="VOICEBOT_RECORDING_FLUSH_SECS")

    # --- observability -----------------------------------------------------
    service_name: str = Field(default="pipecat-voicebot", alias="VOICEBOT_SERVICE_NAME")
    env: str = Field(default="dev", alias="VOICEBOT_ENV")

    metrics_enabled: bool = Field(default=True, alias="VOICEBOT_METRICS_ENABLED")
    metrics_port: int = Field(default=9188, alias="VOICEBOT_METRICS_PORT")

    loki_url: str = Field(default="", alias="VOICEBOT_LOKI_URL")
    loki_batch_secs: float = Field(default=1.0, alias="VOICEBOT_LOKI_BATCH_SECS")
    loki_batch_size: int = Field(default=200, alias="VOICEBOT_LOKI_BATCH_SIZE")
    loki_timeout_secs: float = Field(default=5.0, alias="VOICEBOT_LOKI_TIMEOUT_SECS")

    log_level: str = Field(default="INFO", alias="VOICEBOT_LOG_LEVEL")
    log_dir: str = Field(default="./logs", alias="VOICEBOT_LOG_DIR")
    log_transcripts: bool = Field(default=True, alias="VOICEBOT_LOG_TRANSCRIPTS")

    @field_validator("tts_sample_rate", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat an empty env var as "not set" for optional numeric fields.

        `.env` files have no way to express None: a commented-out line and
        ``VOICEBOT_TTS_SAMPLE_RATE=`` look the same to a human, but pydantic
        receives ``""`` for the latter and refuses to parse it as an int.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def require_credentials(self) -> None:
        """Fail fast with one actionable message listing every missing key."""
        missing = [
            name
            for name, value in (
                ("SONIOX_API_KEY", self.soniox_api_key),
                ("OPENAI_API_KEY", self.openai_api_key),
                ("SARVAM_API_KEY", self.sarvam_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required credentials: "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
