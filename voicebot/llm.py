"""OpenAI-compatible LLM that cannot ask for more room than the model has.

The failure this prevents
------------------------
``max_completion_tokens`` is a *reservation*. The server rejects a request when
``input + reservation > context``, so a reply cap that looks generous silently
becomes a hard 400 as soon as the conversation grows into the space it reserved.
Both halves of that have now broken a live call:

* a 5,000-token cap against a 16,384 window died mid-call once history reached
  11.5k tokens — the call had been running fine for eight turns;
* a 130,000-token cap against a 131,072 window died on the *first* turn, because
  the system prompt alone left no room for the reservation.

Neither is an unreasonable thing for someone to type into a box. The number that
is actually safe depends on the model's context, the size of the prompt, and how
far into the call you are — which is not knowable when the value is configured,
so it should not be configured at all. This service treats the setting as a
*ceiling* and clamps each request to what is genuinely left.

Estimating the input
--------------------
Deliberately pessimistic. Devanagari runs about two characters per token where
Latin runs four — the 26k-character Hindi prompt here is ~13k tokens, not the
~6.5k a flat four-chars rule predicts — so script is counted separately and the
result is rounded up. Over-estimating costs a few tokens of reply; under-
estimating costs the whole call.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from pipecat.services.openai.llm import OpenAILLMService

# Leaves room for chat-template scaffolding and tool definitions, which the
# character estimate below does not see.
_TEMPLATE_MARGIN = 512
# A reply worth speaking. If this will not fit, the context is genuinely full
# and the error should surface rather than being papered over.
_MIN_REPLY = 64


def estimate_tokens(text: str) -> int:
    """Pessimistic token count for mixed Devanagari/Latin text.

    Calibrated against the real thing rather than a rule of thumb: the 26,138
    character Hindi prompt in this repo reports ~13,000 prompt tokens from the
    server, i.e. **two characters per token across the whole text** — including
    its Latin sections, because the tokenizer is not tuned for this mixture.

    A per-script estimate (2 for Devanagari, 4 for Latin) returned 8,004 for
    that same prompt: a 38% under-count. Under-estimating hands back a reply
    budget that does not fit and the request 400s, so the two estimates are
    combined by taking whichever is larger. Over-estimating only costs a few
    tokens of reply.
    """
    if not text:
        return 0
    indic = sum(1 for ch in text if "ऀ" <= ch <= "෿")
    other = len(text) - indic
    per_script = -(-indic // 2) + -(-other // 4)
    flat = -(-len(text) // 2)
    return max(per_script, flat) + 8


def messages_tokens(messages: Any) -> int:
    """Estimate the input size of an OpenAI-style message list."""
    total = 0
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
    return total


def context_limit(base_url: str | None, api_key: str, model: str) -> int | None:
    """Ask the endpoint for its context window, if it will say.

    OpenAI does not report this; self-hosted servers (SGLang, vLLM) do, as
    ``max_model_len``. Returns None when unknown, in which case no clamping
    happens and behaviour is unchanged.
    """
    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/")
    try:
        response = httpx.get(
            f"{endpoint}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        response.raise_for_status()
        for entry in response.json().get("data", []):
            if entry.get("id") == model and entry.get("max_model_len"):
                return int(entry["max_model_len"])
    except Exception:
        pass
    return None


class ContextAwareOpenAILLMService(OpenAILLMService):
    """OpenAI-compatible LLM that fits its reply budget to the space left."""

    def __init__(self, *, context_window: int | None = None, **kwargs):
        """Initialize.

        Args:
            context_window: The model's context in tokens. None disables
                clamping (nothing is known, so nothing is assumed).
            **kwargs: Passed to :class:`OpenAILLMService`.
        """
        super().__init__(**kwargs)
        self._context_window = context_window
        self._clamped = 0
        self._log = logger.bind(component="llm")

    @property
    def clamped_requests(self) -> int:
        """Requests whose reply budget had to be reduced this call."""
        return self._clamped

    def build_chat_completion_params(self, params_from_context: dict) -> dict:
        """Build request params, trimming the reply budget to what fits."""
        params = super().build_chat_completion_params(params_from_context)
        if not self._context_window:
            return params

        wanted = params.get("max_completion_tokens") or params.get("max_tokens")
        if not wanted:
            return params

        used = messages_tokens(params.get("messages"))
        room = self._context_window - used - _TEMPLATE_MARGIN
        if room >= wanted:
            return params

        allowed = max(_MIN_REPLY, room)
        self._clamped += 1
        # Warning: the configured cap is not being honoured, and if this fires
        # every turn the context is nearly full and the call is close to dying.
        self._log.warning(
            "reply budget clamped to fit the context window",
            event="llm_budget_clamped",
            wanted=wanted,
            allowed=allowed,
            estimated_input=used,
            context_window=self._context_window,
        )
        for key in ("max_completion_tokens", "max_tokens"):
            if key in params:
                params[key] = allowed
        return params
