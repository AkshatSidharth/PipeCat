"""LLM-assisted editing of an agent's system prompt.

The one requirement that shapes everything here: the model's reply is written
**straight into the prompt box**. There is no human parsing step between the
response and the bot's instructions. So anything conversational — "Sure! Here's
the updated prompt:", a summary of what changed, a ```` ``` ```` fence — does
not just look untidy, it becomes part of what the bot is told to do on the next
call.

Hence:

* the instruction is emphatic that the whole prompt comes back, nothing else;
* the reply is stripped of code fences and common preambles;
* a reply that looks like commentary rather than a prompt is **rejected**
  rather than saved, because silently writing "Here is your updated prompt:"
  into an agent is worse than telling the user the edit failed;
* the original is returned untouched on any failure.

Editing is deliberately whole-prompt rather than patch-based. These prompts run
to thousands of words with interdependent sections, and a model applying a
targeted diff routinely drops the parts it was not thinking about. Asking for
the complete text back makes the omission visible — the response is either the
whole thing or it fails the length check below.
"""

from __future__ import annotations

import re

import httpx
from loguru import logger

_SYSTEM = """You are editing the system prompt of a production voice bot.

You will be given the CURRENT PROMPT and an EDIT INSTRUCTION.

Return the COMPLETE revised prompt and NOTHING ELSE.

Absolute rules:
- Output the entire prompt, from its first line to its last, with the edit applied.
- Do NOT summarise, explain, or describe what you changed.
- Do NOT write "Here is the updated prompt" or any other preamble.
- Do NOT wrap the output in code fences or quotes.
- Do NOT abbreviate any part with "..." or "[unchanged]" or similar.
- Preserve every section, rule and example that the instruction does not ask
  you to change, verbatim — including formatting, headers and language.
- Keep the original language and script (Hindi/Devanagari stays Devanagari).

Your entire response will be saved as the bot's instructions verbatim."""

# Openings a model reaches for when it ignores the "no preamble" rule.
_PREAMBLE = re.compile(
    r"^\s*(sure|certainly|of course|here('s| is)|below is|updated prompt|revised prompt)"
    r"[^\n]{0,80}[:\n]",
    re.IGNORECASE,
)


def _strip_wrapper(text: str) -> str:
    """Remove code fences and a leading conversational preamble."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1]
            if cleaned[:20].lower().startswith(("json", "text", "markdown", "md")):
                cleaned = cleaned.split("\n", 1)[-1]
    cleaned = cleaned.strip()
    if _PREAMBLE.match(cleaned):
        cleaned = cleaned.split("\n", 1)[-1].strip() if "\n" in cleaned else cleaned
    return cleaned.strip()


def _context_limit(endpoint: str, api_key: str, model: str) -> int | None:
    """The endpoint's context window, if it will tell us.

    Worth one cheap request: the completion budget has to be sized against the
    real limit, and guessing high is a hard 400 rather than a truncation.
    """
    try:
        response = httpx.get(
            f"{endpoint}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        for entry in response.json().get("data", []):
            if entry.get("id") == model and entry.get("max_model_len"):
                return int(entry["max_model_len"])
    except Exception:
        pass
    return None


def revise_prompt(
    settings, prompt: str, instruction: str, timeout: float = 900.0
) -> tuple[str, str | None]:
    """Apply ``instruction`` to ``prompt`` using the agent's own LLM.

    Args:
        settings: Resolved settings — supplies the endpoint, key and model.
        prompt: The current system prompt.
        instruction: Plain-English description of the edit.
        timeout: Request timeout. Generous by necessity — this regenerates the
            whole prompt, so it costs one token per token of prompt. On the
            self-hosted 30B (~20 tok/s) a 7.5k-token Hindi prompt takes about
            six minutes; on a hosted model it is seconds. Point
            ``analysis_model`` at something fast if you edit prompts often.

    Returns:
        ``(revised_prompt, error)``. On any failure ``error`` is set and the
        original prompt is returned unchanged, so a failed edit can never
        truncate or garble what is already saved.
    """
    log = logger.bind(component="prompting")
    base_url, api_key = settings.llm_endpoint()
    if not api_key:
        return prompt, "no LLM API key configured"

    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/")
    model = settings.analysis_model or settings.llm_model

    # Devanagari tokenizes at roughly two characters per token — a 15k-character
    # Hindi prompt is ~7.5k tokens, not the ~3.8k a 4-chars-per-token rule of
    # thumb suggests. Budgeting on the optimistic figure is how this first hit a
    # hard 400 rather than merely running short.
    input_estimate = (len(prompt) + len(instruction)) // 2 + 400
    # The revision is about the same length as the original, plus headroom.
    wanted = int(input_estimate * 1.35) + 512

    limit = _context_limit(endpoint, api_key, model)
    if limit:
        available = limit - input_estimate - 256  # margin for template overhead
        if available < 512:
            return prompt, (
                f"this prompt is ~{input_estimate} tokens and the model's context "
                f"is {limit}; there is no room to write the revision back. Use a "
                "model with a larger context, or shorten the prompt."
            )
        wanted = min(wanted, available)

    user = (
        f"CURRENT PROMPT:\n{prompt}\n\n"
        f"EDIT INSTRUCTION:\n{instruction}\n\n"
        "Return the complete revised prompt now, with no other text."
    )

    try:
        response = httpx.post(
            f"{endpoint}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
                "max_completion_tokens": wanted,
                "temperature": 0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""
    except httpx.TimeoutException:
        estimate = int(wanted / 20)  # local 30B decode rate
        log.warning("prompt edit timed out", event="prompt_edit_timeout")
        return prompt, (
            f"timed out. Rewriting this prompt means generating ~{wanted} tokens; "
            f"at the local model's ~20 tokens/sec that is ~{estimate}s. Set a "
            "faster Analysis model, or edit a shorter prompt."
        )
    except Exception as exc:
        log.warning(f"prompt edit failed: {exc}")
        return prompt, str(exc)[:200]

    revised = _strip_wrapper(content)
    if not revised:
        return prompt, "the model returned nothing"

    # Guard against a model that answered *about* the prompt instead of
    # returning it. A real revision of a substantial prompt is not 20% of its
    # length; a "I've updated the greeting section" reply is.
    if prompt.strip() and len(revised) < 0.4 * len(prompt.strip()):
        log.warning(
            "prompt edit rejected: reply too short to be the whole prompt",
            event="prompt_edit_rejected",
            was_chars=len(prompt),
            got_chars=len(revised),
        )
        return prompt, (
            f"the model returned {len(revised)} characters for a "
            f"{len(prompt)}-character prompt — it summarised instead of "
            "rewriting. Try a more specific instruction."
        )

    log.info(
        "prompt revised",
        event="prompt_edited",
        was_chars=len(prompt),
        now_chars=len(revised),
        instruction=instruction[:120],
    )
    return revised, None
