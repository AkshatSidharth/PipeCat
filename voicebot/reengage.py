"""Re-engagement: check the caller is still there when the line goes quiet.

After the bot finishes a turn, a caller who says nothing is ambiguous — they may
be thinking, they may have put the phone down, or the audio path may have broken
in one direction and they cannot hear a thing. A human agent resolves that in a
couple of seconds with "hello? sun paa rahe hain?".

Pipecat already detects the silence: :class:`UserIdleController` arms a timer on
``BotStoppedSpeakingFrame`` and cancels it the moment anyone speaks, firing
``on_user_turn_idle`` if the timeout elapses. Two properties of that timer shape
this module:

* it fires **once** per arming, and re-arms when the bot next stops speaking —
  so simply speaking a prompt produces the next check automatically, and
  escalation needs no timer of its own; and
* it is suppressed while a user turn is in progress, so a caller who is midway
  through a sentence is never talked over.

What is left to add is *what to say* and *when to stop asking*. Prompts escalate
— a short "hello?" first, then explicitly asking about audio — and stop after a
configured number of attempts, because a bot that keeps talking into a dead line
is worse than one that goes quiet.

The counter resets as soon as the caller speaks, so a normal pause mid-call does
not consume the budget.
"""

from __future__ import annotations

from loguru import logger

# Escalating, in Devanagari: a bare "hello?" first, then progressively more
# explicit about whether the caller can hear anything at all.
DEFAULT_PROMPTS_HI = (
    "हैलो?",
    "जी, मेरी आवाज़ आ रही है?",
    "क्या आप मुझे सुन पा रहे हैं?",
    "हैलो, क्या आपको मेरी आवाज़ आ रही है?",
)

DEFAULT_PROMPTS_EN = (
    "Hello?",
    "Can you hear me?",
    "Are you still there?",
    "Hello, is my voice coming through?",
)


def parse_prompts(raw: str, language: str = "hi") -> list[str]:
    """Parse a newline-separated prompt list, falling back to the defaults.

    Args:
        raw: One prompt per line. Blank lines and stray whitespace are dropped.
        language: Base language code, used only to pick the fallback set.

    Returns:
        The prompts to cycle through, never empty.
    """
    prompts = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if prompts:
        return prompts
    default = (
        DEFAULT_PROMPTS_HI
        if language.split("-")[0].lower() == "hi"
        else DEFAULT_PROMPTS_EN
    )
    return list(default)


class Reengagement:
    """Decides whether to nudge a silent caller, and with what.

    Holds no timer of its own — :class:`UserIdleController` owns that. This is
    the policy: which line comes next, and when to stop.
    """

    def __init__(self, *, prompts: list[str], max_attempts: int = 3):
        """Initialize.

        Args:
            prompts: Lines to use, in escalation order. The last one repeats if
                ``max_attempts`` exceeds the number of prompts.
            max_attempts: Consecutive nudges before giving up until the caller
                speaks again. Keeps a broken audio path from turning into an
                endless monologue.
        """
        self._prompts = prompts
        self._max_attempts = max(0, max_attempts)
        self._attempts = 0
        self._total = 0
        self._log = logger.bind(component="reengage")

    @property
    def attempts_this_silence(self) -> int:
        """Nudges since the caller last spoke."""
        return self._attempts

    @property
    def total_attempts(self) -> int:
        """Nudges over the whole call."""
        return self._total

    def reset(self) -> None:
        """Caller spoke — forget the streak."""
        if self._attempts:
            self._log.debug(
                "caller responded; re-engagement reset",
                event="reengage_reset",
                after_attempts=self._attempts,
            )
        self._attempts = 0

    def next_prompt(self) -> str | None:
        """The line to speak now, or None when the budget is spent."""
        if not self._prompts or self._attempts >= self._max_attempts:
            if self._attempts == self._max_attempts and self._max_attempts:
                self._log.info(
                    "caller unresponsive; no further prompts",
                    event="reengage_exhausted",
                    attempts=self._attempts,
                )
                # Step past the cap so this logs once, not on every idle tick.
                self._attempts += 1
            return None

        # Escalate through the list, then hold on the last (most explicit) line.
        prompt = self._prompts[min(self._attempts, len(self._prompts) - 1)]
        self._attempts += 1
        self._total += 1
        self._log.info(
            "prompting a silent caller",
            event="reengage_prompt",
            attempt=self._attempts,
            text=prompt,
        )
        return prompt


def build_reengagement(settings) -> Reengagement | None:
    """Build from settings, or None when disabled."""
    if not settings.reengage_enabled:
        return None
    return Reengagement(
        prompts=parse_prompts(settings.reengage_prompts, settings.sarvam_language),
        max_attempts=settings.reengage_max_attempts,
    )
