"""Backchannel-aware barge-in: don't let "hmm" and "haan haan" cut the bot off.

Indian callers backchannel constantly. While the agent is talking, a listener
keeps signalling attention — *hmm*, *ji ji*, *haan haan*, *achha*, *uh-huh*.
None of that is a bid to take the floor, but to a turn-taking system that
interrupts on any speech it is indistinguishable from one, and the bot stops
mid-sentence. On a call that reads as the bot being unable to hold a thought.

This module supplies a user-turn-**start** strategy that keeps the floor with
the bot when the incoming speech is only a backchannel.

The one rule that makes this safe
---------------------------------
**Suppression applies only while the bot is speaking.** The same word means
different things depending on who has the floor:

* bot talking, caller says "हाँ"  → *I'm listening.* Do not interrupt.
* bot silent (it just asked something), caller says "हाँ" → *Yes.* That is an
  answer and must start a turn.

So the identical token is suppressed in one context and honoured in the other,
which is what makes an aggressive lexicon safe. Anything not recognised as a
backchannel interrupts exactly as before, so the failure mode is "we missed a
backchannel and interrupted", never "we swallowed what the caller said".

Matching against what Soniox actually returns
---------------------------------------------
The lexicon was built by synthesizing each phrase and transcribing it through
the same Soniox ``stt-rt-v5`` config the bot runs (``language_hints=en,hi``).
The transcripts are not what you would guess:

======================  ==========================  ===============================
spoken                  Soniox returned             consequence
======================  ==========================  ===============================
``hmm`` (English)       ``हम्म।``                    English can come back Devanagari
``ओके`` (Devanagari)    ``Okay.``                   Hindi can come back Latin
``ji ji`` (romanized)   ``जी, जी।``                  romanized input, Devanagari out
``haan haan``           ``Hanhan.``                 repeats merge into one token
``yeah yeah``           ``Yeah.``                   repeats collapse to one
``theek hai``           ``दिखाई।``                   sometimes simply wrong
======================  ==========================  ===============================

Script therefore does **not** follow the spoken language, so every concept is
listed in both scripts. Repeats are handled twice over: token-wise (``जी जी``)
and within a single token (``hanhan``). The last row is why the matcher fails
toward interrupting — a mangled transcript will not match, and the caller gets
their interruption.

Punctuation stripping is deliberately category-based rather than the obvious
``re.sub(r"[^\\w\\s]", ...)``. Devanagari combining marks are ``Mn``/``Mc`` and
are not ``\\w``, so the regex form turns ``हाँ`` into ``ह`` and no Hindi
backchannel ever matches.
"""

from __future__ import annotations

import unicodedata

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.base_user_turn_start_strategy import (
    BaseUserTurnStartStrategy,
)

# Single tokens that carry no floor-taking intent on their own. Both scripts for
# every concept, because Soniox does not keep script and language aligned.
BACKCHANNEL_TOKENS: frozenset[str] = frozenset(
    {
        # --- Devanagari ---
        "हाँ", "हां", "हा", "हूँ", "हूं", "हुह", "हम्म", "हम", "हँ",
        "जी", "जीजी", "अच्छा", "अच्छ", "ठीक", "सही", "बिलकुल", "बिल्कुल",
        "ओके", "ओहो", "अरे", "हम्मम", "समझा", "समझ",
        # --- romanized Hindi / Hinglish ---
        "haan", "han", "haa", "ha", "hanji", "haanji", "ji", "jee", "jij",
        "achha", "accha", "acha", "achcha", "theek", "thik", "teek",
        "bilkul", "bilcul", "sahi", "samjha", "samajh",
        # --- Marathi (Devanagari) ---
        "हो", "होय", "बरं", "बर", "बरोबर", "खरं",
        # --- Bengali / Assamese ---
        "হ্যাঁ", "হ্যা", "হুম", "হম", "আচ্ছা", "জি", "ঠিক", "হয়", "আচ্ছা",
        # --- Gujarati ---
        "હા", "હાજી", "જી", "બરાબર", "સારું", "હમ્મ",
        # --- Punjabi (Gurmukhi) ---
        "ਹਾਂ", "ਜੀ", "ਅੱਛਾ", "ਠੀਕ", "ਸਹੀ", "ਹਾਂਜੀ",
        # --- Tamil ---
        "ஆம்", "ஆமா", "ஆமாம்", "சரி", "ம்ம்", "ம்",
        # --- Telugu ---
        "అవును", "సరే", "అలాగే", "ఊ", "ఊఁ", "మంచిది",
        # --- Kannada ---
        "ಹೌದು", "ಸರಿ", "ಆಯ್ತು", "ಹ್ಮ್", "ಸರಿಸರಿ",
        # --- Malayalam ---
        "അതെ", "ശരി", "ഉം", "ആഹ്",
        # --- Odia ---
        "ହଁ", "ଠିକ୍", "ଆଚ୍ଛା",
        # --- English ---
        "hmm", "hm", "hmmm", "mhm", "mm", "mmm", "uhhuh", "uhuh", "huh",
        "yeah", "yea", "yep", "yup", "yes", "ok", "okay", "okey",
        "right", "sure", "correct", "true", "exactly", "absolutely",
        "alright", "cool", "fine", "great", "good", "nice",
    }
)

# Multi-word backchannels whose individual tokens are too generic to be listed
# above ("है" alone is not a backchannel; "ठीक है" is).
BACKCHANNEL_PHRASES: frozenset[str] = frozenset(
    {
        # --- Devanagari ---
        "ठीक है", "सही है", "अच्छा जी", "हाँ जी", "हां जी", "जी हाँ", "जी हां",
        "समझ गया", "समझ गयी", "समझ गई", "बिल्कुल सही", "ठीक है जी",
        "अच्छा अच्छा", "हाँ हाँ जी", "जी बिल्कुल", "ओके जी",
        # --- romanized ---
        "theek hai", "thik hai", "sahi hai", "achha ji", "accha ji",
        "haan ji", "han ji", "ji haan", "ji han", "samajh gaya", "samjh gaya",
        "bilkul sahi", "ok ji", "okay ji",
        # --- English ---
        "uh huh", "uh-huh", "mm hmm", "i see", "got it", "makes sense",
        "of course", "for sure", "all right", "that's right", "thats right",
        "i understand", "understood", "no problem", "very good",
    }
)

# Explicit floor-claiming words. Any of these present means the caller wants the
# bot to stop, even if every other token is a backchannel ("हाँ हाँ रुकिए").
#
# This is a safety net rather than the primary mechanism: a token outside
# BACKCHANNEL_TOKENS already fails the all-tokens test and interrupts on its
# own. It exists so that widening the lexicon later cannot accidentally create
# a suppressed phrase that means "stop".
INTERRUPT_TOKENS: frozenset[str] = frozenset(
    {
        # --- Devanagari ---
        "नहीं", "नही", "मत", "रुको", "रुकिए", "रुकिये", "रुक", "ठहरो", "ठहरिए",
        "सुनो", "सुनिए", "लेकिन", "पर", "गलत", "अरेरे", "बस", "छोड़ो",
        # --- romanized ---
        "nahi", "nahin", "nahee", "ruko", "rukiye", "rukie", "suno", "suniye",
        "lekin", "galat", "bas",
        # --- English ---
        "no", "not", "nope", "stop", "wait", "hold", "listen", "but",
        "actually", "sorry", "excuse", "hello", "hey", "hang", "pause",
    }
)

# A *varied* string of acknowledgements is capped; a *repetitive* one is not.
_MAX_BACKCHANNEL_TOKENS = 6
_MAX_DISTINCT_BACKCHANNELS = 3


def _strip_punctuation(text: str) -> str:
    """Replace Unicode punctuation with spaces, preserving combining marks.

    Category-based on purpose: the Devanagari danda (``।``, category ``Po``) has
    to go, while the vowel signs and chandrabindu that make up ``हाँ``
    (``Mc``/``Mn``) must survive. A ``\\w``-based regex gets this backwards.
    """
    return "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )


def _collapse_elongation(token: str, keep: int = 2) -> str:
    """Cap runs of identical characters at ``keep`` ("haaaan" -> "haan").

    Both settings are used. ``keep=2`` preserves genuine doubles, so "hmm" and
    "haan" match as listed. ``keep=1`` catches the rest: a drawn-out Devanagari
    "हााा" repeats the vowel *sign*, and only a full collapse reduces it to the
    listed "हा".
    """
    out: list[str] = []
    run = 0
    for char in token:
        if out and char == out[-1]:
            run += 1
            if run >= keep:
                continue
        else:
            run = 0
        out.append(char)
    return "".join(out)


def normalize(text: str) -> str:
    """Fold a transcript to the form the lexicon is written in.

    NFC-composes, lowercases, drops punctuation and collapses whitespace.
    """
    folded = unicodedata.normalize("NFC", text).lower()
    return " ".join(_strip_punctuation(folded).split())


def _repeated_unit(token: str) -> str | None:
    """The backchannel a single token repeats, if it is one ("hanhan" -> "han").

    Soniox runs repeated backchannels together into one token often enough that
    this matters: without it "hanhan" looks like an unknown word and interrupts.
    Returning the unit rather than a bool lets the caller fold repeats together
    when counting how *varied* an utterance is.
    """
    length = len(token)
    for size in range(1, length // 2 + 1):
        if length % size:
            continue
        unit = token[:size]
        if unit * (length // size) == token and unit in BACKCHANNEL_TOKENS:
            return unit
    return None


def canonical_token(token: str) -> str | None:
    """Fold a token to its listed form, or None if it is not a backchannel.

    "haaan", "haan" and "haanhaan" all reduce to the same entry, so a run of
    them counts as *one* distinct backchannel rather than three.
    """
    if token in BACKCHANNEL_TOKENS:
        return token
    for keep in (2, 1):
        collapsed = _collapse_elongation(token, keep)
        if collapsed in BACKCHANNEL_TOKENS:
            return collapsed
    return _repeated_unit(token)


def contains_interrupt_token(text: str) -> bool:
    """True when the caller explicitly claimed the floor ("stop", "रुकिए", "no").

    Checked before every other gate. These are one word often enough that a
    word-count threshold would swallow them, which is the opposite of what the
    caller asked for.
    """
    return any(token in INTERRUPT_TOKENS for token in normalize(text).split())


def is_backchannel(
    text: str,
    *,
    max_tokens: int = _MAX_BACKCHANNEL_TOKENS,
    max_distinct: int = _MAX_DISTINCT_BACKCHANNELS,
) -> bool:
    """True when ``text`` is only a listener signal, not a bid for the floor.

    Length alone does not decide this, because **repetition is how Indian
    callers backchannel**: "हाँ हाँ हाँ हाँ हाँ" or "haan haan haan hmm" runs on
    for as long as the speaker keeps talking and still means nothing more than
    *go on*. Treating a long run as real speech gets it exactly backwards — the
    more it repeats, the more clearly it is a backchannel.

    So the cap applies to *variety*, not length:

    * few distinct acknowledgements, however many times repeated → backchannel
      at any length;
    * many *different* ones strung together → capped, on the theory that a long
      varied utterance is more likely real speech the lexicon happens to cover.

    Args:
        text: Raw transcript text.
        max_tokens: Cap for a *varied* run of acknowledgements.
        max_distinct: Distinct acknowledgements still counted as repetition.
            Repeats fold together first, so "haan", "haaan" and "haanhaan" are
            one, not three.

    Returns:
        Whether the bot should keep talking through this.
    """
    normalized = normalize(text)
    if not normalized:
        # Empty or punctuation-only: nothing was said, so nothing to interrupt
        # for. Never treat this as a floor bid.
        return True

    if normalized in BACKCHANNEL_PHRASES:
        return True

    tokens = normalized.split()
    if any(token in INTERRUPT_TOKENS for token in tokens):
        return False

    canonical = [canonical_token(token) for token in tokens]
    if any(form is None for form in canonical):
        return False

    if len(set(canonical)) <= max_distinct:
        return True
    return len(tokens) <= max_tokens


class BackchannelAwareUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """Starts a user turn on real speech, but not on backchannels.

    Modelled on :class:`MinWordsUserTurnStartStrategy`, which it replaces: it
    keeps that strategy's word-count gate and adds the lexical one. Both apply
    only while the bot is speaking; when the bot is idle a single word starts a
    turn, exactly as before, so ordinary replies stay fast.
    """

    def __init__(
        self,
        *,
        min_words: int = 1,
        max_tokens: int = _MAX_BACKCHANNEL_TOKENS,
        max_distinct: int = _MAX_DISTINCT_BACKCHANNELS,
        use_interim: bool = True,
        **kwargs,
    ):
        """Initialize the strategy.

        Args:
            min_words: Words required to interrupt a speaking bot, applied only
                to utterances the lexicon does not recognise either way. 1
                (the default when suppression is on) disables it: the lexicon
                is a precise test where a word count is a crude proxy for the
                same thing, and running both only adds false negatives.
            max_tokens: Longest *varied* run of acknowledgements still treated
                as a backchannel. A repetitive run is not capped at all.
            max_distinct: Distinct acknowledgements still counted as repetition.
            use_interim: Also evaluate interim transcripts. Interims are what
                make barge-in feel immediate; the cost is that a partial of a
                longer sentence ("haan..." of "haan lekin ek problem hai") can
                look like a backchannel. It only defers the interruption — the
                next interim carrying "lekin" no longer matches and fires.
            **kwargs: Passed to :class:`BaseUserTurnStartStrategy`.
        """
        super().__init__(**kwargs)
        self._min_words = max(1, min_words)
        self._max_tokens = max_tokens
        self._max_distinct = max_distinct
        self._use_interim = use_interim
        self._bot_speaking = False
        self._suppressed = 0
        self._log = logger.bind(component="backchannel")

    @property
    def suppressed_count(self) -> int:
        """Backchannels that did not interrupt the bot, this call."""
        return self._suppressed

    async def handle_user_turn_started(self) -> None:
        """Reset for a new turn, matching MinWords' contract."""
        self._bot_speaking = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Decide whether this frame should open a user turn.

        Returns:
            STOP once a turn has been triggered, CONTINUE otherwise.
        """
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, TranscriptionFrame):
            return await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            return await self._handle_transcription(frame)

        return ProcessFrameResult.CONTINUE

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame
    ) -> ProcessFrameResult:
        text = frame.text
        words = len(text.split())

        if self._bot_speaking and not contains_interrupt_token(text):
            # Order matters. An explicit "stop"/"रुकिए" is handled above and
            # always takes the floor — it is frequently one word, so putting the
            # word gate first would swallow exactly the utterances that most
            # need to get through.
            if is_backchannel(
                text, max_tokens=self._max_tokens, max_distinct=self._max_distinct
            ):
                return await self._hold("backchannel", text, words)
            if words < self._min_words:
                return await self._hold("min_words", text, words)
        elif not words:
            return await self._hold("empty", text, words)

        await self.trigger_user_turn_started()
        return ProcessFrameResult.STOP

    async def _hold(self, reason: str, text: str, words: int) -> ProcessFrameResult:
        """Keep the floor with the bot and discard the aggregated text.

        Dropping the aggregation is what stops a suppressed "hmm" from being
        prepended to the caller's next real utterance — and it is what
        MinWordsUserTurnStartStrategy does in the same situation.
        """
        if reason == "backchannel":
            self._suppressed += 1
            # info, not debug: the default log level is INFO, and this event is
            # the only evidence in Grafana that suppression is doing anything.
            # Volume is a handful per call.
            self._log.info(
                "backchannel ignored, bot keeps talking",
                event="backchannel_suppressed",
                text=text,
                words=words,
            )
        await self.trigger_reset_aggregation()
        return ProcessFrameResult.CONTINUE
