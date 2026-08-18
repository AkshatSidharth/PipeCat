"""Regression tests for backchannel handling.

The lexicon will get edited — someone will add a word for a new deployment. The
failure that matters is not "we missed a backchannel" (the bot interrupts, which
is merely the old behaviour); it is **a false positive**, where a word that
means "stop talking" lands in the backchannel set and the bot talks over the
caller with no error anywhere. The ``MUST_INTERRUPT`` cases below are the guard
against that, and they are the ones to extend when the lexicon grows.

Transcripts marked *(measured)* are literal Soniox ``stt-rt-v5`` output captured
with ``language_hints=["en","hi"]`` — the same config the bot runs — rather than
how the phrase is spelled.
"""

import asyncio

import numpy as np
import pytest

from voicebot.backchannel import (
    BackchannelAwareUserTurnStartStrategy,
    canonical_token,
    contains_interrupt_token,
    is_backchannel,
    normalize,
)
from voicebot.filler import FillerMixer

# --------------------------------------------------------------------------- #
# Matcher
# --------------------------------------------------------------------------- #

IS_BACKCHANNEL = [
    # measured Soniox output
    "हाँ।", "हाँ, हाँ।", "जी।", "जी, जी।", "अच्छा।", "ठीक है।", "बिल्कुल।",
    "हाँ जी।", "जी हाँ।", "समझ गया।", "सही है।", "हम्म।", "हुह।",
    "Okay.", "Uh-huh.", "Yeah.", "Okay, okay.", "Right, right.", "Yes, yes.",
    "I see.", "Got it.", "Sure, sure.",
    "Hanhan.",          # measured: "haan haan" merged into one token
    # variants the matcher should fold
    "haan haan", "ji ji", "hmmmm", "haaaan", "mhm", "achha", "theek hai", "ok",
    "हाँ हाँ", "HMM", "  Yeah.  ",
    # elongation, however it is spelled
    "haaaan", "hmmmmm", "हााा", "जीीी", "yesss", "okkk",
]

# Repetition is how Indian callers backchannel: the run goes on for as long as
# the other side keeps talking. Length must NOT be evidence of real speech here
# — the more it repeats, the more clearly it is a backchannel.
REPETITIVE = [
    "हाँ हाँ हाँ हाँ हाँ",
    "हाँ हाँ हाँ हाँ हाँ हाँ हाँ हाँ",
    "han haan haan haan hmm",
    "जी जी जी जी जी",
    "हाँ जी हाँ जी हाँ जी",
    "hmm hmm haan haan hmm haan",
    "yes yes yes yes yes yes yes",
    "ok ok ok ok ok",
    "हाँ हाँ हम्म हाँ हम्म हाँ हम्म",
]

# The bot serves Hindi and Indian English, but callers switch script freely.
OTHER_INDIC = [
    "হ্যাঁ হ্যাঁ",      # Bengali
    "હા હા જી",         # Gujarati
    "ਹਾਂ ਜੀ ਹਾਂ ਜੀ",     # Punjabi
    "சரி சரி",          # Tamil
    "అవును అవును",      # Telugu
    "ಹೌದು ಸರಿ",         # Kannada
    "ശരി ശരി",          # Malayalam
    "हो हो बरोबर",      # Marathi
    "ହଁ ହଁ",            # Odia
]

MUST_INTERRUPT = [
    # explicit floor claims — one word, so a min-words gate would swallow them
    "Stop.", "नहीं।", "रुकिए।", "no",
    # negation or correction
    "No, wait.", "Actually, no.", "नहीं, नहीं, यह गलत है।", "नहीं जी",
    # backchannel prefix, real content after
    "हाँ हाँ रुकिए", "haan lekin ek problem hai", "yes but I have a question",
    "ok stop", "जी मुझे एक शिकायत है",
    # ordinary speech
    "मेरा ऑर्डर कहाँ है", "kitna hai", "one minute", "hello",
    # repetition does not protect a real interruption hiding inside it
    "हाँ हाँ हाँ हाँ रुकिए",
    "haan haan haan lekin problem hai",
    "ok ok ok stop",
    "हाँ हाँ मेरा ऑर्डर कहाँ है",
    "नहीं नहीं नहीं",
    "yes yes but where is my order",
    # long AND varied: capped, on the theory that this is real speech the
    # lexicon happens to cover
    "हाँ जी बिल्कुल ठीक सही अच्छा ओके समझा",
]


@pytest.mark.parametrize("text", IS_BACKCHANNEL)
def test_backchannel_recognised(text):
    assert is_backchannel(text), f"{text!r} -> {normalize(text)!r}"


@pytest.mark.parametrize("text", REPETITIVE)
def test_repetition_is_never_evidence_of_real_speech(text):
    """A long repetitive run is *more* clearly a backchannel, not less."""
    assert is_backchannel(text), f"{text!r} -> {normalize(text)!r}"


@pytest.mark.parametrize("text", OTHER_INDIC)
def test_other_indian_scripts(text):
    assert is_backchannel(text), f"{text!r} -> {normalize(text)!r}"


@pytest.mark.parametrize("text", MUST_INTERRUPT)
def test_real_speech_is_never_suppressed(text):
    assert not is_backchannel(text), f"{text!r} -> {normalize(text)!r}"


def test_repeats_fold_before_counting_variety():
    """"haan"/"haaan"/"haanhaan" are one acknowledgement, not three."""
    assert canonical_token("haaan") == canonical_token("haan") == "haan"
    assert canonical_token("haanhaan") == "haan"
    assert canonical_token("kitna") is None


def test_normalize_preserves_devanagari_combining_marks():
    """The bug this guards: ``re.sub(r"[^\\w\\s]", ...)`` turns हाँ into ह.

    Combining marks are Mn/Mc and are not ``\\w``, so a regex-based strip
    silently destroys every Hindi backchannel and nothing ever matches.
    """
    assert normalize("हाँ, हाँ।") == "हाँ हाँ"
    assert normalize("जी।") == "जी"


def test_interrupt_tokens_beat_everything():
    assert contains_interrupt_token("हाँ हाँ रुकिए")
    assert contains_interrupt_token("Stop.")
    assert not contains_interrupt_token("हाँ जी")


def test_empty_text_does_not_take_the_floor():
    for text in ["", "   ", ".", "।", "!!!"]:
        assert is_backchannel(text)


# --------------------------------------------------------------------------- #
# Turn-start strategy
# --------------------------------------------------------------------------- #


async def _run(text: str, *, bot_speaking: bool) -> bool:
    """Feed one transcript to the strategy; return whether a turn started."""
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        TranscriptionFrame,
    )
    from pipecat.utils.asyncio.task_manager import TaskManager

    strategy = BackchannelAwareUserTurnStartStrategy(min_words=1)
    await strategy.setup(TaskManager(loop=asyncio.get_running_loop()))
    started: list = []
    strategy.add_event_handler("on_user_turn_started", lambda _s, p: started.append(p))

    await strategy.process_frame(
        BotStartedSpeakingFrame() if bot_speaking else BotStoppedSpeakingFrame()
    )
    await strategy.process_frame(
        TranscriptionFrame(text=text, user_id="u", timestamp="t")
    )
    return bool(started)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    ["हाँ।", "जी, जी।", "Hmm.", "Uh-huh.", "Hanhan.",
     "हाँ हाँ हाँ हाँ हाँ हाँ", "han haan haan haan hmm"],
)
async def test_backchannel_does_not_interrupt_a_speaking_bot(text):
    assert not await _run(text, bot_speaking=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["हाँ।", "जी।", "Yeah.", "Okay."])
async def test_same_words_answer_a_silent_bot(text):
    """The rule the whole design rests on: context decides, not the word."""
    assert await _run(text, bot_speaking=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Stop.", "नहीं।", "रुकिए।", "मेरा ऑर्डर कहाँ है"])
async def test_real_interruptions_still_work(text):
    assert await _run(text, bot_speaking=True)


# --------------------------------------------------------------------------- #
# Mixer
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_mixer_is_a_passthrough_until_it_has_clips():
    mixer = FillerMixer()
    await mixer.start(16000)
    silence = b"\x00" * 1280
    assert not mixer.play("neutral")        # nothing loaded yet
    assert await mixer.mix(silence) == silence


@pytest.mark.asyncio
async def test_mixer_preserves_chunk_size_and_clips_rather_than_wrapping():
    mixer = FillerMixer()
    await mixer.start(16000)
    mixer.set_clips({"neutral": [np.full(640, 20000, dtype=np.int16)]})
    assert mixer.play("neutral")

    loud = np.full(640, 30000, dtype=np.int16).tobytes()
    out = np.frombuffer(await mixer.mix(loud), dtype=np.int16)
    assert len(out) == 640                  # transport requires exact chunk sizes
    assert out.max() == 32767               # saturated, not wrapped to negative
    assert out.min() >= 0


@pytest.mark.asyncio
async def test_mixer_plays_one_clip_at_a_time_and_finishes():
    mixer = FillerMixer()
    await mixer.start(16000)
    mixer.set_clips({"neutral": [np.full(1600, 5000, dtype=np.int16)]})  # 100ms
    assert mixer.play("neutral")
    assert not mixer.play("neutral")        # already busy

    chunks = 0
    while mixer.busy and chunks < 100:
        await mixer.mix(b"\x00" * 320)      # 10ms
        chunks += 1
    assert 9 <= chunks <= 11                # ~100ms of audio
    assert mixer.played == 1


@pytest.mark.asyncio
async def test_mixer_never_raises_on_bad_input():
    """mix() runs on every outgoing chunk — a raise here corrupts all audio."""
    mixer = FillerMixer()
    await mixer.start(16000)
    mixer.set_clips({"neutral": [np.full(160, 100, dtype=np.int16)]})
    mixer.play("neutral")
    assert await mixer.mix(b"\x00\x00\x00") is not None   # odd byte count
