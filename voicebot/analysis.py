"""Post-call analysis: what happened, how it felt, and where the time went.

Two halves, deliberately separated:

**Measured.** Latency percentiles per component, how long the caller took to
answer, interruption and backchannel counts, token spend. These come from the
observer's own record of the call and are arithmetic — they are always right,
they cost nothing, and they are written even when the LLM half fails.

**Inferred.** Disposition, sentiment, summary, action items. These come from an
LLM reading the transcript, and are marked as such in the report. Treat them the
way you would a junior colleague's call notes: useful, not evidence.

Why p50/p90 rather than a mean
------------------------------
A mean hides the shape. One 8-second reply among nine fast ones averages out to
something that looks acceptable, but the caller remembers the 8 seconds. p90 is
what they complain about, so it is reported alongside the median and the worst
case.

The analysis runs *after* the call has ended, so its own latency does not matter
and it never competes with the call for the GPU. It is written next to the
recording, under the same agent directory and timestamp, so a call's audio and
its analysis sit together.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

# What the model is asked to return. Kept small and closed-vocabulary where it
# can be: a free-text disposition is unusable for reporting, and an open
# sentiment scale invites false precision.
DISPOSITIONS = (
    "resolved",
    "information_provided",
    "callback_requested",
    "transferred",
    "not_interested",
    "wrong_number",
    "no_response",
    "caller_hung_up",
    "bot_failure",
    "incomplete",
)

SENTIMENTS = ("positive", "neutral", "frustrated", "angry", "confused")

_SYSTEM = """You analyse completed customer-support phone calls.

You will be given a transcript. Reply with ONE JSON object and nothing else:

{
  "disposition": one of %s,
  "sentiment": one of %s,
  "sentiment_reason": "<one short sentence citing the caller's own words>",
  "summary": "<2-3 sentences: why they called and what happened>",
  "key_points": ["<short bullet>", ...],
  "action_items": ["<what a human should do next>", ...],
  "caller_intent": "<what the caller actually wanted, in a few words>",
  "resolved": true | false,
  "bot_issues": ["<anything the BOT did wrong: talked over the caller, did not understand, wrong information, dead air>", ...]
}

Rules:
- Judge only from the transcript. Do not invent detail.
- "no_response" is for a caller who never meaningfully spoke.
- "bot_failure" is when the bot broke down, not when the caller declined.
- bot_issues may be empty. Be specific and short.
- Write summary and key_points in English even when the call is in Hindi.""" % (
    list(DISPOSITIONS),
    list(SENTIMENTS),
)


def _pct(values: list[float], q: float) -> float | None:
    """Percentile without numpy, nearest-rank."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[index], 3)


def _stats(values: list[float]) -> dict[str, Any]:
    """Median / p90 / worst, plus n. Empty input gives a well-formed blank."""
    if not values:
        return {"n": 0, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "n": len(values),
        "p50": _pct(values, 0.50),
        "p90": _pct(values, 0.90),
        "max": round(max(values), 3),
        "mean": round(statistics.fmean(values), 3),
    }


def measure(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Turn the observer's raw record into the measured half of the report."""
    exchanges = snapshot.get("exchanges", [])

    def column(key: str) -> list[float]:
        return [e[key] for e in exchanges if e.get(key) is not None]

    voice_to_voice = column("voice_to_voice_secs")
    budget = snapshot.get("latency_budget_secs") or 0

    components = {
        name: _stats(values)
        for name, values in sorted(snapshot.get("ttfb_secs", {}).items())
    }

    return {
        "exchanges": len(exchanges),
        "latency": {
            # What the caller feels: their speech ending to the bot speaking.
            "voice_to_voice": _stats(voice_to_voice),
            # VAD + smart turn + STT finalize.
            "turn_detection": _stats(column("turn_detection_secs")),
            # LLM + TTS.
            "response": _stats(column("response_secs")),
            "budget_secs": budget,
            "over_budget": sum(1 for e in exchanges if e.get("over_budget")),
            "over_budget_pct": (
                round(100 * sum(1 for e in exchanges if e.get("over_budget")) / len(exchanges))
                if exchanges
                else None
            ),
        },
        # Per-service time-to-first-byte, so a slow call can be attributed.
        "component_ttfb": components,
        # How quickly the caller answered the bot.
        "user_response": _stats(snapshot.get("user_response_secs", [])),
        "tokens": snapshot.get("tokens", {}),
        "interruptions": snapshot.get("interruptions", 0),
        "backchannels_suppressed": snapshot.get("backchannels_suppressed", 0),
        "reengagement_prompts": snapshot.get("reengagement_prompts", 0),
        # Who hung up. A bot-ended call reached its closing line; a
        # caller-ended one may have been abandoned.
        "ended_by": "bot" if snapshot.get("ended_by_bot") else "caller",
    }


def format_transcript(transcript: list[dict[str, Any]]) -> str:
    """Render the transcript for the model (and for a human reading the file)."""
    lines = []
    for turn in transcript:
        who = "CALLER" if turn.get("role") == "user" else "BOT"
        lines.append(f"[{turn.get('at_secs', 0):>6.1f}s] {who}: {turn.get('text', '')}")
    return "\n".join(lines)


def interpret(
    settings, transcript: list[dict[str, Any]], timeout: float = 60.0
) -> dict[str, Any]:
    """Ask the LLM for disposition, sentiment and a summary.

    Never raises: a failed analysis returns an ``error`` key and the measured
    half of the report is written regardless. Losing the notes is annoying;
    losing the latency record because the notes failed would be worse.
    """
    log = logger.bind(component="analysis")
    if not transcript:
        return {"skipped": "no transcript"}

    base_url, api_key = settings.llm_endpoint()
    if not api_key:
        return {"error": "no LLM API key configured"}

    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/")
    try:
        response = httpx.post(
            f"{endpoint}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": settings.analysis_model or settings.llm_model,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": format_transcript(transcript)},
                ],
                # Roomy: this runs after the call, so tokens cost time nobody
                # is waiting on.
                "max_completion_tokens": 900,
                "temperature": 0,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        log.warning(f"call analysis failed: {exc}")
        return {"error": str(exc)[:200]}

    # Models wrap JSON in prose or fences often enough to be worth handling.
    text = content.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {"error": "model did not return JSON", "raw": content[:400]}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return {"error": f"unparseable JSON: {exc}", "raw": content[:400]}


def analyze_call(
    *,
    settings,
    snapshot: dict[str, Any],
    call_id: str,
    agent: str,
    duration_secs: float,
    output_dir: Path,
    recordings: dict[str, str] | None = None,
    stem: str | None = None,
) -> dict[str, Any]:
    """Build and write the full post-call report.

    Returns:
        The report. Also written as JSON, and as a readable ``.txt`` summary,
        beside the call's recordings.
    """
    from voicebot.agents import slugify

    log = logger.bind(component="analysis")
    transcript = snapshot.get("transcript", [])

    report: dict[str, Any] = {
        "call_id": call_id,
        "agent": agent,
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_secs": round(duration_secs, 2),
        "measured": measure(snapshot),
        "transcript": transcript,
        "recordings": recordings or {},
        "analysis": {"pending": "LLM summary in progress"},
    }

    folder = output_dir / (slugify(agent) or "default")
    folder.mkdir(parents=True, exist_ok=True)
    # Reuse the recorder's stem so audio and analysis pair up; only invent one
    # when recording was off.
    stem = stem or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{call_id}"

    def write() -> None:
        try:
            (folder / f"{stem}-analysis.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (folder / f"{stem}-analysis.txt").write_text(
                render(report), encoding="utf-8"
            )
        except Exception as exc:
            log.warning(f"could not write call analysis: {exc}")

    # Written in two passes on purpose. The measured half is arithmetic — it is
    # ready immediately and costs nothing. The LLM half takes 30-60s on a local
    # 30B, and anything that kills the process in that window (a restart, a
    # deploy, a crash) would otherwise take the latency data and the transcript
    # with it. Two calls' analytics were lost exactly this way. So the report
    # lands complete-but-for-the-summary first, then gets rewritten.
    write()

    if settings.analysis_enabled:
        log.info(
            "analysing call",
            event="call_analysis_started",
            call_id=call_id,
            transcript_turns=len(transcript),
        )
        report["analysis"] = interpret(settings, transcript)
        write()
    else:
        report["analysis"] = {"skipped": "disabled"}
        write()

    a = report["analysis"]
    log.info(
        "call analysed",
        event="call_analysis",
        call_id=call_id,
        agent=agent,
        disposition=a.get("disposition"),
        sentiment=a.get("sentiment"),
        resolved=a.get("resolved"),
        exchanges=report["measured"]["exchanges"],
        v2v_p50=report["measured"]["latency"]["voice_to_voice"]["p50"],
        v2v_p90=report["measured"]["latency"]["voice_to_voice"]["p90"],
    )
    return report


def render(report: dict[str, Any]) -> str:
    """A plain-text report you can read without a JSON viewer."""
    m = report["measured"]
    a = report.get("analysis", {})
    lat = m["latency"]

    def row(label: str, s: dict[str, Any], unit: str = "s") -> str:
        if not s or not s.get("n"):
            return f"  {label:<22} —"
        return (
            f"  {label:<22} p50 {s['p50']:>6.2f}{unit}   p90 {s['p90']:>6.2f}{unit}"
            f"   max {s['max']:>6.2f}{unit}   (n={s['n']})"
        )

    out = [
        "=" * 68,
        f"CALL {report['call_id']}   agent={report['agent']}",
        f"{report['ended_at']}   duration {report['duration_secs']}s"
        f"   exchanges {m['exchanges']}",
        "=" * 68,
        "",
        "OUTCOME" + ("  (inferred by LLM — not evidence)" if a.get("disposition") else ""),
        f"  disposition          {a.get('disposition', '—')}",
        f"  resolved             {a.get('resolved', '—')}",
        f"  sentiment            {a.get('sentiment', '—')}"
        + (f"  ({a['sentiment_reason']})" if a.get("sentiment_reason") else ""),
        f"  caller intent        {a.get('caller_intent', '—')}",
        "",
        "SUMMARY",
        f"  {a.get('summary', a.get('error') or a.get('skipped') or a.get('pending') or '—')}",
        "",
    ]
    for title, key in (("KEY POINTS", "key_points"), ("ACTION ITEMS", "action_items"),
                       ("BOT ISSUES", "bot_issues")):
        items = a.get(key) or []
        if items:
            out.append(title)
            out += [f"  - {i}" for i in items]
            out.append("")

    out += [
        "LATENCY (measured)",
        row("voice-to-voice", lat["voice_to_voice"]),
        row("  turn detection", lat["turn_detection"]),
        row("  response (LLM+TTS)", lat["response"]),
        f"  over budget            {lat['over_budget']}/{m['exchanges']}"
        f" exchanges above {lat['budget_secs']}s"
        + (f" ({lat['over_budget_pct']}%)" if lat["over_budget_pct"] is not None else ""),
        "",
        "COMPONENT TTFB (measured)",
    ]
    out += [row(name, s) for name, s in m["component_ttfb"].items()] or ["  —"]
    out += [
        "",
        "CALLER",
        row("response time", m["user_response"]),
        f"  interruptions          {m['interruptions']}",
        f"  backchannels ignored   {m['backchannels_suppressed']}",
        f"  re-engagement prompts  {m['reengagement_prompts']}",
        f"  ended by               {m['ended_by']}",
        "",
        "TOKENS",
        f"  prompt {m['tokens'].get('prompt', 0)}   completion {m['tokens'].get('completion', 0)}",
        "",
        "TRANSCRIPT",
        format_transcript(report.get("transcript", [])) or "  —",
    ]
    return "\n".join(out)
