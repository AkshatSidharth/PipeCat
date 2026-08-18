"""Static host + agent CRUD for the builder UI.

Runs alongside the Pipecat dev runner rather than inside it: the runner owns the
WebRTC signalling on :7860 and its FastAPI app has no extension point, so this
serves the console on :7861 and the browser talks to both. The runner ships
``allow_origins=["*"]``, so the cross-origin offer POST works without a proxy.

    python ui/server.py            # -> http://localhost:7861

Bind address defaults to loopback. This endpoint can write agent files, so do
not expose it publicly without putting auth in front of it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from voicebot.agents import (
    AgentConfig,
    agent_options,
    delete_agent,
    list_agents,
    load_agent,
    openai_models,
    preflight,
    save_agent,
)
from voicebot.config import get_settings

# The model list changes rarely; one network call per console session is plenty.
_llm_cache: dict[str, dict] = {}

UI_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Voicebot Console", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/options")
async def options() -> dict:
    """Dropdown choices, plus which credentials are actually present.

    The UI greys out services whose key is missing rather than letting you
    build an agent that cannot place a call.
    """
    settings = get_settings()
    return {
        **agent_options(),
        "credentials": {
            "soniox": bool(settings.soniox_api_key),
            "openai": bool(settings.openai_api_key),
            "sarvam": bool(settings.sarvam_api_key),
        },
        "defaults": AgentConfig(
            system_prompt=settings.system_prompt,
            greeting=settings.greeting,
        ).model_dump(),
    }


@app.get("/api/llm-models")
async def llm_models(refresh: bool = False, base_url: str = "") -> dict:
    """Chat models the given endpoint can reach, ranked for voice.

    Kept off ``/api/options`` so a slow or unreachable endpoint never stalls the
    builder — the UI renders with a static shortlist and swaps in the live list
    when this resolves.

    Args:
        refresh: Bypass the cache.
        base_url: OpenAI-compatible endpoint to query. Blank = api.openai.com.
            Cached per endpoint, because a self-hosted server lists only the one
            model it serves and must not be confused with the OpenAI list.
    """
    settings = get_settings()
    endpoint, credential = settings.model_copy(
        update={"llm_base_url": base_url or settings.llm_base_url}
    ).llm_endpoint()
    key = endpoint or ""
    if refresh or key not in _llm_cache:
        _llm_cache[key] = await run_in_threadpool(
            openai_models, credential, endpoint or ""
        )
    return _llm_cache[key]


@app.post("/api/preflight")
async def run_preflight(agent: AgentConfig) -> dict:
    """Make one real request to each service this agent is configured to use."""
    results = await run_in_threadpool(preflight, agent, get_settings())
    return {"results": results, "ok": all(r["ok"] for r in results)}


@app.get("/api/agents")
async def get_agents() -> list[dict]:
    """Every saved agent."""
    return [a.model_dump() for a in list_agents()]


@app.get("/api/agents/{agent_id}")
async def get_agent(agent_id: str) -> dict:
    """One saved agent."""
    agent = load_agent(agent_id)
    if agent is None:
        raise HTTPException(404, "no such agent")
    return agent.model_dump()


@app.put("/api/agents/{agent_id}")
async def put_agent(agent_id: str, agent: AgentConfig) -> dict:
    """Create or replace an agent. The path id wins over the body's."""
    agent.id = agent_id or agent.id
    return save_agent(agent).model_dump()


@app.post("/api/agents")
async def post_agent(agent: AgentConfig) -> dict:
    """Create an agent, deriving its id from the name."""
    return save_agent(agent).model_dump()


@app.delete("/api/agents/{agent_id}")
async def del_agent(agent_id: str) -> dict:
    """Delete an agent."""
    if not delete_agent(agent_id):
        raise HTTPException(404, "no such agent")
    return {"deleted": agent_id}


def _calls_root() -> Path:
    return Path(get_settings().analysis_dir)


def _tracks_for(folder: Path, stem: str, call_id: str | None) -> dict[str, Path]:
    """Locate a call's audio.

    Matches on the shared stem first. Calls recorded before the recorder and
    the analyser were made to share one get a second chance by call id, so
    existing recordings stay reviewable rather than appearing to have no audio.
    """
    found: dict[str, Path] = {}
    for track in ("mixed", "user", "bot"):
        exact = folder / f"{stem}-{track}.wav"
        if exact.exists():
            found[track] = exact
            continue
        if call_id:
            matches = sorted(folder.glob(f"*{call_id}-{track}.wav"))
            if matches:
                found[track] = matches[-1]
    return found


def _safe_stem(agent: str, stem: str) -> Path:
    """Resolve an agent/stem pair to a directory, refusing path traversal."""
    root = _calls_root().resolve()
    folder = (root / agent).resolve()
    if not str(folder).startswith(str(root)) or "/" in stem or "\\" in stem:
        raise HTTPException(400, "bad call id")
    return folder


@app.get("/api/calls")
async def list_calls(limit: int = 100) -> dict:
    """Past sessions, newest first, across every agent.

    Built by scanning the analysis files rather than keeping an index: the
    files are the source of truth, they are already named by agent and
    timestamp, and an index would be one more thing to get out of sync.
    """
    root = _calls_root()
    calls = []
    for path in root.glob("*/*-analysis.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stem = path.name[: -len("-analysis.json")]
        analysis = data.get("analysis") or {}
        measured = data.get("measured") or {}
        latency = measured.get("latency") or {}
        calls.append({
            "agent": path.parent.name,
            "stem": stem,
            "call_id": data.get("call_id"),
            "ended_at": data.get("ended_at"),
            "duration_secs": data.get("duration_secs"),
            "exchanges": measured.get("exchanges"),
            "disposition": analysis.get("disposition"),
            "sentiment": analysis.get("sentiment"),
            "resolved": analysis.get("resolved"),
            "pending": bool(analysis.get("pending")),
            "ended_by": measured.get("ended_by"),
            "v2v_p50": (latency.get("voice_to_voice") or {}).get("p50"),
            "has_audio": bool(_tracks_for(path.parent, stem, data.get("call_id"))),
        })
    calls.sort(key=lambda c: c["stem"], reverse=True)
    return {"calls": calls[:limit], "total": len(calls)}


@app.get("/api/calls/{agent}/{stem}")
async def get_call(agent: str, stem: str) -> dict:
    """The full report for one session: measured, inferred, and transcript."""
    path = _safe_stem(agent, stem) / f"{stem}-analysis.json"
    if not path.exists():
        raise HTTPException(404, "no such call")
    data = json.loads(path.read_text(encoding="utf-8"))
    tracks = _tracks_for(path.parent, stem, data.get("call_id"))
    data["audio"] = {
        track: f"/api/calls/{agent}/{stem}/audio/{track}" for track in tracks
    }
    return data


@app.get("/api/calls/{agent}/{stem}/audio/{track}")
async def get_call_audio(agent: str, stem: str, track: str) -> FileResponse:
    """Serve one recorded track. Range requests work, so seeking works."""
    if track not in ("mixed", "user", "bot"):
        raise HTTPException(400, "unknown track")
    folder = _safe_stem(agent, stem)
    report = folder / f"{stem}-analysis.json"
    call_id = None
    if report.exists():
        try:
            call_id = json.loads(report.read_text(encoding="utf-8")).get("call_id")
        except Exception:
            pass
    found = _tracks_for(folder, stem, call_id).get(track)
    if found is None:
        raise HTTPException(404, "no such recording")
    return FileResponse(found, media_type="audio/wav", filename=found.name)


@app.post("/api/tts-preview")
async def tts_preview(payload: dict) -> Response:
    """Synthesize text with an agent's *unsaved* TTS settings and return a WAV.

    Lets you hear the greeting, or any line, before dialling — using the exact
    provider, model, voice and speed the form currently holds, so what you hear
    is what the call would say. The audio is rendered fresh rather than served
    from the backchannel clip cache, which is keyed to a different purpose.
    """
    import io
    import wave

    from loguru import logger

    from voicebot.filler import render_filler

    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "no text to speak")
    if len(text) > 1000:
        raise HTTPException(400, "text too long for a preview (max 1000 chars)")

    agent_fields = {k: v for k, v in payload.items() if k != "text"}
    try:
        settings = AgentConfig(**agent_fields).apply_to(get_settings())
    except Exception as exc:
        raise HTTPException(400, f"invalid agent config: {exc}") from exc

    # Retries, because some engines drop short Devanagari intermittently — a
    # preview button that fails on a phrase which works two times in five is
    # worse than useless, it makes you distrust a phrase that is fine.
    try:
        pcm, rate = await render_filler(
            settings, text, logger.bind(component="preview")
        )
    except Exception as exc:
        raise HTTPException(502, f"{settings.tts_provider}: {exc}") from exc
    if len(pcm) == 0:
        raise HTTPException(
            502,
            f"{settings.tts_provider} returned no audio for that text after "
            "several attempts — try different wording.",
        )

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"X-Sample-Rate": str(rate), "X-Duration-Secs": f"{len(pcm)/rate:.2f}"},
    )


@app.post("/api/edit-prompt")
async def edit_prompt(payload: dict) -> dict:
    """Rewrite a system prompt from a plain-English instruction.

    Returns the **entire** revised prompt, never a diff or a description of the
    change — the response is written straight into the prompt box, so anything
    conversational would end up as bot instructions.
    """
    from voicebot.prompting import revise_prompt

    prompt = payload.get("prompt") or ""
    instruction = (payload.get("instruction") or "").strip()
    if not instruction:
        raise HTTPException(400, "describe the edit you want")

    agent_fields = {k: v for k, v in payload.items() if k not in ("prompt", "instruction")}
    try:
        settings = AgentConfig(**agent_fields).apply_to(get_settings())
    except Exception as exc:
        raise HTTPException(400, f"invalid agent config: {exc}") from exc

    revised, error = await run_in_threadpool(
        revise_prompt, settings, prompt, instruction
    )
    if error:
        raise HTTPException(502, error)
    return {"prompt": revised, "chars": len(revised), "was_chars": len(prompt)}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the console."""
    return FileResponse(UI_DIR / "index.html")


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


def main() -> None:
    """Run the console server."""
    parser = argparse.ArgumentParser(description="Voicebot builder console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    print(f"\n  Voicebot console  ->  http://{args.host}:{args.port}")
    print("  Bot runner expected on http://localhost:7860 (python bot.py -t webrtc)\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
