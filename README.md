# Pipecat Voicebot — low-latency, with Grafana/Loki observability

An end-to-end voice agent built on **Pipecat 1.7.0**, tuned for a sub-500ms
voice-to-voice budget and wired for production debugging from day one:

- **Smart turn detection** — the local `smart-turn-v3` ONNX model decides when
  the user actually finished speaking instead of guessing from a silence timer.
- **Tunable interruption (barge-in)** handling, including a word-count gate that
  stops stray "mhm"s from cutting the bot off.
- **Call recording** — stereo (user left / bot right) plus isolated per-speaker
  tracks, streamed to disk so memory stays flat on long calls.
- **A latency budget you can actually see** — voice-to-voice split into its two
  additive halves, with a per-response budget counter.
- **Prometheus metrics** — TTFB, TTFA, tokens, turn-detection confidence,
  interruptions, errors.
- **Structured logs shipped to Loki**, correlated per call with `call_id`.
- **A provisioned Grafana dashboard** — 32 panels, no clicking required.

| Layer     | Service                                     | Why this one                        |
| --------- | ------------------------------------------- | ----------------------------------- |
| Transport | Twilio / Telnyx / Plivo / Exotel (inbound PSTN), Daily, SmallWebRTC (browser) | |
| STT       | Soniox `stt-rt-v5`                          | 0.35s P99 speech-end→final transcript — fastest tier Pipecat benchmarks |
| LLM       | OpenAI `gpt-4.1-nano`                       | OpenAI's fastest text model, and **no reasoning** — see below |
| TTS       | Sarvam `bulbul:v2` (websocket)              | Streaming service, `min_buffer_size=30`. Also: ElevenLabs `eleven_flash_v2_5`, OmniVoice — see [TTS providers](#tts-providers) |
| VAD       | Silero (`stop_secs=0.15`)                   | Triggers the turn model; not the decider |
| Turn      | `LocalSmartTurnAnalyzerV3` (bundled ONNX)   | On-device, ~15-40ms, no network hop  |

> **Do not swap the LLM for a reasoning model.** o-series and gpt-5.x models
> spend hundreds of milliseconds thinking before the first token — a cost a
> voice bot cannot absorb. If `gpt-4.1-nano` is not smart enough, step up to
> `gpt-4.1-mini`, not to a reasoning model.

---

## The console — build an agent, then call it

```bash
./run.sh              # bot + console
./run.sh --all        # also brings up Grafana/Loki/Prometheus
```

Open **<http://localhost:7861>**. Two tabs:

**Build** — every knob, in one form: STT model and languages, LLM and service
tier, Sarvam model / voice / opening language, VAD and turn-taking, barge-in,
the system prompt and greeting, latency budget, recording. Save it and it lands
in `agents/<id>.json` — plain JSON that diffs cleanly and can be hand-edited or
committed. The sidebar lists saved agents; the credential lights show which
services actually have a key, so you can't build an agent that cannot dial.

**Test call** — press **Start call** and talk to the agent you just saved. It
shows live transcript, who's speaking, barge-ins, per-service TTFB, and the
**voice-to-voice latency broken into turn-detection and response** — measured
in the browser using the same definition as the Grafana metric. **Save & test
call** does both in one click.

Dropdowns are populated from the installed Pipecat package (Sarvam's own model
and speaker enums, its language map), so the UI cannot drift from what the
services accept. **The OpenAI list is fetched live from your own account** at
`GET /api/llm-models` — no hardcoded list to go stale — filtered to chat models
and grouped fastest-first:

| Group | Contains | Why |
| --- | --- | --- |
| **Fastest — best for voice** | `*-nano` | Smallest, quickest first token |
| **Balanced** | `*-mini`, `*-chat-latest` | `chat-latest` is the non-reasoning tuning of each GPT-5 family |
| **Slower first token** | full-size models | Fine for quality, costly in latency |
| **Avoid for voice** | `*-pro`, `o1`/`o3`/`o4` | Reasoning models — hundreds of ms before the first token |

The grouping is a **naming heuristic, not a benchmark** — OpenAI publishes no
latency figures through the API. It steers you away from models that are
structurally wrong for voice; the console's TTFB panel gives you the real
number. The free-text field beside the picker accepts any model id, and the
whole list falls back to a static shortlist when OpenAI is unreachable so the
builder still works offline.

### How an agent reaches the bot

An agent is a **diff against your `.env`**, not a replacement. Anything left
blank in the builder keeps its environment value — so a blank prompt falls back
to `prompts/system.txt` rather than blanking the bot.

```
console  ──POST /api/offer  request_data:{agent:"kapture-support"}──▶  bot runner
                                                                        │
                                        runner_args.body ──▶ load agents/kapture-support.json
                                                          ──▶ AgentConfig.apply_to(Settings)
```

Because it rides the offer's `request_data`, the same mechanism works for
inbound telephony: put `{"agent": "<id>"}` in your provider's webhook body and
that number gets that agent.

---

## Where everything goes

### 🔑 API keys → `.env`

```bash
cp .env.example .env
```

Then fill in the three at the very top. Nothing else is needed to make a call:

```bash
SONIOX_API_KEY=...      # STT   — console.soniox.com
OPENAI_API_KEY=sk-...   # LLM   — platform.openai.com/api-keys
SARVAM_API_KEY=...      # TTS   — dashboard.sarvam.ai
```

ElevenLabs and OmniVoice are optional alternatives to Sarvam:

```bash
ELEVENLABS_API_KEY=sk_...                          # only if TTS_PROVIDER=elevenlabs
VOICEBOT_OMNIVOICE_URL=ws://172.16.1.4:80/omnivoice-tts   # only if TTS_PROVIDER=omnivoice
VOICEBOT_OMNIVOICE_VOICE_ID=Anika
```

For **inbound phone calls**, also fill in the block for your telephony provider
(section 1 of `.env.example`) — Pipecat's runner reads these directly to build
the frame serializer:

| Provider | Variables |
| -------- | --------- |
| Twilio   | `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` |
| Telnyx   | `TELNYX_API_KEY` |
| Plivo    | `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN` |
| Exotel   | *(none needed)* |
| Daily    | `DAILY_API_KEY` |

`.env` is gitignored. Keys are never logged — the observability layer records
model names and token counts, never credentials.

### 💬 Greeting and system prompt → the console, or `prompts/`

Either edit them per-agent in the **Build** tab (saved into the agent's JSON),
or edit the fallback files that every agent inherits when it leaves them blank:

```
prompts/system.txt      the assistant's persona, tone and rules
prompts/greeting.txt    what it says the moment it picks up
agents/<id>.json        per-agent overrides of both
```

`greeting.txt` is an **instruction to the model**, not a fixed script — so the
greeting comes out in the assistant's own voice and stays in conversation
context. Write "Greet the caller warmly and ask how you can help", not
"Hello, how can I help you?".

Both can be overridden per-environment with `VOICEBOT_SYSTEM_PROMPT` /
`VOICEBOT_GREETING` in `.env`, which take precedence over the files.

---

## The latency budget — read this first

**Voice-to-voice** is the only latency number a caller feels: from the moment
they stop speaking to the moment they hear audio. It splits cleanly in two, and
the dashboard graphs all three lines together:

```
voice_to_voice  =  turn_detection  +  response
                   (VAD wait +        (LLM TTFB +
                    smart turn +       TTS TTFB)
                    STT finalize)
```

Realistic per-stage cost on this stack, at the shipped settings:

| Stage                                    | Typical  | Notes |
| ---------------------------------------- | -------- | ----- |
| VAD silence wait                         | **150ms** | `vad_stop_secs` — pure dead air, the biggest single lever |
| Smart turn inference                     | 15–40ms  | measured; on-device ONNX |
| Soniox finalize (beyond the VAD wait)    | 50–150ms | Pipecat measures Soniox at 0.35s P99 *including* the VAD wait |
| → **turn detection**                     | **220–340ms** | |
| `gpt-4.1-nano` TTFB                      | ~120–250ms | no reasoning; a reasoning model would add 300ms+ here |
| Sarvam `bulbul:v2` TTFB                  | *unmeasured* | Sarvam publishes no P99; read it off the **TTS time-to-first-audio** panel on your own traffic |
| → **response**                           | **≈200–400ms** | |
| **voice-to-voice total**                 | **≈420–740ms** | |

Soniox and the VAD wait are the two numbers here with a published or measured
basis. The Sarvam figure is deliberately blank rather than guessed — the
dashboard will tell you within one call, and it moves with `min_buffer_size`.

**Be clear-eyed about what "sub-500ms" means here.** A cascaded
STT→LLM→TTS pipeline can hit sub-500ms at **p50** with these settings, and this
config is tuned to do that. It will **not** hold sub-500ms at p95: the tail is
dominated by smart turn falling back to its silence ceiling and by LLM TTFB
variance, neither of which config can remove. If you need sub-500ms *at p95*,
the architecture has to change — a speech-to-speech model (Pipecat ships
`pipecat.services.google.gemini_live`) removes the STT and TTS hops entirely and
is the only reliable way there.

Track it honestly rather than guessing: the **Latency budget** row shows p50/p95/p99
and the `voicebot_latency_budget_exceeded_total` counter, and every response logs
its own breakdown:

```logql
{service="pipecat-voicebot"} | json | event="voice_to_voice_latency" | over_budget="true"
```

### If you are over budget, in priority order

1. **Confirm the LLM is not a reasoning model.** `gpt-4.1-nano` is the default
   for exactly this reason. Swapping in an o-series or gpt-5.x model is the
   single most expensive mistake available on this stack.
2. **Lower `VOICEBOT_VAD_STOP_SECS`.** 0.15 → 0.10 buys a flat 50ms. Smart turn
   v3 makes the real decision, so aggressive VAD is safer here than on a
   timer-only setup. Below ~0.1 it starts firing on mid-sentence pauses.
3. **Lower `VOICEBOT_SARVAM_MIN_BUFFER_SIZE`.** Sarvam holds this many
   characters before it starts synthesizing (its own default is 50, we ship
   30). Dropping it starts audio sooner at some cost to prosody.
4. **Shorten replies.** `VOICEBOT_LLM_MAX_TOKENS` and a system prompt that asks
   for a short *first sentence* — in `sentence` aggregation mode, TTS cannot
   start until that first sentence ends. `prompts/system.txt` already asks for
   this; keep it if you rewrite the prompt.
5. **Try `VOICEBOT_OPENAI_SERVICE_TIER=priority`** if your OpenAI account has
   the add-on. It lowers and tightens TTFB. Accounts without it get an error,
   which is why it ships unset.
6. **A/B `VOICEBOT_TTS_TEXT_AGGREGATION`.** See below.
7. **Co-locate.** Region mismatch between your host and Soniox/OpenAI/Sarvam
   silently adds tens of ms per hop, three hops deep. Sarvam is India-hosted.

### TTS providers

Three are selectable per agent (`VOICEBOT_TTS_PROVIDER`, or the builder's
**Text to speech → Provider**).

| Provider | Model / voice | First audio | Use it when |
| --- | --- | --- | --- |
| **Sarvam** *(default)* | `bulbul:v2` | read it off the dashboard | Indic-first, lowest latency of the three |
| **ElevenLabs** | `eleven_flash_v2_5` | ~150–300ms typical | 32 languages, best prosody |
| **OmniVoice** | voice-clone alias | **~450ms floor** | you need a *cloned* voice |

#### OmniVoice

A self-hosted FlowTTS deployment reached over its own websocket protocol
(`voicebot/omnivoice.py`). Two things about it are worth knowing before you
pick it.

**It has a ~450ms floor on first audio, and shortening the text does not help.**
Measured on a warm socket against this deployment:

| text | chars | first audio |
| --- | --- | --- |
| `"Sure."` | 5 | 447ms |
| `"One moment please."` | 18 | 449ms |
| `"Your order will arrive tomorrow by evening."` | 43 | 454ms |
| a 149-char sentence | 149 | 1043ms |

~420ms of that is the server's own reported `decoder_ttft_ms` — a floor no
client-side change moves. It is flat up to roughly a sentence, then grows. That
is most of a sub-500ms voice-to-voice budget spent in TTS alone, so **choose
OmniVoice for voice cloning, not for speed.** Sarvam or ElevenLabs Flash are
the latency picks.

Two implementation choices follow from this:

- **The socket is held open for the whole call.** A cold connection measured
  3.2s versus ~450ms warm, so the service pre-connects on `StartFrame` and
  reuses one socket per call.
- **Barge-in filters on `text_id` instead of reconnecting.** When a turn is
  interrupted the server keeps sending chunks for the old request, which would
  otherwise be picked up as the *next* turn's audio. Every frame carries the
  `text_id` it belongs to, so stale audio is dropped without paying reconnect
  cost on every interruption.

**An unknown voice alias does not error.** The server silently falls back to
OmniVoice's un-cloned auto voice, so a typo in `VOICEBOT_OMNIVOICE_VOICE_ID` is
inaudible as a *failure* — it just is not the voice you asked for. Preflight
says so rather than implying the alias is confirmed. The aliases below were
verified present on the deployment by synthesizing the same Hindi sentence
through each and fingerprinting median F0 against the fallback:

| alias | F0 | |
| --- | --- | --- |
| `Anika` *(default)* | 250 Hz | Indian female — clear, warm, professional |
| `niharika` | 222 Hz | Indian female |
| `saavi` | 217 Hz | Indian female |
| `monisha` | 192 Hz | Indian female |
| `bench-hindi_female` | 134 Hz | Hindi female |
| `hausa-female-1` | 152 Hz | **Nigerian** — the reference notebook's example |
| *(unknown alias)* | ~163–181 Hz, unstable | the un-cloned fallback |

The default was changed from the notebook's `hausa-female-1`, which is a real
Nigerian clone and would have voiced Hindi in a Hausa accent.

> The model is **non-deterministic** — the same text and voice produce
> different audio each call (three runs measured 116640 / 111840 / 114720
> bytes). Do not assert on byte-exact output in tests.

### The TTS aggregation trade-off

Two things gate when audio starts, and they interact — which is why this is a
setting rather than a default to flip blindly:

| Mode | What happens |
| ---- | ------------ |
| `sentence` *(default)* | Pipecat waits for a sentence boundary (~200–300ms) before handing text to Sarvam |
| `token` | Pipecat streams tokens as they arrive; Sarvam's own `min_buffer_size` then decides when synthesis starts |

In `token` mode `min_buffer_size` becomes the real control, so tune the pair
together and read the result off the **TTS time-to-first-audio** panel.

---

## Self-hosted stack (101.53.139.195)

Both the TTS and the LLM run on one L40S box, behind nginx on port 80.

| | endpoint |
| --- | --- |
| TTS synthesis (ws) | `ws://101.53.139.195/omnivoice-tts/ws/<call_id>` |
| Voice clone / list | `http://101.53.139.195/omnivoice-ctrl/voices` |
| LLM (OpenAI-compatible) | `http://101.53.139.195/v1/chat/completions` |
| Health | `http://101.53.139.195/healthz` |

The LLM needs `Authorization: Bearer <key>`; the key is on the box at
`/home/jovyan/voicestack/etc/llm_api_key`. TTS has **no auth** — see the warning
below.

### The one thing to know about this host

It is a **containerd container, not a VM.** Only `/home/jovyan` (a Ceph RBD
volume) survives; the root overlay — `/opt`, `/etc`, `/usr`, and anything `apt`
installed, nginx included — is wiped whenever the container is recreated. That
is not a hypothetical: it happened twice mid-deploy and took a 52 GB download
with it.

So everything lives under `/home/jovyan/voicestack` (plus `/home/jovyan/FlowTTS`),
and after any container restart:

```bash
ssh root@101.53.139.195 'bash /home/jovyan/voicestack/bin/bootstrap.sh'
```

That reinstalls nginx if missing, re-copies the config out of persistent
storage, and starts both services. It is idempotent.

Also: background jobs need `setsid`, not just `nohup` — a plain `nohup … &` is
killed when the SSH session ends.

### TTS — FlowTTS / OmniVoice

All 24 shipped voice clones are loaded. Aliases are **lowercased** by the
registry (`anika`, not `Anika`), and an unknown alias does not error — it
silently falls back to the un-cloned auto voice, so a typo is inaudible as a
failure. The clones live in `voices/voice-npzs/`, a *subdirectory*, and the
registry globs `*.npz` at the top level only; `FLOWTTS_VOICES__VOICES_DIR` has
to point at the subdirectory or nothing loads at all.

Measured TTFB on this box: **297–399 ms**, against 447–540 ms on the old
deployment.

Cloning a new voice (`ref_text` must be the exact transcript of the clip):

```bash
curl -X POST http://101.53.139.195/omnivoice-ctrl/voices \
  -F voice_id=my_voice -F preferred_lang=hi \
  -F ref_text="<exact words spoken in the clip>" \
  -F audio=@clip.wav
```

New voices are usable immediately, no restart.

### LLM — Muse-Glimmer-30B

Served by SGLang. Three things were needed to make it work at all:

1. **SGLang from git main.** `muse_glimmer` support landed 2026-08-11 and is in
   no tagged release. The build also needs Rust (`cargo`) and `ninja` on PATH —
   without ninja it loads the weights fine and then dies during CUDA-graph
   capture.
2. **The pre-quantized FP8 checkpoint** (`RedHatAI/Muse-Glimmer-30B-FP8-block`,
   34.4 GB). The bf16 original is 59.6 GB and does not fit a 46 GB card that is
   also hosting TTS. On-the-fly `--quantization fp8` does *not* work around this
   — it OOMs allocating the FP8 weights, because it needs the bf16 tensors
   resident to convert them. NVFP4 builds exist but need Blackwell; L40S is Ada.
3. **The chat template**, copied from the base repo — the FP8 repo ships none at
   all. Without it the model emits its raw ATEM channel markers (`to=self`,
   visible chain-of-thought) directly into the reply.

#### Reasoning is the latency story

This is a reasoning model, and by default that is fatal for voice. Measured time
to the first token of *user-facing* text:

| `reasoning_strength` | first spoken token |
| --- | --- |
| `high` (the default) | **10.0 s** |
| `medium` | 9.8 s |
| `low` | 6.4 s |
| `none` / `minimal` | 6.8 s / 8.3 s |

Turning the knob down barely helps — the model reasons regardless. What does
work is opening the assistant turn directly on the user channel
(`<|start|>assistant to=user<|message|>`) so the reasoning channel is never
entered. That is what `chat_template_voice.jinja` does, and it is what the
server runs:

| | first token | full reply |
| --- | --- | --- |
| stock template | 6.4–10 s | 8–11 s |
| **voice template** | **139–196 ms** | 626 ms – 2.2 s |

Swap `--chat-template` back to `chat_template.jinja` in
`bin/start_llm.sh` when you want the model to reason.

**Be clear-eyed about the budget.** Decode runs at ~20 tok/s — a dense 30B in
FP8 against the L40S's ~864 GB/s is memory-bandwidth bound, and no amount of
tuning changes that. A 9-token reply takes 626 ms, a 38-token reply 2.2 s. So
this stack does **not** hit the sub-500 ms voice-to-voice target that
`gpt-4.1-nano` was chosen for; realistic voice-to-voice here is **1.5–3 s**.
It is good for testing the self-hosted path end to end. If you want both local
*and* fast, the answer is a smaller model (3–8B would run 60–100 tok/s on this
card), not a different serving flag.

### Choosing it per agent

The endpoint is an **agent** setting, not a deployment-wide one. In the console,
**Language model → Provider**:

| | |
| --- | --- |
| `OpenAI (cloud)` | `api.openai.com`, keyed by `OPENAI_API_KEY` |
| `Self-hosted — Muse-Glimmer 30B` | fills in `http://101.53.139.195/v1` and the model id |
| `Custom OpenAI-compatible` | any server; you supply the base URL |

Picking a provider fills in its base URL and model; the model dropdown then
lists what *that* endpoint actually serves, so a self-hosted box shows
`muse-glimmer-30b` rather than the OpenAI shortlist.

Do **not** set `VOICEBOT_LLM_BASE_URL` in `.env` unless you mean it globally —
it redirects every agent, including ones configured with OpenAI model ids, and
they will fail on their first call. The credential (`VOICEBOT_LLM_API_KEY`)
does belong in `.env`, and is only sent to non-OpenAI endpoints.

### Security

The TTS websocket and clone API are **open on a public IP with no
authentication** — anyone who finds the box can synthesize audio and register
voices. That was the shape you asked for and it is fine for testing, but before
this carries anything real, put it behind a firewall rule or an auth header.
The LLM at least requires a bearer token.

---

## Inbound calls

The runner detects the telephony provider from the first message on the
websocket and builds the matching frame serializer itself, so one command
handles any of them:

```bash
python bot.py -t twilio      # or telnyx / plivo / exotel
```

Point your provider's media-stream webhook at the runner's websocket endpoint
(the runner prints the URL on start; use a tunnel such as `ngrok http 7860`
while developing).

What the bot does differently on an inbound call:

- **Runs the whole pipeline at 8kHz.** PSTN is 8kHz end to end, so resampling
  up to 16/24kHz and back gains nothing and costs latency. `bot.py` detects a
  telephony transport and switches sample rates automatically —
  `VOICEBOT_TELEPHONY_SAMPLE_RATE` if you need to change it.
- **Speaks first.** Silence after pickup reads as a dead line, so the greeting
  fires on `on_client_connected`.
- **Logs the caller.** `from_number`, `to_number` and the provider's own call id
  are attached to the `call_started` log line, so you can join a Grafana trace
  to a record in your telephony console:

  ```logql
  {service="pipecat-voicebot"} | json | from_number = "+919876543210"
  ```

Daily PSTN dial-in works too (`-t daily`); the runner wires the dial-in
settings from the webhook body into `DailyParams` transparently.

---

## Quick start

```bash
# 1. Environment (Python 3.11+ required by Pipecat 1.7)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .

# 2. Credentials + prompts
cp .env.example .env && $EDITOR .env      # Soniox, OpenAI, Sarvam keys
$EDITOR prompts/system.txt prompts/greeting.txt

# 3. Observability stack (Loki + Prometheus + Grafana)
cd observability && docker compose up -d && cd ..

# 4a. Try it in the browser first
python bot.py -t webrtc

# 4b. Or take a real inbound call
python bot.py -t twilio                   # or telnyx / plivo / exotel
```

Then open:

| What                | URL                                            |
| ------------------- | ---------------------------------------------- |
| Voice client        | <http://localhost:7860>                        |
| **Grafana dashboard** | <http://localhost:3001/d/voicebot-overview>   |
| Prometheus          | <http://localhost:9091>                        |
| Loki API            | <http://localhost:3100>                        |
| Bot metrics         | <http://localhost:9188/metrics>                |

Grafana logs in with `admin` / `admin` (anonymous viewers are also allowed).

> **Ports.** Grafana is on **3001** and Prometheus on **9091**, not the usual
> 3000/9090 — this machine already runs another Grafana and Prometheus on those
> ports, so the stack is offset to coexist with them. Override with
> `GRAFANA_PORT`, `PROMETHEUS_PORT`, `LOKI_PORT` when you compose up.

---

## Layout

```
bot.py                          entrypoint; the runner calls bot(runner_args)
run.sh                          starts bot + console (and optionally Grafana)
ui/
  index.html                    agent builder + test-call console (no build step)
  server.py                     static host + agent CRUD + option lists
agents/                         saved agents, one JSON each
prompts/                        fallback system prompt and greeting
voicebot/
  config.py                     all tunables, env-driven (pydantic-settings)
  agents.py                     agent config model, persistence, option lists
  language.py                   bilingual: follows the caller's language
  pipeline.py                   service wiring, turn strategies, pipeline build
  recording.py                  streams call audio to WAV as it arrives
  obs/
    context.py                  call_id / turn contextvars
    logging_setup.py            loguru -> stdout + JSONL + Loki
    loki.py                     batching async Loki push client
    metrics.py                  Prometheus collector definitions
    observer.py                 frames -> metrics + structured logs
observability/
  docker-compose.yml            Loki 3.4 + Prometheus 3.1 + Grafana 11.5
  grafana/dashboards/           provisioned dashboard JSON
  grafana/provisioning/         datasources + dashboard providers
```

### Pipeline shape

```
transport.input()  ->  Soniox STT  ->  LanguageFollower  ->  user aggregator
   ->  OpenAI LLM  ->  Sarvam TTS  ->  transport.output()
   ->  AudioBufferProcessor  ->  assistant aggregator
```

The `AudioBufferProcessor` sits *after* `transport.output()` on purpose: that is
the only point in the graph where both the inbound user audio and the outbound
bot audio are visible, which is what makes a two-track recording possible.

---

## Turn taking and interruptions

This is the part worth understanding before tuning, because Pipecat 1.x changed
the model substantially: `allow_interruptions` and `InterruptionStrategy` no
longer exist. Interruption is now a *consequence* of turn-taking — starting a
user turn while the bot is speaking is what emits an interruption.

Both halves are configured in `voicebot/pipeline.py:build_turn_strategies`.

**Turn start (barge-in).** Strategies are evaluated in order and the **first one
to fire wins**. That detail matters: pairing a VAD strategy with a word-count
strategy lets VAD win every time and silently defeats the word gate. So the two
configurations are mutually exclusive:

| `VOICEBOT_INTERRUPT_MIN_WORDS` | Strategy                                     | Behaviour |
| ------------------------------ | -------------------------------------------- | --------- |
| `2` (default)                  | `MinWordsUserTurnStartStrategy`               | Needs 2 words to interrupt while the bot speaks; 1 word when it is idle. Ignores backchannels. |
| `0`                            | `VADUserTurnStartStrategy` + `Transcription…` | Interrupts on the first hint of speech. Lowest latency, most false triggers. |

The min-words path relies on Soniox's interim (non-final) tokens, which the
realtime websocket emits by default.

**Turn stop (smart turn).** `LocalSmartTurnAnalyzerV3` runs an ONNX model over
the last 8 seconds of audio and reads prosody to tell "I'm done" from "I'm
thinking". It ships inside `pipecat-ai` (an 8.7 MB bundled model) and runs on
CPU, so there is no network hop and no extra dependency to install.

> **Soniox has its own endpoint detection, and it is deliberately off.**
> `SonioxSTTService` takes `vad_force_turn_endpoint`, which defaults to `True`
> — meaning Soniox endpointing is disabled and local VAD + smart turn own the
> turn boundary. Setting it to `False` hands turn-taking to Soniox and
> **bypasses smart turn entirely**, and Pipecat will override your
> `user_turn_strategies` to match. The consequence of keeping the default:
> Soniox's `max_endpoint_delay_ms` / `endpoint_sensitivity` /
> `endpoint_latency_adjustment_level` settings are inert, so tuning them here
> would be a no-op.

| Variable | Effect |
| -------- | ------ |
| `VOICEBOT_SMART_TURN_STOP_SECS` | Silence the model tolerates before deciding the turn ended. Raise it for users who pause mid-thought. |
| `VOICEBOT_VAD_STOP_SECS`        | Silero's own silence threshold. Keep it well below the smart-turn value. |
| `VOICEBOT_WAIT_FOR_TRANSCRIPT`  | Require a finalized transcript before ending the turn. Set `false` to take STT off the critical path. |

Watch **Smart turn confidence** on the dashboard while tuning. Values hovering
near 0.5 mean the model genuinely cannot tell turns apart on your audio, and no
amount of timeout tuning will fix that.

### Backchannels

Indian callers acknowledge continuously while the other side is talking — *hmm*,
*haan haan*, *ji ji*, *achha*, *uh-huh*. None of it is a bid for the floor, but
to a turn-taking system that interrupts on speech it is indistinguishable from
one, and the bot stops mid-sentence every few seconds.

**Caller backchannels no longer interrupt** (`VOICEBOT_BACKCHANNEL_SUPPRESSION`,
on by default). `BackchannelAwareUserTurnStartStrategy` replaces the word-count
strategy with a lexical test. The rule that makes it safe:

> Suppression applies **only while the bot is speaking**.

The same word means different things depending on who holds the floor. Bot
talking + "हाँ" = *I'm listening*, ignore it. Bot silent, having just asked a
question, + "हाँ" = *yes*, which is an answer and must start a turn. That single
distinction is what lets the lexicon be aggressive without swallowing anything.
Anything unrecognised interrupts exactly as before, so the failure mode is
"missed a backchannel and interrupted", never "swallowed what the caller said".
`stop` / `रुकिए` / `नहीं` / `wait` are checked **first** and always take the floor.

Turning this on makes `VOICEBOT_INTERRUPT_MIN_WORDS` inert, deliberately. A word
count is a crude proxy for the same goal and the two stack badly: with
`min_words=2`, the one-word commands that most need to get through — "stop",
"रुको" — are exactly the ones it blocks. The lexicon tests the thing the word
count was approximating, so it replaces it rather than joining it.

The lexicon was built from **measured** Soniox output, not from how the words are
spelled. Each phrase was synthesized and transcribed through the same
`stt-rt-v5` config the bot runs (`language_hints=en,hi`):

| spoken | Soniox returned | why it matters |
| --- | --- | --- |
| `hmm` (English) | `हम्म।` | English can come back in Devanagari |
| `ओके` (Devanagari) | `Okay.` | …and Hindi in Latin |
| `ji ji` (romanized) | `जी, जी।` | script does not follow the input |
| `haan haan` | `Hanhan.` | repeats merge into one token |
| `yeah yeah` | `Yeah.` | repeats collapse to one |
| `theek hai` | `दिखाई।` | sometimes simply wrong |

So every concept is listed in both scripts, and repeats are handled token-wise
*and* within a single token. The last row is why the matcher fails toward
interrupting.

**Length is not evidence of real speech.** Repetition is *how* Indian callers
backchannel — "हाँ हाँ हाँ हाँ हाँ", "haan haan haan hmm" — and the run goes on
for as long as the other side keeps talking. A blanket word cap gets this
exactly backwards: the more it repeats, the more clearly it is a backchannel. So
the cap applies to **variety**, not length:

| utterance | | |
| --- | --- | --- |
| `हाँ हाँ हाँ हाँ हाँ हाँ हाँ` | 1 distinct | backchannel — no length limit |
| `han haan haan haan hmm` | 3 distinct | backchannel — no length limit |
| `हाँ जी बिल्कुल ठीक सही अच्छा ओके समझा` | 8 distinct | capped → interrupts |
| `हाँ हाँ हाँ हाँ रुकिए` | contains `रुकिए` | interrupts, always |

Repeats fold before variety is counted, so `haan`, `haaan` and `haanhaan` are
one acknowledgement rather than three. Elongation is folded however it is
spelled, including Devanagari that repeats the vowel *sign* (`हााा` → `हा`).

Beyond Hindi and Indian English, the lexicon also covers backchannels in
Bengali, Gujarati, Punjabi, Marathi, Tamil, Telugu, Kannada, Malayalam and
Odia — callers switch script more readily than they switch language.

> One trap worth naming: the obvious `re.sub(r"[^\w\s]", "", text)` for
> stripping punctuation **destroys Devanagari**. Combining marks are `Mn`/`Mc`
> and are not `\w`, so `हाँ` becomes `ह` and no Hindi backchannel ever matches.
> Punctuation is stripped by Unicode category instead.

> **On spelling *hmm* in Devanagari.** It is `ह्म्म` — ह् + म्म, no vowel. The
> obvious `हम्म` is not: it carries a real vowel and is pronounced "hum", which
> is audible and wrong (round-tripping `हम्म्म` through Soniox returns `हम`).
>
> Engines fail on a phrase by returning **silence rather than an error**, so a
> correct phrase can vanish from the pool with no symptom. Sarvam says
> `ह्म्म ह्म्म` cleanly. OmniVoice is *intermittent* on it — measured 2
> successes in 5 identical attempts, same text and voice — which is why
> `render_filler` retries the wanted phrase three times before trying it
> doubled, and only then substitutes. The configured spelling is the goal, and
> a clip is cached the first time it works, so retrying costs nothing after
> that. A substitution logs at **warning** level naming both the wanted and the
> used phrase; the OmniVoice fallback is `हूँ हूँ`, the only spelling it renders
> reliably and that Soniox transcribes back as "Hmm."
>
> Phrases are two words each for the same reason: single tokens often render as
> silence, and an empty render now also retries once doubled.

**The bot backchannels too** (`VOICEBOT_FILLER_ENABLED`, on by default). Once the
caller has been talking for `VOICEBOT_FILLER_AFTER_SECS` (3s), it plays a quiet
"hmm" / "ji ji" so the line does not feel dead — capped per utterance and never
while the bot itself is speaking.

This does **not** go through the TTS service, and that is the whole design.
`BaseOutputTransport` has one `_audio_queue` per destination drained serially at
wall-clock pace, so anything queued ahead of the reply delays it by its own
duration; and `TTSAudioRawFrame` triggers `BotStartedSpeakingFrame`, which drives
turn tracking, recording segmentation and the barge-in gate. Fillers instead go
through the **output audio mixer**, the one genuinely parallel path in the
framework: when the queue is empty the transport synthesizes a frame from the
mixer alone, as a plain `OutputAudioRawFrame`. That cannot delay the response by
even one chunk, carries no bot-speaking bookkeeping, and is still recorded.

Clips are synthesized once in the **agent's own voice** and cached under
`assets/backchannels/` — a filler in a different voice is worse than none.
Rendering happens in the background on first use, so it never delays call setup.

The cache key is everything that changes how a clip *sounds*: provider, model,
voice, language, speed, and a hash of the configured wording. Keying on provider
and voice alone is not enough and would quietly defeat the point — Sarvam's
`bulbul:v2` and `bulbul:v3` share speaker names but not their delivery,
ElevenLabs' `eleven_flash_v2_5` and `eleven_v3` share voice ids, and OmniVoice's
`speed` changes the reading. Sample rate is deliberately *not* in the key: clips
are stored at the engine's rate and resampled on load, so one file serves every
transport.

The phrases themselves are editable per agent in the console, one per line for
each of the three tones. Blank uses the built-in set.

Tone is picked from the caller's speech energy relative to their own running
baseline: louder gets a firmer "ji ji", quieter gets a soft "hmm". That is
**prosody, not emotion recognition** — three seconds of streaming audio without a
dedicated speech-emotion model does not support a claim about how someone feels,
and the code does not make one. It is enough to keep the acknowledgement from
sounding tone-deaf.

| Variable | Default | |
| --- | --- | --- |
| `VOICEBOT_BACKCHANNEL_SUPPRESSION` | `true` | ignore caller backchannels while speaking |
| `VOICEBOT_BACKCHANNEL_MAX_TOKENS` | `6` | cap on a *varied* run; repetitive runs are uncapped |
| `VOICEBOT_BACKCHANNEL_MAX_DISTINCT` | `3` | distinct acknowledgements still counted as repetition |
| `VOICEBOT_FILLER_ENABLED` | `true` | bot plays its own acknowledgements |
| `VOICEBOT_FILLER_AFTER_SECS` | `3.0` | caller speech before the first filler |
| `VOICEBOT_FILLER_INTERVAL_SECS` | `4.5` | gap between fillers in one utterance |
| `VOICEBOT_FILLER_MAX_PER_TURN` | `3` | cap per utterance |
| `VOICEBOT_FILLER_GAIN` | `0.55` | above ~0.7 reads as interrupting |

Both emit Loki events at INFO — `backchannel_suppressed` and `filler_played` —
so you can confirm from Grafana that they are firing rather than guessing.

There is a useful interaction between the two: the filler plays while the
caller's mic is open, so it can echo back into the STT. An echoed "hmm" is
itself a backchannel, so the suppression side ignores it.

---

## Ending the call

The bot hangs up once it has **finished speaking** a closing line. The timing is
the whole trick: ending on the *text* is too early — the words exist but have
not been played, and the caller hears the line die mid-sentence. So it arms on
the goodbye and fires on the transport's `BotStoppedSpeakingFrame`, then pushes
`EndWorkerFrame` to close the pipeline gracefully, flushing what is queued.

**Matched against the end of a reply, not anywhere in it.** A substring match
hangs up on a mid-call "धन्यवाद"; a goodbye is by definition the last thing
said. Verified — a reply containing `धन्यवाद आपका दिन शुभ हो` *mid-sentence*
does not end the call, while the same words as the closing line do. The failure
modes are not symmetric: a call that fails to auto-end is an annoyance, one that
hangs up on a talking customer is a complaint.

`VOICEBOT_END_CALL_LINGER_SECS` (0.4) covers the transport's jitter buffer,
which still holds a little audio when the bot-stopped signal fires; without it
the final syllable can clip on a PSTN leg.

Closing lines are editable per agent in the console, one per line. The report
records `ended by: bot | caller` — a bot-ended call reached its closing line, a
caller-ended one may have been abandoned.

---

## Two things the console can do before you dial

**Hear the voice.** *Text to speech → Hear it before you dial* synthesizes any
line — the greeting by default — using the settings **currently in the form**,
including unsaved changes, so what you hear is what the call would say. It
retries a few times, because some engines drop short Devanagari intermittently
and a preview that fails on a phrase which works two times in five teaches you
to distrust a phrase that is fine.

**Edit the prompt in English.** *Behaviour → describe an edit* rewrites the
system prompt with your change applied and drops the result back in the box for
review before saving.

Editing is whole-prompt, not patch-based, on purpose: these prompts run to
thousands of words with interdependent sections, and a model applying a targeted
diff routinely drops the parts it was not thinking about. Demanding the complete
text back makes an omission visible — and `voicebot/prompting.py` rejects a reply
shorter than 40% of the original rather than saving it, because the response goes
**straight into the prompt box** with no human parsing step. A model that answers
"Here's the updated prompt:" would otherwise write that sentence into the bot's
instructions. Preambles and code fences are stripped; on any failure the original
is returned untouched.

> **This is slow on a local model, unavoidably.** Rewriting costs one generated
> token per token of prompt. Devanagari runs about two characters per token, so
> the 15k-character Hindi prompt is ~7,500 tokens — roughly six minutes at the
> self-hosted 30B's ~20 tok/s. The budget is sized against the endpoint's real
> `max_model_len` (asking for more is a hard 400, not a truncation), and a
> timeout explains the arithmetic rather than just failing. Point **Analysis
> model** at something faster if you edit prompts often.

---

## Reviewing calls in the console

The **Calls** tab lists every ended session — agent, call id, disposition,
duration, p50 voice-to-voice — and opens each one with its recording and
transcript side by side. Clicking a transcript line seeks the audio to it, and
the line currently being spoken highlights as the audio plays.

### Getting the alignment right

Two things had to be fixed before the transcript could be trusted against the
audio, and both were silent constant offsets — the worst kind, because nothing
looks broken.

**One clock.** The observer is built during pipeline construction; the audio
buffer starts recording later, when it sees the `StartFrame`. Timestamping
transcripts from the observer's own birth put every line ahead of the audio by
that gap. The recorder now calls `observer.mark_timeline_start()` the instant
recording begins, so both share an origin.

**Onset, not finalization.** A `TranscriptionFrame` arrives when the STT
*finalizes* a turn — after the speaker stopped. Timestamping there puts every
caller line *later* than the words it describes, so clicking it lands you past
them. Verified against a real recording by checking, for each line, which
channel was actually loud at that timestamp: 6/7 lines landed on the right
speaker, and the one that missed was a caller line stamped 0.8s late, into the
bot's reply. Lines are now stamped at `UserStartedSpeakingFrame` for the caller
and `BotStartedSpeakingFrame` for the bot (TTS begins before the LLM finishes
streaming, so that frame is the right mark for the reply that follows).

Audio and analysis also now share one filename stem. They used to generate a
timestamp each — a second apart — so the console could not pair them and every
call showed as having no recording. Calls recorded before that fix are matched
by call id instead, so nothing already on disk is stranded.

---

## Post-call analysis

When a call ends, a report is written next to its recording — same agent
directory, same timestamp — as both `-analysis.json` and a readable
`-analysis.txt`:

```
CALL demo1234   agent=selfhosted-stack
2026-08-14T21:14:57+00:00   duration 31.4s   exchanges 4

OUTCOME  (inferred by LLM — not evidence)
  disposition          information_provided
  resolved             False
  sentiment            frustrated  (Caller says 'ये बहुत देर हो गई, मुझे रिफंड चाहिए'.)
  caller intent        Check delayed order status and request refund

LATENCY (measured)
  voice-to-voice         p50   2.05s   p90   3.80s   max   3.80s   (n=4)
    turn detection       p50   0.31s   p90   0.33s   max   0.33s   (n=4)
    response (LLM+TTS)   p50   1.77s   p90   3.51s   max   3.51s   (n=4)
  over budget            4/4 exchanges above 0.5s (100%)

COMPONENT TTFB (measured)
  LLM                    p50   0.20s   p90   1.22s   max   1.22s   (n=4)
  STT                    p50   0.43s   p90   0.44s   max   0.44s   (n=4)
  TTS                    p50   0.35s   p90   0.60s   max   0.60s   (n=4)

CALLER
  response time          p50   3.40s   p90   6.80s   max   6.80s   (n=4)
  interruptions          1
  backchannels ignored   3
  re-engagement prompts  1
```

Plus summary, key points, action items, and a `bot_issues` list (what the *bot*
did wrong — talked over the caller, misunderstood, dead air).

**The two halves are separated on purpose.** Latency, counts and tokens are
arithmetic over the observer's record of that call: always correct, free, and
written even when the LLM half fails. Disposition, sentiment and the summary are
one model's reading of a transcript — useful, but labelled `(inferred by LLM —
not evidence)` in the report so nobody builds a metric on them by accident.

Percentiles rather than a mean, because a mean hides the shape: one 8-second
reply among nine fast ones averages away, and the 8 seconds is what the caller
remembers.

The analysis runs *after* the call, so it never competes with it — which also
means `VOICEBOT_ANALYSIS_MODEL` can point at a bigger, slower model than the one
driving the conversation, at no cost to the caller.

| Variable | Default | |
| --- | --- | --- |
| `VOICEBOT_ANALYSIS_ENABLED` | `true` | gates only the LLM half |
| `VOICEBOT_ANALYSIS_MODEL` | *(blank)* | blank reuses the call's model |
| `VOICEBOT_ANALYSIS_DIR` | `./recordings` | filed beside the audio |

---

## Silence and follow-up

**Re-engagement.** If the caller goes quiet for `VOICEBOT_REENGAGE_AFTER_SECS`
(5s) after the bot finishes, the bot checks they are still there, escalating:

```
हैलो?
जी, मेरी आवाज़ आ रही है?
क्या आप मुझे सुन पा रहे हैं?
```

Pipecat's `UserIdleController` supplies the timer, and two of its properties do
most of the work: it is suppressed while a user turn is in progress, so a caller
mid-sentence is never talked over; and it re-arms on every `BotStoppedSpeaking`,
so speaking a prompt schedules the next check for free — escalation needs no
timer of its own. After `VOICEBOT_REENGAGE_MAX_ATTEMPTS` it stops, because a bot
talking into a dead line is worse than one that goes quiet. The counter resets
the moment the caller speaks. Prompts are editable per agent in the console, one
per line.

**The stranded-transcript watchdog.** Soniox turns tokens into a
`TranscriptionFrame` only when an **end token** arrives, which is normally
provoked by a `finalize` sent on `VADUserStoppedSpeakingFrame`. If that finalize
is missed, nothing else ever asks for one — the buffered text is released by the
*next* utterance's end token and arrives glued to it, one turn late. That is the
"transcription sticks on the last chunk until you speak again" symptom.

Pipecat 1.7.0 records `_last_tokens_received` and mentions an "auto finalize
delay", but **nothing reads that attribute** — there is no timer and so no
recovery. `voicebot/stt.py` supplies the missing one: text buffered plus
`VOICEBOT_STT_FINALIZE_AFTER` (1.5s) of silence from Soniox re-sends `finalize`.
On a healthy call it never fires.

---

## Recording

Every call writes three files under `VOICEBOT_RECORDINGS_DIR`, in a directory
named after the agent and prefixed with the call's start time:

```
recordings/
  pooja-kapture/
    20260815-014526-a1b2c3d4-mixed.wav    stereo: user left, bot right
    20260815-014526-a1b2c3d4-user.wav     mono, user only
    20260815-014526-a1b2c3d4-bot.wav      mono, bot only
  selfhosted-stack/
    ...
```

`ls` in an agent's directory is therefore already a call log in order. The call
id stays on the end because two calls can start within the same second. Calls
that name no agent land in `default/`.

> **Recording depends on an explicit flush, not on the pipeline draining.** The
> audio buffer only emits when it fills (30s) or when the processor is stopped,
> and the processor is normally stopped by an `EndFrame` travelling the
> pipeline. If anything upstream is wedged — a TTS service in a read timeout —
> that frame never arrives and the whole call's audio is dropped with no error
> anywhere. `CallRecorder.close()` therefore calls `stop_recording()` itself,
> then waits for the write handlers: Pipecat dispatches async event handlers as
> tasks *without awaiting them*, so closing the files immediately after would
> race the writes. Both bugs were live — three consecutive calls logged
> "recording started" and wrote nothing.

The three files are:

```
<call_id>-mixed.wav   stereo — user left, bot right
<call_id>-user.wav    mono
<call_id>-bot.wav     mono
```

Audio is flushed every `VOICEBOT_RECORDING_FLUSH_SECS` (30s default) and written
off the event loop, so a long call does not grow the process heap. The stereo
mix is the one to open when debugging turn-taking: overlapping speech is
visible at a glance in any waveform viewer.

Set `VOICEBOT_RECORDING_ENABLED=false` to drop the processor from the pipeline
entirely.

---

## What gets measured

Metrics require `enable_metrics=True` and `enable_usage_metrics=True` on
`PipelineParams` — without them the services emit no `MetricsFrame` at all and
every latency panel stays empty. Both are set in `voicebot/pipeline.py`.

| Metric | Meaning |
| ------ | ------- |
| `voicebot_voice_to_voice_latency_seconds` | VAD speech end → bot audio out. **The number the caller actually feels.** |
| `voicebot_turn_detection_latency_seconds` | VAD speech end → turn end (VAD wait + smart turn + STT finalize) |
| `voicebot_response_latency_seconds` | Turn end → bot audio out (LLM + TTS) |
| `voicebot_latency_budget_exceeded_total` | Responses that missed `VOICEBOT_LATENCY_BUDGET_MS` |
| `voicebot_ttfb_seconds{service_type}` | Time to first byte per tier (stt/llm/tts) — shows which stage owns the latency |
| `voicebot_ttfa_seconds`, `voicebot_tts_leading_silence_seconds` | Time to first *audible* sample, and how much of it was vendor silence padding |
| `voicebot_llm_tokens_total{kind}` | prompt / completion / cache_read / cache_creation / reasoning |
| `voicebot_smart_turn_*` | Verdicts, confidence distribution, inference latency |
| `voicebot_turns_total{interrupted}`, `voicebot_interruptions_total` | Turn-taking health |
| `voicebot_recorded_seconds_total{track}` | Recording throughput per track |
| `voicebot_requests_total`, `voicebot_errors_total`, `voicebot_function_calls_total` | Volume and failures |

### Logs

Logs go to three places at once: colored stdout, `logs/voicebot.jsonl`, and
Loki. Every line carries `call_id` and `turn`, injected from context variables —
including logs emitted inside Pipecat itself.

Loki **labels** are deliberately low-cardinality (`service`, `env`, `host`,
`level`, `component`). `call_id` lives *inside* the JSON line, not in a label:
a per-call label would create one Loki stream per call and melt the ingester.
Query it with the JSON parser instead:

```logql
# One call, end to end
{service="pipecat-voicebot"} | json | call_id = "a1b2c3d4e5f6"

# Every LLM turn that took longer than half a second to first byte
{service="pipecat-voicebot"} | json | event = "ttfb" | service_type = "llm" | ttfb_secs > 0.5

# Turns the user cut short
{service="pipecat-voicebot"} | json | event = "turn_ended" | was_interrupted = true
```

The dashboard's **Call ID** template variable feeds the first query, so pasting
a call id filters the log panels to that conversation.

If you would rather not push directly, leave `VOICEBOT_LOKI_URL` empty and point
Promtail or Grafana Alloy at `logs/voicebot.jsonl` — it is the same JSON.

---

## Notes on the implementation

A few things are easy to get wrong against Pipecat 1.x, and are handled here:

- **`PipelineTask` and `PipelineRunner` are deprecated** (1.3.0). This uses
  `PipelineWorker` and `WorkerRunner` from `pipecat.pipeline.worker` /
  `pipecat.workers.runner`.
- **VAD is no longer a transport parameter.** `TransportParams` has no
  `vad_analyzer` or `turn_analyzer` field in 1.x; VAD belongs on
  `LLMUserAggregatorParams`, which builds the `VADController` internally.
- **Observers see each frame once per processor edge.** A frame crossing N
  processors fires `on_push_frame` N times, so `TelemetryObserver` dedups on
  frame id — otherwise one LLM response would be counted once per hop.
- **`TTFAMetricsData` re-reports its own `ttfb`.** That same value also arrives
  as a separate `TTFBMetricsData`, so only the TTFA-specific fields are recorded
  to avoid double counting.
- Service `model` / `voice_id` constructor arguments are deprecated in favour of
  `settings=Service.Settings(...)`, which is what `pipeline.py` uses.
- **Sarvam ships two TTS services.** `SarvamTTSService` is the websocket
  streaming one used here; `SarvamHttpTTSService` is the slower non-streaming
  sibling and is easy to grab by mistake.
- **Service-type inference is token-based, not substring-based.** Several
  vendor class names upper-case into a string containing `STT` even though they
  are TTS services, so naive substring matching misclassifies them.

---

## Troubleshooting

| Symptom | Cause |
| ------- | ----- |
| Voice-to-voice way over budget | Check the LLM is not a reasoning model first. Then the **Budget breakdown** panel: whichever half is nearest the total is the one to fix. |
| `turn_detection` dominates | Lower `VOICEBOT_VAD_STOP_SECS`; check smart-turn confidence — a p50 near 0.5 means the model can't read your audio and is hitting the fallback ceiling. |
| `response` dominates | LLM TTFB (thinking level, region) or TTS aggregation mode. Split it on the **TTFB p95 by service** panel. |
| Grafana panels empty | Prometheus can't reach the bot. Check `http://localhost:9091/api/v1/targets` — the `voicebot` job should be `up`. |
| No logs in Grafana | `VOICEBOT_LOKI_URL` unset, or Loki unreachable. The sink prints `[loki] push error:` to stdout and keeps the call running. |
| Latency panels empty but logs fine | `enable_metrics` / `enable_usage_metrics` were turned off. |
| Bot interrupted by background noise | Raise `VOICEBOT_INTERRUPT_MIN_WORDS`. |
| Bot cuts users off mid-sentence | Raise `VOICEBOT_SMART_TURN_STOP_SECS`; check the smart-turn confidence panel. |
| Port already allocated on compose up | Another Grafana/Prometheus is running. Set `GRAFANA_PORT` / `PROMETHEUS_PORT`. |

---

## Verification status

Exercised against the real package and a live Loki/Prometheus/Grafana stack:

- Pipeline builds with the correct 10-stage graph on Soniox → OpenAI → Sarvam.
- Resolved config asserted at runtime: `stt-rt-v5` / `language_hints=['en']`,
  `gpt-4.1-nano` with `service_tier=None` (unset, so not sent), Sarvam
  `bulbul:v2` / `anushka` / `en-IN` / `min_buffer_size=30`, and the **websocket**
  Sarvam service rather than the HTTP one.
- Inbound telephony asserted: all four providers produce `FastAPIWebsocketParams`
  at 8000/8000, provider detection is correct for every transport type, and the
  pipeline builds with `PipelineParams` at 8kHz.
- Prompts confirmed loading from `prompts/system.txt` and `prompts/greeting.txt`.
- Latency split verified arithmetically: `turn_detection + response == voice_to_voice`
  to 4 decimal places; budget counter and over-budget warning both fire correctly.
- Observer records every metric type; frame dedup verified by double-pushing.
- Recorder produces valid WAVs (stereo mixed + two mono tracks, correct durations).
- Loki push payload verified against an HTTP receiver: nanosecond timestamps,
  one stream per label set, `call_id` absent from labels.
- **All dashboard PromQL and LogQL queries returned live data**; 0 push errors,
  0 dropped lines across ~490 log lines.

**Not verified:** a real call (needs live vendor keys), and therefore the actual
achieved latency — the per-stage figures in the budget table are vendor/Pipecat
published numbers, not measurements from this machine, and the Sarvam row is
left blank rather than guessed. To check what OpenAI models your key can reach:

```bash
curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" \
  | grep -oE '"gpt-[0-9a-z.\-]+"' | sort -u
```
