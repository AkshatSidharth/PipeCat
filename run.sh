#!/usr/bin/env bash
# Start everything needed to build and test an agent.
#
#   ./run.sh          bot + console  (assumes observability is already up)
#   ./run.sh --all    also brings up Loki / Prometheus / Grafana
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] || { echo "No .venv — run: uv venv --python 3.12 .venv && uv pip install -e ."; exit 1; }
[ -f .env ]  || { echo "No .env — run: cp .env.example .env  and add your keys"; exit 1; }
source .venv/bin/activate

# Work whether or not `pip install -e .` has been run: ui/server.py and bot.py
# both import the `voicebot` package from the repo root.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

if [ "${1:-}" = "--all" ]; then
  echo "==> observability stack"
  (cd observability && docker compose up -d)
fi

pids=()
cleanup() { echo; echo "==> stopping"; for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "==> bot runner    http://localhost:7860"
python bot.py -t webrtc & pids+=($!)

echo "==> console       http://localhost:7861"
python ui/server.py & pids+=($!)

cat <<'BANNER'

  ─────────────────────────────────────────────
   Console   http://localhost:7861   ← open this
   Grafana   http://localhost:3001/d/voicebot-overview
  ─────────────────────────────────────────────
   Ctrl-C to stop.

BANNER
wait
