#!/usr/bin/env bash
# run.sh — one-command start for Quiver Signals.
#   ./run.sh            create/reuse ./.venv, install deps, health-check, launch
#   ./run.sh --check    everything except the launch (CI / troubleshooting)
#
# Uses its own .venv so it can't disturb conda base or other projects.
# (If a conda env is active, .venv still takes precedence once sourced.)
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
  echo "[run.sh] creating virtual environment (.venv)…"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[run.sh] using: $(python -c 'import sys; print(sys.executable)')"
python -m pip install --quiet --upgrade pip
echo "[run.sh] installing/updating dependencies…"
python -m pip install --quiet --upgrade -r requirements.txt

echo "[run.sh] health check…"
python check_setup.py

if [ "${1:-}" = "--check" ]; then
  echo "[run.sh] --check passed; not launching."
  exit 0
fi

echo "[run.sh] launching — opens at http://localhost:8501 (Ctrl+C to stop)"
exec python -m streamlit run app.py
