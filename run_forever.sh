#!/usr/bin/env bash
# run_forever.sh — robust launcher for Buddy middleware without systemd
# - auto-detects venv (.venv or audiovenv or current $VIRTUAL_ENV)
# - restarts on crash with backoff
# - rotates logs
# - refuses to start if an identical instance is already running

set -Eeuo pipefail

### CONFIG ###
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ENTRY="serve.py"
LOG_DIR="${APP_DIR}/logs"
LOG_FILE="${LOG_DIR}/buddy.out"
PID_FILE="${APP_DIR}/buddy.pid"
PORT="8787"               # keep in sync with serve.py
BACKOFF_START=2           # seconds
BACKOFF_MAX=30            # seconds

### FUNCTIONS ###
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_FILE"; }

ensure_dirs() {
  mkdir -p "$LOG_DIR"
  # rotate if > 5MB
  if [ -f "$LOG_FILE" ] && [ "$(wc -c <"$LOG_FILE")" -gt $((5*1024*1024)) ]; then
    mv "$LOG_FILE" "${LOG_FILE%.*}.$(date +%Y%m%d-%H%M%S).out"
  fi
  : > "$LOG_FILE"  # new session log
}

find_python() {
  # 1) use current venv if already active
  if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PY="${VIRTUAL_ENV}/bin/python"; echo "$PY"; return
  fi
  # 2) .venv
  if [[ -x "${APP_DIR}/.venv/bin/python" ]]; then
    PY="${APP_DIR}/.venv/bin/python"; echo "$PY"; return
  fi
  # 3) audiovenv (your prompt showed this one)
  if [[ -x "${APP_DIR}/audiovenv/bin/python" ]]; then
    PY="${APP_DIR}/audiovenv/bin/python"; echo "$PY"; return
  fi
  # 4) system python
  if command -v python3 >/dev/null 2>&1; then
    echo "$(command -v python3)"; return
  fi
  echo ""
}

already_running() {
  if [ -f "$PID_FILE" ]; then
    local p; p="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      return 0
    else
      rm -f "$PID_FILE"
    fi
  fi
  return 1
}

wait_port_free() {
  local tries=10
  while lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do
    log "Port :$PORT busy; waiting..."
    sleep 1
    tries=$((tries-1))
    [ "$tries" -le 0 ] && { log "Port still busy. Trying anyway."; break; }
  done
}

### MAIN ###
cd "$APP_DIR"

# normalize line endings (in case the file was edited on Windows)
if grep -q $'\r' "$0"; then
  tr -d '\r' <"$0" >"$0.tmp" && mv "$0.tmp" "$0"
  chmod +x "$0"
fi

ensure_dirs

if already_running; then
  log "Another instance is already running (PID $(cat "$PID_FILE")). Aborting."
  exit 0
fi

PYTHON_BIN="$(find_python)"
if [[ -z "$PYTHON_BIN" ]]; then
  log "ERROR: No python interpreter found (tried VIRTUAL_ENV, .venv, audiovenv, system)."
  exit 1
fi

log "Using python: $PYTHON_BIN"
log "Working dir: $APP_DIR"
log "Logging to:  $LOG_FILE"
log "PID file:    $PID_FILE"

BACKOFF="$BACKOFF_START"

# trap to cleanup PID on exit
cleanup() { rm -f "$PID_FILE"; }
trap cleanup EXIT

echo $$ > "$PID_FILE"

while true; do
  wait_port_free
  log "Starting app: $PYTHON_BIN $APP_ENTRY"
  # launch and capture exit code; keep this process as the supervisor
  set +e
  "$PYTHON_BIN" "$APP_ENTRY" >> "$LOG_FILE" 2>&1
  EXIT=$?
  set -e
  log "App exited with code $EXIT"

  # graceful backoff
  sleep "$BACKOFF"
  BACKOFF=$(( BACKOFF * 2 ))
  [ "$BACKOFF" -gt "$BACKOFF_MAX" ] && BACKOFF="$BACKOFF_MAX"
done
