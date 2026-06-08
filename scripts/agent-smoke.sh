#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${SCRIBY_BIN:-${ROOT_DIR}/scriby}"
STATE_DIR="${SCRIBY_AGENT_SMOKE_STATE_DIR:-$(mktemp -d)}"

cleanup() {
  if [[ -z "${SCRIBY_AGENT_SMOKE_STATE_DIR:-}" && -d "$STATE_DIR" ]]; then
    rm -rf "$STATE_DIR"
  fi
}
trap cleanup EXIT

if [[ ! -x "$BIN" ]]; then
  (cd "$ROOT_DIR" && go build -o "$BIN" .)
fi

require_json_status() {
  local expected="$1"
  shift
  local output
  output="$("$@")"
  SCRIBY_AGENT_SMOKE_OUTPUT="$output" python3 - "$expected" <<'PY'
import json
import os
import sys

expected = sys.argv[1]
payload = json.loads(os.environ["SCRIBY_AGENT_SMOKE_OUTPUT"])
status = payload.get("status")
if status != expected:
    raise SystemExit(f"status={status!r}, want {expected!r}: {payload}")
if "schema_version" not in payload or "command" not in payload or "run_id" not in payload:
    raise SystemExit(f"missing envelope keys: {payload}")
PY
}

require_json_status succeeded "$BIN" --agent history path --state-dir "$STATE_DIR"
require_json_status succeeded "$BIN" --agent history list --state-dir "$STATE_DIR" --limit 5
require_json_status succeeded "$BIN" --agent history schema --state-dir "$STATE_DIR"
require_json_status succeeded "$BIN" --agent history sql --state-dir "$STATE_DIR" "select name from sqlite_master where type = 'table' order by name"

echo "agent smoke ok: $STATE_DIR"
