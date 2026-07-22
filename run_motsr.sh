#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

python llm_engine/engine_new.py &
ENGINE_PID=$!

cleanup() {
  kill "$ENGINE_PID" 2>/dev/null || true
}

trap cleanup EXIT

python main.py
