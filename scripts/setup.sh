#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/setup.sh — one-command bootstrap: build llama.cpp (with
# -DGGML_CPU_KLEIDIAI=ON) + build kernels/ + fetch the demo GGUF model.
# This is what the Devpost "Setup Instructions" section points at.
#
# Usage:
#   ./scripts/setup.sh
#
# Idempotent: safe to re-run. Each stage skips its own work if it already
# completed with the same inputs (see scripts/lib/build_llamacpp.sh and
# scripts/lib/fetch_model.sh for exactly what is cached and how). Set
# FORCE=1 to force every stage to redo its work from scratch:
#   FORCE=1 ./scripts/setup.sh
#
# All build/download state lives under $CACHE_DIR (default:
# $TMPDIR/arm-dispatch-ledger-cache), never inside the repo checkout itself
# — see scripts/common.sh for why. Override CACHE_DIR, LLAMA_CPP_REF,
# MODEL_DIR, JOBS, etc. as documented in scripts/common.sh.
#
# Requires on PATH: git, cmake, a C/C++ compiler (clang or gcc), curl,
# python3 (only needed if a discovered kernels/tools/tests entrypoint is a
# .py file). This script does NOT install system packages for you — if a
# required tool is missing it prints the exact package name to install and
# exits non-zero, rather than silently sudo-installing things on a machine
# it doesn't own. CI workflows install these explicitly in a prior step
# instead (see .github/workflows/*.yml).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

require_tool() {
    local tool="$1" hint="$2"
    if ! command -v "$tool" >/dev/null 2>&1; then
        log_fail "required tool '$tool' not found on PATH. $hint"
        return 1
    fi
}

missing=0
require_tool git "Install with your OS package manager (e.g. 'brew install git', 'apt-get install git')." || missing=1
require_tool cmake "Install with your OS package manager (e.g. 'brew install cmake', 'apt-get install cmake')." || missing=1
require_tool curl "Install with your OS package manager (e.g. 'apt-get install curl'; curl ships with macOS)." || missing=1
if ! command -v cc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1; then
    log_fail "no C/C++ compiler found (looked for cc, clang, gcc). Install Xcode Command Line Tools on macOS ('xcode-select --install') or build-essential on Debian/Ubuntu ('apt-get install build-essential')."
    missing=1
fi
if [[ "$missing" -eq 1 ]]; then
    log_fail "one or more required tools are missing; see messages above."
    exit 1
fi

log_info "=== setup.sh: bootstrap starting (CACHE_DIR=$CACHE_DIR) ==="

log_info "--- stage: build llama.cpp (KleidiAI) ---"
bash "$SCRIPT_DIR/lib/build_llamacpp.sh"

log_info "--- stage: fetch model ---"
bash "$SCRIPT_DIR/lib/fetch_model.sh"

log_info "--- stage: build kernels ---"
bash "$SCRIPT_DIR/lib/build_kernels.sh"

log_info "=== setup.sh: bootstrap complete ==="
log_info "llama-cli:  $LLAMA_CLI"
log_info "model:      $MODEL_PATH"
log_info "Next: ./scripts/run_all.sh to verify dispatch, bench, and emit the ledger."
