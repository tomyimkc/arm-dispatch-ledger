#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/build_llamacpp.sh — clone (or reuse) a pinned llama.cpp
# checkout and build ONLY the `llama-cli` and `llama-bench` targets with
# -DGGML_CPU_KLEIDIAI=ON. (llama-cli is what tools/verify_dispatch.py drives;
# llama-bench is what tools/bench.py drives.)
#
# Idempotent: re-running with the same LLAMA_CPP_REF and no FORCE=1 is a
# no-op once both binaries exist and a matching build stamp is present. This
# is what makes the same script usable both as a local one-shot bootstrap
# and as a cheap re-run once `actions/cache` has restored $LLAMA_CPP_DIR.
#
# Why only these two targets (not `all`), and why JOBS is capped: see the
# long comment in scripts/common.sh (build_jobs_cap). Short version:
# building the full `all` target with -j == nproc SIGKILLed (OOM) on a 48 GB
# Apple M4 Max while compiling several large translation units in parallel,
# including llama-bench's own impl library — a target this repo DOES need,
# just not by building `all` to get it. Building `llama-bench` as its own
# isolated `--target` (verified separately, ~3s once ggml-cpu is warm, no
# OOM) gets the same binary without the memory pressure of building
# llama-server / examples / tests alongside it.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

: "${JOBS:=$(build_jobs_cap)}"
: "${FORCE:=0}"

STAGE="build_llamacpp"
STAMP_FILE="$LLAMA_CPP_DIR/.arm-dispatch-ledger-build-stamp"
# Stamp encodes everything that would require a rebuild if it changed.
STAMP_WANT="ref=$LLAMA_CPP_REF cmake_build_type=Release kleidiai=ON targets=llama-cli,llama-bench"

if [[ "$FORCE" != "1" && -x "$LLAMA_CLI" && -x "$LLAMA_BENCH" && -f "$STAMP_FILE" ]] && [[ "$(cat "$STAMP_FILE" 2>/dev/null)" == "$STAMP_WANT" ]]; then
    record_stage "$STAGE" OK "reusing cached build at $LLAMA_CLI / $LLAMA_BENCH (stamp matched, set FORCE=1 to rebuild)"
    exit 0
fi

log_info "llama.cpp dir: $LLAMA_CPP_DIR (ref=$LLAMA_CPP_REF)"
mkdir -p "$LLAMA_CPP_DIR"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    log_info "cloning $LLAMA_CPP_REMOTE"
    git init -q "$LLAMA_CPP_DIR"
    (cd "$LLAMA_CPP_DIR" && git remote add origin "$LLAMA_CPP_REMOTE")
fi

# Fetch just the pinned ref, shallow — GitHub supports fetching by full SHA
# for public repos (verified against ggml-org/llama.cpp on 2026-08-04), so
# this avoids ever cloning full history.
(
    cd "$LLAMA_CPP_DIR"
    log_info "fetching $LLAMA_CPP_REF (depth 1)"
    git fetch --depth 1 origin "$LLAMA_CPP_REF"
    git checkout -q --detach FETCH_HEAD
    log_info "checked out $(git rev-parse HEAD)"
)

log_info "configuring (this reuses CMakeCache.txt on re-runs)"
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_KLEIDIAI=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    >"$LOG_DIR/llamacpp-configure.log" 2>&1 \
    || {
        record_stage "$STAGE" FAIL "cmake configure failed, see $LOG_DIR/llamacpp-configure.log"
        tail -n 40 "$LOG_DIR/llamacpp-configure.log" >&2 || true
        exit 1
    }

log_info "building --target llama-cli llama-bench with -j$JOBS"
if ! cmake --build "$LLAMA_CPP_DIR/build" --target llama-cli --target llama-bench -j"$JOBS" \
    >"$LOG_DIR/llamacpp-build.log" 2>&1; then
    record_stage "$STAGE" FAIL "build failed, see $LOG_DIR/llamacpp-build.log"
    tail -n 60 "$LOG_DIR/llamacpp-build.log" >&2 || true
    exit 1
fi

if [[ ! -x "$LLAMA_CLI" || ! -x "$LLAMA_BENCH" ]]; then
    record_stage "$STAGE" FAIL "build reported success but $LLAMA_CLI and/or $LLAMA_BENCH is missing"
    exit 1
fi

echo "$STAMP_WANT" >"$STAMP_FILE"
record_stage "$STAGE" OK "built $LLAMA_CLI and $LLAMA_BENCH ($(cd "$LLAMA_CPP_DIR" && git rev-parse --short HEAD))"
