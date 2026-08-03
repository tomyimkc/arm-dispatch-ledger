# SPDX-License-Identifier: Apache-2.0
# scripts/common.sh — shared helpers sourced by every script in scripts/ and
# scripts/lib/, and (indirectly, via those scripts) by all three CI lanes.
#
# This file is NOT executable on its own: `source` it, don't run it.
#
# =============================================================================
# CROSS-PACKAGE CONTRACT
# =============================================================================
# This "ci" work package (.github/workflows/ + scripts/) does not own
# kernels/, tools/, tests/, mcp/, or docs/ — those are built by other agents
# working in this repo concurrently. To avoid a hard dependency on file names
# that may not exist yet (or may land with a slightly different name), every
# stage below is a *best-effort dispatcher*: it searches an ordered list of
# candidate entrypoints and runs the first one it finds. If none exist yet,
# the stage is recorded as SKIPPED (not failed, not passed) so CI stays green
# during development and automatically starts exercising real code the moment
# a matching file lands — no CI edit required.
#
# Expected entrypoints (first match wins), all resolved relative to the repo
# root:
#
#   Stage              Candidates tried in order
#   -----------------  -----------------------------------------------------
#   build kernels      kernels/Makefile        (targets: all, test, bench)
#                       kernels/CMakeLists.txt  (configure+build into kernels/build)
#                       kernels/build.sh
#   correctness tests   tests/run_tests.sh
#                       tests/Makefile          (target: test)
#                       kernels/Makefile        (target: test, if the above two absent)
#   verify dispatch     tools/verify_dispatch.py
#                       tools/verify_dispatch.sh
#                       tools/verify_dispatch      (extensionless executable)
#   reduced bench       tools/bench.py
#                       tools/bench.sh
#                       tools/run_bench.py
#                       tools/run_bench.sh
#   emit ledger         tools/emit_ledger.py
#                       tools/emit_ledger.sh
#
# If your tool needs a different name, either rename to match one of the
# candidates above, or add it to the relevant `find_entrypoint` call in
# scripts/lib/*.sh — those are the only places the candidate lists live.
#
# Contract for verify_dispatch / bench invocations: this file's
# run_entrypoint() always passes MODEL, LLAMA_CLI, THREADS, OUTPUT_DIR as
# environment variables (see scripts/lib/verify_dispatch.sh and
# scripts/lib/run_bench.sh) rather than assuming a fixed CLI flag spelling,
# since flags are cheaper for a tool author to read from env than for this
# dispatcher to guess.
#
# IMPORTANT correctness note for anyone implementing tools/verify_dispatch:
# llama-cli defaults to offloading layers to the GPU backend (Metal on
# macOS) when one is compiled in. The KleidiAI CPU dispatch under test in
# Finding 1 only executes when computation actually runs on the CPU backend.
# The verified reproduction this repo is built on used exactly this
# invocation (see docs / README for the full write-up):
#   llama-cli -m MODEL -p "..." -n 16 -no-cnv -st --simple-io -t N
# and got the expected thread-gated hit counts on the CPU backend without
# passing -ngl, on this specific 0.5B model. If you change the model, the
# backend build flags, or add GPU layers, re-verify the CPU path is actually
# what is being measured before trusting a hit count of 0 as "gated" rather
# than "not running on CPU at all".
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repo root: prefer git (works from any subdirectory / worktree), fall back to
# "two directories up from this file" (scripts/common.sh -> scripts -> root)
# so the scripts still work in a source tarball with no .git present.
if command -v git >/dev/null 2>&1 && git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel >/dev/null 2>&1; then
    REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export REPO_ROOT

# All cache / build state lives OUTSIDE the repo tree by default. This repo is
# shared by several concurrently-working agents each restricted to their own
# subtree; keeping build/download caches out of the working tree means
# running these scripts never dirties `git status` for paths this package
# does not own, and CI runners can point CACHE_DIR at their own ephemeral
# temp dir (e.g. $RUNNER_TEMP) for actions/cache without touching the
# checkout at all.
: "${CACHE_DIR:=${TMPDIR:-/tmp}/arm-dispatch-ledger-cache}"
CACHE_DIR="${CACHE_DIR%/}"
export CACHE_DIR

# results/ is the one path in the repo this package IS expected to write to
# at runtime (the workflows upload it as a build artifact). Authoring scripts
# never seed content into it ahead of time; only running them does.
: "${RESULTS_DIR:=$REPO_ROOT/results}"
export RESULTS_DIR
LOG_DIR="$RESULTS_DIR/logs"
export LOG_DIR

mkdir -p "$CACHE_DIR" "$RESULTS_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# llama.cpp / model defaults
# ---------------------------------------------------------------------------

# Pinned to the exact commit this project's findings were hand-verified
# against (see docs/ for the full writeup). Override LLAMA_CPP_REF to track
# upstream deliberately; the dispatch logic in ggml-cpu/kleidiai.cpp is young
# and has changed shape before, so floating HEAD is opt-in, not default.
: "${LLAMA_CPP_REMOTE:=https://github.com/ggml-org/llama.cpp.git}"
: "${LLAMA_CPP_REF:=dbadb68eecdfb3ab0e86872d011738fc937f0364}"
: "${LLAMA_CPP_DIR:=$CACHE_DIR/llama.cpp}"
export LLAMA_CPP_REMOTE LLAMA_CPP_REF LLAMA_CPP_DIR

# Small (~409 MiB), Apache-2.0-licensed instruct GGUF used for every lane.
# sha256 captured 2026-08-04 via:
#   curl -sIL <url> | grep -i x-linked-etag
# HF's Xet/LFS storage serves this as a content hash, confirmed here by a
# full download + `shasum -a 256` match. If Qwen re-uploads the file this
# constant goes stale; fetch_model.sh treats a mismatch as a loud WARNING
# (recorded in results/), never a hard failure, since this repo does not
# control the upstream artifact.
: "${HF_REPO:=Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
: "${HF_FILE:=qwen2.5-0.5b-instruct-q4_0.gguf}"
: "${HF_FILE_SHA256:=7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed}"
: "${MODEL_DIR:=$CACHE_DIR/models}"
: "${MODEL_PATH:=$MODEL_DIR/$HF_FILE}"
export HF_REPO HF_FILE HF_FILE_SHA256 MODEL_DIR MODEL_PATH

LLAMA_CLI="$LLAMA_CPP_DIR/build/bin/llama-cli"
LLAMA_BENCH="$LLAMA_CPP_DIR/build/bin/llama-bench"
export LLAMA_CLI LLAMA_BENCH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

log_info() { printf '[%s] [INFO] %s\n' "$(_ts)" "$*" >&2; }
log_warn() { printf '[%s] [WARN] %s\n' "$(_ts)" "$*" >&2; }
log_fail() { printf '[%s] [FAIL] %s\n' "$(_ts)" "$*" >&2; }
log_skip() { printf '[%s] [SKIP] %s\n' "$(_ts)" "$*" >&2; }
log_ok()   { printf '[%s] [ OK ] %s\n' "$(_ts)" "$*" >&2; }

# ---------------------------------------------------------------------------
# Host introspection
# ---------------------------------------------------------------------------

detect_nproc() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1 && sysctl -n hw.ncpu >/dev/null 2>&1; then
        sysctl -n hw.ncpu
    else
        getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
    fi
}

# Parallel build job cap. Measured on an Apple M4 Max (48 GB RAM): building
# llama.cpp's `all` target with -j == nproc (16) SIGKILLed
# (exit "Killed: 9", OOM) while compiling tools/llama-bench's impl library in
# parallel with several other large translation units. We only ever build
# the `llama-cli` target (see scripts/lib/build_llamacpp.sh), which sidesteps
# that specific TU, but we still cap parallelism as a second line of defence
# for smaller/self-hosted runners with less RAM than this dev machine.
build_jobs_cap() {
    local n
    n="$(detect_nproc)"
    if [[ "$n" -gt 8 ]]; then
        echo 8
    else
        echo "$n"
    fi
}

# Ascending, deduplicated thread sweep used by the dispatch-verification
# stage. Always includes 1, 2, 4, 8 (the values Finding 1 was hand-verified
# with on the M4 Max: SME2 hit at 1 and 2 threads, zero hits at 4 and 8) plus
# the host's own core count (llama.cpp's real default n_threads), each
# capped at the host's actual core count so a 4-core CI runner doesn't try
# to run at -t 8.
default_thread_sweep() {
    local nproc_val out=() seen="," t
    nproc_val="$(detect_nproc)"
    for t in 1 2 4 8 "$nproc_val"; do
        if [[ "$seen" != *",$t,"* ]] && [[ "$t" -le "$nproc_val" ]]; then
            out+=("$t")
            seen="${seen}${t},"
        fi
    done
    local IFS=,
    echo "${out[*]}"
}

# Portable wall-clock timeout. macOS ships no `timeout(1)`; this falls back
# to a background watcher + kill when neither `timeout` nor `gtimeout`
# (coreutils via Homebrew) is present, so every lane behaves the same way
# regardless of host.
#
# Deliberately NOT delegated to the external timeout(1)/gtimeout(1)
# binaries: every call site in this repo passes run_entrypoint (a bash
# *function*, see below) as the command to run, and an external `timeout`
# execs its argv literally — it cannot see shell functions, so
# `timeout 5 run_entrypoint ...` fails with "No such file or directory"
# regardless of PATH. (Hit exactly this while developing verify_dispatch.sh
# / run_bench.sh; every invocation silently "failed" via this codepath.)
# The manual background+watcher approach below runs the command in this
# same shell process, so it works uniformly for shell functions and
# external binaries alike.
run_with_timeout() {
    local secs="$1"
    shift
    "$@" &
    local pid=$!
    (
        sleep "$secs"
        kill -TERM "$pid" 2>/dev/null || true
    ) &
    local watcher=$!
    local status=0
    wait "$pid" 2>/dev/null || status=$?
    kill "$watcher" 2>/dev/null || true
    wait "$watcher" 2>/dev/null || true
    return "$status"
}

# ---------------------------------------------------------------------------
# Entrypoint discovery (see CROSS-PACKAGE CONTRACT above)
# ---------------------------------------------------------------------------

# find_entrypoint <relative-dir> <stem> [extra-verbatim-candidate ...]
# Tries "<stem>", "<stem>.py", "<stem>.sh" inside <relative-dir>, then any
# extra verbatim relative paths given. Prints the first match's absolute
# path and returns 0; returns 1 with no output if nothing matched.
find_entrypoint() {
    local rel_dir="$1"
    local stem="$2"
    shift 2 || true
    local candidate
    for candidate in \
        "$rel_dir/$stem" \
        "$rel_dir/$stem.py" \
        "$rel_dir/$stem.sh" \
        "$@"; do
        if [[ -f "$REPO_ROOT/$candidate" ]]; then
            echo "$REPO_ROOT/$candidate"
            return 0
        fi
    done
    return 1
}

# run_entrypoint <path> [args...]
# Dispatches on extension so callers don't need to care whether a tool
# author wrote Python, shell, or a compiled/extensionless executable.
run_entrypoint() {
    local path="$1"
    shift || true
    case "$path" in
        *.py)
            python3 "$path" "$@"
            ;;
        *.sh)
            bash "$path" "$@"
            ;;
        *)
            if [[ -x "$path" ]]; then
                "$path" "$@"
            else
                bash "$path" "$@"
            fi
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Stage bookkeeping — every scripts/lib/*.sh stage reports through this so
# run_all.sh and the CI job summaries have one consistent record to read.
# Appends a machine-readable line to results/stage-status.tsv:
#   <unix-ts>\t<stage-name>\t<STATUS>\t<human message>
# STATUS is one of: OK SKIP FAIL
# ---------------------------------------------------------------------------

STAGE_STATUS_FILE="$RESULTS_DIR/stage-status.tsv"
export STAGE_STATUS_FILE

record_stage() {
    local name="$1" status="$2" message="${3:-}"
    printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$name" "$status" "$message" >>"$STAGE_STATUS_FILE"
    case "$status" in
        OK) log_ok "$name: $message" ;;
        SKIP) log_skip "$name: $message" ;;
        FAIL) log_fail "$name: $message" ;;
        *) log_info "$name: $status $message" ;;
    esac
}
