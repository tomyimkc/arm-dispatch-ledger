#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/run_bench.sh — drives tools/bench.py (the SME2-vs-NEON
# interleaved throughput sweep). Flags below match the real, landed CLI
# (`python3 tools/bench.py --help`), discovered dynamically via
# find_entrypoint (see CROSS-PACKAGE CONTRACT in scripts/common.sh) so a
# rename is still picked up automatically.
#
# BENCH_REDUCED=1 (default, what CI uses): --reps 2 instead of the tool's
# own default of 5, keeping this a fast CI smoke measurement rather than the
# publication-quality sweep a developer would run locally with
# BENCH_REDUCED=0.
#
# --skip-dispatch-verify is ALWAYS passed, in both modes, regardless of
# BENCH_REDUCED. Why: measured directly on this machine, bench.py's
# built-in per-configuration lldb dispatch verification (its fallback path
# when --verify-dispatch-cmd is not given) was SIGKILLed (OOM) after ~82s,
# partway through verifying its 18 unique configurations, on a 48 GB Apple
# M4 Max. Dispatch verification is already this pipeline's dedicated
# verify_dispatch.sh stage (see that file) — running it a second time,
# inside bench.py, is both redundant and the thing that crashed. If a future
# bench.py revision fixes that memory issue, --verify-dispatch-cmd can be
# passed instead of --skip-dispatch-verify to get bench.py's own richer
# per-row dispatch labels; until then this is the safe default.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

: "${BENCH_REDUCED:=1}"
: "${BENCH_TIMEOUT_SECS:=900}"
STAGE="run_bench"

entrypoint="$(find_entrypoint tools bench tools/run_bench.py tools/run_bench.sh || true)"
if [[ -z "$entrypoint" ]]; then
    record_stage "$STAGE" SKIP "no tools/bench(.py|.sh) or tools/run_bench(.py|.sh) found yet"
    exit 0
fi

if [[ ! -x "$LLAMA_BENCH" ]]; then
    record_stage "$STAGE" FAIL "llama-bench not built at $LLAMA_BENCH; run build_llamacpp first"
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    record_stage "$STAGE" FAIL "model not present at $MODEL_PATH; run fetch_model first"
    exit 1
fi

OUTPUT_DIR="$RESULTS_DIR/bench"
mkdir -p "$OUTPUT_DIR"
logfile="$LOG_DIR/run_bench.log"

args=(
    --llama-bin-dir "$(dirname "$LLAMA_BENCH")"
    --model "$MODEL_PATH"
    --out-dir "$OUTPUT_DIR"
    --skip-dispatch-verify
    --per-call-timeout 60
    --quiet
)
if [[ "$BENCH_REDUCED" == "1" ]]; then
    args+=(--reps 2)
    log_info "BENCH_REDUCED=1: using tool defaults for --threads/--phases (already a reduced grid per tools/protocol.md), --reps 2"
else
    log_info "BENCH_REDUCED=0: full local run, tool defaults (--reps 5, threads=1,2,8)"
fi

log_info "found bench tool: $entrypoint (timeout=${BENCH_TIMEOUT_SECS}s)"
if run_with_timeout "$BENCH_TIMEOUT_SECS" run_entrypoint "$entrypoint" "${args[@]}" >"$logfile" 2>&1; then
    record_stage "$STAGE" OK "bench completed, results in $OUTPUT_DIR (log: $logfile)"
else
    record_stage "$STAGE" FAIL "bench failed or timed out after ${BENCH_TIMEOUT_SECS}s, see $logfile"
    tail -n 40 "$logfile" >&2 || true
    exit 1
fi

# Optional: render tools/plot_results.py figures from whatever bench-*.json
# the run above produced. Best-effort and non-fatal — a plotting failure
# (e.g. matplotlib missing on a minimal runner) should never fail the bench
# stage itself, since plot_results.py degrades to a markdown table on its
# own when matplotlib is unavailable (see its --help).
plot_entrypoint="$(find_entrypoint tools plot_results || true)"
if [[ -n "$plot_entrypoint" ]]; then
    bench_json="$(find "$OUTPUT_DIR" -maxdepth 1 -name 'bench-*.json' -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -n1 || true)"
    if [[ -n "$bench_json" ]]; then
        log_info "found plot tool: $plot_entrypoint -> $bench_json"
        if run_entrypoint "$plot_entrypoint" "$bench_json" --out-dir "$OUTPUT_DIR/figures" \
            >"$LOG_DIR/plot_results.log" 2>&1; then
            log_ok "plot_results: wrote figures under $OUTPUT_DIR/figures"
        else
            log_warn "plot_results failed (non-fatal), see $LOG_DIR/plot_results.log"
        fi
    else
        log_warn "plot_results.py found but no bench-*.json in $OUTPUT_DIR to render"
    fi
fi
