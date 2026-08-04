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
#
# CLOUD_THROUGHPUT=1 (opt-in, set only by verify-free-arm64.yml's dedicated
# "Cloud-class throughput benchmark" step): a SECOND, independent code path
# in this same file, gated at the very top so it never touches tools/bench.py
# or the BENCH_REDUCED logic above. Why this exists: every quantified
# throughput number elsewhere in this repo (results/AUTODEFAULTS.md,
# results/REMEASURE-2026-08-04-QUIET.md, results/OPTIMIZATION.md) was
# measured on an Apple M4 Max laptop/desktop, not cloud hardware — but this
# is a Track 2 (Cloud AI) submission, and the free-hosted ubuntu-24.04-arm
# runner (Neoverse-N2 class) is the one lane in this whole project that
# actually runs on cloud-class Arm64. This path drives llama-bench directly
# (not tools/bench.py — that tool is owned by a different work package in
# this repo, and its SME-on/off axis is meaningless here: Neoverse-N2 has no
# SME2 at all, and its SVE2 is 128-bit, below KleidiAI's 256-bit SVE
# dispatch gate, so the SVE kernel family can never be selected here either
# — see results/GROUND-TRUTH-DISPATCH.md, Finding 2) and measures decode and
# prefill throughput, separately, across a small thread sweep capped at the
# runner's own core count.
#
# Methodology (mirrors tools/bench.py's own, documented in tools/protocol.md,
# so the two lanes' numbers stay comparable in spirit even though this path
# is a separate, smaller implementation): round-robin interleaved across
# every (phase, threads) cell — one full pass over every cell per round,
# `CLOUD_THROUGHPUT_REPS` rounds total — never all reps of one cell before
# another (see this repo's HARD RULES: that anti-pattern produced a
# previously-retracted fake +57.3% claim). Each round-robin call is a single
# `llama-bench -r 1` invocation (its own discarded warmup, per protocol.md's
# convention), so no cell's samples are clustered early or late in wall-clock
# time. Median/stddev/min/max are computed across rounds — never a bare
# mean. A per-stage wall-clock budget (CLOUD_THROUGHPUT_BUDGET_SECS) is
# checked between cells so a slow runner degrades gracefully (fewer
# completed rounds, explicitly listed as skipped/partial in the emitted
# JSON + markdown) rather than blowing the job's overall 20-minute budget —
# see verify-free-arm64.yml's timeout-minutes comment. A per-call
# subprocess failure drops that one sample (never fabricates a number for
# it); if literally nothing was measured, the stage fails loudly rather than
# emitting an all-empty table.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

: "${BENCH_REDUCED:=1}"
: "${BENCH_TIMEOUT_SECS:=900}"
: "${CLOUD_THROUGHPUT:=0}"
: "${CLOUD_THROUGHPUT_REPS:=5}"
: "${CLOUD_THROUGHPUT_DECODE_TOKENS:=32}"
: "${CLOUD_THROUGHPUT_PREFILL_TOKENS:=128}"
: "${CLOUD_THROUGHPUT_PER_CALL_TIMEOUT:=60}"
# Soft internal deadline (checked between round-robin cells, so the stage
# always has time to write out whatever it DID measure) and a hard
# safety-net timeout wrapped around the whole stage (in case a single
# llama-bench call itself wedges past its own per-call timeout somehow).
# 240s / 480s: measured single-cell calls for this repo's ~0.5-1.5B models
# take well under a second on an Apple M4 Max (see this package's OKF run
# log); even at several seconds per call on a slower Neoverse-N2 vCPU, the
# default grid (2 phases x up to 3 threads x 5 reps = up to 30 calls) fits
# comfortably inside 240s, leaving the 480s hard cap as pure headroom, and
# both fit inside the job's existing 20-minute total budget (see
# verify-free-arm64.yml).
: "${CLOUD_THROUGHPUT_BUDGET_SECS:=240}"
: "${CLOUD_THROUGHPUT_STAGE_TIMEOUT_SECS:=480}"
STAGE="run_bench"

# ---------------------------------------------------------------------------
# CLOUD_THROUGHPUT=1 path — see file header. Early-exits the whole script
# (never falls through to the tools/bench.py logic below) so this is always
# a strictly additive second invocation, not a replacement for the existing
# reduced-bench step.
# ---------------------------------------------------------------------------

run_cloud_throughput_bench() {
    local stage="cloud_throughput_bench"

    if [[ ! -x "$LLAMA_BENCH" ]]; then
        record_stage "$stage" FAIL "llama-bench not built at $LLAMA_BENCH; run build_llamacpp first"
        return 1
    fi
    if [[ ! -f "$MODEL_PATH" ]]; then
        record_stage "$stage" FAIL "model not present at $MODEL_PATH; run fetch_model first"
        return 1
    fi

    local out_dir="$RESULTS_DIR/bench"
    mkdir -p "$out_dir"
    local raw_dir
    raw_dir="$(mktemp -d "${CACHE_DIR}/cloud-throughput-raw.XXXXXX")"
    local platform_slug
    platform_slug="$(uname -m)"
    local out_json="$out_dir/cloud-throughput-${platform_slug}.json"
    local out_md="$out_dir/cloud-throughput-${platform_slug}.md"
    local logfile="$LOG_DIR/cloud_throughput_bench.log"
    : >"$logfile"

    local nproc_val
    nproc_val="$(detect_nproc)"

    # Small thread sweep, capped at this runner's actual core count. On the
    # free-hosted ubuntu-24.04-arm runner nproc is 4 (see this file's header
    # comment / scripts/common.sh's default_thread_sweep, which the rest of
    # this pipeline's dispatch-verification stage already relies on the same
    # fact for), so this resolves to "1 2 4".
    local t seen="," threads=()
    for t in 1 2 4 "$nproc_val"; do
        if [[ "$seen" != *",$t,"* ]] && [[ "$t" -le "$nproc_val" ]]; then
            threads+=("$t")
            seen="${seen}${t},"
        fi
    done
    local threads_csv
    threads_csv="$(IFS=,; echo "${threads[*]}")"

    # decode: n_prompt=0, n_gen=CLOUD_THROUGHPUT_DECODE_TOKENS.
    # prefill: n_prompt=CLOUD_THROUGHPUT_PREFILL_TOKENS, n_gen=0.
    # Always reported as separate rows/cells — never blended (see this
    # repo's HARD RULES).
    local phase_names=(decode prefill)
    local phase_nprompt=(0 "$CLOUD_THROUGHPUT_PREFILL_TOKENS")
    local phase_ngen=("$CLOUD_THROUGHPUT_DECODE_TOKENS" 0)

    local total_cells=$(( ${#phase_names[@]} * ${#threads[@]} ))
    local total_calls=$(( total_cells * CLOUD_THROUGHPUT_REPS ))
    log_info "cloud_throughput_bench: ${#phase_names[@]} phases x ${#threads[@]} threads ($threads_csv) x ${CLOUD_THROUGHPUT_REPS} reps = $total_calls llama-bench calls, round-robin interleaved, budget=${CLOUD_THROUGHPUT_BUDGET_SECS}s"

    local start_ts
    start_ts="$(date +%s)"
    local call_no=0
    local budget_hit=0
    local round pi ti phase nprompt ngen th elapsed rc cell_json

    for (( round=1; round<=CLOUD_THROUGHPUT_REPS; round++ )); do
        for (( pi=0; pi<${#phase_names[@]}; pi++ )); do
            phase="${phase_names[$pi]}"
            nprompt="${phase_nprompt[$pi]}"
            ngen="${phase_ngen[$pi]}"
            for (( ti=0; ti<${#threads[@]}; ti++ )); do
                th="${threads[$ti]}"
                elapsed=$(( $(date +%s) - start_ts ))
                if [[ "$elapsed" -ge "$CLOUD_THROUGHPUT_BUDGET_SECS" ]]; then
                    budget_hit=1
                    break 3
                fi
                call_no=$((call_no + 1))
                cell_json="$raw_dir/r${round}-${phase}-t${th}.json"
                log_info "cloud_throughput_bench: call $call_no/$total_calls round=$round phase=$phase threads=$th"
                rc=0
                run_with_timeout "$CLOUD_THROUGHPUT_PER_CALL_TIMEOUT" \
                    "$LLAMA_BENCH" -m "$MODEL_PATH" -ngl 0 -p "$nprompt" -n "$ngen" -t "$th" -r 1 -o json \
                    >"$cell_json" 2>>"$logfile" || rc=$?
                if [[ "$rc" -ne 0 ]]; then
                    log_warn "cloud_throughput_bench: call failed (rc=$rc) round=$round phase=$phase threads=$th; sample dropped (not fabricated), see $logfile"
                    rm -f "$cell_json"
                fi
            done
        done
        [[ "$budget_hit" -eq 1 ]] && break
    done

    local generated_at
    generated_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    local sweep_wall_seconds=$(( $(date +%s) - start_ts ))

    log_info "cloud_throughput_bench: $call_no/$total_calls llama-bench calls issued in ${sweep_wall_seconds}s (budget_hit=$budget_hit); aggregating"

    if ! CLOUD_RAW_DIR="$raw_dir" \
        CLOUD_OUT_JSON="$out_json" \
        CLOUD_OUT_MD="$out_md" \
        CLOUD_PLATFORM_SLUG="$platform_slug" \
        CLOUD_THREADS_CSV="$threads_csv" \
        CLOUD_REPS="$CLOUD_THROUGHPUT_REPS" \
        CLOUD_DECODE_TOKENS="$CLOUD_THROUGHPUT_DECODE_TOKENS" \
        CLOUD_PREFILL_TOKENS="$CLOUD_THROUGHPUT_PREFILL_TOKENS" \
        CLOUD_MODEL_ID="${MATRIX_MODEL:-${HF_FILE:-unknown}}" \
        CLOUD_MODEL_PATH="$MODEL_PATH" \
        CLOUD_LLAMA_BENCH="$LLAMA_BENCH" \
        CLOUD_GENERATED_AT="$generated_at" \
        CLOUD_SWEEP_WALL_SECONDS="$sweep_wall_seconds" \
        CLOUD_BUDGET_HIT="$budget_hit" \
        CLOUD_BUDGET_SECS="$CLOUD_THROUGHPUT_BUDGET_SECS" \
        python3 - <<'PYEOF' >>"$logfile" 2>&1
import json
import os
import statistics
from pathlib import Path

raw_dir = Path(os.environ["CLOUD_RAW_DIR"])
out_json = Path(os.environ["CLOUD_OUT_JSON"])
out_md = Path(os.environ["CLOUD_OUT_MD"])
platform_slug = os.environ["CLOUD_PLATFORM_SLUG"]
threads = [int(x) for x in os.environ["CLOUD_THREADS_CSV"].split(",") if x]
reps = int(os.environ["CLOUD_REPS"])
decode_tokens = int(os.environ["CLOUD_DECODE_TOKENS"])
prefill_tokens = int(os.environ["CLOUD_PREFILL_TOKENS"])
model_id = os.environ.get("CLOUD_MODEL_ID", "unknown")
model_path = os.environ.get("CLOUD_MODEL_PATH", "")
llama_bench = os.environ.get("CLOUD_LLAMA_BENCH", "")
generated_at = os.environ.get("CLOUD_GENERATED_AT", "")
sweep_wall_seconds = float(os.environ.get("CLOUD_SWEEP_WALL_SECONDS", "0"))
budget_hit = os.environ.get("CLOUD_BUDGET_HIT", "0") == "1"
budget_secs = int(os.environ.get("CLOUD_BUDGET_SECS", "0"))

# phase name -> (n_prompt, n_gen). decode and prefill are always separate
# cells/rows, never blended.
phases = {
    "decode": (0, decode_tokens),
    "prefill": (prefill_tokens, 0),
}

samples = {}  # (phase, threads) -> [tok/s samples]
for f in sorted(raw_dir.glob("r*-*.json")):
    try:
        data = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if not isinstance(data, list) or not data:
        continue
    entry = data[0]
    th = int(entry.get("n_threads", 0))
    n_prompt = int(entry.get("n_prompt", 0))
    n_gen = int(entry.get("n_gen", 0))
    phase = next(
        (name for name, (pp, ng) in phases.items() if pp == n_prompt and ng == n_gen),
        None,
    )
    if phase is None:
        continue
    vals = entry.get("samples_ts") or []
    samples.setdefault((phase, th), []).extend(float(v) for v in vals)

rows = []
skipped_cells = []
partial_cells = []
for phase_name, (pp, ng) in phases.items():
    for th in threads:
        vals = samples.get((phase_name, th), [])
        n = len(vals)
        if n == 0:
            skipped_cells.append(f"{phase_name}/threads={th}")
            rows.append({
                "phase": phase_name, "n_prompt": pp, "n_gen": ng,
                "threads": th, "agg": None,
                "note": "not measured this run (see meta.skipped_cells / meta.budget_hit)",
            })
            continue
        if n < reps:
            partial_cells.append(f"{phase_name}/threads={th} ({n}/{reps})")
        rows.append({
            "phase": phase_name, "n_prompt": pp, "n_gen": ng, "threads": th,
            "agg": {
                "n": n,
                "median_ts": statistics.median(vals),
                "stddev_ts": statistics.stdev(vals) if n > 1 else 0.0,
                "min_ts": min(vals),
                "max_ts": max(vals),
                "samples_ts": vals,
            },
        })

meta = {
    "platform": platform_slug,
    "lane": "verify-free-arm64 (GitHub-hosted ubuntu-24.04-arm, Neoverse-N2 class)",
    "model_id": model_id,
    "model_path": model_path,
    "llama_bench": llama_bench,
    "threads_swept": threads,
    "phases_swept": {name: list(v) for name, v in phases.items()},
    "reps_requested": reps,
    "generated_at": generated_at,
    "sweep_wall_seconds": sweep_wall_seconds,
    "budget_secs": budget_secs,
    "budget_hit": budget_hit,
    "skipped_cells": skipped_cells,
    "partial_cells": partial_cells,
    "protocol": (
        "Round-robin interleaved across (phase, threads): one pass over "
        "every cell per round, one llama-bench call per cell per round "
        "(-r 1, warmup enabled and discarded), never all reps of one cell "
        "before another. Median/stddev/min/max computed across rounds -- "
        "never a bare mean. This is a NEON/i8mm-only platform (Neoverse-N2's "
        "SVE2 is 128-bit, below KleidiAI's 256-bit SVE dispatch gate, and it "
        "has no SME2 at all -- see results/GROUND-TRUTH-DISPATCH.md, Finding "
        "2), so these are cloud-class Arm64 baseline throughput and "
        "thread-scaling numbers, never framed as an SME2 result."
    ),
    "candidateOnly": True,
    "canClaimAGI": False,
}

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n")

lines = []
lines.append(f"## Cloud-class throughput -- {platform_slug} (verify-free-arm64, Neoverse-N2 class)")
lines.append("")
lines.append(f"- Generated: {generated_at}")
lines.append(f"- Model: {model_id}")
lines.append(f"- Reps requested per cell: {reps}")
lines.append(
    "- NEON/i8mm-only platform: Neoverse-N2's SVE2 is 128-bit, below "
    "KleidiAI's 256-bit SVE dispatch gate, and it has no SME2 at all (see "
    "results/GROUND-TRUTH-DISPATCH.md, Finding 2). These are cloud-class "
    "Arm64 baseline throughput and thread-scaling numbers, not an SME2 "
    "result."
)
lines.append(
    "- Round-robin interleaved (phase x threads, one pass per round), "
    "median/stddev/min/max across rounds -- never a bare mean."
)
if budget_hit:
    lines.append(
        f"- Stage time budget ({budget_secs}s) was reached before every "
        "planned round-robin round completed; remaining cells were skipped "
        "rather than fabricated -- see skipped/partial counts below."
    )
lines.append("")
lines.append("| phase | threads | median tok/s | stddev | min | max | n/reps |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for row in rows:
    agg = row["agg"]
    if agg is None:
        lines.append(f"| {row['phase']} | {row['threads']} | _[not measured]_ | | | | 0/{reps} |")
    else:
        lines.append(
            f"| {row['phase']} | {row['threads']} | {agg['median_ts']:.1f} | "
            f"{agg['stddev_ts']:.2f} | {agg['min_ts']:.1f} | {agg['max_ts']:.1f} | "
            f"{agg['n']}/{reps} |"
        )
lines.append("")
if skipped_cells:
    lines.append(f"**Skipped cells (0 samples):** {', '.join(skipped_cells)}")
    lines.append("")
if partial_cells:
    lines.append(f"**Partial cells (fewer than {reps} samples):** {', '.join(partial_cells)}")
    lines.append("")

out_md.write_text("\n".join(lines) + "\n")

total_expected = len(phases) * len(threads) * reps
total_measured = sum(row["agg"]["n"] for row in rows if row["agg"] is not None)
print(
    f"cloud_throughput_bench: {total_measured}/{total_expected} samples "
    f"collected ({len(skipped_cells)} cell(s) fully skipped, "
    f"{len(partial_cells)} partial)"
)
if total_measured == 0:
    raise SystemExit(1)
PYEOF
    then
        record_stage "$stage" FAIL "cloud-throughput aggregation produced zero samples or errored, see $logfile"
        rm -rf "$raw_dir"
        return 1
    fi

    rm -rf "$raw_dir"
    record_stage "$stage" OK "wrote $out_json / $out_md ($call_no llama-bench calls, budget_hit=$budget_hit; log: $logfile)"
    return 0
}

if [[ "$CLOUD_THROUGHPUT" == "1" ]]; then
    if run_with_timeout "$CLOUD_THROUGHPUT_STAGE_TIMEOUT_SECS" run_cloud_throughput_bench; then
        exit 0
    else
        record_stage "$STAGE" FAIL "cloud_throughput_bench timed out or failed after ${CLOUD_THROUGHPUT_STAGE_TIMEOUT_SECS}s"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Default path (CLOUD_THROUGHPUT unset/0): unchanged tools/bench.py sweep.
# ---------------------------------------------------------------------------

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
