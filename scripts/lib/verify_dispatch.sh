#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/verify_dispatch.sh — drives tools/verify_dispatch.py (the
# lldb/gdb symbol-level dispatch verifier for Finding 1 / Finding 2) across
# the thread sweep.
#
# This calls the REAL, landed CLI (`--binary --model --threads ... --out`),
# discovered dynamically so a rename or reimplementation under a different
# name in tools/ is still picked up — see find_entrypoint / the
# CROSS-PACKAGE CONTRACT in scripts/common.sh. The exact flags below were
# derived from `python3 tools/verify_dispatch.py --help` and confirmed by
# actually running it against this machine's llama-cli + GGUF (see
# results/dispatch-ledger-*.json and the OKF run log for this package for
# the real output).
#
# Deliberately never pass --assert here: a NO_DISPATCH_OBSERVED /
# SILENT_FALLBACK verdict is not a bug in this project — for several of the
# configurations in this sweep it IS Finding 1 (the whole point is that the
# accelerated kernel silently doesn't run at the runner's default thread
# count). Asserting on that would fail CI for demonstrating the exact thing
# this repo exists to demonstrate. Verdicts are recorded in the ledger for a
# human to read, never used to gate this step's exit status.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

: "${THREAD_SWEEP:=$(default_thread_sweep)}"
: "${VERIFY_DISPATCH_L2_TIMEOUT:=}"   # empty = let the tool use its own default
: "${VERIFY_DISPATCH_L3_TIMEOUT:=}"   # empty = let the tool use its own default
: "${VERIFY_DISPATCH_SKIP_L3:=0}"     # 1 = fast L1+L2-only smoke test, no debugger
STAGE="verify_dispatch"

entrypoint="$(find_entrypoint tools verify_dispatch || true)"
if [[ -z "$entrypoint" ]]; then
    record_stage "$STAGE" SKIP "no tools/verify_dispatch(.py|.sh) found yet"
    exit 0
fi

if [[ ! -x "$LLAMA_CLI" ]]; then
    record_stage "$STAGE" FAIL "llama-cli not built at $LLAMA_CLI; run build_llamacpp first"
    exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    record_stage "$STAGE" FAIL "model not present at $MODEL_PATH; run fetch_model first"
    exit 1
fi

platform_slug="$(uname -s)-$(uname -m)"
out_json="$RESULTS_DIR/dispatch-ledger-${platform_slug}.json"
logfile="$LOG_DIR/verify_dispatch.log"

args=(--binary "$LLAMA_CLI" --model "$MODEL_PATH" --threads "$THREAD_SWEEP" --out "$out_json")
[[ -n "$VERIFY_DISPATCH_L2_TIMEOUT" ]] && args+=(--l2-timeout "$VERIFY_DISPATCH_L2_TIMEOUT")
[[ -n "$VERIFY_DISPATCH_L3_TIMEOUT" ]] && args+=(--l3-timeout "$VERIFY_DISPATCH_L3_TIMEOUT")
[[ "$VERIFY_DISPATCH_SKIP_L3" == "1" ]] && args+=(--skip-l3)

log_info "found dispatch verifier: $entrypoint"
log_info "threads=$THREAD_SWEEP -> $out_json"

# 20 minutes is generous headroom: measured on an Apple M4 Max, 4 configs
# (2 threads x 2 built-in workloads) took ~17s total (~3.5s/config with the
# lldb L3 debugger attached), so even the full default 5-thread x
# 2-workload = 10-config sweep should land well under a minute in practice.
if run_with_timeout 1200 run_entrypoint "$entrypoint" "${args[@]}" >"$logfile" 2>&1; then
    record_stage "$STAGE" OK "wrote $out_json (log: $logfile)"
else
    # A non-zero exit here means the tool itself errored (bad path, crashed
    # subprocess, timed out) — NOT a dispatch verdict (see file header: we
    # never pass --assert, so verdict content never affects exit status).
    record_stage "$STAGE" FAIL "verify_dispatch.py exited non-zero (tool error, not a verdict) — see $logfile"
    tail -n 40 "$logfile" >&2 || true
    exit 1
fi
