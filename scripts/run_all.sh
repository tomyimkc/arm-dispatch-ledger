#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/run_all.sh — bootstrap + verify + bench + emit ledger, in one
# command. This is the full local reproduction path: bootstrap (setup.sh),
# capture hardware feature facts, run correctness tests, verify dispatch
# across the thread sweep, run a reduced bench, then emit results/LEDGER.md.
#
# Usage:
#   ./scripts/run_all.sh
#
# Every stage is independently idempotent/best-effort (see
# scripts/common.sh's CROSS-PACKAGE CONTRACT): a stage whose tools/kernels/
# tests entrypoint has not landed yet is recorded as SKIP, not FAIL, so this
# script is safe to run at any point during development and will pick up
# more real work automatically as kernels/, tools/, and tests/ land.
#
# Exit status: non-zero if any stage that DID find an entrypoint to run
# failed. A SKIP never causes a non-zero exit on its own.
set -uo pipefail # deliberately not -e: we want every stage to run and be
                  # recorded even if an earlier one fails; overall exit
                  # status is computed explicitly at the end.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$SCRIPT_DIR/common.sh"

# Fresh run: start stage-status.tsv clean so this run's summary doesn't mix
# with a previous one. (Per-stage logs in results/logs/ are still
# overwritten in place by each stage, which is fine — only the latest run's
# logs are ever interesting.)
: >"$STAGE_STATUS_FILE"

overall_status=0

run_stage_script() {
    local script="$1"
    if bash "$SCRIPT_DIR/lib/$script"; then
        return 0
    else
        overall_status=1
        return 1
    fi
}

log_info "=== run_all.sh starting ==="

log_info "--- bootstrap (scripts/setup.sh) ---"
if ! bash "$SCRIPT_DIR/setup.sh"; then
    log_fail "bootstrap failed; aborting (verify/bench/ledger stages need a built llama-cli + model)"
    overall_status=1
    # Still emit whatever ledger we can, so a failed run's artifact is
    # legible rather than empty.
    bash "$SCRIPT_DIR/lib/emit_ledger.sh" || true
    exit "$overall_status"
fi

log_info "--- capture hardware features ---"
run_stage_script capture_hw_features.sh || true

log_info "--- correctness tests ---"
run_stage_script run_correctness_tests.sh || true

log_info "--- verify dispatch (thread sweep) ---"
run_stage_script verify_dispatch.sh || true

log_info "--- reduced bench ---"
run_stage_script run_bench.sh || true

log_info "--- emit ledger ---"
run_stage_script emit_ledger.sh || true

log_info "=== run_all.sh finished (exit status will be $overall_status) ==="
log_info "see $RESULTS_DIR/LEDGER.md for the human-readable summary"

exit "$overall_status"
