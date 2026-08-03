#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/run_correctness_tests.sh — run whatever correctness tests have
# landed, in this order (all that exist are run; none existing is a SKIP):
#   1. tests/run_tests.sh
#   2. tests/Makefile              (target: test)
#   3. kernels/build (CTest)       -> `ctest` if kernels/CMakeLists.txt was
#      built via scripts/lib/build_kernels.sh (enable_testing() + add_test();
#      this repo's kernels/ package uses this — e.g. kernel_test comparing
#      the hand-written SME2 kernel against tests/kernels/test_correctness.c)
#   4. kernels/Makefile            (target: test) — fallback for a
#      Makefile-based kernels/ build instead of CMake.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

STAGE="correctness_tests"
ran_any=0
failed=0

run_and_record() {
    local label="$1" logfile="$2"
    shift 2
    log_info "running: $* (log: $logfile)"
    if "$@" >"$logfile" 2>&1; then
        log_ok "$label passed"
    else
        log_fail "$label FAILED, see $logfile"
        tail -n 60 "$logfile" >&2 || true
        failed=1
    fi
    ran_any=1
}

if [[ -x "$REPO_ROOT/tests/run_tests.sh" ]]; then
    run_and_record "tests/run_tests.sh" "$LOG_DIR/tests-run_tests.log" \
        "$REPO_ROOT/tests/run_tests.sh"
elif [[ -f "$REPO_ROOT/tests/run_tests.sh" ]]; then
    run_and_record "tests/run_tests.sh" "$LOG_DIR/tests-run_tests.log" \
        bash "$REPO_ROOT/tests/run_tests.sh"
fi

if [[ -f "$REPO_ROOT/tests/Makefile" ]]; then
    run_and_record "make -C tests test" "$LOG_DIR/tests-make.log" \
        make -C "$REPO_ROOT/tests" test
fi

if [[ -f "$REPO_ROOT/kernels/build/CTestTestfile.cmake" ]] && command -v ctest >/dev/null 2>&1; then
    run_and_record "ctest (kernels/build)" "$LOG_DIR/kernels-ctest.log" \
        ctest --test-dir "$REPO_ROOT/kernels/build" --output-on-failure
elif [[ -f "$REPO_ROOT/kernels/CMakeLists.txt" && ! -f "$REPO_ROOT/kernels/build/CTestTestfile.cmake" ]]; then
    log_warn "kernels/CMakeLists.txt exists but kernels/build/CTestTestfile.cmake does not — run scripts/lib/build_kernels.sh first"
fi

if [[ -f "$REPO_ROOT/kernels/Makefile" ]] && grep -qE '^test:' "$REPO_ROOT/kernels/Makefile" 2>/dev/null; then
    run_and_record "make -C kernels test" "$LOG_DIR/kernels-test.log" \
        make -C "$REPO_ROOT/kernels" test
fi

if [[ "$ran_any" -eq 0 ]]; then
    record_stage "$STAGE" SKIP "no test entrypoint found yet (tests/run_tests.sh, tests/Makefile, kernels ctest, or kernels/Makefile[test])"
elif [[ "$failed" -eq 1 ]]; then
    record_stage "$STAGE" FAIL "one or more correctness test suites failed, see logs in $LOG_DIR"
    exit 1
else
    record_stage "$STAGE" OK "all discovered correctness test suites passed"
fi
