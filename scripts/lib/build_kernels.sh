#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/build_kernels.sh — build whatever is under kernels/, if
# anything has landed there yet. See the CROSS-PACKAGE CONTRACT in
# scripts/common.sh for the candidate build systems tried, in order:
#   1. kernels/Makefile        -> `make -C kernels all`
#   2. kernels/CMakeLists.txt  -> configure+build into kernels/build/
#   3. kernels/build.sh        -> run directly
# If none exist yet this is a SKIP, not a failure — the kernels/ package may
# simply not have landed yet.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

: "${JOBS:=$(build_jobs_cap)}"
STAGE="build_kernels"

if [[ -f "$REPO_ROOT/kernels/Makefile" ]]; then
    log_info "kernels/Makefile found; running 'make -C kernels all'"
    if make -C "$REPO_ROOT/kernels" all -j"$JOBS" >"$LOG_DIR/build_kernels.log" 2>&1; then
        record_stage "$STAGE" OK "make -C kernels all succeeded"
    else
        record_stage "$STAGE" FAIL "make -C kernels all failed, see $LOG_DIR/build_kernels.log"
        tail -n 40 "$LOG_DIR/build_kernels.log" >&2 || true
        exit 1
    fi
elif [[ -f "$REPO_ROOT/kernels/CMakeLists.txt" ]]; then
    log_info "kernels/CMakeLists.txt found; configuring+building kernels/build"
    if {
        cmake -S "$REPO_ROOT/kernels" -B "$REPO_ROOT/kernels/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$REPO_ROOT/kernels/build" -j"$JOBS"
    } >"$LOG_DIR/build_kernels.log" 2>&1; then
        record_stage "$STAGE" OK "kernels/CMakeLists.txt build succeeded"
    else
        record_stage "$STAGE" FAIL "kernels CMake build failed, see $LOG_DIR/build_kernels.log"
        tail -n 40 "$LOG_DIR/build_kernels.log" >&2 || true
        exit 1
    fi
elif [[ -x "$REPO_ROOT/kernels/build.sh" ]]; then
    log_info "kernels/build.sh found; running it"
    if (cd "$REPO_ROOT/kernels" && ./build.sh) >"$LOG_DIR/build_kernels.log" 2>&1; then
        record_stage "$STAGE" OK "kernels/build.sh succeeded"
    else
        record_stage "$STAGE" FAIL "kernels/build.sh failed, see $LOG_DIR/build_kernels.log"
        tail -n 40 "$LOG_DIR/build_kernels.log" >&2 || true
        exit 1
    fi
else
    record_stage "$STAGE" SKIP "no kernels/Makefile, kernels/CMakeLists.txt, or kernels/build.sh found yet"
fi
