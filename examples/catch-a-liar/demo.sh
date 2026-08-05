#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
#
# examples/catch-a-liar/demo.sh -- the 2-minute, no-Arm-hardware,
# no-model-download proof that `polygraph check` catches a real silent
# fallback. Compiles two builds of examples/catch-a-liar/liar.c (see that
# file's header) and runs `tools/polygraph check` against both:
#
#   build/liar    prints "using fast path: yes" but never calls it  -> MISMATCH, exit 1
#   build/honest  prints the same line and genuinely calls it       -> MATCH,    exit 0
#
# Both directions matter: a detector that always says "mismatch" is
# worthless. This script shows both, with the literal exit code from each.
#
# Usage:
#   ./examples/catch-a-liar/demo.sh          # via `make demo` from repo root
#
# Design rules (mirrors demo/demo.sh):
#   - Every command that matters is echoed before it runs.
#   - Never abort mid-script: `tools/polygraph` not existing yet is a real,
#     expected state while it is still being built -- this script reports
#     that plainly (exit 2) instead of dying on "command not found".
#   - Idempotent: writes only under examples/catch-a-liar/build/, which is
#     gitignored (see .gitignore's blanket `build/` rule).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 2

EXAMPLE_DIR="examples/catch-a-liar"
BUILD_DIR="$EXAMPLE_DIR/build"
CC="${CC:-cc}"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
run()  { printf '$ %s\n' "$*"; eval "$@"; }

say "== Step 1/3: compile two builds of ${EXAMPLE_DIR}/liar.c =="
mkdir -p "$BUILD_DIR"
run "$CC -O0 -g -o $BUILD_DIR/liar   $EXAMPLE_DIR/liar.c"
liar_cc_rc=$?
run "$CC -O0 -g -o $BUILD_DIR/honest $EXAMPLE_DIR/liar.c -DACTUALLY_FAST"
honest_cc_rc=$?
if [ "$liar_cc_rc" -ne 0 ] || [ "$honest_cc_rc" -ne 0 ]; then
    echo "ERROR: compilation failed (liar rc=$liar_cc_rc, honest rc=$honest_cc_rc)." >&2
    exit 2
fi

say "== Both binaries print the identical banner -- that line proves nothing =="
run "$BUILD_DIR/liar"
run "$BUILD_DIR/honest"

POLYGRAPH="tools/polygraph"
if [ ! -e "$POLYGRAPH" ]; then
    say "== tools/polygraph not found in this checkout =="
    echo "This example is built against the CLI contract (docs/QUICKSTART.md), but" >&2
    echo "tools/polygraph itself is not present yet. Nothing further to run --" >&2
    echo "once it lands, this script will invoke:" >&2
    echo "  $POLYGRAPH check --binary $BUILD_DIR/liar   --symbols '^fast_path_sum\$' --run $BUILD_DIR/liar   --level 3" >&2
    echo "  $POLYGRAPH check --binary $BUILD_DIR/honest --symbols '^fast_path_sum\$' --run $BUILD_DIR/honest --level 3" >&2
    exit 2
fi
if [ -x "$POLYGRAPH" ]; then
    RUN_POLYGRAPH="$POLYGRAPH"
else
    RUN_POLYGRAPH="python3 $POLYGRAPH"
fi

say "== Step 2/3: polygraph check on the LIE (expect MISMATCH, exit 1) =="
run "$RUN_POLYGRAPH check catch-a-liar"
liar_rc=$?
echo "exit code: $liar_rc"

say "== Step 3/3: polygraph check on the TRUTH (expect MATCH, exit 0) =="
run "$RUN_POLYGRAPH check --binary $BUILD_DIR/honest --symbols '^fast_path_sum\$' --run $BUILD_DIR/honest --level 3"
honest_rc=$?
echo "exit code: $honest_rc"

say "== Summary =="
overall=0
if [ "$liar_rc" -eq 1 ]; then
    echo "PASS: liar   -> exit 1 (MISMATCH), as expected"
else
    echo "FAIL: liar   -> exit $liar_rc, expected 1"
    overall=1
fi
if [ "$honest_rc" -eq 0 ]; then
    echo "PASS: honest -> exit 0 (MATCH), as expected"
else
    echo "FAIL: honest -> exit $honest_rc, expected 0"
    overall=1
fi

exit "$overall"
