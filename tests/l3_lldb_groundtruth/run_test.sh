#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
#
# Ground-truth test for the L3 lldb dispatch probe (macOS/Darwin only).
#
# WHY THIS EXISTS
# ---------------
# 100% of the SME2 dispatch evidence in this repo (results/GROUND-TRUTH-DISPATCH.md,
# results/AUTODEFAULTS.md, results/REMEASURE-2026-08-04-QUIET.md, and every "SME2
# actually dispatched N times" number in the ledger) comes from tools/dispatch_probe.lldb
# via tools/verify_dispatch.py's run_l3_lldb(). Until this test existed, that probe had
# NEVER been checked against a call count we actually knew was true -- every number it
# ever produced was trusted on the strength of "the breakpoint resolved and lldb didn't
# error", not on a measured ground truth.
#
# That gap mattered: tests/l3_gdb_groundtruth/ (the Linux counterpart) exists because
# the FIRST revision of dispatch_probe.gdb silently reported 0 hits for every config on
# real CI, with no error and exit status 0 -- `rbreak` (unlike `break`) does not create
# pending breakpoints, so a regex evaluated before ggml dlopen's its CPU backend matches
# nothing. lldb's regex breakpoints auto-resolve when a solib loads later, which is
# exactly why the macOS lane never showed that symptom -- but "different bug class"
# is not the same as "no bug class", and nobody had ever proven it with a fixture whose
# true call count was known in advance. This test closes that gap for lldb.
#
# It ALSO found a real bug in ITS OWN fixture while being written: an early cut of
# libkai_fake.c, built with -O2 and no noinline hint, let the compiler inline both
# kai_run_matmul_* calls into their same-translation-unit caller. The exported symbol
# still existed and lldb's regex breakpoint still resolved against it (shown as
# "resolved"), but the probe reported 0 hits for a real 7/11-call workload -- a
# "resolved but silent" false negative, self-inflicted by the fixture's build flags,
# not by dispatch_probe.lldb. See libkai_fake.c's header comment for the fix
# (__attribute__((noinline))), which makes the result independent of optimization
# level. Left in as a cautionary note: a ground-truth fixture can lie too.
#
# WHAT THIS TEST DOES DIFFERENTLY FROM tests/l3_gdb_groundtruth/run_test.sh
# ---------------------------------------------------------------------------
# The gdb version hand-rolls its own `gdb -q -batch -x ... --args ...` invocation
# because run_l3_gdb() needs an explicit symbol list (enumerated via nm) and a
# lib_path it doesn't have here. lldb needs neither: `breakpoint set --func-regex`
# resolves against symbols in ANY module, loaded now or later, with no upfront
# enumeration. That means this test can call tools/verify_dispatch.py's
# run_l3_lldb(binary, model, threads, prompt, n_predict, dispatch_regex, timeout, env)
# DIRECTLY -- the exact function verify_dispatch.py's own sweep calls for every
# (thread, workload) config on Darwin -- instead of reimplementing any part of its
# command construction or output parsing. fake_llama (main.c) exists purely so this
# test has something that accepts run_l3_lldb()'s hardcoded llama-cli-shaped argv
# (`-m ... -p ... -n ... -no-cnv -st --simple-io -t ...`) and turns `-n` into a known
# per-symbol call count.
#
# Ground truth: dotprod-family symbol called exactly CALLS times, sme2-family symbol
# called exactly CALLS + 4 times (deliberately different counts, so this also checks
# L3Result.kernel_family_executed -- the argmax-by-hit-count family -- not just the
# raw per-symbol counts). Keep the "+ 4" in sync with SME2_EXTRA_CALLS in main.c.
#
# Usage:  tests/l3_lldb_groundtruth/run_test.sh [calls_per_dotprod_symbol]
# Exit:   0 on pass, non-zero on failure. Skips (0) with a clear message on non-Darwin
#         or when lldb is not on PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CALLS="${1:-7}"
SME2_EXTRA_CALLS=4 # must match SME2_EXTRA_CALLS in main.c

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[skip] L3 lldb ground-truth test is Darwin-only (this is $(uname -s); Linux uses the gdb probe -- see tests/l3_gdb_groundtruth/)."
    exit 0
fi
if ! command -v lldb >/dev/null 2>&1; then
    echo "[skip] lldb not on PATH; cannot run the L3 lldb ground-truth test. Install Xcode Command Line Tools (xcode-select --install)." >&2
    exit 0
fi
if ! command -v cc >/dev/null 2>&1; then
    echo "[skip] cc (clang) not on PATH; cannot build the ground-truth fixture." >&2
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

cc -dynamiclib -fPIC -O0 -o libkai_fake.dylib "$HERE/libkai_fake.c"
cc -O0 -o fake_llama "$HERE/main.c" -ldl

DOTPROD_EXPECTED="$CALLS"
SME2_EXPECTED=$((CALLS + SME2_EXTRA_CALLS))
echo "[info] built fake backend + fake_llama; ground truth = dotprod:$DOTPROD_EXPECTED sme2:$SME2_EXPECTED"

# Drive the REAL run_l3_lldb() implementation, not a copy, so this tests what ships.
python3 - "$REPO_ROOT" "$DOTPROD_EXPECTED" "$SME2_EXPECTED" "$WORK" <<'PY'
import json, os, sys

repo_root, dotprod_expected, sme2_expected, work = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, os.path.join(repo_root, "tools"))
import verify_dispatch as vd

binary = os.path.join(work, "fake_llama")
run_env = dict(os.environ)
# Sidestep any ambiguity about lldb's inferior launch cwd (undocumented/not
# guaranteed): main.c resolves the dylib via this env var, which run_l3_lldb()
# threads straight through to the lldb (and thus the inferior's) environment.
run_env["L3_LLDB_GT_LIBDIR"] = work

print(f"[info] dispatch_regex={vd.DEFAULT_DISPATCH_REGEX!r}")
print(f"[info] driving the real tools/dispatch_probe.lldb via verify_dispatch.run_l3_lldb()")

l3 = vd.run_l3_lldb(
    binary=binary,
    model="fake-model.gguf",
    threads=1,
    prompt="L3 lldb ground-truth probe",
    n_predict=dotprod_expected,
    dispatch_regex=vd.DEFAULT_DISPATCH_REGEX,
    timeout=90.0,
    env=run_env,
)

print(f"[info] L3Result: completed={l3.completed} timed_out={l3.timed_out} error={l3.error}")
print(f"[info] command: {' '.join(l3.command)}")
print(f"[info] hits_by_symbol: {json.dumps(l3.hits_by_symbol, indent=1)}")
print(f"[info] hits_by_family: {l3.hits_by_family}")
print(f"[info] kernel_family_executed: {l3.kernel_family_executed}")
print(f"[info] total_hits: {l3.total_hits}  wall_time_sec: {l3.wall_time_sec:.2f}")

failures = []
if not l3.completed:
    failures.append(f"L3 probe did not complete: error={l3.error!r}")
if l3.timed_out:
    failures.append("L3 probe timed out")
if l3.error:
    failures.append(f"L3 probe reported an error even though it completed: {l3.error!r}")
if len(l3.hits_by_symbol) != 2:
    failures.append(f"expected hits for exactly 2 symbols, got {len(l3.hits_by_symbol)}: {l3.hits_by_symbol}")

dotprod_syms = [s for s in l3.hits_by_symbol if vd.classify_symbol_family(s) == "dotprod"]
sme2_syms = [s for s in l3.hits_by_symbol if vd.classify_symbol_family(s) == "sme2"]
if len(dotprod_syms) != 1:
    failures.append(f"expected exactly 1 dotprod-classified symbol, got {dotprod_syms}")
if len(sme2_syms) != 1:
    failures.append(f"expected exactly 1 sme2-classified symbol, got {sme2_syms}")

if dotprod_syms and l3.hits_by_symbol[dotprod_syms[0]] != dotprod_expected:
    failures.append(
        f"{dotprod_syms[0]}: expected exactly {dotprod_expected} calls, counted {l3.hits_by_symbol[dotprod_syms[0]]}"
    )
if sme2_syms and l3.hits_by_symbol[sme2_syms[0]] != sme2_expected:
    failures.append(
        f"{sme2_syms[0]}: expected exactly {sme2_expected} calls, counted {l3.hits_by_symbol[sme2_syms[0]]}"
    )

if l3.hits_by_family.get("dotprod") != dotprod_expected:
    failures.append(f"hits_by_family['dotprod']: expected {dotprod_expected}, got {l3.hits_by_family.get('dotprod')}")
if l3.hits_by_family.get("sme2") != sme2_expected:
    failures.append(f"hits_by_family['sme2']: expected {sme2_expected}, got {l3.hits_by_family.get('sme2')}")

expected_total = dotprod_expected + sme2_expected
if l3.total_hits != expected_total:
    failures.append(f"total_hits: expected {expected_total}, got {l3.total_hits}")

# sme2 is deliberately called more often (ground truth = dotprod_expected + 4), so this
# also exercises the argmax logic that picks kernel_family_executed, not just raw counts.
if l3.kernel_family_executed != "sme2":
    failures.append(f"kernel_family_executed: expected 'sme2' (the majority family), got {l3.kernel_family_executed!r}")

if failures:
    for f in failures:
        print("FAIL: " + f)
    sys.exit(1)
print(
    f"PASS: L3 lldb probe (run_l3_lldb(), the real shipped code path) recovered exact "
    f"ground-truth call counts (dotprod={dotprod_expected}, sme2={sme2_expected}) and "
    f"family classification (incl. kernel_family_executed) is correct"
)
PY
