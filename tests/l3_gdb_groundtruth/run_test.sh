#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
#
# Ground-truth test for the L3 gdb dispatch probe (Linux/aarch64 only).
#
# WHY THIS EXISTS
# ---------------
# The first revision of tools/dispatch_probe.gdb reported 0 hits for every config on
# GitHub's free ubuntu-24.04-arm runner, with no error and exit status 0 -- it used
# `rbreak <regex>` before `run`, but ggml dlopen's its CPU backend after process start
# and `rbreak` (unlike `break`) does not create pending breakpoints. A tool whose entire
# purpose is distinguishing "the kernel ran" from "the kernel did not run" shipped a
# path that could not tell either from "we never instrumented anything."
#
# This test builds a shared library with a KNOWN call count, dlopen's it exactly the way
# ggml does, and asserts the probe recovers that count exactly. It fails loudly if the
# probe is ever silently uninstrumented again.
#
# Usage:  tests/l3_gdb_groundtruth/run_test.sh [calls_per_symbol]
# Exit:   0 on pass, non-zero on failure. Skips (0) with a clear message on non-Linux.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CALLS="${1:-7}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "[skip] L3 gdb ground-truth test is Linux-only (this is $(uname -s); macOS uses the lldb probe)."
    exit 0
fi
if ! command -v gdb >/dev/null 2>&1; then
    echo "[skip] gdb not on PATH; cannot run the L3 gdb ground-truth test." >&2
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

cc -shared -fPIC -g -O0 -o libkai_fake.so "$HERE/libkai_fake.c"
cc -g -O0 -o fake_llama "$HERE/main.c" -ldl

echo "[info] built fake backend; ground truth = $CALLS calls per symbol"

# Drive the REAL probe implementation, not a copy, so this tests what ships.
python3 - "$REPO_ROOT" "$CALLS" <<'PY'
import json, os, sys

repo_root, calls = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, os.path.join(repo_root, "tools"))
import verify_dispatch as vd

lib = os.path.abspath("libkai_fake.so")
syms = vd.enumerate_dispatch_symbols(lib, vd.DEFAULT_DISPATCH_REGEX)
print(f"[info] enumerate_dispatch_symbols found {len(syms)}: {syms}")
if len(syms) != 2:
    print(f"FAIL: expected 2 kai_run_matmul_* symbols, got {len(syms)}")
    sys.exit(1)

script = vd._render_template(vd.GDB_TEMPLATE_PATH, {"DP_SYMBOLS": repr(syms)})
with open("probe.gdb", "w") as fh:
    fh.write(script)

rc, out, timed_out, wall = vd._run_with_timeout(
    ["gdb", "-q", "-batch", "-x", "probe.gdb", "--args", "./fake_llama", str(calls)],
    timeout=120.0,
)
if timed_out:
    print("FAIL: gdb probe timed out")
    sys.exit(1)

created = None
hits = {}
in_block = False
for line in out.splitlines():
    if line.startswith("DP_BREAKPOINTS_CREATED "):
        created = int(line.split()[1])
    elif line.strip() == "DISPATCH_PROBE_RESULT_BEGIN":
        in_block = True
    elif line.strip() == "DISPATCH_PROBE_RESULT_END":
        in_block = False
    elif in_block and line.startswith("DP_HIT "):
        _, name, count = line.split(" ", 2)
        hits[name] = int(count)

print(f"[info] breakpoints created: {created}")
print(f"[info] hits: {json.dumps(hits, indent=1)}")

failures = []
if created != 2:
    failures.append(f"expected 2 breakpoints created, got {created}")
if len(hits) != 2:
    failures.append(f"expected hits for 2 symbols, got {len(hits)}")
for sym, n in hits.items():
    if n != calls:
        failures.append(f"{sym}: expected exactly {calls} calls, counted {n}")

# The counts must be true CALL counts, so family classification must work on them too.
fams = {}
for sym, n in hits.items():
    fams[vd.classify_symbol_family(sym)] = fams.get(vd.classify_symbol_family(sym), 0) + n
print(f"[info] hits by family: {fams}")
if set(fams) != {"dotprod", "sme2"}:
    failures.append(f"expected families {{dotprod, sme2}}, got {set(fams)}")

if failures:
    for f in failures:
        print("FAIL: " + f)
    sys.exit(1)
print(f"PASS: L3 gdb probe recovered exact ground-truth call counts ({calls} per symbol, 2 families)")
PY
