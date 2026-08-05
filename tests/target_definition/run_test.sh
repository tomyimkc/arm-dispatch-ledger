#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
#
# Ground-truth test for tools/polygraph's DECLARATIVE target-JSON pipeline.
#
# WHY THIS EXISTS
# ---------------
# tests/l3_lldb_groundtruth/ and tests/l3_gdb_groundtruth/ prove run_l3_lldb()/
# run_l3_gdb() recover exact ground-truth call counts -- but they call those
# functions directly from Python. Neither one ever runs tools/polygraph itself,
# and neither one ever loads a tools/targets/*.json file. This test closes that
# gap: it builds a tiny fixture with a KNOWN call count (like the other two),
# but drives it through the REAL, SHIPPED `tools/polygraph check <target.json>`
# CLI end to end -- L1 (nm symbol scan) -> L2 (selection-log regex match) ->
# L3 (breakpoint dispatch count) -> verdict -> exit code -- exactly the path a
# judge running `polygraph check <preset> --binary ...` takes.
#
# It also asserts the one behavior none of the real presets can cleanly prove
# in isolation, because none of them control ground truth: that a declarative
# target whose L2 "advertised" claim does NOT match what L3 actually dispatched
# is correctly classified SILENT_FALLBACK with exit code 1, and a target whose
# claim DOES match is DISPATCHED with exit code 0 -- the CLI contract's core
# promise, checked against a fixture where we know which one is true.
#
# Runs on both macOS (lldb) and Linux (gdb) via tools/polygraph's own
# --l3-debugger auto, using ONE shared C fixture (libfake_accel.c / main.c)
# rather than two platform-specific copies.
#
# Usage:  tests/target_definition/run_test.sh
# Exit:   0 on pass, non-zero on failure. Skips (0) with a clear message if
#         this platform has neither lldb nor gdb, or no C compiler.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
POLYGRAPH="$REPO_ROOT/tools/polygraph"

OS="$(uname -s)"
if ! command -v cc >/dev/null 2>&1; then
    echo "[skip] cc not on PATH; cannot build the target_definition fixture." >&2
    exit 0
fi
if [[ "$OS" == "Darwin" ]]; then
    if ! command -v lldb >/dev/null 2>&1; then
        echo "[skip] Darwin but lldb not on PATH; install Xcode Command Line Tools." >&2
        exit 0
    fi
    LIB_NAME="libfake_accel.dylib"
    CC_LIB_FLAGS=(-dynamiclib -fPIC -O0)
elif [[ "$OS" == "Linux" ]]; then
    if ! command -v gdb >/dev/null 2>&1; then
        echo "[skip] Linux but gdb not on PATH." >&2
        exit 0
    fi
    LIB_NAME="libfake_accel.so"
    CC_LIB_FLAGS=(-shared -fPIC -O0)
else
    echo "[skip] target_definition test not implemented for $OS." >&2
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

cc "${CC_LIB_FLAGS[@]}" -o "$LIB_NAME" "$HERE/libfake_accel.c"
cc -O0 -o fake_main "$HERE/main.c" -ldl 2>/dev/null || cc -O0 -o fake_main "$HERE/main.c"

echo "[info] built $LIB_NAME + fake_main in $WORK"

export POLY_TD_LIBDIR="$WORK"

# ---------------------------------------------------------------------------
# Scenario 1: the advertised family (avx) is exactly what dispatches -> MATCH.
# Scenario 2: avx is advertised (main.c always prints it) but scalar actually
#             dispatches -> SILENT_FALLBACK. This is the load-bearing check:
#             a declarative target correctly catching a claim/reality mismatch
#             it has never seen before, on a fixture where we control both.
# ---------------------------------------------------------------------------
python3 - "$REPO_ROOT" "$POLYGRAPH" "$WORK" "$LIB_NAME" <<'PY'
import json
import os
import subprocess
import sys

repo_root, polygraph, work, lib_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
target_json = os.path.join(repo_root, "tests", "target_definition", "target.json")
binary = os.path.join(work, "fake_main")
lib = os.path.join(work, lib_name)

failures = []


def run_check(n, actual):
    cmd = [
        sys.executable, polygraph, "check", target_json,
        "--binary", binary, "--lib-name", lib,
        "--param", f"n={n}", "--param", f"actual={actual}",
        "--json", "--quiet", "--l3-debugger", "auto",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"FAIL: --json output was not valid JSON for n={n} actual={actual}: {e}")
        print("stdout:", proc.stdout[:2000])
        print("stderr:", proc.stderr[:2000])
        sys.exit(1)
    return proc.returncode, result


# --- L1 sanity (shared ground truth for both scenarios: 2 matching symbols) ---
rc, res = run_check(7, "avx")
l1 = res["l1"]
print(f"[info] L1: matched={l1['matched_symbol_count']} total={l1['total_symbol_count']} "
      f"families={l1['symbols_by_family']}")
if l1["matched_symbol_count"] != 2:
    failures.append(f"L1 matched_symbol_count: expected 2, got {l1['matched_symbol_count']}")
if l1["symbols_by_family"] != {"avx": 1, "scalar": 1}:
    failures.append(f"L1 symbols_by_family: expected {{'avx': 1, 'scalar': 1}}, got {l1['symbols_by_family']}")

# --- Scenario 1: advertised == actual (avx, n=7) -> DISPATCHED, exit 0 ---
print(f"[info] scenario 1 (match): verdict={res['verdict']} exit_code={res['exit_code']} "
      f"advertised={res['advertised']} l3.total_hits={res['l3']['total_hits']} "
      f"l3.hits_by_family={res['l3']['hits_by_family']}")
if res["advertised"] != "avx":
    failures.append(f"scenario 1: expected advertised='avx', got {res['advertised']!r}")
if res["verdict"] != "AVX_DISPATCHED":
    failures.append(f"scenario 1: expected verdict AVX_DISPATCHED, got {res['verdict']!r}")
if rc != 0:
    failures.append(f"scenario 1: expected exit code 0, got {rc}")
if res["l3"]["hits_by_family"].get("avx") != 7:
    failures.append(f"scenario 1: expected 7 avx hits, got {res['l3']['hits_by_family']}")
if res["l3"]["hits_by_family"].get("scalar", 0) != 0:
    failures.append(f"scenario 1: expected 0 scalar hits, got {res['l3']['hits_by_family']}")
if res["l3"]["total_hits"] != 7:
    failures.append(f"scenario 1: expected total_hits 7, got {res['l3']['total_hits']}")

# --- Scenario 2: advertised (avx, fixed) != actual (scalar, n=6) -> SILENT_FALLBACK, exit 1 ---
rc2, res2 = run_check(6, "scalar")
print(f"[info] scenario 2 (silent fallback): verdict={res2['verdict']} exit_code={rc2} "
      f"advertised={res2['advertised']} l3.total_hits={res2['l3']['total_hits']} "
      f"l3.hits_by_family={res2['l3']['hits_by_family']}")
if res2["advertised"] != "avx":
    failures.append(f"scenario 2: expected advertised='avx' (main.c always claims it), got {res2['advertised']!r}")
if res2["verdict"] != "SILENT_FALLBACK":
    failures.append(f"scenario 2: expected verdict SILENT_FALLBACK, got {res2['verdict']!r}")
if rc2 != 1:
    failures.append(f"scenario 2: expected exit code 1 (mismatch), got {rc2}")
if res2["l3"]["hits_by_family"].get("scalar") != 6:
    failures.append(f"scenario 2: expected 6 scalar hits, got {res2['l3']['hits_by_family']}")
if res2["l3"]["hits_by_family"].get("avx", 0) != 0:
    failures.append(f"scenario 2: expected 0 avx hits (the advertised family never fired), got {res2['l3']['hits_by_family']}")
if res2["l3"]["total_hits"] != 6:
    failures.append(f"scenario 2: expected total_hits 6, got {res2['l3']['total_hits']}")

# --- Sanity: every shipped preset under tools/targets/*.json is valid, loadable JSON ---
targets_dir = os.path.join(repo_root, "tools", "targets")
preset_names = sorted(
    os.path.splitext(f)[0] for f in os.listdir(targets_dir) if f.endswith(".json")
)
print(f"[info] shipped presets: {preset_names}")
list_proc = subprocess.run([sys.executable, polygraph, "list"], capture_output=True, text=True, timeout=30)
if list_proc.returncode != 0:
    failures.append(f"`polygraph list` exited {list_proc.returncode}: {list_proc.stderr[:500]}")
for name in preset_names:
    explain_proc = subprocess.run([sys.executable, polygraph, "explain", name],
                                   capture_output=True, text=True, timeout=30)
    if explain_proc.returncode != 0:
        failures.append(f"`polygraph explain {name}` exited {explain_proc.returncode}: {explain_proc.stderr[:500]}")

if failures:
    for f in failures:
        print("FAIL: " + f)
    sys.exit(1)

print("PASS: tools/polygraph's declarative target pipeline (L1/L2/L3/verdict/exit-code) "
      "recovered exact ground truth for both a MATCH and a SILENT_FALLBACK scenario, "
      "and all 6 shipped presets under tools/targets/*.json are valid and `explain`-able.")
PY
