# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
#
# dispatch_probe.gdb -- GDB layer-3 (DISPATCH) probe template for verify_dispatch.py.
#
# *** STATUS: written to the documented GDB Python API, NOT yet exercised on
# *** real hardware -- there is no gdb on the macOS development machine this
# *** repo was built on (`gdb` is absent from Xcode's toolchain; lldb is used
# *** instead, see dispatch_probe.lldb). This script is the Linux-side
# *** counterpart, intended to run on the DGX Spark / ubuntu-24.04-arm CI
# *** lanes where gdb is available. Treat any Linux dispatch-ledger row
# *** produced via this script as "verified via this template" only once it
# *** has actually been run in that CI job -- do not assume it is bug-free
# *** from inspection alone.
#
# verify_dispatch.py renders this template exactly like dispatch_probe.lldb:
# it substitutes __DISPATCH_REGEX__ with a concrete extended-regex pattern
# (default `^kai_run_matmul`), writes the result to a temp .gdb file, and
# drives it with:
#
#   gdb -q -batch -x <rendered-script> --args <binary> <arg1> <arg2> ...
#
# `--args` binds the target binary and its arguments once, so this template
# only needs a bare `run` to reuse them -- no per-run substitution beyond the
# regex is required, same reusability property as the lldb template.
#
# Why this needs a Python block instead of plain `rbreak` + `commands`:
# `rbreak <regex>` creates one real breakpoint per matching symbol, and
# GDB's `commands` (without an explicit breakpoint range) only attaches to
# the LAST breakpoint `rbreak` created -- not all of them. Enumerating the
# exact breakpoint number range up front is fragile (it depends on how many
# symbols in the binary match the regex, which varies by platform/build).
# Instead this script uses `gdb.events.stop`, which fires for every stop
# regardless of which breakpoint caused it, counts a hit keyed by the
# current frame's function name, and reschedules `continue` via
# `gdb.post_event` (the documented safe way to act from inside a stop-event
# callback without reentering the event loop directly).
python
import gdb

_dp_hits = {}

def _dp_stop_handler(event):
    try:
        frame = gdb.selected_frame()
        name = frame.name() or "<unknown>"
    except Exception:
        name = "<unknown>"
    _dp_hits[name] = _dp_hits.get(name, 0) + 1
    gdb.post_event(lambda: gdb.execute("continue", to_string=True))

gdb.events.stop.connect(_dp_stop_handler)
end

rbreak __DISPATCH_REGEX__
run

python
print("DISPATCH_PROBE_RESULT_BEGIN")
for _dp_name, _dp_count in sorted(_dp_hits.items()):
    print("DP_HIT %s %d" % (_dp_name, _dp_count))
print("DISPATCH_PROBE_RESULT_END")
end

quit
