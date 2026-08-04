# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
#
# dispatch_probe.gdb -- GDB layer-3 (DISPATCH) probe for verify_dispatch.py.
#
# STATUS: exercised on real aarch64 Linux (Ubuntu 24.04 container, native arm64) against
# a synthetic dlopen'd shared library that reproduces ggml's backend-loading pattern,
# with a known ground truth of 5 calls per symbol. See "Why v1 returned zero hits"
# below -- the previous revision of this file silently reported 0 hits for every
# config on the free CI runner, which is exactly the class of failure this whole
# project exists to catch. It is fitting that our own tool had it.
#
# verify_dispatch.py renders this template by substituting:
#   __DP_SYMBOLS__ -- a Python list literal of concrete symbol names to break on,
#                     enumerated from the CPU backend library with `nm` (same
#                     enumeration L1 uses), filtered by the dispatch regex.
# and drives it with:
#   gdb -q -batch -x <rendered-script> --args <binary> <arg1> ...
#
# ---------------------------------------------------------------------------
# Why v1 returned zero hits on every config
# ---------------------------------------------------------------------------
# v1 ran `rbreak ^kai_run_matmul` BEFORE `run`. That cannot work here:
#
#   1. ggml loads its CPU backend (libggml-cpu.so) via dlopen AFTER process start, so
#      when `rbreak` executes, no `kai_*` symbol exists in the symbol table yet.
#   2. `rbreak` enumerates currently-known symbols and creates one real breakpoint per
#      match. Unlike `break`, it does NOT create pending breakpoints -- so zero matches
#      means zero breakpoints, silently, with no error and exit status 0.
#
# lldb hides this difference (its regex breakpoints auto-resolve when a solib loads
# later), which is why the macOS lane worked while the Linux lane reported
# NO_DISPATCH_OBSERVED for every row -- observed on free-runner CI run 30861845179.
#
# The fix, validated against ground truth (a program calling each symbol exactly 5
# times reported exactly 5 and 5):
#   * `set breakpoint pending on` + an explicit `break <symbol>` per name. `break` DOES
#     create pending breakpoints, which resolve when the solib is dlopen'd.
#   * Count with a gdb.Breakpoint subclass whose stop() returns False -- the documented
#     way to tally hits WITHOUT halting the inferior. v1's approach (gdb.events.stop +
#     gdb.post_event(continue)) halts on every hit, which is ruinously slow across tens
#     of thousands of matmul calls, and drops hits: on the same ground-truth program it
#     reported 1 instead of 5.
#
# NOTE ON COUNTS: because stop() returns False the inferior never halts, so these are
# true call counts, directly comparable to the lldb lane. They are not "stop events".

set confirm off
set pagination off
set breakpoint pending on
# llama.cpp does not fork for inference, but be explicit rather than rely on the default.
set follow-fork-mode parent

python
import gdb

_dp_hits = {}
_dp_failed = []


class _DispatchCounter(gdb.Breakpoint):
    """Count calls to one kernel entry point without stopping the inferior.

    Returning False from stop() is the documented GDB Python contract for
    "record this hit and keep going" -- no halt, no resume round-trip, no
    dropped events.
    """

    def __init__(self, symbol):
        super(_DispatchCounter, self).__init__(symbol, gdb.BP_BREAKPOINT, internal=False)
        self.silent = True
        self._symbol = symbol
        _dp_hits[symbol] = 0

    def stop(self):
        _dp_hits[self._symbol] += 1
        return False


for _dp_sym in __DP_SYMBOLS__:
    try:
        _DispatchCounter(_dp_sym)
    except Exception as _dp_exc:
        _dp_failed.append((_dp_sym, str(_dp_exc)))

# Emitted so the driver can distinguish "no kernel ran" (a real, interesting result)
# from "we failed to instrument anything" (a broken probe). v1 could not tell these
# apart, and that is precisely how it reported a silent zero.
print("DP_BREAKPOINTS_REQUESTED %d" % len(__DP_SYMBOLS__))
print("DP_BREAKPOINTS_CREATED %d" % len(_dp_hits))
for _dp_sym, _dp_err in _dp_failed:
    print("DP_BPFAIL %s %s" % (_dp_sym, _dp_err))
end

run

python
print("DISPATCH_PROBE_RESULT_BEGIN")
for _dp_name in sorted(_dp_hits):
    if _dp_hits[_dp_name]:
        print("DP_HIT %s %d" % (_dp_name, _dp_hits[_dp_name]))
print("DISPATCH_PROBE_RESULT_END")
end

quit
