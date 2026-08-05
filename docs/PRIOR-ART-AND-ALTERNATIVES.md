# Prior art and alternatives: what else answers "did the fast path run?"

This page exists for the same reason [`RELATED-WORK.md`](RELATED-WORK.md) does. That page records
that another project published this project's Finding 2 mechanism two days earlier. This one
records something less comfortable: **the L3 technique at the centre of this repo is not novel.**
Every element of it is documented standard practice, and for some questions there are simpler,
cheaper, lower-overhead tools that answer the same thing.

A reader deciding whether to use this project deserves to know that before they read the results,
not after. So: here is everything that overlaps, what each alternative does better, and the
narrow set of things this project actually adds.

## The technique, stated plainly

L3 sets breakpoints on kernel entry symbols and counts how many times they are hit, without
stopping the program. Concretely:

- **Pending breakpoints** so a symbol inside a library that is `dlopen`'d after process start
  still resolves. GDB's manual documents this as a built-in feature; the only real trap is that
  `rbreak` does not create pending breakpoints while `break` does, which is why
  [`tools/verify_dispatch.py`](../tools/verify_dispatch.py) enumerates symbol names from `nm`
  first and sets them explicitly.
- **A non-halting counter** — a `gdb.Breakpoint` subclass whose `stop()` returns `False`, so the
  hit is recorded and the inferior continues. LLDB's own tutorial demonstrates this exact pattern.

Both are in the vendors' own documentation. Neither was invented here.

## Alternatives that answer the same or an adjacent question

| tool | what it measures | where it beats L3 |
|---|---|---|
| **`bpftrace` uprobes** — `uprobe:/path/lib.so:kai_run_matmul_* { @[probe] = count(); }` | function entry counts | Far lower overhead, and uprobes attach by inode+offset so the `dlopen` ordering problem this repo worked around **does not arise at all**. Needs root and a kernel with BPF enabled. |
| **Arm PMU instruction counters** (`SVE_INST_SPEC`, `ASE_INST_SPEC`, via `perf stat`) | how many SVE / Advanced-SIMD instructions the silicon actually issued | Answers "did the vector hardware do work?" at the hardware level, with **no code and near-zero overhead** — *where it is available*. Cannot attribute to a named function. On the one Arm server we tested it was not available at all; see below. |
| **Intel `processwatch`** | per-process ISA-extension instruction mix on x86 | A shipped, maintained tool for the x86 form of exactly this question. Good evidence the *category* is established. |
| **`perf probe` + `perf stat`** | function entry counts | No scripting, no debugger, standard toolchain. |
| **Intel SDE `-mix`** | full instruction histogram | Strictly more information about which instructions executed. Emulation-speed, x86 only. |
| **`ltrace` / `uftrace` / SystemTap / DTrace** | call tracing | Mature, general, well documented. |
| **Intel Pin, DynamoRIO, Valgrind/Callgrind** | arbitrary dynamic binary instrumentation | Vastly more powerful and more general. Much heavier. |
| **`ONEDNN_VERBOSE`, `MKL_VERBOSE`** | the library's own kernel-selection log | Free and authoritative *when the library implements it*. This is L2, and where it exists it is better than L3. |

## What this project actually adds

Given the table above, the honest list is short. It is not the mechanism.

1. **A specific, previously undocumented defect, found and reported.** llama.cpp's own documented
   KleidiAI build line compiles zero `kai_run_matmul` kernels on gcc 13.3 + Cortex-X925 while the
   banner still prints `KLEIDIAI = 1`, costing 4.57x on 7B prefill. Filed upstream. The tool was
   the means; the finding is the contribution.
2. **Three independent levels that can disagree, and a record of them disagreeing.** L1 (symbols),
   L2 (logs), L3 (execution) are separately reported rather than collapsed into one verdict. The
   entire Finding 3 result *is* a disagreement between L1 and the banner. A single-number tool
   cannot express that.
3. **It works where the privileged tools do not — measured, not assumed.** We went looking for
   the PMU alternative on real Arm server silicon and could not use it, for two *independent*
   reasons ([`results/pmu/pmu-crosscheck.json`](../results/pmu/pmu-crosscheck.json)):

   - **Permission.** `perf_event_paranoid = 4` blocks `perf_event_open()` without `CAP_PERFMON`.
     Notably it blocked even `perf stat -e task-clock`, a pure *software* event that the
     documented paranoid ladder says should be exempt at that level.
   - **The events do not exist.** Independent of permission, the kernel's own event registry
     (`/sys/bus/event_source/devices/armv8_pmuv3_0/events/`) enumerates **78 events, none of them
     SVE-, SME- or ASE-class**. There is one undifferentiated `inst_spec` counter and no way to
     ask "how many SVE instructions". `SVE_INST_SPEC` is simply not registered by this driver.

   This is a Cortex-X925 + Cortex-A725 Armv9.2 machine on **bare metal** (`systemd-detect-virt`
   reports none) — not a container, not a VM, which is the caveat usually attached to PMU
   availability. A hardened default plus a driver that never enumerated the finer-grained event
   codes was enough. A debugger attaching to a process you already own needed no privilege and
   worked.

   Two honest caveats. First, this is **one machine**; it is not evidence that PMU counters are
   generally unavailable on Arm. Second, we could not test whether a root session could reach the
   events via *raw* event codes, bypassing the symbolic registry — `perf_event_paranoid = 4`
   blocked every event type at our privilege level, so there was nothing further to learn. The
   PMU-vs-L3 numeric cross-validation we set out to run therefore **remains unperformed**, and is
   recorded as unanswered rather than answered.
4. **Ground-truth tests for the probe itself.** `tests/l3_gdb_groundtruth/` and
   `tests/l3_lldb_groundtruth/` drive the real shipped probe against programs with known call
   counts. This exists because an earlier version of the probe **silently reported zero hits** —
   it looked like a clean negative result and was a bug. A verification tool that is not itself
   verified is worse than no tool, because it produces confident wrong answers.
5. **A claims registry that fails CI on drift.** `tools/check_claims.py` refuses to let a number
   appear in prose unless it is registered and JSON-backed, and refuses to let a retracted figure
   appear unmarked. It has caught 12 real drift errors.

Items 4 and 5 are the parts a competitor would find least convenient to copy, because they are
discipline rather than code.

## What L3 costs

Measured on the same box, round-robin interleaved A,B,A,B,A,B,A,B,A,B, 5 reps per arm, at
threads=4 on a short decode workload:

| | median wall clock | spread | stdev |
|---|---|---|---|
| plain run | 1.2056 s | 0.21 s | 0.088 s |
| under the L3 gdb probe | 4.379 s | 2.68 s | 1.04 s |

**Median overhead 3.63x**, and the debugger's own attach and scheduling cost dominates the
run-to-run spread. The *dispatch count itself* was exactly reproducible — 15,936 hits on every one
of the five probed runs — so the measurement is deterministic even though its wall clock is not.
That is the right trade for a verification tool and the wrong one for a profiler. **Do not use
this to measure performance; use it to find out what ran.** This figure covers one configuration
(15,936 hits); it has not been shown to generalise.

## The honest summary

If you want to know whether vector hardware did work on a machine you control, try `perf stat`
with the Arm PMU events first — it is free, it has no overhead, and it needs no code. Check that
the events are actually enumerated on your machine before relying on it; on ours they were not.

If you want to know whether a **specific named kernel** was invoked, in an environment where PMU
events and BPF are not available to you, and you want the answer cross-checked against static
symbols and the runtime's own logs, that is what this project does.

That is a narrower claim than "a new way to verify hardware acceleration." It is the one the
evidence supports.
