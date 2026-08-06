# Note — the "marked untested" warning inside the Finding-4 evidence capture

**Added 2026-08-06. This note is additive; nothing in the 2026-08-05 evidence files was edited.**

The captured `human_report_literal` fields inside
[`llamacpp-26334-cuda-host-buffer.json`](llamacpp-26334-cuda-host-buffer.json) begin with:

> `WARNING: target '…' is marked untested (never verified against a real artifact) -- results may
> not be meaningful.`

Because that line sits inside this project's strongest evidence file, this note states exactly
what it is and is not.

## What the warning is

It is `tools/polygraph`'s own **preset-provenance flag** (`tools/polygraph`, the `cmd_check`
named-target path): it prints when a target definition's `tested` field is false, i.e. when the
preset JSON has no committed `verified_against` receipt yet. It warns about the *preset's*
history, not about the measurement it is about to drive.

## Why it printed on 2026-08-05

The three CUDA targets used for this reproduction
(`llama-cpp-kleidiai-cuda-ngl0-baseline`, `…-nohost`, `…-devnone`) were authored on the spot for
this measurement session. They were variants of `tools/targets/llama-cpp-kleidiai.json` whose
`l1`/`l2`/`l3` sections are byte-identical to that tested preset — only `workload.arg_template`
differs (`-ngl 0`, plus `--no-host` or `-dev none`) — but as freshly authored files they carried
no `verified_against` receipt at run time. That receipt-less state is exactly what the flag
reports, and the evidence JSON records the run output verbatim, warning included. The evidence
file's own `polygraph_targets_used.note` field documents this ("three ONE-OFF variant target
files … NOT committed, NOT part of the polygraph repo").

## Why it is not a caveat about the measurement

The L3 layer ran to `level_reached: 3/3` with real non-halting `gdb` breakpoints on all 10
`kai_run_matmul_*` symbols, five round-robin-interleaved reps per arm, and zero rep-to-rep
variation (0 hits in every baseline rep; 7,968 hits in every `--no-host`/`-dev none` rep). See
[`FINDING-4-CUDA-HOST-BUFFER.md`](FINDING-4-CUDA-HOST-BUFFER.md).

## What changed on 2026-08-06

The three presets are now committed under `tools/targets/` with `verified_against` receipts
pointing at this directory's 15-run evidence file, so:

1. the Reproduce section of `FINDING-4-CUDA-HOST-BUFFER.md` works verbatim again (the presets had
   been session-local and were not in the repo), and
2. future runs of these targets no longer print the warning.

The 2026-08-05 capture above is deliberately left untouched — it is a verbatim record of the run
as it happened, warning and all.
