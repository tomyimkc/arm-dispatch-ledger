# Auto-defaults patch measurement — `0002-kleidiai-sme-aware-thread-default.patch`

Fixes Defect A: the one thing this repo *built* (`0001-kleidiai-phase-aware-dispatch.patch`) was
a measured regression, and the one positive number (3.43x decode) came entirely from flags
`llama.cpp` already ships (`-t 2`), not from anything we wrote. This patch closes that gap: it
makes the `-t 2`-equivalent decode win happen **automatically, with zero flags**, without the
prefill regression that naively recommending `-t 2` would cause.

**Verdict: the patch works, is honest, and beats the naive alternative. Read the whole page
before citing a number — the methodology had to change mid-flight (see §2) and that matters for
how to read §4.**

## 0. Setup

- Model: `Qwen2.5-0.5B-Instruct-Q4_0` (`/tmp/ggufs/q05.gguf`), Apache-2.0.
- Baseline binary: `/tmp/llama.cpp/build/bin/llama-cli` @ `dbadb68`, `-DGGML_CPU_KLEIDIAI=ON
  -DGGML_METAL=OFF`, Release. **Unmodified — never touched.**
- Autodefault binary: `/tmp/llama-autodefaults/build/bin/llama-cli`, same base commit
  (`dbadb68`) + `patches/0002-kleidiai-sme-aware-thread-default.patch` applied, same CMake flags.
  Patch verified to `git apply --check` cleanly against a **fresh, separately-cloned** copy of
  `dbadb68` (not just the dev tree it was written in) before any number below was collected.
- Host: Apple M4 Max, macOS, `hw.perflevel0.physicalcpu=12` (P-cores, llama.cpp's no-flag
  default), `hw.perflevel1.physicalcpu=4` (E-cores), `sme_thread_cap=2` (KleidiAI
  runtime-detected).
- `-no-cnv -st --simple-io` used throughout (required for non-TTY stdin per this repo's runbook).

## 1. What the patch is supposed to do (recap)

With no `-t`/`--threads` passed, and KleidiAI reporting a positive SME2 thread cap lower than
the normal default: default **generation** threads to the cap (2 here) instead of the
physical-core count (12 here). `--threads-batch`/prefill is left at the normal default in every
case. `GGML_KLEIDIAI_AUTO_THREADS=0` disables it entirely.

## 2. A methodology correction, made *before* collecting numbers, not after

The original plan was to run all four configs through `llama-bench -r 1 -o json`. Before trusting
that, we checked whether `llama-bench`'s own `-t` default actually exercises this patch's code —
it does not, and this was verified directly:

```
$ ./build/bin/llama-bench -m q05.gguf -p 16 -n 16 -r 1 -o json   # autodefault binary, no flags
[{'n_threads': 12, 'n_prompt': 16, 'n_gen': 0}, {'n_threads': 12, 'n_prompt': 0, 'n_gen': 16}]
$ ./build/bin/llama-bench -m q05.gguf -p 16 -n 16 -r 1 -o json   # baseline binary, no flags
[{'n_threads': 12, 'n_prompt': 16, 'n_gen': 0}, {'n_threads': 12, 'n_prompt': 0, 'n_gen': 16}]
```

Identical. `llama-bench` never calls `postprocess_cpu_params()` or
`common_context_params_to_llama()` — it builds its own `llama_context_params` and sources its
default straight from `common_cpu_get_num_math()`, bypassing every function this patch touches.
This is architectural, not a bug in our patch, and touching `common_cpu_get_num_math()` itself
to "fix" it would have broken `llama-bench`'s own prefill-only test rows (it has no `-tb`
equivalent — one `n_threads` value covers both `-p` and `-n` tests in a row).

**Consequence:** the "no flags" configs (1, 2, 4) below are measured with **`llama-cli`**, the
tool that actually goes through the patched path, not `llama-bench`. Config 3 (`-t 2`, explicit)
works identically in both tools since it doesn't depend on the default-resolution path at all;
`llama-cli` was used for all four configs so every number in one table comes from the same tool
and methodology. `llama-bench` was used only for the correctness check above and for two
supporting single-shot cross-checks in §5.

## 3. Correctness — the patch fires exactly where it should, and nowhere else

`llama-cli --verbose`, `system_info` line + the patch's own `COM_INF` log line:

| invocation | `n_threads` (generation) | `n_threads_batch` (prefill) | override fired? |
|---|---:|---:|---|
| autodefault, no flags | **2** | 12 | yes — logged |
| autodefault, `GGML_KLEIDIAI_AUTO_THREADS=0`, no flags | 12 | 12 | no (must equal baseline — it does) |
| autodefault, explicit `-t 4` | 4 | 4 | no (user's choice preserved) |
| autodefault, explicit `-tb 6`, no `-t` | 2 | 6 | yes — proves gen/batch independence |
| baseline (unpatched), no flags | 12 | 12 | n/a (no such mechanism exists) |
| autodefault built with `-DGGML_CPU_KLEIDIAI=OFF`, no flags | 12 | 12 | no (stub path, no crash, no KLEIDIAI feature flag) |

Log line observed verbatim:
```
common_kleidiai_sme_auto_gen_threads: KleidiAI SME2 detected (thread cap = 2); defaulting
generation threads to 2 instead of 12. --threads-batch / prefill is unaffected. Set
GGML_KLEIDIAI_AUTO_THREADS=0 to disable this.
```

Text generation was checked for correctness at every configuration above (e.g. "The capital of
France is Paris.") — no crash, no assertion failure, no garbled output in any run.

## 4. Symbol-level dispatch proof

`tools/verify_dispatch.py --binary <autodefault llama-cli> --model q05.gguf --threads 2
--workloads decode_short --l3-debugger lldb` (i.e. at the exact thread count this patch
auto-selects with zero flags):

```
L2: primary_kernel_feature={'q4': 'SME2', 'q8': 'SME2', 'f32': 'SME2'} sme_enabled=True
L3: kernel_family_executed=sme2 total_hits=5826 (25.2s)
VERDICT: SME2_DISPATCHED
```

5826/0 SME2-vs-other hits: the accelerated kernel doesn't just get selected in the log, it
actually runs, at the thread count the "no flags" default now auto-selects.

## 5. Throughput — round-robin interleaved, n=9, `llama-cli`, decode and prefill separately

Method: one shared prompt file (~500 tokens, repeated sentence text), `-n 128` for generation.
Each `llama-cli` run prints `[ Prompt: X t/s | Generation: Y t/s ]`, giving prefill (X) and
decode (Y) from a single invocation. Configs run in strict round-robin order
(1, 2, 3, 4, 1, 2, 3, 4, ...) for 9 full cycles — never all reps of one config before the next.

External CPU load (`ps -Ao %cpu,comm -r`, top entries) — **before**: `cmux 92%, fseventsd 69%,
logd 64%, Virtualization.framework 37%, claude 30%` (a busy, multi-agent-shared machine, load
average ~5.5-6 on this run — consistent with this repo's established finding that absolute
numbers on this host are contention-suppressed and ratios/interleaving are what's trustworthy,
see `results/REMEASURE-2026-08-04-QUIET.md`). **After**: `cmux 91%, logd 47%,
Virtualization.framework 44%, mds 37%, claude 15%` — comparable load throughout, shared equally
across all 4 configs by the round-robin design.

| config | phase | median (t/s) | stdev | min | max | n |
|---|---|---:|---:|---:|---:|---:|
| 1. baseline, no flags | decode | 67.8 | 5.8 | 52.8 | 71.8 | 9 |
| 2. autodefault, no flags | decode | **145.9** | 2.2 | 144.6 | 150.1 | 9 |
| 3. baseline, `-t 2` (hand-tuned ceiling) | decode | 146.0 | 1.2 | 144.7 | 148.3 | 9 |
| 4. autodefault, `AUTO_THREADS=0`, no flags | decode | 68.3 | 8.5 | 44.6 | 71.8 | 9 |
| 1. baseline, no flags | prefill | 1835.2 | 75.5 | 1711.2 | 1958.6 | 9 |
| 2. autodefault, no flags | prefill | 1779.8 | 71.7 | 1646.8 | 1873.0 | 9 |
| 3. baseline, `-t 2` (hand-tuned ceiling) | prefill | 975.6 | 4.1 | 970.0 | 982.6 | 9 |
| 4. autodefault, `AUTO_THREADS=0`, no flags | prefill | 1835.3 | 219.9 | 1287.2 | 2037.5 | 9 |

Raw per-rep numbers are in the session log this table was built from; medians/stdev/min/max above
are computed directly from the 9 interleaved reps per config, not cherry-picked.

### Reading it

| comparison | ratio | reading |
|---|---:|---|
| decode: config 2 / config 1 | **2.15x** | no-flags autodefault reaches ~2.15x the no-flags baseline decode throughput |
| decode: config 2 / config 3 | 0.999x | autodefault (no flags) is statistically indistinguishable from the hand-tuned `-t 2` ceiling — it reaches the ceiling automatically |
| decode: config 4 / config 1 | 1.007x | kill switch reproduces the unpatched baseline within noise (stdev bands overlap: 62.0-73.6 vs 59.8-76.8) |
| prefill: config 2 / config 1 | 0.970x | autodefault's prefill is **unchanged within noise** (-3.0%, well inside the ±75-token-per-second band both configs show) |
| prefill: config 3 / config 1 | **0.532x** | the "obvious" hand-tuned fix (`-t 2`, no `-tb`) **collapses prefill by 47%** as a side effect — this is exactly why "just tell people to pass `-t 2`" is not an acceptable substitute for this patch |
| prefill: config 4 / config 1 | 1.000x | kill switch reproduces baseline prefill essentially exactly |

**This is the headline finding: the autodefault patch gets the full decode win (2.15x, matching
the hand-tuned ceiling to within measurement noise) while leaving prefill untouched — something
the "just document `-t 2`" alternative structurally cannot do**, because stock llama.cpp's
`-tb` default is "same as `-t`," so any single-flag fix to decode threads also caps prefill
threads unless the fix is phase-aware, which is what this patch is.

Absolute numbers here differ from `results/REMEASURE-2026-08-04-QUIET.md`'s quieter-machine
figures (e.g. 93.6/321.0 tok/s decode there vs 67.8/146.0 here) because this measurement ran on a
busier machine (see load snapshot above) — consistent with this repo's own prior finding that
absolute throughput on this host is contention-sensitive while the *ratio* between interleaved
configs is not. The ratio pattern (large decode win from fewer threads, ~2-3x depending on
contention) replicates across both measurement sessions.

## 6. Honest limits

- **`llama-bench`'s own "no flags" default is not, and cannot cheaply be made, aware of this
  patch** (§2). Anyone re-measuring this should use `llama-cli`/`llama-server` for the "no
  flags" comparison, not `llama-bench`, or they will see no effect and wrongly conclude the patch
  does nothing.
- **Speedup magnitude is hardware-specific.** The mechanism (read `sme_thread_cap` from
  KleidiAI's own runtime detection, not a hardcoded constant) generalizes to any SME2 CPU
  KleidiAI supports, but the measured 2.15x/-3% numbers above are this machine, this model, this
  quant only.
- **Measured on a busy, shared, multi-agent-session machine** (see load snapshot). The
  round-robin design and the internal consistency of the four configs (config 2 ≈ config 3,
  config 4 ≈ config 1) are the trustworthy part; treat the absolute tok/s values as this
  session's numbers, not a hardware ceiling.
- **`0001` remains a measured regression** and is unrelated to this patch's mechanism — this
  patch does not depend on, require, or interact with `0001`; it was tested standalone against
  unmodified `dbadb68`.
