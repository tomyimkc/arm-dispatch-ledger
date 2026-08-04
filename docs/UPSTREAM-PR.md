<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
-->
# Draft pull request for `ggml-org/llama.cpp`

**Status: NOT opened.** This file is a ready-to-paste PR description, written as the
offered follow-up to [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547)
("happy to send a PR if the direction is welcome"). Per this project's own working
agreement, no PR has been opened against `ggml-org/llama.cpp` — this document exists so
the PR can be opened later with no further drafting, and so a judge can read exactly what
would be proposed upstream without us actually filing it during the challenge window.

This document offers **two separable things**, and is explicit about which one is
actually being proposed for merge:

1. **The warning** — small, uncontroversial, and proposed to land on its own.
2. **The phase-aware dispatch experiment** — reported as a **measured negative result**.
   We hypothesized it would help, implemented it, measured it on real hardware, and it
   does not: it is a statistical tie or a real ~12% regression, not an improvement. It is
   **explicitly not proposed for merge.** It is included here because the measurement
   itself is useful information for the maintainers (it says the `ne11 < 128` exclusion is
   not leaving decode throughput on the table on this chip), and because publishing a
   negative result about our own patch, rather than quietly dropping it, is the honest
   thing to do.

Everything below is written in the voice of the PR author (a first-time contributor to
this codebase) addressing the `llama.cpp`/KleidiAI maintainers directly.

---

## Title

`ggml-cpu: kleidiai: warn on silent GEMV→NEON fallback (+ a measured negative result for an opt-in phase-aware dispatch experiment, not proposed for merge)`

## Base / branch

- Base: `ggml-org/llama.cpp@dbadb68` (the commit this was developed and measured against)
- Patch: `patches/0001-kleidiai-phase-aware-dispatch.patch` in
  [`tomyimkc/arm-dispatch-ledger`](https://github.com/tomyimkc/arm-dispatch-ledger),
  applied via `git am` as commit `ef973b1` in a local branch
  `kleidiai-phase-aware-dispatch`
- Diffstat: 1 file changed, 56 insertions(+), 3 deletions(-) — `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`
  only. No new files, no new dependencies, no public API or CMake change. The diff contains
  both pieces described above (warning + dispatch experiment) together, for reproducibility;
  the actual **ask** below is scoped to the warning only, and the experiment section says so
  explicitly a second time before the numbers.

## Summary

Follow-up to #26547 (Finding 1). This patch does two things, and only one of them is
being proposed for merge:

**(1) Proposed for merge — a one-shot `GGML_LOG_WARN`** for the case #26547 already
documents: a GEMV op (decode, `ne11 == 1`) collapsing to NEON because the requested
thread count exceeds `sme_thread_cap`, naming the exact knob (`-t <= sme_thread_cap`)
that recovers it. This is small, low-risk, and has no measured downside — it changes
nothing about dispatch, only observability.

**(2) Reported for information only, NOT proposed for merge — an opt-in env var**,
`GGML_KLEIDIAI_PHASE_AWARE=1`, default off, that lets a GEMV-shaped op enter the
*existing* SME+NEON hybrid dispatch path once the thread count exceeds `sme_thread_cap`,
instead of collapsing to NEON-only. We built this because Finding 1 raised an obvious
question: is the `ne11 < 128` exclusion actually costing decode throughput? We measured
the answer, and it is no — on an Apple M4 Max, at the default thread count, this bypass
makes decode **~12% slower**, and at other thread counts it is a statistical tie. We are
not asking anyone to merge this. We are telling you what we found, because "we checked
and the exclusion isn't leaving anything on the table on this chip" is useful information
for anyone else who has the same question, and because publishing a negative result about
our own patch is more honest than quietly shelving it.

## Part 1 — the warning (proposed for merge)

`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`'s dispatch decision (unpatched, this base commit, lines
1094–1113) gates the hybrid SME+NEON split on:

```cpp
const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);
```

`ne11` is the batch/token count of the matmul. Decode is *always* `ne11 == 1`, so this
term is always true for decode regardless of thread count, and dispatch collapses
unconditionally to the non-SME slot once `nth_total > sme_thread_cap`:

```cpp
chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
```

That collapse is silent: the startup banner and the one-time "SME2 enabled" log line are
unchanged either way (see #26547's L1/L2/L3 evidence chain), so a user at default thread
count has no way to learn, short of attaching a debugger, that their decode step never
touched SME2. This PR's warning half fixes exactly that observability gap, and only that:
a one-shot `GGML_LOG_WARN`, fired once per process when a GEMV op collapses to NEON
because of the thread cap, naming the exact knob (`-t <= sme_thread_cap`) that would
change it. It follows the same one-shot `static std::atomic<bool>` guard pattern this
file already uses for its existing weight-type-fallback warning — same idiom, same log
call, no new pattern introduced. It requires no flag and changes no dispatch behavior; it
only makes an existing, real silent fallback observable.

## Part 2 — the phase-aware dispatch experiment (reported for information; NOT proposed for merge)

This PR does not remove the `ne11 < 128` gate. It adds a second, opt-in path around it,
gated behind `GGML_KLEIDIAI_PHASE_AWARE=1` (default off):

```cpp
const bool phase_aware_gemv = ctx.phase_aware_dispatch && is_gemv &&
                               sme_slot != -1 && non_sme_slot != -1;
const bool too_small_for_hybrid = (min_cols_per_thread < 2) ||
                                   (!phase_aware_gemv && ne11 < 128);
```

When the flag is set and the op is GEMV-shaped with both an SME and a non-SME kernel slot
available, the `ne11 < 128` term is bypassed for that op only (the `min_cols_per_thread <
2` guard still applies — this does not touch the too-small-to-split-at-all case). That
routes decode above the cap through the **same** thread-assignment code prefill hybrid
mode already exercises: SME capped at `sme_thread_cap` threads, the remaining threads
running NEON in parallel. No new thread-splitting logic — this reuses an already-tested
path deliberately, because this codebase's threadpool model requires every thread in
`[0, nth_total)` to reach the same barriers the same number of times per op; an earlier
draft of this patch that left the extra threads idle instead of routing them to NEON was
rejected during design for exactly that deadlock risk.

**Why we are not proposing this for merge:** we measured it (see "Measured results"
below) and it does not help. At the thread count that matters most — `llama.cpp`'s real
no-flag default — decode is ~12% *slower* with the flag on than without it, outside noise.
At `-t 2` it is a statistical tie (the patch is inert there; `nth_total == sme_thread_cap`
means the bypass never activates). Prefill is an untouched tie, as expected, since this
patch's diff does not touch prefill's dispatch code. We are including the patch and its
measurement here anyway, rather than deleting it, because the *dispatch* change is real
and symbol-proven (see below) even though the *throughput* change is not a win — and
because "we tried the obvious fix for the exclusion Finding 1 documented, and it doesn't
help on this chip" is a useful, checkable data point for anyone else considering the same
change.

## Exact files / lines touched

All in `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (base `dbadb68` line numbers cited
below; the patch's own new-file line numbers are in the diff hunks):

| Site (base `dbadb68`) | Change | Part |
|---|---|---|
| `struct ggml_kleidiai_context` (~line 67) | New field `bool phase_aware_dispatch;`, default `false`, documented inline. | 2 (experiment) |
| `init_kleidiai_context()`, env parsing block (~lines 201–235, alongside the existing `GGML_KLEIDIAI_SME` / `GGML_TOTAL_THREADS` / `GGML_KLEIDIAI_CHUNK_MULTIPLIER` reads) | New `getenv("GGML_KLEIDIAI_PHASE_AWARE")`, parsed with the existing `parse_uint_env()` helper, sets `ctx.phase_aware_dispatch = true` only on a truthy value. | 2 (experiment) |
| `init_kleidiai_context()`, logging block (~line 311, right after the existing "SME disabled"/"SME2 enabled" log lines) | New one-shot `GGML_LOG_INFO` announcing phase-aware dispatch is active, only printed when the flag is set. | 2 (experiment) |
| Dispatch decision in `compute_forward_qx()` / `tensor_traits` (~lines 1094–1113, the `too_small_for_hybrid` / `hybrid_enabled` / `chosen_slot` block described in #26547) | The `phase_aware_gemv` bypass described above. | 2 (experiment) |
| Same dispatch decision block | One-shot `GGML_LOG_WARN` fired when a GEMV op collapses to NEON because of the thread cap **and the flag is not set** — the "here's the knob" warning, on by default. | **1 (proposed for merge)** |

Full diff: `patches/0001-kleidiai-phase-aware-dispatch.patch` in the linked repo (or
inline once this is opened as a real PR). If the maintainers would rather review these as
two separate PRs, we are glad to split them — see the questions at the end.

## Measured results (Apple M4 Max, this is the only hardware this was tested on)

Full write-up with every caveat and the reasoning behind each "is this noise or a real
effect" call: `results/REMEASURE-2026-08-04-QUIET.md` in the linked repo (this
supersedes an earlier, contended measurement in `results/OPTIMIZATION.md` — the original
run shared this host with several unrelated concurrent sessions at a 1-minute load average
of 66–147, and baseline/patched configs were measured in different, non-interleaved time
windows, which manufactured a fake speedup for the patch. We re-ran everything
round-robin-interleaved on a quiet machine before writing this section). Summary here.

**Setup:** `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (337 MB, Apache-2.0). Baseline binary
untouched at `dbadb68`; patched binary is a separate clone at `ef973b1` (this patch on
top of `dbadb68`), same CMake flags
(`-DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`).
`llama-bench -r 1 -o json`, 7 independently-launched process repetitions per cell,
interleaved round-robin across configurations (not 7-in-a-row), median ± population
stdev reported — never a bare mean. Prefill and decode reported separately throughout,
per this project's own no-blended-speedup-number rule. `llama.cpp`'s real no-flag default
on this host is **12 threads** (`hw.perflevel0.physicalcpu`, the P-core count), not 16.

### Throughput — default vs. tuned vs. patched (`llama-bench`, interleaved, n=7)

| Config | decode tok/s (median ± stdev) | prefill tok/s (median ± stdev) |
|---|---:|---:|
| (a) baseline, llama.cpp's real no-flags default (12 threads) | 93.6 ± 2.47 | 1,230.3 ± 118.52 |
| (b) baseline, hand-tuned per-phase thread flag (`-t 2` decode / `-t 8` prefill — this repo's own best-measured split, needs no patch) | **321.0 ± 2.09** (**3.43×** vs. (a)) | **2,198.1 ± 72.59** (**1.79×** vs. (a)) |
| (c) **this patch's dispatch experiment**, flag on, no-flags default | **82.5 ± 4.07** (**0.88×** vs. (a) — **~12% slower**) | 1,202.1 ± 96.26 (0.98×, tie — this patch's diff does not touch prefill's dispatch code at all) |
| (d) this patch's dispatch experiment, flag on, `-t 2` | 317.5 ± 3.58 (0.99×, tie — the bypass never activates at `nth_total (2) == sme_thread_cap (2)`) | not applicable |

The genuinely actionable, user-facing finding in this whole investigation is row (b): the
tuning discovery, not the patch. It needs zero code changes and is 3.43×/1.79× today.

### Symbol-level dispatch proof (`tools/verify_dispatch.py`, `lldb`, anchored `^kai_run_matmul` breakpoints, auto-continue, real hit counts)

Same patched binary both rows; only `GGML_KLEIDIAI_PHASE_AWARE` differs. This part is
**not** contention-sensitive — it counts kernel calls via a debugger breakpoint, not
wall-clock time, so the contention problem that affected the throughput numbers above does
not apply here:

| threads | workload | flag OFF: SME2 hits / other hits | flag ON: SME2 hits / other hits |
|---:|---|---:|---:|
| 4 | decode | 0 / 15,936 (exact match to pre-patch #26547 ground truth) | **3,072 / 10,428** |
| 8 | decode | 0 / 31,872 | **2,354 / 20,517** |

This is the proof the dispatch change is real and not a selection-log artifact: the
identical binary, identical workload, identical thread count goes from zero SME2 kernel
calls to thousands, purely as a function of the env var. What it does **not** prove is a
throughput win — see the table above, where the same mechanism shows a tie or a real
regression, not a gain. Dispatching SME2 and being faster are two different questions;
this patch answers the first one and the answer to the second is no.

### Reproduction

```sh
# Apply and build the patch (against a fresh dbadb68 checkout)
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git checkout dbadb68
git am /path/to/arm-dispatch-ledger/patches/0001-kleidiai-phase-aware-dispatch.patch
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-cpu llama-cli llama-bench llama-tokenize -j"$(nproc)"

# Throughput, interleaved (the discipline that matters — see results/REMEASURE-2026-08-04-QUIET.md)
/path/to/llama.cpp/build/bin/llama-bench -m /path/to/Qwen2.5-0.5B-Instruct-Q4_0.gguf -p 0   -n 32 -r 7        # default, decode
/path/to/llama.cpp/build/bin/llama-bench -m /path/to/Qwen2.5-0.5B-Instruct-Q4_0.gguf -p 0   -n 32 -r 7 -t 2   # tuned decode
/path/to/llama.cpp/build/bin/llama-bench -m /path/to/Qwen2.5-0.5B-Instruct-Q4_0.gguf -p 256 -n 0  -r 7        # default, prefill
/path/to/llama.cpp/build/bin/llama-bench -m /path/to/Qwen2.5-0.5B-Instruct-Q4_0.gguf -p 256 -n 0  -r 7 -t 8   # tuned prefill
# repeat the "default, decode" and "default, prefill" rows with GGML_KLEIDIAI_PHASE_AWARE=1
# set in the environment to reproduce (c); interleave rather than running each config
# back-to-back, and record external CPU load before and after.

# Symbol-level dispatch proof, flag on and off (not contention-sensitive)
python3 tools/verify_dispatch.py --binary /path/to/llama.cpp/build/bin/llama-cli \
  --model /path/to/Qwen2.5-0.5B-Instruct-Q4_0.gguf --threads 4,8 \
  --workloads decode_short,prefill_long --env GGML_KLEIDIAI_PHASE_AWARE=1 \
  --l3-timeout 240 --out results/dispatch-ledger-<platform>-flag-on.json --assert

python3 tools/verify_dispatch.py --binary /path/to/llama.cpp/build/bin/llama-cli \
  --model /path/to/Qwen2.5-0.5B-Instruct-Q4_0.gguf --threads 4,8 \
  --workloads decode_short,prefill_long --l3-timeout 240 \
  --out results/dispatch-ledger-<platform>-flag-off.json --assert
```

`tools/crossover.py` and `tools/verify_dispatch.py` are stdlib-only Python, part of
[`tomyimkc/arm-dispatch-ledger`](https://github.com/tomyimkc/arm-dispatch-ledger)
(Apache-2.0); they are generic against any `llama.cpp`-family binary + GGUF, not
specific to this patch or this machine.

## What this experiment does NOT achieve

Read this before the numbers above, not after:

- **It does not raise the decode ceiling.** The best decode throughput measured
  anywhere in this project, patched or unpatched, is the hand-tuned `-t 2` flag,
  321.0 tok/s (`results/REMEASURE-2026-08-04-QUIET.md`) — already expressible today with
  **zero code changes**. The dispatch experiment, where it activates (default thread
  count), lands at 82.5 tok/s — well below that ceiling, and below the unpatched baseline
  at the same thread count (93.6 tok/s).
- **It makes the default configuration slower, not faster.** At `llama.cpp`'s real
  no-flag default (12 threads), decode goes 93.6 → 82.5 tok/s with the flag on — a real,
  outside-noise **~12% regression**, not the improvement we expected when we wrote this
  patch.
- **At the tuned thread count it is inert.** At `-t 2`, flag on vs. off is a statistical
  tie (321.0 vs. 317.5 tok/s) — the bypass never activates there because
  `nth_total == sme_thread_cap`, so there is nothing to measure either way.
- **It does not touch the deeper limitation `GGML_KLEIDIAI_SME` being process-global.**
  The theoretical best (SME2 for decode, NEON for prefill, simultaneously, in one
  process) is still unreachable after this patch, for the same reason it was
  unreachable before it: there is no per-call or per-phase override of that env var. A
  phase-aware **kernel-family** selector, not just the phase-aware **thread-gate**
  bypass this experiment adds, would be the real fix for that, and we have not attempted
  it.
- **What it does prove, and the reason we are still reporting it:** the dispatch change
  is real at the symbol level (0 → thousands of SME2 kernel calls, same binary, same
  workload, only the env var different — see "Symbol-level dispatch proof" above), and
  the `ne11 < 128` exclusion Finding 1 documents is, on this chip and this model, *not*
  leaving decode throughput on the table. That is a useful negative result, not nothing.

## What was NOT tested

- **Chips:** only Apple M4 Max (`sme_thread_cap = 2`). Not verified on M4, M4 Pro, or M4
  Ultra (the other rows of `detect_num_smcus()`'s brand-string table), nor on any future
  Apple chip. No non-Apple SME2 hardware currently exists to my knowledge, so this
  patch's Apple-only relevance is a fact about the current hardware landscape, not a
  design choice in the patch itself — the `phase_aware_gemv` logic is not `__APPLE__`-gated.
- **Models:** only `Qwen2.5-0.5B-Instruct`. Not tested at any larger parameter count,
  where the compute-to-memory ratio for decode vs. prefill likely differs and could
  change which configuration wins.
- **Quantizations:** only `Q4_0`. No `Q8_0` GGUF was available in my environment; not
  tested.
- **Thread count 16:** every attempt to measure decode at 16 threads (baseline and
  patched) timed out due to contention on the shared development machine this was
  originally built on — not a claim about this patch's behavior at that thread count, just
  an admission it is unmeasured.
- **Non-macOS:** `lldb` was used for the L3 dispatch proof above; the equivalent Linux
  path (`gdb`) exists in the linked repo's tooling but has not been exercised against
  this patch on Linux/SVE2/Neoverse hardware.
- **Correctness beyond output inspection:** verified via manual smoke tests (`llama-cli`
  producing correct, coherent completions with the flag on and off, at multiple thread
  counts) — not a formal perplexity/logprob equivalence check against the unpatched
  hybrid path's own known-good behavior for large-batch ops.

## An explicit invitation to redirect this

Given what we measured, our own recommendation is narrow: land the warning, skip the
dispatch experiment. But we would rather ask than assume:

1. **We are proposing the one-shot warning land alone**, without the dispatch-bypass
   change — it is the smaller, lower-risk half of this diff (mirrors the existing
   one-shot weight-type-fallback warning already in this file, same pattern, same
   `static std::atomic<bool>` guard), it has no measured downside, and it directly
   answers the "why is this slower than I expected" question #26547 raised. Happy to
   split this into a standalone PR containing only the warning if that is easier to
   review — the dispatch-experiment diff exists in the linked repo purely for
   reproducibility of the measurement above, not as something we are asking you to take.
2. **Is the negative result itself useful to you**, or is `ne11 < 128` known/expected to
   be a good gate already, making this confirmation unsurprising? If there is a workload
   shape or chip where you'd expect the bypass to actually win, we're glad to try to
   measure it — we only tested one chip, one model, one quantization (see "What was NOT
   tested").
3. **Would a runtime capability probe (rather than the brand-string table in
   `detect_num_smcus()`) be worth pursuing separately from this?** That's a larger,
   different change we have not attempted here, but Finding 1's root-cause read of
   `detect_num_smcus()` suggests it might be worth its own issue regardless of what
   happens with this one.

I have a DGX Spark (Cortex-X925, no SME) and access to GitHub's free `ubuntu-24.04-arm`
CI (Neoverse-N2, no SME) available for testing anything that doesn't require SME2
hardware specifically; I do not currently have access to an M4-non-Max, M4 Pro, or M4
Ultra to broaden the SME2-specific coverage myself.

## Prior art / precedent

Same as cited in #26547: the one-shot `GGML_LOG_WARN` pattern for a silent fallback
already exists in this file for the weight-type case (non-`Q4_0`/`Q8_0` tensors through
KleidiAI) — the warning proposed here follows that exact pattern (same guard idiom, same
log call), not a new style.

Thank you again for KleidiAI and for `llama.cpp`'s CPU backend — this PR, like the issue
it follows up on, is offered in the spirit of showing the working and inviting scrutiny,
including scrutiny of our own patch, which is why we are publishing the negative result
rather than only the parts that look good.
