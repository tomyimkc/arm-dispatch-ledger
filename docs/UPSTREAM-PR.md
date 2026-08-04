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

Everything below is written in the voice of the PR author (a first-time contributor to
this codebase) addressing the `llama.cpp`/KleidiAI maintainers directly.

---

## Title

`ggml-cpu: kleidiai: opt-in phase-aware SME dispatch for GEMV ops (decode)`

## Base / branch

- Base: `ggml-org/llama.cpp@dbadb68` (the commit this was developed and measured against)
- Patch: `patches/0001-kleidiai-phase-aware-dispatch.patch` in
  [`tomyimkc/arm-dispatch-ledger`](https://github.com/tomyimkc/arm-dispatch-ledger),
  applied via `git am` as commit `ef973b1` in a local branch
  `kleidiai-phase-aware-dispatch`
- Diffstat: 1 file changed, 56 insertions(+), 3 deletions(-) — `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`
  only. No new files, no new dependencies, no public API or CMake change.

## Summary

Follow-up to #26547 (Finding 1). This PR adds an **opt-in, default-off** env var,
`GGML_KLEIDIAI_PHASE_AWARE=1`, that lets a GEMV-shaped op (decode, `ne11 == 1`) enter the
*existing* SME+NEON hybrid dispatch path once the requested thread count exceeds
`sme_thread_cap`, instead of collapsing to NEON-only. It also adds a one-shot
`GGML_LOG_WARN` for the case this PR leaves alone by default — a GEMV op that *does*
collapse to NEON because of the thread cap — naming the exact knob that would change
that.

**I want to be upfront about scope before anyone reviews the diff: I measured this
patch's actual throughput effect, and it is smaller than I expected, and in some
configurations a statistical tie or a slight regression. I'm not asking you to merge
this because it's a clear performance win — I don't think the evidence supports that
claim. I'm proposing it because (a) the one-shot warning is a clear, low-risk win on
its own, and (b) the opt-in hybrid-dispatch path is a real, correct, measured behavior
change that closes part of the gap for the single most common case — a user who never
passes `-t`/`-tb` at all — even though it does not raise the ceiling. Full numbers
below; please read section "What this PR does NOT achieve" before the "Measured
results" section, not after.**

## What changes, and why

`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`'s dispatch decision (unpatched, this base
commit, lines 1094–1113) gates the hybrid SME+NEON split on:

```cpp
const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);
```

`ne11` is the batch/token count of the matmul. Decode is *always* `ne11 == 1`, so this
term is always true for decode regardless of thread count, and dispatch collapses
unconditionally to the non-SME slot once `nth_total > sme_thread_cap`:

```cpp
chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
```

That is architecturally sound for *why* the gate exists — it avoids hybrid-split
overhead on genuinely tiny batches — but it conflates two different questions: "is this
batch too small to split efficiently" and "is this op a GEMV". Prefill legitimately
benefits from staying out of hybrid mode when `ne11` is small; decode never gets
*evaluated* on its own merits, because `ne11 == 1 < 128` trips the same gate every time,
on every thread count, on every chip in the `detect_num_smcus()` table.

This PR does not remove that gate. It adds a second, opt-in path around it:

```cpp
const bool phase_aware_gemv = ctx.phase_aware_dispatch && is_gemv &&
                               sme_slot != -1 && non_sme_slot != -1;
const bool too_small_for_hybrid = (min_cols_per_thread < 2) ||
                                   (!phase_aware_gemv && ne11 < 128);
```

When `GGML_KLEIDIAI_PHASE_AWARE=1` and the op is GEMV-shaped with both an SME and a
non-SME kernel slot available, the `ne11 < 128` term is bypassed for that op only (the
`min_cols_per_thread < 2` guard still applies — this PR does not touch the
too-small-to-split-at-all case). That routes decode above the cap through the **same**
thread-assignment code prefill hybrid mode already exercises: SME capped at
`sme_thread_cap` threads, the remaining threads running NEON in parallel. No new
thread-splitting logic — this reuses an already-tested path deliberately, because this
codebase's threadpool model requires every thread in `[0, nth_total)` to reach the same
barriers the same number of times per op; an earlier draft of this patch that left the
extra threads idle instead of routing them to NEON was rejected during design for
exactly that deadlock risk.

## Exact files / lines touched

All in `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (base `dbadb68` line numbers cited
below; the patch's own new-file line numbers are in the diff hunks):

| Site (base `dbadb68`) | Change |
|---|---|
| `struct ggml_kleidiai_context` (~line 67) | New field `bool phase_aware_dispatch;`, default `false`, documented inline. |
| `init_kleidiai_context()`, env parsing block (~lines 201–235, alongside the existing `GGML_KLEIDIAI_SME` / `GGML_TOTAL_THREADS` / `GGML_KLEIDIAI_CHUNK_MULTIPLIER` reads) | New `getenv("GGML_KLEIDIAI_PHASE_AWARE")`, parsed with the existing `parse_uint_env()` helper, sets `ctx.phase_aware_dispatch = true` only on a truthy value. |
| `init_kleidiai_context()`, logging block (~line 311, right after the existing "SME disabled"/"SME2 enabled" log lines) | New one-shot `GGML_LOG_INFO` announcing phase-aware dispatch is active, only printed when the flag is set. |
| Dispatch decision in `compute_forward_qx()` / `tensor_traits` (~lines 1094–1113, the `too_small_for_hybrid` / `hybrid_enabled` / `chosen_slot` block described in #26547) | The `phase_aware_gemv` bypass described above, plus a one-shot `GGML_LOG_WARN` fired when a GEMV op collapses to NEON because of the thread cap **and the flag is not set** — this is the "here's the knob" warning, on by default. |

Full diff: `patches/0001-kleidiai-phase-aware-dispatch.patch` in the linked repo (or
inline once this is opened as a real PR).

## The opt-in flag, and why it defaults off

`GGML_KLEIDIAI_PHASE_AWARE=1`, unset by default. Three reasons this is opt-in rather
than the new default behavior:

1. **With the flag unset, every line this patch touches evaluates to exactly what it
   evaluated to before.** This was verified directly, not just argued: running the
   patched binary with the flag unset against a decode workload at `-t 4` reproduces
   the pre-patch ground truth from #26547 *exactly* — `0/15936` SME2 hits, bit-for-bit
   the same "silent fallback" count reported in the issue (`results/dispatch-ledger-darwin-arm64-patched-flag-off.json`
   in the linked repo). I did not want to ship this as a maintainer-facing PR without
   that A/B check, because a "should be a no-op" claim about dispatch logic is exactly
   the kind of claim this whole issue is about not trusting on faith.
2. **My own measurement of the *on* case is a mixed result** (see below) — a real,
   proven dispatch change that does not clearly move the needle on throughput, and at
   one thread count (4) is nominally a small regression versus the unpatched NEON
   collapse, though within overlapping noise bands. That is not evidence strong enough
   to justify changing default behavior for every KleidiAI user.
3. **This has only been measured on one chip, one model, one quantization** (see
   "What was NOT tested" below) — a change this narrow in scope should not become the
   default until it has been reproduced more broadly, and I have not done that work
   yet.

## Measured results (Apple M4 Max, this is the only hardware this was tested on)

Full write-up with every caveat and the reasoning behind each "is this noise or a real
effect" call: `results/OPTIMIZATION.md` in the linked repo. Summary here.

**Setup:** `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (337 MB, Apache-2.0). Baseline binary
untouched at `dbadb68`; patched binary is a separate clone at `ef973b1` (this patch on
top of `dbadb68`), same CMake flags
(`-DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release`).
`n=5` independently-launched process repetitions per cell, interleaved across cells (not
5-in-a-row), median ± stddev reported — never a bare mean. Prefill and decode reported
separately throughout, per this project's own no-blended-speedup-number rule.

### Throughput — four configurations (`tools/crossover.py`)

| Config | decode tok/s (median ± stddev) | prefill tok/s (median ± stddev) |
|---|---:|---:|
| (a) baseline, llama.cpp's real no-flags default (12 threads on this machine, not 16 — see repro note) | 45.5 ± 7.71 | 1145.0 ± 136.05 |
| (b) baseline, hand-tuned split `-t 2 -tb 8`, SME off (this repo's own best-measured split-phase config) | 198.9 ± 16.85 | 2257.5 ± 170.03 |
| (c) **this patch**, flag on, no-flags default | **71.6 ± 11.18** (**+57.3% vs. (a)**) | 1328.4 ± 267.51 (noise — this patch's diff does not touch prefill's dispatch code at all) |
| (d) this patch, flag on, best hand-tuned thread settings | resolves to the identical `-t 2 -tb 8` as (b) — the patch's own gate never activates at `nth_total (2) == sme_thread_cap (2)`, so (d) cannot exceed (b) |

**Where the patch's hybrid path actually activates** (only above `sme_thread_cap`, i.e.
threads 4/8/16 — at 1/2 threads the flag is a provable no-op because
`nth_total <= sme_thread_cap`), decode only, baseline vs. patched (flag on), both
`GGML_KLEIDIAI_SME` unset:

| threads | baseline (no patch) | patched (flag on) | delta |
|---:|---:|---:|---:|
| 4 | 258.2 ± 11.44 | 246.7 ± 6.66 | −4.5% (1-sigma bands overlap: [246.7, 269.6] vs. [240.0, 253.3]) |
| 8 | 143.3 ± 10.66 | 149.2 ± 10.94 | +4.1% (fully overlapping bands — statistical tie) |
| 16 | measurement failed (timeout, both) | measurement failed (timeout, both) | not measured (shared-machine contention, not attributable to this patch) |

### Symbol-level dispatch proof (`tools/verify_dispatch.py`, `lldb`, anchored `^kai_run_matmul` breakpoints, auto-continue, real hit counts)

Same patched binary both rows; only `GGML_KLEIDIAI_PHASE_AWARE` differs:

| threads | workload | flag OFF: SME2 hits / other hits | flag ON: SME2 hits / other hits |
|---:|---|---:|---:|
| 4 | decode | 0 / 15,936 (exact match to pre-patch #26547 ground truth) | **3,072 / 10,428** |
| 8 | decode | 0 / 31,872 | **2,354 / 20,517** |

This is the proof the dispatch change is real and not a selection-log artifact: the
identical binary, identical workload, identical thread count goes from zero SME2 kernel
calls to thousands, purely as a function of the env var. What it does *not* prove is a
throughput win — see the table above, where the same thread counts show a tie/slight
regression, not a gain.

### Reproduction

```sh
# Apply and build the patch (against a fresh dbadb68 checkout)
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git checkout dbadb68
git am /path/to/arm-dispatch-ledger/patches/0001-kleidiai-phase-aware-dispatch.patch
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-cpu llama-cli llama-bench llama-tokenize -j"$(nproc)"

# Four-configuration throughput measurement (from arm-dispatch-ledger)
GGML_KLEIDIAI_PHASE_AWARE=1 python3 tools/crossover.py \
  --llama-bin-dir /path/to/llama.cpp/build/bin --model /path/to/Qwen2.5-0.5B-Instruct-Q4_0.gguf \
  --threads 1,2,4,8,16 --sme-modes on,off --reps 5 --per-call-timeout 60 --retries 2 \
  --platform <your-platform> --out-dir results/crossover/patched

# Symbol-level dispatch proof, flag on and off
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

## What this PR does NOT achieve

Read this before the numbers above, not after:

- **It does not raise the decode ceiling.** The best decode throughput measured
  anywhere in my testing — patched or unpatched — is `-t 2`, SME on, ~305 tok/s
  (`results/crossover/crossover-apple-m4-max.md` and
  `results/crossover/patched/crossover-apple-m4-max-patched-phase-aware.md` in the
  linked repo; both sweeps independently rediscover the same optimum). That
  configuration is already expressible today with **zero code changes** via `-t`. This
  PR's hybrid path, where it activates (threads 4 and 8), tops out at 246.7 and 149.2
  tok/s respectively — both well below that ceiling.
- **It does not beat hand-tuned `-t`/`-tb` flags.** (b) above (198.9–214.7 tok/s decode
  across the states I measured) is roughly 2.8–3× the patched default-config number
  (71.6 tok/s). A user who already knows to pass `-t 2` gets nothing from this patch.
- **Its one clean, attributable win is narrow: the llama.cpp *default* configuration.**
  A user who passes no `-t`/`-tb` flags at all gets decode 45.5 → 71.6 tok/s
  automatically with the flag set (+57.3% relative, but only 26.1 of the 259.5 tok/s
  absolute gap to the `-t 2` ceiling — roughly 10%).
- **It does not touch the deeper limitation `GGML_KLEIDIAI_SME` being process-global.**
  The theoretical best (SME2 for decode, NEON for prefill, simultaneously, in one
  process) is still unreachable after this patch, for the same reason it was
  unreachable before it: there is no per-call or per-phase override of that env var. A
  phase-aware **kernel-family** selector, not just the phase-aware **thread-gate**
  bypass this PR adds, would be the real fix for that, and this PR does not attempt it.

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
  built on — not a claim about this patch's behavior at that thread count, just an
  admission it is unmeasured.
- **Non-macOS:** `lldb` was used for the L3 dispatch proof above; the equivalent Linux
  path (`gdb`) exists in the linked repo's tooling but has not been exercised against
  this patch on Linux/SVE2/Neoverse hardware.
- **Correctness beyond output inspection:** verified via manual smoke tests (`llama-cli`
  producing correct, coherent completions with the flag on and off, at multiple thread
  counts) — not a formal perplexity/logprob equivalence check against the unpatched
  hybrid path's own known-good behavior for large-batch ops.

## An explicit invitation to redirect this

I am not confident this is the right shape for the fix, and would rather ask than
guess:

1. **Would you prefer the one-shot warning land alone, without the opt-in dispatch
   change?** It is the smaller, lower-risk half of this diff (mirrors the existing
   one-shot weight-type-fallback warning already in this file, same pattern, same
   `static std::atomic<bool>` guard), it has no measured downside I'm aware of, and it
   directly answers the "why is this slower than I expected" question #26547 raised —
   independent of whether the hybrid-dispatch opt-in is wanted at all. I'd be glad to
   split this into two PRs if that's easier to review, or drop the dispatch-change half
   entirely if the maintainers would rather solve this a different way (e.g. a runtime
   capability probe instead of the brand-string table, which is a separate, larger
   change I have not attempted here).
2. **Is an env var the right mechanism, or would a `GGML_KLEIDIAI_SME` value (rather
   than a second, separate flag) be preferred** — e.g. a mode that means "hybrid for
   everything, including GEMV"? I chose a new, narrowly-scoped flag to minimize
   interaction with existing `GGML_KLEIDIAI_SME` semantics, but I have not thought hard
   about whether that is the cleanest long-term API.
3. **Is the measured tie/slight-regression at threads=4 (−4.5%, within noise) worth
   worrying about, or is it noise I'm over-reading?** I called it "noise, not confidently
   a regression" in my own write-up because the 1-sigma bands overlap, but I only have
   `n=5` per cell on a shared, contended machine (load average up to ~100 on a
   16-physical-core box during this run — full contention log in
   `results/OPTIMIZATION.md` §5) and would trust a maintainer's read on a quieter box
   more than mine.
4. **Should this wait for broader hardware/model coverage before being proposed at
   all?** I filed the issue first specifically to get a read on whether this was worth
   pursuing before writing more code; happy to hear "not yet" here too.

I have a DGX Spark (Cortex-X925, no SME) and access to GitHub's free `ubuntu-24.04-arm`
CI (Neoverse-N2, no SME) available for testing anything that doesn't require SME2
hardware specifically; I do not currently have access to an M4-non-Max, M4 Pro, or M4
Ultra to broaden the SME2-specific coverage myself.

## Prior art / precedent

Same as cited in #26547: the one-shot `GGML_LOG_WARN` pattern for a silent fallback
already exists in this file for the weight-type case (non-`Q4_0`/`Q8_0` tensors through
KleidiAI) — the warning added here follows that exact pattern (same guard idiom, same
log call), not a new style.

Thank you again for KleidiAI and for `llama.cpp`'s CPU backend — this PR, like the issue
it follows up on, is offered in the spirit of showing the working and inviting scrutiny,
not asserting a finished performance win.
