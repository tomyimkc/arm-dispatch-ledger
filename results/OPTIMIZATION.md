<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
-->
# Optimization verdict: does the phase-aware KleidiAI patch actually help?

> **Superseded on throughput, still authoritative on mechanism.** The throughput/verdict
> numbers below were retracted and corrected by
> **[`results/REMEASURE-2026-08-04-QUIET.md`](REMEASURE-2026-08-04-QUIET.md)** — read
> that file first. It supersedes every tok/s figure and every percentage-improvement
> claim in this document. This file remains authoritative for the parts unaffected by
> the retraction: what was built (§1), the symbol-level dispatch proof (§4), and the
> documentation of the full threads×SME grid (§2/§3, now annotated below). If any other
> file in the submission (README, `docs/SUBMISSION.md`, the demo) disagrees with
> `REMEASURE-2026-08-04-QUIET.md` on a number, the re-measurement file is right and they
> must be corrected.

## TL;DR verdict (corrected 2026-08-04, quiet/interleaved re-measurement)

**The tuning discovery is the real, headline result, and it needs zero code changes:**
on a quiet, round-robin-interleaved re-measurement, `-t 2` takes decode from a
93.6 tok/s default to **321.0 tok/s — a 3.43x improvement** — and `-t 8` takes prefill
from a 1230.3 tok/s default to **2198.1 tok/s — a 1.79x improvement.** Both flags already
ship in stock `llama.cpp`; nothing about this needs the patch, and nothing in
`llama.cpp`'s own banner or docs currently tells a user this gap exists.

**The phase-aware KleidiAI patch does not help, and at default thread count it measurably
hurts.** With `GGML_KLEIDIAI_PHASE_AWARE=1` at the no-flags default thread count (12 on
this machine), decode goes from 93.6 to **82.5 tok/s — 0.88x, ~12% *slower*** — a real
regression, not noise (93.6 ± 2.47 and 82.5 ± 4.07 do not overlap). At `-t 2` the patch is
inert (321.0 → 317.5 tok/s, a statistical tie, because `-t 2` never exceeds
`sme_thread_cap` so the patch's branch is never entered). Prefill at default is also a
tie (1230.3 → 1202.1 tok/s), since the patch's diff never touches the prefill GEMM path.

**What the patch *does* prove, and what remains true about it:** `tools/verify_dispatch.py`
shows the dispatch change is real and mechanically working exactly as designed —
decode@t=4 goes from 0 to 3,072 SME2 kernel hits, decode@t=8 from 0 to 2,354, flag off
vs. flag on, same binary (§4, unchanged, still valid — these are symbol-level breakpoint
counts, not timings, so they are unaffected by the contention that invalidated the
throughput numbers). **Dispatching SME2 is not the same as being faster**: at 12 threads
the hybrid split gives SME2 only 2 of them while 10 run NEON, and coordinating that split
costs more than the SME lane returns at this shape, so pure NEON on all 12 threads wins.
Upstream's existing exclusion of decode from the hybrid path is, on this chip and this
model, the *better* default — the patch's own premise (that the exclusion was costing
throughput) is not supported by measurement, even though the exclusion itself is real and
the *warning* half of the patch (telling a user SME2 is silently unused) still stands on
its own merit and costs nothing to upstream independently of the dispatch change.

**Why the numbers changed:** the sweep this file originally reported (§2/§3 below) ran
while this machine's 1-minute load average was 66–147 on 16 cores, and — critically —
measured the baseline and patched binaries in separate, non-interleaved time windows, so
each saw different amounts of external contention. That combination manufactured a fake
"+57.3%" win. The 2026-08-04 re-measurement fixed both problems (round-robin interleaving,
quieter host) and reversed the verdict. See `REMEASURE-2026-08-04-QUIET.md` for full
method, data, and reasoning.

---

## 1. What was built and verified before any measurement

- **Baseline binary**: `/tmp/llama.cpp` (untouched throughout this session), built at
  `dbadb68`, `-DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON
  -DCMAKE_BUILD_TYPE=Release`. `llama-bench`'s own `build_commit` field reports
  `dbadb68` in every baseline JSON row below.
- **Patched binary**: `/tmp/llama-phase-aware`, a separate clone, branch
  `kleidiai-phase-aware-dispatch`, with `patches/0001-kleidiai-phase-aware-dispatch.patch`
  applied via `git am` on top of `dbadb68` (commit `ef973b1`). Same cmake flags as
  baseline. This session additionally built the `llama-bench` and `llama-tokenize`
  targets in this clone (only `llama-cli` had been built previously by the patch's
  own work package), so `tools/crossover.py` could run its full methodology against
  the patched binary. `llama-bench`'s own `build_commit` field reports `ef973b1` in
  every patched-sweep JSON row below — direct machine-readable confirmation that the
  patched binary, not the baseline, produced those numbers.
- **Smoke test**: `GGML_KLEIDIAI_SME=2 ./build/bin/llama-cli -m /tmp/ggufs/q05.gguf -p
  "The capital of France is" -n 8 -no-cnv -st --simple-io -t 4` on the patched binary
  produced correct, coherent output ("The capital of France is Paris.") before any
  benchmark ran.

## 2. The four configurations (`tools/crossover.py`, ≥5 reps, interleaved, median+stddev)

> **This section's numbers are SUPERSEDED for cross-run (baseline-vs-patched)
> comparison — see `REMEASURE-2026-08-04-QUIET.md`.** Each cell below *was* interleaved
> across its own 20-cell grid, but the (a)/(b) baseline sweep and the (c)/(d) patched
> sweep were run as two separate invocations roughly 40 minutes apart, under heavy and
> independently-varying contention (1-minute load average 66–147 on 16 cores; see §5).
> That means any (a)-vs-(c) or (b)-vs-(d) delta below reflects contention drift as much
> as it reflects the patch, which is exactly what produced the retracted "+57.3%" figure.
> Kept here as the honest raw record and because it is the only place in this repo with
> the full 1/2/4/8/16-thread × on/off-SME grid (the quiet re-measurement only re-ran
> `-t 2`/`-t 8`/default). Read the interpretive sentences after each table as **historical
> — do not cite the percentages below as current.**

All numbers below are `median_ts` (tok/s) ± `stddev_ts` across `n` independently
launched process reps, exactly as `tools/crossover.py` reports (never a bare mean).
Full raw data: `results/crossover/crossover-apple-m4-max.json` ((a)/(b), baseline
binary) and `results/crossover/patched/crossover-apple-m4-max-patched-phase-aware.json`
((c)/(d), patched binary, run with `GGML_KLEIDIAI_PHASE_AWARE=1` exported into the
harness's own subprocess environment for every call).

### (a) BASELINE, llama.cpp default thread count (no `-t`/`-tb` flags)

`crossover.py`'s own instrumentation confirmed the real no-flags default on this
machine is **12** threads (`hw.perflevel0.physicalcpu`), not 16 — see
`tools/crossover.md` §6. `GGML_KLEIDIAI_SME` unset (auto-detect, the real default).

| phase | median tok/s | stddev | n |
|---|---:|---:|---:|
| decode | 45.5 | 7.71 | 5 |
| prefill | 1145.0 | 136.05 | 5 |

### (b) BASELINE, best hand-tuned split (`llama-cli -t 2 -tb 8`)

`-t 2`/`-tb 8` are this repo's own sweep-derived per-phase optima (decode:
threads=2/SME-on; prefill: threads=8/SME-off). `GGML_KLEIDIAI_SME` is still a single
process-global setting, so both states were measured; **SME=off wins both metrics**
and is the config actually compared against below.

| GGML_KLEIDIAI_SME | prefill median tok/s | stddev | decode median tok/s | stddev | n |
|---|---:|---:|---:|---:|---:|
| on  | 2148.3 | 363.57 | 197.9 | 16.66 | 5 |
| **off** | **2257.5** | 170.03 | **198.9** | 16.85 | 5 |

### (c) PATCHED (`GGML_KLEIDIAI_PHASE_AWARE=1`), default thread count

Same no-flags invocation as (a), same `GGML_KLEIDIAI_SME` auto-detect state, run
against the patched binary with the flag exported.

| phase | median tok/s | stddev | n |
|---|---:|---:|---:|
| decode | **71.6** | 11.18 | 5 |
| prefill | 1328.4 | 267.51 | 5 |

**Historical note — retracted:** this cell (71.6) was originally reported as "+26.1 tok/s,
+57.3% vs (a)" and read as the patch's one clean win. That comparison is **retracted**: (a)
and (c) were measured in separate, non-interleaved contention windows, which is precisely
the failure mode that manufactures a fake delta. The quiet, interleaved re-measurement in
`REMEASURE-2026-08-04-QUIET.md` found the opposite at default threads: 93.6 → 82.5 tok/s,
**0.88x, ~12% slower**, a real and outside-noise regression. The **prefill** delta seen
here (+183.4 tok/s, +16.0% vs (a)) was correctly flagged as noise even at the time — the
patch's diff is scoped entirely to the GEMV (`ne11==1`, decode) branch, and prefill's
`too_small_for_hybrid` gate (`ne11 >= 128`) is untouched — and the quiet re-measurement
confirms it: prefill patched-vs-baseline at default is a tie (1230.3 vs 1202.1, 0.98x).

### (d) PATCHED, phase-aware flag + best thread settings

`crossover.py`'s own full sweep against the patched binary (§3) re-derives the
per-phase optima independently — they land on the **same** thread counts as the
baseline sweep (decode: threads=2/SME-on; prefill: threads=8/SME-off), so the
split-phase command is again `llama-cli -t 2 -tb 8`.

| GGML_KLEIDIAI_SME | prefill median tok/s | stddev | decode median tok/s | stddev | n |
|---|---:|---:|---:|---:|---:|
| on  | 2309.1 | 216.21 | 201.8 | 5.82 | 5 |
| off | 2552.9 | 31.15 | 214.7 | 4.83 | 5 |

**This is not a fair "does the patch help" comparison, and must not be read as one.**
At `-t 2`, `nth_total (2) == sme_thread_cap (2)`, never `>`, so the patch's own
`phase_aware_gemv` branch (which only fires when `nth_total > sme_thread_cap`) is
**never entered** for decode at this thread count, patched or not. The prefill side
(`-tb 8`) is untouched by the patch by design. So (d) at `-t 2 -tb 8` is, architecturally,
the *same unpatched code path* as (b) at `-t 2 -tb 8` — the small numeric differences
above (e.g., prefill 2552.9 vs (b)'s 2257.5, both historical/superseded figures, +13.1%)
are shared-machine contention-window noise between two runs made roughly 40 minutes apart
(see §5's load-average numbers), **not evidence the patch changed anything here.** The
correct, and now independently confirmed, statement is: **the best achievable
configuration is unchanged by this patch** — the quiet re-measurement's own `-t 2` tie
(321.0 vs 317.5 tok/s, §TL;DR) reaches the identical conclusion architecturally, without
relying on a non-interleaved cross-run comparison.

## 3. Where the patch's hybrid path actually activates: the full sweep, cell by cell

> This table's *within-run* comparisons (baseline vs. patched, same contention window,
> same 20-cell interleaved sweep) are more informative than §2's cross-run deltas, since
> both sides of each row below come from binaries measured under the same drifting load.
> It agrees directionally with the quiet re-measurement's Verdict 2 (no clear win for the
> patch at the thread counts where it activates) — see `REMEASURE-2026-08-04-QUIET.md`
> for the clean, quiet-host confirmation at default thread count specifically.

`too_small_for_hybrid`'s `ne11 < 128` term is only bypassed for decode when
`nth_total > sme_thread_cap` (2 here) — i.e., the patch can only matter at threads
4, 8, 16. Comparing the same cells across the baseline sweep
(`results/crossover/crossover-apple-m4-max.json`) and the patched sweep
(`results/crossover/patched/crossover-apple-m4-max-patched-phase-aware.json`), both
`GGML_KLEIDIAI_SME` unset ("on"/auto-detect, required for the patch to have any
effect at all):

| threads | decode, baseline (no patch) | decode, patched (`PHASE_AWARE=1`) | delta |
|---:|---:|---:|---:|
| 1 (≤cap, patch inert) | 187.6 ± 19.86 | 197.4 ± 6.38 | +5.2% (noise) |
| 2 (≤cap, patch inert) | 305.4 ± 52.88 | 305.0 ± 8.01 | −0.1% (identical, as expected) |
| **4 (patch active)** | **258.2 ± 11.44** | **246.7 ± 6.66** | **−4.5%** |
| **8 (patch active)** | **143.3 ± 10.66** | **149.2 ± 10.94** | **+4.1% (within combined noise)** |
| 16 (patch active) | measurement failed (timeout, both runs) | measurement failed (timeout, both runs) | not measured |

**Neither above-cap cell where the patch is actually active shows a clear win.** At
threads=4 the patched hybrid path (246.7 ± 6.66) is nominally 4.5% lower than the
unpatched NEON collapse (258.2 ± 11.44), but the two 1-sigma bands overlap
substantially (baseline [246.7, 269.6], patched [240.0, 253.3] — overlap region
[246.7, 253.3]), so this is a small, directionally-negative result that should be read
as "no measured benefit, possibly a small cost," not a confidently-proven regression.
At threads=8 the two are a clear statistical tie (143.3 ± 10.66 vs. 149.2 ± 10.94,
fully overlapping bands). **At no thread count does the patched hybrid path approach
the 305 tok/s ceiling that plain `-t 2` (no patch, no flag) already achieves** — that
part of the comparison is not close enough to be a noise question. This is the core reason (d) cannot beat (b): the patch never
produces a decode number competitive with the existing `-t 2` optimum, so routing the
"best settings" search through the patched binary rediscovers the same `-t 2`
optimum the baseline sweep already found.

Threads=16 failed to complete in both the baseline and patched sweeps (`llama-bench`
exit 124, timeout, all 5 reps × 2 SME states in each run) — this reproduces the same
failure this repo's crossover baseline run already documented
(`results/crossover/crossover-apple-m4-max.md`, "Contention note"), not a patch-caused
regression; see §5.

## 4. Symbol-level dispatch proof (`tools/verify_dispatch.py`) — did the kernel family actually change?

A throughput delta alone is not proof of anything (per this project's own standard,
`results/GROUND-TRUTH-DISPATCH.md`): the L2 selection log claims `SME2` in every
config below regardless of what actually ran. Only L3 (lldb, `kai_run_matmul.*`
breakpoints, auto-continuing, real per-symbol call counts) is decisive. All four runs
below used the **same patched binary** (`/tmp/llama-phase-aware/build/bin/llama-cli`),
varying only whether `GGML_KLEIDIAI_PHASE_AWARE=1` was set — an apples-to-apples,
single-binary A/B on the dispatch decision itself.

**Flag OFF (control — patched binary, unset flag, must reproduce pre-patch behavior
exactly):** `results/dispatch-ledger-darwin-arm64-patched-flag-off.json`

| threads | workload | advertised (L2) | executed (L3) | hits (adv/other) | verdict |
|---:|---|---|---|---|---|
| 4 | decode_short | SME2 | dotprod | **0/15936** | **SILENT_FALLBACK** |
| 8 | decode_short | SME2 | dotprod | **0/31872** | **SILENT_FALLBACK** |
| 4 | prefill_long | SME2 | dotprod | 2362/6582 | SME2_HYBRID_DISPATCH |
| 8 | prefill_long | SME2 | dotprod | 1618/13621 | SME2_HYBRID_DISPATCH |

The decode rows (`0/15936`, `0/31872`) are an **exact reproduction** of the baseline
ground truth in `results/GROUND-TRUTH-DISPATCH.md` (`t=4 decode -> dotprod, 0
sme2/15936`) and `results/SUMMARY.md`'s dispatch table — proof the patched binary,
with the flag unset, is bit-for-bit behaviorally identical to the unpatched baseline
on this decisive metric, exactly as the patch's own design goal claims ("no behavior
change when the env var is unset").

**Flag ON — the patch's actual effect:** `results/dispatch-ledger-darwin-arm64-patched-flag-on.json`

| threads | workload | advertised (L2) | executed (L3) | hits (adv/other) | verdict |
|---:|---|---|---|---|---|
| 4 | decode_short | SME2 | dotprod | **3072/10428** | **SME2_HYBRID_DISPATCH** |
| 8 | decode_short | SME2 | dotprod | **2354/20517** | **SME2_HYBRID_DISPATCH** |
| 4 | prefill_long | SME2 | dotprod | 2528/6410 | SME2_HYBRID_DISPATCH |
| 8 | prefill_long | SME2 | dotprod | 2006/13144 | SME2_HYBRID_DISPATCH |

**This is the symbol-level proof the patch does what it says**: decode at threads=4
goes from **0 SME2 kernel calls (flag off) to 3072 SME2 kernel calls (flag on)** in
the identical binary, identical workload, identical thread count — a genuine dispatch
change, not a selection-log artifact. Decode at threads=8 shows the same pattern
(0 → 2354). Prefill's hit counts move only by ordinary run-to-run variance (2362→2528,
1618→2006 — both still comfortably in the "hundreds to low-thousands of hits" band
prefill_long showed throughout this project's baseline measurements), consistent with
prefill's dispatch logic being untouched by the patch.

**The combined result of §3 and §4 is the real finding of this work package: the
patch's dispatch change is real and proven, but it is not sufficient — a changed
kernel family at threads=4/8 does not translate into a throughput win over the
already-collapsed NEON path at those same thread counts, and it does not approach the
`-t 2` ceiling.** More silicon activity (SME2 + NEON running concurrently) does not
automatically mean more useful throughput once thread-assignment overhead and the
2-core SME cap's serialization are accounted for — exactly the kind of result the
patch's own README (`patches/README.md`) flagged in advance as unmeasured
("plausible... but 'plausible' is not 'measured'").

## 5. Contention context (read before trusting any absolute magnitude above)

**This is the section that turned out to matter most.** The gap this section documents —
baseline and patched swept ~40 minutes apart, under independently-varying heavy load —
is exactly the flaw `REMEASURE-2026-08-04-QUIET.md` was written to fix (round-robin
interleaving + a quieter host), and fixing it reversed §2/§6's original verdict. Read this
section as the diagnosis of *why* the original numbers were wrong, not as a caveat that
merely widens their error bars.

Both crossover sweeps, and both dispatch probes, ran on a machine shared with other
concurrent agent sessions (this repo's own multi-agent working agreement).
`uptime` 1-minute load average during this session's runs ranged from ~5 to **over
100** on a 16-physical-core machine:

- Baseline sweep ((a)/(b)): `results/crossover/crossover-apple-m4-max.json`,
  `n_retries_used=10`, all 10 in `decode threads=16` (both SME states) — pre-existing,
  documented in that file's own "Contention note."
- Patched sweep ((c)/(d)): `results/crossover/patched/crossover-apple-m4-max-patched-phase-aware.json`,
  `n_retries_used=20`, same `decode threads=16` failure pattern (all 5 reps × 2 SME
  states), plus load average **peaked at 1-minute=100.66 mid-run** (`uptime` snapshot
  taken directly during this session, 11:32). The patched sweep took 2113.3s
  (35 min) vs. the baseline sweep's 1205.7s (20 min) for the identical grid size —
  consistent with heavier concurrent load during this run, not a patch-caused slowdown
  (the two sweeps ran ~40 minutes apart on the same machine under different,
  independently-observed load).
- The dispatch-verify runs (§4) completed cleanly (33–90s each, no timeouts) at a
  quieter point in the same session.

Threads=16 decode failed to produce **any** successful measurement in either sweep (10
timeouts baseline, 10 timeouts patched) — this is the same failure this project's own
`results/crossover/crossover-apple-m4-max.md` already documents as contention-linked,
not interpolated, not estimated, and not attributable to the patch (the patched binary
was never successfully measured at threads=16 for decode, so no claim is made about
that cell either way).

**Relative comparisons within a phase (which config wins) are more robust to this
contention than absolute tok/s magnitudes** — the same caveat this project's own
`tools/crossover.md` §5 states for the baseline data applies equally here, and was
treated as a real constraint (not a corrected number) throughout this write-up: every
"noise, not a patch effect" call above (the prefill deltas in (c) and (d)) was made
because the stddev bands overlap or the code path is provably unaffected by the patch
— never because the number was inconvenient.

## 6. Verdict, explicitly (corrected 2026-08-04 — supersedes the original §6)

**Does the phase-aware patch beat the llama.cpp default?**
**No — at default thread count it is a measured regression.** The quiet, interleaved
re-measurement (`REMEASURE-2026-08-04-QUIET.md`) found decode goes from **93.6 to 82.5
tok/s, 0.88x, ~12% slower**, with non-overlapping error bands (93.6 ± 2.47 vs. 82.5 ±
4.07) — real, not noise. Prefill at default is a tie (1230.3 → 1202.1, 0.98x). The
originally-reported "45.5 → 71.6 tok/s, +57.3%" win is **retracted**: it was an artifact
of measuring the baseline and patched binaries in separate, non-interleaved contention
windows (§5).

**Does the phase-aware patch beat the hand-tuned split already achievable today
(`-t 2` for decode, `-t 8` for prefill)?**
**No, and it isn't close.** The quiet re-measurement's tuned decode number is 321.0
tok/s — roughly **3.9x** the patch's regressed default-config number (82.5 tok/s). At
`-t 2` itself the patch is architecturally inert (`nth_total == sme_thread_cap`, so its
branch never fires) and measures as a statistical tie with the unpatched binary (321.0 vs
317.5 tok/s, 0.99x).

**Does the patch raise the ceiling — the best decode/prefill numbers achievable at all,
patch or no patch?**
**No.** The decode ceiling in this repo is `-t 2`, ~305–321 tok/s depending on
contention at measurement time — a config the patch's own logic never touches
(`nth_total ≤ sme_thread_cap`). The patch's hybrid path only activates at threads 4/8,
where even the original contended sweep (§3) showed no clear win over the unpatched
NEON-collapse path, and the quiet default-thread-count re-measurement shows an outright
loss.

**What is real and stays in this submission:** the dispatch change itself. `tools/verify_dispatch.py`
proves, at the symbol level and independent of any timing contention, that
`GGML_KLEIDIAI_PHASE_AWARE=1` moves decode from 0 SME2 kernel hits to 3,072 (t=4) / 2,354
(t=8) — §4, unaffected by this retraction. The honest conclusion is that this dispatch
change is real but not sufficient: routing more work through SME2 at these thread counts
costs more in split/coordination overhead than it returns, so the upstream exclusion this
patch tried to lift is, on this chip and this model, the *better* default. Report the
phase-aware dispatch half of the patch to upstream as a **measured negative result** —
useful information, not a performance fix.

**Honest summary:** the genuinely actionable, user-facing finding in this whole
submission is the **tuning** one, and it needs no patch at all: `llama-cli -t 2` for
decode is **3.43x** the no-flags default, and `-t 8` for prefill is **1.79x** the
no-flags default, today, in stock `llama.cpp`, and nothing currently tells a user this
gap exists. The `GGML_KLEIDIAI_SME`-is-process-global limitation this project's
diagnosis phase identified — that the *theoretical* best (SME2-decode + NEON-prefill,
simultaneously, in one process) is `[NOT YET ACHIEVABLE]` — remains exactly as
unachievable after this patch as before it, and this patch does not narrow that gap
either; a phase-aware **kernel-family** selector (not just a phase-aware **thread-count**
gate, which is all this patch adds) is still the open problem.

## 7. Reproduce

```sh
# Patched binary (already applied/built in this session at /tmp/llama-phase-aware):
cd /tmp/llama-phase-aware && git log --oneline -1   # ef973b1 ggml-cpu: kleidiai: phase-aware SME dispatch...
cmake --build build --target ggml-cpu llama-cli llama-bench llama-tokenize -j"$(sysctl -n hw.ncpu)"

# (a)/(b) baseline — already committed, not re-run for this file:
#   results/crossover/crossover-apple-m4-max.{json,md}

# (c)/(d) patched — historical, superseded evidence (§2 disclaimer applies):
GGML_KLEIDIAI_PHASE_AWARE=1 python3 tools/crossover.py \
  --llama-bin-dir /tmp/llama-phase-aware/build/bin --model /tmp/ggufs/q05.gguf \
  --threads 1,2,4,8,16 --sme-modes on,off --reps 5 --per-call-timeout 60 --retries 2 \
  --platform apple-m4-max-patched-phase-aware --out-dir results/crossover/patched

# Symbol-level dispatch proof (both flag states, same patched binary) — still valid:
python3 tools/verify_dispatch.py --binary /tmp/llama-phase-aware/build/bin/llama-cli \
  --model /tmp/ggufs/q05.gguf --threads 4,8 --workloads decode_short,prefill_long \
  --env GGML_KLEIDIAI_PHASE_AWARE=1 --l3-timeout 240 \
  --out results/dispatch-ledger-darwin-arm64-patched-flag-on.json --assert

python3 tools/verify_dispatch.py --binary /tmp/llama-phase-aware/build/bin/llama-cli \
  --model /tmp/ggufs/q05.gguf --threads 4,8 --workloads decode_short,prefill_long \
  --l3-timeout 240 \
  --out results/dispatch-ledger-darwin-arm64-patched-flag-off.json --assert

# Corrected throughput verdict — the authoritative reproduce commands for §TL;DR/§6:
# see "Reproduce" in results/REMEASURE-2026-08-04-QUIET.md (round-robin interleaved,
# 7 reps/config, quieter host).
```

## 8. Caveats

1. This entire measurement is single-machine (Apple M4 Max), single-model
   (Qwen2.5-0.5B-Instruct Q4_0) — the same scope limitation the rest of this repo's
   results carry. Not yet reproduced on the DGX Spark, GitHub's `ubuntu-24.04-arm`
   runner (no SME2 there at all — this patch is Apple-only by construction, since
   `detect_num_smcus()` is an `__APPLE__`-only code path), or a larger model.
2. Threads=16 decode is entirely unmeasured for both baseline and patched binaries in
   this session (100% timeout rate, both runs) — no claim is made about that cell.
   `patches/README.md` separately documents that its own `-t 16` *dispatch* check
   (not throughput) was done via a temporary source-level log rather than a completed
   lldb sweep, for the same reason (this machine's shared contention). Not re-tested by
   the quiet re-measurement either (it only re-ran default, `-t 2`, `-t 8`).
3. **The (c)/(d) patched sweep and the (a)/(b) baseline sweep were run ~40 minutes apart
   under measurably different shared-machine load (§5), and this — not any property of
   the patch — is what produced the original "+57.3%" figure.** This is no longer just a
   caveat: it is the confirmed root cause of the retraction. See
   `REMEASURE-2026-08-04-QUIET.md` for the round-robin-interleaved fix and the corrected
   numbers (§TL;DR, §6).
4. `results/dispatch-ledger-darwin-arm64.json` (the pre-existing, committed baseline
   ledger from this project's diagnosis phase) was deliberately **not** overwritten;
   this file's new dispatch evidence lives in the two new
   `dispatch-ledger-darwin-arm64-patched-flag-{on,off}.json` files instead. This evidence
   is unaffected by the throughput retraction (§4).
5. No claim in this file has been folded into a single blended "speedup" number across
   phases, per the task brief's explicit instruction — prefill and decode are reported
   and judged separately throughout, including in the corrected §TL;DR/§6.
