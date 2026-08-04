<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
-->
# Optimization verdict: does the phase-aware KleidiAI patch actually help?

> **Authoritative.** Measured on Apple M4 Max, this repo, 2026-08-04. This file is the
> factual basis for any optimization claim made elsewhere in the submission (README,
> `docs/SUBMISSION.md`, the demo). If those files disagree with this one on a number,
> this file is right and they must be corrected.

## TL;DR verdict

**The patch is real, opt-in, and its dispatch effect is proven at the symbol level: with
`GGML_KLEIDIAI_PHASE_AWARE=1`, decode above `sme_thread_cap` genuinely enters a hybrid
SME2+NEON path instead of collapsing to NEON-only — confirmed by non-zero SME2 kernel
hit counts where the unpatched/flag-off binary shows exactly zero.**

**But the patch does not raise the performance ceiling.** The best decode configuration
found in this repo (`-t 2`, SME on, ~305 tok/s, first documented in `results/SUMMARY.md`)
is *already* expressible today with zero code changes, and the patch cannot beat it,
match it, or even meaningfully approach it at the thread counts where it actually
activates (4, 8 threads: 149–247 tok/s, both patched and unpatched). The one place the
patch shows a real, repeatable, honest win is **the llama.cpp *default* configuration**
— the thread count (12 on this machine) a user gets by passing **no** `-t`/`-tb` flags
at all — where decode throughput goes from **45.5 tok/s (unpatched) to 71.6 tok/s
(patched), a +57% improvement, automatically, with no user action.** That is a real,
if modest and non-ceiling-raising, result: it does not require a user to know that `-t 2`
exists. It does **not** beat the hand-tuned split-phase config `-t 2 -tb 8` (decode
199–215 tok/s in this session), which remains strictly better on both phases and is
already achievable today without any patch.

The honest framing, matching the task brief's own words: **the tuning is expressible
today via `-t`/`-tb`; the patch's only demonstrated value is that a *fraction* of that
tuning happens automatically for a user who passes no flags at all** — it narrows, but
does not close, the gap between "default llama.cpp" and "expert-tuned llama.cpp," and it
does not touch the `GGML_KLEIDIAI_SME`-is-process-global limitation that keeps the
*theoretical* best (SME2-decode + NEON-prefill simultaneously) out of reach for either
the patched or unpatched binary.

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

**Decode: +26.1 tok/s, +57.3% vs (a).** This is the patch's one clean, attributable win
— see §4 for why it is real and why it is bounded. **Prefill: +183.4 tok/s, +16.0% vs
(a), but this cannot be attributed to the patch** — the patch's diff is scoped
entirely to the GEMV (`ne11==1`, decode) branch; prefill's `too_small_for_hybrid` gate
(`ne11 >= 128`) is untouched. The prefill delta here is inside the combined stddev
band (136.05 + 267.51) and is measurement noise from running at a different point in
this shared machine's contention cycle (see §5), not a patch effect. Report it as
noise, not as a second win.

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
above (e.g., prefill 2552.9 vs 2257.5, +13.1%) are shared-machine contention-window
noise between two runs made roughly 40 minutes apart (see §5's load-average numbers),
**not evidence the patch changed anything here.** The correct statement is: **the best
achievable configuration is unchanged by this patch.**

## 3. Where the patch's hybrid path actually activates: the full sweep, cell by cell

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

## 6. Verdict, explicitly

**Does the phase-aware patch beat (a), the llama.cpp default?**
**Yes, on decode, by a real and repeatable margin: 45.5 → 71.6 tok/s (+57.3%),
automatically, with the flag set and no thread flags at all.** This is the patch's one
clean, defensible win. (Its apparent prefill improvement in the same comparison is
noise, not a patch effect — see §2(c).)

**Does the phase-aware patch beat (b), the hand-tuned split already achievable today?**
**No.** (b)'s decode throughput (198.9–214.7 tok/s across both SME states measured in
this session) is roughly **2.8–3.0x higher** than the patch's best default-config
decode number (71.6 tok/s), and the patch's own best-thread-settings measurement (d)
literally cannot exceed (b) because it resolves to the identical, patch-inert `-t 2
-tb 8` configuration (§2(d), §3).

**Does the patch raise the ceiling — the best decode/prefill numbers achievable at
all, patch or no patch?**
**No.** The global decode optimum in every sweep run in this repo, baseline or
patched, is `-t 2`, SME on, ~305 tok/s — a config the patch's own logic never touches
(`nth_total ≤ sme_thread_cap`). The patch's hybrid path, where it does activate
(threads 4 and 8), tops out at 246.7 and 149.2 tok/s respectively — both well below
the `-t 2` ceiling, and at threads=4 nominally (not confidently, given overlapping
noise bands — see §3) *below* even the unpatched NEON-collapse number at that same
thread count.

**Honest summary, in the words this task brief asked for:** the tuning this
submission is built around — SME2-for-decode, NEON-for-prefill — **is expressible
today via `llama-cli -t 2 -tb 8`, with zero code changes**, and remains strictly the
best measured configuration in this repo. The phase-aware patch's real,
symbol-level-proven contribution is narrower than "the fix": it automatically claws
back about **10% of the gap** between "a user who passes no flags" (45.5 tok/s) and
the `-t 2` ceiling (~305 tok/s) — 45.5 → 71.6 tok/s is a real, repeatable +57.3%
*relative* improvement on a low baseline, but only a modest dent in the *absolute*
259.5 tok/s gap to the ceiling (26.1 tok/s closed of 259.5) — for the specific,
common case of a user who never tunes thread counts at all. It does not close that gap,
it does not beat expert manual tuning, and it does not touch the deeper limitation
this project's diagnosis phase already identified — `GGML_KLEIDIAI_SME` is still a
single process-global setting, so the *theoretical* best (SME2-decode + NEON-prefill,
simultaneously, in one process) documented as `[NOT YET ACHIEVABLE]` in
`results/crossover/crossover-apple-m4-max.md` remains exactly as unachievable after
this patch as before it. A phase-aware **kernel-family** selector (not just a
phase-aware **thread-count** gate, which is all this patch adds) is still the
open problem.

## 7. Reproduce

```sh
# Patched binary (already applied/built in this session at /tmp/llama-phase-aware):
cd /tmp/llama-phase-aware && git log --oneline -1   # ef973b1 ggml-cpu: kleidiai: phase-aware SME dispatch...
cmake --build build --target ggml-cpu llama-cli llama-bench llama-tokenize -j"$(sysctl -n hw.ncpu)"

# (a)/(b) baseline — already committed, not re-run for this file:
#   results/crossover/crossover-apple-m4-max.{json,md}

# (c)/(d) patched — this file's new evidence:
GGML_KLEIDIAI_PHASE_AWARE=1 python3 tools/crossover.py \
  --llama-bin-dir /tmp/llama-phase-aware/build/bin --model /tmp/ggufs/q05.gguf \
  --threads 1,2,4,8,16 --sme-modes on,off --reps 5 --per-call-timeout 60 --retries 2 \
  --platform apple-m4-max-patched-phase-aware --out-dir results/crossover/patched

# Symbol-level dispatch proof (both flag states, same patched binary):
python3 tools/verify_dispatch.py --binary /tmp/llama-phase-aware/build/bin/llama-cli \
  --model /tmp/ggufs/q05.gguf --threads 4,8 --workloads decode_short,prefill_long \
  --env GGML_KLEIDIAI_PHASE_AWARE=1 --l3-timeout 240 \
  --out results/dispatch-ledger-darwin-arm64-patched-flag-on.json --assert

python3 tools/verify_dispatch.py --binary /tmp/llama-phase-aware/build/bin/llama-cli \
  --model /tmp/ggufs/q05.gguf --threads 4,8 --workloads decode_short,prefill_long \
  --l3-timeout 240 \
  --out results/dispatch-ledger-darwin-arm64-patched-flag-off.json --assert
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
   lldb sweep, for the same reason (this machine's shared contention).
3. The (c)/(d) patched sweep and the (a)/(b) baseline sweep were run ~40 minutes apart
   under measurably different shared-machine load (§5) — every comparison in this file
   that could plausibly be contention noise rather than a patch effect is called out
   explicitly above, rather than left implicit or omitted.
4. `results/dispatch-ledger-darwin-arm64.json` (the pre-existing, committed baseline
   ledger from this project's diagnosis phase) was deliberately **not** overwritten;
   this file's new dispatch evidence lives in the two new
   `dispatch-ledger-darwin-arm64-patched-flag-{on,off}.json` files instead.
5. No claim in this file has been folded into a single blended "speedup" number across
   phases, per the task brief's explicit instruction — prefill and decode are reported
   and judged separately throughout.
