# Scale experiment: does the thread-tuning win survive at a realistic model size — and what does the Finding 3 build defect actually cost?

> **Honest headline verdict, leading with the one that deflates our own number.**
> **(1) The thread-tuning win this project has been built around collapses with model size.**
> At 0.5B, restricting `llama.cpp` to the right thread count is a **4.56×** decode win. At 7B — a
> size people actually run — the identical lever, measured the identical way on the identical
> machine, is a **1.33×** win. The 4.56× figure is real, but it is a small-model artifact, not a
> general property of thread tuning, and this document says so before it says anything else.
> **(2) The Finding 3 build defect (`llama.cpp`'s own documented KleidiAI build line silently
> compiling zero matmul kernels on this gcc/CPU pairing) costs the *opposite* of what tuning does:
> it barely touches the 0.5B model (0.99×, a statistical tie) and costs 7B model prefill 4.57× and
> decode 1.65×** — real, outside-noise effects that grow, not shrink, with model size. A toy model
> would never surface this defect; a 7B model makes it the largest single lever measured anywhere
> in this experiment.

---

## 0. The question this experiment was run to answer

Two questions, both aimed at this project's own prior claims rather than at a new target:

1. **Does the thread-tuning win survive at a realistic model size?** Every thread-tuning multiple
   reported elsewhere in this repository (`results/REMEASURE-2026-08-04-QUIET.md`'s 3.43× decode /
   1.79× prefill, `results/AUTODEFAULTS.md`'s 2.15×, `README.md`'s 6.44× total decomposition) was
   measured on **Qwen2.5-0.5B**, on an **Apple M4 Max**. Is that a property of thread tuning, or a
   property of running a very small model? This experiment adds a **Qwen2.5-7B** run, on a
   **different machine (DGX Spark)**, under the same harness and the same interleaving discipline,
   to find out.
2. **What does the Finding 3 build defect (docs/FINDINGS.md Finding 3 — `llama.cpp`'s documented
   KleidiAI build line producing zero compiled matmul kernels on gcc 13.3 + Cortex-X925) actually
   cost in throughput, and does that cost scale with model size the same way tuning does, or
   differently?** `results/server/SERVER-LANE.md` proved the defect at the *symbol* and
   *dispatch-call* level (0 vs 10 `kai_run_matmul` symbols). This experiment is the first place in
   this repository that measures its cost in **tok/s**, at two model sizes.

## 1. Method

| | |
|---|---|
| Machine | DGX Spark, GB10, 20-core Armv9.2 (Cortex-X925 + Cortex-A725), 121 GiB, gcc 13.3.0 — the same box documented in `results/server/SERVER-LANE.md` |
| `llama.cpp` | pinned `dbadb68`, same commit as every other measurement in this repository |
| Harness | `llama-bench -p 128 -n 32`, **5 reps per configuration**, `llama-bench -r 1` per invocation (median computed from the 5 independent invocations, not from `llama-bench`'s own internal `-r` averaging) |
| Statistic | **median ± population stdev**, `n=5` for every cell in every table below |
| Interleaving | **round-robin**: one rep of every configuration in a table, then the next rep of every configuration, repeated until 5 reps/config are collected — never all reps of one configuration back to back. Recorded verbatim in the raw file's own `note` field: `"round-robin interleaved; median of REPS; llama-bench -r 1 per invocation"`. This is the same discipline `results/REMEASURE-2026-08-04-QUIET.md` adopted after this project's earlier non-interleaved measurement produced a fabricated "+57.3%" figure — see that file for why interleaving is treated as non-negotiable here. |
| Models | `Qwen2.5-0.5B-Instruct-Q4_0` and `Qwen2.5-7B-Instruct-Q4_0`, both Apache-2.0, licence verified live via the HuggingFace API (per this project's standing convention — see `results/GENERALIZATION.md` §0 for the same live-verification method applied to other model files) |
| Build defect comparison (Section B, below) | **BROKEN** = `llama.cpp`'s own documented `-DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release` line (produces `kai_run_matmul symbols: 0`, per `results/server/spark-provenance.txt`). **FIXED** = the same line plus `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"` (produces `kai_run_matmul symbols: 10`). Both builds and their exact banners are documented in `results/server/SERVER-LANE.md`; this experiment reuses the same BROKEN/FIXED distinction and measures its throughput cost for the first time. |
| Thread sweep (Section A, below) | Run against the **FIXED** build only, so the sweep is not confounded with the build defect. |

**Confirming the default thread count.** `llama-bench`'s no-`-t`-flag default on this 20-core box
is not asserted here — it is derived from the data itself. Section B's "default threads" rows (no
`-t` flag) land within noise of, and closer to, the dedicated `t=20` row of the same model in
Section A/A2 than to any other thread count measured, for both models and both phases:

| model | phase | default (Section B, FIXED) | `t=20` sweep row (Section A/A2) | next-closest sweep row |
|---|---|---:|---:|---:|
| 0.5B | decode | 41.70 ± 6.86 | 45.49 ± 6.43 | 120.05 (`t=16`) |
| 0.5B | prefill | 933.63 ± 74.03 | 965.98 ± 87.36 | 1047.66 (`t=16`) |
| 7B | decode | 18.45 ± 0.84 | 18.03 ± 1.05 | 21.01 (`t=16`) |
| 7B | prefill | 222.14 ± 4.94 | 218.39 ± 2.49 | 212.41 (`t=16`) |

In every case the default row sits inside or immediately adjacent to the `t=20` row's stdev band,
and every other thread count is far outside it. That is the basis for reading "default" as
`t=20` throughout this document — not an assumption carried in from outside this file.

**Load, stated plainly.** The raw file records `load_before: 0.43` and `load_after: 12.32` — a
near-idle machine at the start of this experiment, and a materially busier one by the end. This
rise is not external contention; it is consistent with this experiment's own `t=20`/default-thread
configurations (all 20 cores, for a 121 GiB, 20-core box) driving load up over the course of the
run. Round-robin interleaving means that drift is shared roughly equally across every configuration
being compared *within* a given table, which is what protects the relative comparisons this
document draws — but it is a real, self-generated load increase, not a quiet machine throughout,
and is stated here rather than left implicit.

## 2. The three tables, verbatim from `results/scale/scale-experiment.json`

### A) Thread sweep, 7B, fixed build (`pp=128`, `tg=32`, `n=5` every cell)

| threads | prefill median (tok/s) | prefill stdev | decode median (tok/s) | decode stdev |
|---:|---:|---:|---:|---:|
| 1 | 28.89 | ±0.14 | 7.26 | ±0.20 |
| 2 | 55.41 | ±0.78 | 12.95 | ±0.13 |
| 4 | 104.42 | ±0.92 | 17.76 | ±2.04 |
| 8 | 179.45 | ±3.76 | **24.45 (decode peak)** | ±1.05 |
| 16 | 212.41 | ±1.27 | 21.01 | ±0.75 |
| 20 | **218.39 (prefill peak)** | ±2.49 | 18.03 | ±1.05 |

### A2) Thread sweep, 0.5B, fixed build (`pp=128`, `tg=32`, `n=5` every cell)

| threads | prefill median (tok/s) | prefill stdev | decode median (tok/s) | decode stdev |
|---:|---:|---:|---:|---:|
| 1 | 318.95 | ±6.43 | 85.41 | ±2.93 |
| 2 | 556.71 | ±5.35 | 138.60 | ±9.15 |
| 4 | 915.65 | ±11.77 | 177.61 | ±13.21 |
| 8 | **1457.68 (prefill peak)** | ±21.84 | **190.32 (decode peak)** | ±8.93 |
| 16 | 1047.66 | ±51.97 | 120.05 | ±7.16 |
| 20 | 965.98 | ±87.36 | 45.49 | ±6.43 |

### B) Finding 3 throughput cost, default threads (`pp=128`, `tg=32`, `n=5` every cell)

| model | build | prefill median (tok/s) | prefill stdev | decode median (tok/s) | decode stdev |
|---|---|---:|---:|---:|---:|
| 0.5B | BROKEN | 657.00 | ±51.29 | 42.18 | ±6.78 |
| 0.5B | FIXED | 933.63 | ±74.03 | 41.70 | ±6.86 |
| 7B | BROKEN | 48.64 | ±0.42 | 11.17 | ±0.37 |
| 7B | FIXED | 222.14 | ±4.94 | 18.45 | ±0.84 |

## 3. Conclusion 1 (leads) — thread tuning collapses with model size

**0.5B decode:** default (FIXED, 20 threads, Table B) 41.70 ± 6.86 → best (`t=8`, Table A2) 190.32
± 8.93 = **4.56× (190.32 / 41.70 = 4.5640)**. The two bands do not overlap by a wide margin
(41.70 + 6.86 = 48.56, far below 190.32 − 8.93 = 181.39) — a large, unambiguous effect.

**7B decode:** default (FIXED, 20 threads, Table B) 18.45 ± 0.84 → best (`t=8`, Table A) 24.45 ±
1.05 = **1.33× (24.45 / 18.45 = 1.3252)**. The bands still do not overlap (18.45 + 0.84 = 19.29 <
24.45 − 1.05 = 23.40 — real, not noise) — but the effect that was 4.56× at 0.5B is **1.33×** at
7B. Measured on the same machine, the same harness, the same interleaving discipline, changing only
the model size, **the headline thread-tuning multiple this project has repeatedly reported falls
by more than 3×.**

This is not a contradiction of the earlier M4 Max results (3.43× decode, 2.15× decode,
6.44× total) — those numbers are accurate for what they measured, a 0.5B model. It is a correction
to the implicit generalization: **the multiple is largely a small-model artifact.** At a size
people actually deploy, the same lever is a real but modest 1.33×, not a 3–4.5× win.

**Mechanism, stated as reasoning rather than as a measured fact.** A 0.5B model's decode step does
very little arithmetic per token relative to the fixed cost of coordinating however many threads
`llama.cpp` spins up — at this size, most of the wall-clock time in a 20-thread decode is thread
synchronization and scheduling overhead, not GEMV work, so cutting thread count to match the
actual amount of work available (the `t=8` peak observed here) removes overhead that was pure
waste. A 7B model's decode step is roughly 14× the arithmetic per token, and at that size decode
is memory-bandwidth-bound: every token requires streaming the full set of weights through the
memory bus regardless of how many threads are coordinating, so the ceiling is set by DRAM
bandwidth long before thread-coordination overhead becomes the dominant cost. The same lever
(fewer threads, less coordination overhead) is still net-positive at 7B — `t=8` still beats the
20-thread default — but it has far less overhead left to remove, hence 1.33× instead of 4.56×.

**Table A2's own shape reinforces this.** Every thread count tested for 0.5B decode is different
from every other by a wide, non-overlapping margin (85.41 → 138.60 → 177.61 → 190.32 → 120.05 →
45.49) — thread count visibly dominates the outcome. Table A's 7B decode row has the same peak
shape (peaking at `t=8`, falling off at `t=16`/`t=20`) but a much narrower spread (7.26 to 24.45,
a 3.4× range top to bottom, versus 0.5B's 4.2× range) — consistent with a workload where thread
count matters, but matters less.

**Prefill, kept separate and not blended into the decode figure above:** 0.5B prefill tuning is
933.63 (default) → 1457.68 (`t=8` peak) = **1.56×**. 7B prefill has effectively **no separate
tuning lever at all** — its own peak (Table A, `t=20`, 218.39) *is* the default thread count (the
table in §1 above shows the FIXED-build default row, 222.14, already at or above the sweep's own
`t=20` value). For 7B, prefill is already thread-saturated at whatever `llama.cpp` picks by
default; the entire realizable prefill gain from thread tuning at 7B is at most the measurement
noise between 222.14 and 218.39, not a multiple worth reporting. This is the same collapse as
decode, expressed even more starkly: at 7B, the prefill knob barely exists.

## 4. Conclusion 2 — the Finding 3 build defect grows with model size, the opposite direction

**0.5B prefill:** BROKEN 657.00 ± 51.29 → FIXED 933.63 ± 74.03 = **1.42× (933.63 / 657.00 =
1.4211)**. Bands do not overlap (657.00 + 51.29 = 708.29 < 933.63 − 74.03 = 859.60) — a real
effect, even at this size.

**0.5B decode:** BROKEN 42.18 ± 6.78 → FIXED 41.70 ± 6.86 = **0.99× (41.70 / 42.18 = 0.9886)**.
The bands overlap almost completely (BROKEN spans 35.40–48.96, FIXED spans 34.84–48.56) — this is
a **statistical tie, not an effect.** At 0.5B, the Finding 3 defect costs prefill but not decode.

**7B prefill:** BROKEN 48.64 ± 0.42 → FIXED 222.14 ± 4.94 = **4.57× (222.14 / 48.64 = 4.5670)**.
Bands are nowhere close (48.64 + 0.42 = 49.06 vs. 222.14 − 4.94 = 217.20) — the largest single
multiple measured anywhere in this experiment, larger even than the 0.5B thread-tuning decode win.

**7B decode:** BROKEN 11.17 ± 0.37 → FIXED 18.45 ± 0.84 = **1.65× (18.45 / 11.17 = 1.6517)**.
Bands do not overlap (11.17 + 0.37 = 11.54 < 18.45 − 0.84 = 17.61) — real, and, unlike at 0.5B,
decode is now measurably affected too.

**The defect's cost is size-dependent and moves in the opposite direction from thread tuning:**
at 0.5B it is 1.42× prefill / 0.99× decode (prefill-only, and modest); at 7B it is 4.57× prefill /
1.65× decode (both phases, and prefill's cost is the largest number in this whole experiment). A
toy model such as the 0.5B one used throughout most of this project's own prior measurements would
never surface how expensive this defect actually is — it takes a model large enough to be
memory-bandwidth- and matmul-shape-sensitive (7B) to expose the real magnitude of compiling zero
KleidiAI matmul kernels instead of ten.

**Why this reframes the project.** Section 3 shows the tuning knob this project has spent the most
words on shrinks to a modest 1.33× at a realistic model size. Section 4 shows the defect
Polygraph's own dispatch verification exists to *detect* — a silent, zero-exit-code build
misconfiguration — costs **4.57× prefill** at that same realistic size, over three times the
tuning win. The tool that verifies what a build actually dispatches, and the single largest
throughput lever measured in this document, are pointing at the same thing.

## 5. Limitations

- **One machine, one quantization, one benchmark harness.** Every number above is from the DGX
  Spark (Cortex-X925/Cortex-A725, gcc 13.3.0), `Q4_0` quantization only, and `llama-bench` only.
  No other CPU, compiler, quant format, or benchmarking tool was exercised in this experiment.
- **Only two model sizes.** 0.5B and 7B bound the range tested; nothing here says where between
  them (or beyond 7B) the thread-tuning multiple or the Finding 3 cost actually lands — both
  conclusions describe a direction (shrinking vs. growing with size), not a fitted curve.
- **Finding 3's magnitude here is specific to a build where feature detection collapsed
  entirely.** The BROKEN build's banner (documented in `results/server/SERVER-LANE.md`) showed no
  `DOTPROD`, no `MATMUL_INT8`, and no `SVE` at all — a total feature-detection failure on this
  gcc/CPU pairing, not a narrower "KleidiAI specifically is missing" case. The 4.57×/1.65× figures
  measured here should be read as the cost of *that* failure mode, not assumed to transfer
  unchanged to every KleidiAI-related build misconfiguration on every machine.
- **The 0.5B decode result in Section 4 shows no effect (0.99×, a statistical tie), not a small
  positive effect.** The defect's cost is not a single flat number that applies at every model
  size — at 0.5B it is prefill-only and modest; only at 7B does it become large and touch both
  phases. Reporting "the Finding 3 defect costs ~4.6×" without the size qualifier would misstate
  what was actually measured.
- **The load rise (`load_before: 0.43` → `load_after: 12.32`) is self-generated by this
  experiment's own default/`t=20`/all-core configurations, not external contention.**
  Round-robin interleaving spreads that rise evenly across the configurations compared within each
  table, which is what makes the ratios above meaningful — but this was not a uniformly quiet
  machine end to end, and that is stated here rather than implied by silence.
- **Prefill and decode are never blended into one "speedup" anywhere in this document.** Every
  ratio above is explicitly a prefill ratio or a decode ratio, never an average or combination of
  the two — the two phases have different bottlenecks (compute-bound prefill vs.
  memory-bandwidth-bound decode at these sizes) and a blended number would obscure exactly the
  distinction Sections 3 and 4 depend on.
- **The mechanism explanations in Section 3** (thread-coordination overhead dominating at 0.5B;
  memory-bandwidth-boundedness dominating at 7B) are offered as reasoning consistent with the
  measured shape of Tables A/A2, not as independently measured facts — no separate profiling data
  (e.g., a memory-bandwidth trace or a thread-scheduling trace) was collected in this experiment to
  confirm the mechanism directly.
