# Re-measurement on a quiet machine — 2026-08-04

> **AUTHORITATIVE. This file supersedes the throughput numbers in `results/OPTIMIZATION.md`
> and `results/crossover/`.** Where they disagree, this file is correct. Any document in this
> repo still quoting a "+57.3% decode" figure for the phase-aware patch is **wrong** and must
> be corrected to match this page.

## Why this re-measurement exists

`results/crossover/` was collected while this machine's 1-minute load average was **66–147 on a
16-core host** — several unrelated concurrent agent sessions were competing for the same cores.
The harness itself flagged this in its own contention note, correctly, and warned that absolute
magnitudes might be suppressed.

They were, and worse: the baseline and patched configurations were measured **in different time
windows rather than interleaved against each other**, so they experienced *different* amounts of
external interference. That is exactly the failure mode that manufactures a fake speedup. It did:
the reported "patched default decode 45.5 → 71.6 tok/s, **+57.3%**" was an artifact of the
baseline window being more contended than the patched window, not of the patch doing anything good.

## Method

- Every configuration run **round-robin, one rep at a time** (A,B,C,…,A,B,C,…), so any drift in
  external load hits all configurations equally. This is the fix for the flaw above.
- `llama-bench -r 1 -o json`, 7 reps per configuration, median + population stdev + min + max.
- Model: `Qwen2.5-0.5B-Instruct-Q4_0` (`/tmp/ggufs/q05.gguf`), Apache-2.0.
- Baseline binary: `/tmp/llama.cpp/build/bin/llama-bench` @ `dbadb68`, `-DGGML_CPU_KLEIDIAI=ON`.
- Patched binary: `/tmp/llama-phase-aware/build/bin/llama-bench` @ `ef973b1` (same base + the patch).
- Host: Apple M4 Max, macOS 27, 16 cores. `llama.cpp`'s no-flag default is **12 threads**
  (`hw.perflevel0.physicalcpu`, the P-core count), not 16.
- External (non-benchmark) CPU load: **236% at start, 326% at end** — i.e. roughly **2.4–3.3 of 16
  cores** were busy with unrelated work throughout. Not a perfectly quiet machine, but ~40× lower
  contention than the original run, and, critically, **shared equally across all configurations**
  by the round-robin design.

## Results

| configuration | median tok/s | stdev | min | max | n |
|---|---:|---:|---:|---:|---:|
| decode — default (no flags, 12 threads) | 93.6 | 2.47 | 87.8 | 94.5 | 7 |
| **decode — `-t 2`** | **321.0** | 2.09 | 318.4 | 324.3 | 7 |
| decode — patched + `GGML_KLEIDIAI_PHASE_AWARE=1`, default | **82.5** | 4.07 | 79.0 | 92.0 | 7 |
| decode — patched + flag, `-t 2` | 317.5 | 3.58 | 313.9 | 324.1 | 7 |
| prefill — default (no flags) | 1230.3 | 118.52 | 1092.9 | 1440.6 | 7 |
| **prefill — `-t 8`** | **2198.1** | 72.59 | 2096.9 | 2335.9 | 7 |
| prefill — patched + flag, default | 1202.1 | 96.26 | 1088.7 | 1408.7 | 7 |

## Verdict 1 — the tuning win is REAL and replicates

| | default | tuned | ratio |
|---|---:|---:|---:|
| decode | 93.6 | 321.0 (`-t 2`) | **3.43×** |
| prefill | 1230.3 | 2198.1 (`-t 8`) | **1.79×** |

Decode stdevs are 2.47 and 2.09 on medians of 93.6 and 321.0 — the bands are nowhere near
overlapping. This is a large, clean, reproducible effect, obtainable **today with flags
`llama.cpp` already ships and zero code changes**.

Note this is *lower* than the previously published 4.4×, because the old figure divided by a
contention-suppressed baseline of 45.5 tok/s. The true baseline is ~93.6. **3.43× is the honest
number.**

## Verdict 2 — the phase-aware patch is a REGRESSION, not a win

| comparison | ratio | reading |
|---|---:|---|
| patched+flag vs baseline, at default threads | **0.88×** | 93.6 → 82.5 tok/s: **~12% SLOWER** |
| patched+flag vs baseline, at `-t 2` | 0.99× | 321.0 → 317.5: statistical tie (patch is inert here) |
| prefill, patched+flag vs baseline, at default | 0.98× | 1230.3 → 1202.1: tie, within noise (patch does not touch GEMM) |

93.6 ± 2.47 versus 82.5 ± 4.07 do not overlap. The regression is **real and outside noise**, not a
measurement wobble.

**Mechanism, stated honestly:** the patch works exactly as designed at the dispatch level — it
routes GEMV (decode) work into the existing SME+NEON hybrid split above `sme_thread_cap`, and
`tools/verify_dispatch.py` proves the change at the symbol level (decode@t=4: 0 → 3,072 SME2 hits;
decode@t=8: 0 → 2,354). That dispatch evidence is a symbol-level fact and remains valid — it is not
a timing measurement and is unaffected by contention.

But **dispatching SME2 is not the same as being faster.** At 12 threads the hybrid split gives SME
only 2 of them while the other 10 run NEON, and coordinating that split costs more than the SME
lane returns for a shape this small. Pure NEON on all 12 threads wins. The upstream code's
existing behaviour is, on this chip and this model, the better default — and our patch's premise
that decode was being unfairly excluded is **not supported by throughput**, even though the
exclusion itself is real.

## What this means for the upstream contribution

- The **warning** half of `patches/0001-kleidiai-phase-aware-dispatch.patch` stands on its own and
  should be proposed independently: a user at default thread count genuinely cannot tell that SME2
  is not being used, and telling them costs nothing.
- The **phase-aware dispatch** half should be reported to maintainers as a **measured negative
  result**, not offered as a performance improvement. It is useful information — it says the
  `ne11 < 128` exclusion is not leaving throughput on the table on this chip — but it is not a fix.
- The genuinely actionable user-facing finding is the **tuning** one: `-t 2` for decode is 3.43×
  the default on an M4 Max, and nothing in `llama.cpp` currently tells anyone that.

## Reproduce

```bash
# baseline vs tuned (the real optimization)
/tmp/llama.cpp/build/bin/llama-bench -m /tmp/ggufs/q05.gguf -p 0   -n 32 -r 7        # default
/tmp/llama.cpp/build/bin/llama-bench -m /tmp/ggufs/q05.gguf -p 0   -n 32 -r 7 -t 2   # tuned decode
/tmp/llama.cpp/build/bin/llama-bench -m /tmp/ggufs/q05.gguf -p 256 -n 0  -r 7        # default
/tmp/llama.cpp/build/bin/llama-bench -m /tmp/ggufs/q05.gguf -p 256 -n 0  -r 7 -t 8   # tuned prefill
```

Interleave the invocations rather than running all reps of one config back to back, and record
external CPU load before and after. That discipline is the entire reason this page exists.
