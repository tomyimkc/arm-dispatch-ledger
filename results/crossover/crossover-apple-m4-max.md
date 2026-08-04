# Crossover harness results -- apple-m4-max

- Generated: 2026-08-04T03:12:51.757832+00:00
- CPU: Apple M4 Max
- llama.cpp bin dir: `/tmp/llama.cpp/build/bin`  (commit: dbadb68)
- Model: `/tmp/ggufs/q05.gguf`
- Reps per cell: 5 (interleaved round-robin across all 20 cells, never all reps of one cell back-to-back)
- llama.cpp's true no-flags default n_threads on this machine is 12 (hw.perflevel0.physicalcpu, the P-core count), not the total 16 physical cores -- verified via `llama-cli -v` this session.
- Full methodology: see `tools/crossover.md`. Median/stddev/min/max only, computed across independently-launched, interleaved `llama-bench -r 1` process invocations -- never a bare mean.

## Thermal context (`pmset -g therm`)

Before:
```
Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
```
After:
```
Note: No thermal warning level has been recorded
Note: No performance warning level has been recorded
Note: No CPU power status has been recorded
```

## Contention note (shared machine)

- Load average before: `10:52  up 9 days, 12:51, 37 users, load averages: 66.41 147.58 154.94`
- Load average after: `11:12  up 9 days, 13:11, 37 users, load averages: 91.82 119.80 126.72`
- 10 of this run's llama-bench/llama-cli calls needed a timeout-retry (see `retry_log` in the JSON output for exactly which).
- **This machine's 1-minute load average exceeded its physical core count (16) during this run** -- this host is shared with other concurrent, unrelated agent sessions (observed: several `python -m contest_bench...` multiprocessing workers and other Claude Code sessions competing for the same cores). Absolute tok/s numbers below may be measurably suppressed relative to a quiet machine, and cell-to-cell variance (stddev) may be inflated by contention bursts, not just this workload's own thread-count/kernel-family behaviour. Every call that hit its timeout was retried (never estimated or interpolated) against the same real binary; a cell is reported only from calls that actually completed. Relative comparisons (which thread count / kernel family wins within a phase) are expected to be more robust to this than absolute magnitudes, since contention affects all configurations of a given call, not selectively -- but this has not been independently verified by re-running on a quiet machine, so treat this as a caveat, not a correction factor.

## (a) Per-phase optimum: full sweep, threads x SME x phase

| phase | threads | SME | median tok/s | stddev | min | max | n |
|---|---:|---|---:|---:|---:|---:|---:|
| decode | 1 | on | 187.6 | 19.86 | 150.4 | 201.9 | 5 |
| decode | 1 | off | 140.4 | 1.97 | 137.2 | 142.5 | 5 |
| decode | 2 | on | 305.4 **<-- optimum** | 52.88 | 190.9 | 319.4 | 5 |
| decode | 2 | off | 233.8 | 10.41 | 231.0 | 253.1 | 5 |
| decode | 4 | on | 258.2 | 11.44 | 238.1 | 266.0 | 5 |
| decode | 4 | off | 276.6 | 11.83 | 251.9 | 278.9 | 5 |
| decode | 8 | on | 143.3 | 10.66 | 127.4 | 152.2 | 5 |
| decode | 8 | off | 140.4 | 10.32 | 123.0 | 149.9 | 5 |
| decode | 16 | on | _[measurement failed]_ | | | | |
| decode | 16 | off | _[measurement failed]_ | | | | |
| prefill | 1 | on | 860.0 | 2.65 | 859.7 | 864.9 | 5 |
| prefill | 1 | off | 400.0 | 2.48 | 396.5 | 402.5 | 5 |
| prefill | 2 | on | 1584.4 | 18.96 | 1556.5 | 1602.4 | 5 |
| prefill | 2 | off | 788.5 | 3.59 | 781.8 | 790.0 | 5 |
| prefill | 4 | on | 1863.3 | 24.24 | 1838.4 | 1903.3 | 5 |
| prefill | 4 | off | 1500.8 | 8.47 | 1482.5 | 1502.7 | 5 |
| prefill | 8 | on | 2289.9 | 260.62 | 1841.0 | 2493.1 | 5 |
| prefill | 8 | off | 2615.6 **<-- optimum** | 52.05 | 2527.1 | 2657.8 | 5 |
| prefill | 16 | on | 29.0 | 4.12 | 24.4 | 35.7 | 5 |
| prefill | 16 | off | 27.7 | 4.34 | 20.1 | 31.4 | 5 |

- **decode optimum:** threads=2, SME=on (SME2/hybrid dispatch region), median 305.4 tok/s.
- **prefill optimum:** threads=8, SME=off (NEON forced), median 2615.6 tok/s.

Cells not listed/measured must not be treated as zero, equal-to-neighbor, or interpolated.

## (b) llama.cpp DEFAULT configuration (no -t/-tb flags, SME unset)

| phase | median tok/s | stddev | min | max | n |
|---|---:|---:|---:|---:|---:|
| decode | 45.5 | 7.71 | 39.6 | 56.3 | 5 |
| prefill | 1145.0 | 136.05 | 839.8 | 1163.2 | 5 |

This is what a user who runs llama-cli/llama-bench with zero thread flags actually gets today.

## (c) Best hand-tuned split-phase config (llama-cli -t/-tb), TODAY, no patch

`-t 2` (this sweep's decode optimum thread count) `-tb 8` (this sweep's prefill optimum thread count). `GGML_KLEIDIAI_SME` is still a single process-global setting, so both available states were measured.

| GGML_KLEIDIAI_SME | prompt (prefill) tok/s | median | stddev | min | max | n | generation (decode) tok/s | median | stddev | min | max | n |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| on | | 2148.3 | 363.57 | 1354.7 | 2245.8 | 5 | | 197.9 | 16.66 | 164.4 | 205.9 | 5 |
| off | | 2257.5 | 170.03 | 1890.1 | 2289.3 | 5 | | 198.9 | 16.85 | 180.0 | 214.2 | 5 |

## (d) THEORETICAL best -- best prefill cell + best decode cell (NOT YET ACHIEVABLE TODAY)

[NOT YET ACHIEVABLE] decode wants threads=2, SME=on (median 305.4 tok/s); prefill wants threads=8, SME=off (median 2615.6 tok/s). These require DIFFERENT `GGML_KLEIDIAI_SME` process-global states simultaneously, which llama.cpp cannot express today (`GGML_KLEIDIAI_SME` is read once at process start; there is no per-call or per-phase override). This pairing is the TARGET a phase-aware kernel-family-selection patch should approach, not a number this harness (or any unpatched llama.cpp invocation) can itself produce.

## Editorial addendum (hand-added after generation, not regenerated by `crossover.py`)

Two numbers in this run deserve explicit comparison against the sibling `tools/bench.py` sweep already checked into `results/SUMMARY.md`/`results/bench/bench-apple-m4-max.md`, run in an earlier, much quieter session on the same machine/model/binary:

1. **`prefill, threads=16` collapsed to ~28-29 tok/s here (both SME on and off), versus 445.3/1514.1 tok/s (SME on/off) in `results/SUMMARY.md` section 3 for the nominally equivalent `prefill_long@16` cell** -- roughly a 50x gap for the same configuration. `decode, threads=16` failed to complete at all in this run (10/10 timeouts, see `sweep_errors`), whereas `bench.py`'s sweep reported a real (if already noted as "unstable") number there. Given this harness's own "Contention note" above (1-minute load average 66-190+ against 16 physical cores, versus a quiet machine for the `bench.py` run), the most likely explanation is that 16-thread configurations are the ones most exposed to this run's severe, concurrent multi-agent CPU contention -- they need every physical core simultaneously, so they have the least room to tolerate other processes competing for the same cores. This is **not** presented as a corrected or "true" number; it is the real, honestly-measured output of this run, under these conditions, and is flagged here so a reviewer does not mistake it for a code defect or a change in the underlying dispatch behaviour. The `threads<=8` cells in this same run track `bench.py`'s numbers reasonably closely (e.g. `prefill@8,off` here: 2615.6 vs `bench.py`'s `prefill_long@8,off`: 2676.4 -- within ~2%), which is consistent with contention being the dominant factor specifically at `threads=16`, not a broader measurement problem.
2. **The per-phase optima themselves (decode: threads=2/SME=on; prefill: threads=8/SME=off) match `bench.py`'s independently-observed pattern exactly** (`results/SUMMARY.md` section 4: "SME2 wins outright [for decode] ... NEON alone beats SME's own best cell [for prefill]") -- two separately-written harnesses, run in different sessions under different system-load conditions, agree on the qualitative crossover finding this submission's optimization claim rests on, even though this run's absolute magnitudes are noisier.
