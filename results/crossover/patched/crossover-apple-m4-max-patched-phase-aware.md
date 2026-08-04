# Crossover harness results -- apple-m4-max-patched-phase-aware

> **SUPERSEDED — read before trusting any cross-run comparison from this file.**
> These numbers were collected under heavy external load (1-minute load average up to
> ~100+ on this 16-core host — see "Contention note" below), and this file's patched
> sweep and the sibling baseline sweep (`results/crossover/crossover-apple-m4-max.md`)
> were **not interleaved against each other** -- they were run as two separate
> invocations roughly 40 minutes apart, so any comparison drawn *across* the two files
> (patched vs. baseline) is invalid and must not be cited. That non-interleaved,
> different-contention-window comparison is exactly what produced the retracted
> "decode +57.3%" figure previously reported in `results/OPTIMIZATION.md`. **Superseded
> by [`results/REMEASURE-2026-08-04-QUIET.md`](../../REMEASURE-2026-08-04-QUIET.md)**,
> which re-measured the same comparison round-robin-interleaved on a quieter host and
> found the phase-aware patch to be a ~12% *regression* at default thread count, not a
> win. The **within-run** relative ordering in the tables below (which thread count / SME
> state wins within *this one file's own* sweep) is still informative and was not itself
> contradicted by the re-measurement. Nothing below has been edited or deleted -- this is
> the honest raw record of a contended run, kept as evidence, not as a current claim.

- Generated: 2026-08-04T03:54:49.357269+00:00
- CPU: Apple M4 Max
- llama.cpp bin dir: `/tmp/llama-phase-aware/build/bin`  (commit: ef973b1)
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

- Load average before: `11:19  up 9 days, 13:17, 37 users, load averages: 5.07 34.69 80.72`
- Load average after: `11:54  up 9 days, 13:53, 37 users, load averages: 69.53 106.94 112.26`
- 20 of this run's llama-bench/llama-cli calls needed a timeout-retry (see `retry_log` in the JSON output for exactly which).
- **This machine's 1-minute load average exceeded its physical core count (16) during this run** -- this host is shared with other concurrent, unrelated agent sessions (observed: several `python -m contest_bench...` multiprocessing workers and other Claude Code sessions competing for the same cores). Absolute tok/s numbers below may be measurably suppressed relative to a quiet machine, and cell-to-cell variance (stddev) may be inflated by contention bursts, not just this workload's own thread-count/kernel-family behaviour. Every call that hit its timeout was retried (never estimated or interpolated) against the same real binary; a cell is reported only from calls that actually completed. Relative comparisons (which thread count / kernel family wins within a phase) are expected to be more robust to this than absolute magnitudes, since contention affects all configurations of a given call, not selectively -- but this has not been independently verified by re-running on a quiet machine, so treat this as a caveat, not a correction factor.

## (a) Per-phase optimum: full sweep, threads x SME x phase

| phase | threads | SME | median tok/s | stddev | min | max | n |
|---|---:|---|---:|---:|---:|---:|---:|
| decode | 1 | on | 197.4 | 6.38 | 187.2 | 203.6 | 5 |
| decode | 1 | off | 142.1 | 1.36 | 141.7 | 144.9 | 5 |
| decode | 2 | on | 305.0 **<-- optimum** | 8.01 | 302.7 | 320.0 | 5 |
| decode | 2 | off | 244.2 | 7.67 | 237.8 | 255.9 | 5 |
| decode | 4 | on | 246.7 | 6.66 | 236.2 | 250.2 | 5 |
| decode | 4 | off | 278.5 | 6.70 | 271.3 | 288.6 | 5 |
| decode | 8 | on | 149.2 | 10.94 | 142.3 | 170.3 | 5 |
| decode | 8 | off | 151.8 | 4.33 | 146.2 | 158.3 | 5 |
| decode | 16 | on | _[measurement failed]_ | | | | |
| decode | 16 | off | _[measurement failed]_ | | | | |
| prefill | 1 | on | 862.8 | 7.71 | 858.0 | 875.5 | 5 |
| prefill | 1 | off | 400.8 | 2.83 | 396.5 | 403.7 | 5 |
| prefill | 2 | on | 1576.0 | 21.32 | 1548.7 | 1602.6 | 5 |
| prefill | 2 | off | 788.0 | 6.21 | 775.3 | 790.6 | 5 |
| prefill | 4 | on | 1861.7 | 29.43 | 1832.6 | 1900.8 | 5 |
| prefill | 4 | off | 1493.3 | 7.02 | 1486.3 | 1504.7 | 5 |
| prefill | 8 | on | 2324.3 | 199.96 | 1972.6 | 2491.4 | 5 |
| prefill | 8 | off | 2533.0 **<-- optimum** | 152.61 | 2293.9 | 2676.7 | 5 |
| prefill | 16 | on | 32.4 | 7.02 | 24.5 | 42.1 | 5 |
| prefill | 16 | off | 34.1 | 52.53 | 22.8 | 147.2 | 5 |

- **decode optimum:** threads=2, SME=on (SME2/hybrid dispatch region), median 305.0 tok/s.
- **prefill optimum:** threads=8, SME=off (NEON forced), median 2533.0 tok/s.

Cells not listed/measured must not be treated as zero, equal-to-neighbor, or interpolated.

## (b) llama.cpp DEFAULT configuration (no -t/-tb flags, SME unset)

| phase | median tok/s | stddev | min | max | n |
|---|---:|---:|---:|---:|---:|
| decode | 71.6 | 11.18 | 59.3 | 87.6 | 5 |
| prefill | 1328.4 | 267.51 | 917.6 | 1607.8 | 5 |

This is what a user who runs llama-cli/llama-bench with zero thread flags actually gets today.

## (c) Best hand-tuned split-phase config (llama-cli -t/-tb), TODAY, no patch

`-t 2` (this sweep's decode optimum thread count) `-tb 8` (this sweep's prefill optimum thread count). `GGML_KLEIDIAI_SME` is still a single process-global setting, so both available states were measured.

| GGML_KLEIDIAI_SME | prompt (prefill) tok/s | median | stddev | min | max | n | generation (decode) tok/s | median | stddev | min | max | n |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| on | | 2309.1 | 216.21 | 1869.4 | 2400.9 | 5 | | 201.8 | 5.82 | 199.2 | 213.5 | 5 |
| off | | 2552.9 | 31.15 | 2493.2 | 2573.4 | 5 | | 214.7 | 4.83 | 211.4 | 223.4 | 5 |

## (d) THEORETICAL best -- best prefill cell + best decode cell (NOT YET ACHIEVABLE TODAY)

[NOT YET ACHIEVABLE] decode wants threads=2, SME=on (median 305.0 tok/s); prefill wants threads=8, SME=off (median 2533.0 tok/s). These require DIFFERENT `GGML_KLEIDIAI_SME` process-global states simultaneously, which llama.cpp cannot express today (`GGML_KLEIDIAI_SME` is read once at process start; there is no per-call or per-phase override). This pairing is the TARGET a phase-aware kernel-family-selection patch should approach, not a number this harness (or any unpatched llama.cpp invocation) can itself produce.
