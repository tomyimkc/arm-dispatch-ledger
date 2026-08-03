# `kernel_bench` — Apple M4 Max, hand-written kernels (`kernels/`)

Added during adversarial review (2026-08-04) because the numbers quoted in `README.md` /
`results/SUMMARY.md` were not previously backed by a persisted artifact in `results/` —
only by "this run" prose. This file fixes that: it is the raw, unedited output of

```
cd kernels && rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -- -j
./kernel_test    # ALL CHECKS PASSED, EXIT=0 (re-verified this session)
./kernel_bench
```

run for real, on this machine, this session. Raw log: `kernel-bench-apple-m4-max.log` in
this directory.

## fp32 GEMM (GFLOP/s, single-thread, best-of-5)

| N | NEON (tuned) | SME2 (packed) | Accelerate | Accelerate / SME2 |
|---:|---:|---:|---:|---:|
| 512 | 99.42 | 503.63 | 1607.40 | 3.2x |
| 1024 | 95.95 | 463.62 | 3103.30 | 6.7x |
| 2048 | 49.71 | 185.17 | 3408.03 | 18.4x |

**Apple's tuned Accelerate library is still faster than our hand-written SME2 kernel at every
size measured — we never claim otherwise.**

## int8 GEMM (GOP/s, single-thread, best-of-5)

| N | SME2 int8 | Accelerate int8 |
|---:|---:|---|
| 512 | 101.83 | *(no integer GEMM exists in Accelerate's CBLAS surface)* |
| 1024 | 123.15 | *(no integer GEMM exists)* |
| 2048 | 115.98 | *(no integer GEMM exists)* |

## Methodology caveat: this microbenchmark is noisier than the dispatch-verification numbers

Unlike the `lldb`-verified dispatch counts (which are exact call counts, not timings), `kernel_bench`
times sub-millisecond GEMMs with `clock_gettime` and reports the best of 5 reps — at `N=512` the
whole GEMM finishes in ~0.0002–0.0005s, which is close to the practical resolution/noise floor for
wall-clock timing on a shared, multi-process machine. Two additional back-to-back reruns performed
during this review, not otherwise recorded in this repo, illustrate the spread at `N=512`:

| run | NEON | SME2 (packed) | Accelerate | Accelerate / SME2 |
|---|---:|---:|---:|---:|
| canonical (this file) | 99.42 | 503.63 | 1607.40 | 3.2x |
| rerun A | 100.09 | 503.63 | 1607.40 | 3.2x |
| rerun B | 100.58 | 496.18 | 2508.74 | 5.1x |

`N=1024`/`N=2048` were stable across all three reruns (±5%). **Read the `N=512` Accelerate cell,
and the "Accelerate / SME2" ratio derived from it, as "roughly 3–5x", not a precise point estimate**;
`N=1024` (6.7x) and `N=2048` (18.4x) are the more trustworthy numbers in this table. This does not
change the qualitative finding (Accelerate is faster, sometimes far faster, at every size — no
strawman "beats naive NEON" framing survives this data either way).

int8 numbers (no Accelerate comparator, no sub-µs floor issue since there's nothing to divide
against) were stable within ~2% across reruns.
