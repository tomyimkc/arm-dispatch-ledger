# Bench results -- apple-m4-max

- Generated: 2026-08-03T22:30:54.663025+00:00
- CPU: Apple M4 Max
- llama.cpp bin dir: `/tmp/llama.cpp/build/bin`
- Reps per cell: 5
- Full methodology: see `tools/protocol.md`. Median/stddev/min/max are computed across independently-warmed-up, interleaved process invocations -- never a bare mean.

## Thermal context

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

## Results

| phase | quant | threads | SME | median tok/s | stddev | min | max | n | dispatch |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| decode | Q4_0 | 1 | on | 208.9 | 2.87 | 203.5 | 210.2 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q4_0 | 1 | off | 149.9 | 4.30 | 146.1 | 157.8 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q4_0 | 2 | on | 327.6 | 4.56 | 318.2 | 329.9 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q4_0 | 2 | off | 266.4 | 6.44 | 255.2 | 272.9 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q4_0 | 8 | on | 154.6 | 1.79 | 153.0 | 156.7 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q4_0 | 8 | off | 155.3 | 0.83 | 154.0 | 155.9 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q4_0 | 16 | on | 34.9 | 6.62 | 29.8 | 44.6 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q4_0 | 16 | off | 28.7 | 7.85 | 21.9 | 42.9 | 5 | unverified (--skip-dispatch-verify passed) |
| decode | Q8_0 | 1 | on | _[not available]_ | | | | | model file not present |
| decode | Q8_0 | 1 | off | _[not available]_ | | | | | model file not present |
| decode | Q8_0 | 2 | on | _[not available]_ | | | | | model file not present |
| decode | Q8_0 | 2 | off | _[not available]_ | | | | | model file not present |
| decode | Q8_0 | 8 | on | _[not available]_ | | | | | model file not present |
| decode | Q8_0 | 8 | off | _[not available]_ | | | | | model file not present |
| decode | Q8_0 | 16 | on | _[not available]_ | | | | | model file not present |
| decode | Q8_0 | 16 | off | _[not available]_ | | | | | model file not present |
| prefill_short | Q4_0 | 1 | on | 903.3 | 5.58 | 892.2 | 906.6 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q4_0 | 1 | off | 422.0 | 8.02 | 420.4 | 438.8 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q4_0 | 2 | on | 1579.0 | 8.45 | 1565.5 | 1585.0 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q4_0 | 2 | off | 806.4 | 17.31 | 803.6 | 844.2 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q4_0 | 8 | on | 2136.0 | 91.87 | 1998.2 | 2227.9 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q4_0 | 8 | off | 2188.4 | 60.99 | 2092.7 | 2262.8 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q4_0 | 16 | on | 888.9 | 185.13 | 724.9 | 1175.9 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q4_0 | 16 | off | 996.7 | 366.74 | 366.4 | 1147.2 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_short | Q8_0 | 1 | on | _[not available]_ | | | | | model file not present |
| prefill_short | Q8_0 | 1 | off | _[not available]_ | | | | | model file not present |
| prefill_short | Q8_0 | 2 | on | _[not available]_ | | | | | model file not present |
| prefill_short | Q8_0 | 2 | off | _[not available]_ | | | | | model file not present |
| prefill_short | Q8_0 | 8 | on | _[not available]_ | | | | | model file not present |
| prefill_short | Q8_0 | 8 | off | _[not available]_ | | | | | model file not present |
| prefill_short | Q8_0 | 16 | on | _[not available]_ | | | | | model file not present |
| prefill_short | Q8_0 | 16 | off | _[not available]_ | | | | | model file not present |
| prefill_long | Q4_0 | 1 | on | 896.2 | 4.66 | 889.2 | 901.8 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q4_0 | 1 | off | 415.1 | 5.95 | 412.4 | 427.0 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q4_0 | 2 | on | 1629.1 | 8.60 | 1616.7 | 1640.0 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q4_0 | 2 | off | 805.2 | 13.14 | 803.3 | 834.8 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q4_0 | 8 | on | 1830.1 | 203.51 | 1762.1 | 2270.8 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q4_0 | 8 | off | 2676.4 | 30.63 | 2661.1 | 2727.6 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q4_0 | 16 | on | 445.3 | 100.45 | 255.5 | 526.4 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q4_0 | 16 | off | 1514.1 | 198.85 | 1297.4 | 1733.1 | 5 | unverified (--skip-dispatch-verify passed) |
| prefill_long | Q8_0 | 1 | on | _[not available]_ | | | | | model file not present |
| prefill_long | Q8_0 | 1 | off | _[not available]_ | | | | | model file not present |
| prefill_long | Q8_0 | 2 | on | _[not available]_ | | | | | model file not present |
| prefill_long | Q8_0 | 2 | off | _[not available]_ | | | | | model file not present |
| prefill_long | Q8_0 | 8 | on | _[not available]_ | | | | | model file not present |
| prefill_long | Q8_0 | 8 | off | _[not available]_ | | | | | model file not present |
| prefill_long | Q8_0 | 16 | on | _[not available]_ | | | | | model file not present |
| prefill_long | Q8_0 | 16 | off | _[not available]_ | | | | | model file not present |

Threads/quants not listed above were not measured in this run and must not be treated as zero, equal-to-neighbor, or interpolated -- see `tools/protocol.md` section 7.
