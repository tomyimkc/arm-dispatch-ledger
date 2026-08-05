# DRAFT — comment for ggml-org/llama.cpp#26076 (do not post without owner approval)

Tested this PR's Apple perf-level SMCU derivation on an M4 Max — not among the devices listed in the PR description — at the sysctl level and at the kernel-execution level, against the merge-base.

**Derivation.** On this machine `hw.nperflevels = 2`; `hw.perflevel0` (Performance) reports `physicalcpu = 12`, `cpusperl2 = 6`, so `(12 + 6 - 1) / 6` counts 2 units; `hw.perflevel1` (Efficiency) is skipped. Total = 2, matching the `{ "M4 Max", 2 }` entry this PR removes — and a different core-count/`cpusperl2` grouping on the Apple positive path than the M4 Pro results in the description.

**Execution.** Built PR head `7d83248` and merge-base `a035a8887` identically (`-DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF`, AppleClang 21.0.0.21000323), and counted `kai_run_matmul_*` entries with non-halting lldb breakpoints running Qwen2.5-0.5B-Instruct Q4_0 at 1/2/4/8/16 threads, decode and prefill workloads:

| threads | decode, merge-base | decode, PR head |
|---:|---|---|
| 1 | 1014 SME2 / 0 other | 1014 / 0 |
| 2 | 5952 SME2 / 0 other | 5952 / 0 |
| 4 | 0 SME2 / 16224 dotprod | 0 / 16224 |
| 8 | 0 / 32447 | 0 / 32445 |
| 16 | 0 / 52221 | 0 / 52221 |

Prefill shows the same equivalence: SME2-only at 1–2 threads (672 and 3937 hits, identical on both builds); the hybrid SME2+dotprod path at 4/8/16 threads, totals within 1 hit (9124 vs 9124, 15516 vs 15515, 23332 vs 23331). The SME2/dotprod split within a hybrid run varies with scheduling; the ≤1-hit total deltas match the run-to-run noise at 8-thread decode (32447 vs 32445), not a dispatch difference. Compiled kernel symbols are identical in both builds (264 `kai_*`, 17 of them `kai_run_matmul_*`).

So on this hardware the runtime derivation reproduces the removed hand-tuned value exactly: the effective SME2 thread cap is still 2, and the SME2/dotprod dispatch outcome is unchanged in all 10 configurations. The automated review earlier in the thread asked for concrete on-hardware data for the new detection paths — treat this as one data point for the Apple branch.

Limits: one machine (M4 Max), one OS (macOS 27.0 pre-release), one model/quant, the thread counts listed. Hit counts scale with workload length, so only same-configuration comparisons are meaningful. Raw JSON and reproduction commands: [results/upstream/pr26076](https://github.com/tomyimkc/polygraph/tree/main/results/upstream/pr26076) in polygraph, which produced the counts.
