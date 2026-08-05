<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Polygraph contributors
-->
# PR #26076 (ggml-org/llama.cpp) — before/after dispatch verification on Apple M4 Max

**Date:** 2026-08-06 (measurement session; ledger `generated_at_utc` stamps are in the JSON).
**Machine:** Apple M4 Max (the Arm test laptop; `hw.nperflevels = 2`, perflevel0 "Performance"
`physicalcpu = 12` / `cpusperl2 = 6`, perflevel1 "Efficiency" `physicalcpu = 4`).
**OS:** macOS 27.0 (build 26A5378n, pre-release). **Compiler:** AppleClang 21.0.0.21000323.
**Debugger:** lldb-2103.0.25.1.

## What was tested

[ggml-org/llama.cpp#26076](https://github.com/ggml-org/llama.cpp/pull/26076) replaces the
hardcoded `ModelSMCU` brand-string table (including `{ "M4 Max", 2 }`) with a runtime
derivation from `hw.nperflevels` / `hw.perflevelN.physicalcpu` / `hw.perflevelN.cpusperl2`.
Question: does the derived SME thread cap reproduce the removed hand-tuned value *at the
kernel-execution level*, not just arithmetically?

- **PR head:** `7d8324875075f29cb0fc98e236f52092e48acf00` (branch
  `jonclo01/aarch64_runtime_feature_detection`, fetched via `refs/pull/26076/head`)
- **Baseline:** merge-base with `origin/master`, `a035a88878ad4d48c1e1b41cf83b0c11aea64bdb`
- Both built: `cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF`
  then `cmake --build build --target llama-cli -j 10`
- Model: `qwen2.5-0.5b-instruct-q4_0.gguf`, sha256
  `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed` (matches
  `scripts/models.txt`; re-downloaded and re-hashed this session because the cached copy on
  disk had a different hash and was discarded)
- Probe: `tools/verify_dispatch.py --threads 1,2,4,8,16 --workloads all --quant Q4_0`
  (non-halting lldb breakpoints on `^kai_run_matmul`)

## Results

Ledgers (committed verbatim in this directory):

- `dispatch-base-a035a8887-m4max-q4_0.json`
- `dispatch-pr-7d8324875-m4max-q4_0.json`

Verdicts are identical in all 10 (threads × workload) cells. Hits below are
`advertised-family (SME2) / other-family` counts from L3.

| threads | workload | merge-base | PR head | verdict (both) |
|---:|---|---|---|---|
| 1 | decode_short | 1014 / 0 | 1014 / 0 | SME2_DISPATCHED |
| 2 | decode_short | 5952 / 0 | 5952 / 0 | SME2_DISPATCHED |
| 4 | decode_short | 0 / 16224 | 0 / 16224 | SILENT_FALLBACK (dotprod) |
| 8 | decode_short | 0 / 32447 | 0 / 32445 | SILENT_FALLBACK (dotprod) |
| 16 | decode_short | 0 / 52221 | 0 / 52221 | SILENT_FALLBACK (dotprod) |
| 1 | prefill_long | 672 / 0 | 672 / 0 | SME2_DISPATCHED |
| 2 | prefill_long | 3937 / 0 | 3937 / 0 | SME2_DISPATCHED |
| 4 | prefill_long | 2457 / 6667 | 2356 / 6768 | SME2_HYBRID_DISPATCH |
| 8 | prefill_long | 1925 / 13591 | 1566 / 13949 | SME2_HYBRID_DISPATCH |
| 16 | prefill_long | 1419 / 21913 | 1447 / 21884 | SME2_HYBRID_DISPATCH |

L1 (static): both builds carry 264 `kai_*` symbols, 17 `kai_run_matmul_*` entry points,
same family split (sme2 6, sme 3, dotprod 6, i8mm 2). L2 (selection log): both builds
advertise SME2 for q4/q8/f32 at every thread count.

## Reading

- The PR's runtime derivation yields an effective SME2 thread cap of **2** on M4 Max —
  SME2 kernels execute at 1–2 threads and never for decode at ≥4 — matching the
  `{ "M4 Max", 2 }` entry the PR deletes. **No dispatch regression observed.**
- Pure (single-family) cells reproduce exactly across builds except 8-thread decode
  (32447 vs 32445, a 2-hit difference out of ~32k). Hybrid prefill totals match within
  1 hit (9124/9124, 15516/15515, 23332/23331); the SME2-vs-dotprod *split* inside a
  hybrid cell varies with scheduling, while the totals differ by at most 1 hit — the
  same run-to-run noise scale as the 8-thread decode cell, not a dispatch difference.

## Limits

One machine, one OS (a pre-release macOS build), one model/quant, one compiler, the five
thread counts listed. Hit counts scale with workload length; only same-config comparisons
are meaningful. This measures dispatch equivalence only — no timing was recorded, and no
throughput claim is made.
