---
title: "The silicon wasn't the problem: a silent zero-kernel build on Cortex-X925"
date: 2026-08-05
tags: [arm, kleidiai, llama.cpp, sve2, i8mm, dgx-spark]
---

# The silicon wasn't the problem: a silent zero-kernel build on Cortex-X925

Short version, for anyone evaluating Arm server CPUs for LLM inference: the hardware is
fine. A DGX Spark's Cortex-X925 — 20 cores, Armv9.2, SVE2, i8mm, bf16, no GPU involved —
sustains **440.4 tok/s aggregate** across 16 concurrent `llama-server` clients once it's
built correctly. What we found is not a silicon gap. It's one CMake line straight out of
`llama.cpp`'s own docs that ships zero accelerated matmul kernels while printing a
banner that reads like success.

This is [Polygraph](https://github.com/tomyimkc/polygraph)'s third finding: we attach a
debugger to real kernel entry points and count actual calls, because timing alone can't
tell you whether an accelerated code path ran at all. Everything below is quoted from
files committed in `results/server/`.

## What we ran

`llama.cpp`'s own documented KleidiAI build command, no unusual flags, gcc 13.3.0,
aarch64:

```
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
```

It compiles cleanly, exit 0. `nm` on the resulting binary shows 36 `kai_` symbols — all
packing helpers — and **0** `kai_run_matmul` symbols: zero matmul micro-kernels of any
family. At runtime:

```
system_info: ... | NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: no compatible q4 kernels found for CPU features mask 0
kleidiai: no compatible q8 kernels found for CPU features mask 0
```

`KLEIDIAI = 1` is the exact flag value a working build also prints. SVE, DOTPROD, and
MATMUL_INT8 are simply missing from the feature list — mask 0, nothing detected — and
the only place that surfaces is a `kleidiai:` log line most people never read.

## Root cause: the probe, not the chip

Diagnosed by compiling probes directly with gcc 13.3 on the box. CMake can't resolve an
explicit `-march`/`-mcpu` here, so it falls back to probing feature suffixes on
`-mcpu=native`. `-mcpu=native` alone compiles; `-mcpu=native+dotprod` and
`-mcpu=cortex-x925` are both **rejected** — gcc 13.3 predates Cortex-X925 in its own
`-mcpu` table. `-march=armv9.2-a+i8mm`, bypassing that table entirely, compiles cleanly.

The tell that this is a broken probe rather than a missing feature: the *negative*
controls fail too — a probe meant to confirm a feature's absence fails to even compile.
Every suffixed probe collapses to "no," KleidiAI compiles with no matmul kernels
selected, and nothing in the build or the banner says so.

**Scope:** this is gcc 13.3.0 paired with Cortex-X925, a core that postdates that
compiler release — not a claim that KleidiAI is broken on Arm generally. The reusable
lesson: this recurs on any CPU newer than the toolchain building for it, because
`-mcpu=native` probing has nothing valid to test against until the compiler knows the
chip's name.

## The fix, and how to check your own build

One flag pair, added to the same documented command:

```
-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
```

Same commit, compiler, machine: 149 `kai_` symbols, 10 `kai_run_matmul` kernels (6
dotprod, 2 i8mm, 2 sve), banner reading `MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 |
SVE_CNT = 16`, and `kleidiai: primary q4 kernel feature I8MM`.

If you're building `llama.cpp` with KleidiAI, don't trust `KLEIDIAI = 1` alone — both
builds above set it. Instead: `nm` the binary for `kai_run_matmul` and confirm the count
is non-zero, and read the `kleidiai:` log lines at startup, not just the feature banner
(`mask 0` means nothing was selected). If either comes back empty, pass
`GGML_NATIVE=OFF` with an explicit `GGML_CPU_ARM_ARCH` naming your target's ISA
extensions instead of relying on `-mcpu=native` to guess.

## The throughput this CPU delivers, once built right

`llama-server`, continuous batching, `Qwen2.5-0.5B-Instruct-Q4_0`, 3 rounds per row,
median reported, all from `results/server/server-bench.json`:

| clients | aggregate tok/s | per-client tok/s | TTFT p50 | TTFT p99 | peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1  |  14.9 | 14.9 | 89ms  | 89ms  | 724 MiB |
| 4  |  56.6 | 14.2 | 92ms  | 221ms | 761 MiB |
| 8  | 271.8 | 34.0 | 62ms  | 117ms | 809 MiB |
| 16 | 440.4 | 27.6 | 120ms | 168ms | 901 MiB |

Zero errors across every row. Throughput scales close to linearly with concurrency —
roughly 30× from 1 to 16 clients — while peak RSS stays under a gigabyte. Worth flagging
honestly: TTFT p99 isn't monotonic with load. The 4-client row (221ms) is the least
favorable point in the sweep, worse than the 16-client row (168ms). We don't have a
confirmed explanation for that and aren't guessing at one here.

## One more gap worth knowing about

Two of the 10 compiled kernels are SVE kernels. Under 8 concurrent clients, a
`gdb`-attached trace counting every real call recorded `{"dotprod": 11360, "i8mm":
364444}` — zero SVE calls out of 375,804 total. `kleidiai.cpp` gates the whole SVE
family on an *exact* 256-bit vector width, not a minimum; Cortex-X925 implements SVE2 at
128 bits, so that gate can never open on this core, regardless of how the build is
flagged. This mechanism was published first by
[`luongs3/arm-dispatch-audit`](https://github.com/luongs3/arm-dispatch-audit), two days
before this repository existed — credit for identifying it goes to them.

## Where this stands

Findings 1 and 2 (SME2 thread-gating on Apple Silicon, and the SVE width gate) are
already filed upstream as
[ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547). This
build-flag finding is new as of this session and hasn't been written up as a separate
upstream issue yet. Full evidence and methodology: [the repo](https://github.com/tomyimkc/polygraph)
(Apache-2.0); live dashboard at <https://tomyimkc.github.io/polygraph/>.

Arm server CPUs can clearly do this work. The gap is making sure your build actually
uses them.
