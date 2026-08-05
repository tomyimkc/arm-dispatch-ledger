# Upstream issue draft for ggml-org/llama.cpp — Finding 3

> **Filed 2026-08-05 as [ggml-org/llama.cpp#26630](https://github.com/ggml-org/llama.cpp/issues/26630).** The text below this banner is the report as sent, and is frozen; corrections belong in a comment on the issue.


Status: **DRAFT — not filed.** This file is prepared for the repository owner to post manually to
`ggml-org/llama.cpp`; nothing below has been submitted. It is a separate, previously-unreported
defect from [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547) (filed
2026-08-04, open), which covers Finding 1 (SME2 thread-gating on Apple Silicon) and Finding 2 (the
exact-256-bit SVE width gate). This issue is a follow-up from the same project, on a different
machine, and is linked to #26547 at the end.

Everything under "Evidence" was produced by code in `tomyimkc/polygraph` (Apache-2.0) and can be
reproduced with the commands shown.

---

## Title

Build: `-DGGML_CPU_KLEIDIAI=ON`'s documented build line silently compiles zero `kai_run_matmul`
kernels on gcc 13.3 + Cortex-X925 (banner still reports `KLEIDIAI = 1`); cost grows with model size

## Environment

| | |
|---|---|
| llama.cpp commit | `dbadb68eecdfb3ab0e86872d011738fc937f0364` |
| Build flags (documented, broken) | `-DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release` |
| Build flags (workaround, fixed) | add `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"` |
| Hardware | NVIDIA DGX Spark (GB10), Cortex-X925 + Cortex-A725, 20 cores, 121 GiB, Armv9.2 |
| Kernel | Linux `6.17.0-1021-nvidia`, aarch64 |
| Compiler | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| Models | `Qwen2.5-0.5B-Instruct-Q4_0.gguf` and `Qwen2.5-7B-Instruct-Q4_0.gguf` (both Apache-2.0) |

## Summary

Following `llama.cpp`'s own documented KleidiAI build line on this machine —

```
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
```

— produces a binary with **zero** `kai_run_matmul` symbols of any kernel family (dotprod, i8mm, or
sve), and a runtime banner that still prints `KLEIDIAI = 1` while the load-time log reports:

```
kleidiai: no compatible q4 kernels found for CPU features mask 0
kleidiai: no compatible q8 kernels found for CPU features mask 0
kleidiai: no compatible f32 kernels found for CPU features mask 0
```

`DOTPROD`, `MATMUL_INT8`, and `SVE` are also entirely absent from that banner — this is not a
KleidiAI-specific failure, it is a total CPU-feature-detection failure that happens to be most
visible through KleidiAI's own diagnostic lines. The build completes and exits 0. No warning is
printed anywhere in the build or at runtime that acceleration failed.

Adding `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"` to the same
command line, same source, same commit, same compiler, produces a binary with 10 `kai_run_matmul`
symbols and a banner reporting `MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1` plus two successful I8MM
kernel selections. We measured what that difference costs in throughput: on a 7B model it is up to
a **4.57x prefill** and **1.65x decode** loss; on a 0.5B model the same comparison is much smaller
and, for decode, a statistical tie. The cost scales with model size, which is exactly why a small
smoke-test model would not surface this defect.

## Reproduction

```bash
# Broken build — llama.cpp's own documented KleidiAI build line, unmodified
cmake -S . -B build-broken -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-broken -j

# Fixed build — same source, same commit, same compiler, one added flag pair
cmake -S . -B build-fixed -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
cmake --build build-fixed -j

# Static symbol count, either build
nm -D build-broken/bin/libggml-cpu.so | grep -c '^[0-9a-f]* T kai_run_matmul'   # -> 0
nm -D build-fixed/bin/libggml-cpu.so  | grep -c '^[0-9a-f]* T kai_run_matmul'   # -> 10

# Runtime banner + kleidiai selection log, either build
./build-broken/bin/llama-cli -m <model>.gguf -p "hi" -n 8 -no-cnv --simple-io -v
./build-fixed/bin/llama-cli  -m <model>.gguf -p "hi" -n 8 -no-cnv --simple-io -v
```

## Observed vs expected

Both banners below are quoted verbatim from `results/server/spark-provenance.txt` in the linked
project repository.

**Broken build** (`-DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release` only):

```
system_info: ... n_threads = 4 (n_threads_batch = 4) / 20 | CPU : NEON = 1 | ARM_FMA = 1 |
  LLAMAFILE = 1 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: no compatible q4 kernels found for CPU features mask 0
kleidiai: no compatible q8 kernels found for CPU features mask 0
kleidiai: no compatible f32 kernels found for CPU features mask 0
```

**Fixed build** (`+ -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"`):

```
system_info: ... n_threads = 4 (n_threads_batch = 4) / 20 | CPU : NEON = 1 | ARM_FMA = 1 |
  FP16_VA = 1 | MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 | SVE_CNT = 16 | OPENMP = 1 |
  KLEIDIAI = 1 | REPACK = 1 |
kleidiai: primary q4 kernel feature I8MM
kleidiai: primary q8 kernel feature I8MM
kleidiai: no compatible f32 kernels found for CPU features mask 3
```

`KLEIDIAI = 1` is printed identically in both banners, as are `NEON = 1`, `ARM_FMA = 1`,
`OPENMP = 1`, and `REPACK = 1`. Nothing in the broken banner says acceleration is off; the only
signal is what is *missing* (`FP16_VA`, `MATMUL_INT8`, `SVE`, `DOTPROD`, `SVE_CNT`) and the three
`kleidiai:` lines reporting `mask 0` instead of two successful `I8MM` selections. Static symbol
counts (`results/server/kai-symbols.txt` in the linked repository): 36 `kai_` symbols total but 0
`kai_run_matmul` entry points in the broken build; 149 `kai_` symbols and 10 `kai_run_matmul` entry
points (2 sve, 2 i8mm, 6 dotprod) in the fixed build.

## Root cause

**Confirmed, from the two banners above:** the broken build's CPU-feature bitmask is 0 across the
board — not just for KleidiAI. `DOTPROD`, `MATMUL_INT8`, and `SVE` are absent from the broken
banner even though this exact CPU reports all three once the fixed flags are used. That rules out
a KleidiAI-specific bug; the feature-detection step that runs before KleidiAI's kernel selection
is producing an empty result for this compiler/CPU pair.

**Reported diagnosis, not itself a committed artifact — flagged here as exactly that.** In
diagnosing this, we observed CMake's configure step log that it could not resolve an explicit
`-march`/`-mcpu` for this target and falls back to probing feature suffixes appended to
`-mcpu=native` (`+dotprod`, `+i8mm`, `+sve`, etc.). On this gcc 13.3 + Cortex-X925 pairing, every
suffixed probe we observed failed — including the negative-control probes (a probe that is
*supposed* to fail, failing), which is the specific signature of a broken probe rather than a
genuinely featureless CPU: `-mcpu=native` alone compiles, `-mcpu=cortex-x925` is rejected (gcc 13.3
predates Cortex-X925 in its own `-mcpu` table), and `-mcpu=native+dotprod` — the exact suffixed
form the probing produces — is also rejected. An explicit `-march=armv9.2-a+i8mm`-style target,
bypassing `-mcpu=native` entirely, compiles cleanly, which is the fix used above. We did not
capture the individual probe compiler invocations as a separate log file in our evidence
repository, so we are presenting this as the diagnostic narrative behind the two verified banners
above, not as a numbered artifact — happy to re-run and capture the raw compiler transcript if that
would help triage.

**The general lesson, not a claim about Arm broadly:** the feature probe as currently implemented
cannot distinguish "this CPU lacks the feature" from "this compiler cannot express this flag for
this `-mcpu`." Any CPU newer than the toolchain compiling for it can hit this same silent-zero
outcome — gcc 13.3 predating Cortex-X925 in its `-mcpu` table is the specific trigger here, but the
mechanism is general to any `-mcpu=native+<suffix>` probe against an unrecognized `-mcpu` base.

## Measured cost

`llama-bench -p 128 -n 32`, 5 reps per configuration (`llama-bench -r 1` per invocation, median
computed across the 5 independent invocations, not `llama-bench`'s own internal averaging),
**round-robin interleaved** (one rep of every configuration, then the next rep of every
configuration, repeated — never all reps of one configuration back to back), default thread count
(20, confirmed to sit inside or immediately adjacent to the dedicated `t=20` sweep row for both
models and both phases — see `results/scale/SCALE-EXPERIMENT.md` §1 in the linked repository for
that derivation). Source: `results/scale/scale-experiment.json`, `note` field records the
interleaving discipline verbatim.

| model | build | prefill median (tok/s) | prefill stdev | decode median (tok/s) | decode stdev | n |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct-Q4_0 | broken | 657.00 | ±51.29 | 42.18 | ±6.78 | 5 |
| Qwen2.5-0.5B-Instruct-Q4_0 | fixed  | 933.63 | ±74.03 | 41.70 | ±6.86 | 5 |
| Qwen2.5-7B-Instruct-Q4_0   | broken | 48.64  | ±0.42  | 11.17 | ±0.37 | 5 |
| Qwen2.5-7B-Instruct-Q4_0   | fixed  | 222.14 | ±4.94  | 18.45 | ±0.84 | 5 |

Prefill and decode are kept separate throughout; they are never blended into one number.

- **0.5B prefill:** 657.00 → 933.63 = **1.42x** (933.63 / 657.00 = 1.4211). Bands do not overlap
  (657.00 + 51.29 = 708.29 < 933.63 − 74.03 = 859.60) — real, even at this size.
- **0.5B decode:** 42.18 → 41.70 = **0.99x** (41.70 / 42.18 = 0.9886). Bands overlap almost
  completely (broken spans 35.40–48.96, fixed spans 34.84–48.56) — a **statistical tie, not an
  effect**. At 0.5B the defect costs prefill but not decode.
- **7B prefill:** 48.64 → 222.14 = **4.57x** (222.14 / 48.64 = 4.5670). Bands are nowhere close
  (48.64 + 0.42 = 49.06 vs. 222.14 − 4.94 = 217.20) — the largest single multiple we measured
  anywhere in this comparison.
- **7B decode:** 11.17 → 18.45 = **1.65x** (18.45 / 11.17 = 1.6517). Bands do not overlap
  (11.17 + 0.37 = 11.54 < 18.45 − 0.84 = 17.61) — real, and unlike at 0.5B, decode is now
  measurably affected too.

The cost is size-dependent and grows, not shrinks, with model size: prefill-only and modest at
0.5B, both-phases and large at 7B. A toy model would not surface how expensive this defect actually
is — it takes a model large enough to be memory-bandwidth- and matmul-shape-sensitive to expose the
real magnitude of compiling zero KleidiAI matmul kernels instead of ten.

## Suggested fix

We are not proposing a specific patch — offering three options in increasing order of
intrusiveness, and happy to send a PR for whichever direction the maintainers prefer, or to be
redirected to a different approach entirely:

1. **Least intrusive — a warning.** When every feature-suffix probe fails (dotprod, i8mm, sve, and
   any others CMake tries), emit a `message(WARNING ...)` at configure time. That combination —
   every probe failing, including probes checked as negative controls — is almost certainly a
   broken `-mcpu=native` probe on an unrecognized CPU, not a genuinely featureless Armv9 CPU, and a
   one-line warning would have told us that immediately instead of requiring a `kai_run_matmul`
   symbol count and a throughput regression to notice.
2. **A step further — treat both-failed as inconclusive.** If a feature's positive probe
   (`-mcpu=native+feature`) *and* its negative control both fail to compile, that should be treated
   as "probe inconclusive" rather than silently folded into "feature unsupported." Right now both
   outcomes appear to collapse to the same "not supported" result, which is what let this reach a
   released banner claiming `KLEIDIAI = 1` with zero compiled kernels.
3. **Most intrusive — a fallback path.** When `-mcpu=native+<feature>` is rejected by the compiler,
   fall back to probing `-march=<detected-arch>+<feature>` (bypassing `-mcpu=native` entirely, the
   same substitution that fixes this build manually) before giving up on that feature.

## Scope — what this is, and what it is not

This is **gcc 13.3.0 + Cortex-X925** on Ubuntu 24.04 — a CPU that postdates that compiler's
`-mcpu` support table. **This is not a claim that KleidiAI or llama.cpp is broken on Arm
generally.** What we did not test, and are not claiming anything about:

- Other compilers (gcc 14+, clang) — a newer gcc that recognizes `cortex-x925` may not exhibit this
  at all.
- Other CPUs — this may be specific to Cortex-X925's absence from gcc 13.3's `-mcpu` table, or it
  may recur on any sufficiently-new Arm core paired with a sufficiently-old default-distro gcc; we
  have not surveyed other pairings.
- Other Linux distributions.
- Cross-compilation (all builds here were native, on-device).

The general lesson we think is worth recording regardless of this specific pairing: the current
feature probe cannot distinguish "the CPU lacks this feature" from "the compiler cannot express
this flag for this `-mcpu`," and any CPU newer than the toolchain building for it can hit the same
silent-zero-kernel outcome.

## Relationship to #26547

This is a follow-up finding from the same project (`tomyimkc/polygraph`, Apache-2.0) that filed
[ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547), measured on
different hardware (DGX Spark / Cortex-X925 vs. that issue's Apple M4 Max and GitHub-hosted
Neoverse-N2 runner) and reporting a different mechanism (a build-time feature-probe failure, not a
runtime dispatch gate). We are filing it separately because the root cause, the affected code path,
and the fix are unrelated to either finding in #26547 — #26547 is about kernels that compile but
are silently *not selected* at runtime; this issue is about kernels that never compile at all.

Thank you for KleidiAI and llama.cpp's CPU backend generally — this is meant in the spirit of
closing a gap between what the build banner claims and what actually got compiled, not a complaint
about KleidiAI's design.
