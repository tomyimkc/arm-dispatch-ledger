# Server lane: DGX Spark, `llama-server`, continuous batching

> **Authoritative for this lane's numbers.** Every figure in this document is drawn verbatim from
> `results/server/server-bench.json`, `results/server/server-dispatch.json`, or
> `results/server/spark-provenance.txt`, or is an arithmetic derivation of those numbers computed
> in this document (shown with its inputs, never asserted bare). Where the underlying diagnostic
> narrative is broader than what is captured in those three files, that is flagged explicitly in
> place — see "Provenance note" under Finding 3 below. Anything not backed by one of the three
> files or a shown computation is marked `[not captured in this lane's evidence]`.

Measured 2026-08-05. This is a new, additive lane: it does not modify, and is not required to
agree numerically with, any pre-existing `results/` artifact (see `results/RENAME-NOTE.md` for why
existing artifacts are never edited after the fact — the same discipline applies here: nothing
above already committed to `results/` was touched to produce this page).

---

## Hardware / provenance

| | |
|---|---|
| Host | `spark-2f2d`, NVIDIA DGX Spark (GB10) |
| Kernel | Linux `6.17.0-1021-nvidia`, aarch64 |
| CPU | Cortex-X925 + Cortex-A725, 20 cores |
| Memory | 121 GiB |
| Compiler | gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 |
| `llama.cpp` | `dbadb68ee` — "ggml: use dynamic allocation for split graph inputs (#22789)" |

The commit is the short form of `dbadb68eecdfb3ab0e86872d011738fc937f0364` — the same commit pinned
for every other piece of evidence in this repository (`docs/FINDINGS.md`,
`results/GROUND-TRUTH-DISPATCH.md`). Source: `results/server/spark-provenance.txt`.

This is genuine Arm server-class silicon (20-core Armv9.2, no SME), distinct from every prior lane
in this repository, which ran on Apple Silicon (M4 Max, SME2-capable) or GitHub's free
`ubuntu-24.04-arm` CI runner (Neoverse-N2, 4 cores). It is the first evidence in this project
collected under **concurrent multi-client serving load** rather than single-user `llama-cli`
decode/prefill.

---

## Finding 3 — the documented KleidiAI build line silently produces zero matmul kernels on this compiler/CPU pair

### The headline fact, verified verbatim in `spark-provenance.txt`

Following `llama.cpp`'s own documented build line on this machine —
`cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release` — produces a binary with:

```
kai_run_matmul symbols: 0
```

Not "SME2 unreachable," not "falls back to a slower kernel" — **zero** compiled-in KleidiAI matmul
micro-kernel entry points, of any family (dotprod, i8mm, or sve). The build completes and exits 0.
A second build on the same source, same commit, same compiler, adding only
`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"`, produces:

```
kai_run_matmul symbols: 10
```

Both counts are read directly from `results/server/spark-provenance.txt`; neither is a symbol count
this document computes or estimates.

### Both builds' banners, side by side (verbatim from `spark-provenance.txt`)

**Default build** (`-DGGML_CPU_KLEIDIAI=ON` only):

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

**The tell, read straight off these two banners:** `KLEIDIAI = 1` is printed in *both*. So are
`NEON = 1`, `ARM_FMA = 1`, `OPENMP = 1`, `REPACK = 1`. A user reading only the default build's
banner sees nothing that says "acceleration is off" — the `KLEIDIAI = 1` flag reads exactly like
success. What differs is everything the default banner is *missing* versus the fixed one:
`FP16_VA`, `MATMUL_INT8`, `SVE`, `DOTPROD`, and `SVE_CNT` are present in the fixed banner and simply
absent from the default one, and the `kleidiai:` lines flip from three "no compatible kernels
found ... mask 0" failures to two successful "primary kernel feature I8MM" selections. `mask 0`
in the default build's log is the CPU-feature bitmask KleidiAI's kernel-selection logic computed
for this run — zero features detected, hence zero eligible kernels, hence the log's own explicit
admission that it found none. Nothing about this failure is loud: no warning, no non-zero exit
code, no line in the default banner that says "acceleration failed."

### Root cause — reported diagnosis, with an explicit provenance note

The build's own CMake configure step logs (per this lane's build investigation) that it could not
resolve an explicit `-march`/`-mcpu` for this target and falls back to probing feature suffixes on
top of `-mcpu=native` (`+dotprod`, `+i8mm`, `+sve`, etc.). On this gcc 13.3 + Cortex-X925 pairing,
those probe compiles were reported to fail — including, notably, the negative-control probes
(a probe expected to fail, failing) — which is the specific signature of a broken *probe*, not an
absent feature: `-mcpu=native` alone compiles, `-mcpu=cortex-x925` is rejected (gcc 13.3 predates
Cortex-X925 in its `-mcpu` table), and `-mcpu=native+dotprod` — the exact suffixed form CMake's
probing produces — is also rejected. An explicit `-march=armv9.2-a+i8mm` target, bypassing
`-mcpu=native` entirely, compiles cleanly, which is exactly the fix applied above.

**Provenance note:** the individual gcc probe invocations and their pass/fail outcomes described in
this paragraph are the diagnostic narrative behind this finding, but — unlike the two build banners
quoted verbatim above — they are not themselves captured as a separate committed artifact under
`results/server/`. The three files this lane commits (`server-bench.json`, `server-dispatch.json`,
`spark-provenance.txt`) capture the two build banners and the runtime consequences, which is the
evidence this document treats as authoritative. The causal story above is offered as the
root-cause explanation for why one build produces `kai_run_matmul symbols: 0` and the other
produces `10` — a difference that *is* directly and verbatim confirmed in `spark-provenance.txt` —
but the probe-by-probe compiler transcript itself should be read as reported methodology, not as a
number sourced from a committed JSON/log file. No symbol-family breakdown of the 10
`kai_run_matmul` entry points (e.g. how many are dotprod vs. i8mm vs. sve vs. neon), and no count
of KleidiAI's packing-helper symbols in either build, is present in `spark-provenance.txt` — this
document does not state those numbers.

### The fix

```
-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
```

One flag pair, added to `llama.cpp`'s own documented KleidiAI build line, turns 0 compiled matmul
kernels into 10, and turns three "no compatible kernels found" failures into two successful I8MM
kernel selections — all verified verbatim above.

### Severity and scope — read this precisely

This is a **build-time, silent, complete loss of all Arm-specific matmul acceleration** —
reproducible by following `llama.cpp`'s own documented KleidiAI build instructions, with no error,
no warning, and a banner (`KLEIDIAI = 1`) that reads as if acceleration succeeded.

It is **not** a claim that every Arm machine is affected. It is specifically this
**gcc 13.3.0 + Cortex-X925** combination: gcc 13.3 predates Cortex-X925 in its own `-mcpu` support
table (confirmed by the `-mcpu=cortex-x925` rejection above), so CMake's `-mcpu=native`-based
feature-suffix probing has nothing valid to probe against on this CPU with this compiler, and every
suffixed probe fails — including, per the reported diagnosis, the probes that are *supposed* to
fail, which is what marks the probing mechanism itself as broken rather than the hardware as
lacking the features. **The general lesson is not "Cortex-X925 is broken"; it is that this failure
mode will recur on any CPU newer than the toolchain compiling for it** — any time a distribution's
default gcc predates a chip's addition to gcc's `-mcpu` table, the same silent zero-kernel outcome
is the predictable result of `llama.cpp`'s current documented build line on that pairing, not a
one-off Spark quirk.

---

## Finding 2 — confirmed on a second core family, for the first time under concurrent serving load

`docs/FINDINGS.md`'s Finding 2 (the `kleidiai.cpp:209` gate,
`ggml_cpu_has_sve() && ggml_cpu_get_sve_cnt() == QK8_0`, i.e. an *exact* 256-bit/32-byte SVE width
requirement) was previously measured only on GitHub's free `ubuntu-24.04-arm` runner
(Neoverse-N2, 4 cores, single-user `llama-cli`). This lane adds a second, independent core family
and, for the first time in this repository, a measurement taken while the process is serving
concurrent clients rather than at single-shot load time.

The fixed build's banner (verbatim, quoted above) reports:

```
SVE = 1 | DOTPROD = 1 | SVE_CNT = 16
```

`SVE_CNT = 16` means Cortex-X925's SVE2 implementation is 16 bytes wide — 128 bits — exactly half
of the 32-byte/256-bit width `kleidiai.cpp:209`'s equality check requires. The gate predicts SVE
must be permanently unreachable on this core, and I8MM must be selected instead — which is exactly
what the same banner's `kleidiai: primary q4/q8 kernel feature I8MM` lines already confirm at
selection time (quoted above). This is a second, independent core family (Cortex-X925, in addition
to the previously-confirmed Neoverse-N2) landing on the identical predicted outcome.

### Confirmed at the dispatch layer, under real concurrent load

`results/server/server-dispatch.json` (gdb attached to a running `llama-server`, 10
`kai_run_matmul` breakpoints set — one per symbol reported in the fixed build's
`kai_run_matmul symbols: 10` count above — while 8 concurrent clients were driven against the
server, the same `parallel=8`/`clients=8` configuration measured in the throughput table below):

```json
{
 "dotprod": 11360,
 "i8mm": 364444
}
```

Two things this file makes verifiable rather than merely predicted:

- **No `sve` key appears in the file at all.** Across every one of `11360 + 364444 = 375,804` total
  recorded `kai_run_matmul` calls, zero were attributed to the SVE kernel family. This is the L3
  dispatch-layer confirmation the Neoverse-N2 CI run and Cortex-X925's own SVE_CNT arithmetic both
  predicted, now obtained from a real `gdb`-attached trace on a live server process, not a
  single-shot `llama-cli` run.
- **The dispatch shape inverts versus single-user decode.** i8mm accounts for 364,444 of 375,804
  calls (roughly 97% by call count) and dotprod for the remaining 11,360 (roughly 3%) — overwhelmingly
  i8mm-dominated. This is the opposite of Apple Silicon's single-user decode trace in
  `docs/FINDINGS.md` Finding 1, where the decode phase dispatches through the dotprod/NEON slot,
  never the batched-GEMM-shaped kernel. The mechanism is continuous batching: `llama-server -cb`
  turns per-token GEMV ops (`ne11 == 1`, the shape that dominates single-user decode) into batched
  GEMM ops (`ne11 > 1`) once multiple clients' tokens are packed into one matmul call, and it is the
  GEMM-shaped kernel family (i8mm) that KleidiAI's dispatcher prefers at that shape. SVE, again, is
  never entered — the `== QK8_0` gate excludes it regardless of GEMV/GEMM shape.

### Status upgrade

Finding 2 in `docs/FINDINGS.md` previously read "architecturally derived, dispatch confirmation
pending" for anything beyond the single Neoverse-N2 CI measurement. With this lane's evidence,
`docs/FINDINGS.md` records the additional confirmation: a second core family (Cortex-X925), and for
the first time, confirmation under concurrent multi-client serving load rather than only at
single-shot load/decode time. See `docs/FINDINGS.md` for the updated status line; this document
does not restate it a second time to avoid the two drifting apart.

---

## Cloud-AI throughput / TTFT / memory

`llama-server`, continuous batching (`-cb`), 3 rounds per configuration, median reported. All rows
below are reproduced verbatim from `results/server/server-bench.json` (`agg_tps_median`,
`ttft_p50`, `ttft_p99`, `per_client_tps_median`, `peak_rss_mib`, `rounds`, `errors`):

| parallel | threads | threads_batch | clients | aggregate tok/s | per-client tok/s | TTFT p50 | TTFT p99 | peak RSS | rounds | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 20 | 20 | 1  |  14.9 | 14.9 | 0.089s (89ms)  | 0.089s (89ms)  | 724 MiB | 3 | 0 |
| 4  | 20 | 20 | 4  |  56.6 | 14.2 | 0.092s (92ms)  | 0.221s (221ms) | 761 MiB | 3 | 0 |
| 8  | 20 | 20 | 8  | 271.8 | 34.0 | 0.062s (62ms)  | 0.117s (117ms) | 809 MiB | 3 | 0 |
| 8  |  4 | 20 | 8  | 264.8 | 33.1 | 0.062s (62ms)  | 0.094s (94ms)  | 809 MiB | 3 | 0 |
| 16 | 20 | 20 | 16 | 440.4 | 27.6 | 0.120s (120ms) | 0.168s (168ms) | 901 MiB | 3 | 0 |

Every row reports `errors: 0` — no failed requests across any configuration or round.

### Two readings that matter

**(a) Aggregate throughput scales close to linearly with concurrency, and cost stays flat.**
14.9 → 440.4 tok/s from 1 to 16 concurrent clients is a **29.6×** increase (440.4 / 14.9 = 29.56,
computed from the two rows above), while TTFT p99 at those two endpoints — 0.089s and 0.168s — is
comfortably under 170ms, and peak RSS grows only 724 → 901 MiB — **+177 MiB (+24.4%)** for a 16×
increase in concurrent clients — computed directly from the table's own `peak_rss_mib` column. The
sweep is **not perfectly monotonic**, though: the interior `parallel=4` row spikes to 0.221s
(221ms) TTFT p99, the highest value anywhere in the table — higher than the 16-client row. For a
Cloud AI / dedicated on-prem server use case, the shape that matters is still favorable — throughput
scales with load and memory does not scale proportionally with concurrency — but latency is not
uniformly bounded across every row measured, and that outlier should not be smoothed over.

**(b) At `parallel=8`, thread count barely matters — the server is batch-bound, not thread-bound.**
Dropping `threads` (the generation thread count) from 20 to 4 at `parallel=8` costs almost nothing:
271.8 → 264.8 tok/s, a **2.6% drop** (computed: 1 − 264.8/271.8 = 0.0258), and TTFT p99 actually
*improves*, 0.117s → 0.094s, a **19.7% reduction** (computed: 1 − 0.094/0.117 = 0.1966). This is a
striking contrast with this repository's other findings: `docs/FINDINGS.md` Finding 1 and the
`OPTIMIZATION.md` crossover measurements show single-user decode throughput is highly sensitive to
thread count (fewer threads help token generation, sometimes dramatically). Under continuous
batching with 8 concurrent clients already saturating the batch, that thread-count sensitivity
**largely disappears** — the bottleneck has moved from per-thread scheduling to how many requests
are packed into each batched matmul call. This is a genuinely different regime from every prior
single-user measurement in this repository, and it is the honest counterweight to the project's
thread-tuning story: the tuning that helps a single-user decode workload the most is close to
irrelevant once the server is busy enough to be batch-bound.

---

## What this lane does NOT show

- **No SME, and Finding 1 is not reproducible here.** The DGX Spark's Cortex-X925 does not
  implement SME/SME2 — neither build banner above reports `SME` or `SME2` at all (compare the
  Apple M4 Max banners in `results/GROUND-TRUTH-DISPATCH.md`, which show `SME = 1 | SME2 = 1`).
  Finding 1 (the hardcoded per-chip SME thread cap and its `ne11`-gated hybrid rescue path) is an
  SME-specific mechanism and simply does not apply to this hardware. Nothing in this lane confirms,
  contradicts, or extends Finding 1.
- **Single model, single quantization, single serving binary.** All throughput/TTFT/RSS numbers and
  the dispatch trace were collected against one model at one quantization level, served through
  `llama-server` only. No other model size, quant format, or serving binary (e.g. a different
  inference server, a different batching scheduler) was exercised in this lane.
- **No symbol-family breakdown of the fixed build's 10 `kai_run_matmul` entries.**
  `spark-provenance.txt` reports the count (10) but not which of dotprod/i8mm/sve/neon each
  resolved symbol belongs to, and no packing-helper (`kai_lhs_quant_pack_*` / `kai_rhs_pack_*`)
  symbol count was captured for either build. `server-dispatch.json`'s two dispatch-count buckets
  (`i8mm`, `dotprod`) are call-level dispatch data, not a static symbol inventory, and are not a
  substitute for that missing static breakdown.
- **No gcc-probe transcript committed as a separate artifact.** Finding 3's root-cause narrative
  (which specific `-mcpu`/`-march` probes gcc 13.3 accepted or rejected) is reported diagnosis, not
  a number read from a committed file — see the provenance note under Finding 3.
- **No repeated measurement across different concurrency levels for the dispatch trace.** The
  dispatch-under-load numbers in `server-dispatch.json` were captured at one configuration
  (`parallel=8`/`clients=8`); the shape-inversion argument above (i8mm-dominated under batching vs.
  dotprod-dominated at single-user decode) is drawn from comparing that one configuration against
  Finding 1's separately-measured single-user Apple Silicon trace, not from sweeping serving
  concurrency and watching the dispatch mix change in this lane.
- **This lane's throughput/TTFT/RSS table is the full set of configurations measured** — five rows,
  three rounds each — not a broader sweep across prompt lengths, output lengths, or context sizes.
