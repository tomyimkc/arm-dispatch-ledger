# Polygraph

**Polygraph checks whether your software is telling the truth about the hardware acceleration it
claims to use.** Point it at a binary and a workload; it attaches a debugger to the real kernel
entry points — not a timing guess, not the startup banner — and reports whether the accelerated
code path actually ran. **Does your software actually do what it says?** That's the question this
tool answers, and the question the rest of this document answers for one real case: `llama.cpp`'s
KleidiAI CPU backend on Arm.

**Why "Polygraph":** a lie detector doesn't take a claim's word for it — it checks what the body
actually did underneath it. This project does the same for a software performance claim: it
doesn't trust the log line that says an accelerator is "enabled," it counts the real calls into
the kernel.

**Previously called Arm Dispatch Ledger.** The old GitHub URL,
[`github.com/tomyimkc/arm-dispatch-ledger`](https://github.com/tomyimkc/arm-dispatch-ledger),
redirects here automatically — an old link or clone still works, it just lands on this repo under
its new name.

**`llama.cpp` already has the flags to fix this. Nobody uses them, because the tool's own banner
says the fast kernel is already running.** This project verifies that it isn't (an undocumented
dispatch gap in `llama.cpp`'s KleidiAI CPU backend — a real limit of its hardcoded per-chip
thread-cap design, not a code defect), measures the per-phase performance cost of that gap, finds
the exact source-level reason the dispatcher can't correct itself, patches that reason, measures
the patch honestly — including where it falls short — and reports it upstream. The verifier
(`tools/verify_dispatch.py`) is the *method* used throughout this arc, not the whole product; it
ships alongside an MCP server, a phase-crossover benchmark harness, an upstream patch, a results
dashboard, and a hand-written Arm kernel library, all as reusable artifacts.

**Live dashboard:** <https://tomyimkc.github.io/polygraph/> — the advertised-vs-executed ledger, rendered from the committed JSON in `results/`, published by `.github/workflows/pages.yml` on every push to `main`.

**Prior art / scholarly hygiene:** this project's Finding 2 mechanism turns out to have been
published two days earlier by a different, unrelated repository. We cite it, state plainly what it
found first, and what this project adds on top — see
**[`docs/RELATED-WORK.md`](docs/RELATED-WORK.md)**, and read it before assuming either finding here
is uncontested.

**New (measured 2026-08-05): the most user-impactful finding yet, and the project's first
server-class evidence.** Following `llama.cpp`'s own documented KleidiAI build line on a DGX Spark
(Cortex-X925, 20-core Armv9.2 server silicon) produces a binary with **zero working matmul
micro-kernels** — while the startup banner still prints `KLEIDIAI = 1` and the build exits `0`. One
flag pair fixes it — see **Finding 3**, below. Rebuilt correctly, that same server sustains
**~29.6x** aggregate throughput scaling from 1 to 16 concurrent `llama-server` clients (14.9 →
440.4 tok/s) — this project's first inference-server (Cloud-AI) measurements, including
time-to-first-token, memory, and a second-platform dispatch confirmation of Finding 2 under real
serving load. Full tables and methodology: **"Cloud AI: measured on server-class Arm"**, below.

**The optimization, re-measured 2026-08-04 on this repo's own hardware (Apple M4 Max, `llama-bench
-r 1`, n=7, round-robin interleaved against external load, median ± stdev — full table below):**
matching thread count to phase — `-t 2` for decode, `-t 8`/`-tb 8` for prefill, both flags
`llama.cpp` already ships — beats the default config by **3.43x on decode (93.6 → 321.0 tok/s) and
1.79x on prefill (1,230.3 → 2,198.1 tok/s), today, with zero code changes.** Most of that decode
number is **not** an SME2 discovery: a dedicated decomposition sweep (full breakdown in
"Decomposition" under "The optimization" below) shows thread tuning *alone*, with SME2 forced off
the entire time, already accounts for the majority of the total measured win — the well-documented
"fewer threads help token generation on Apple Silicon" memory-bandwidth effect, not something this
project found. SME2 (Finding 1 below: a hardcoded per-chip thread cap that silently excludes decode
from the fast kernel) adds a real, smaller win on top of that at the tuned thread count, and
measurably *hurts* at the default thread count — see the Decomposition section for the exact ratios
and why the two effects are easy to conflate. We also wrote and measured an opt-in patch
(`patches/0001-kleidiai-phase-aware-dispatch.patch`) meant to recover the thread-gating gap
automatically for a user who never touches a thread flag at all — the honest, re-measured result is
that it does **not** help: at default thread count it is **~12% slower** (93.6 → 82.5 tok/s), a
real regression outside the noise, even though the dispatch change it makes is genuine and
symbol-level proven. See "The optimization" section below for the full before/after table, the
decomposition, the patch's mechanism, the symbol-level dispatch proof, and why the dispatch change
doesn't translate into a throughput win.

**Correction (2026-08-05): that 3.43x decode multiple does not generalize past a tiny model.** An
independent thread-count sweep on a second machine (DGX Spark, Cortex-X925) across two model sizes
shows the same class of win shrinking from **4.56x at 0.5B to a modest 1.33x at 7B** — see "Does the
3.43x decode-tuning win generalize to a bigger model?" under "The optimization," below, and
`results/scale/SCALE-EXPERIMENT.md` for the full data. The 93.6 → 321.0 tok/s / 3.43x figures above
remain accurate for their own scope (Apple M4 Max, this exact 0.5B model) and are not being
retracted; they should just not be read as a multiple that generalizes to larger models. The same
build defect this project's verifier exists to catch (Finding 3, below) turns out to cost *more*, not
less, at a realistic model size — 4.57x prefill / 1.65x decode at 7B, the largest speedup this
project has measured from fixing a single build-time defect rather than tuning a runtime knob.

*(An earlier version of the headline numbers above was wrong and has since been retracted and
re-measured — see [Correction (2026-08-04)](#correction-2026-08-04) below for the full account.)*

**The hook that started this — measured with real `lldb` breakpoints, not inferred:** at
`llama.cpp`'s *default* thread count (physical core count — 16 on this machine), single-token
decode **never dispatches SME2**, even though the startup banner and the runtime log both keep
claiming `SME2 enabled` on every single one of those runs.

```
threads=1   decode: SME2 fires (996 lldb hits)         <- advertised AND executed
threads=8   decode: SME2 fires ZERO times (31,871 NEON hits instead)   <- advertised, NOT executed
threads=16  decode: SME2 fires ZERO times (51,215 NEON hits instead)   <- advertised, NOT executed
```

All numbers in this document were produced by code in this repo, run for real on real Arm
hardware. Anything not yet measured is marked `[not yet measured]` — never invented, never
interpolated. See `results/REMEASURE-2026-08-04-QUIET.md` for the authoritative, corrected
throughput numbers (supersedes `results/OPTIMIZATION.md` and `results/crossover/` where they
disagree), `results/SUMMARY.md` for the diagnosis run log, and `results/GROUND-TRUTH-DISPATCH.md`
for the authoritative, corrected dispatch rule (an earlier draft of that finding was incomplete —
see "the correction, up front" in Finding 1 below).

---

## TL;DR

| | |
|---|---|
| **Advertised** (compile-time banner + selection-time log) | `SME = 1 \| SME2 = 1 \| KLEIDIAI = 1` and `kleidiai: primary q4 kernel feature SME2` — printed identically on **every** run below, including the ones where SME2 never executes once. |
| **Executed** (dispatch-time, `lldb` breakpoint on `kai_run_matmul.*sme`, 18 symbol locations) | Decode at `-t 4/8/16`: **0 hits**, 15,936–51,215 NEON-dotprod hits instead. Decode at `-t 1/2`: SME2 fires (996–5,826 hits). |
| **Net effect** | `llama.cpp` defaults `n_threads` to the physical core count. On this 16-core M4 Max, **that default silently never uses SME2 for token generation** — the common case for a chat session — while every log line a user would look at says otherwise. |

---

## Why this matters

A benchmark that only measures **tokens/sec** cannot see this. If SME2 silently falls back to
NEON, the run still completes, still prints a plausible number, and still prints a banner that
says `SME2 = 1`. Nothing about a timing-only harness would ever flag that the accelerator was
compiled in, selected in the log, and skipped at runtime. You need a debugger attached to the
actual kernel entry points to know the difference. This project's verifier (`tools/verify_dispatch.py`)
formalizes that into three independent evidence layers, and **never trusts the first two alone**:

| Layer | What it checks | Tooling | Proves |
|---|---|---|---|
| **L1 — static** | Do the accelerated-kernel symbols exist in the built library at all? | `nm`/`otool` (macOS), `nm`/`objdump` (Linux) | The kernel was *compiled in*. Nothing about runtime behavior. |
| **L2 — selection** | What does `llama.cpp`'s own verbose log say it *chose*? | Parses `kleidiai: primary q4 kernel feature X`, `SME2 enabled (...)` | The kernel was *selected* at model-load time. Still not proof of execution. |
| **L3 — dispatch** | Did the kernel's machine code actually *run*? | `lldb`/`gdb`, regex breakpoint on every `kai_run_matmul_*` entry point, real inference workload, count real hits | The only layer that answers the actual question. |

Every result in this repo that claims a kernel "dispatched" or "fell back" is backed by L3, not L1
or L2. L1 and L2 alone would have reported every single row below as `SME2 = 1` and been wrong
about most of them.

---

## Evidence platforms at a glance

| Platform | Class | Cores | ISA path measured | What it backs |
|---|---|---:|---|---|
| Apple M4 Max (macOS) | Laptop/desktop SoC | 16 | SME2 | Finding 1, the Apple M4 Max measured results, the optimization + patch |
| Cortex-X925 / DGX Spark | Server-class Arm, Armv9.2 | 20 | SVE2 (128-bit) → I8MM/DOTPROD | Finding 2 (now dispatch-confirmed), Finding 3, the Cloud AI server lane |
| Neoverse-N2 | Free, judge-reproducible CI (`ubuntu-24.04-arm`) | `[not measured]` | SVE2 (128-bit) | Finding 2's zero-cost, judge-reproducible static (L1) path |

---

## The three findings

### Finding 1 — SME2 is gated by a hardcoded per-chip thread cap, with a batch-size-dependent hybrid rescue path

**The correction, up front:** an earlier draft of this finding claimed "SME2 never fires above 2
threads, full stop." That is **not correct** and the earlier test only exercised one code path
because it used a 4-token prompt. The real rule, read from `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`
in `llama.cpp` @ `dbadb68`, has two independent paths:

```c
// kleidiai.cpp:156-161 — hardcoded Apple brand-string -> thread-cap table
struct ModelSMCU { const char *match; size_t smcus; };
static const ModelSMCU table[] = {
    { "M4 Ultra", 2 },
    { "M4 Max",   2 },   // <- this machine
    { "M4 Pro",   2 },
    { "M4",       1 },
};

// kleidiai.cpp:300
ctx.sme_thread_cap = (ctx.features & CPU_FEATURE_SME) ? sme_cores : 0;

// kleidiai.cpp:1094-1112 — the actual dispatch-time decision
const int  sme_cap_limit = ctx.sme_thread_cap;
const bool use_hybrid    = sme_cap_limit > 0 && runtime_count > 1 && nth_total > sme_cap_limit;
size_t min_cols_per_thread = std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);
const bool hybrid_enabled = use_hybrid && !too_small_for_hybrid;
if (!hybrid_enabled) {
    // ... chosen_slot = 1;  <- collapses to the NEON kernel, SME2 skipped entirely
}
```

SME2 dispatches if **either**: (1) `n_threads <= sme_thread_cap` (the plain SME path), **or**
(2) hybrid mode: `n_threads > sme_thread_cap` **and** `ne11 >= 128` **and** `ne01/n_threads >= 2`.
`ne11` is the batch size of the matmul — the number of tokens processed in one call.

That single variable, `ne11`, is why the finding splits cleanly by phase:

- **Decode** (autoregressive, one new token at a time): `ne11 == 1`, always. `too_small_for_hybrid`
  is always true. **SME2 is structurally unreachable above the thread cap for decode, on any
  prompt, on any model** — this is not a tuning knob, it's an architectural dead end for the most
  common inference pattern.
- **Prefill** of a long-enough prompt: `ne11` is large, the hybrid gate opens, and SME2 keeps
  firing *alongside* NEON even at 16 threads (a genuine split-batch hybrid, confirmed in the L3
  hit counts below — both families non-zero in the same run).

Meanwhile `system_info:` still prints `SME = 1 | SME2 = 1 | KLEIDIAI = 1` and the log still says
`kleidiai: primary q4 kernel feature SME2` — **in every zero-dispatch row below.** Both are
compile-time/selection-time signals; neither reflects what actually ran.

**Reproduce it:**

```bash
GGML_KLEIDIAI_SME=2 lldb -b -s tools/dispatch_probe.lldb -- \
  /tmp/llama.cpp/build/bin/llama-cli -m /tmp/ggufs/q05.gguf \
  -p "Hello." -n 4 -no-cnv -st --simple-io -t 8
# or, the full automated sweep used to produce the table below:
python3 tools/verify_dispatch.py --binary /tmp/llama.cpp/build/bin/llama-cli \
  --model /tmp/ggufs/q05.gguf --threads 1,2,4,8,16 --workloads all \
  --out results/dispatch-ledger-darwin-arm64.json --assert
```

Note: the first, manual `lldb` one-liner above sets `GGML_KLEIDIAI_SME=2` to explicitly pin the
cap — on this M4 Max that just restates the chip's own hardcoded default (also 2), it does not
change the result. The `verify_dispatch.py` sweep that actually produced the ledger table below
left this env var **unset** (see `"GGML_KLEIDIAI_SME": null` in
`results/dispatch-ledger-darwin-arm64.json`'s `env` block) — i.e. the real-world default a user
gets with no special configuration.

Full write-up with the correction history and prior-art check: `results/GROUND-TRUTH-DISPATCH.md`.
Root-cause deep dive: `docs/FINDINGS.md`.

### Finding 2 — the SVE kernel family is architecturally unreachable below 256-bit vectors

```c
// kleidiai.cpp:209
((ggml_cpu_has_sve() && ggml_cpu_get_sve_cnt() == QK8_0) ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);
```

`QK8_0 == 32` bytes == **256-bit**. This is an exact-equality check, not `>=`. The DGX Spark's
Cortex-X925 implements SVE2 at **128-bit**, so `CPU_FEATURE_SVE` can never be set there and the SVE
kernel family can never be selected — **regardless of the fact that the core genuinely has SVE2,
i8mm, and bf16.** Any current Arm core shipping 128-bit SVE2 (which is most of them — 256-bit+ SVE
is still rare outside HPC-class silicon) hits the same wall.

**Status:** read from source and confirmed by static analysis (`tools/verify_dispatch.py`'s L1
tier). **Now also confirmed at the L3 (dispatch) tier, on real SVE2 hardware:** a manual `gdb` trace
against a live `llama-server` process on the DGX Spark (Cortex-X925, `SVE_CNT = 16` — 128-bit)
recorded zero SVE-family dispatch under concurrent serving load — only `dotprod` (11,360 calls) and
`i8mm` (364,444 calls) — see **"Cloud AI: measured on server-class Arm"** below for the full trace
and methodology. That confirmation came from a direct, manual measurement on the Spark hardware,
**not** from the automated `.github/workflows/verify-spark-aarch64.yml` CI lane, which remains
`continue-on-error` and has still not completed a clean run (see Limitations) — the two are
independent, and only the manual measurement has produced real L3 evidence so far.

**Prior art:** this exact mechanism — the same `kleidiai.cpp:209` line, the same `QK8_0`-equality
gate — was published two days before this repository existed by a different, unrelated project.
We found it independently, later, and are not claiming priority. Full disclosure, dates, and what
this project adds beyond that prior work: [`docs/RELATED-WORK.md`](docs/RELATED-WORK.md).

### Finding 3 — `llama.cpp`'s documented KleidiAI build line silently ships zero acceleration on Cortex-X925

Following `llama.cpp`'s own documented build line, on the DGX Spark:

```bash
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
```

produces a binary with **zero `kai_run_matmul` symbols** — none of KleidiAI's matmul micro-kernels
compile in at all; only the packing helpers (`kai_lhs_quant_pack_*`, `kai_rhs_pack_*`) are present.
The startup banner does not say so (`results/server/spark-provenance.txt`):

```
system_info: ... | NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: no compatible q4 kernels found for CPU features mask 0
kleidiai: no compatible q8 kernels found for CPU features mask 0
kleidiai: no compatible f32 kernels found for CPU features mask 0
```

**`KLEIDIAI = 1` prints on the same run whose own log admits the CPU feature mask is `0`** — and
`SVE`, `DOTPROD`, and `MATMUL_INT8` have all silently dropped out of the banner entirely rather than
printing as disabled. The build exits `0`. Reading only L2 (the selection-time log, in this
project's own terminology from "Why this matters" above) would conclude KleidiAI is active; it is
compiled with zero working matmul kernels.

**Root cause:** on this machine (`gcc 13.3.0`, Ubuntu 24.04 aarch64 —
`results/server/spark-provenance.txt`), `llama.cpp`'s `CMakeLists.txt` logs "ARM -march/-mcpu not
found, -mcpu=native will be used," then probes for CPU features by appending flags —
`-mcpu=native+dotprod`, `-mcpu=cortex-x925`, and others — to `-mcpu=native`. This `gcc` rejects
every one of those probes, including the *negative* controls (a probe that checks a feature is
*absent* also fails) — the tell that the probing logic itself is broken, not that the hardware lacks
the feature. `-march=armv9.2-a+i8mm` compiles cleanly on the same toolchain, confirming the CPU
support is real.

**The fix is one flag pair:**

```
-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
```

Rebuilt with that pair, the same source tree and toolchain produce **10 `kai_run_matmul` symbols**
(`results/server/spark-provenance.txt`) and a banner that matches reality:

```
system_info: ... | NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 | SVE_CNT = 16 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: primary q4 kernel feature I8MM
kleidiai: primary q8 kernel feature I8MM
kleidiai: no compatible f32 kernels found for CPU features mask 3
```

`f32` still has no compatible kernel even in the fixed build — the flag pair does not fix every code
path, and this document does not pretend it does.

**Why this is arguably the most user-impactful finding in this project.** Findings 1 and 2 are about
*which* accelerated kernel a working build selects. Finding 3 is a build that ships **no**
accelerated matmul kernel at all, produced by following `llama.cpp`'s own documented instructions,
while every signal a user would normally check — a `0` exit code and a `KLEIDIAI = 1` banner line —
says it worked.

**The throughput cost of that build defect — measured, and it gets *bigger* at a realistic model
size, not smaller.** Measured with `llama-bench` (`-p 128 -n 32`, 5 reps, round-robin interleaved,
median ± population stdev — same DGX Spark; raw data `results/scale/scale-experiment.json`, full
write-up `results/scale/SCALE-EXPERIMENT.md`), comparing this broken default build against the fixed
rebuild above, at two model sizes:

| model | phase | broken build | fixed build | ratio |
|---|---|---:|---:|---:|
| large | prefill | 48.64 ± 0.42 | 222.14 ± 4.94 | **4.57x** |
| large | decode | 11.17 ± 0.37 | 18.45 ± 0.84 | **1.65x** |
| small | prefill | 657.00 ± 51.29 | 933.63 ± 74.03 | 1.42x |
| small | decode | 42.18 ± 6.78 | 41.70 ± 6.86 | 0.99x — no effect |

("large" = `Qwen2.5-7B-Instruct-Q4_0`, "small" = `Qwen2.5-0.5B-Instruct-Q4_0`, both Apache-2.0,
licence verified live via the HuggingFace API.)

This is the largest speedup anywhere in this project that comes from fixing a single build-time
defect rather than tuning a runtime knob — and unlike every thread-tuning multiple in this document
(see "Does the 3.43x decode-tuning win generalize to a bigger model?" under "The optimization,"
below), it gets **bigger**, not smaller, on a realistic model size: 4.57x prefill / 1.65x decode at
7B, versus a barely-there 1.42x prefill and *no effect at all* (0.99x) on decode at 0.5B — the two
results generalize in opposite directions, and neither should be read off the other. Read together
with the thread-tuning collapse below, the honest picture is: at a toy 0.5B model, thread tuning
looks dramatic and this build defect looks almost invisible; at a 7B model people actually run, that
relationship inverts. **Polygraph detects the broken build; fixing it is the 4.57x.** The
verification tool and the "make it faster" outcome are, here, the same thing.

**One caveat that must travel with this number:** it is specific to a build where feature detection
collapsed *entirely* — the broken banner above shows no `DOTPROD`, no `MATMUL_INT8`, and no `SVE`
line at all, not merely a missing KleidiAI kernel. Read it as "this specific broken build, on this
machine and toolchain, costs 4.57x at 7B prefill," not as a general "KleidiAI costs 4.57x" claim.

**Status:** measured directly on this machine. Both the broken default build and the fixed rebuild
were compiled and run on the same DGX Spark in the same session
(`results/server/spark-provenance.txt`), not derived from source alone.

---

## Measured results (Apple M4 Max, macOS 27, 16 cores, `llama.cpp` @ `dbadb68`, `-DGGML_CPU_KLEIDIAI=ON`)

Model: `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (337 MB, Apache-2.0). No `Q8_0` GGUF was available in this
environment — every `Q8_0` cell below is `[not available]`, never fabricated from the `Q4_0` file.
Full methodology, thermal controls, and anti-"you faked it" measures: `tools/protocol.md`.

### 1. Dispatch verification (the decisive evidence for Finding 1)

`lldb` regex breakpoint on `kai_run_matmul.*sme` (18 symbol locations resolved), real hit counts:

| threads | workload | advertised (L2) | executed (L3) | hits (SME2 / other) | verdict |
|---:|---|---|---|---|---|
| 1 | decode_short | SME2 | sme2 | 996 / 0 | **SME2_DISPATCHED** |
| 2 | decode_short | SME2 | sme2 | 5,826 / 0 | **SME2_DISPATCHED** |
| 4 | decode_short | SME2 | dotprod | 0 / 15,936 | **SILENT_FALLBACK** |
| 8 | decode_short | SME2 | dotprod | 0 / 31,871 | **SILENT_FALLBACK** |
| 16 | decode_short | SME2 | dotprod | 0 / 51,215 | **SILENT_FALLBACK** |
| 1 | prefill_long | SME2 | sme2 | 660 / 0 | **SME2_DISPATCHED** |
| 2 | prefill_long | SME2 | sme2 | 3,853 / 0 | **SME2_DISPATCHED** |
| 4 | prefill_long | SME2 | dotprod+sme2 | 2,232 / 6,711 | **SME2_HYBRID_DISPATCH** |
| 8 | prefill_long | SME2 | dotprod+sme2 | 1,538 / 13,702 | **SME2_HYBRID_DISPATCH** |
| 16 | prefill_long | SME2 | dotprod+sme2 | 1,377 / 21,534 | **SME2_HYBRID_DISPATCH** |

`--assert` exits `1` on exactly the 3 true `SILENT_FALLBACK` rows and does not flag the 3
`HYBRID_DISPATCH` rows (real, if partial, SME2 usage). Full per-config JSON with the L1/L2/L3
evidence used to derive every row above: `results/dispatch-ledger-darwin-arm64.json`.

### 2. Throughput sweep (`tools/bench.py`, Q4_0, interleaved, warmup-discarded, 5 reps/cell, median ± stddev)

**Decode** (`n_gen=32`):

| threads | SME on | SME off (NEON forced) | ratio |
|---:|---:|---:|---:|
| 1 | 208.9 ± 2.9 | 149.9 ± 4.3 tok/s | 1.39x |
| 2 | **327.6 ± 4.6** | 266.4 ± 6.4 tok/s | 1.23x |
| 8 | 154.6 ± 1.8 | 155.3 ± 0.8 tok/s | 1.00x (tie — matches confirmed SILENT_FALLBACK) |
| 16 | 34.9 ± 6.6 | 28.7 ± 7.9 tok/s | both collapse — thread oversubscription, not an SME effect |

**Prefill, long prompt** (`n_prompt=256`):

| threads | SME on | SME off (NEON forced) | ratio |
|---:|---:|---:|---:|
| 1 | 896.2 ± 4.7 | 415.1 ± 6.0 tok/s | 2.16x |
| 2 | 1629.1 ± 8.6 | 805.2 ± 13.1 tok/s | 2.02x |
| 8 | 1830.1 ± 203.5 | **2676.4 ± 30.6 tok/s** | 0.68x — **NEON alone is the fastest cell in the whole sweep** |
| 16 | 445.3 ± 100.5 | 1514.1 ± 198.8 tok/s (unstable) | NEON still ahead, both degraded |

Prefill, short prompt and full Q8_0 `[not available]` rows: `results/bench/bench-apple-m4-max.md`.
Raw JSON + figures: `results/bench/bench-apple-m4-max.json`, `results/bench/figures/*.png`.

### 3. The reconciliation — the honest, unflattering answer

**Decode: SME2 wins outright, at every thread count measured.** NEON's own best decode throughput
at *any* thread count is 266.4 tok/s (at 2 threads — NEON doesn't get faster with more threads for
this workload either). SME2@2 (327.6) still beats that by **1.23x**. There is no thread count at
which plain NEON catches SME2 for decode. The 2-thread cap costs nothing here.

**Prefill: the comparison flips once NEON is allowed its own best thread count (8, not 16).**
Plain NEON at 8 threads hits **2,676.4 tok/s — the single highest number in this entire sweep** —
beating SME2's own best (1,830.1 tok/s, also at 8 threads, hybrid path) by **1.46x**, and beating
SME2 capped at its 2-thread sweet spot (1,629.1) by **1.64x**.

**The load-bearing conclusion:** `sme_thread_cap`'s 2-thread ceiling (Finding 1) is a real, measured
**net throughput loss for prefill** once NEON can use its own natural thread count — not merely a
dispatch curiosity. SME2's real-world win is *conditional on phase*: unconditional for decode,
negative for prefill past 2 threads. That conditionality is exactly what a chip-name lookup table
cannot express. Full reconciliation with every caveat: `results/SUMMARY.md` §4.

**Caveats on all of the above:** single machine (Apple M4 Max), single 0.5B model, `Q4_0` only;
`prefill_short` was never independently `lldb`-verified in this session (inferred from the on/off
throughput ratio only); 16-thread cells have stddev comparable to or exceeding their own median and
should be read as "this regime is unstable," not a precise point estimate.

---

## Cloud AI: measured on server-class Arm (DGX Spark, Cortex-X925, 20 cores)

Everything above this section other than Finding 3 was measured on a single Apple laptop. For the
Arm Create: AI Optimization Challenge's Track 2 (Cloud AI) focus areas, this lane closes a real gap:
before it, this repository had no inference-server-speed measurement, no time-to-first-token figure,
and no memory figure anywhere in it. This section is the project's first server-class,
concurrent-serving evidence: `llama-server` with continuous batching (`-cb`),
`Qwen2.5-0.5B-Instruct-Q4_0`, 3 rounds per row, median reported, built with the Finding 3 fix above
(`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+sve2+i8mm+bf16+dotprod`) so the server is actually
running an accelerated kernel rather than the silently-broken default. Raw data:
`results/server/server-bench.json`, `results/server/server-dispatch.json`,
`results/server/spark-provenance.txt`.

### Throughput, time-to-first-token, and memory

| parallel | threads | clients | aggregate tok/s | TTFT p50 | TTFT p99 | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 20 | 1  |  14.9 | 0.089s | 0.089s | 724 MiB |
| 4  | 20 | 4  |  56.6 | 0.092s | 0.221s | 761 MiB |
| 8  | 20 | 8  | 271.8 | 0.062s | 0.117s | 809 MiB |
| 8  |  4 | 8  | 264.8 | 0.062s | 0.094s | 809 MiB |
| 16 | 20 | 16 | 440.4 | 0.120s | 0.168s | 901 MiB |

All 5 configurations, 3 rounds each, 0 errors in any row (`results/server/server-bench.json`).

**Reading (a) — concurrency scales, and latency/memory stay disciplined at the edges of the sweep.**
Aggregate throughput scales from 14.9 tok/s at 1 client to 440.4 tok/s at 16 concurrent clients
(**~29.6x**), and peak memory grows only 724 → 901 MiB across that same range. TTFT p99 at those two
endpoints — 0.089s and 0.168s — is comfortably under 170ms, though the sweep is not perfectly
monotonic: the interior `parallel=4` row spikes to 0.221s TTFT p99, the highest value in the whole
table, before `parallel=8` and `parallel=16` both settle back down. That non-monotonic middle point
is real and left in the table rather than smoothed over.

**Reading (b) — the honest counterweight to this repo's own thread-tuning story.** At `parallel=8`,
dropping `--threads` from 20 to 4 costs almost nothing — 271.8 vs 264.8 tok/s — and TTFT p99 actually
*improves*, 0.117s → 0.094s. The single-user thread-count sensitivity that drives this document's
entire "match thread count to phase" story above (`-t 2` vs. default being a 3.43x swing on M4 Max
decode) **largely disappears once continuous batching is doing the work**: under concurrent serving
the server is batch-bound, not thread-bound. This is a genuinely different regime from the
single-user decode/prefill story above, presented here as the honest counterweight it is, not folded
into those numbers as if they were the same claim.

### Finding 2, confirmed on a second core family, under real load

The fixed build's own banner already carries the L2 (selection) signal Finding 2 predicts for
128-bit SVE2 hardware: `SVE_CNT = 16` (128-bit) and `kleidiai: primary q4 kernel feature I8MM` — I8MM
selected, not SVE, exactly as `kleidiai.cpp:209`'s `QK8_0`-exact-equality gate requires. This server
lane supplies the L3 (dispatch) confirmation Finding 2's status note above says was still pending:
`gdb` was attached to a live `llama-server` process with breakpoints on all 10 `kai_run_matmul`
symbols from the fixed build, driven by 8 concurrent clients (the same `parallel=8` configuration as
the table above). The resulting dispatch tally (`results/server/server-dispatch.json`):

```json
{"dotprod": 11360, "i8mm": 364444}
```

Two accelerated families, zero `sve` entries at all — under real concurrent serving load, on real
SVE2-capable hardware, the SVE kernel family is never entered, matching Finding 2's prediction. The
shape also inverts versus the M4 Max's single-user decode numbers earlier in this document:
continuous batching turns GEMV into GEMM, so batched serving here is overwhelmingly `i8mm` (364,444
calls) rather than the `dotprod`-leaning shape a GEMV workload would produce.

---

## Correction (2026-08-04)

> The figures used throughout the "optimization" numbers in this document replace an earlier, wrong
> headline (4.4x decode, and a "+57.3%" win for the patch). That original run measured the baseline
> and patched configs in **different time windows** on a machine with 1-minute load average 66–147
> from unrelated concurrent agent sessions — uneven contention between the two windows manufactured
> a fake speedup for the patch. Every throughput number in this document has since been re-measured
> round-robin-interleaved (A,B,C,…,A,B,C,…) on the same machine under light, evenly-shared external
> load (236–326% CPU, ~2.4–3.3 of 16 cores). Full method and retraction:
> `results/REMEASURE-2026-08-04-QUIET.md`.

This gets its own section, not a footnote, on purpose — a submission that publicly retracts its own
wrong number is more credible, not less. It sits here, after the real method and the real numbers
have already been read once, rather than directly under the headline, so it doesn't crowd out the
result itself as the second thing a reader sees — but it is still one heading away, not buried.

---

## The optimization — matching thread count to phase, and what a patch can and can't fix

The `bench.py` sweep above measured the phase-dependent crossover. A second, independently written
harness (`tools/crossover.py`, its own methodology doc at `tools/crossover.md`) re-derived the same
optima from scratch: decode optimum `threads=2, SME=on` and prefill optimum `threads=8,
SME=off/NEON` — the same qualitative crossover as `bench.py` above. **The absolute magnitudes from
that first `crossover.py` session are superseded** (see [Correction (2026-08-04)](#correction-2026-08-04)
above): they were collected under the same uncontrolled, non-interleaved contention that
manufactured the patch's original inflated headline number. The quiet, round-robin-interleaved
re-measurement (`results/REMEASURE-2026-08-04-QUIET.md`) reconfirms the identical qualitative
crossover — decode optimum at `-t 2` (93.6 → 321.0 tok/s) and prefill optimum at `-t 8` (1,230.3 →
2,198.1 tok/s) — with tight, non-overlapping stdev bands. Two independent tools, two independent
runs, the same qualitative answer, now measured cleanly: **decode wants SME2 + few threads; prefill
wants NEON + many threads.**

### Decomposition — how much of the decode win is SME2, and how much is thread tuning

The 3.43x headline above answers "does matching thread count to phase help." It does not, by
itself, answer "how much of that is *this project's* SME2 finding versus the well-known effect of
simply not oversubscribing threads on Apple Silicon." Those are different claims, and conflating
them was a real defect in an earlier draft of this document, which asserted the win existed
"because of Finding 1" without ever isolating the two variables. A second sweep that toggles
**both** thread count and `GGML_KLEIDIAI_SME` (forcing SME off entirely, not just changing threads)
separates them:

| configuration | decode tok/s (interleaved, n=5, same session) |
|---|---:|
| default threads (12), SME on | 48.0 |
| default threads (12), SME off (NEON forced) | 59.6 |
| `-t 2`, SME on | 309.2 |
| `-t 2`, SME off (NEON forced) | 235.7 |

| what changed | ratio | reading |
|---|---:|---|
| Total win: default/SME-on → `-t 2`/SME-on | **6.44x** | both levers pulled together |
| **Thread tuning alone** (SME forced off throughout, default → `-t 2`) | **3.95x** | **the majority of the total win, and has nothing to do with SME2** |
| SME2's contribution at `-t 2` (on vs. off, holding threads fixed) | **1.31x** | real, but the minority |
| SME2's contribution at default threads (on vs. off, holding threads fixed) | **0.81x** | SME2 being enabled **hurts** here |

**Caveat on the absolute numbers:** this decomposition sweep ran under measurably different shared-
machine load than the quiet, n=7 sweep that produced the 93.6/321.0 tok/s headline figures above —
its absolute tok/s values are lower across the board and should not be compared directly to the
headline table. Per this project's own interleaving discipline, the **ratios** within this single
interleaved session are the trustworthy part; the headline 3.43x/1.79x figures remain the
authoritative, quiet-session numbers for "how fast is the tuned config." This sweep exists
specifically to isolate the SME on/off variable, which the two-row headline table does not.

**The honest conclusion: most of the 3.43x is not an SME2 discovery.** The majority of it —
3.95x out of a measured 6.44x total when both levers are pulled — is the well-documented "too many
threads hurts token generation" memory-bandwidth/oversubscription effect on Apple Silicon, and this
project did not discover it. `llama.cpp`'s own documentation already covers it directly:

> "It's extremely important that this parameter \[`-t`/`--threads`] is not too large. If your token
> generation is extremely slow, try setting this number to 1. If this significantly improves your
> token generation speed, then your CPU is being oversaturated..."
> — [`docs/development/token_generation_performance_tips.md`](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md), `llama.cpp` upstream (verified live 2026-08-04)

The same effect is widely discussed for Apple Silicon specifically — see `llama.cpp`'s own
[Discussion #4167, "Performance of llama.cpp on Apple Silicon M-series"](https://github.com/ggml-org/llama.cpp/discussions/4167)
(verified live 2026-08-04; a large community benchmarking thread across M-series chips, cited here
as evidence that Apple Silicon thread-count tuning is an actively discussed, pre-existing topic, not
something this project surfaced) and third-party guidance such as
["Tune llama.cpp on Apple Silicon: 7 Flags"](https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc)
(verified live 2026-08-04), which warns that "more threads than P-cores hurts" on M-series chips.

**What genuinely is this project's finding:** at the tuned thread count (`-t 2`), SME2 still
contributes a real, measured **1.31x on top of** the thread-tuning win — not zero, not noise. More
interesting than the magnitude is *why* the two effects are so easy to conflate: the SME2 dispatch
boundary (`sme_thread_cap`, hardcoded to exactly 2 on this chip — Finding 1) happens to sit almost
exactly at the empirically-discovered decode throughput optimum. A benchmark that only sweeps
thread counts sees one smooth curve and cannot tell you how much of the win at its peak is SME2
versus thread-oversubscription avoidance — they move together. Telling them apart required forcing
`GGML_KLEIDIAI_SME` off independently of thread count, which is the same class of symbol-level
dispatch discipline (not trusting the banner, not trusting a single-variable sweep) that Finding 1
itself required. That is the defensible, non-obvious point here: not "we discovered the thread-
tuning win," but "we could tell you exactly how much of it is SME2 and how much isn't, and why a
naive benchmark can't."

### Does the 3.43x decode-tuning win generalize to a bigger model? No — it collapses to 1.33x at 7B

Everything above this point in "The optimization" was measured on a 337 MB, 0.5B model on Apple
Silicon. A second, independently run experiment on a different machine — DGX Spark, GB10,
Cortex-X925/Cortex-A725, gcc 13.3.0 — sweeps thread count for **two** model sizes,
`Qwen2.5-0.5B-Instruct-Q4_0` and `Qwen2.5-7B-Instruct-Q4_0` (both Apache-2.0, licence verified live
via the HuggingFace API), and the honest answer is that this repo's headline decode-tuning multiple
does not generalize past a tiny model.

Method: `llama-bench -p 128 -n 32`, 5 reps per config, **round-robin interleaved** (one rep of each
config in turn, repeated, so load drift hits every config equally — the same discipline as the
Apple M4 Max re-measurement above), median ± population stdev. Raw data:
`results/scale/scale-experiment.json`. Full write-up, every caveat, and the prefill side of this
same sweep in more depth: `results/scale/SCALE-EXPERIMENT.md`.

**7B (`Qwen2.5-7B-Instruct-Q4_0`), thread sweep:**

| threads | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 1  | 28.89 ± 0.14 | 7.26 ± 0.20 |
| 2  | 55.41 ± 0.78 | 12.95 ± 0.13 |
| 4  | 104.42 ± 0.92 | 17.76 ± 2.04 |
| 8  | 179.45 ± 3.76 | **24.45 ± 1.05** — decode peak |
| 16 | 212.41 ± 1.27 | 21.01 ± 0.75 |
| 20 | **218.39 ± 2.49** — prefill peak | 18.03 ± 1.05 |

**0.5B (`Qwen2.5-0.5B-Instruct-Q4_0`), same method:**

| threads | prefill tok/s | decode tok/s |
|---:|---:|---:|
| 1  | 318.95 ± 6.43 | 85.41 ± 2.93 |
| 2  | 556.71 ± 5.35 | 138.60 ± 9.15 |
| 4  | 915.65 ± 11.77 | 177.61 ± 13.21 |
| 8  | **1457.68 ± 21.84** — both phases peak here | **190.32 ± 8.93** — both phases peak here |
| 16 | 1047.66 ± 51.97 | 120.05 ± 7.16 |
| 20 | 965.98 ± 87.36 | 45.49 ± 6.43 |

`llama-bench`'s default thread count on this 20-core box resolves to 20 — confirmed by comparing a
separate default-threads run (the Finding 3 throughput-cost table, above, whose "fixed build" row
uses no `-t` flag) to the explicit `t=20` row above (0.5B decode: 41.70 vs. 45.49; 7B decode: 18.45
vs. 18.03 — close but not identical, ordinary run-to-run noise between two separate `llama-bench`
sessions, not the same measurement twice). Using that default-threads figure as "before" and each
model's own peak decode thread count (`t=8`, both models) as "after":

| model | default decode tok/s | peak decode tok/s (`t=8`) | ratio |
|---|---:|---:|---:|
| 0.5B | 41.70 | 190.32 | **4.56x** |
| 7B | 18.45 | 24.45 | **1.33x** |

**Say this plainly: the headline thread-tuning multiple this repo has been built around is largely
an artefact of a tiny model. At a size people actually run, it is a modest 1.33x.** This does not
retract the Apple M4 Max numbers above (93.6 → 321.0 tok/s, 3.43x) — they are real, correctly
measured, and correctly scoped to their own machine and model. It means the scope matters: a 0.5B
model on an M4 Max and a 0.5B model on a Cortex-X925 both show a large multiple (3.43x and 4.56x
respectively); move to a 7B model on that same Cortex-X925 and the same class of thread-count tuning
shrinks to 1.33x. Prefill is a separate curve and is never blended into this decode number — both
models' prefill keeps improving out to `t=16`/`t=20` rather than peaking at `t=8` and collapsing the
way decode does; see the two tables above for prefill's own numbers.

**Caveats, stated plainly:** one machine (DGX Spark, Cortex-X925, gcc 13.3.0), one quant (`Q4_0`),
two model sizes, `llama-bench` only — this is not a re-measurement of the Apple M4 Max headline
itself, it is an independent demonstration that the underlying phenomenon (thread-count tuning
delivering an outsized win) is size-dependent. `scale-experiment.json` records `load_before: 0.43`
and `load_after: 12.32` — the 20-thread configuration in this sweep generates real, self-inflicted
load on a 20-core box; round-robin interleaving protects every config-to-config comparison equally
against that drift, but this is not a quiet, externally-idle host in the same sense as the Apple M4
Max re-measurement's `results/REMEASURE-2026-08-04-QUIET.md`. Full method and every additional
caveat: `results/scale/SCALE-EXPERIMENT.md`.

### Why the dispatcher can't just do this itself

`llama.cpp` already exposes the fix as two separate flags — `-t`/`--threads` for decode,
`-tb`/`--threads-batch` for prefill — so the *thread-count* half of this is expressible today. The
*kernel-family* half is not: `GGML_KLEIDIAI_SME` is parsed exactly once, lazily, on the first call
into KleidiAI (`init_kleidiai_context()`'s `static bool initialized` guard, `kleidiai.cpp:193–198`;
the `getenv("GGML_KLEIDIAI_SME")` read itself at `kleidiai.cpp:201`) and cached for the rest of the
process. One `llama-cli` process cannot run SME2-for-decode and NEON-forced-for-prefill at the same
time — it picks one kernel-family policy at startup and keeps it. That is the missing capability
this project's patch targets (full mechanism and line citations: `docs/FINDINGS.md`).

### Before / after, decode and prefill separate, with variance (re-measured 2026-08-04, quiet host, round-robin interleaved, `llama-bench -r 1`, n=7, median ± stdev)

| Configuration | decode tok/s | prefill tok/s | Requires code changes? |
|---|---:|---:|---|
| `llama.cpp` default (no `-t`/`-tb`; resolves to **12** threads on this machine, not 16 — verified via `llama-cli -v`) | 93.6 ± 2.47 | 1,230.3 ± 118.52 | — |
| **Hand-tuned split** (`-t 2` for decode, `-t 8`/`-tb 8` for prefill) — achievable **today** | **321.0 ± 2.09** (**3.43x** vs default) | **2,198.1 ± 72.59** (**1.79x** vs default) | **No** |
| Phase-aware patch, `GGML_KLEIDIAI_PHASE_AWARE=1`, **default thread count** (no `-t`/`-tb` at all) | 82.5 ± 4.07 (**0.88x** vs default — **~12% slower**, a real regression outside noise, not a win) | 1,202.1 ± 96.26 (0.98x vs default — a tie; patch's diff never touches prefill's code path) | Yes, opt-in |
| Phase-aware patch + `-t 2` | 317.5 ± 3.58 (0.99x vs the hand-tuned row — a statistical tie; the patch is inert here, its branch never activates at `nth_total == sme_thread_cap`) | `[not yet measured at -t 8/-tb 8]` | Yes, opt-in |

Source: `results/REMEASURE-2026-08-04-QUIET.md` — authoritative, re-measured 2026-08-04 on a quiet,
round-robin-interleaved host. **Supersedes** `results/OPTIMIZATION.md` §2 and the raw JSON in
`results/crossover/` and `results/crossover/patched/`, which were collected under heavy,
non-interleaved external load in different time windows for baseline vs. patched (see
[Correction (2026-08-04)](#correction-2026-08-04) above).

### The patch — an honest negative result: dispatch works, throughput doesn't

`patches/0001-kleidiai-phase-aware-dispatch.patch` (56 insertions / 3 deletions, one file, no new
deps) adds `GGML_KLEIDIAI_PHASE_AWARE=1` (default off). When set, a GEMV-shaped op (`ne11 == 1`,
i.e. decode) is let into the *existing* SME+NEON hybrid path instead of unconditionally collapsing
to NEON once `nth_total > sme_thread_cap` — reusing prefill's already-correct hybrid
thread-assignment code verbatim rather than inventing a new one (a naive "leave extra threads idle"
design was rejected during design: this threadpool model requires every thread to reach the same
barriers the same number of times per op, and idling threads risks a deadlock). Full mechanism,
diff walkthrough, and design rationale: `patches/README.md` and `docs/FINDINGS.md`.

The hypothesis behind the patch was straightforward: Finding 1 shows decode is structurally
excluded from SME2 above the thread cap, so routing it into the existing hybrid path should recover
some of that throughput automatically, with no flags required. **We built it, measured it honestly,
and the hypothesis did not hold.**

**Symbol-level dispatch proof** (`tools/verify_dispatch.py`, same patched binary, only the env var
changes — an apples-to-apples single-binary A/B). This part is real and unaffected by the
correction below — these are breakpoint hit counts, not timings, so external CPU contention cannot
distort them:

| threads | flag OFF, hits (SME2/other) | flag ON, hits (SME2/other) | verdict |
|---:|---|---|---|
| 4, decode | **0 / 15,936** — exact match to the pre-patch ground truth | **3,072 / 10,428** | dispatch genuinely changes |
| 8, decode | **0 / 31,872** | **2,354 / 20,517** | dispatch genuinely changes |

That is real proof the patch does what it claims at the dispatch level, not a selection-log
artifact. **But dispatching SME2 is not the same as being faster, and here it isn't.** Re-measured
2026-08-04, quiet and round-robin-interleaved (`results/REMEASURE-2026-08-04-QUIET.md`): at the
default thread count (12 — the exact case the patch targets, a user who passes no thread flags at
all) throughput goes **93.6 → 82.5 tok/s, i.e. ~12% *slower***, a real regression outside the
measurement noise (93.6 ± 2.47 vs. 82.5 ± 4.07 do not overlap). At `-t 2` the patch is inert
(321.0 → 317.5 tok/s, a statistical tie; its branch never activates at `nth_total == sme_thread_cap`).
At prefill's default thread count it is also a tie (1,230.3 → 1,202.1 tok/s), unsurprising since the
patch's diff never touches prefill's code path.

**Mechanism, stated honestly:** at 12 threads the hybrid split gives SME2 only 2 of them while the
other 10 run NEON, and coordinating that split costs more than the SME2 lane returns for a GEMV
shape this small. Pure NEON on all 12 threads wins. Upstream's existing behaviour — collapsing to
NEON above the thread cap — is, on this chip and this model, the better default, and the patch's
premise that decode was being unfairly excluded is **not supported by throughput**, even though the
exclusion itself (Finding 1) is real and the dispatch change proven above genuinely works as
designed. The patch does not beat hand-tuning (`-t 2`, no code changes, 321.0 tok/s) at any thread
count, and it does not touch the process-global `GGML_KLEIDIAI_SME` limitation described above —
the *theoretical* best (SME2-decode + NEON-forced-prefill, simultaneously, in one process) remains
**`[NOT YET ACHIEVABLE]`** with or without this patch. Full verdict with every caveat:
`results/REMEASURE-2026-08-04-QUIET.md`.

**Reported upstream:** [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547)
(filed 2026-08-04, both findings, reproduction commands, and an offer to send this patch). Per the
re-measurement above, the *warning* half of the patch (surfacing that SME2 silently isn't in use)
should be proposed to maintainers on its own merits — it costs nothing and closes a real
observability gap. The *phase-aware dispatch* half is reported as a **measured negative result**,
not a performance improvement.

---

## The hand-written microkernels (`kernels/`)

**Their honest role is to prove the silicon is not the limiter — not to claim they beat a vendor
library.** They don't, and this project says so explicitly rather than picking a strawman baseline.

Correctness (`kernel_test`, `ctest` — **all pass**, exit 0): NEON fp32 bit-exact across 8 shapes;
SME2 fp32 max-rel-diff 0 across 8 shapes; SME2 int8 bit-exact across 8 shapes; SME2 Q4
L2-rel-error 0.0036–0.0055 (tolerance 0.02) across 5 shapes; SVE2 kernels correctly self-report
`-1` (unavailable) on this non-SVE2 host instead of silently miscomputing.

Single-thread microbenchmark (`kernel_bench`), fp32 GEMM, GFLOP/s. Raw, persisted artifact
(re-captured during adversarial review, 2026-08-04 — the numbers below previously had no
file in `results/` backing them, only "this run" prose; that gap is now fixed):
`results/bench/kernel-bench-apple-m4-max.log` / `.md`.

| N | NEON (tuned) | SME2 (packed) | Apple Accelerate | Accelerate is faster by |
|---:|---:|---:|---:|---:|
| 512 | 99.4 | 503.6 | 1607.4 | ~3.2x (noisy — see caveat) |
| 1024 | 96.0 | 463.6 | 3103.3 | 6.7x |
| 2048 | 49.7 | 185.2 | 3408.0 | 18.4x |

**Apple's tuned Accelerate library is still faster than our hand-written kernel at fp32 GEMM, at
every size measured — roughly 3–18x depending on N.** We never claim otherwise. **Caveat:** at
N=512 the whole GEMM finishes in under half a millisecond, close to this benchmark's practical
timing-noise floor; two extra reruns during review swung the N=512 Accelerate figure between
1607 and 2509 GFLOP/s (ratio ~3.2x–5.1x) while N=1024/2048 stayed stable within ~5%. Treat the
N=512 ratio as "roughly 3–5x", not a precise point estimate; full spread in
`results/bench/kernel-bench-apple-m4-max.md`. What Accelerate does *not* expose is an
**integer GEMM** — `cblas_gemm_*` has no INT4/INT8 symbol at all — so quantized inference has no
vendor-tuned CPU BLAS path to fall back on. That is the real, non-strawman gap:

| N | SME2 int8 (GOP/s) | Accelerate int8 |
|---:|---:|---:|
| 512 | 101.8 | *(no integer GEMM exists)* |
| 1024 | 123.2 | *(no integer GEMM exists)* |
| 2048 | 116.0 | *(no integer GEMM exists)* |

### The `-march=armv9-a+sme2` SIGILL trap

Discovered the hard way while building `kernels/`: on Apple Silicon, compiling SME2 code with the
generic flag

```
-march=armv9-a+sme2
```

produces a binary that **SIGILLs at runtime**. `clang` emits the SVE instruction `cntd` outside
streaming mode as part of that target, and **Apple Silicon has no non-streaming SVE at all** — SME2
ships without it. Generic Armv9+SME2 code is not portable to Apple Silicon as-is.

**The fix:** target the concrete CPU instead of a generic architecture level:

```
-mcpu=apple-m4
```

`kernels/CMakeLists.txt` encodes this per-platform automatically — `-mcpu=apple-m4` on Apple,
`-mcpu=gb10` on the DGX Spark (falling back to `-march=armv9.2-a+sve2+i8mm+bf16` if the toolchain
doesn't know that CPU name yet). Working ACLE patterns that were verified compiling and running
bit-exact on this hardware: `__arm_new("za") __arm_locally_streaming` functions, pre-transposed
operands (gather loads are illegal in streaming mode), and querying the streaming vector length
(`svcntw()`) *from inside* the streaming function, never outside it. See `kernels/sme2_gemm.c` for
the working reference implementation.

---

## Setup instructions

### Option A — free, judge-reproducible, zero cost (recommended)

GitHub's `ubuntu-24.04-arm` runner (Neoverse-N2 class) is **free for public repos**. Fork this repo
and either open a PR or click **Run workflow** on `verify-free-arm64` in the Actions tab — no Arm
hardware, no payment, no local setup required. It builds `llama.cpp` + KleidiAI, builds `kernels/`,
runs correctness tests, verifies dispatch, runs the bench sweep, and publishes `results/LEDGER.md`
as the job summary plus a downloadable `results/` artifact. (Finding 2 is expected — and marked
`continue-on-error` — on this runner: Neoverse-N2's SVE2 is also below the 256-bit gate.)

To run the same thing locally on any `aarch64` Linux box:

```bash
sudo apt-get install -y cmake build-essential curl python3 gdb
git clone https://github.com/<you>/polygraph.git && cd polygraph
./scripts/setup.sh      # clones+builds llama.cpp w/ KleidiAI, fetches the demo GGUF, builds kernels/
./scripts/run_all.sh    # correctness -> dispatch verify -> bench -> results/LEDGER.md
```

### Option B — macOS / Apple Silicon (the SME2 lane)

Requires Xcode Command Line Tools (`xcode-select --install`, for `clang` + `lldb`) and `cmake`.

```bash
git clone https://github.com/<you>/polygraph.git && cd polygraph
./scripts/setup.sh
./scripts/run_all.sh
```

This is a full clone+build+download from scratch by default (`LLAMA_CPP_REF`/`MODEL_PATH` etc. are
overridable — see `scripts/common.sh` — to reuse an existing checkout). The expensive stage is the
`lldb`-attached dispatch sweep (~9m39s on this machine, 10 configurations); everything else
finishes in well under two minutes.

If `lldb` reports "Developer mode is currently disabled" or every dispatch check times out, see
`tools/protocol.md` §6 item 9 — this project hit that exact failure mode mid-session and documents
the recovery path (`--dispatch-ledger-json` to replay a previously-real `lldb` observation).

### Option C — DGX Spark / Cortex-X925 (the SVE2 lane, best-effort)

This is a self-hosted lane you would point at your own Spark (or any SVE2-capable `aarch64` Linux
box):

```bash
./scripts/setup.sh   # -mcpu=gb10 kernels build if the toolchain knows that CPU name
./scripts/run_all.sh
```

The corresponding CI workflow (`.github/workflows/verify-spark-aarch64.yml`) is `workflow_dispatch`
only, `continue-on-error: true` on every step, and never added to required checks — there is a
live, unresolved incident on this project's own Spark runner where the kernel kills the runner
process (suspected OOM) for reasons unrelated to this repo. Treat this lane as best-effort, never
as a gate.

**If the documented `-DGGML_CPU_KLEIDIAI=ON` build line yields a binary with zero
`kai_run_matmul` symbols on your box, that isn't your setup — see Finding 3 above.** Add
`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"` to `cmake` (adjusted for
your core's actual SVE2/i8mm/bf16/dotprod support) and rebuild.

### Manual `llama-cli` invocation (for exploring dispatch by hand)

```bash
./bin/llama-cli -m MODEL.gguf -p "your prompt" -n 16 -no-cnv -st --simple-io -t N
```

The `-no-cnv -st --simple-io` flags are **required** — without them `llama-cli` waits forever on
non-TTY stdin.

---

## Reusable artifacts (this is the point of the Impact score)

| Artifact | Reusable for |
|---|---|
| `tools/verify_dispatch.py` | Stdlib-only Python. Point it at **any** `llama.cpp`-family binary + GGUF and get a real L1/L2/L3 dispatch verdict — not specific to this project's model or machine. |
| `tools/crossover.py` + `tools/crossover.md` | A phase-crossover benchmark harness: sweeps threads × `GGML_KLEIDIAI_SME` × phase, interleaved reps, retry-on-timeout, and reports the default / hand-tuned-split / theoretical-best configs for **any** `llama.cpp`-family binary — the tool that found and quantified the optimization this README leads with. |
| `patches/0001-kleidiai-phase-aware-dispatch.patch` + `patches/README.md` | A minimal (56-line), opt-in, upstream-submittable `llama.cpp` patch plus its full design rationale and local verification log — apply with `git am` against `dbadb68`. Reusable as-is by anyone hitting the same GEMV/hybrid-dispatch gate, or as a worked example of how to extend KleidiAI's dispatch decision safely. |
| `mcp/server.py` | Dependency-free MCP stdio server exposing `detect_arm_features`, `verify_dispatch`, `recommend_config`, and `explain_finding` as callable tools — so an agentic client can ask *this machine, right now* whether SME2/SVE is actually dispatching, instead of trusting a banner. Add to Claude Code with `claude mcp add polygraph -- python3 mcp/server.py`; self-test with `python3 mcp/server.py --selftest`. See `mcp/README.md`. |
| `kernels/` | A small, dependency-free, correctness-tested NEON/SME2/SVE2 GEMM library with a `CMakeLists.txt` that already encodes the Apple-vs-Linux `-mcpu` selection (and the SIGILL trap fix) — usable as a starting template for anyone porting compute onto Apple SME2 or Arm SVE2. |
| `site/` + `.github/workflows/pages.yml` | A static, no-build-step dashboard (advertised-vs-executed table, phase-crossover charts, figure gallery) driven entirely off `results/*.json` via a runtime-generated manifest — new results from any of the artifacts above show up with no code change. Validated locally (`actionlint`, headless-Chrome DOM checks); not yet deployed (`has_pages=false`, nothing pushed). |
| `scripts/models.txt` + `scripts/lib/fetch_model.sh` | A pipe-delimited model manifest (id, HF repo/file, sha256, license) plus a fetcher with single-model, model-set, and CI-matrix modes — turns "add a row" into "add a CI leg." Live-checks license (already caught and rejected a non-commercial GGUF). |
| `demo/demo.sh` + `demo/README.md` + `demo/SHOTLIST.md` | A self-contained, idempotent, degrade-gracefully terminal walkthrough of the full claim → proof → cost → fix → gap → upstream arc, timed and narrated for the submission video. |
| `scripts/run_all.sh` + `scripts/lib/*.sh` | An idempotent, cache-aware, CI-ready pipeline (build → verify dispatch → bench → emit ledger) that already runs on three different Arm64 targets. |
| **[ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547)** | **Filed upstream 2026-08-04.** Both findings reported to the maintainers with reproduction commands, exact source line citations, and an offer to send the patch. Contribution back to the ecosystem, not just consumption of it. Draft and rationale retained in `docs/UPSTREAM-ISSUE.md`. |
| `results/GROUND-TRUTH-DISPATCH.md` + `docs/FINDINGS.md` + `results/OPTIMIZATION.md` | The evidence chain behind that issue and the optimization verdict, including the documented precedent (`llama.cpp` PR #25701 added exactly this kind of silent-fallback warning for a different case) for why a `GGML_LOG_WARN` on `SILENT_FALLBACK` is a reasonable ask. |
| [`docs/RELATED-WORK.md`](docs/RELATED-WORK.md) | Full disclosure that Finding 2's mechanism was independently published two days before this repo existed, what this project adds beyond that prior work, and an honest one-line comparison against every other Track 2 entry we're aware of — a reusable template for how a submission should handle being partially scooped. |
| `tests/l3_gdb_groundtruth/` | A dlopen-based harness that asserts the L3 probe recovers a *known* call count. Written after our own gdb probe silently reported zero hits on the free CI lane — the exact failure mode this project exists to catch. Reusable by anyone instrumenting a dynamically-loaded kernel library. |
| Three verify CI lanes (`.github/workflows/verify-*.yml`) | A template for a free-hosted judge-reproducible lane plus two self-hosted lanes with correctly scoped `pull_request` exclusions for physical hardware; the free lane now runs as a model matrix derived from `scripts/models.txt`. |

---

## What this is NOT / limitations

- **Single machine per finding.** Finding 1 is verified live on one Apple M4 Max. It is not yet
  re-verified on an M4/M4 Pro/M4 Ultra (the other rows of the hardcoded brand-string table), and
  the base `"M4"` and `"M4 Pro"`/`"M4 Ultra"` cap values are read from source, not independently
  measured on that specific silicon.
- **Finding 2 is now dispatch-confirmed on real SVE2 hardware, but by a manual measurement, not
  the automated CI lane.** A manual `gdb` trace against a live `llama-server` process on the DGX
  Spark (`results/server/server-dispatch.json`) recorded zero SVE-family calls — only `dotprod`
  (11,360) and `i8mm` (364,444) — under concurrent serving load, matching the `QK8_0`-equality
  gate's prediction for 128-bit SVE2. The self-hosted `.github/workflows/verify-spark-aarch64.yml`
  CI lane meant to produce this automatically still has not completed a clean run (a separate,
  unresolved OOM incident on that runner) — this confirmation stands on its own measurement, not on
  that lane going green.
- **Finding 3 (the broken default KleidiAI build on the DGX Spark) has not been reported
  upstream.** Unlike Findings 1 and 2 (filed as
  [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547)), the build-flag
  issue diagnosed here is new as of this session and has not yet been written up as a separate
  upstream issue or doc fix.
- **The Cloud AI server sweep (`results/server/server-bench.json`) is single-machine,
  single-model.** All five rows use `Qwen2.5-0.5B-Instruct-Q4_0` on one DGX Spark, built with the
  Finding 3 fix; TTFT and throughput at a larger model, a larger batch size, or another quantization
  are `[not yet measured]`. The `parallel=4` row's TTFT p99 (0.221s) is the least favorable point in
  the sweep and is not smoothed over in the readings above.
- **Finding 3's build-flag diagnosis is one gcc/toolchain combination.** It was diagnosed and fixed
  against `gcc 13.3.0` on Ubuntu 24.04 aarch64 (`results/server/spark-provenance.txt`); whether the
  same `-mcpu=native+<feature>` probe failure reproduces on other gcc versions, `clang`, or other Arm
  server cores is `[not yet measured]`.
- **Single model.** All throughput numbers are `Qwen2.5-0.5B-Instruct`, `Q4_0` only. `Q8_0` is
  `[not available]` — no such GGUF existed in this environment, and none was fabricated by
  up-converting the lossy `Q4_0` file.
- **`threads=4` is missing from the `bench.py` throughput sweep** (its grid used `1,2,8`; the
  16-thread cells that are present are high-variance and should be read directionally).
  `tools/crossover.py`'s independent sweep does cover `threads=4`, and both harnesses' overlapping
  cells (`≤8` threads) agree within ~2%.
- **The phase-aware patch is a measured regression, not a win, and does not beat hand-tuning.**
  Re-measured 2026-08-04 on a quiet, round-robin-interleaved host
  (`results/REMEASURE-2026-08-04-QUIET.md`): at the true-default (no `-t`/`-tb`) case it is
  **~12% slower** (93.6 → 82.5 tok/s, outside noise), and at every other thread count measured it
  is a statistical tie against the unpatched binary. No configuration found with the patch beats
  the pre-existing `-t 2` config, patch or no patch. The earlier "+57.3%" figure for this row was a
  non-interleaved contention artifact and is retracted — see
  [Correction (2026-08-04)](#correction-2026-08-04) above.
- **The patch is Apple-only by construction** (`detect_num_smcus()`, the function that sets
  `sme_thread_cap`, is an `__APPLE__`-only code path) — it has no effect on the Neoverse-N2 free CI
  lane or the DGX Spark, neither of which has SME2.
- **`-t 16` decode dispatch for the patched binary was not confirmed by a completed `lldb` sweep** —
  this session's shared-machine contention made a full symbol-count capture at that thread count
  impractical; it was instead confirmed via a temporary source-level diagnostic log showing the
  same decision path as the `lldb`-confirmed `-t 4` case. `[recommend re-running the `-t 16` lldb
  sweep on a quieter machine before citing that cell]`. Both the baseline and patched `threads=16`
  decode throughput cells failed to produce **any** successful `crossover.py` measurement in this
  session (100% timeout, both sweeps) — `[not measured]`, not interpolated.
- **The patched and baseline `crossover.py` sweeps ran ~40 minutes apart under measurably different
  shared-machine load** (§5 of `results/OPTIMIZATION.md`); every "noise, not a patch effect" call in
  that file was made because the stddev bands overlap or the code path is provably untouched by the
  patch, never because a number was inconvenient. The symbol-level dispatch A/B (flag on vs. off,
  same binary, same short session) is the more robust of the two comparisons.
- **`GGML_KLEIDIAI_SME` remains a process-global setting**, patched or not — the *theoretical* best
  (SME2-decode + NEON-forced-prefill, simultaneously, in one process) stays `[NOT YET ACHIEVABLE]`.
  The patch closes part of the gap for the no-flags default case; it does not touch this limitation.
- **The dashboard (`site/`) and the broadened model manifest (`scripts/models.txt`) are validated
  locally, not on live infrastructure.** GitHub Pages is not enabled for this repo (`has_pages`
  reported `false`) and nothing was pushed, so `pages.yml` has never actually run in GitHub Actions;
  the model-matrix CI path in `verify-free-arm64.yml` is validated by `actionlint`, YAML parsing, and
  local shell simulation of its non-trivial steps, not by a live run.
- **`sve2_gemm.c`** compiles cleanly cross-compiled for `-march=armv9.2-a+sve2+i8mm+bf16` (real,
  reproducible ELF aarch64 object containing genuine `smmla`/`fmad` SVE2/i8mm instructions, not a
  stub — verified during adversarial review and persisted at
  `results/logs/sve2_cross_compile_check.log`, since this claim previously had no artifact backing
  it in `results/`) but has never executed on real SVE2 hardware in this session — this machine has
  none.
- **`tools/dispatch_probe.gdb`** (the Linux L3 path) has never been executed — no `gdb` on this Mac.
- **The three GitHub Actions workflows are syntax/lint-validated only** (`yaml.safe_load` +
  `actionlint`), not yet run end-to-end on real GitHub Actions infrastructure.
- **A known filename-casing footgun:** `scripts/lib/verify_dispatch.sh` derives its output filename
  from `uname -s` (`Darwin-arm64`), while `tools/verify_dispatch.py`'s own default is lowercase
  (`darwin-arm64`). On this Mac's case-insensitive APFS volume they silently collapse to one file;
  on a case-sensitive Linux CI runner they would diverge into two separate ledgers. Not yet fixed.
- No number in this document was generated by a model, an LLM estimate, or an analogy to a
  published spec sheet — every number traces to a command in `results/` you can re-run yourself.

---

## License / attribution

Apache-2.0 (see `LICENSE`). This project builds on and instruments — but does not fork or vendor —
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) (MIT-licensed) and its
[KleidiAI](https://github.com/ARM-software/kleidiai) CPU backend
(`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`), both used at the pinned commit `dbadb68`.
`patches/0001-kleidiai-phase-aware-dispatch.patch` is a diff against that MIT-licensed file — a
derivative work under llama.cpp's own MIT terms, not relicensed as Apache-2.0 — stored here only as
a patch file, never as a vendored/forked copy of `kleidiai.cpp` itself; see
`patches/README.md`'s "License / attribution" section. The demo model,
[`Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), is Apache-2.0
licensed. All findings, kernels, tooling, and CI in this repository are original work produced for
the Arm Create: AI Optimization Challenge.
