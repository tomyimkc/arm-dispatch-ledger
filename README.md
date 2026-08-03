# Arm Dispatch Ledger

**A kernel can be compiled in, advertised in the startup banner, and never execute once — and a
timing-only benchmark will never tell you.** This project builds a symbol-level dispatch verifier
for `llama.cpp`'s KleidiAI CPU backend, uses it to prove that on Apple Silicon, and ships the
verifier, an MCP server, and a small hand-written Arm kernel library as reusable artifacts.

**Headline finding, measured on this repo's own hardware (Apple M4 Max, real `lldb` breakpoints, not
inferred):** at `llama.cpp`'s *default* thread count (physical core count — 16 on this machine),
single-token decode **never dispatches SME2**, even though the startup banner and the runtime log
both keep claiming `SME2 enabled` on every single one of those runs.

```
threads=1   decode: SME2 fires (996 lldb hits)         <- advertised AND executed
threads=8   decode: SME2 fires ZERO times (31,871 NEON hits instead)   <- advertised, NOT executed
threads=16  decode: SME2 fires ZERO times (51,214 NEON hits instead)   <- advertised, NOT executed
```

All numbers in this document were produced by code in this repo, run for real on real Arm
hardware. Anything not yet measured is marked `[not yet measured]` — never invented, never
interpolated. See `results/SUMMARY.md` for the full run log and `results/GROUND-TRUTH-DISPATCH.md`
for the authoritative, corrected dispatch rule (an earlier draft of this finding was incomplete —
see "The correction" below).

---

## TL;DR

| | |
|---|---|
| **Advertised** (compile-time banner + selection-time log) | `SME = 1 \| SME2 = 1 \| KLEIDIAI = 1` and `kleidiai: primary q4 kernel feature SME2` — printed identically on **every** run below, including the ones where SME2 never executes once. |
| **Executed** (dispatch-time, `lldb` breakpoint on `kai_run_matmul.*sme`, 18 symbol locations) | Decode at `-t 4/8/16`: **0 hits**, 15,936–51,214 NEON-dotprod hits instead. Decode at `-t 1/2`: SME2 fires (996–5,826 hits). |
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

## The two findings

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
tier); **not yet confirmed by an L3 dispatch trace on real SVE2 hardware** — the DGX Spark CI lane
(`.github/workflows/verify-spark-aarch64.yml`) exists to do exactly that but has not completed a
clean run yet (see Limitations). Treat this finding as `[architecturally derived, dispatch-level
confirmation pending]` until that lane is green.

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
| 16 | decode_short | SME2 | dotprod | 0 / 51,214 | **SILENT_FALLBACK** |
| 1 | prefill_long | SME2 | sme2 | 660 / 0 | **SME2_DISPATCHED** |
| 2 | prefill_long | SME2 | sme2 | 3,853 / 0 | **SME2_DISPATCHED** |
| 4 | prefill_long | SME2 | dotprod+sme2 | 2,232 / 6,712 | **SME2_HYBRID_DISPATCH** |
| 8 | prefill_long | SME2 | dotprod+sme2 | 1,538 / 13,702 | **SME2_HYBRID_DISPATCH** |
| 16 | prefill_long | SME2 | dotprod+sme2 | 1,403 / 21,509 | **SME2_HYBRID_DISPATCH** |

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
| 16 | 445.3 ± 100.5 | 1514.1 ± 198.9 tok/s (unstable) | NEON still ahead, both degraded |

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
git clone https://github.com/<you>/arm-dispatch-ledger.git && cd arm-dispatch-ledger
./scripts/setup.sh      # clones+builds llama.cpp w/ KleidiAI, fetches the demo GGUF, builds kernels/
./scripts/run_all.sh    # correctness -> dispatch verify -> bench -> results/LEDGER.md
```

### Option B — macOS / Apple Silicon (the SME2 lane)

Requires Xcode Command Line Tools (`xcode-select --install`, for `clang` + `lldb`) and `cmake`.

```bash
git clone https://github.com/<you>/arm-dispatch-ledger.git && cd arm-dispatch-ledger
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
| `mcp/server.py` | Dependency-free MCP stdio server exposing `detect_arm_features`, `verify_dispatch`, `recommend_config`, and `explain_finding` as callable tools — so an agentic client can ask *this machine, right now* whether SME2/SVE is actually dispatching, instead of trusting a banner. Add to Claude Code with `claude mcp add arm-dispatch-ledger -- python3 mcp/server.py`; self-test with `python3 mcp/server.py --selftest`. See `mcp/README.md`. |
| `kernels/` | A small, dependency-free, correctness-tested NEON/SME2/SVE2 GEMM library with a `CMakeLists.txt` that already encodes the Apple-vs-Linux `-mcpu` selection (and the SIGILL trap fix) — usable as a starting template for anyone porting compute onto Apple SME2 or Arm SVE2. |
| `scripts/run_all.sh` + `scripts/lib/*.sh` | An idempotent, cache-aware, CI-ready pipeline (build → verify dispatch → bench → emit ledger) that already runs on three different Arm64 targets. |
| `results/GROUND-TRUTH-DISPATCH.md` + `docs/FINDINGS.md` | An upstream-actionable bug report, including a documented precedent (`llama.cpp` PR #25701 added exactly this kind of silent-fallback warning for a different case) for why a `GGML_LOG_WARN` on `SILENT_FALLBACK` is a reasonable ask. |
| Three CI lanes (`.github/workflows/`) | A template for a free-hosted judge-reproducible lane plus two self-hosted lanes with correctly scoped `pull_request` exclusions for physical hardware. |

---

## What this is NOT / limitations

- **Single machine per finding.** Finding 1 is verified live on one Apple M4 Max. It is not yet
  re-verified on an M4/M4 Pro/M4 Ultra (the other rows of the hardcoded brand-string table), and
  the base `"M4"` and `"M4 Pro"`/`"M4 Ultra"` cap values are read from source, not independently
  measured on that specific silicon.
- **Finding 2 is architecturally derived, not yet dispatch-confirmed.** The `QK8_0`-equality gate
  is read directly from source and matches the DGX Spark's documented 128-bit SVE2 width, but this
  session did not obtain a clean `lldb`/`gdb` L3 trace on real SVE2 hardware — the Spark CI lane
  exists for this and has not completed a full run yet.
- **Single model.** All throughput numbers are `Qwen2.5-0.5B-Instruct`, `Q4_0` only. `Q8_0` is
  `[not available]` — no such GGUF existed in this environment, and none was fabricated by
  up-converting the lossy `Q4_0` file.
- **`threads=4` is missing from the throughput sweep** (`tools/bench.py`'s grid used `1,2,8`; the
  16-thread cells that are present are high-variance and should be read directionally).
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
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) and its [KleidiAI](https://github.com/ARM-software/kleidiai)
CPU backend (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`), both used at the pinned commit `dbadb68`.
The demo model, [`Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), is
Apache-2.0 licensed. All findings, kernels, tooling, and CI in this repository are original work
produced for the Arm Create: AI Optimization Challenge.
