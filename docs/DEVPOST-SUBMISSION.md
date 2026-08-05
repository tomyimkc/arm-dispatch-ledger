# Devpost submission copy — Polygraph

Paste-ready copy only. Every number below is sourced from `results/REMEASURE-2026-08-04-QUIET.md`,
`results/AUTODEFAULTS.md`, `results/GENERALIZATION.md`, `README.md`, and `docs/RELATED-WORK.md` in
this repository.

> **Renamed 2026-08-04.** This project shipped the challenge as `arm-dispatch-ledger`; the repo has
> since been renamed to **Polygraph** (same code, same history, same GitHub account —
> `github.com/tomyimkc/polygraph`; the old URL 301-redirects). "Polygraph" names what the tool has
> always done: it is a lie detector for software — it checks whether the accelerated code path your
> program claims to use actually ran.

---

### FIELD: Project name

Polygraph

---

### FIELD: Elevator pitch / tagline

Polygraph: a lie detector for software. It checks with a debugger, not a timer, whether the accelerated code path your program claims to use actually runs.

---

### FIELD: About the project

## Project Overview

`llama.cpp`'s own startup banner and verbose log both print `SME2 enabled` and `kleidiai: primary
q4 kernel feature SME2` on an Apple Silicon Mac — on **every single run**, including runs where
SME2 never executes once. At the tool's own default thread count (12 threads on this Apple M4
Max), single-token decode dispatches SME2 **zero times**; a `dotprod` NEON kernel runs instead,
31,871 times, while the log keeps claiming the accelerated path is active. We proved this with a
debugger attached to the real kernel entry points (`lldb`, regex breakpoint on every
`kai_run_matmul_*` symbol, real hit counts) — not by inferring it from a timing number, which is
exactly what every timing-only benchmark in this space currently does and exactly why nobody had
caught it.

**Why it should win:** this project does what the host's own rubric asks for, in the order it
asks for it. It states **what** was optimized (KleidiAI's per-chip SME2 thread-cap dispatch gate
on Apple Silicon, and the mismatch between llama.cpp's default thread count and that gate), **what
technical changes were made** (a working, additive, on-by-default upstream-style patch —
`patches/0002-kleidiai-sme-aware-thread-default.patch` — plus a second experimental dispatch patch
reported as an honest negative result), and **how much those changes helped**, measured
round-robin-interleaved against external load, decode and prefill reported separately, with the
raw tables committed so a judge can re-derive every ratio by hand. It also does the harder, less
flattering thing most hackathon submissions skip: it retracted its own first wrong number in
public (`README.md`, "Correction (2026-08-04)"), decomposed its headline win to show most of it is
a well-known effect and not a novel discovery, and cited two-days-earlier prior art on its second
finding instead of letting a judge discover the overlap first (`docs/RELATED-WORK.md`). That
discipline — real measurement, real patch, honest attribution, honest limits — is the thing worth
rewarding here, not just the raw multiplier.

A second, independent machine (`results/server/`, DGX Spark, Cortex-X925, added 2026-08-05) extends
that same discipline to genuine Arm server-class silicon and to the challenge's own "inference
server speed" focus area, which this repository had nothing on before: no TTFT number, no memory
figure, no concurrent-serving throughput anywhere in it. That lane now supplies all three, confirms
Finding 2's 256-bit SVE-width gate on a second core family under real concurrent serving load (not
just at single-user load time), and surfaces Finding 3 — a silent llama.cpp build defect this
project's own verify-what-actually-ran method caught rather than went looking for: the tool's own
CMake build line produces a binary with zero working matmul micro-kernels while its startup banner
still prints `KLEIDIAI = 1`.

## Functionality / Output

The final output is six things, all reproducible from this repository:

1. **A symbol-level dispatch verifier** (`tools/verify_dispatch.py`) that answers, for any
   `llama.cpp`-family binary and GGUF, whether an advertised accelerated kernel actually *executed*
   — not just whether it was compiled in or selected in a log line. It attaches `lldb`/`gdb` to the
   real kernel entry points and counts real hits.
2. **A working patch that ships a real optimization by default**
   (`patches/0002-kleidiai-sme-aware-thread-default.patch`): with zero flags, generation
   (decode) threads now default to KleidiAI's own detected SME2 thread cap (2 on this machine)
   instead of the physical-core count (12), while prefill/batch threads are left untouched. Measured
   round-robin-interleaved (`llama-cli`, n=9): no-flags decode goes **67.8 → 145.9 tok/s (2.15x)**,
   matching the hand-tuned `-t 2` ceiling (146.0 tok/s) within noise, while prefill stays
   **unchanged within noise** (1835.2 → 1779.8 tok/s, -3.0%) — unlike the naive "just pass `-t 2`"
   workaround, which reaches the same decode ceiling but collapses prefill by 47% (1835.2 → 975.6
   tok/s) because stock `llama.cpp`'s `-tb` default silently inherits `-t`.
3. **An honestly-reported negative result** (`patches/0001-kleidiai-phase-aware-dispatch.patch`):
   an experimental patch that lets decode into KleidiAI's existing SME+NEON hybrid path above the
   thread cap. The dispatch change is symbol-level proven (0 → 3,072 SME2 hits at 4 threads), but
   throughput is **~12% slower** at default thread count (93.6 → 82.5 tok/s) — reported as a
   regression, not spun as a win.
4. **A quantified, decomposed optimization case study**: matching thread count to phase
   (`-t 2` for decode, `-t 8`/`-tb 8` for prefill — flags `llama.cpp` already ships) is
   **3.43x faster decode** (93.6 → 321.0 tok/s) and **1.79x faster prefill** (1,230.3 → 2,198.1
   tok/s) than the tool's own default, with a dedicated decomposition sweep showing **3.95x of
   that decode win is thread-oversubscription avoidance alone** (SME2 forced off throughout — the
   well-documented Apple Silicon effect, not this project's discovery) and **1.31x is SME2's own,
   genuine contribution** on top, at the tuned thread count — and that SME2 actively **hurts**
   (0.81x) at the untuned default. All of this is backed by a live ledger
   (`results/*.json` + `site/`), a hand-written NEON/SME2/SVE2 microkernel library proving the
   silicon itself isn't the limiter, and an upstream issue filed with reproduction steps
   ([ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547)).
5. **A newly found silent build defect in llama.cpp's own KleidiAI feature detection** (Finding 3,
   `results/server/spark-provenance.txt`): following llama.cpp's own documented build line
   (`-DGGML_CPU_KLEIDIAI=ON`) on a DGX Spark (Cortex-X925, gcc 13.3) silently produces a binary
   with **zero `kai_run_matmul` symbols** — the startup banner still prints `KLEIDIAI = 1`, but
   every real matmul micro-kernel is missing, and the runtime log says `no compatible q4 kernels
   found for CPU features mask 0`. The build succeeds and exits 0. Root cause: gcc rejects every
   `-mcpu=native+<feature>` probe llama.cpp's CMake tries (its own negative-control probes fail
   too — the tell that the probe itself is broken, not the feature). The fix is one flag pair
   (`-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+sve2+i8mm+bf16+dotprod`), which restores
   **10 `kai_run_matmul` symbols** and a correct `MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1` banner.
   This is exactly the class of defect this project's verifier exists to catch — a log claiming
   the accelerated path is active on a binary where it silently is not.
6. **A concurrent-serving benchmark and dispatch-under-load capture** on the same DGX Spark
   (`results/server/server-bench.json`, `server-dispatch.json`): `llama-server` with continuous
   batching, `Qwen2.5-0.5B-Instruct-Q4_0`, swept 1→16 concurrent clients — aggregate throughput
   **14.9 → 440.4 tok/s (~29.6x)** while peak RSS grows only **724 → 901 MiB** and TTFT p99 ranges
   **89–221ms** across the sweep, peaking at 4 concurrent clients rather than climbing smoothly
   with concurrency. A second reading: at 8 concurrent clients, dropping `--threads` from 20 to 4
   costs almost nothing (271.8 → 264.8 tok/s aggregate) and TTFT p99 actually *improves*
   (117 → 94ms) — under continuous batching the server is batch-bound, not thread-bound, unlike
   this project's single-user decode story above. The same symbol-level method now runs under
   `gdb` against a live `llama-server` process: 8 concurrent clients dispatch **364,444 I8MM** and
   **11,360 DOTPROD** `kai_run_matmul` calls and zero SVE calls — confirming Finding 2's 256-bit
   SVE-width gate on a second core family (Cortex-X925, `SVE_CNT = 16` i.e. 128-bit SVE) and, for
   the first time, under real concurrent serving load rather than only at single-user decode.

## How this maps to the challenge's optimization focus areas

The challenge names six optimization focus areas. Coverage below is uneven by design, not by
omission — this project's actual contribution is concentrated in speed, server speed, developer
experience, and Arm-specific work; model size and model quality are measured honestly, not claimed
as wins.

| Optimization area | This project's evidence |
|---|---|
| **Model size** | Peak RSS measured across concurrency: **724 → 901 MiB** (`results/server/server-bench.json`). No size-reduction work (quantization, pruning, distillation) is claimed. |
| **Model quality** | Byte-identical output at a fixed seed between patched and unpatched builds — an equivalence guarantee, not an accuracy/quality improvement claim. |
| **Model speed** | Decode **67.8 → 145.9 tok/s (2.15x)** via `patches/0002` (`results/AUTODEFAULTS.md`); TTFT is now measured (see the row below). |
| **Inference server speed** | **14.9 → 440.4 tok/s** aggregate across 1–16 concurrent `llama-server` clients (~29.6x); TTFT p99 ranges **89–221ms** across that sweep — peaking at 4 concurrent clients, not the highest-concurrency row (`results/server/server-bench.json`). |
| **Developer experience** | The symbol-level dispatch verifier, MCP server, free-CI lane, dashboard, and this claims gate — plus Finding 3, a silent llama.cpp build defect this project's own method caught: the banner says `KLEIDIAI = 1` on a binary with zero working matmul kernels. |
| **Arm-specific optimization** | `patches/0002` into llama.cpp; SME2/SVE2/I8MM micro-kernels; three Arm microarchitectures measured (Apple M4 Max, GitHub's free Neoverse-N2 runner, DGX Spark Cortex-X925). |

## Setup Instructions

**Recommended — free, judge-reproducible, zero cost, no Arm hardware needed:**

GitHub's `ubuntu-24.04-arm` runner (Neoverse-N2 class) is free for public repos. Fork this repo and
either open a PR or click **Run workflow** on `verify-free-arm64` in the Actions tab. It builds
`llama.cpp` + KleidiAI, builds the kernel library, runs correctness tests, verifies dispatch, runs
the benchmark sweep, and publishes `results/LEDGER.md` as the job summary plus a downloadable
`results/` artifact. (Finding 2 is expected — and marked `continue-on-error` — on this runner:
Neoverse-N2's SVE2 is also below the 256-bit gate the finding describes.)

To run the identical pipeline locally on any `aarch64` Linux box:

```bash
sudo apt-get install -y cmake build-essential curl python3 gdb
git clone https://github.com/tomyimkc/polygraph.git && cd polygraph
./scripts/setup.sh      # clones+builds llama.cpp w/ KleidiAI, fetches the demo GGUF, builds kernels/
./scripts/run_all.sh    # correctness -> dispatch verify -> bench -> results/LEDGER.md
```

**Local Apple Silicon path (the SME2 lane this submission's headline numbers come from):**

Requires Xcode Command Line Tools (`xcode-select --install`, for `clang` + `lldb`) and `cmake`.

```bash
git clone https://github.com/tomyimkc/polygraph.git && cd polygraph
./scripts/setup.sh
./scripts/run_all.sh
```

This is a full clone+build+download from scratch (overridable via `scripts/common.sh` to reuse an
existing checkout). The expensive stage is the `lldb`-attached dispatch sweep (~9m39s on an M4 Max,
10 configurations); everything else finishes in well under two minutes. If `lldb` reports
"Developer mode is currently disabled" or a dispatch check times out, see `tools/protocol.md` §6
item 9 for the documented recovery path.

**To apply and verify the two patches directly** (against a separate `dbadb68` checkout of
`llama.cpp`):

```bash
git apply patches/0002-kleidiai-sme-aware-thread-default.patch   # the real, on-by-default win
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
./build/bin/llama-cli -m model.gguf -no-cnv -st --simple-io -p "your prompt" -n 128
# compare: GGML_KLEIDIAI_AUTO_THREADS=0 ./build/bin/llama-cli ... (reproduces stock behavior exactly)
```

**Optional — DGX Spark / Cortex-X925 (the SVE2 lane, best-effort, not a gate):**
`./scripts/setup.sh && ./scripts/run_all.sh` against your own SVE2-capable `aarch64` Linux box.
The corresponding `verify-spark-aarch64.yml` workflow is `workflow_dispatch`-only and
`continue-on-error` on every step — there is a live, unresolved incident on this project's own
Spark runner unrelated to this repo's code, so this lane is documented as best-effort.

## What changed after 2026-06-04

This entire repository is new work created for this challenge: `tomyimkc/polygraph` (created and
submitted under the name `arm-dispatch-ledger`, renamed 2026-08-04 — same account, same commit
history, old URL 301-redirects) was created **2026-08-03T23:19:09Z**, and every commit, finding,
patch, benchmark, and line of code in it was produced between then and this submission. Nothing
here predates the challenge window — there is no earlier version of this project to compare
against.

## Honest limitations

- **Single machine per finding.** Finding 1 (SME2's hardcoded per-chip thread cap) is verified live
  on one Apple M4 Max; the base "M4"/"M4 Pro"/"M4 Ultra" cap values for other chips are read from
  source, not independently measured.
- **Finding 2 (the SVE2 256-bit exact-width gate) is now dispatch-confirmed on real SVE2
  hardware.** `results/server/` (DGX Spark, Cortex-X925, `SVE_CNT = 16`, i.e. 128-bit SVE) shows
  I8MM selected over SVE exactly as the gate predicts, including under concurrent `llama-server`
  load (364,444 I8MM vs. 0 SVE calls). That confirmation was gathered manually on the box, not via
  the automated `verify-spark-aarch64.yml` CI lane, which remains `continue-on-error` and
  best-effort (see Setup Instructions) because of a separate, unrelated incident on this project's
  own Spark runner. It was also **published two days before this repository existed** by a
  different, unrelated project, [`luongs3/arm-dispatch-audit`](https://github.com/luongs3/arm-dispatch-audit)
  — full disclosure and what this project adds beyond it: `docs/RELATED-WORK.md`. This project does
  not claim priority on that finding.
- **The `0001` phase-aware dispatch patch is a measured regression, not a win** (~12% slower at
  default thread count) — reported honestly as a negative result, not offered as a performance
  improvement. Only `0002` (the auto-defaults patch) is claimed as a genuine speedup.
- **Most of the headline 3.43x decode number is not an SME2 discovery.** A decomposition sweep
  shows 3.95x of it is thread-oversubscription avoidance alone (SME2 forced off), a well-documented
  Apple Silicon effect this project did not discover; SME2's own contribution at the tuned thread
  count is a real but smaller 1.31x.
- **Single model, single quant for the headline 3.43x/1.79x tuning number.** That number
  (decode/prefill vs. the no-flags default) is `Qwen2.5-0.5B-Instruct`, `Q4_0` only. A separate
  follow-up did test the `0002` auto-defaults patch specifically against two further configs — see
  the next bullet — but that is still not a full model-size × quant grid, and still Qwen-family only.
- **The `0002` patch's speedup magnitude is this machine's numbers, not a general claim** — the
  mechanism (reading the cap from KleidiAI's own runtime detection) generalizes to any SME2 CPU
  KleidiAI supports, but 2.15x/-3% was only measured on this Apple M4 Max.
- **The `0002` patch's auto-selected thread count is not the true per-model decode optimum.** A
  generalization study (`results/GENERALIZATION.md`) confirms the *mechanism* holds beyond the
  original 0.5B/Q4_0 config: across the three model/quant configs tested (0.5B/Q4_0, 1.5B/Q4_0,
  0.5B/Q8_0), the patch beats the no-flags baseline by **1.47x-3.00x decode** and reaches its own
  `-t <cap>` target within 1-2% every time, never regressing relative to the unpatched baseline.
  But the *specific* thread count the patch hardcodes to (the SME cap — 2 on this chip) is only the
  true per-model decode optimum at 0.5B. At 1.5B/Q4_0, an isolated thread sweep shows decode peaks
  at `-t 4` (122.1 tok/s), not the cap's `-t 2` (103.9 tok/s) — a real, measured **~17.5% miss**.
  Stated plainly: the auto-defaults mechanism generalizes; the fixed SME-thread-cap heuristic does
  not generalize to model size.
- **`GGML_KLEIDIAI_SME` remains a process-global setting.** The theoretical best (SME2-decode +
  NEON-forced-prefill, simultaneously, in one process) stays `[NOT YET ACHIEVABLE]` with either
  patch.
- Absolute throughput numbers throughout were collected on a busy, multi-agent-shared machine; the
  interleaved **ratios**, not the absolute tok/s figures, are the trustworthy part — stated
  explicitly everywhere a number is quoted.
- **The DGX Spark has no SME/SME2.** Finding 1 (SME2's hardcoded per-chip decode thread cap) was
  not, and cannot be, reproduced there — the Spark lane adds Finding 3 (a new build defect), a
  second-core confirmation of Finding 2, and the concurrent-serving numbers above; it says nothing
  new about Finding 1.
- **The server-lane numbers are one model, one quant, one machine.** `results/server/` is
  `Qwen2.5-0.5B-Instruct-Q4_0` only, on one DGX Spark unit — not a model-size × quant grid, and not
  independently reproduced on a second Spark.
- **`llama-server` only — not vLLM, TGI, or any other serving stack.** The concurrent-serving
  throughput/TTFT/memory numbers and the dispatch-under-load capture are specific to `llama.cpp`'s
  own server; nothing here measures or claims anything about other inference servers.

---

### FIELD: Built with

llama.cpp, Arm KleidiAI, Arm SME2, Arm SVE2, Arm NEON, C, C++, Python, Bash, CMake, lldb, gdb, GitHub Actions, Qwen2.5-0.5B-Instruct (GGUF, Q4_0)

---

### FIELD: Repository URL

https://github.com/tomyimkc/polygraph

---

### FIELD: Dashboard URL

https://tomyimkc.github.io/polygraph/

---

### FIELD: Demo video

[PASTE YOUTUBE URL AFTER UPLOAD]

Runtime **1:43 (103.06s)**, comfortably under the contest's 3-minute cap (`docs/VIDEO-PRODUCTION.md`).
Captions are burned in, with a sidecar `.srt` (36 cues) also provided for the YouTube upload.

---

### FIELD: Upstream issue URL

https://github.com/ggml-org/llama.cpp/issues/26547

---

### FIELD: Q1 — What was the hardest part of building or optimizing your project? Select all that apply

Measuring performance, Debugging runtime or compatibility issues, Understanding Arm-specific guidance

---

### FIELD: Q2 — What would have made it easier to complete your project? Select all that apply.

More benchmarking examples, More Arm-specific optimization guidance, Better documentation

---

### FIELD: Q3 — Did this challenge change your likelihood of building on Arm in the future?

Yes, significantly more likely

---

### FIELD: Q4 — How likely are you to continue developing, optimizing, or deploying this project after the challenge?

Very likely

---

### FIELD: Q5 — What is one thing Arm could improve to better support developers like you?

Two concrete things hit during this project. First, `-march=armv9-a+sme2` — the generic, documented
way to target SME2 — SIGILLs at runtime on Apple Silicon, because `clang` emits the SVE instruction
`cntd` outside streaming mode as part of that target, and Apple ships SME2 without any
non-streaming SVE at all. The fix is to target the concrete CPU (`-mcpu=apple-m4`) instead of the
generic architecture level, but nothing in the generic Armv9+SME2 guidance says this — it has to be
discovered by hitting the SIGILL. Second, kernel dispatch on Arm CPUs is currently unobservable
without attaching a debugger to the actual kernel entry points: a runtime log or startup banner can
say an accelerated kernel is selected while the code that actually runs falls back to a slower
kernel family, and nothing short of an `lldb`/`gdb` breakpoint on the real symbols catches the gap.
Official tooling that makes dispatch decisions inspectable at runtime — without requiring every
developer to build their own debugger harness, as this project had to — would close a real, silent
performance gap that a timing-only benchmark cannot see.
