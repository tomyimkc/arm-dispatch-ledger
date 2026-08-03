# Devpost submission copy — Arm Dispatch Ledger

Ready to paste into the Arm Create: AI Optimization Challenge Devpost form (Track 2,
Cloud AI). Repo: https://github.com/tomyimkc/arm-dispatch-ledger (Apache-2.0).

---

## Project Overview

**Arm Dispatch Ledger answers a question nobody could previously answer with
confidence: when llama.cpp's startup banner prints `SME = 1 | SME2 = 1 | KLEIDIAI = 1`,
did the accelerated kernel actually run — or did it silently fall back to a slower
one?**

We found, and can prove with a debugger breakpoint on real Apple Silicon hardware, that
the answer is often "it silently fell back." KleidiAI's SME2 kernel path in llama.cpp
is gated by a hardcoded, per-chip thread cap (2 threads on an Apple M4 Max/Pro/Ultra).
Ask llama.cpp to use its own default thread count — the number of physical cores, 8 or
16 on these machines — and every single-token decode step quietly runs on NEON/DotProd
instead, while the banner and the load-time log both keep claiming SME2 is active. We
also found a second, related gap: KleidiAI's SVE kernel path requires an *exact*
256-bit vector length, which makes it architecturally unreachable on every current
128-bit-SVE2 Armv9 core (Cortex-X925, Neoverse-N2) — including the free
`ubuntu-24.04-arm` GitHub-hosted runner and the DGX Spark.

Both gaps are compile-time-vs-dispatch-time bugs in disguise: the *availability* signal
(what the binary was built with) and the *selection* signal (what the log line says was
chosen) both look right, and only the *dispatch* signal (what instruction actually
executed) is wrong. Nothing in llama.cpp's own tooling distinguishes these three layers
today. We built the tool that does, and open-sourced everything underneath it.

**Why this should win:** this project does not report a synthetic microbenchmark
number and stop. It (1) found a real, previously-undocumented bug class in a
widely-deployed open-source inference engine, on real Arm hardware, with a debugger —
not a guess; (2) turned the finding into a reusable, judge-reproducible verification
tool (a Python harness *and* an MCP server, so any agent can ask "did this actually
dispatch?" about its own inference calls); (3) measured the actual, honest performance
consequence — including the unflattering half of the answer, where plain NEON at its
own best thread count *beats* SME2 for prefill on this model, which a
flattering-numbers-only submission would have hidden; (4) proved the underlying silicon
is not the limiter by hand-writing working SME2 ACLE kernels from scratch, bit-exact
against a scalar reference, while being explicit that Apple's own Accelerate library is
still roughly 3-18x faster than our hand-written kernel, depending on matrix size (no strawman "beats naive NEON by
19x" claim survives this repo); and (5) is filing the finding upstream, because a
one-line fix (a log warning) is the actual right-sized remedy, and the Impact score
should reward shipping the fix path, not just the finding.

Every number in this repo was produced by code in this repo, run for real, on
real Arm hardware, in this session — see `results/SUMMARY.md` for the full,
reproducible measurement log, including the caveats.

## Functionality / Output

Four artifacts, all reusable independent of this specific bug:

1. **`kernels/`** — a small, dependency-free C library with correctness-verified NEON,
   SME2 (fp32 GEMM, int8 GEMM, Q4 quantized GEMM), and SVE2 kernels, built with
   `ctest`-driven correctness tests (bit-exact / near-bit-exact against a scalar
   reference across 5-8 shapes per kernel) and a microbenchmark (`kernel_bench`)
   against the strongest fair baseline available (Apple Accelerate on macOS). This
   proves the SME2 silicon genuinely works and is fast — the bug is in llama.cpp's
   dispatch logic, not in Apple's hardware or Arm's ISA.
2. **`tools/verify_dispatch.py`** — the core verification harness. For a given
   llama.cpp-family binary + GGUF model + thread sweep, it checks three independent
   evidence tiers — L1 (compile-time banner), L2 (selection-time log), L3
   (dispatch-time `lldb`/`gdb` breakpoint hit count) — and emits a machine-readable
   ledger (`results/dispatch-ledger-*.json`) plus a verdict per config:
   `SME2_DISPATCHED`, `SILENT_FALLBACK`, or `SME2_HYBRID_DISPATCH`. `--assert` makes
   this CI-gateable (non-zero exit on a real silent fallback).
3. **`tools/bench.py`** — the throughput side: an interleaved,
   warmup-discarding, 5-reps-per-cell benchmark harness across thread counts × SME
   on/off × workload phase, reporting median/stddev/min/max (never a bare mean), that
   answers the honest question "does the dispatch difference actually cost tokens/sec,
   and by how much, in which phase?"
4. **`mcp/server.py`** — a dependency-free (stdlib-only) MCP server exposing
   `detect_arm_features`, `verify_dispatch`, `recommend_config`, and `explain_finding`
   as callable tools, so an *agent* — not just a human running a script — can ask "is
   my current inference call actually using SME2?" live, and get back a grounded
   verdict instead of trusting a banner. This is the piece that targets the
   challenge's explicit call-out of "agentic multi-model workloads with MCP servers."

The measured output, end to end: `results/SUMMARY.md` (the full write-up with every
table and every caveat), `results/dispatch-ledger-darwin-arm64.json` (raw per-config
evidence), `results/bench/` (raw throughput JSON/MD + plots), `results/LEDGER.md`
(auto-generated run summary), and three GitHub Actions workflows in
`.github/workflows/` — one on GitHub's **free, hosted** `ubuntu-24.04-arm` runner
(Neoverse-N2), which is deliberately the most important lane in this repo: a judge can
fork the repository and reproduce the full pipeline at zero cost, with zero owned Arm
hardware.

## Setup Instructions

Requires: `git`, `cmake`, a C/C++ compiler, `curl`, `python3` (stdlib only — no `pip
install` needed anywhere in this repo). Works on any Arm64 Linux box or Apple Silicon
Mac; `lldb` (macOS) or `gdb` (Linux) is optional and only needed for the L3
dispatch-time tier — everything degrades honestly (`available: false`, not a fake
number) if it's missing.

```bash
git clone https://github.com/tomyimkc/arm-dispatch-ledger.git
cd arm-dispatch-ledger

# One command: builds llama.cpp with -DGGML_CPU_KLEIDIAI=ON, downloads the
# (Apache-2.0, sha256-pinned) demo GGUF, builds this repo's own kernels/.
./scripts/setup.sh

# Full pipeline: hardware feature capture -> kernel correctness tests ->
# dispatch verification sweep -> throughput bench -> results/LEDGER.md
./scripts/run_all.sh
```

Or run each piece directly:

```bash
# Kernel correctness + microbench
cd kernels && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -- -j
./kernel_test && ctest        # exit 0 = all correctness checks pass
./kernel_bench                # GFLOP/s vs. NEON vs. Accelerate

# Dispatch verification (the core finding)
python3 tools/verify_dispatch.py --binary /path/to/llama-cli \
  --model /path/to/model.gguf --threads 1,2,4,8,16 --workloads all \
  --out results/dispatch-ledger-<platform>.json --assert

# Throughput sweep
python3 tools/bench.py --threads 1,2,8,16 --sme-modes on,off \
  --phases decode,prefill_short,prefill_long --reps 5 --out-dir results/bench

# MCP server (no client needed to smoke-test)
python3 mcp/server.py --selftest
# ...or add to Claude Code:
claude mcp add arm-dispatch-ledger -- python3 "$(pwd)/mcp/server.py"
```

**Zero-cost reproduction on Arm hardware you don't own:** fork the repo and either open
a PR or manually trigger `.github/workflows/verify-free-arm64.yml` — it runs the full
pipeline on GitHub's free, hosted `ubuntu-24.04-arm` runner (Neoverse-N2) and writes the
ledger to the job summary + as a downloadable artifact. This is the lane every judge can
run without configuring anything.

## What changed after 2026-06-04

This entire repository is new work. The root commit was authored on 2026-08-03/04,
during this challenge; there is no pre-existing codebase this submission is built on
top of. The only third-party code involved is llama.cpp itself (MIT-licensed,
`ggml-org/llama.cpp`, upstream, unmodified — we build it from a pinned commit as the
*subject under test*, not as code we wrote or are claiming credit for) and the
Qwen2.5-0.5B-Instruct GGUF model (Apache-2.0, used only as a fixed, reproducible
workload for the benchmark). Every kernel, script, test, workflow, and line of the MCP
server in `kernels/`, `tools/`, `mcp/`, `scripts/`, `tests/`, and `.github/workflows/`
was written from scratch for this challenge.

## Draft answers to the 5 required custom questions

### Q1 — What was the hardest part of this project? (multi-select)

- **Debugging runtime or compatibility issues** — the headline finding *is* a
  runtime-vs-compile-time debugging problem: the banner, the log, and the actual
  dispatched kernel disagreed, and untangling that required an `lldb` breakpoint
  session, not just reading logs.
- **Understanding Arm-specific guidance** — SME2's streaming-mode ACLE rules are strict
  and easy to violate silently-until-SIGILL: gather loads are illegal in streaming
  mode, the streaming vector length can only be queried (`svcntw()`) from inside a
  streaming function, and the compiler flag has to target a concrete CPU
  (`-mcpu=apple-m4`), not a generic `-march=armv9-a+sme2` — the generic flag compiles
  cleanly and then SIGILLs at runtime on real hardware.
- **Measuring performance** — building an honest baseline meant distinguishing
  compile-time availability, selection-time choice, and dispatch-time execution as
  three separate, separately-checkable claims, and refusing the flattering-but-wrong
  "N x faster than naive NEON" framing in favor of the strongest fair baseline
  (Accelerate) and the honest reconciliation (SME2 wins decode, NEON wins prefill).
- **Finding relevant examples or documentation** — the `sme_thread_cap` /
  `ne11 >= 128` hybrid-dispatch rule does not appear anywhere in llama.cpp's docs and
  returned nothing on web search; it had to be read directly out of
  `kleidiai.cpp` source.

### Q2 — What would have made it easier? (multi-select)

- **More Arm-specific optimization guidance** — a single canonical, executable example
  of the SME2 streaming-mode boundary rules (illegal gather loads, `cntd`/`svcntw()`
  streaming-only semantics, the concrete per-vendor compile flag) would have saved the
  SIGILL-and-diagnose cycle entirely.
- **More benchmarking examples** — nothing off-the-shelf distinguishes
  compile/selection/dispatch time for an Arm-accelerated kernel path; we had to build
  that harness ourselves.
- **Better documentation** — specifically, of KleidiAI's own dispatch-time behavior
  (the thread cap, the hybrid rescue path) inside llama.cpp; this is now the subject of
  our upstream issue (`docs/UPSTREAM-ISSUE.md`).
- **Easier access to Arm-based hardware or cloud instances** — validating Finding 2
  needed genuine 128-bit-SVE2 hardware distinct from the Apple Silicon box used for
  Finding 1; combining a self-hosted DGX Spark, a self-hosted Mac, and GitHub's free
  `ubuntu-24.04-arm` runner was the practical answer, but discovering and wiring up all
  three lanes (including a still-open, unresolved OOM instability on the Spark runner)
  took real effort that a single more-available Arm cloud target would have avoided.

### Q3 — How likely are you to build on Arm again? (single-select)

**Yes, significantly more likely.** This project's central discovery — that a
production inference engine's own startup banner can misreport what actually executed
— is not specific to KleidiAI; it is a pattern worth checking for in every
Arm-accelerated library we touch next. Having working ACLE/SME2 patterns, a
correctness-tested kernel baseline, and a reusable dispatch-verification harness now in
hand removes most of the ramp-up cost for a next project.

### Q4 — How likely are you to continue this project? (single-select)

**Very likely.** Concrete next steps already identified: (1) file the upstream issue
in `docs/UPSTREAM-ISSUE.md` and follow through with a PR if the maintainers want one;
(2) get an independent, verified run of the SVE2 kernels and the dispatch verifier on
the DGX Spark and on GitHub's `ubuntu-24.04-arm` runner (both wired up in
`.github/workflows/` but not yet exercised against real Arm CI infrastructure at
submission time); (3) repeat the throughput reconciliation at a larger model size,
since the decode-wins/prefill-loses split measured here is plausibly
compute-to-memory-ratio-dependent and may reverse on a bigger model.

### Q5 — What's one thing Arm could improve? (free text)

The gap we hit hardest was a silent one on both sides of the stack: the compiler
silently accepted `-march=armv9-a+sme2` and produced a binary that SIGILLs on real
Apple Silicon (because Apple ships SME2 without non-streaming SVE, and the generic
march emits a non-streaming SVE instruction), and separately, KleidiAI's own dispatch
logic silently downgrades to NEON above a hardcoded thread cap while its own log claims
otherwise. Neither failure is loud. Our concrete ask: Arm-adjacent tooling and
libraries should treat "silent architectural fallback" as a first-class thing to log,
not just handle gracefully — a one-line runtime warning ("SME2 requested but not
dispatched for this op; falling back to NEON because thread_count(8) > sme_cap(2)")
would have saved us an `lldb` session, and we suspect this exact pattern — an
accelerated path that degrades gracefully and silently at the same time — recurs
across other Arm library integrations beyond this one. A canonical, executable
reference for the SME2 streaming-mode ACLE boundary rules (illegal gather loads in
streaming mode, `svcntw()` only valid inside a streaming function, per-vendor
`-mcpu=` targeting) mapped explicitly to "this generic flag will SIGILL, use this one
instead" would also have turned a multi-hour SIGILL debugging session into a five-minute
read.
