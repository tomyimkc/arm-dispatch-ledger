# Devpost submission copy — Arm Dispatch Ledger

Ready to paste into the Arm Create: AI Optimization Challenge Devpost form (Track 2,
Cloud AI). Repo: https://github.com/tomyimkc/arm-dispatch-ledger (Apache-2.0).

---

## Project Overview

**Arm Dispatch Ledger answers a question nobody could previously answer with
confidence: when llama.cpp's startup banner prints `SME = 1 | SME2 = 1 | KLEIDIAI = 1`,
did the accelerated kernel actually run — or did it silently fall back to a slower
one? And once you know the answer, can you actually fix the dispatch, measure whether
your fix helped, and say so honestly either way?**

We found, and can prove with a debugger breakpoint on real Apple Silicon hardware, that
the answer to the first question is often "it silently fell back." KleidiAI's SME2
kernel path in llama.cpp is gated by a hardcoded, per-chip thread cap (2 threads on an
Apple M4 Max/Pro/Ultra). Ask llama.cpp to use its own default thread count — 12 physical
performance-cores on this machine, measured directly, not assumed — and every
single-token decode step quietly runs on NEON/DotProd instead, while the banner and the
load-time log both keep claiming SME2 is active. We also found a second, related gap:
KleidiAI's SVE kernel path requires an *exact* 256-bit vector length, which makes it
architecturally unreachable on every current 128-bit-SVE2 Armv9 core (Cortex-X925,
Neoverse-N2) — including the free `ubuntu-24.04-arm` GitHub-hosted runner and the DGX
Spark.

That diagnosis led directly to an actionable, user-facing discovery: on an Apple M4 Max,
today, with **zero code changes**, splitting the per-phase thread count (`-t 2` for
decode, `-t 8` for prefill) is **3.43× faster for decode and 1.79× faster for prefill**
than `llama.cpp`'s own default thread count — measured on a quiet machine, round-robin
interleaved against the baseline specifically so external contention cannot manufacture
the number (`results/REMEASURE-2026-08-04-QUIET.md`). Nothing in `llama.cpp` currently
tells a user this: the banner claims SME2 is already in use at the default thread count,
while the finding above shows it structurally is not, for decode.

We then went one step further than diagnosis and tuning: we **wrote and measured an
actual patch** (`patches/0001-kleidiai-phase-aware-dispatch.patch`) that changes the
dispatch decision itself, on the hypothesis that the thread-cap exclusion documented
above was silently costing decode throughput. We proved at the `lldb` symbol level that
the patch does exactly what it says — SME2 kernel calls go from zero to thousands with
the flag on, same binary, same workload — and then measured its real throughput effect
against the tuned baseline above. The honest result: the patch does **not** help. At
`llama.cpp`'s own default thread count it makes decode **~12% slower**; at the tuned
thread count it is a statistical tie. We report that plainly, as a measured negative
result, rather than rounding it up or quietly dropping the patch. That reconciliation is
written up in full in `results/REMEASURE-2026-08-04-QUIET.md` and summarized in
Functionality/Output below.

Both dispatch gaps are compile-time-vs-dispatch-time bugs in disguise: the
*availability* signal (what the binary was built with) and the *selection* signal (what
the log line says was chosen) both look right, and only the *dispatch* signal (what
instruction actually executed) is wrong. Nothing in llama.cpp's own tooling
distinguishes these three layers today. We built the tool that does, open-sourced
everything underneath it, and used that same tooling to verify our own patch rather
than taking our own fix on faith.

**Why this should win:** this project does not report a synthetic microbenchmark
number and stop. It (1) found a real, previously-undocumented bug class in a
widely-deployed open-source inference engine, on real Arm hardware, with a debugger —
not a guess; (2) turned the finding into a reusable, judge-reproducible verification
tool (a Python harness *and* an MCP server, so any agent can ask "did this actually
dispatch?" about its own inference calls); (3) turned that diagnosis into a genuine,
user-facing throughput win **without writing a line of code**: an interleaved,
contention-controlled re-measurement shows `-t 2`/`-t 8` per-phase tuning is **3.43×
faster for decode and 1.79× faster for prefill** than `llama.cpp`'s own default, today,
with flags it already ships — a real optimization, not a proposal; (4) proved the
underlying silicon is not the limiter by hand-writing working SME2 ACLE kernels from
scratch, bit-exact against a scalar reference, while being explicit that Apple's own
Accelerate library is still roughly 3-18x faster than our hand-written kernel, depending
on matrix size (no strawman "beats naive NEON by 19x" claim survives this repo);
(5) **wrote, applied, and measured an actual opt-in dispatch patch** against the
diagnosed bug — not just a proposal — proved at the symbol level that it does exactly
what it claims (zero SME2 kernel calls to thousands, same binary, same workload), and
then reported its real throughput effect honestly: it does **not** improve performance —
a real ~12% regression at the default thread count, a statistical tie at the tuned
thread count — and we say so plainly instead of quietly dropping the negative result;
and (6) is filing the finding upstream and has a ready-to-open follow-up PR
(`docs/UPSTREAM-PR.md`) that separates the two pieces honestly: a small, uncontroversial
warning proposed for merge, and the dispatch-patch experiment offered purely as a
measured negative result, explicitly not proposed for merge — the Impact score should
reward shipping a real, reproducible optimization and being honest about a patch that
didn't pan out, not rounding a regression up into a win.

Every number in this repo was produced by code in this repo, run for real, on
real Arm hardware, in this session — see `results/SUMMARY.md` for the diagnosis-phase
measurement log and `results/REMEASURE-2026-08-04-QUIET.md` for the authoritative,
interleaved optimization-phase measurement log (superseding the earlier, contended
`results/OPTIMIZATION.md` run), both including every caveat.

## Functionality / Output

**What the final output *is*: a diagnosis tool, a real zero-code-change optimization
discovered from that diagnosis, an actual dispatch patch built and measured against the
same diagnosis (reported as a negative result), a verifier that checks the diagnosis and
the patch the same way, an MCP tool that exposes all of it to an agent, and a public
dashboard that publishes the evidence.** Seven artifacts, all reusable independent of
this specific bug:

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
   this CI-gateable (non-zero exit on a real silent fallback). This same tool, unmodified,
   is what verified our own patch below — it was not special-cased to be flattering to
   our fix.
3. **`patches/0001-kleidiai-phase-aware-dispatch.patch`** — **an actual dispatch
   experiment, reported as a measured negative result.** An opt-in
   (`GGML_KLEIDIAI_PHASE_AWARE=1`, default off), 56-line patch to
   `ggml-cpu/kleidiai/kleidiai.cpp` that lets decode (a GEMV, `ne11 == 1`) enter the
   existing SME2+NEON hybrid dispatch path above the chip's thread cap instead of
   collapsing to NEON-only, plus a one-shot warning naming the knob when it doesn't.
   `tools/verify_dispatch.py` proves the dispatch change is real at the symbol level
   (decode at `-t 4`: 0 SME2 kernel calls with the flag off, 3,072 with it on, same
   binary) — but the interleaved throughput re-measurement
   (`results/REMEASURE-2026-08-04-QUIET.md`) found the dispatch change does **not**
   help: decode is **~12% slower** with the flag on at `llama.cpp`'s own default thread
   count, and a statistical tie at the tuned thread count. We report that honestly
   rather than dropping it, because it is useful information (the `ne11 < 128`
   exclusion is not leaving throughput on the table on this chip) even though it is not
   a performance win. A ready-to-open follow-up pull request offering only the
   warning half for merge — and reporting the dispatch-bypass half purely as a measured
   negative result, explicitly not proposed for merge — is drafted at
   `docs/UPSTREAM-PR.md` (not opened, per this project's working agreement).
4. **`tools/crossover.py`** — the dedicated per-phase-optimum harness
   (`tools/crossover.md` documents its methodology): pins the real llama.cpp default
   thread count by measurement (12 on this machine, not the assumed 16 — a small,
   separate finding in its own right), and the best split-phase config expressible
   today with `-t`/`-tb`. This is the instrument that, re-run interleaved on a quiet
   machine, produced this project's actual optimization result: **3.43× faster decode,
   1.79× faster prefill**, today, with zero code changes
   (`results/REMEASURE-2026-08-04-QUIET.md`) — and that made the claim falsifiable
   rather than asserted, since the same harness caught the original, contention-inflated
   version of this number as wrong and forced the re-measurement that produced the
   honest figure.
5. **`tools/bench.py`** — the throughput side: an interleaved,
   warmup-discarding, 5-reps-per-cell benchmark harness across thread counts × SME
   on/off × workload phase, reporting median/stddev/min/max (never a bare mean), that
   answers the honest question "does the dispatch difference actually cost tokens/sec,
   and by how much, in which phase?"
6. **`mcp/server.py`** — a dependency-free (stdlib-only) MCP server exposing
   `detect_arm_features`, `verify_dispatch`, `recommend_config`, and `explain_finding`
   as callable tools, so an *agent* — not just a human running a script — can ask "is
   my current inference call actually using SME2?" live, and get back a grounded
   verdict instead of trusting a banner. This is the piece that targets the
   challenge's explicit call-out of "agentic multi-model workloads with MCP servers."
7. **The GitHub Pages dashboard** (`site/`, published by `.github/workflows/pages.yml`)
   — renders the advertised-vs-executed dispatch ledger and the throughput sweep
   directly from the JSON files committed to `results/`. It never fetches
   cross-origin, never invents a number, and degrades to an explicit "no data yet"
   state for anything not actually committed — the same anti-overclaim discipline as
   every other artifact in this repo, applied to the one artifact a judge is most
   likely to look at first.

The measured output, end to end: `results/SUMMARY.md` (the diagnosis-phase write-up
with every table and every caveat), `results/OPTIMIZATION.md` (the optimization-phase
write-up — did the patch help, measured against the strongest baselines, honestly),
`results/dispatch-ledger-darwin-arm64.json` and the two patched-binary
`results/dispatch-ledger-darwin-arm64-patched-flag-{on,off}.json` ledgers (raw
per-config evidence), `results/bench/` and `results/crossover/` (raw throughput
JSON/MD + plots), `results/LEDGER.md` (auto-generated run summary), and three GitHub
Actions workflows in `.github/workflows/` — one on GitHub's **free, hosted**
`ubuntu-24.04-arm` runner (Neoverse-N2), which is deliberately the most important lane
in this repo: a judge can fork the repository and reproduce the full pipeline at zero
cost, with zero owned Arm hardware.

## Setup Instructions

### Step 0 (recommended, zero cost, no Arm hardware needed): the free CI lane

**Lead with this if you just want to validate the diagnosis without owning any Arm
hardware.** GitHub's `ubuntu-24.04-arm` runner (Neoverse-N2 class) is free for public
repos. Fork the repo, then either open a PR or click **Run workflow** on
`verify-free-arm64` in the Actions tab:

```bash
gh repo fork tomyimkc/arm-dispatch-ledger --clone
cd arm-dispatch-ledger
gh workflow run verify-free-arm64.yml
gh run watch   # or: check the Actions tab
```

No local setup, no payment, no owned Arm hardware. It builds `llama.cpp` + KleidiAI,
builds `kernels/`, runs correctness tests, verifies dispatch, runs the bench sweep, and
publishes `results/LEDGER.md` as the job summary plus a downloadable `results/`
artifact — this lane is already **green** (this is the lane referenced throughout this
submission as "already GREEN on GitHub's ubuntu-24.04-arm").

This lane validates Finding 1's SME-thread-cap logic path and Finding 2 (the SVE
256-bit gate, expected and asserted on Neoverse-N2's 128-bit SVE2) — it does **not**
exercise the SME2-specific dispatch patch, since Neoverse-N2 has no SME hardware.
See Step 3 below for validating the patch itself, which requires Apple Silicon (the
only SME2 hardware this project has access to).

### Step 1 — local build and full diagnosis pipeline (any Arm64 Linux box, or Apple Silicon Mac)

Requires: `git`, `cmake`, a C/C++ compiler, `curl`, `python3` (stdlib only — no `pip
install` needed anywhere in this repo). `lldb` (macOS) or `gdb` (Linux) is optional and
only needed for the L3 dispatch-time tier — everything degrades honestly
(`available: false`, not a fake number) if it's missing.

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

### Step 2 — validate the per-phase optimum claim (`tools/crossover.py`)

This is the harness this submission's actual optimization claim rests on (the per-phase
thread tuning, independent of the code patch), and it also pins the exact baseline the
dispatch patch (Step 3) is measured against — run it first:

```bash
python3 tools/crossover.py --threads 1,2,4,8,16 --sme-modes on,off --reps 5 \
  --per-call-timeout 180 --out-dir results/crossover
```

Compare your output against `results/crossover/crossover-apple-m4-max.md` — expect the
same *qualitative* result (decode wants low threads + SME on, prefill wants more
threads + SME off) even if your machine's absolute tok/s differ from ours. That
committed file's own absolute numbers were later found to have been collected under
heavy, unequal contention; `results/REMEASURE-2026-08-04-QUIET.md` is the authoritative
source for this project's own measured decode/prefill optimum tok/s (3.43×/1.79× over
`llama.cpp`'s default).

### Step 3 — validate the dispatch patch experiment itself (Apple Silicon with SME2 only)

This is the step that answers "does the patch's dispatch change actually help
throughput" — the honest answer is no, and do not skip straight to trusting
`results/REMEASURE-2026-08-04-QUIET.md`; the same commands that produced it are
reproducible here:

```bash
# Apply the patch to a fresh llama.cpp checkout at the pinned base commit
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama-phase-aware
cd /tmp/llama-phase-aware && git checkout dbadb68
git am /path/to/arm-dispatch-ledger/patches/0001-kleidiai-phase-aware-dispatch.patch
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-cpu llama-cli llama-bench llama-tokenize -j"$(sysctl -n hw.ncpu)"

# 1. Prove the dispatch change is real at the symbol level (flag off must reproduce
#    the pre-patch ground truth EXACTLY; flag on must show real SME2 hits where flag
#    off shows zero)
cd /path/to/arm-dispatch-ledger
python3 tools/verify_dispatch.py --binary /tmp/llama-phase-aware/build/bin/llama-cli \
  --model /path/to/q05.gguf --threads 4,8 --workloads decode_short,prefill_long \
  --l3-timeout 240 --out /tmp/flag-off.json --assert
python3 tools/verify_dispatch.py --binary /tmp/llama-phase-aware/build/bin/llama-cli \
  --model /path/to/q05.gguf --threads 4,8 --workloads decode_short,prefill_long \
  --env GGML_KLEIDIAI_PHASE_AWARE=1 --l3-timeout 240 --out /tmp/flag-on.json --assert

# 2. Measure whether the dispatch change actually moves throughput (this is the step
#    that produces the honest "does it help" verdict, not just "did it dispatch")
GGML_KLEIDIAI_PHASE_AWARE=1 python3 tools/crossover.py \
  --llama-bin-dir /tmp/llama-phase-aware/build/bin --model /path/to/q05.gguf \
  --threads 1,2,4,8,16 --sme-modes on,off --reps 5 --per-call-timeout 60 --retries 2 \
  --platform <your-platform>-patched --out-dir /tmp/crossover-patched
```

Full methodology, every table, and the honest verdict this exact procedure produced on
our own machine: `results/REMEASURE-2026-08-04-QUIET.md` (authoritative; supersedes the
original, contended `results/OPTIMIZATION.md` run). Expect your dispatch proof (step 1)
to match closely; expect your throughput numbers (step 2) to vary with machine load, but
the *qualitative* verdict — real dispatch change, no throughput win: a real ~12%
regression at the default thread count, a statistical tie at the tuned thread count — to
reproduce.

### Zero-cost reproduction, restated

If you don't have Apple Silicon (needed for Step 3, the SME2-specific patch) or don't
want to build locally at all, Step 0's free CI lane is the recommended starting point —
it validates the diagnosis (both findings' dispatch logic and, for Finding 2, the actual
128-bit-SVE2 exclusion) at zero cost and with zero local setup.

## What changed after 2026-06-04

This entire repository is new work. The root commit was authored on 2026-08-03/04,
during this challenge; there is no pre-existing codebase this submission is built on
top of. The only third-party code involved is llama.cpp itself (MIT-licensed,
`ggml-org/llama.cpp`, upstream — we build it from a pinned commit, `dbadb68`, as the
*subject under test*, not as code we wrote or are claiming credit for) and the
Qwen2.5-0.5B-Instruct GGUF model (Apache-2.0, used only as a fixed, reproducible
workload for the benchmark). Every kernel, script, test, workflow, MCP server file, and
line of the dashboard in `kernels/`, `tools/`, `mcp/`, `scripts/`, `tests/`, `site/`, and
`.github/workflows/` was written from scratch for this challenge.

The one exception to "llama.cpp is unmodified" is `patches/0001-kleidiai-phase-aware-dispatch.patch`
— our own original patch, written for this challenge and stored *in this repo* as a
patch file, not as a vendored fork of llama.cpp. It was applied to a separate, disposable
clone of `dbadb68` (never committed to this repository, never distributed as modified
llama.cpp source) purely to build and measure it; this repository does not redistribute
llama.cpp source, modified or otherwise. `docs/UPSTREAM-PR.md` is the ready-to-open
follow-up to the already-filed upstream issue
([ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547))
offering that patch back to the project it targets — contribution, not appropriation.

## Draft answers to the 5 required custom questions

### Q1 — What was the hardest part of this project? (multi-select)

- **Debugging runtime or compatibility issues** — the headline finding *is* a
  runtime-vs-compile-time debugging problem: the banner, the log, and the actual
  dispatched kernel disagreed, and untangling that required an `lldb` breakpoint
  session, not just reading logs. Writing the dispatch patch made this *worse*
  before it got better: proving the patch's own dispatch change was real (not just
  "should be real" by code inspection) required the identical `lldb` A/B methodology
  a second time — flag off had to reproduce the pre-patch `0/15936` hit count
  *exactly* before we trusted a single number from the flag-on run.
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
  This got harder, not easier, once we had an actual patch to evaluate: the honest
  answer for the patch's own effect turned out to be "real dispatch change, no
  throughput win — in fact a real regression at the default thread count" — a
  genuinely negative result that took a dedicated crossover harness
  (`tools/crossover.py`), and later a fully interleaved re-measurement on a quiet
  machine, to state precisely instead of rounding toward a cleaner story.
- **Finding relevant examples or documentation** — the `sme_thread_cap` /
  `ne11 >= 128` hybrid-dispatch rule does not appear anywhere in llama.cpp's docs and
  returned nothing on web search; it had to be read directly out of
  `kleidiai.cpp` source, and the same was true in reverse when writing the patch: no
  prior art for a phase-aware bypass of that gate existed to check our approach against.

### Q2 — What would have made it easier? (multi-select)

- **More Arm-specific optimization guidance** — a single canonical, executable example
  of the SME2 streaming-mode boundary rules (illegal gather loads, `cntd`/`svcntw()`
  streaming-only semantics, the concrete per-vendor compile flag) would have saved the
  SIGILL-and-diagnose cycle entirely. On the patch side, the threadpool's own
  requirement — every thread in `[0, nth_total)` must reach the same barriers the same
  number of times per op — is not documented anywhere we found; an earlier draft of
  the patch that left extra threads idle instead of routing them to NEON was a real
  deadlock risk we only caught by reading the barrier code directly.
- **More benchmarking examples** — nothing off-the-shelf distinguishes
  compile/selection/dispatch time for an Arm-accelerated kernel path; we had to build
  that harness ourselves, and then build a second, narrower one (`tools/crossover.py`)
  once "does it dispatch" and "does it actually help throughput" turned out to need
  different instruments to answer honestly.
- **Better documentation** — specifically, of KleidiAI's own dispatch-time behavior
  (the thread cap, the hybrid rescue path) inside llama.cpp; this is now the subject of
  our upstream issue (`docs/UPSTREAM-ISSUE.md`) and the follow-up patch we're offering
  (`docs/UPSTREAM-PR.md`) proposes exactly this as a one-line runtime warning.
- **Easier access to Arm-based hardware or cloud instances** — validating Finding 2
  needed genuine 128-bit-SVE2 hardware distinct from the Apple Silicon box used for
  Finding 1; combining a self-hosted DGX Spark, a self-hosted Mac, and GitHub's free
  `ubuntu-24.04-arm` runner was the practical answer, but discovering and wiring up all
  three lanes (including a still-open, unresolved OOM instability on the Spark runner)
  took real effort that a single more-available Arm cloud target would have avoided.
  The same gap blocked broader validation of the dispatch patch experiment itself — we
  only have one SME2-capable chip (M4 Max) to measure it on, and `docs/UPSTREAM-PR.md`
  says so explicitly rather than implying broader coverage than we have.

### Q3 — How likely are you to build on Arm again? (single-select)

**Yes, significantly more likely.** This project's central discovery — that a
production inference engine's own startup banner can misreport what actually executed
— is not specific to KleidiAI; it is a pattern worth checking for in every
Arm-accelerated library we touch next. Having working ACLE/SME2 patterns, a
correctness-tested kernel baseline, a reusable dispatch-verification harness, and now a
worked example of writing, self-verifying, and *honestly grading* an actual dispatch
patch against that harness removes most of the ramp-up cost for a next project.

### Q4 — How likely are you to continue this project? (single-select)

**Very likely.** Concrete next steps already identified: (1) open the follow-up PR
drafted in `docs/UPSTREAM-PR.md` against `ggml-org/llama.cpp` — proposing the one-shot
warning for merge, and reporting the phase-aware dispatch bypass as a measured negative
result rather than proposing it — held back during the challenge window per this
project's own no-external-PRs working agreement, not because it isn't ready to send;
(2) get an independent, verified run of the SVE2 kernels, the dispatch verifier, and the
dispatch patch's own symbol-level proof on the DGX Spark and on GitHub's
`ubuntu-24.04-arm` runner (both wired up in `.github/workflows/` but the patch itself is
Apple-SME2-specific and has only been measured on one chip so far); (3) repeat the
throughput reconciliation — and the patch's own before/after — at a larger model size,
since the decode-favors-SME/prefill-favors-NEON split measured here is plausibly
compute-to-memory-ratio-dependent and may reverse on a bigger model, which would also
change whether the patch's default-thread-count regression holds up.

### Q5 — What's one thing Arm could improve? (free text)

The gap we hit hardest was a silent one on both sides of the stack: the compiler
silently accepted `-march=armv9-a+sme2` and produced a binary that SIGILLs on real
Apple Silicon (because Apple ships SME2 without non-streaming SVE, and the generic
march emits a non-streaming SVE instruction), and separately, KleidiAI's own dispatch
logic silently downgrades to NEON above a hardcoded thread cap while its own log claims
otherwise. Neither failure is loud, and neither got easier once we had our own patch to
verify — proving *our* dispatch change was real took the same `lldb`-breakpoint session
as proving the *original* bug was real, because there is still no way to ask the
process "what did you actually dispatch just now" without attaching a debugger to it.
Our concrete ask: Arm-adjacent tooling and libraries should treat "silent architectural
fallback" as a first-class thing to log, not just handle gracefully — a one-line
runtime warning ("SME2 requested but not dispatched for this op; falling back to NEON
because thread_count(8) > sme_cap(2)") would have saved us an `lldb` session, and we
suspect this exact pattern — an accelerated path that degrades gracefully and silently
at the same time — recurs across other Arm library integrations beyond this one. This
is literally what we proposed in our upstream follow-up PR (`docs/UPSTREAM-PR.md`): the
warning-only half of that patch exists specifically because dispatch should be
observable without a debugger, and we think that's a fix worth landing on its own even
if the more ambitious dispatch-change half isn't. A canonical, executable reference for
the SME2 streaming-mode ACLE boundary rules (illegal gather loads in streaming mode,
`svcntw()` only valid inside a streaming function, per-vendor `-mcpu=` targeting, and —
learned while writing the patch, not just measuring it — the threadpool barrier
invariant that every thread in a call must reach the same synchronization points the
same number of times) mapped explicitly to "this will SIGILL/deadlock, do this instead"
would also have turned multi-hour debugging sessions into five-minute reads.
