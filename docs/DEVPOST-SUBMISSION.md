# Devpost submission copy — Polygraph

Paste-ready copy only. Every number below is sourced from `docs/CLAIMS.md`'s registry (which in
turn cites `results/REMEASURE-2026-08-04-QUIET.md`, `results/AUTODEFAULTS.md`,
`results/scale/scale-experiment.json`, `results/server/`, `docs/PRODUCT.md`, `README.md`, and
`docs/RELATED-WORK.md`) in this repository. Run `python3 tools/check_claims.py` before pasting —
it fails the build on any number here that drifts from that registry.

> **Renamed 2026-08-04.** This project shipped the challenge as `arm-dispatch-ledger`; the repo has
> since been renamed to **Polygraph** (same code, same history, same GitHub account —
> `github.com/tomyimkc/polygraph`; the old URL 301-redirects). "Polygraph" names what the tool has
> always done: it is a lie detector for software — it checks whether the accelerated code path your
> program claims to use actually ran.

> **Rewritten 2026-08-05 to match the actual scored rubric.** The Official Rules score four
> criteria: **Technological Implementation (40 pts)**, **"WOW" factor (25 pts)**, **Potential
> Impact (20 pts)**, and **User Experience / Developer Experience (15 pts)**. Everything below is
> now organized around those four, in that order, with space proportional to their weight — not
> around the challenge's separate "optimization focus areas" list (that list still gets an
> appendix, since it's still worth covering, but it is not what gets scored).

---

### FIELD: Project name

Polygraph

---

### FIELD: Elevator pitch / tagline

Polygraph: a lie detector for software. It checks with a debugger, not a timer, whether the accelerated code path your program claims to use actually runs.

---

### FIELD: About the project

## The one-sentence version

`llama.cpp`'s own documented KleidiAI build line compiles a binary whose startup banner still says
`KLEIDIAI = 1` — while every one of the fast matmul kernels it needs is silently missing, costing
up to **4.57x** slower prefill on a 7B model, and the cost gets *worse*, not better, at realistic
model sizes. We found that by attaching a debugger to the real kernel entry points instead of
trusting a log line — the same three-layer method this project packages as a reusable tool
(`tools/verify_dispatch.py`, and the `tools/polygraph` CLI built around it) for anyone else to run
against their own binary.

## Where each judging criterion's evidence lives

| Criterion | Points | What to look for |
|---|---:|---|
| Technological Implementation | 40 | The L1/L2/L3 method, the fail-closed exit-code contract, the ground-truth harnesses, the claims-registry CI gate, cross-platform gdb/lldb, the target system, the two upstream patches |
| "WOW" factor | 25 | The build-defect finding (banner lies, 0 working kernels, 4.57x cost) and the second wow: we re-ran our own method on our own headline number and published that it shrank |
| Potential Impact | 20 | Filed upstream (#26547) + a drafted second report, a reusable CI pattern, and an honest statement of exactly who this affects (and who it doesn't) |
| User Experience / Developer Experience | 15 | One command, stdlib-only, JSON output, graceful degradation, a documented exit-code contract that never silently reports success |

---

## Technological Implementation — 40 points

### Three independent layers, and the discipline of never trusting the first two alone

A benchmark that only measures tokens/sec cannot see this class of bug: a binary can still run,
still print a plausible number, and still print a banner that says the accelerator is on, while
the accelerator never once executes. So this project checks three separate things and reports them
separately, rather than collapsing them into one verdict:

| Layer | What it checks | Tooling | Proves |
|---|---|---|---|
| **L1 — static** | Do the accelerated-kernel symbols exist in the built library at all? | `nm`/`otool` (macOS), `nm`/`objdump` (Linux) | The kernel was *compiled in*. Nothing about runtime behavior. |
| **L2 — selection** | What does the runtime's own verbose log say it *chose*? | Parses `llama.cpp`'s own `kleidiai: primary q4 kernel feature X` / `SME2 enabled (...)` lines | The kernel was *selected* at load time. Still not proof of execution. |
| **L3 — dispatch** | Did the kernel's machine code actually *run*? | `lldb`/`gdb`, a regex breakpoint on every real kernel entry point, a real inference workload, real hit counts | The only layer that answers the actual question. |

Every dispatch verdict in this project is backed by L3, not L1 or L2 alone — L1 and L2 alone would
have reported this project's own headline finding (Finding 3, below) as fully accelerated, and
would have been wrong.

### Fail-closed by design, at both the tool level and the CLI level

`tools/verify_dispatch.py` already ships a `--assert` flag that exits non-zero if any swept
configuration's verdict lands in `NO_DISPATCH_OBSERVED` or `SILENT_FALLBACK` — wiring the
verifier directly into a CI gate rather than a report a human has to read. This submission's unified
entry point, `tools/polygraph`, generalizes that into a contract every command honors: exit `0`
means the advertised capability matches what executed, exit `1` means a real mismatch, and exit `2`
means undetermined (no debugger, no permission, binary not found) — **never silently `0`.** A tool
that answers "did the fast path really run?" cannot itself be the kind of tool that quietly claims
success when it doesn't know.

### Ground-truth harnesses that exist because the probe once lied to us

`tests/l3_gdb_groundtruth/` is a small dlopen-based harness that asserts the L3 probe recovers a
*known*, pre-computed call count against a synthetic library. It exists because our own `gdb`
probe once silently reported zero hits on the free CI lane — exactly the failure mode this whole
project exists to catch, this time inside our own tooling instead of `llama.cpp`'s. We do not ask
a judge to trust that our dispatch counts are real; we ship the test that would fail if they
weren't.

### A claims registry that fails CI on drift, not just on invention

This repo has shipped a wrong number before: a fabricated **"+57.3%"** win, produced by comparing
a baseline and a patched config measured in unevenly-contended time windows
(`results/REMEASURE-2026-08-04-QUIET.md` carries the full retraction). `docs/CLAIMS.md` and
`tools/check_claims.py` are the structural fix: every ratio, throughput figure, percentage, and
dispatch-hit count anywhere in `README.md`/`docs/*.md`/`site/` must resolve to either a
hand-curated, source-cited registry entry or a real leaf value committed under `results/**/*.json`
— and every previously-retracted figure is permanently banned from reappearing unmarked. It runs on
every push and PR (`.github/workflows/claims.yml`), stdlib-only, no build step. It has already
caught real pre-existing drift on its first run against a hand-reviewed tree (documented in
`docs/CLAIMS.md`'s "Resolved findings" section) — the argument for building it in the first place.

### Cross-platform debugger work, including against a live server process

The L3 layer runs identically on macOS (`lldb`, `dispatch_probe.lldb`) and Linux (`gdb`,
`dispatch_probe.gdb`), auto-selected per platform. It has been exercised against a one-shot
`llama-cli` process (the Apple M4 Max results) *and* attached live to a running `llama-server`
process under concurrent load on a DGX Spark, where it recorded **364,444** I8MM and **11,360**
DOTPROD `kai_run_matmul` calls across 8 concurrent clients — the same technique, extended from a
single inference call to a live serving process.

### The target system: built-in presets, plus any binary at all

`tools/polygraph` ships built-in, one-word presets (`polygraph check llama-cpp-kleidiai`, for
example) that encode the exact symbol regex and workload this project already validated — but the
underlying mechanism is not `llama.cpp`-specific. `polygraph check --binary PATH --symbols REGEX
--run "CMD"` runs the identical L1/L2/L3 method against *any* binary and *any* claimed accelerated
symbol, with no preset required. This is what "generalizes beyond `llama.cpp`" means concretely:
the presets are a convenience layer over a mechanism that only needs a binary, a symbol pattern,
and a command to run.

### Two upstream patches, one real win and one honest negative result

- **`patches/0002-kleidiai-sme-aware-thread-default.patch`** — a real, on-by-default win. With
  zero flags, generation (decode) threads now default to KleidiAI's own detected SME2 thread cap
  instead of the physical-core count. Measured round-robin-interleaved (`llama-cli`, n=9): no-flags
  decode goes **67.8 → 145.9 tok/s (2.15x)**, matching the hand-tuned ceiling within noise, while
  prefill stays essentially unchanged (**-3.0%**, within noise) — unlike the naive "just pass
  `-t 2`" workaround, which reaches the same decode ceiling but collapses prefill by **47%**
  because stock `llama.cpp`'s `-tb` default silently inherits `-t`.
- **`patches/0001-kleidiai-phase-aware-dispatch.patch`** — an experimental patch that lets decode
  into KleidiAI's existing SME+NEON hybrid path above the thread cap. The dispatch change is
  symbol-level proven, but throughput is **~12% slower** at default thread count — reported as a
  measured regression, not spun as a win. Rigor cuts both ways here: a companion decomposition
  sweep shows most of this project's headline decode number (3.95x of a 6.44x total) is
  thread-oversubscription avoidance alone, a well-documented effect this project did not discover;
  SME2's own contribution at the tuned thread count is a real but smaller **1.31x**, and it
  actively **hurts** (0.81x) at the untuned default.

### And the agentic-workload piece the challenge calls out by name

`mcp/server.py` is a dependency-free MCP stdio server exposing `detect_arm_features`,
`verify_dispatch`, `recommend_config`, and `explain_finding` as callable tools, so an agentic
client can ask *this machine, right now* whether SME2/SVE is actually dispatching instead of
trusting a compile-time banner — every tool degrades honestly (missing `lldb` → an explicit
"unavailable" field, not a fabricated hit count) rather than guessing.

**Full honesty about limits is itself part of this section's evidence**, not a separate liability —
see "Honest limitations," below, for the complete, unabridged list of what is single-machine,
single-model, or not yet independently reproduced.

---

## "WOW" factor — 25 points

### The finding: a banner that lies, and a cost that grows exactly where it matters

Following `llama.cpp`'s own documented KleidiAI build line (`-DGGML_CPU_KLEIDIAI=ON`) on a DGX
Spark (Cortex-X925, gcc 13.3) produces a binary with **zero** working `kai_run_matmul` kernels —
while the startup banner still prints `KLEIDIAI = 1`, the load-time log reports "no compatible q4
kernels found," and the build exits `0`. No warning anywhere. The cost is not small, and it is not
flat across model sizes — it gets *worse* the more realistic the model:

| Model | Phase | Cost of the broken build |
|---|---|---|
| 0.5B (toy) | prefill | 1.42x |
| 0.5B (toy) | decode | 0.99x — no measurable effect |
| 7B (realistic) | prefill | **4.57x** |
| 7B (realistic) | decode | 1.65x |

A team validating a build on a small model for a quick sanity check would see almost nothing wrong
and ship it anyway. The one-flag-pair fix (`-DGGML_NATIVE=OFF
-DGGML_CPU_ARM_ARCH=armv9.2-a+sve2+i8mm+bf16+dotprod`) restores 10 working `kai_run_matmul` symbols
and a correct feature banner.

### The second wow: we pointed the tool at our own headline claim

Most submissions stop at the flattering number. We didn't. This project's own original headline —
thread-count tuning is a **3.43x** decode win — was measured on a tiny 0.5B model on Apple Silicon.
We re-ran the identical method on a second machine (DGX Spark) across two model sizes and published
what we found: the same class of thread-tuning win shrinks from **4.56x at 0.5B to 1.33x at 7B**.
We did not quietly narrow the claim in a footnote — `README.md`'s own "Correction" section leads
with the shrinkage, and `docs/PRODUCT.md` documents rejecting an entire product framing (an
"auto-tuner for local LLM runners") because this exact re-measurement showed the headline number
does not survive contact with a realistic model size. Publishing your own tool's finding against
your own prior claim, and letting it cost you a product direction, is not a normal thing for a
hackathon entry to do — and it is exactly what a debugger-based verifier should do when pointed at
anything, including its own project.

---

## Potential Impact — 20 points

### Filed upstream, plus a second report ready to go

Both original findings (the SME2 thread-cap gate and the SVE exact-width gate) were filed as
[ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547) on 2026-08-04, with
reproduction commands, exact source-line citations, and an offer to send the patch. The newer
build-defect finding (Finding 3, the zero-kernel KleidiAI build) is drafted and ready to file —
`docs/issues/finding3-title.txt` and `docs/issues/finding3-body.md` — but has **not yet been
submitted** as its own issue; we say so plainly rather than implying it's already in the
maintainers' queue.

### A CI pattern any project can adopt, not just this one

`docs/CLAIMS.md` is explicit that its check is "deliberately a reusable pattern, not a one-off
script": any project that measures things, writes results into more than one document, and has
ever shipped a stale number can adopt the same three checks with the same stdlib-only tool and one
CI job. That is a piece of infrastructure other maintainers can drop in, independent of whether they
ever touch KleidiAI.

### The target system generalizes past this one library

Because `tools/polygraph`'s ad-hoc mode (`--binary --symbols --run`, no preset required) needs
nothing project-specific, the verification method — not just this one finding — is available to
any maintainer who has ever wondered whether their own "accelerated" build flag actually does
anything.

### Being honest about how many people this actually affects

Judges punish inflated impact claims far harder than modest ones, so here is the real scope. We
checked whether the binaries ordinary users actually download carry the Finding 3 defect. They
don't: the official `llama.cpp` release (`b10276`, `ubuntu-arm64`) ships all eight
runtime-dispatch-variant libraries with **0** `kai_*` symbols but **204** `ggml` repack matmul
kernels compiled into each one — the release is accelerated, it just doesn't route through
KleidiAI at all, because `GGML_CPU_KLEIDIAI` defaults **off** and neither `llama.cpp`'s nor
Ollama's release CI ever turns it on. There is no advertised-vs-executed gap in a stock install.
**The exposed population is source-builders on new Arm silicon and release pipelines** — people who
manually pass `-DGGML_CPU_KLEIDIAI=ON` on a native build with a compiler whose Arm CPU-name table
predates their specific core. Real, verified (`results/prevalence/shipped-binaries-2026-08-05.json`,
`docs/PRODUCT.md`), and small — not "every Ollama user," and we don't claim otherwise.

---

## User Experience / Developer Experience — 15 points

### One command, three verbs

```
polygraph list                        # list built-in targets, one per line
polygraph check <target> [options]    # run a built-in preset
polygraph check --binary PATH --symbols REGEX --run "CMD"   # ad-hoc, no preset needed
polygraph explain <target>            # print what the preset looks for and why
```

Human output leads with a one-line verdict a non-expert can understand — e.g. `MISMATCH: binary
advertises KleidiAI but 0 of 10 kernel symbols ever executed` — with `--json` available for
machine consumption and `--quiet` for scripting.

### No dependencies beyond a debugger already on the box

Every piece of this project's own tooling — `tools/verify_dispatch.py`,
`tools/check_claims.py`, `mcp/server.py` — is stdlib-only Python, deliberately, since judges (and
any real user) will run this on a fresh machine. The only external requirement is a debugger
(`lldb` on macOS, `gdb` on Linux) that a working Arm dev box already has.

### Two minutes to first result, on a laptop, with no Arm hardware

```bash
git clone https://github.com/tomyimkc/polygraph && cd polygraph
make demo
```

`examples/catch-a-liar/liar.c` is thirty lines of C compiled two ways. Both builds print
`using fast path: yes` and return the same answer; one is lying. `polygraph check catch-a-liar`
exits `1` on the liar and `0` on the honest build. A judge can verify the core claim of this
entire project before deciding whether to read any of the rest of it. A recording of that run is
the first thing in the README.

### Graceful degradation that says so, and an exit-code contract that never lies

`--level` caps how far a check attempts to go and falls back cleanly when a layer isn't available
(`--level 2` for a fast L1+L2-only smoke test; auto debugger selection with an explicit "none"
fallback) — and always says which layer it actually reached, rather than silently reporting a
lower-confidence result as a full pass. Exit codes are contractual: `0` match, `1` mismatch, `2`
undetermined — CI can depend on the distinction between "acceleration failed" and "we couldn't
check," which is exactly the distinction a timing-only benchmark collapses.

### The rest of the developer experience already shipped

A dependency-free MCP server (`claude mcp add polygraph -- python3 mcp/server.py`, no `pip
install`), a static dashboard with no build step (`site/`, published by
`.github/workflows/pages.yml`), and one-command setup/run scripts (`./scripts/setup.sh &&
./scripts/run_all.sh`) validated on three different Arm64 targets, including a free,
judge-reproducible GitHub-hosted `ubuntu-24.04-arm` runner that needs no Arm hardware at all.

---

## Appendix: the challenge's stated optimization focus areas (reference only — not the scored rubric)

The challenge separately names six optimization focus areas. These are not the judging criteria
above, but they're still worth a maintainer's quick scan. Coverage is uneven by design: this
project's actual contribution concentrates in speed, server speed, developer experience, and
Arm-specific work; model size and model quality are measured honestly, not claimed as wins.

| Optimization area | This project's evidence |
|---|---|
| **Model size** | Peak RSS measured across concurrency: **724 → 901 MiB** (`results/server/server-bench.json`). No size-reduction work (quantization, pruning, distillation) is claimed. |
| **Model quality** | Byte-identical output at a fixed seed between patched and unpatched builds — an equivalence guarantee, not an accuracy/quality improvement claim. |
| **Model speed** | Decode **67.8 → 145.9 tok/s (2.15x)** via `patches/0002` (`results/AUTODEFAULTS.md`). |
| **Inference server speed** | **14.9 → 440.4 tok/s** aggregate across 1–16 concurrent `llama-server` clients (~29.6x); TTFT p99 ranges **89–221ms** across that sweep, peaking at 4 concurrent clients, not the highest-concurrency row (`results/server/server-bench.json`). |
| **Developer experience** | The symbol-level dispatch verifier, MCP server, free-CI lane, dashboard, and the claims gate — plus Finding 3, a silent llama.cpp build defect this project's own method caught. |
| **Arm-specific optimization** | `patches/0002` into llama.cpp; SME2/SVE2/I8MM micro-kernels; three Arm microarchitectures measured (Apple M4 Max, GitHub's free Neoverse-N2 runner, DGX Spark Cortex-X925). |

---

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
  load (364,444 I8MM vs. zero SVE calls). That confirmation was gathered manually on the box, not
  via the automated `verify-spark-aarch64.yml` CI lane, which remains `continue-on-error` and
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
  own server; nothing here measures or claims anything about other inference servers. (`vLLM`'s Arm
  CPU path doesn't share the KleidiAI mechanism at all — see `docs/PRODUCT.md`.)
- **Finding 3 has not yet been filed upstream.** It is drafted and ready
  (`docs/issues/finding3-title.txt`, `docs/issues/finding3-body.md`) but, unlike Findings 1 and 2
  (`ggml-org/llama.cpp#26547`), has not been submitted as its own issue as of this submission.
- **The Finding 3 defect does not affect stock releases.** The official `llama.cpp` release
  (`b10276`) ships `GGML_CPU_KLEIDIAI` off by default and is not broken by this defect; the exposed
  population is people manually building with `-DGGML_CPU_KLEIDIAI=ON` on a native build against a
  compiler whose Arm CPU-name table predates their specific core — real, but small and expert.
  (`results/prevalence/shipped-binaries-2026-08-05.json`, `docs/PRODUCT.md`.)

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

## Devpost narrative fields

Devpost's own "About the project" box conventionally uses these headers. Paste them, in this
order, after the judging-criteria sections above — the sections above are the depth a judge scores
against; these are the plain-language story a general reader skims first.

### FIELD: What it does

Polygraph checks whether the "fast mode" a piece of software claims to be using is actually
running — not by timing it, but by attaching a debugger to the real code and counting how many
times the fast function actually gets called. We pointed it at `llama.cpp`, the software behind
many local AI chatbots, running Arm's KleidiAI acceleration, and found a real, previously-unknown
bug: following Arm's own official build instructions produces a program whose startup message
claims acceleration is on, but that has zero of the actual fast kernels compiled in. It silently
runs slow the entire time, and nothing in the log, the banner, or the exit code tells you. We
measured exactly how much that costs (up to 4.57x slower on a realistic model size), built the
one-flag fix, verified the fix with the same debugger method, and reported it to the people who
maintain `llama.cpp`.

### FIELD: How we built it

We built a three-layer check: first, does the binary even contain the fast code (a static symbol
check)? Second, what does the program's own log say it picked (its self-reported log line)? Third
— the layer nobody else was doing — did that code actually execute, proven by literally setting a
breakpoint on the real function and counting real hits, without stopping the program. That third
layer is what caught the gap the first two would have missed. Around that core method we built a
patch that fixes a real, measured performance problem (and a second patch we tried, measured, and
reported as *not* working, honestly); a small command-line tool and an MCP server so an AI agent
can ask the same question; a results dashboard; and continuous-integration checks — including one
that fails the build if any number in our own writeup drifts from what we actually measured, since
we've been burned by a stale number before. We ran the whole pipeline on three different Arm
machines: a MacBook, a free GitHub-hosted cloud runner, and a DGX Spark server.

### FIELD: Challenges we ran into

Getting a debugger to catch a function that lives inside a library the program only loads after it
starts running (`dlopen`) took real work — the fix is "pending breakpoints," a feature most people
never need. We shipped a wrong headline number once (a fabricated +57.3%), caused by comparing two
runs measured under different, uneven background load, and had to publicly retract it and build a
mechanical check so it can't happen silently again. On the DGX Spark, the exact same silent-failure
pattern this project hunts for happened to our own build: the compiler rejected `llama.cpp`'s own
CPU-detection probe, which is the root cause of the headline finding. A promising shortcut —
reading hardware performance counters instead of using a debugger — turned out to be locked down on
that same server (`perf_event_paranoid=4`, and the specific counters we needed don't exist on that
chip at all), so we had to keep the debugger-based approach rather than switch to something faster.
And targeting Arm's newest CPU feature (SME2) the way the documentation says to crashes instantly on
Apple hardware — the real fix had to be discovered by hitting the crash, not by reading the manual.

### FIELD: Accomplishments that we're proud of

We found and reported a real bug to a major open-source project, with exact reproduction steps and
an offer to send the fix. We retracted our own wrong number in public instead of quietly fixing it,
and then built a tool that makes that mistake structurally hard to repeat. We re-ran our own
headline result on a bigger, more realistic model size and published that it shrank — instead of
only publishing the number that made us look best. And we tested the "how big is this actually?"
question honestly: rather than assume the bug affects everyone, we checked what real, official
release binaries look like, found the bug doesn't affect them, and said so plainly instead of
inflating the impact.

### FIELD: What we learned

A timing-only benchmark genuinely cannot see this class of bug — you need a debugger on the real
entry points, because the log and the exit code will both look fine. A number measured on a tiny
demo-sized model can be dramatically different from the same measurement on a model people actually
run, which is a lesson we learned about our own results, not just about auditing someone else's.
And some of Arm's own documented guidance has real gaps that only show up as a crash or a silent
failure — the fix for both had to be found by hand, not by following the docs as written.

### FIELD: What's next

File the drafted Finding 3 report upstream (it's written, just not submitted yet). Extend the CLI's
ad-hoc mode to a couple of other Arm-accelerated projects beyond `llama.cpp`, to prove the target
system really does generalize rather than just claiming it does. Get the automated SVE2
confirmation lane green once the unrelated Spark runner incident is resolved, so that result stops
depending on a manual measurement. And explore per-model auto-tuning that's honest about the thing
this project already found the hard way: the current thread-cap heuristic is a good default, not a
universal optimum, and any auto-tuner has to earn that claim per model size instead of assuming it.

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
