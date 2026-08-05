# Product thesis — what this project is, now that the data has decided it

This document exists because the obvious product idea and the correct one turned out to be
different, and the difference was settled by measurement, not opinion. It is a decision document,
not a pitch deck: it states the framing the data rules out, the framing it supports, what that
means for positioning, and what is still unknown.

The evidence behind every number below is `results/scale/scale-experiment.json` — a DGX Spark
(GB10, 20-core Armv9.2, Cortex-X925 + Cortex-A725, 121 GiB, gcc 13.3.0) run of `llama.cpp` @
`dbadb68`, `llama-bench -p 128 -n 32`, 5 repetitions, round-robin interleaved across configs so
load drift hits every config equally, median ± population stdev. Two Apache-2.0 models with
verified licences: Qwen2.5-0.5B-Instruct-Q4_0 and Qwen2.5-7B-Instruct-Q4_0.

## The tempting framing: an auto-tuner for local LLM runners

The natural product pitch, once you have a thread-count sweep in hand, is: point a tool at your
Ollama / `llama.cpp` / vLLM install, sweep the runtime knobs, and tell the user the fastest
setting. It is an easy demo and an easy story — "we found you free speed."

The data does not support this as the core of a product, for two separate reasons, both visible
in the same experiment.

**The win collapses as the model gets realistic.** On the toy model, decode throughput at this
box's default thread count (20 threads — confirmed by comparing the default-thread rows against
the explicit thread sweep: 0.5B decode 41.70 tok/s vs. 45.49 tok/s at t=20, 7B decode 18.45 tok/s
vs. 18.03 tok/s at t=20, close enough on both models to confirm the resolution) is 41.70 tok/s;
the best observed decode throughput anywhere in the sweep is 190.32 tok/s, at 8 threads. That is a
**4.56x** tuning win — a genuinely large number. Run the identical experiment on the 7B model
people actually deploy, and default decode throughput is 18.45 tok/s against a best of 24.45 tok/s,
also at 8 threads: a tuning win of roughly one and a third. The headline multiple this repo's
earlier work was built around is, at a size a real user runs, less than a third as large. A
product whose flagship number shrinks by roughly 3.4x the moment the customer's model gets bigger
is not a strong foundation for a launch pitch, especially since the natural demo model for a fast,
cheap live demonstration is exactly the small model where the effect looks best.

**The optimum is not even a stable target to search for.** Decode throughput peaks at 8 threads
for both models on this box — that part is at least consistent. Prefill does not agree: the 0.5B
model's prefill throughput peaks at 8 threads (1457.68 tok/s), but the 7B model's prefill
throughput keeps climbing all the way to 20 threads (218.39 tok/s, still rising past the point
where 7B decode had already fallen off its own peak). So "the right thread count" is at minimum a
function of model size *and* of whether the workload is prefill- or decode-shaped, on one machine,
with one compiler, before any framework- or hardware-generalization question is even asked. An
auto-tuner has to re-discover this per model, per phase, per machine, forever — a search problem
whose payoff shrinks exactly where the product would need it to hold up.

None of this means thread tuning is fake or useless — 4.56x on the toy model and roughly a third
faster on the 7B model are both real, measured, reproducible numbers, and the smaller of the two is
still a genuine free win on a model people actually run. It
means thread tuning cannot be the headline of what this project sells, because the headline number
depends on picking a small enough model to demo, and that is precisely the kind of number this
project's own prior retraction (`docs/CLAIMS.md`) exists to make impossible to get away with
twice.

## The framing the data supports: the verifier is the accelerator

Run a second experiment against the same two models: build `llama.cpp` with its own documented
KleidiAI command, and compare that build against a build where feature detection worked. This is
not a tuning knob — it is the difference between a binary whose own startup banner claims
acceleration is on and one that actually has working kernels behind that claim, which is exactly
what `tools/verify_dispatch.py` exists to check by attaching to the real kernel entry points
instead of trusting the banner.

The cost of the broken build **grows with model size — the opposite direction from thread
tuning**:

| Model | Phase | Broken build | Fixed build | Multiple |
|---|---|---|---|---|
| 0.5B | prefill | 657.00 ± 51.29 tok/s | 933.63 ± 74.03 tok/s | 1.42x |
| 0.5B | decode | 42.18 ± 6.78 tok/s | 41.70 ± 6.86 tok/s | 0.99x — no effect |
| 7B | prefill | 48.64 ± 0.42 tok/s | 222.14 ± 4.94 tok/s | **largest gap in this table** |
| 7B | decode | 11.17 ± 0.37 tok/s | 18.45 ± 0.84 tok/s | 1.65x |

**the 7B prefill recovery (48.64 → 222.14 tok/s) is the single largest gap in this entire
experiment** — larger, in proportional terms, than the best-case thread-tuning multiple at either
model size. And unlike thread tuning, it does not shrink as the model gets more realistic; it
grows. On the toy model the defect barely registers (1.42x on prefill, no measurable effect at all
on decode) — which is exactly why a team validating their build on a small model for a quick
sanity check would never catch it.

This is why the verifier is the accelerator, not a separate feature bolted onto one: the largest
measured speedup available anywhere in this dataset does not come from searching a parameter
space, it comes from detecting that a specific, real capability — working accelerated matmul
kernels — was silently never enabled, despite the build exiting `0` and the banner saying it was.
Fixing it requires zero code changes to the model and zero runtime tuning; it requires knowing the
capability was missing in the first place, which is a detection problem, not a search problem. A
tuner searches a space that already contains everything the hardware can do. A verifier tells you
when the space you're searching is smaller than you think it is — and that gap, once you can see
it, is the bigger number.

## What that means concretely for positioning

The product answers one question: **"why is my local LLM slow, and am I actually getting the
hardware I paid for?"** The differentiator is not that it produces a faster number — a timing-only
benchmark can already tell a user a number went up or down. It is that the answer is provable at
the symbol level: a debugger attached to the real kernel entry point, counting actual calls, can
say *why* — a specific kernel family never got dispatched — rather than leaving the user to guess
whether a delta came from a disabled accelerator, thermal throttling, scheduler placement, or
noise. That distinction is also what determines whether the fix is durable. A verified capability
gap is a one-time build fix; a tuned thread count is a setting you have to remember to reapply
every time the model, the box, or the phase mix changes.

Thread tuning still belongs in the product — just not as the headline. Once a build is verified
correct, reporting the phase-dependent optimum on top of that ("your build is verified correct,
and on this box decode is fastest at 8 threads while prefill wants 20") is a reasonable secondary
feature. It should never be the first claim, because it is the claim whose size depends on how
small a model the reader picked.

## Honest limits

- **One machine, one compiler, one quantization, two model sizes, one benchmarking tool.** Cortex-
  X925 + Cortex-A725 (DGX Spark, GB10), gcc 13.3.0, Q4_0 only, 0.5B and 7B only, `llama-bench`
  only. None of the numbers above are claimed to hold on a different chip, compiler, quant, or
  framework.
- **The 7B prefill recovery (48.64 → 222.14 tok/s) is specific to a build where feature detection
  collapsed entirely** — the broken configuration's own startup banner showed no `DOTPROD`, no
  `MATMUL_INT8`, and no `SVE`, not merely "KleidiAI missing" in isolation. A build with a narrower
  detection failure would very plausibly show a smaller gap. This is a measured fact about this
  specific defect on this specific machine, not a general constant for "a broken accelerated
  build."
- **The defect's cost is phase-dependent and must never be blended.** 0.5B decode shows no
  measurable effect (0.99x, and both broken and fixed decode medians sit within roughly one
  stdev of each other). Prefill and decode are different bottlenecks — batch matmul-bound vs.
  memory-bandwidth-bound — and reporting one combined "speedup" across both, or across model
  sizes, would misrepresent a result that is genuinely size- and phase-dependent. This repo has
  already retracted one number for exactly this kind of blending; see `docs/CLAIMS.md`.
- **The load context is not fully external.** `scale-experiment.json` records `load_before: 0.43`
  and `load_after: 12.32` — the elevated load after the run is generated by the preceding 20-
  thread benchmark sweep itself, not by an unrelated process. Round-robin interleaving of configs
  protects the *relative* comparisons between configs, but the absolute load level during the run
  is not independent of the workload being measured.
- **n=5 reps per config.** Population stdev is reported alongside every median above precisely so
  a reader can see which comparisons are tight (7B prefill broken-vs-fixed, ±0.42 vs. ±4.94, a
  clean separation) and which are noisier (0.5B prefill, ±51.29 and ±74.03 on medians roughly 277
  tok/s apart — still a real gap, but a much wider band).

## What would have to be true for this to be a product, not a finding

The generalizable claim this project can currently stand behind is a **method** — verify execution
at the symbol level instead of inferring capability from timing — not a magnitude. Before "the
verifier is the accelerator" is a product claim rather than a two-model, one-machine finding, the
following would need to be measured, not assumed:

1. **Other frameworks.** Does the same class of defect — a build that reports success and prints
   an "enabled" banner while the accelerated kernel path is never actually reachable — occur in
   ONNX Runtime or vLLM, or is it specific to `llama.cpp`'s KleidiAI integration?
2. **Other toolchain/CPU pairs.** This defect was found on one gcc version against one Arm core
   combination. Is "compiler feature-detection logic older than the CPU it's compiling for" a
   general class of build defect, or a one-off collision specific to gcc 13.3.0 and Cortex-X925?
3. **More model sizes.** Only two points (0.5B, 7B) are measured. Where does the thread-tuning
   payoff actually cross below the build-defect payoff, and does the build defect's cost keep
   growing past 7B or plateau?
4. **Whether the defect class exists outside the toolchain-newer-than-CPU case.** This repo found
   the mechanism in one specific circumstance. Whether "banner says on, kernel never dispatches"
   shows up for other reasons — different flags, different backends, different silent fallback
   paths — is open.
5. **Repetition across machines.** This project's other findings (the SME2 thread-gating and SVE
   exact-width-gate work) were measured on a different machine (Apple M4 Max) than this scale
   experiment (DGX Spark). No number in this document is claimed to reproduce there or anywhere
   else until it is actually measured there.

Until those are answered, what ships is the verifier and the method it demonstrates — not a
promise about how large the win will be on hardware nobody here has measured yet.

---

## The third framing, tested and rejected: "an accelerator for Ollama / llama.cpp / vLLM"

*Added 2026-08-05, after the verifier framing was judged unattractive twice.*

The proposal was to reposition this project as a fundamental tool that speeds up local LLM
inference across engines. The whole thesis rested on one number nobody had measured: **prevalence**
— specifically, whether the binaries ordinary users download carry the Finding 3 defect. Finding 3
was measured on a build *we* compiled. The product claim is about builds *other people* compiled.
Those are different populations, and the gap had never been tested.

It has now been tested, and the answer is no.

### The decisive measurement

Official `llama.cpp` release `b10276`, `ubuntu-arm64` archive, all eight runtime-dispatch variants:

| library | `kai_*` symbols | ggml repack matmul kernels |
|---|---|---|
| `libggml-cpu-armv8.0_1.so` … `libggml-cpu-armv9.2_2.so` (8 files) | **0** | **present in all 8** |

Zero KleidiAI symbols — **and that is not a defect.** Each variant carries ggml's own
`ggml::cpu::repack::tensor_traits<block_q4_0, ...>` matmul kernels, and the binary exports
`ggml_cpu_has_dotprod`, `ggml_cpu_has_matmul_int8`, `ggml_cpu_has_sve`, `ggml_cpu_has_sme`,
`ggml_cpu_has_sme2`. The release is built with `-DGGML_CPU_ALL_VARIANTS=ON`, which compiles one
library per ISA tier and selects at load time. The shipped binary is ISA-dispatched and
accelerated; it simply does not route through KleidiAI.

**`GGML_CPU_KLEIDIAI` defaults to OFF, and neither llama.cpp's nor Ollama's release CI ever turns
it on.** So there is no advertised-vs-executed gap in a stock install: the banner correctly says
KleidiAI is off, because it is off. The failure this project detects — a banner claiming
`KLEIDIAI = 1` over a binary with no KleidiAI kernels — cannot occur in a stock install at all.

### Why the "zero symbols means free speed on the table" reading is wrong

An earlier working hypothesis in this project was that a `kai_*` count of zero on a shipped binary
implied unexploited hardware, and therefore a large recoverable speed-up for every Arm user. The
symbol table above refutes it directly: **zero KleidiAI symbols is not zero acceleration.** The
comparison that would justify the product claim is "shipped binary vs. best possible build", and
the shipped binary already carries per-ISA repack kernels. The 4.57x in
`results/scale/scale-experiment.json` is measured against a *broken* build, not against the
official release, and must not be quoted as headroom available to a stock user.

### What Finding 3 actually requires

Three conditions simultaneously, none of them defaults: a **native** build (not the portable
`GGML_CPU_ALL_VARIANTS` mode every official release uses), a manually-passed
`-DGGML_CPU_KLEIDIAI=ON`, and a compiler whose Arm CPU-name table predates the specific core
(gcc 13.3.0 shipped 2024-05-21; `cortex-x925` was backported only in November 2024). The exposed
population is people hand-building llama.cpp for new Arm silicon — real, but small, and expert.

### What else the review found

- **The mechanism is being fixed upstream, for free, on a visible timeline.** An Arm-affiliated
  engineer opened a three-PR series reworking KleidiAI build integration and adding runtime
  feature detection on 2026-07-24 — the exact code path Finding 3 exploits.
- **No demand signal.** No issue in `ollama/ollama` mentions KleidiAI.
- **A tempting piece of evidence does not hold.** Ollama issue #13860 — a real 3–10x arm64
  slowdown affecting a Pi 5 and three Qualcomm SoCs for roughly a month — was a missing `-O3`
  CGO flag, not a dispatch failure. The same kernels ran before and after, so this project's
  method would most likely have shown **identical** results across that regression. It must not
  be cited as a case this tool would have caught.
- **vLLM does not share the mechanism.** Its Arm CPU path uses oneDNN + Arm Compute Library and
  no KleidiAI, and its dominant path is GPU, where L3's `gdb`/`lldb` breakpoint technique does not
  transfer. "One tool for three engines" is architecturally false: Ollama vendors llama.cpp
  (one mechanism, not two), and vLLM is a third, unrelated stack.

### The conclusion

The verifier framing stands, narrowed and stated precisely: **a CI gate that proves whether a
build's claimed accelerated kernel actually ran**, aimed at people compiling for new or unusual
Arm silicon and at release pipelines that want a regression gate — not at end users of
`ollama run`. This is a smaller claim than "an accelerator for local LLMs," and it is the one the
measurements support.

---

## Defensibility: what is available, and what was already given away

*Added 2026-08-05, after asking whether this work could be made unique, private, or patentable.*

Three constraints were checked against primary sources before considering any strategy, and two of
them had already decided the question.

**1. The contest forbids it.** The Official Rules state: *"The repository must be public and open
source by including an open source license file"* (MIT or Apache 2.0), available *"for testing,
evaluation and use by the Sponsor, Administrator and Judges until the Judging Period ends"* —
2026-09-04 16:00 PT. Going private is a rules violation, not an option. The rules also confirm the
upside: *"All Submissions remain the intellectual property of the individuals or organizations
that developed them."* The sponsor receives only a non-exclusive licence for judging plus
promotional rights.

**2. Apache-2.0 already granted it away, irrevocably.** This repo has been public under Apache-2.0
since its first commit (`c8834e4`, 2026-08-04 06:37:38 +0800). Section 2 grants every recipient a
*"perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license"*,
and **Section 3 grants a patent licence on the same terms**, covering *"those patent claims
licensable by such Contributor that are necessarily infringed by their Contribution(s)"*. A patent
filed later on the mechanism as published here would already be licensed to anyone using this
code — that is, to precisely the population a patent would be meant to restrain. Making the repo
private cannot retract either grant from anyone who already holds a copy.

**3. The technique is not novel.** See
[`docs/PRIOR-ART-AND-ALTERNATIVES.md`](PRIOR-ART-AND-ALTERNATIVES.md). Pending breakpoints and a
non-halting `stop()` counter are documented GDB/LLDB features; `bpftrace` uprobes answer the same
question with lower overhead and without the `dlopen` ordering problem; Arm's own
`SVE_INST_SPEC` PMU counter answers the hardware-level version with no code at all.

Patent rights outside the US were extinguished by publication before filing — the EPC (Art. 54/55)
and China (Patent Law Art. 24) apply absolute novelty with no self-disclosure grace period. The US
retains a §102(b)(1)(A) one-year window from first disclosure, but combined with the Apache-2.0 §3
grant and the prior art above, the honest expected value is negative. *(Research, not legal
advice.)*

### What is left, and it is not nothing

The scored rubric for this contest is Technological Implementation 40, "WOW" factor 25, Potential
Impact 20, Developer Experience 15. **No criterion rewards defensibility.** What the rubric
rewards is what this project should therefore optimise, and it happens to coincide with the only
durable advantages actually available:

- **Being the credible first discloser, with receipts.** Upstream issue #26547 is filed and dated.
- **A published record of self-correction.** This project retracted a fabricated +57.3% figure and
  later published that its own headline shrank from 4.56x to 1.33x at a realistic model size.
  In benchmarking, that record is the asset. Secrecy would destroy the thing that makes it worth
  anything.
- **The accumulated measurement corpus.** One verified configuration today. Every additional
  (silicon x compiler x engine x flags) record is one a competitor must re-measure on real
  hardware to match. This is the Phoronix/MLCommons shape: the moat is the corpus, not the
  harness. It takes months, not days.
- **Verification of the verifier.** The ground-truth harnesses and the claims registry are
  discipline rather than code, and are the least convenient part to copy.

The mechanism was never the moat. Being right, in public, repeatedly, is.
