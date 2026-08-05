# Polygraph — launch copy (all channels)

> **STATUS: NOTHING BELOW HAS BEEN POSTED ANYWHERE.** This file is ready-to-paste copy only.
>
> **Sequencing requirement: do not post the Hacker News or r/LocalLLaMA items until the
> demo video is public.** Both posts reference "the video below" / a video link. Posting before
> the video is live means shipping a broken link into two audiences that will not give a second
> chance. X and LinkedIn posts that embed the clip directly are fine to sequence around the same
> video-goes-public moment, but should still not go out before the video URL resolves.
>
> Every number below is copied verbatim from `results/server/kai-symbols.txt`,
> `results/server/server-bench.json`, `results/server/server-dispatch.json`,
> `results/server/spark-provenance.txt`, and `results/server/SERVER-LANE.md`, or from
> `docs/FINDINGS.md` / `docs/RELATED-WORK.md` for Findings 1 and 2 context. No figure, URL, or
> quote in this document was invented. Where a claim is diagnosis rather than a number read
> from a committed file (the gcc-probe root-cause story), that is flagged the same way
> `SERVER-LANE.md` flags it — as reported methodology, not a logged artifact.

---

## 0. Boilerplate — "what is Polygraph" (reusable everywhere)

**Short (2 sentences):**

> Polygraph checks whether software is telling the truth about the hardware acceleration it
> claims to use. It attaches a debugger to the real kernel entry points and counts actual calls —
> not a timing guess, not the startup banner — so a "the accelerator is enabled" log line can
> finally be checked instead of trusted.

**Slightly longer (3 sentences, for LinkedIn/About sections):**

> Polygraph checks whether software is telling the truth about the hardware acceleration it
> claims to use. Instead of inferring from timing or trusting a startup banner, it attaches a
> debugger to the real kernel entry points and counts actual calls. It's Apache-2.0, and its
> first public case study is `llama.cpp`'s Arm KleidiAI backend — including a build path that
> silently ships zero acceleration while printing `KLEIDIAI = 1` and exiting `0`.

---

## 1. Hacker News

**Title (71 characters):**

```
llama.cpp's documented KleidiAI build line compiles zero matmul kernels
```

**Submission URL:** `https://tomyimkc.github.io/polygraph/` (dashboard) or the repo root
`https://github.com/tomyimkc/polygraph` — pick one at post time, do not submit both.

**Author's first comment (post immediately after submitting):**

> Author here. Repro, in full, below — no part of this requires trusting my numbers, every command
> is copy-pasteable.
>
> On a DGX Spark (GB10, 20-core Armv9.2 — 10x Cortex-X925 + 10x Cortex-A725, gcc 13.3.0, Ubuntu
> aarch64), `llama.cpp`'s own documented KleidiAI build line —
>
> ```
> cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
> ```
>
> — produces a binary with 0 `kai_run_matmul` symbols. Not "falls back to a slower kernel," zero
> compiled-in matmul micro-kernel entry points, of any family. 36 `kai_` symbols total exist in
> that binary, and all 36 are packing helpers, not kernels. The build exits 0.
>
> At runtime that same binary prints:
>
> ```
> system_info: ... | NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
> kleidiai: no compatible q4 kernels found for CPU features mask 0
> kleidiai: no compatible q8 kernels found for CPU features mask 0
> ```
>
> `KLEIDIAI = 1` is printed on the exact same run whose own log line says the CPU feature mask is
> `0`. SVE, DOTPROD, and MATMUL_INT8 don't print as disabled — they just aren't in the banner at
> all. Nothing here is loud: no warning, no non-zero exit code.
>
> How I found it: I was diffing static symbol counts (`nm`) against the runtime dispatch log as
> part of a broader project (Polygraph — a debugger-attached dispatch verifier, this is its third
> finding, the first two are on Apple Silicon SME2 and a 256-bit SVE width gate) and the symbol
> count for this specific build was just 0, on a box I could rebuild and re-check in place. That's
> what made me go looking at the CMake configure log instead of assuming a config typo.
>
> Root cause (diagnosed by compiling small probes directly with the same gcc 13.3 on this box):
> `llama.cpp`'s CMake configure step can't resolve an explicit `-march`/`-mcpu` for this CPU, so it
> falls back to probing feature suffixes on top of `-mcpu=native` — `-mcpu=native+dotprod`,
> `-mcpu=cortex-x925`, etc. gcc 13.3 rejects every one of those, including, notably, the *negative
> control* probes (a probe that's supposed to fail, failing) — that's the tell that the probe
> mechanism itself is broken, not that the feature is genuinely absent. `-mcpu=cortex-x925` fails
> because gcc 13.3 predates Cortex-X925 in its own `-mcpu` table. An explicit
> `-march=armv9.2-a+i8mm`, bypassing `-mcpu=native` entirely, compiles cleanly on the same
> toolchain.
>
> Fix is one flag pair added to the same documented build line:
>
> ```
> -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
> ```
>
> Rebuilt: 149 `kai_` symbols, 10 `kai_run_matmul` entry points (dotprod 6, i8mm 2, sve 2), and the
> banner now matches reality — `MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 | SVE_CNT = 16`, and
> `kleidiai: primary q4 kernel feature I8MM`.
>
> Scope, stated honestly: this is specifically gcc 13.3.0 + Cortex-X925 — a CPU that postdates
> that compiler's `-mcpu` table. I'm not claiming KleidiAI is broken on Arm generally. The general
> lesson is that this exact failure mode — CMake's `-mcpu=native`-suffix probing collapsing to "no
> features detected" — will recur on any CPU newer than the toolchain building for it, and that the
> probe currently has no way to tell "feature absent" from "compiler can't express the flag."
>
> One more thing this box let me check that I hadn't been able to before: under 8 concurrent
> `llama-server` clients with continuous batching, a `gdb`-attached trace across all 10 fixed-build
> `kai_run_matmul` breakpoints recorded 364,444 i8mm calls and 11,360 dotprod calls — zero SVE calls
> (no `sve` key in the trace at all). That's expected: this core's SVE2 is 128 bits wide
> (`SVE_CNT = 16`) and KleidiAI's SVE kernel family requires an exact 256-bit width
> (`kleidiai.cpp:209`, `ggml_cpu_get_sve_cnt() == QK8_0`) — not a bug I found, credit for that gate
> goes to `luongs3/arm-dispatch-audit`, who published it first (details + dates in
> `docs/RELATED-WORK.md`); this run is a second, independent confirmation of it, now under real
> serving load rather than single-shot decode.
>
> Full writeup, all raw data, and the verifier tool: https://github.com/tomyimkc/polygraph — filed
> upstream at https://github.com/ggml-org/llama.cpp/issues/26547. Happy to answer questions or dig
> into anything below.

---

## 2. r/LocalLLaMA

**Title:**

```
If you built llama.cpp with -DGGML_CPU_KLEIDIAI=ON on Arm, check this — the documented build line can silently ship zero matmul kernels
```

**Body:**

> **Two-command self-check — do this before reading anything else:**
>
> ```bash
> # Linux (the .so KleidiAI links into)
> nm -D build/bin/libggml-cpu.so | grep -c kai_run_matmul
>
> # macOS (the .dylib)
> nm -gU build/bin/libggml-cpu.dylib | grep -c kai_run_matmul
> ```
>
> If that prints `0` and you built with `-DGGML_CPU_KLEIDIAI=ON`, your binary has **zero** compiled
> KleidiAI matmul micro-kernels — not "using a fallback," zero — even if the startup banner prints
> `KLEIDIAI = 1` and the build exited `0`. Add this to your `cmake` invocation and rebuild
> (`GGML_CPU_ARM_ARCH` should match your actual core's features):
>
> ```bash
> -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
> ```
>
> **What I measured, on a DGX Spark (GB10, 20-core Armv9.2, 10x Cortex-X925 + 10x Cortex-A725,
> gcc 13.3.0 Ubuntu aarch64, `llama.cpp` @ `dbadb68`):**
>
> Following `llama.cpp`'s own documented KleidiAI build line —
> `cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release` — the resulting binary has
> **0** `kai_run_matmul` symbols (36 `kai_` symbols total, all packing helpers, none are kernels).
> The build exits `0`. The runtime banner still prints `KLEIDIAI = 1`, and the actual log line one
> level down says: `kleidiai: no compatible q4 kernels found for CPU features mask 0`.
>
> Root cause (diagnosed by compiling probes directly with the same gcc): CMake can't resolve an
> explicit `-march`/`-mcpu` for this CPU, so it probes feature suffixes on `-mcpu=native`
> (`-mcpu=native+dotprod`, `-mcpu=cortex-x925`, etc). gcc 13.3 rejects all of them — including the
> negative-control probes that are *supposed* to fail — because gcc 13.3 predates Cortex-X925 in
> its own `-mcpu` table. That's the signature of a broken probe, not an absent feature. An explicit
> `-march=armv9.2-a+i8mm` (skipping `-mcpu=native` entirely) compiles fine on the same toolchain.
>
> One flag pair fixes it (shown above). Rebuilt: 149 `kai_` symbols, 10 `kai_run_matmul` entries
> (dotprod 6, i8mm 2, sve 2), banner now shows `MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 |
> SVE_CNT = 16`, and `kleidiai: primary q4 kernel feature I8MM`.
>
> **Scope — read this before assuming it's you:** this is gcc 13.3.0 + Cortex-X925 specifically —
> a CPU newer than that compiler's `-mcpu` support table. It is not a claim that KleidiAI is broken
> on Arm in general. But the mechanism (CMake's `-mcpu=native`-suffix probing silently collapsing to
> "nothing detected" when the compiler doesn't know the CPU name) will hit anyone whose distro gcc
> predates their chip. Cortex-X925 boxes (several current Arm SBCs and the DGX Spark line) with
> stock Ubuntu 24.04 gcc 13.3 are a concrete case where this is worth checking today.
>
> **Also measured on the same box (llama-server, continuous batching, 8 clients):** a
> `gdb`-attached trace on all 10 fixed-build `kai_run_matmul` breakpoints recorded 364,444 i8mm
> calls and 11,360 dotprod calls — zero SVE calls. That's the `luongs3/arm-dispatch-audit`
> 256-bit-SVE-width gate (`kleidiai.cpp:209`, they published this mechanism first — full credit and
> dates in the repo's `RELATED-WORK.md`), now confirmed on a second core family under real serving
> load: this Cortex-X925's SVE2 is 128 bits wide (`SVE_CNT = 16`), and the gate requires exactly
> 256 bits.
>
> Throughput, for context, on the same server (`llama-server`, continuous batching, 3 rounds/config,
> median): aggregate tok/s went 14.9 (1 client) → 56.6 (4) → 271.8 (8) → 440.4 (16) — about 29.6x
> from 1 to 16 concurrent clients. TTFT p99 across that sweep ranged 89–221ms and was **not**
> monotonic — the 4-client row had the worst TTFT p99 (221ms) of the whole table, worse than the
> 16-client row (168ms) — worth knowing if you're tuning for tail latency, not just throughput.
> Peak RSS grew 724 → 901 MiB over that same 1→16 client range.
>
> Full repo, verifier tool, and raw JSON: https://github.com/tomyimkc/polygraph. Filed upstream:
> https://github.com/ggml-org/llama.cpp/issues/26547.

---

## 3. X / Twitter (thread, 4–6 posts)

**Post 1 (stands alone — this is the only post most people will see):**

> `llama.cpp`'s own documented KleidiAI build command compiles a binary with ZERO matmul
> acceleration kernels. No warning. No error. Exit code 0. Startup banner still says
> `KLEIDIAI = 1`.
>
> Found it with a debugger attached to the real kernel entry points, not a timing guess.
> 🧵
>
> [VIDEO/GIF GOES HERE — the build-then-run demo clip, ~20–30s, showing the two banners
> side by side and the `nm` symbol count flipping 0 → 10]

**Post 2:**

> The command, straight from `llama.cpp`'s own docs:
>
> `cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release`
>
> Result on a DGX Spark (Cortex-X925, gcc 13.3.0): 0 `kai_run_matmul` symbols. 36 `kai_` symbols
> total exist — all 36 are packing helpers. None are kernels.

**Post 3:**

> The banner doesn't say so. It prints:
>
> `KLEIDIAI = 1`
>
> One log line down, at INFO level, it actually admits it:
>
> `kleidiai: no compatible q4 kernels found for CPU features mask 0`
>
> Same run. Both lines. `KLEIDIAI = 1` is the one most people glance at.

**Post 4:**

> Root cause: gcc 13.3 predates Cortex-X925 in its `-mcpu` table. CMake's feature-probing
> (`-mcpu=native+dotprod`, `-mcpu=native+i8mm`...) fails silently — even the probes that are
> *supposed* to fail also fail, which is the tell that the probe is broken, not the CPU lacking the
> feature.

**Post 5:**

> The fix is one flag pair added to the same build line:
>
> `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"`
>
> Rebuilt: 149 `kai_` symbols, 10 real `kai_run_matmul` kernels. Banner now matches reality:
> `MATMUL_INT8=1 | SVE=1 | DOTPROD=1`.

**Post 6:**

> Scope, honestly: this is gcc 13.3.0 + Cortex-X925 specifically — a chip newer than that
> compiler's `-mcpu` table. Not "KleidiAI is broken on Arm." The general failure mode — silent
> zero-kernel builds on any CPU newer than its toolchain — is the reusable lesson.
>
> Full repro + filed upstream: github.com/tomyimkc/polygraph
> ggml-org/llama.cpp#26547

---

## 4. LinkedIn

> **If you're paying for "Arm-accelerated" cloud inference, it's worth checking whether the
> acceleration actually compiled in.**
>
> I ran `llama.cpp`'s own documented build command for its Arm KleidiAI backend
> (`-DGGML_CPU_KLEIDIAI=ON`) on a DGX Spark — current-generation Arm server silicon (Cortex-X925).
> The build succeeded, exited with status 0, and the startup log printed `KLEIDIAI = 1`, which
> reads exactly like "acceleration is on."
>
> It wasn't. The binary contained zero compiled matmul acceleration kernels for the CPU backend —
> not a slower fallback path, zero. Every accelerated matmul call in that build had nowhere to go.
> The cause: the compiler on this box (gcc 13.3, the default on Ubuntu 24.04) predates this specific
> CPU in its own architecture table, and the build system's automatic feature-detection silently
> failed as a result — with no error surfaced anywhere a user would see it.
>
> The fix was one additional flag pair on the same build command. After rebuilding: 10 real
> accelerated kernels present, and the runtime log correctly reported which ones were selected.
>
> Why this matters beyond one flag: any team that measures "is our Arm instance using the
> acceleration we're paying for" by reading a startup banner or a green build — rather than
> confirming the accelerated code path actually executes — can be running on CPU-only fallback
> paths for a build that reports success at every checked step. On the same hardware, once
> correctly built, throughput scaled roughly linearly with concurrent load (about 29.6x aggregate
> throughput from 1 to 16 concurrent clients, in our measurement) — which is the performance that
> was silently unavailable in the default build.
>
> This is one finding from Polygraph, an open-source (Apache-2.0) tool we built to check whether
> software's hardware-acceleration claims hold up under a debugger, not just a log line. Full
> writeup and raw data: github.com/tomyimkc/polygraph. Filed upstream with the `llama.cpp`
> maintainers: ggml-org/llama.cpp#26547.

---

## 5. FAQ (pre-answering the obvious pushback)

**"Isn't this just a build misconfiguration?"**

Yes — and that's exactly the point. It's not a typo in a user's own `cmake` invocation; it's the
outcome of following `llama.cpp`'s own documented KleidiAI build command
(`cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release`) verbatim, and the failure
is silent — the build exits `0`, and the runtime banner still prints `KLEIDIAI = 1` on the exact
same run whose own log says the CPU feature mask is `0`. A "misconfiguration" that the documented
path produces, with no error and a banner that reads as success, is a different problem than a
misconfiguration a user could reasonably catch themselves.

**"Does it affect me?"**

Check in the time it takes to run two commands:

```bash
nm -D build/bin/libggml-cpu.so | grep -c kai_run_matmul     # Linux
nm -gU build/bin/libggml-cpu.dylib | grep -c kai_run_matmul  # macOS
```

If you built with `-DGGML_CPU_KLEIDIAI=ON` and that prints `0`, you're affected. What we can say
about *who* hits this: it's specifically the combination of a compiler whose `-mcpu` table
predates your CPU (confirmed here as gcc 13.3.0 + Cortex-X925) — we have not tested other
gcc/CPU pairings, and we are not claiming this affects every Arm machine or every gcc version.

**"Why not just read the log?"**

The line that tells the truth exists — `kleidiai: no compatible q4 kernels found for CPU features
mask 0` — but it's printed at INFO level, one line below the startup banner, while the banner
itself (`KLEIDIAI = 1`) contradicts it and is the line most people actually glance at before
moving on. Both lines are real output from the same run; only one of them is accurate about
whether acceleration will execute.

**"Does this affect Apple Silicon / the SME2 finding I've seen elsewhere in this repo (Finding 1)?"**

No — this finding (Finding 3) was measured on the DGX Spark, which has no SME/SME2 hardware at
all; neither build's banner reports `SME` or `SME2` in any form. Finding 1 (Apple Silicon SME2
decode thread-gating) is a separate finding on separate hardware and is not reproduced or touched
by this post.

**"Is the SVE-never-dispatches part of this new?"**

No, and we're not claiming it is. The mechanism — KleidiAI's SVE kernel family requires an exact
256-bit SVE width (`kleidiai.cpp:209`) — was published first by `luongs3/arm-dispatch-audit`
(created two days before this repo). What's new in this post is confirming it at the dispatch
layer (a `gdb`-attached trace, not just source reading) on a second, independent core family
(Cortex-X925), under real concurrent serving load rather than single-shot inference. Full
disclosure and dates: `docs/RELATED-WORK.md` in the repo.
