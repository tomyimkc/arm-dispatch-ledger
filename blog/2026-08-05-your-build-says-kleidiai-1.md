---
title: "Your build says KLEIDIAI = 1. Your CPU feature mask is 0."
date: 2026-08-05
tags: [arm, kleidiai, llama.cpp, gcc, build-systems, dgx-spark]
canonical: https://github.com/tomyimkc/polygraph
---

# Your build says KLEIDIAI = 1. Your CPU feature mask is 0.

Here is what a `llama.cpp` build printed at startup, on a machine we build on regularly,
after following the project's own documented instructions with no unusual flags:

```
system_info: ... | NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: no compatible q4 kernels found for CPU features mask 0
```

`KLEIDIAI = 1` on the first line. `CPU features mask 0` on the third. Same process, same
startup, same log stream. One line says the accelerator is on. The next line, from the
accelerator's own code, says it found zero usable CPU features to select a kernel with.
Both are true simultaneously, and the build exited 0.

## How we noticed

We were setting up a serving benchmark on a DGX Spark — `llama-server` under concurrent
client load — and before running the load test we do a cheap sanity check: count how
many `kai_run_matmul_*` symbols exist in the shared library KleidiAI actually links
into, so we know there's something for a debugger to break on later. The command was
`nm -D build/bin/libggml-cpu.so | grep -c kai_run_matmul` (that symbol table lives in
`libggml-cpu.so`, not in the `llama-server` executable itself — `nm` on the executable
will read `0` regardless of whether your build is broken, because `llama-server` never
references those symbols directly). It came back **zero**. Not low, zero. The build had followed
`llama.cpp`'s documented KleidiAI line —

```
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
```

— compiled cleanly, and linked a binary with no KleidiAI matmul micro-kernels at all. `nm`
found 36 `kai_` symbols total in that binary, and every one of them was a packing helper
(`kai_lhs_quant_pack_*`, `kai_rhs_pack_*`) — the plumbing that would feed a matmul kernel,
with no matmul kernel behind it to feed.

## Diagnosis: the probe is broken, not the CPU

Once the symbol count is zero, the next question is why. `llama.cpp`'s CMake configure
step can't resolve an explicit `-march`/`-mcpu` for this target, so it logs a fallback and
starts probing feature suffixes on top of `-mcpu=native` — appending things like
`+dotprod`, `+i8mm`, `+sve` and compiling a tiny test file for each, to see which ones the
toolchain accepts. We reproduced that probing directly, compiling each candidate flag with
the same `gcc 13.3.0` on the same box:

| Probe | Result |
|---|---|
| `-mcpu=native` | compiles |
| `-mcpu=native+dotprod` | **rejected** |
| `-mcpu=cortex-x925` | **rejected** |
| `-march=armv9.2-a+i8mm` | compiles |

`-mcpu=native` alone is fine. The moment CMake appends a feature suffix to it —
`-mcpu=native+dotprod`, the exact form its probing logic generates — gcc 13.3 rejects it.
`-mcpu=cortex-x925` is rejected too, for a simpler reason: gcc 13.3.0 predates
Cortex-X925's addition to gcc's own `-mcpu` table, so it doesn't know the name.

The detail that actually pins this down is the *negative* control. `llama.cpp`'s CMake
also runs a check meant to confirm a feature is correctly reported absent —
`GGML_MACHINE_SUPPORTS_nodotprod` — and on this pairing, that check fails to compile too.
When a probe for "feature present" fails and the probe for "feature absent" *also* fails,
the CPU isn't the variable that changed — the probe itself can't compile anything in this
suffixed form. Every suffix CMake tries collapses to "no," feature detection reports
nothing, and KleidiAI compiles against a CPU mask of zero. `-march=armv9.2-a+i8mm`, which
sidesteps `-mcpu=native` and its suffix probing entirely, compiles without incident on the
identical toolchain — which is the tell that the hardware and compiler are both capable,
and only the detection path between them is broken.

## The fix

One flag pair, added to the same documented command, on the same commit
(`llama.cpp` @ `dbadb68`), same compiler, same machine:

```
-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
```

Before: 36 `kai_` symbols, 0 `kai_run_matmul`.
After: 149 `kai_` symbols, 10 `kai_run_matmul` (6 dotprod, 2 i8mm, 2 sve).

The banner changes shape, not just value:

```
system_info: ... | NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 | SVE_CNT = 16 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: primary q4 kernel feature I8MM
kleidiai: primary q8 kernel feature I8MM
```

`MATMUL_INT8`, `SVE`, `DOTPROD`, and `SVE_CNT` — absent from the broken build's banner
entirely, not printed as `= 0` — are now present, and the `kleidiai:` lines flip from
three "no compatible kernels found" failures to two successful kernel selections.
`KLEIDIAI = 1` is unchanged. It was never the signal that was wrong.

## Why this is invisible

A binary with zero KleidiAI matmul kernels doesn't crash and doesn't error. `ggml` still
has generic NEON and repacking code paths, so the model still loads, still generates
correct tokens, and the process still looks busy under `top`. If you build once, run it
once, and the numbers look plausible, there is nothing in a wall-clock timing to tell you
the fast kernel never ran — you'd have to already know what the accelerated number should
be, on this exact model and hardware, to notice its absence. The build exits 0. The banner
prints `KLEIDIAI = 1`. Everything about the experience of running this binary says it
worked.

That gap — between what a log line claims and what the hardware actually executed — is
why [Polygraph](https://github.com/tomyimkc/polygraph) exists: a debugger attached to the
real `kai_run_matmul_*` entry points in a running process, counting actual calls, instead
of trusting a startup banner or a timer. What actually flagged this
particular bug was cheaper than a full debugger trace — a plain symbol count landing at
zero is what sent us to the CMake log in the first place. The debugger comes in once
there's something to break on: with the fixed build compiled, an 8-client
`llama-server` load test with breakpoints on all 10 `kai_run_matmul` symbols recorded
375,804 real calls, split 364,444 i8mm / 11,360 dotprod — and zero on either of the two
compiled-in SVE kernels, because Cortex-X925's SVE2 is 128 bits wide and KleidiAI's
dispatcher only enables its SVE family at an exact 256-bit width. That specific gate was
published first by [`luongs3/arm-dispatch-audit`](https://github.com/luongs3/arm-dispatch-audit),
two days before our repository existed; we're not claiming priority on it, only reporting
that our trace on this CPU lands on the same prediction they made.

## Check your own build in two commands

First, count the actual matmul kernels in the shared library, not the packing helpers —
and note this has to point at `libggml-cpu`, not at the `llama-server`/`llama-cli`
executable, which never references these symbols directly and will read `0` either way:

```
# Linux
nm -D build/bin/libggml-cpu.so | grep -c kai_run_matmul
# macOS
nm -gU build/bin/libggml-cpu.dylib | grep -c kai_run_matmul
```

If that's `0`, you have no accelerated matmul path, regardless of what the startup banner
says. Second, run the binary and read the `kleidiai:` log lines at startup, not just the
feature flag line:

```
./build/bin/llama-server -m your-model.gguf ... 2>&1 | grep -E "system_info:|kleidiai:"
```

Look specifically for `no compatible ... kernels found for CPU features mask 0` versus a
`primary ... kernel feature` line. `KLEIDIAI = 1` will be identical either way — it tells
you the option was compiled in, not that it selected anything. If your build comes back
empty on the symbol count or "mask 0" on the log lines, pass `-DGGML_NATIVE=OFF` with an
explicit `-DGGML_CPU_ARM_ARCH` naming your target's ISA extensions, instead of letting
`-mcpu=native` guess.

## Scope

This is one compiler and one core: gcc 13.3.0 building for Cortex-X925 on a DGX Spark. It
is not a claim that KleidiAI is broken on Arm in general, and we haven't reproduced it on
any other chip. The reusable part is the mechanism, not the specific hardware: gcc 13.3
predates Cortex-X925 in its own `-mcpu` table, so `llama.cpp`'s `-mcpu=native`-plus-suffix
feature probing has nothing valid to test against on this pairing, and it fails silently
rather than loudly. Any distribution's default compiler paired with a CPU newer than that
compiler's `-mcpu` table is a candidate for the same outcome, on this exact build line,
until upstream changes how the feature probe reports failure.

Findings 1 (SME2 thread-gating on Apple Silicon) and 2 (the SVE width gate above) from
this same project are already filed upstream:
[ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547). This
build-flag finding was measured after that issue was filed and hasn't been written up as
its own upstream report yet. Full evidence, the fixed build's throughput under real
concurrent load (14.9 → 440.4 tok/s aggregate, 1 to 16 clients, on `results/server/`),
and reproduction scripts: [github.com/tomyimkc/polygraph](https://github.com/tomyimkc/polygraph)
(Apache-2.0). Live dashboard: <https://tomyimkc.github.io/polygraph/>.
