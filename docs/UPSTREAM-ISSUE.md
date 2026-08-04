# Upstream issue for ggml-org/llama.cpp

Status: **FILED 2026-08-04 — [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547)** (open).

This file is retained as the source of record for what was reported and why; the text below
the `---` is verbatim what was submitted. Suggested labels were `bug`, `kleidiai`, `ARM`,
`documentation` — triage is the maintainers' call.

Everything under "Evidence" was produced by code in this repository
(`tomyimkc/arm-dispatch-ledger`, Apache-2.0) and can be reproduced with the commands shown.
The offer in "Suggested fix" stands: if the maintainers confirm the direction, we will send a
PR adding the one-shot `GGML_LOG_WARN`, mirroring the precedent already in the codebase for
the weight-type fallback.

---

## Title

KleidiAI: SME2 dispatch is silently thread-gated on Apple Silicon (banner still says
"SME2" even when the kernel never runs); SVE path is architecturally unreachable on
128-bit-SVE2 cores

## Environment

| | |
|---|---|
| llama.cpp commit | `dbadb68eecdfb3ab0e86872d011738fc937f0364` |
| Build flags | `-DGGML_CPU_KLEIDIAI=ON` (KleidiAI CPU backend) |
| Hardware (measured directly) | Apple M4 Max, macOS 27, 16 cores (12P+4E), Apple clang 21 |
| ISA (measured via `sysctl`) | `FEAT_SME=1`, `FEAT_SME2=1`, `sme_max_svl_b=64` (512-bit SVL), `FEAT_I8MM=1`, `FEAT_BF16=1`, `FEAT_DotProd=1`; **`FEAT_SVE` absent** (Apple ships SME2 without non-streaming SVE) |
| Model | `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (Apache-2.0) |
| Hardware for Finding 2 (**measured**) | GitHub-hosted `ubuntu-24.04-arm` runner — Neoverse-N2, 4 cores, SVE2 @ 128-bit, i8mm, bf16, no SME. Free and re-runnable by anyone; CI run [`30862916023`](https://github.com/tomyimkc/arm-dispatch-ledger/actions/runs/30862916023) |

## Summary

`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` has two dispatch gates that never surface to
the user, even though llama.cpp's own startup banner and per-load log lines strongly
imply the accelerated kernel is in use:

1. **A hardcoded, brand-string-keyed SME thread cap** (`detect_num_smcus()`) silently
   collapses SME2 matmuls back to the NEON/DotProd kernel once the thread count exceeds
   that cap, for the exact shapes (`ne11 < 128`, i.e. single-token decode) where the
   "hybrid" rescue path can't engage. The banner and log never change to reflect this.
2. **The SVE feature gate requires an exact 256-bit vector length**
   (`ggml_cpu_get_sve_cnt() == QK8_0`), which structurally excludes every current
   128-bit-SVE2 Arm core (Cortex-X925/DGX Spark, Neoverse-N2/GitHub's
   `ubuntu-24.04-arm` runner) from the SVE kernel family, with no log line explaining
   why.

Both are real, reproducible, and (as far as we could find — see "Prior art" below)
undocumented. We think both are worth a one-line runtime log message so a user
debugging "why is this slower than I expected" doesn't need `lldb` to find out, which
is what we had to do.

## Finding 1 — SME2 silently thread-gated, with a "hybrid" nuance

### Reproduction

```bash
# Build (see this repo's kernels/CMakeLists.txt comment for why
# -mcpu=apple-m4 is required over a generic -march=armv9-a+sme2 target on
# Apple Silicon; not relevant to this llama.cpp issue but included for
# completeness in case a reviewer rebuilds from source).
cmake -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# An lldb regex breakpoint on every KleidiAI matmul entry point, anchored
# (^kai_run_matmul) and set to auto-continue on every hit, so `breakpoint
# list`'s per-location hit count is a real call count -- not just a
# stop/no-stop signal. (An UNANCHORED pattern like "kai_run_matmul.*sme"
# also matches the compiler-generated template thunk that tail-calls into
# the real symbol, silently doubling the count -- we hit this ourselves:
# 1992 vs. the correct 996 for the same run. Anchoring avoids it.)
breakpoint set --func-regex "^kai_run_matmul"
breakpoint command add 1 -o "continue"
process launch -- -m q05.gguf -p "<PROMPT>" -n <N> -no-cnv -st --simple-io -t <THREADS>
breakpoint list
quit
```

run as `lldb -b -s probe.lldb -- ./build/bin/llama-cli`, one process per
thread count (this anchored pattern resolves to 18 breakpoint locations in this
binary — every exported `kai_run_matmul_*` symbol KleidiAI compiled in, across
all kernel families, not just SME). We wrapped this into a small harness
(`tools/verify_dispatch.py` in the repo linked below) that sweeps thread
counts automatically and classifies each run's hit counts into a verdict —
the raw command above is the minimal reproduction if you'd rather not clone
that.

### Observed vs expected

Real per-config `kai_run_matmul*` call counts (anchored breakpoint,
auto-continue), Apple M4 Max, `-t <threads>`, `Qwen2.5-0.5B-Instruct-Q4_0.gguf`:

| threads | workload (`ne11`) | expected (banner says `SME2=1`) | **observed SME2 hits / other-kernel hits** | verdict |
|---:|---|---|---:|---|
| 1 | decode (short, `ne11==1`) | SME2 kernel runs | 996 / 0 | SME2 dispatched |
| 2 | decode | SME2 kernel runs | 5826 / 0 | SME2 dispatched |
| 4 | decode | SME2 kernel runs | **0 / 15936** | **silent fallback** |
| 8 | decode | SME2 kernel runs | **0 / 31871** | **silent fallback** |
| 16 | decode | SME2 kernel runs | **0 / 51214** | **silent fallback** |
| 1 | prefill (long, `ne11>=128`) | SME2 kernel runs | 660 / 0 | SME2 dispatched |
| 2 | prefill (long) | SME2 kernel runs | 3853 / 0 | SME2 dispatched |
| 4 | prefill (long) | SME2 kernel runs | 2232 / 6712 | **hybrid** — some SME2, some NEON |
| 8 | prefill (long) | SME2 kernel runs | 1547 / 13692 | **hybrid** |
| 16 | prefill (long) | SME2 kernel runs | 1403 / 21509 | **hybrid** |

Across every one of these rows, `system_info:` still prints `SME = 1 | SME2 = 1 |
KLEIDIAI = 1`, and the load-time log still prints `kleidiai: primary q4 kernel feature
SME2` and `kleidiai: SME2 enabled (runtime-detected SME cores=2)`. Both lines are
compile-time / selection-time signals; neither reflects what actually executed. A user
running with `-t 8` (a completely reasonable default on a 16-core machine) sees every
indication that SME2 is active for decode, and it never runs.

### Root cause (exact lines, this commit)

`sme_thread_cap` is derived from a hardcoded per-chip table, not a runtime capability
query:

```cpp
// ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:148-169 (detect_num_smcus(), __APPLE__ branch)
// table for known M4 variants. Users can override via GGML_KLEIDIAI_SME=<n>.
char chip_name[256] = {};
size_t size = sizeof(chip_name);
if (sysctlbyname("machdep.cpu.brand_string", chip_name, &size, nullptr, 0) == 0) {
    const std::string brand(chip_name);
    struct ModelSMCU { const char *match; size_t smcus; };
    static const ModelSMCU table[] = {
        { "M4 Ultra", 2 },
        { "M4 Max",   2 },
        { "M4 Pro",   2 },
        { "M4",       1 },
    };
    for (const auto &e : table) {
        if (brand.find(e.match) != std::string::npos) {
            return e.smcus;
        }
    }
}
return 0;
```

That value becomes the cap actually enforced at dispatch time:

```cpp
// ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:300
ctx.sme_thread_cap = (ctx.features & CPU_FEATURE_SME) ? sme_cores : 0;
```

And the per-op dispatch decision (this is the part our first draft of this issue got
wrong — see the correction below) is:

```cpp
// ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:1094-1113
const int sme_cap_limit = ctx.sme_thread_cap;
const bool use_hybrid = sme_cap_limit > 0 &&
                         runtime_count > 1 &&
                         nth_total > sme_cap_limit;
size_t min_cols_per_thread = /* ne01 / nth_total, floor 1 */;
const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);
const bool hybrid_enabled = use_hybrid && !too_small_for_hybrid;

if (!hybrid_enabled) {
    int chosen_slot = 0;
    if (too_small_for_hybrid && sme_slot != -1) {
        chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
    } else if (runtime_count > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
        chosen_slot = 1; // <- collapses to the non-SME (NEON) slot
    }
    ...
}
```

So the real rule (confirmed by the lldb sweep above) is: SME2 dispatches if **either**
(a) `n_threads <= sme_thread_cap`, or (b) the "hybrid" rescue path engages, which
additionally requires `ne11 >= 128` (batch size / columns of `src1`) **and**
`ne01/n_threads >= 2`. Decode is always `ne11 == 1`, so condition (b) can never help it
— decode falls all the way back to NEON above the cap, on every thread count and every
prompt. Prefill of a long-enough prompt satisfies `ne11 >= 128` and gets the hybrid
path, so it keeps *some* SME2 throughput even above the cap — just not all of it.

**Correction to our own first draft:** we initially reported "SME2 never fires above 2
threads" as a flat rule from a 4-token-prompt-only test. That is incomplete — it is
true for decode-shaped ops and false for large-batch prefill, for the reason above. We
mention this because it is exactly the kind of nuance a one-line log message (below)
would have surfaced immediately, instead of requiring an `lldb` breakpoint sweep across
two different prompt shapes to discover.

## Finding 2 — SVE kernel family requires an exact 256-bit vector length

```cpp
// ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:209
((ggml_cpu_has_sve() && ggml_cpu_get_sve_cnt() == QK8_0) ? CPU_FEATURE_SVE : CPU_FEATURE_NONE)
```

`QK8_0` is 32 bytes (256 bits). Any core with genuine, working SVE2 at a different
vector length — most notably every current 128-bit-SVE2 Armv9 core, including Cortex-
X925 (DGX Spark) and Neoverse-N2 (GitHub's free `ubuntu-24.04-arm` runner) — fails this
`==` check and gets `CPU_FEATURE_NONE` for SVE, silently, with no log line. `ggml_cpu_has_sve()`
can be true while `CPU_FEATURE_SVE` in this context is never set.

**Measured, not just inferred.** We confirmed this end to end on GitHub's own free
`ubuntu-24.04-arm` runner (Neoverse-N2, 4 cores), so it reproduces at zero cost. From CI
run [`30862916023`](https://github.com/tomyimkc/arm-dispatch-ledger/actions/runs/30862916023):

- `/proc/cpuinfo` advertises `sve sve2 sveaes svebitperm svesha3 svesm4 svei8mm svebf16
  i8mm bf16` — the hardware genuinely has SVE2.
- A static scan of the built `libggml-cpu.so` shows the SVE kernels are compiled in:
  `kai_symbols_by_family = {dotprod: 6, i8mm: 2, sve: 2}`, plus 26,629 SVE z-register
  operands in the disassembly.
- The load-time log nonetheless reports `primary q4 kernel feature I8MM` and `primary q8
  kernel feature I8MM` — never SVE.
- Debugger breakpoints on every `kai_run_matmul*` entry point confirm execution is
  `i8mm` / `dotprod`; the SVE family is never entered.

So the SVE kernels ship, the silicon supports SVE2, and the dispatcher still cannot
select them.

We have **not** run our verifier against a 256-bit-SVE2 core to observe a genuine
positive case, so we cannot rule out that the exact-width requirement is intentional —
the SVE microkernels may be hand-tuned for one vector length rather than being
vector-length-agnostic. We would value the maintainers' read on that; see the question
in "Suggested fix" below. If it is intentional, a single log line saying so would still
have saved us a debugger session.

## Suggested fix

We are not proposing a specific patch here (happy to send one if the direction below
is welcome) — just the smallest change that would have saved us the `lldb` session:

1. **Finding 1:** emit one `GGML_LOG_WARN` at context-init time (same place the
   existing "kleidiai: SME2 enabled (runtime-detected SME cores=N)" line already lives)
   when `ctx.sme_thread_cap > 0` and the effective thread count (`GGML_TOTAL_THREADS`
   hint, or the thread count llama.cpp is about to run with) exceeds it — something
   like:

   ```
   kleidiai: SME2 thread cap is 2 on this CPU, but N threads were requested;
   decode-shaped ops (batch size < 128) will run on NEON/DotProd instead of SME2,
   larger-batch ops may use a hybrid SME2+NEON split. Set GGML_KLEIDIAI_SME=<=2 to
   pin single-slot SME2, or ignore this if NEON's throughput at N threads already
   meets your needs (our own measurements on M4 Max found NEON@8 threads actually
   *beats* SME2 for prefill on a 0.5B model, but SME2 wins outright for decode --
   your workload's phase mix matters more than the cap itself).
   ```

   This mirrors the precedent already in this codebase for the *weight-type* silent
   fallback (the one-shot `GGML_LOG_WARN` for non-Q4_0/Q8_0 tensors), so it would be
   consistent with existing style, not a new pattern.

   Optionally, a `GGML_KLEIDIAI_DEBUG_DISPATCH=1` env var that logs the chosen slot
   per matmul call (rate-limited / first-N-calls only) would remove the need for
   `lldb` entirely for anyone debugging this in the future — that is effectively what
   our project's `tools/verify_dispatch.py` L3 tier does externally today, and it
   would be strictly better done from inside the process.

2. **Finding 2:** a genuine question rather than a confident suggestion — is the exact
   `== QK8_0` (256-bit) requirement a hard correctness constraint of the current SVE
   microkernels (in which case, a one-line `GGML_LOG_INFO` explaining *why* 128-bit-
   SVE2 was excluded would help), or would `>=` be safe, or is a second,
   vector-length-agnostic SVE kernel variant the right long-term answer for the
   128-bit-SVE2 cores that are becoming the common case on Armv9 (Cortex-X925,
   Neoverse-N2/V2)? We would like to help verify whichever direction the maintainers
   confirm is correct — we have a DGX Spark and free-tier `ubuntu-24.04-arm` CI
   available to test a candidate patch on real 128-bit-SVE2 hardware.

## Methodology note (so this issue is falsifiable)

The counts above come from an **anchored** (`^kai_run_matmul`), **auto-continuing**
breakpoint, so `breakpoint list`'s per-location hit count is a genuine call count, not
just a stop/no-stop signal. We flag this explicitly because our own first attempt used
an unanchored pattern (`kai_run_matmul.*sme`) and a non-auto-continuing breakpoint,
which (a) double-counted every real call — the unanchored pattern also matches the
compiler-generated template thunk that tail-calls into the real exported symbol, so we
saw 1992 hits where the correct count was 996 for the same run — and (b) could only
report "fired at least once," not a call count. Both are worth knowing if anyone tries
to reproduce this with their own quick lldb one-liner. Every number here was produced
by code in
[`tomyimkc/arm-dispatch-ledger`](https://github.com/tomyimkc/arm-dispatch-ledger) (a
small, Apache-2.0, dependency-free verification harness we built specifically to make
this compile-time-vs-dispatch-time gap checkable without a fresh debugger session each
time); happy to share the raw ledger JSON or open a PR if useful.

## Prior art

- llama.cpp already added a one-shot `GGML_LOG_WARN` for the unrelated *weight-type*
  silent-fallback case (non-Q4_0/Q8_0 tensors going through KleidiAI) — precedent for
  the style of fix suggested above.
- Issue #22182 independently notes `ggml_cpu_has_sme()` is a compile-time-only check,
  which is consistent with (but does not by itself describe) the thread-cap /
  `ne11 >= 128` runtime dispatch rule documented here.
- We could not find any existing issue, PR, or doc describing the `sme_thread_cap` /
  hybrid-dispatch rule itself.

Thank you for KleidiAI and for llama.cpp's CPU backend generally — this issue is meant
in the spirit of "the banner should not be able to lie by omission," not a complaint
about the underlying design, which we think is reasonable given the hardware
constraints (Apple's 2-SME-core M4 Max/Pro/Ultra topology is a real limit, not a bug).
