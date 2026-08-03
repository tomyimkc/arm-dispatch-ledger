# Findings — deep root-cause analysis

> **Authoritative source:** `results/GROUND-TRUTH-DISPATCH.md`. Any statement in this document that
> contradicts that file is wrong and should be corrected to match it. This document expands on it
> with the full methodology, the correction history, and the evidence trail; it does not supersede
> it.

Target: `llama.cpp` @ `dbadb68eecdfb3ab0e86872d011738fc937f0364`, built
`-DGGML_CPU_KLEIDIAI=ON`. Source file throughout: `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`.

---

## Finding 1 — SME2 dispatch is gated by a hardcoded per-chip thread cap, rescued only by a batch-size-dependent hybrid path

### The question

Does `llama.cpp`'s KleidiAI backend actually execute an SME2 kernel when its own banner and log say
it will? The answer is "it depends on two things a user never sees: which Apple chip the binary
thinks it's on, and how many tokens are in the current matmul batch."

### The evidence chain

**L1 (static, `nm`/`otool` on `libggml-cpu.dylib`):** 264 `kai_*` symbols present, 17
`kai_run_matmul_*` entry points, split by family: `sme2` (6), `sme` (3), `dotprod` (6), `i8mm` (2).
The SME2 kernels are unambiguously compiled in. This alone proves nothing about runtime behavior.

**L2 (selection, `llama-cli --verbose` log), identical on every configuration below:**

```
system_info: n_threads = N (n_threads_batch = N) / 16 | CPU : NEON = 1 | ARM_FMA = 1 |
  FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | SME = 1 | SME2 = 1 | ACCELERATE = 1 |
  KLEIDIAI = 1 | REPACK = 1 |
kleidiai: primary q4 kernel feature SME2
kleidiai: primary q8 kernel feature SME2
kleidiai: primary f32 kernel feature SME2
kleidiai: SME2 enabled (runtime-detected SME cores=2)
```

This is printed **verbatim, byte-for-byte the same,** whether or not SME2 ever executes. It is a
compile-time-feature-flag readout plus a one-time "this is the preferred kernel *if selectable*"
log line — neither is conditioned on the actual per-call dispatch decision made deep in the matmul
op handler.

**L3 (dispatch, `lldb` regex breakpoint on `kai_run_matmul.*sme`, 18 resolved locations, real
per-symbol hit counts via an auto-continuing breakpoint command):** this is where the two behaviors
diverge. Two representative real traces, `-t 8`, same binary, same model, same session:

```
# t=1, decode_short: kernel_family_executed = "sme2"
hits_by_family: { sme2: 996, sme: 0, dotprod: 0, i8mm: 0 }
  kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot: 504
  kai_run_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa: 486

# t=8, decode_short: kernel_family_executed = "dotprod"
hits_by_family: { sme2: 0, sme: 0, dotprod: 31871, i8mm: 0 }   <- SME2 never entered, at all

# t=8, prefill_long: kernel_family_executed = "dotprod" (mixed)
hits_by_family: { sme2: 1538, sme: 0, dotprod: 13702, i8mm: 0 }  <- BOTH fire: this is hybrid mode
```

Full per-config JSON (all 10 configurations, every symbol's individual hit count):
`results/dispatch-ledger-darwin-arm64.json`.

### Root cause, read from source

**Step 1 — the cap is a hardcoded brand-string lookup, not a hardware query:**

```c
// kleidiai.cpp:96  (detect_num_smcus)
static size_t detect_num_smcus() {
    ...
    struct ModelSMCU { const char *match; size_t smcus; };
    static const ModelSMCU table[] = {   // kleidiai.cpp:156-161
        { "M4 Ultra", 2 },
        { "M4 Max",   2 },   // <- this machine's brand string matches here
        { "M4 Pro",   2 },
        { "M4",       1 },
    };
    for (const auto &e : table) {
        if (brand.find(e.match) != std::string::npos) {
            return e.smcus;
        }
    }
    // (no match falls through — an M5/M6/unlisted brand gets a different,
    //  unverified-in-this-session default)
}
```

`detect_num_smcus()` calls `sysctlbyname("machdep.cpu.brand_string", ...)` and does a substring
match against four literal strings. There is no query of an actual SME hardware capability register
here — the cap is a fact about *marketing names*, not silicon. An M4 Max is hardcoded to 2 SME
cores/threads.

**Step 2 — that number becomes a hard cap on the context:**

```c
// kleidiai.cpp:300
ctx.sme_thread_cap = (ctx.features & CPU_FEATURE_SME) ? sme_cores : 0;
```

**Step 3 — the actual per-call dispatch decision (the part the earlier draft of this finding
missed):**

```c
// kleidiai.cpp:1094-1112
const int  sme_cap_limit = ctx.sme_thread_cap;
const bool use_hybrid    = sme_cap_limit > 0 && runtime_count > 1 && nth_total > sme_cap_limit;

size_t min_cols_per_thread = std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

const bool hybrid_enabled = use_hybrid && !too_small_for_hybrid;

if (!hybrid_enabled) {
    ...
    } else if (runtime_count > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
        chosen_slot = 1;          // <-- collapses to the NON-SME (NEON) slot, SME2 never called
    }
}
```

`ne11` is the number of columns of `src1` — i.e., the batch dimension, the number of tokens being
multiplied in this one matmul call. This single variable is the whole story:

- **Decode** (`llama.cpp` generating one token at a time): `ne11 == 1`, unconditionally, on every
  model, every prompt. `too_small_for_hybrid` is therefore *always* `true`. Once `n_threads` exceeds
  the 2-thread cap, `hybrid_enabled` is `false` and the kernel chain unconditionally collapses to
  `chosen_slot = 1` — the NEON `dotprod`/`i8mm` path. **This is not a probabilistic or
  workload-dependent effect. It is a structural dead end for token-by-token decoding above 2
  threads, on this chip, full stop.**
- **Prefill** of a long-enough prompt: `ne11` is large (256 in our `prefill_long` phase), the
  `ne11 >= 128` gate opens, `hybrid_enabled` is `true`, and the batch is split across both kernel
  families — confirmed directly in the L3 hit counts above, where `sme2` and `dotprod` both fire
  non-zero in the *same* run.

### Why the earlier draft was wrong, and why that matters methodologically

The first pass at this finding used a 4-token prompt for every thread count and concluded "SME2
never fires above 2 threads." That statement is true *only* for decode-shaped workloads (`ne11 <
128`), and the 4-token prompt used to test it never exercised the hybrid branch at all — it looked
like a clean thread-count-only gate because every test case happened to sit in the same regime. Two
independent prompt lengths (`decode_short`, `~4 tok`; `prefill_long`, `~400 tok`) were required to
separate the two code paths and catch this. This is exactly the trap `tools/protocol.md` §6 item 6
documents under "missing the hybrid-dispatch rescue path," and it's why `tools/bench.py`'s sweep
grid deliberately includes both a below-gate and an above-gate prompt length rather than one
"representative" prompt.

### Load-bearing consequence

On Apple Silicon, **SME2 accelerates prefill but is structurally unreachable for token-by-token
decode** unless you manually drop to `-t 2` — which on a 16-core M4 Max means giving up 14 of 16
cores for every other workload sharing the machine. That is a real, quantifiable engineering
trade-off a user makes silently and unknowingly every time they run `llama.cpp` with its default
thread count. See the README's throughput sweep for whether that trade-off is actually worth it —
the answer is phase-dependent and not what a demo would pick (decode: yes, unconditionally;
prefill: no, once NEON is given its own natural thread count).

### Methodology caveat: stop counts vs. call counts

`lldb -b` with no auto-continue stops the process at the *first* breakpoint hit and the script ends
there — so the boolean fires/doesn't-fire signal is solid, but a raw count from that mode is not a
true call count. The dispatch-ledger table's hit counts (996, 5826, 31871, etc.) come from an
**auto-continuing** breakpoint command that lets the whole run complete and tallies every hit, which
*is* a true per-symbol call count — but the two modes must never be conflated. See
`results/GROUND-TRUTH-DISPATCH.md`'s own "Methodology caveat" section and `tools/protocol.md` §6
item 7 for why the auto-continuing mode is *not* used for the decode phase at high thread counts (it
can stall for minutes under heavy multi-threaded contention — a tooling fact, not a dispatch fact).

### A confound this project specifically checked and ruled out

This build also links Apple's Accelerate BLAS backend (`ggml-blas`). Its `supports_op` for `MUL_MAT`
requires `ne0 >= 32 && ne1 >= 32 && ne10 >= 32`. For decode, `ne1 == 1` always, so BLAS can never
claim the op. For prefill, the batch dimension can exceed 32, which in principle makes BLAS
eligible — but the same empirical `lldb` check shows a KleidiAI symbol (SME2 or NEON) firing for
*every* configuration actually tested, never a silent fall-through to Accelerate. This does not
generalize to every prompt length or model shape untested here, but it held for every configuration
this project measured — see `tools/protocol.md` §6 item 5.

### Prior-art check (2026-08-04)

- `sme_thread_cap` appears nowhere in `llama.cpp`'s own documentation and returns nothing on web
  search — this specific gate is undocumented upstream.
- `llama.cpp` *did* previously add a one-shot `GGML_LOG_WARN` for a different silent-fallback case
  (non-`Q4_0`/`Q8_0` weight types) in PR #25701 — direct precedent that upstream has accepted this
  exact category of fix before, which is the concrete ask this finding supports: a `GGML_LOG_WARN`
  when `hybrid_enabled` is `false` and `nth_total > sme_cap_limit`, i.e. exactly the
  `SILENT_FALLBACK` rows in the dispatch ledger.
- Issue #22182 independently notes that `ggml_cpu_has_sme()` is a compile-time check, so the banner
  can already be known to mislead about compile-time-vs-runtime — but nothing found documents the
  thread-cap / `ne11 >= 128` dispatch rule itself.

---

## Finding 2 — the SVE kernel family requires an exact 256-bit vector width

```c
// kleidiai.cpp:209
((ggml_cpu_has_sve() && ggml_cpu_get_sve_cnt() == QK8_0) ? CPU_FEATURE_SVE : CPU_FEATURE_NONE);
```

`ggml_cpu_get_sve_cnt()` returns the SVE vector length in bytes. `QK8_0` is `32` — 256 bits. This
is an **exact equality**, not a minimum-width check (`>=`). Any core whose SVE2 implementation is
narrower than exactly 256 bits — which, per the Arm architecture, is a legal and common
implementation choice (SVE/SVE2 vector length is implementation-defined, from 128 to 2048 bits in
128-bit increments) — can never set `CPU_FEATURE_SVE`, and the entire SVE kernel family becomes
permanently unreachable on that core, independent of whether it also implements i8mm and bf16.

The DGX Spark's Cortex-X925 implements SVE2 at **128 bits**. `ggml_cpu_get_sve_cnt()` on that core
returns `16`, not `32`; the equality fails; `CPU_FEATURE_SVE` is never set. This is a static,
architectural fact derivable from the source and the publicly documented Cortex-X925
microarchitecture spec — it does not require running on the chip to derive, only to *confirm at the
dispatch layer*.

**What has and hasn't been done for this finding:**

- Read from source and independently checked against Cortex-X925's documented SVE2 width: **done**.
- L1 (static) check on a Neoverse-N2 / Cortex-X925 binary, confirming the SVE symbols are compiled
  in but the feature bit is unreachable: **implemented in `tools/verify_dispatch.py` and
  `mcp/server.py`'s Linux code path, not yet exercised against real Linux/aarch64 hardware in this
  session** (see `mcp/README.md`'s own caveat: `"verified_on_this_session": false` on Linux).
- L3 (dispatch-time `lldb`/`gdb` trace) on real SVE2 hardware confirming zero SVE-family hits:
  **not yet obtained.** The self-hosted Spark CI lane (`.github/workflows/verify-spark-aarch64.yml`)
  exists specifically to produce this, gated `continue-on-error: true` end-to-end because of a
  separate, unrelated live incident on that runner (a suspected-OOM kernel kill, tracked in a
  still-open PR in this repo's history) — that incident is why this finding is reported as
  architecturally derived rather than fully dispatch-confirmed.
- `kernels/sve2_gemm.c` (this project's own hand-written SVE2 kernel, using `svwhilelt` for a
  padding-free tail and `svmmla_s32`/i8mm for the int8 path) compiles cleanly cross-compiled for
  `-march=armv9.2-a+sve2+i8mm+bf16` — verified as a real ELF aarch64 object containing genuine
  `smmla`/`fmad` SVE2/i8mm instructions (not merely a syntax check), persisted at
  `results/logs/sve2_cross_compile_check.log` — and correctly self-reports `-1` (unavailable) when
  run on this non-SVE2 Apple machine, per the correctness suite — but has never executed on real
  SVE2 hardware in this session.

**Do not upgrade this finding's confidence beyond "architecturally derived, dispatch confirmation
pending" until the Spark lane (or an equivalent SVE2 host) produces a real L3 trace.**

---

## What would change either finding

- **A different Apple chip generation** (M5/M6, or an unlisted brand string) exercising the
  fallthrough branch of `detect_num_smcus()`'s table — not yet observed.
- **A model/quant with different tensor shapes**, which changes whether `ne11 >= 128` and
  `ne01/n_threads >= 2` hold at a given thread count — the hybrid gate is workload-shape-dependent,
  not a fixed threshold independent of the model.
- **A real SVE2 dispatch trace on the Spark** (or any other narrow-SVE2 Arm core) turning Finding 2
  from "derived" into "confirmed," or — if it somehow doesn't reproduce — falsifying it, which would
  itself be a significant and reportable result given how directly it follows from the source.
