# Findings — deep root-cause analysis

> **Authoritative source:** `results/GROUND-TRUTH-DISPATCH.md`. Any statement in this document that
> contradicts that file is wrong and should be corrected to match it. This document expands on it
> with the full methodology, the correction history, and the evidence trail; it does not supersede
> it.

> **Prior art (2026-08-04):** Finding 2's mechanism below (the `kleidiai.cpp:209` exact-256-bit SVE
> gate) was independently published two days before this repository existed, by a different,
> unrelated project. We are not claiming priority on it. Finding 1 (the SME2 thread-gating below)
> is, as far as we could verify, still original to this repository. Full disclosure and what this
> project adds beyond the prior work: `docs/RELATED-WORK.md`.

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

## Optimization — the measured phase crossover, and a patch for part of it

> **Authoritative for this section's throughput numbers:** `results/REMEASURE-2026-08-04-QUIET.md`.
> The original `results/OPTIMIZATION.md` and `results/crossover/` numbers were collected on a host
> under heavy, unequal contention (1-minute load average 66–147) with baseline and tuned configs
> measured in different, non-interleaved time windows — a combination that manufactured a fake
> speedup. `results/REMEASURE-2026-08-04-QUIET.md` re-ran everything round-robin-interleaved on a
> quiet machine and is correct where the two disagree. This section adds the exact `kleidiai.cpp`
> line citations for the patch's mechanism and summarizes the corrected measured evidence; it does
> not supersede the remeasurement file.

### The measured crossover

Finding 1's root cause (`ne11`-gated hybrid dispatch) predicts a phase-dependent throughput
crossover: decode should prefer SME2 at the capped thread count, prefill should prefer NEON once
given its own natural thread count. Two independently written harnesses (`tools/bench.py`,
`results/SUMMARY.md`; `tools/crossover.py`, `results/crossover/crossover-apple-m4-max.md`) measured
this from scratch and agreed qualitatively on the direction. Both runs, however, were later found to
have been collected on this shared 16-core host while its 1-minute load average was **66–147**
(concurrent unrelated agent sessions), with baseline and tuned configurations measured in
**different, non-interleaved time windows** — a combination that manufactures a fake speedup. Their
absolute tok/s figures are therefore **not reliable** and are not repeated here; see
`results/REMEASURE-2026-08-04-QUIET.md` for the full account. Only the qualitative direction (decode
wants fewer threads with SME on; prefill wants more threads with SME off) still stands, and it is
confirmed below by an interleaved re-measurement.

`llama.cpp` already exposes the *thread-count* half of this as two separate CLI flags,
`-t`/`--threads` (generation) and `-tb`/`--threads-batch` (prompt/batch) — so tuning each phase
separately is expressible today, with zero code changes. A dedicated, round-robin-interleaved
re-measurement on a quiet machine (`results/REMEASURE-2026-08-04-QUIET.md`, 2026-08-04, `llama-bench
-r 1 -o json`, 7 reps/config, median + population stdev, external non-benchmark CPU load 236–326%
shared equally across configs by design) makes the before/after concrete.
`llama.cpp`'s no-flag default on this host is **12 threads** (`hw.perflevel0.physicalcpu`, the
P-core count), not 16:

| Configuration | decode tok/s (median, n=7) | prefill tok/s (median, n=7) |
|---|---:|---:|
| `llama.cpp` default (no `-t`/`-tb`) | 93.6 ± 2.47 | 1,230.3 ± 118.52 |
| decode: `-t 2`  ·  prefill: `-t 8` | **321.0 ± 2.09** (**3.43×**) | **2,198.1 ± 72.59** (**1.79×**) |

Source: `results/REMEASURE-2026-08-04-QUIET.md`. This supersedes the previously reported
45.5 → 198.9 tok/s (4.4×) decode figure and 1,145.0 → 2,257.5 tok/s (2.0×) prefill figure, both
artifacts of the non-interleaved, contended measurement described above. **3.43× decode / 1.79×
prefill, today, with flags `llama.cpp` already ships and zero code changes, is the honest number.**

**A correction to how this section previously framed causality:** Finding 1's root cause correctly
predicts the *qualitative* direction of this crossover (decode prefers SME2 at the capped thread
count; prefill prefers NEON at its own natural thread count) — but the qualitative direction is not
the same claim as "the 3.43× is caused by Finding 1," which an earlier draft of this project's
README asserted and which is not supported by the data. A dedicated decomposition sweep (SME2
forced on vs. off, independently of thread count) shows the **majority** of the 3.43× decode figure
is the well-documented "fewer threads help token generation on Apple Silicon"
memory-bandwidth/oversubscription effect — see `docs/development/token_generation_performance_tips.md`
in upstream `llama.cpp` — and is not attributable to SME2 dispatch at all. SME2 (Finding 1) adds a
real, smaller contribution on top at the tuned thread count, and *reduces* throughput at the default
thread count. The full decomposition table, exact ratios, and prior-art citations live in the
README's "Decomposition — how much of the decode win is SME2, and how much is thread tuning"
section (this file does not duplicate the full table to avoid the two documents drifting apart on a
number that matters).

### Why the dispatcher can't do this itself: `GGML_KLEIDIAI_SME` is process-global, read once

The *kernel-family* half of the split (SME2 for decode, NEON forced for prefill, in the same
process) is not expressible, because `GGML_KLEIDIAI_SME` is read exactly once, lazily, on the first
KleidiAI call, and cached for the rest of the process:

```c
// kleidiai.cpp:193-198 (init_kleidiai_context) — lazy, one-shot init guard
static void init_kleidiai_context(void) {
    ggml_critical_section_start();
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        // kleidiai.cpp:201
        const char *env_sme = getenv("GGML_KLEIDIAI_SME");
        ...
```

One process picks one SME policy at startup and keeps it for every `MUL_MAT` call thereafter,
decode and prefill alike. That is the missing capability: the thread-count knob is per-invocation
(`-t`/`-tb`), but the kernel-family knob is per-process. This project's patch targets the
*thread-count-gating* side of the problem (letting decode reach the existing hybrid path above the
cap); it does **not** touch this process-global limitation — see the verdict below.

### The patch's mechanism, with exact line citations

`patches/0001-kleidiai-phase-aware-dispatch.patch` (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`, 56
insertions / 3 deletions, applied on top of `dbadb68` as commit `ef973b1` in
`/tmp/llama-phase-aware`). Line numbers below are from the **patched** file:

1. **New context field**, `kleidiai.cpp:70`: `bool phase_aware_dispatch;` added to
   `struct ggml_kleidiai_context`, default-initialized `false`.
2. **New env var read**, `kleidiai.cpp:208`: `getenv("GGML_KLEIDIAI_PHASE_AWARE")`, parsed at
   `kleidiai.cpp:232-238` and stored into `ctx.phase_aware_dispatch` only if the value is truthy —
   symmetric with how `GGML_KLEIDIAI_SME` itself is parsed a few lines above it in the same
   init-once block. Logged once at `kleidiai.cpp:330-331` (`GGML_LOG_INFO`) when set.
3. **The actual dispatch-site change**, `kleidiai.cpp:1127-1140` (the same call site documented in
   Finding 1 above, originally `kleidiai.cpp:1094-1112` pre-patch):

   ```c
   // kleidiai.cpp:1137-1140 (patched)
   const bool phase_aware_gemv = ctx.phase_aware_dispatch && is_gemv &&
                                  sme_slot != -1 && non_sme_slot != -1;
   const bool too_small_for_hybrid = (min_cols_per_thread < 2) ||
                                      (!phase_aware_gemv && ne11 < 128);
   ```

   With the flag unset, `phase_aware_gemv` is always `false`, so `too_small_for_hybrid` evaluates
   identically to the pre-patch expression — the patch is a no-op unless explicitly opted into. With
   the flag set, a GEMV op (`is_gemv`, i.e. `ne11 == 1` — decode) with both an SME and a non-SME
   kernel slot available bypasses only the `ne11 < 128` term; the `min_cols_per_thread < 2` guard is
   untouched. This routes decode through the **same** hybrid thread-assignment code prefill already
   uses (`hybrid_enabled` at `kleidiai.cpp:1142`) — deliberately reused rather than reimplemented, to
   avoid a barrier-deadlock risk: this threadpool model requires every thread in `[0, nth_total)` to
   reach the same barriers the same number of times per op, so an earlier draft that left excess
   threads idle instead of routing them to the NEON slot was rejected during design.
4. **One-shot warning for the default (flag-off) path**, `kleidiai.cpp:1157-1160`: mirrors the
   file's existing weight-type-fallback warning pattern (`static std::atomic<bool> warned` guard),
   fires once per process when a GEMV op collapses to NEON because `nth_total > sme_thread_cap`,
   naming the exact knob (`-t <= sme_thread_cap`, or `GGML_KLEIDIAI_PHASE_AWARE=1`) that recovers it.

### The dispatch-level proof

Symbol-level (`tools/verify_dispatch.py`, `lldb`, auto-continuing breakpoint on
`kai_run_matmul.*sme`, true per-symbol call counts), same patched binary, only
`GGML_KLEIDIAI_PHASE_AWARE` toggled — an apples-to-apples single-binary A/B on the dispatch decision
itself:

| threads | workload | flag OFF (hits: SME2 / other) | flag ON (hits: SME2 / other) |
|---:|---|---|---|
| 4 | decode_short | **0 / 15,936** — exact match to Finding 1's pre-patch ground truth | **3,072 / 10,428** (3,072/13,500 total) |
| 8 | decode_short | **0 / 31,872** | **2,354 / 20,517** (2,354/22,871 total) |

Full JSON: `results/dispatch-ledger-darwin-arm64-patched-flag-{off,on}.json`. This is the decisive
evidence that the patch changes *dispatch*, not merely the selection-log text: SME2 goes from zero
calls to thousands of calls in the identical binary, workload, and thread count, with only the env
var different.

### Measured verdict — the patch is a REGRESSION, not a win

The interleaved re-measurement (`results/REMEASURE-2026-08-04-QUIET.md`) retracts the previously
reported threads=4/8 deltas and the previously reported "decode 45.5 → 71.6 tok/s, +57.3%"
default-configuration figure — both were artifacts of the same non-interleaved, contended
measurement described above. The clean re-run, at the thread count that actually matters most
(`llama.cpp`'s real no-flag default, 12 threads on this host) and at the tuned `-t 2` decode
setting, shows the opposite of a win:

| comparison | ratio | reading |
|---|---:|---|
| patched+flag vs baseline, default threads (12) | **0.88×** | 93.6 → 82.5 tok/s: **~12% SLOWER** |
| patched+flag vs baseline, `-t 2` | 0.99× | 321.0 → 317.5 tok/s: statistical tie (patch is inert here) |
| prefill, patched+flag vs baseline, default threads | 0.98× | 1,230.3 → 1,202.1 tok/s: tie, within noise (patch doesn't touch the GEMM path) |

93.6 ± 2.47 versus 82.5 ± 4.07 do not overlap — the regression at default thread count is **real
and outside noise**, not measurement wobble. (The previously reported threads=4/8 deltas came from
the same contended, non-interleaved protocol as the retracted +57.3% figure and are not repeated
here; they have not been re-measured under the quiet, interleaved protocol.)

**Mechanism, stated honestly:** the patch works exactly as designed at the dispatch level — it
routes GEMV (decode) work into the existing SME+NEON hybrid split above `sme_thread_cap`, and
`tools/verify_dispatch.py` proves the change at the symbol level (decode@t=4: 0 → 3,072 SME2 hits;
decode@t=8: 0 → 2,354). That dispatch evidence is a symbol-level fact, unaffected by contention, and
remains valid — dispatch counts, not timings. But **dispatching SME2 is not the same as being
faster.** At 12 threads the hybrid split gives SME only 2 of them while the other 10 run NEON, and
coordinating that split costs more than the SME lane returns for a shape this small; pure NEON on
all 12 threads wins. The upstream code's existing behaviour is, on this chip and this model, the
better default, and the patch's premise that decode was being unfairly excluded is **not supported
by throughput**, even though the exclusion itself is real.

It does not beat the hand-tuned per-phase thread split (`-t 2` decode / `-t 8` prefill: 321.0 /
2,198.1 tok/s), which needs no patch at all, and it does not touch the process-global
`GGML_KLEIDIAI_SME` limitation above — the theoretical best (SME2-decode + NEON-forced-prefill,
simultaneously, in one process) remains `[NOT YET ACHIEVABLE]`, patched or not. Full numbers, every
caveat: `results/REMEASURE-2026-08-04-QUIET.md` (authoritative); `results/OPTIMIZATION.md` (original
methodology, superseded).

**Reported upstream, with an offer to send this patch:**
[ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547).

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
  **obtained 2026-08-05**, via a direct `gdb`-attached trace on the DGX Spark itself
  (`results/server/server-dispatch.json`, full account in `results/server/SERVER-LANE.md`) — **not**
  via the self-hosted Spark CI lane (`.github/workflows/verify-spark-aarch64.yml`), which remains
  gated `continue-on-error: true` and unresolved for the same reason recorded here previously (a
  suspected-OOM kernel kill, tracked in a still-open PR in this repo's history). The server lane's
  trace also covers a workload the CI lane was never built to exercise: 8 concurrent `llama-server`
  clients under continuous batching, not single-user `llama-cli` decode/prefill. See the new
  subsection below.
- `kernels/sve2_gemm.c` (this project's own hand-written SVE2 kernel, using `svwhilelt` for a
  padding-free tail and `svmmla_s32`/i8mm for the int8 path) compiles cleanly cross-compiled for
  `-march=armv9.2-a+sve2+i8mm+bf16` — verified as a real ELF aarch64 object containing genuine
  `smmla`/`fmad` SVE2/i8mm instructions (not merely a syntax check), persisted at
  `results/logs/sve2_cross_compile_check.log` — and correctly self-reports `-1` (unavailable) when
  run on this non-SVE2 Apple machine, per the correctness suite — but has never executed on real
  SVE2 hardware in this session.

**Status, updated 2026-08-05:** upgraded from "architecturally derived, dispatch confirmation
pending" to **confirmed**, on a second independent core family (Cortex-X925 / DGX Spark, in
addition to the previously-confirmed Neoverse-N2 via free CI), and for the first time under
concurrent multi-client serving load rather than only single-user `llama-cli` load/decode time. The
self-hosted Spark CI lane referenced above is still unresolved and is **not** the source of this
confirmation — a direct, manually-driven `gdb`-attached trace on the Spark is.

### Confirmed on Cortex-X925, under concurrent serving load (2026-08-05)

`results/server/spark-provenance.txt` reports, on a KleidiAI-enabled build fixed per Finding 3
below, `SVE_CNT = 16` — a 16-byte (128-bit) SVE2 implementation, exactly half the 32-byte/256-bit
width `kleidiai.cpp:209`'s exact-equality gate requires — and the same banner's
`kleidiai: primary q4/q8 kernel feature I8MM` lines confirm I8MM, not SVE, was selected, precisely
as the gate predicts. `results/server/server-dispatch.json`, captured with `gdb` attached to a live
`llama-server` process serving 8 concurrent clients (10 `kai_run_matmul` breakpoints set — one per
symbol in the fixed build's own `kai_run_matmul symbols: 10` count), records:

```json
{ "dotprod": 11360, "i8mm": 364444 }
```

No `sve` key appears in the file: across all `11360 + 364444 = 375,804` recorded calls, zero were
attributed to the SVE kernel family. Full account, including the dispatch-shape inversion this
lane also observed (i8mm-dominated under batched serving vs. dotprod-dominated in Finding 1's
single-user Apple Silicon decode trace): `results/server/SERVER-LANE.md`.

**Prior art:** this exact `kleidiai.cpp:209` line and the `QK8_0`-equality reasoning above were
published, independently, two days before this repository existed, by a different project
(`luongs3/arm-dispatch-audit`, created 2026-08-01, this repo created 2026-08-03). We derived this
finding from source before we were aware of that repository, but they published it first and we are
not claiming priority. Full disclosure, verification, and what this project adds beyond that prior
work (most notably Finding 1, which their repository does not contain): `docs/RELATED-WORK.md`.

---

## Finding 3 — `llama.cpp`'s documented KleidiAI build line silently compiles zero matmul kernels on gcc 13.3 + Cortex-X925

> **Authoritative for this finding's full evidence, banners, and scope discussion:**
> `results/server/SERVER-LANE.md`. This section summarizes it at the same rigor as Findings 1 and
> 2; it does not duplicate every table or caveat there.

### The question

Does following `llama.cpp`'s own documented KleidiAI build command actually produce a binary with
Arm-accelerated matmul kernels? On the DGX Spark (Cortex-X925, gcc 13.3.0), the answer is no — and
nothing about the build or its runtime banner says so.

### The evidence

`cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DCMAKE_BUILD_TYPE=Release` — `llama.cpp`'s own
documented KleidiAI build line, no unusual flags — produces, per `results/server/spark-provenance.txt`:

```
kai_run_matmul symbols: 0
system_info: ... CPU : NEON = 1 | ARM_FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: no compatible q4 kernels found for CPU features mask 0
kleidiai: no compatible q8 kernels found for CPU features mask 0
kleidiai: no compatible f32 kernels found for CPU features mask 0
```

Zero compiled-in KleidiAI matmul micro-kernel entry points, of any family. The build exits 0. The
banner prints `KLEIDIAI = 1` — the exact same flag value a working build also prints — with nothing
in it that says acceleration failed. Only the `kleidiai:` log lines, and only their explicit
`mask 0` (zero CPU features detected for kernel selection), give it away.

Adding one flag pair, `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"`,
to the same documented command, on the same commit, same compiler, same machine, produces:

```
kai_run_matmul symbols: 10
system_info: ... CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | SVE = 1 | DOTPROD = 1 |
  SVE_CNT = 16 | OPENMP = 1 | KLEIDIAI = 1 | REPACK = 1 |
kleidiai: primary q4 kernel feature I8MM
kleidiai: primary q8 kernel feature I8MM
```

Both symbol counts (`0` and `10`) and both banners are quoted verbatim from
`results/server/spark-provenance.txt`.

### Root cause

Reported diagnosis from this lane's build investigation (see `results/server/SERVER-LANE.md`'s
provenance note for exactly which parts of this paragraph are a committed artifact versus reported
methodology): `llama.cpp`'s CMake configure step cannot resolve an explicit `-march`/`-mcpu` for
this target and falls back to probing feature suffixes (`+dotprod`, `+i8mm`, `+sve`, ...) on top of
`-mcpu=native`. On gcc 13.3.0 + Cortex-X925, those suffixed probes were reported to fail —
including the negative-control probes, which is the signature of a broken probe rather than an
absent feature — because gcc 13.3 predates Cortex-X925 in its own `-mcpu` support table. An
explicit `-march=armv9.2-a+i8mm`-style target, bypassing `-mcpu=native` probing entirely, compiles
cleanly, which is exactly the fix quoted above. What *is* independently, verbatim confirmed by the
committed evidence is the causal prediction this diagnosis makes: `kai_run_matmul symbols` goes
from `0` to `10` between the two builds.

### Severity and scope

This is a **build-time, silent, complete loss of all Arm-specific matmul acceleration**,
reproducible from `llama.cpp`'s own documented KleidiAI build instructions, with no error, no
warning, and a `KLEIDIAI = 1` banner flag that reads as success either way.

This is **not** a claim that every Arm machine is affected — it is specifically the
**gcc 13.3.0 + Cortex-X925** pairing tested here: gcc 13.3 predates Cortex-X925 in its own `-mcpu`
table, so `llama.cpp`'s `-mcpu=native`-based feature probing has nothing valid to probe against on
this CPU with this compiler, and every suffixed probe fails. **The general, reusable lesson is that
this failure mode recurs on any CPU newer than the toolchain compiling for it** — any pairing of a
distribution's default gcc with a chip gcc added to its `-mcpu` table later than that gcc release
is predicted to hit the same silent zero-kernel outcome on `llama.cpp`'s current documented build
line, not a Spark-specific defect.

---

## What would change these findings

- **A different Apple chip generation** (M5/M6, or an unlisted brand string) exercising the
  fallthrough branch of `detect_num_smcus()`'s table — not yet observed.
- **A model/quant with different tensor shapes**, which changes whether `ne11 >= 128` and
  `ne01/n_threads >= 2` hold at a given thread count — the hybrid gate is workload-shape-dependent,
  not a fixed threshold independent of the model.
- ~~A real SVE2 dispatch trace on the Spark (or any other narrow-SVE2 Arm core) turning Finding 2
  from "derived" into "confirmed."~~ **Done, 2026-08-05** — see Finding 2's "Confirmed on
  Cortex-X925, under concurrent serving load" subsection and `results/server/SERVER-LANE.md`. What
  would still extend it further: a real SVE2 dispatch trace on a *third* independent core family, or
  a wider SVE2 implementation (256-bit or above) that should, per the same gate, finally let the SVE
  kernel family dispatch — neither has been observed.
- **A committed gcc-probe transcript** for Finding 3 (the individual `-mcpu`/`-march` probe
  compiles and their pass/fail outcomes, captured as a raw log rather than reported diagnosis) would
  move that finding's root-cause explanation from "reported" to the same directly-cited-artifact
  standard as the rest of this document — see the provenance note in
  `results/server/SERVER-LANE.md`.
- **A newer gcc release on the same Cortex-X925 hardware**, or the same gcc 13.3.0 on a chip gcc 13.3
  already supports, would isolate whether Finding 3 is specifically a toolchain-predates-chip
  mismatch (as diagnosed) rather than something else about this machine's build environment —
  neither has been tried.
