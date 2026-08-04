<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
-->

# `0001-kleidiai-phase-aware-dispatch.patch`

**Target:** `ggml-org/llama.cpp` at `dbadb68` (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`).
**Status:** local, opt-in, unverified for speedup (see "What this patch does NOT claim" below).
**Upstream issue this answers:** https://github.com/ggml-org/llama.cpp/issues/26547

## The problem, in one sentence

KleidiAI's SME dispatch decision caps SME threads uniformly for every `MUL_MAT` shape, which
has the side effect of permanently excluding decode (`ne11 == 1`, a GEMV) from the SME kernel
once the requested thread count exceeds `sme_thread_cap` — even on hardware where SME
measurably wins that exact shape at the capped thread count.

## Root cause (read from source, not guessed)

`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`, `compute_forward_qx()`, around the dispatch decision:

```c
const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);
```

`ne11` is the batch/token count of the matmul. Decode is *always* `ne11 == 1`, so this term is
always true for decode, regardless of thread count. When `too_small_for_hybrid` is true and
`nth_total > sme_thread_cap`, dispatch collapses to the non-SME (NEON) slot unconditionally:

```c
chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
```

This is architecturally sound for *why* the gate exists (avoid hybrid overhead on genuinely
small batches), but it conflates two different things: "batch too small to split efficiently"
and "this op is a GEMV". Prefill (`ne11 >= 128`) legitimately benefits from staying out of
hybrid mode when small; decode never gets the *chance* to be evaluated on its own merits,
because `ne11 == 1 < 128` always trips the same gate.

Measured on Apple M4 Max (`results/SUMMARY.md`, this repo): `decode SME2@2thr = 327.6 tok/s`
vs. `decode NEON@2thr = 266.4 tok/s` — SME2 wins at the capped thread count, but nothing in the
dispatcher can reach it once more than 2 threads are requested for generation.

## The fix

Minimal and surgical, at the same call site:

1. **Opt-in flag.** `GGML_KLEIDIAI_PHASE_AWARE=1` (default off). Parsed once in
   `init_kleidiai_context()`, stored as `ctx.phase_aware_dispatch`, logged at `GGML_LOG_INFO`
   when set. With the flag unset, every line this patch touches evaluates to exactly what it
   evaluated to before — this is not a heuristic change, it is a gated one.

2. **Let GEMV into the existing hybrid path instead of building a new one.** When the flag is
   set and the op is GEMV-shaped (`is_gemv`, i.e. `ne11 == 1`) with both an SME and a non-SME
   kernel slot available, the `ne11 < 128` term of `too_small_for_hybrid` is bypassed for that
   op (the `min_cols_per_thread < 2` term still applies — this patch does not touch the
   too-small-to-split-at-all guard). That makes `hybrid_enabled` true for decode above the cap,
   which routes it through the *same* thread-assignment code prefill hybrid mode already uses:
   SME gets capped at `sme_thread_cap` threads, the rest run NEON in parallel. No new thread-
   splitting logic was written; this reuses code that was already exercised (and correct) for
   prefill. That reuse is deliberate: it is the smallest change that gets decode into a tested
   path, and it avoids a real deadlock risk — this codebase's threadpool model requires every
   thread in `[0, nth_total)` to reach the same barriers the same number of times per op, so an
   earlier draft of this patch that tried to leave the extra threads idle (rather than routed to
   the NEON slot) was rejected during design for exactly that risk. See the patch's inline
   comment at the `too_small_for_hybrid` computation for the full reasoning.

3. **One-shot warning for the default (flag-off) path.** Mirrors the existing one-shot
   weight-type-fallback warning later in the same file (`static std::atomic<bool> warned`
   guard). Fires once per process when a GEMV op collapses to NEON because
   `nth_total > sme_thread_cap`, naming the exact knob (`-t <= sme_thread_cap` for generation,
   or `GGML_KLEIDIAI_PHASE_AWARE=1`) that gets SME back.

Diffstat: 1 file changed, 56 insertions(+), 3 deletions(-). No new files, no new dependencies,
no change to any public API or CMake target.

## What this patch does NOT claim

This patch is scoped to **correct, minimal, compiles, dispatches as intended**. It does **not**
claim a speedup. Running SME2 on 2 threads concurrently with NEON on the remaining threads is a
plausible win (it's strictly more silicon doing useful work than either single-kernel option),
but "plausible" is not "measured" — throughput measurement with this flag on vs. off, across
thread counts and prompt lengths, is a separate, later step and should be reported (or not) on
its own evidence, not folded into this patch's claims.

## How to apply

```sh
cd /path/to/llama.cpp   # at or near dbadb68
git apply patches/0001-kleidiai-phase-aware-dispatch.patch
# or, to keep the commit (message includes full rationale + local verification notes):
git am patches/0001-kleidiai-phase-aware-dispatch.patch
```

Verified to apply cleanly (`git apply --check` and `git am`) against a fresh clone of the
`dbadb68` baseline used throughout this repo.

## How to build and run it

```sh
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-cpu llama-cli -j"$(sysctl -n hw.ncpu)"

# default (flag unset): identical to pre-patch behavior
GGML_KLEIDIAI_SME=2 ./build/bin/llama-cli -m model.gguf -p "..." -n 32 -no-cnv -st --simple-io -t 8

# opt-in: decode gets an SME2 (capped)+NEON hybrid split above sme_thread_cap
GGML_KLEIDIAI_SME=2 GGML_KLEIDIAI_PHASE_AWARE=1 ./build/bin/llama-cli -m model.gguf -p "..." -n 32 -no-cnv -st --simple-io -t 8
```

## Verification performed locally (Apple M4 Max, `FEAT_SME2`, `sme_thread_cap=2`)

- **Compiles clean**: `-DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON`, Release build,
  zero warnings/errors attributed to `kleidiai.cpp`.
- **Correctness**: `llama-cli` produces correct, coherent generations with the flag on and off,
  at `-t 8` and `-t 16` (e.g. "The capital of France is Paris." both ways). No crash, no
  assertion failure, no garbled output, in every run across this verification pass.
- **Dispatch, flag OFF, `-t 4`, decode** (`lldb`, anchored `^kai_run_matmul` breakpoint,
  auto-continue, per-symbol hit counts via `arm-dispatch-ledger/tools/dispatch_probe.lldb`):
  `kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot` (the Q4_0 SME2 GEMV
  kernel) hit count **0**; total hits **15936** — matches
  `results/GROUND-TRUTH-DISPATCH.md`'s independently-measured baseline exactly
  (`t=4 decode -> dotprod (0 sme2/15936)`), confirming this environment reproduces the
  documented ground truth before any patch behavior is exercised.
- **Dispatch, flag ON, `-t 4`, decode**, same methodology: the same SME2 GEMV kernel symbol
  fires **3515** times, and the NEON GEMV kernel
  (`kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod`) fires **2112** times in
  the same run — both kernel families active concurrently, i.e. the hybrid split is real, not
  just "selected but never entered." This is the core deliverable: SME2 now dispatches for
  decode above `sme_thread_cap`, opt-in only.
- **Dispatch decision at `-t 16`** was confirmed via a temporary source-level log (removed
  before finalizing the patch) rather than a full `lldb` sweep: at `nth_total=16` the same run
  showed `hybrid_enabled=1, phase_aware_gemv=1, sme_slot=0, non_sme_slot=1` for the model's
  MUL_MAT ops — i.e., the same correct hybrid decision as `-t 4`, since the branch does not
  depend on `nth_total` beyond the existing `sme_thread_cap`/`min_cols_per_thread` terms. A full
  `lldb`-driven `-t 16` sweep was attempted but not completed cleanly in this session: this
  machine runs several other concurrent agent sessions against the same physical cores per this
  project's working agreement, and a `-t 16` decode run competing with that load became slow
  enough (observed as low as 0.2 tok/s at one point) to make a full `lldb` capture at that
  thread count impractical in the session time available. This is a measurement-environment
  limitation, not a defect in the dispatch logic — the decision path itself was confirmed
  independent of thread count by the source-level check above, and is not sensitive to timing.
  Re-running the `-t 16` `lldb` sweep on a quieter machine is recommended before citing a `-t 16`
  dispatch number anywhere.

## Files in this directory

- `0001-kleidiai-phase-aware-dispatch.patch` — the patch itself (`git format-patch` output,
  applies with `git apply` or `git am`).
- `README.md` — this file.
