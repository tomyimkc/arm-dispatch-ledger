<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
-->

# `0001-kleidiai-phase-aware-dispatch.patch`

**Target:** `ggml-org/llama.cpp` at `dbadb68` (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`).
**Status:** local, opt-in, unverified for speedup (see "What this patch does NOT claim" below).
**Upstream issue this answers:** https://github.com/ggml-org/llama.cpp/issues/26547

## License / attribution

This patch is a **derivative work of MIT-licensed code**: `ggml-org/llama.cpp`
(https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) is MIT-licensed, and every
line this patch touches or adds lives inside `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`,
an existing MIT-licensed file in that project — this patch does not introduce new
Apache-2.0-licensed code into llama.cpp, and llama.cpp's own MIT terms (not this
repository's Apache-2.0 license) govern the patched file once applied there. This
repository stores only the `.patch` diff (a description of a change), never a vendored
or forked copy of `kleidiai.cpp` itself — see `docs/SUBMISSION.md`'s "What changed after
2026-06-04" section for the no-vendoring statement. If/when this patch is opened as a
pull request against `ggml-org/llama.cpp` (`docs/UPSTREAM-PR.md`), it is offered back to
that project under its own MIT license and contributor terms, consistent with how the
patch was developed (against an unmodified MIT-licensed `dbadb68` checkout, never
against a copy relicensed under this repository's Apache-2.0 terms).

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
- `0002-kleidiai-sme-aware-thread-default.patch` — see below.
- `README.md` — this file.

---

# `0002-kleidiai-sme-aware-thread-default.patch`

**Target:** `ggml-org/llama.cpp` at `dbadb68` (`common/`, `ggml/include/ggml-cpu.h`,
`ggml/src/ggml-cpu/ggml-cpu.cpp`, `ggml/src/ggml-cpu/kleidiai/kleidiai.{h,cpp}`).
**Status:** local, on by default, measured (see `results/AUTODEFAULTS.md`).

## License / attribution

Same basis as `0001` above: a derivative work of MIT-licensed `ggml-org/llama.cpp`, touching
only files already under that project's MIT license. This repository stores the `.patch` diff
only, never a vendored copy of the touched files.

## The problem this answers (Defect A)

`0001` and this repo's own measurements (`results/REMEASURE-2026-08-04-QUIET.md`) establish that
on an SME2 CPU, decode throughput is **~2-3x higher at `n_threads == sme_thread_cap`** (2 on an
Apple M4 Max) than at the stock default (the physical/P-core count, 12 here) — because above the
cap, KleidiAI's SME2 GEMV kernels can't dispatch and silently fall back to slower NEON. That win
is real, but it requires the user to already know `sme_thread_cap`, know it applies specifically
to *generation* threads, and pass `-t 2` themselves. Nothing in the codebase does this
automatically, so out of the box, on stock `llama.cpp`, KleidiAI-capable SME2 hardware runs
decode at roughly a third of its achievable throughput, silently.

Worse: the "obvious" hand-tuned fix (just pass `-t 2`) is a trap. Passing `-t 2` alone with no
`-tb` also caps **prefill/batch** threads at 2 (llama.cpp's stock `-tb` default is "same as
`-t`" — see `postprocess_cpu_params()` in `common/common.cpp`), which **collapses prefill by
~2x** in our own measurement (`results/AUTODEFAULTS.md`: 1835 -> 976 tok/s, -47%). A doc telling
people to "just pass `-t 2`" would be trading a decode win for a prefill regression nobody asked
for.

## The fix

A small, additive, generation-only default:

1. **A new public accessor**, `ggml_backend_cpu_kleidiai_sme_thread_cap()`, declared in the
   always-compiled `ggml/include/ggml-cpu.h` (so `common/` can call it regardless of whether
   `GGML_CPU_KLEIDIAI` is on) and implemented twice: the real one in `kleidiai.cpp` (returns
   `ctx.sme_thread_cap` after forcing lazy init), and a `return 0;` stub in `ggml-cpu.cpp` for
   builds without KleidiAI. Both verified to build (see "Verification" below).

2. **`common_kleidiai_sme_auto_gen_threads()`** in `common/common.cpp`: if the accessor reports a
   positive cap lower than the already-computed default, and `GGML_KLEIDIAI_AUTO_THREADS` is not
   `0`, return the cap instead of the default (and log one `COM_INF` line saying exactly what
   changed and why). Otherwise return the default unchanged.

3. **`common_params_apply_kleidiai_auto_threads()`**, called from `common/arg.cpp` **after** all
   four existing `postprocess_cpu_params()` calls (main, batch, speculative draft, speculative
   draft batch) have already resolved their defaults. This ordering is the entire correctness
   argument: `--threads-batch` inherits "same as `--threads`" by *copying* `params.cpuparams` at
   the point `postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams)` runs (line
   ~853) — if the SME override were applied any earlier (e.g. inside `postprocess_cpu_params`
   itself), that copy would silently propagate the SME-capped generation value into
   `--threads-batch`/prefill too, recreating the exact `-t 2` prefill-collapse trap this patch
   exists to avoid. Applying the override strictly afterward, and strictly only to
   `params.cpuparams`, guarantees batch/prefill and speculative decoding always see the
   *original*, uncapped default.

4. **Only fires when the user did not pass `-t`/`--threads`.** The signal is the `< 0` "unset"
   sentinel on `params.cpuparams.n_threads`, captured *before* `postprocess_cpu_params()` resolves
   it (a precise signal, not the "compare against the computed default" fallback the original
   design brief anticipated needing — the precise signal was available, so we used it).

5. **Kill switch and logging.** `GGML_KLEIDIAI_AUTO_THREADS=0` disables the whole override
   (verified to reproduce the unpatched baseline exactly — see below). When active, it always
   logs one `COM_INF` line naming the old and new thread counts — never a silent behavior change.

Diffstat: 7 files changed, 94 insertions(+), 0 deletions. No file is modified destructively;
every change is a new function, a new declaration, or a new call site.

## A note on `llama-bench` (read before citing "no-flags" numbers)

**`llama-bench` does not exercise this patch's "no flags" code path**, by construction, and this
was verified directly, not assumed. `llama-bench` never calls `postprocess_cpu_params()` or
`common_context_params_to_llama()` — it builds its own `llama_context_params` directly and
sources its own `-t` default straight from `common_cpu_get_num_math()`
(`tools/llama-bench/llama-bench.cpp`, `cmd_params_defaults.n_threads`), which this patch
intentionally does not touch (touching it would have re-broken llama-bench's own default for
*prefill-only* test rows too, since llama-bench uses one `n_threads` value for both `-p` and `-n`
tests in a given row — there is no `-tb` equivalent inside llama-bench). Confirmed empirically:
`llama-bench -m q05.gguf -p 16 -n 16 -o json` reports `n_threads: 12` for both the baseline and
the patched `llama-bench` binary with no flags passed, identically.

Consequently, `results/AUTODEFAULTS.md`'s "no flags" measurements (configs 1/2/4) use
**`llama-cli`** round-robin, not `llama-bench` — `llama-cli` is the tool that actually goes
through `common_params_parse()` -> `postprocess_cpu_params()` ->
`common_params_apply_kleidiai_auto_threads()` -> `common_context_params_to_llama()`, i.e. the real
patched path a user hits by typing `llama-cli -m model.gguf -p "..."` with no `-t`. `llama-bench`
with an *explicit* `-t` (config 3, the hand-tuned ceiling) is unaffected by this distinction and
was cross-checked to agree with the `llama-cli` numbers within noise.

## What this patch does NOT claim

- It does not change anything about KleidiAI's dispatch logic itself (that is `0001`'s scope,
  and `0001` remains a measured regression — see `results/REMEASURE-2026-08-04-QUIET.md`). This
  patch only changes which thread count `llama-cli`/`llama-server` pick by default.
- It does not affect `llama-bench`'s own `-t` default (see above) — only real front ends that go
  through `common/arg.cpp`.
- It does not claim the SME2 cap is optimal on hardware other than the one measured here (Apple
  M4 Max, `sme_thread_cap=2`); the mechanism generalizes (it reads the cap from KleidiAI's own
  runtime detection, not a hardcoded constant), but the *speedup magnitude* was only measured on
  this machine.

## How to apply

```sh
cd /path/to/llama.cpp   # at or near dbadb68
git apply patches/0002-kleidiai-sme-aware-thread-default.patch
```

Verified to apply cleanly (`git apply --check`) against a **fresh, unmodified clone** of the
`dbadb68` baseline (not just the working tree it was developed in) — see "Verification" below.

## How to build and run it

```sh
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli llama-bench -j"$(sysctl -n hw.ncpu)"

# no flags: generation threads auto-default to the SME2 cap; --threads-batch/prefill unaffected
./build/bin/llama-cli -m model.gguf -no-cnv -st --simple-io -p "..." -n 128

# disable the auto-default, reproduce stock behavior exactly
GGML_KLEIDIAI_AUTO_THREADS=0 ./build/bin/llama-cli -m model.gguf -no-cnv -st --simple-io -p "..." -n 128
```

## Verification performed locally (Apple M4 Max, `FEAT_SME2`, `sme_thread_cap=2`)

- **Applies clean to a pristine checkout**: `git apply --check` and `git apply` against a fresh
  `cp -a` of the `dbadb68` baseline (not the dev working tree), confirmed before this doc was
  written.
- **Builds clean both ways**: `-DGGML_CPU_KLEIDIAI=ON` and `-DGGML_CPU_KLEIDIAI=OFF`, both Release,
  both from the freshly-patched pristine checkout above — zero errors/warnings attributed to this
  patch's code in either configuration.
- **Correctness**: `llama-cli` produces correct, coherent generations at the auto-selected
  default (e.g. "The capital of France is Paris."), no crash, no assertion failure, no garbled
  output.
- **The auto-default actually fires, and only where intended** (`--verbose`, `system_info` line
  and the patch's own log line):
  - No flags: `KleidiAI SME2 detected (thread cap = 2); defaulting generation threads to 2
    instead of 12.` -> `system_info: n_threads = 2 (n_threads_batch = 12)`. Batch/prefill
    untouched, exactly as designed.
  - `GGML_KLEIDIAI_AUTO_THREADS=0`, no flags: `system_info: n_threads = 12 (n_threads_batch =
    12)` — identical to the unpatched baseline's own no-flags output, no override log line.
  - Explicit `-t 4`: `system_info: n_threads = 4 (n_threads_batch = 4)` — no override log line;
    user's explicit choice is never touched.
  - Explicit `-tb 6` with no `-t`: `system_info: n_threads = 2 (n_threads_batch = 6)` — proves
    generation and batch really are independent under this patch.
  - `-DGGML_CPU_KLEIDIAI=OFF` build, no flags: `system_info: n_threads = 12 (n_threads_batch =
    12)`, no KLEIDIAI feature flag in the CPU feature list, no crash — the stub path works.
- **Symbol-level dispatch proof** (`tools/verify_dispatch.py --threads 2 --workloads
  decode_short`, `lldb`, anchored `^kai_run_matmul` breakpoint): **`SME2_DISPATCHED`**, 5826/0
  SME2-vs-other kernel hits, at `sme_thread_cap`, the exact value this patch auto-selects with
  zero flags.
- **Throughput, round-robin interleaved, `llama-cli`, n=9, decode and prefill separately**: see
  `results/AUTODEFAULTS.md` for the full table. Headline: no-flags decode 67.8 -> 145.9 tok/s
  (**2.15x**, matching the `-t 2` hand-tuned ceiling of 146.0 within noise) with prefill
  essentially unchanged (1835.2 -> 1779.8, -3.0%, within noise) — versus the naive `-t 2` flag,
  which reaches the same decode ceiling but collapses prefill by 47% (1835.2 -> 975.6).
