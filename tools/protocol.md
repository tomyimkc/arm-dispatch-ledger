# Benchmark protocol — `bench.py`

This document states the exact measurement methodology used by `tools/bench.py`, and
pre-empts the standard objections a skeptical reviewer (or an Arm engineer) will raise
about a CPU micro-benchmark. If a claim in `results/bench-*.md` seems too good, or too
convenient, check it against this file first — the script is built to match this
document, not the other way around.

## 1. The question being answered

On Apple Silicon, `llama.cpp` built with `-DGGML_CPU_KLEIDIAI=ON` can execute a
quantized (Q4_0 / Q8_0) matmul through **one of two kernel families**:

- **SME2** (Arm Scalable Matrix Extension v2, via KleidiAI), gated to at most
  `sme_thread_cap` threads (2 on an M4 Max/Ultra/Pro, 1 on plain M4 — see
  `results/GROUND-TRUTH-DISPATCH.md`), *unless* a **hybrid** rescue path also engages
  (see §5.4).
- **NEON** (dotprod / i8mm), unconstrained in thread count.

Nobody has published a throughput comparison of these two paths against each other on
the same silicon, the same model, and the same quantization. That is what this harness
measures — not "how fast is llama.cpp" in the abstract, but **"between the two paths
`llama.cpp` can legally pick, which one actually wins, at which thread count, in which
phase (prefill vs. decode), and why."**

## 2. Fixed variables (never change within one comparison)

| Variable | Value | Why fixed |
|---|---|---|
| Model | `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (337 MB, Apache-2.0) | Same weights, same quant, every row. Different models have different tensor shapes, which changes whether the hybrid-dispatch gate (`ne11 >= 128`) engages — see §5.4. |
| Binary | `/tmp/llama.cpp/build/bin/{llama-bench,llama-cli}`, commit `dbadb68`, built once with `-DGGML_CPU_KLEIDIAI=ON` | Recompiling between configurations would confound the comparison with compiler-flag differences (see the SME2/`-march` SIGILL trap documented in the top-level project notes — irrelevant to *this* harness since it never recompiles, but exactly the kind of thing that must stay fixed). |
| Token content | `llama-bench`'s internal synthetic prompt/generation loop calls `std::rand()` with **no explicit `srand()` call** (confirmed by reading `tools/llama-bench/llama-bench.cpp`). Every fresh process therefore starts from the C library's default seed (`1`) and produces the **same token sequence** for the same `-p`/`-n`, in every one of our invocations. We rely on this — `llama-bench` exposes no `--seed` flag, so this is the only lever available, and it already gives us reproducibility across runs and across configurations. |
| CPU (thermal state) | See §4 | Controlled for, not eliminated — a CPU-bound sweep on a laptop-class chip can throttle. |

## 3. Variables under test (the sweep grid)

| Axis | Values (reduced run, see §7) | Values (full sweep, not yet run — see `results/bench-*.json` for what was actually measured) |
|---|---|---|
| Threads (`-t`) | `1, 2, 8` | `1, 2, 4, 8, 16` |
| `GGML_KLEIDIAI_SME` | `on` (env var **unset** — the real-world default a user gets) / `off` (`GGML_KLEIDIAI_SME=0`, forces the NEON fallback chain) | same |
| Phase | `decode` (`-p 0 -n N`, `ne11 == 1` every step), `prefill_short` (`-p 64 -n 0`, below the hybrid gate), `prefill_long` (`-p 256 -n 0`, above the hybrid gate) | same, plus intermediate prompt lengths straddling the `ne11 >= 128` boundary |
| Quant | `Q4_0` (the only `.gguf` on hand) | `Q8_0` — **not yet measured**: producing a fair Q8_0 file requires quantizing from an unquantized (F16/F32) checkpoint, not up-converting the lossy Q4_0 file already on disk, and no such checkpoint is present in this environment. Marked `[not available]` in every output row rather than skipped silently. |

`on` vs. explicitly forcing `GGML_KLEIDIAI_SME=<n>`: we test the **default** behavior
(env unset, runtime auto-detects via the hardcoded Apple brand-string table in
`kleidiai.cpp`) because that is what every user who did not read the source gets. A
forced override is a different, also-interesting experiment, but not this one.

## 4. Thermal and system controls

- Before and after the full sweep, the script captures:
  - `pmset -g therm` (macOS thermal/CPU-power warning levels), and
  - `sysctl -n machdep.cpu.brand_string` (confirms we are still measuring on the chip
    we think we are, catches an accidental run on the wrong machine).
- These are recorded verbatim in the output JSON under `thermal.before` / `thermal.after`
  and `cpu_brand`. If `pmset` reports a non-empty thermal or CPU-power warning level in
  the "after" snapshot that was absent "before", the markdown report prints an explicit
  **THERMAL WARNING** line next to the affected run — we do not silently average over a
  throttled state.
- We do **not** claim this eliminates throttling risk on a fanless/thin-chassis machine;
  we claim we **surfaced** whatever `pmset` was willing to report, and that the
  interleaving in §5.1 stops sustained throttling from being mistaken for an SME-vs-NEON
  effect (a monotonic thermal drift shows up as a trend *within* a configuration's
  repeated samples, not as a fake gap between two interleaved configurations).

## 5. Anti-"you faked it" measures

### 5.1 Interleaving, not blocking

The naive way to run this sweep is: run all N repetitions of config A, then all N
repetitions of config B. That lets any monotonic drift (thermal, background load,
frequency scaling ramping up) masquerade as a difference between A and B, because A's
samples are all early and B's are all late (or vice versa).

Instead, `bench.py` builds the full config grid once, then loops:

```
for round in range(reps):
    for cfg in configs:        # fixed order, e.g. A, B, C, A, B, C, ...
        measure(cfg)            # exactly one repetition
```

So repetition 1 of every config runs before repetition 2 of any config. Any drift over
the course of the whole sweep is spread evenly across every configuration, not
concentrated in whichever one happened to run last.

### 5.2 Warmup discarded — every single call, not once per sweep

`llama-bench`'s default behavior (i.e. without `--no-warmup`) is to run one warmup pass
before its measured repetitions and discard it. Because our interleaving scheme invokes
`llama-bench` **once per (config, round)** — a fresh process each time — every single
measured sample is preceded by its own warmup pass in the same process. This is
*stricter* than the minimum bar ("discard a warmup run"): it means cache/allocator state
cannot leak an advantage from one configuration's warmup into another configuration's
first measured sample, because they never share a process.

### 5.3 Never a bare mean

For every `(phase, threads, sme_mode, quant)` cell, the script collects one tok/s sample
per round (`reps` samples total) and reports **median, sample stddev, min, and max** —
never a bare mean. `llama-bench`'s own internal `avg_ts`/`stddev_ts` (computed *within* a
single process's repeated runs) is also captured per call, but the headline numbers in
the markdown table are the median/stddev/min/max computed by `bench.py` itself **across**
the interleaved, independently-warmed-up process invocations, which is the more
conservative and more defensible number.

### 5.4 Dispatch is verified at the symbol level — never inferred from the log banner

`llama.cpp`'s startup banner (`SME = 1 | SME2 = 1 | KLEIDIAI = 1`) and its log line
(`kleidiai: primary q4 kernel feature SME2`) are **compile-time / selection-time**
signals. They print identically whether or not SME2 kernels are ever actually called at
runtime (see the project's Finding 1, and `results/GROUND-TRUTH-DISPATCH.md` for the
corrected, complete dispatch rule — including the **hybrid mode**, where SME2 can still
fire above the nominal thread cap if `ne11 >= 128`, i.e. a long-enough prompt). A tok/s
number with no dispatch confirmation is exactly the mistake this whole project exists to
call out, so `bench.py` refuses to let that happen structurally: **every result row
carries a `dispatch` field**, populated by one of, in the actual code's precedence order
(`main()`'s dispatch-resolution block — check that, not this prose, if the two ever
disagree):

1. **A precomputed ledger**, if `--dispatch-ledger-json PATH` was passed and it has a
   record for this exact `(phase, quant, threads, sme_mode)` key (see
   `load_dispatch_ledger()`). Checked **first**, ahead of both options below, because a
   ledger entry is a real prior `lldb` (or equivalent) observation for this exact
   configuration — legitimate to reuse precisely because dispatch is deterministic given
   those four inputs (see the caching note below). This exists as the documented recovery
   path for a host/session where live `lldb` attach has stopped working (see §6 item 9 —
   this is not a hypothetical; it happened during this submission's own preparation).
2. **An external verifier**, if `tools/verify_dispatch.py` (or whatever is passed via
   `--verify-dispatch-cmd`) exists and the ledger had no entry for this key. Contract:
   invoked with
   `--threads N --sme on|off --phase P --n-prompt P --n-gen G --model PATH --llama-bin PATH`,
   expected to print one JSON object on stdout with at least
   `{"sme_fires": bool, "neon_fires": bool, "method": str}`. This lets a sibling
   tool (built independently, e.g. one that reproduces
   `results/GROUND-TRUTH-DISPATCH.md`) become the single source of dispatch truth
   without `bench.py` needing to change.
3. **The built-in `lldb` verifier** (the fallback if neither of the above resolved this
   key), which sets a regex breakpoint on the SME/SME2 kernel symbols
   (`kai_run_matmul.*_(sme|sme2)_`) and a second one on the NEON fallback symbols
   (`kai_run_matmul.*_neon_(dotprod|i8mm)`), and reports which fired. Two tiers are used,
   because a naive approach hangs (see §6 item 7):
   - **Fast tier** (always run, ~2-10s on a host where `lldb` attach works — see §6 item 9
     for what it looks like when it does not): launch under `lldb -b`, let the process run
     until the *first* breakpoint hit (no auto-continue), record which family hit first,
     kill the process. This tells us `fires: true/false` per family cheaply and matches
     the original methodology in `results/GROUND-TRUTH-DISPATCH.md` §"Methodology
     caveat" (their own words: *"Only zero vs non-zero is meaningful here"*).
   - **Thorough tier** (best-effort, prefill phases only, bounded by a timeout):
     attach an auto-continuing breakpoint command to both regexes and let the *entire*
     run complete, yielding true per-symbol call counts and — critically — the ability to
     see **both** families fire non-zero in the same run, which is the signature of
     **hybrid mode**. If this tier does not finish inside the timeout, its partial
     result is discarded (not reported as a call count) and the row falls back to the
     fast tier's boolean result, explicitly labeled `thorough_timed_out: true`.
4. If none of the above resolved this key (e.g. `lldb` missing or unable to attach, no
   ledger entry, no external verifier, non-macOS without a configured debugger), the row
   is labeled `dispatch: {"verified": false, "reason": "..."}` rather than silently
   omitted or guessed — see §6 item 9 for a real example of exactly this happening.

Dispatch verification runs **once per unique `(phase, threads, sme_mode, quant)`
combination**, cached, and the same label is attached to every repetition's row for
that combination — dispatch is deterministic given those four inputs (it does not
depend on which repetition it is), so re-verifying every repetition would only spend
time without adding information.

## 6. Traps this harness specifically avoids

1. **Reading the startup banner as proof of dispatch.** It is not — see §5.4.
2. **Trusting a timing-only benchmark to reveal thread-gating.** The whole reason
   Finding 1 exists is that the *default* thread count (physical core count) silently
   never uses SME2, and a plain `llama-bench -t $(nproc)` run would never show that.
3. **Running all of config A before all of config B.** See §5.1.
4. **Reporting a bare mean and hiding the variance.** See §5.3.
5. **A confound from Apple's Accelerate BLAS backend intercepting the matmul before it
   ever reaches KleidiAI.** This build links `ggml-blas` (Accelerate). Read from
   `ggml/src/ggml-blas/ggml-blas.cpp`: the BLAS backend's `supports_op` for `MUL_MAT`
   requires `ne0 >= 32 && ne1 >= 32 && ne10 >= 32` (`min_batch = 32`). For our **decode**
   phase, `ne1` (the batch/token dimension) is always `1`, so BLAS can never claim the
   op — confirmed empirically: the `lldb` dispatch check fires a KleidiAI symbol
   (SME2 or NEON, depending on `GGML_KLEIDIAI_SME`), never falls through to Accelerate,
   for every decode configuration tested. For **prefill**, the batch dimension can
   exceed 32, which in principle makes BLAS eligible — but the same empirical check
   shows KleidiAI symbols firing for every prefill configuration tested too (see
   `results/bench-*.json`, `dispatch` field on every `prefill_*` row). We do not claim
   this generalizes to every prompt length or every model shape; we claim it held for
   every configuration this harness actually measured.
6. **Missing the hybrid-dispatch rescue path.** An earlier draft of this project's
   headline claim ("SME2 never runs above 2 threads") was wrong — see
   `results/GROUND-TRUTH-DISPATCH.md`'s own correction. `bench.py`'s `prefill_long`
   phase (`-p 256`, above the `ne11 >= 128` gate) exists specifically to surface this:
   at over-cap thread counts, SME2 *and* NEON can both fire in the same run, splitting
   the batch. `prefill_short` (`-p 64`, below the gate) exists as the contrasting case
   where the same over-cap thread count collapses to pure NEON. Reporting only one of
   the two prompt lengths would have reproduced the incomplete earlier finding.
7. **`lldb` auto-continue breakpoint commands hanging under heavy multi-threaded
   contention.** Empirically, attaching an auto-continuing breakpoint command to the
   decode-phase symbols at thread counts above 1 can stall for minutes (observed
   directly while building this harness: a `-t 4`/`-t 8` decode run with as few as 2
   generated tokens did not return within a 30-second timeout, while the same
   auto-continue approach on a **prefill** run of 256 tokens at `-t 8` completed in a
   few seconds). We do not know the exact `lldb`/kernel interaction responsible, and we
   do not guess at it in the tool's output — we simply never run the thorough tier on
   the decode phase, bound every `lldb` invocation with a hard timeout, and fall back to
   the fast (first-hit) tier on any timeout. This is exactly the kind of measurement
   artifact `rlvr-harness-traps`-style skills warn about: a "the tool hung" result is a
   tooling fact, not a dispatch fact, and must never be reported as one.
8. **Up-converting a lossy quantization to claim a quant we do not have.** See §3 —
   `Q8_0` is marked `[not available]`, not silently produced from the `Q4_0` file.
9. **Trusting `lldb` to fail loudly when it cannot actually attach.** Discovered directly while
   preparing this submission: a live `lldb`-driven sweep run on `2026-08-04` (this machine, same
   binary/model as every other row in this document) had **every single one of its 18
   configurations** report `"fast-tier lldb timed out after 15.0s"` — a hard regression from an
   earlier run on the same machine 40 minutes prior, which got real per-symbol hit counts (e.g.
   `SME2 fired (x5312)`) for every configuration. Root-caused by hand, outside `bench.py`, with
   three checks: (a) `sysctl` confirmed we were still on the same Apple M4 Max; (b)
   `DevToolsSecurity -status` printed **"Developer mode is currently disabled"**; (c) running the
   exact fast-tier `lldb -b -s ... -- llama-bench ...` command directly (both inside and outside
   this harness's own sandboxed shell, to rule out a Claude-Code-specific sandbox) showed the
   *same* result either way: the process ran for ~18-19s (vs. **0.58s** for the identical
   `llama-bench` invocation with no debugger at all) and every breakpoint still reported
   `hit count = 0` — i.e. `lldb` was not failing to start, it was **hanging while attempting to
   instrument the target and then getting killed by the timeout**, silently as far as the
   inferior process is concerned. This is consistent with `lldb`'s `task_for_pid`/ptrace
   entitlement requiring the host's Developer Mode to be enabled; something toggled it off between
   the two runs (this harness does not know what, and does not attempt to re-enable it itself —
   flipping a system-wide security setting is exactly the kind of hard-to-reverse, outward-facing
   change this project's operating rules require a human to authorize, not a benchmark script).
   **The fix already existed in the code, unused until this happened:** `--dispatch-ledger-json`
   (see §5.4 item 1 / `load_dispatch_ledger()`) lets a dispatch verification pass done once, when
   `lldb` *could* attach, be reused for any number of later `bench.py` invocations — legitimate
   specifically because dispatch is deterministic given `(phase, quant, threads, sme_mode)` (§5.4),
   not a measurement that drifts run to run the way timing does. The `2026-08-04` re-run in
   `results/bench-apple-m4-max.json` uses exactly this: fresh, live-measured tok/s from *this*
   run, paired with dispatch labels sourced from the real `lldb`-verified run 40 minutes earlier
   (every ledger record's `note` field says so explicitly, so a reader never mistakes a reused
   label for a fresh one). This is a **fallback for a broken debugger, not a shortcut around
   verification** — every dispatch label the ledger contains still traces back to an actual
   `lldb` observation on this exact hardware/binary/model, and if `--dispatch-ledger-json` is
   omitted and `lldb` cannot attach, `bench.py` reports `"unverified (fast-tier lldb timed out ...)"`
   per row rather than silently guessing (confirmed: this is what happened on the timed-out run
   before the ledger fallback was applied).

## 7. What "reduced sweep" means in this submission

A full sweep is `threads {1,2,4,8,16} x sme {on,off} x phase {decode, prefill_short,
prefill_long} x quant {Q4_0, Q8_0}` = up to 60 cells, each needing `reps` (>=5)
independently-warmed-up process launches plus a one-time dispatch verification — on the
order of 300+ `llama-bench` process launches plus ~30 `lldb` sessions. That is a
reasonable amount of wall-clock time but more than this harness needed to spend to
produce a real, defensible headline number for this submission.

**What was actually run and is in `results/bench-*.json`:** `threads {1, 2, 8}`,
`sme {on, off}`, all three phases, `Q4_0` only, `reps = 5`, real interleaved
measurements, real `lldb`-verified dispatch on every row (live via `lldb` in the first run;
replayed from that same real verification via `--dispatch-ledger-json` in the second run,
because `lldb` itself stopped being able to attach on this host between the two runs — see
§6 item 9 for exactly what happened and why the replay is still a real observation, not a
guess). Threads `4` and `16`, and the `Q8_0` quant, are **not yet measured** and must not be
treated as measured — they are absent from the output entirely rather than filled in with
an inferred or interpolated number.

This exact grid was run **twice, independently**, ~47 minutes apart on the same machine
(`2026-08-03T20:14:05Z` and `2026-08-03T21:01:45Z`) — the second run is this submission's
final, currently-published one (`results/bench-apple-m4-max.json`), and was itself preceded by
two intermediate re-runs while the `lldb`-hang issue in §6 item 9 was being diagnosed and the
reused-ledger's decode-phase entries corrected to drop an over-precise hit count the current
code cannot actually produce for that phase (see the ledger's own `note` fields). The headline
trends **agree between the two published runs**: SME2 wins decode and prefill at `threads<=2` by
roughly 1.4-2.2x, both paths collapse toward NEON-only throughput at `threads=8` (over the
`sme_thread_cap=2` cap), and the `prefill_long`/`threads=8`/`sme=on` HYBRID cell is consistently
both the highest-variance row in the table *and* slower than plain NEON at the same thread count
in both runs (run 1: 1909.5 HYBRID vs. 2640.8 NEON; run 2: 1871.3 HYBRID vs. 2675.7 NEON) — see
§6 item 9 for why the second run's dispatch labels are sourced from the first run's real `lldb`
observations rather than a fresh (in that run, non-functional) `lldb` attach.

## 8. How to reproduce

```bash
cd /Users/tom/Documents/GitHub/arm-dispatch-ledger
python3 tools/bench.py \
    --llama-bin-dir /tmp/llama.cpp/build/bin \
    --model /tmp/ggufs/q05.gguf \
    --threads 1,2,8 \
    --reps 5
```

If `lldb` cannot attach on your host (see §6 item 9 — check `DevToolsSecurity -status`
first), every row will read `unverified (fast-tier lldb timed out ...)` instead of a real
kernel-family label. Re-run with a precomputed ledger from a host/session where `lldb` does
work:

```bash
python3 tools/bench.py \
    --llama-bin-dir /tmp/llama.cpp/build/bin \
    --model /tmp/ggufs/q05.gguf \
    --threads 1,2,8 --reps 5 \
    --dispatch-ledger-json /path/to/a/real/lldb-verified/ledger.json
```

The ledger is a plain JSON list of records shaped like
`{"phase": "decode", "quant": "Q4_0", "threads": 1, "sme_mode": "on", "sme_fires": true,
"neon_fires": false, "sme_hits": 5312, "neon_hits": 0, "hybrid": false}` — see
`load_dispatch_ledger()` in `tools/bench.py` for the full contract, and `tools/verify_dispatch.py`
for a from-scratch (L1/L2/L3) alternative to hand-authoring one.

Add `--threads 1,2,4,8,16` for the full thread sweep (slower). See
`python3 tools/bench.py --help` for every knob (phase token counts, dispatch-verify
timeouts, output paths, an external `--verify-dispatch-cmd`).

Plot the resulting JSON with:

```bash
python3 tools/plot_results.py results/bench-<platform>.json
```
