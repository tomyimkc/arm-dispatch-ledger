<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Dispatch Ledger contributors
SPDX-License-Identifier: Apache-2.0
-->
# Crossover harness protocol — `crossover.py`

This document states the exact methodology behind `tools/crossover.py` and
`results/crossover/crossover-*.{json,md}`. Those artifacts are the **baseline
the optimization claim in this submission rests on**: `tools/bench.py`
already showed *that* SME2 and NEON trade wins by phase (see
`results/SUMMARY.md`); this harness is the narrower, dedicated instrument
that pins down the *exact* per-phase optimum, what a default install actually
gets, the best config expressible today, and the (currently unreachable)
target a phase-aware patch should aim for.

## 1. The question being answered

For **decode** and **prefill** separately, across `threads {1,2,4,8,16}` and
`GGML_KLEIDIAI_SME {unset ("on"), 0 ("off")}`:

1. What is the real per-phase optimum (thread count + kernel family)?
2. What does the llama.cpp **default** configuration (no `-t`/`-tb` flags at
   all) actually give a user today?
3. What is the best **split-phase** config expressible today, using
   llama.cpp's existing separate `-t` (generation/decode threads) and `-tb`
   (batch/prompt-processing threads) flags?
4. What is the **theoretical best** — the per-phase optimum for decode paired
   with the per-phase optimum for prefill, independently — and is that
   pairing actually reachable with a single process-global
   `GGML_KLEIDIAI_SME` setting, or not?

(4) is the crux of the whole submission: `GGML_KLEIDIAI_SME` is read once at
process start (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`) and there is no
per-call or per-phase override. If decode's optimum and prefill's optimum
land on *different* `GGML_KLEIDIAI_SME` states, (4) is provably unreachable
without a code change — which is exactly the gap a phase-aware
kernel-family-selection patch would close.

## 2. Output format: JSON, not CSV

A previous ad-hoc attempt in this project failed to parse `llama-bench`'s CSV
output. Before writing any parser, this script's development ran the exact
target command once and inspected the raw output directly:

```
llama-bench -m /tmp/ggufs/q05.gguf -p 256 -n 0 -t 2 -r 3 -o json
```

```json
[
  {
    "build_commit": "dbadb68", "cpu_info": "Accelerate, Apple M4 Max",
    "model_type": "qwen2 1B Q4_0", "n_threads": 2,
    "n_prompt": 256, "n_gen": 0,
    "avg_ns": 159347416, "stddev_ns": 243519,
    "avg_ts": 1606.555049, "stddev_ts": 2.450724,
    "samples_ns": [ 159479791, 159495334, 159067125 ],
    "samples_ts": [ 1605.22, 1605.06, 1609.38 ]
  }
]
```

`-o json` returns a uniformly-shaped array with one object per test, and
`samples_ts`/`samples_ns` already exclude `llama-bench`'s own internal
warm-up run (with `-r N`, the array has exactly `N` elements, not `N+1`).
There is no header/units row to misalign, unlike CSV. `crossover.py`
therefore always invokes `llama-bench -o json` and never touches CSV.

`llama-bench` has `-t`/`--threads` but **no** `-tb`/`--threads-batch` (checked
via `llama-bench --help`; confirmed absent). `llama-cli` has both (`-t` for
generation, `-tb` for batch/prompt processing — confirmed via `llama-cli
--help`). This is *why* the split-phase measurement (question 3 above) falls
back to `llama-cli` and parses its `[ Prompt: X t/s | Generation: Y t/s ]`
completion-summary line instead of `llama-bench`'s JSON — not a style choice,
a capability gap in `llama-bench` itself.

## 3. Axes and phase definitions

| Axis | Values |
|---|---|
| Threads | `1, 2, 4, 8, 16` |
| `GGML_KLEIDIAI_SME` | `on` (env var **unset** — real-world default, auto-detect) / `off` (`GGML_KLEIDIAI_SME=0`, forces NEON) |
| Phase | `decode` (`-p 0 -n 32`), `prefill` (`-p 256 -n 0`) |

`decode`/`prefill` token counts match `tools/bench.py`'s `decode` /
`prefill_long` definitions exactly, so numbers from the two harnesses are
directly comparable:

- `decode`: `ne11 == 1` every step (always below the SME2 hybrid-dispatch
  gate; see `results/GROUND-TRUTH-DISPATCH.md`).
- `prefill`: `ne11 == 256 >= 128` (in the regime where the SME2 "hybrid"
  rescue path can engage even past `sme_thread_cap`).

5 threads × 2 SME states × 2 phases = **20 cells**, ≥5 repetitions per cell
(default 5).

## 4. Anti-drift methodology

- Every repetition is its **own fresh `llama-bench -r 1` process** (never
  `-r 5` in one process for the main sweep), so the 20 cells can be genuinely
  **interleaved**: for `round in range(reps): for cell in cells: run once`.
  The fixed cell order (same every round) puts the two `GGML_KLEIDIAI_SME`
  states back-to-back within each `(phase, threads)` pair — the most
  decision-relevant comparison is also the most thermally-adjacent one — and
  rotates across thread counts/phases within a round so no single cell is
  systematically favoured by its position in the run.
- Every repetition still gets `llama-bench`'s own internal warm-up pass (not
  `--no-warmup`); `samples_ts` already excludes that sample (see §2), so
  "discard warm-up" is satisfied by construction, never double-counted.
- `pmset -g therm` and `os.getloadavg()` are captured before and after the
  full run and persisted verbatim (`meta.thermal`, `meta.load_average` in the
  JSON) so a reviewer can check for thermal throttling or CPU contention
  independently, rather than take a clean-room result on faith.
- Reports **median/stddev/min/max** per cell — never a bare mean.
- A failed measurement is recorded as an entry in `sweep_errors` /
  `default_errors` / `split_phase_errors` with the real error text, never
  silently dropped, estimated from neighbors, or interpolated. A cell missing
  from the tables was **not measured** — it must not be read as zero or
  equal-to-neighbor.

## 5. Contention note (read this before trusting an absolute number here)

This measurement ran on a machine **shared with other concurrent, unrelated
agent sessions** (this repo's own multi-agent setup: several sibling work
packages plus, observed directly during this run, unrelated
`python -m contest_bench...` multiprocessing workers from a different
project entirely, each pegged near 95-100% CPU). During development of this
harness, `uptime` reported **1-minute load averages of 100-185 on a
16-physical-core machine** (`sysctl -n hw.physicalcpu` = 16) — i.e. 6-11x
oversubscribed — and a single calibration `llama-bench` decode call
(normally sub-second) was observed to **not complete within a 180-second
timeout** during a contention burst.

To keep measuring *real* numbers rather than giving up or fabricating clean
ones, `crossover.py`:

- Uses a generous per-call timeout (180s default, `--per-call-timeout` to
  change it).
- Retries a call up to twice on a timeout (`call_with_retries()`) — this
  re-runs the **identical real subprocess command**; it never estimates,
  reuses a stale sample, or fabricates a number. Non-timeout errors (bad
  JSON, a real non-zero exit) are never retried.
- Records `n_retries_used` and a full `retry_log` (which cell, which round,
  which attempt) in the JSON output, and prints the load-average
  before/after into the markdown report's "Contention note" section
  whenever the 1-minute load average exceeded the physical core count during
  the run.

**Read every number in `results/crossover/crossover-apple-m4-max.md` with
this in mind.** Higher-thread cells (8, 16, and the no-flags default which
resolves to 12 — see §6) contend more directly with other processes for the
same physical cores and are the ones most likely to show inflated stddev or
suppressed medians from contention, not from this workload's own
thread-count/kernel-family behaviour. The *relative* comparison within a
phase (which thread count / kernel family wins) is expected to be more
robust to shared-machine contention than the *absolute* tok/s magnitudes,
since contention is not selective per configuration — but that expectation
has not been independently verified by re-running on a quiet machine, so it
is a caveat here, not a correction applied to any number.

## 6. llama.cpp's real default thread count (measured, not assumed)

`results/SUMMARY.md` (a sibling work package's output) describes the
no-flags default as "physical core count, 16 here". This harness measured
the actual value directly on this build/machine and found it to be **12**,
not 16:

```
$ llama-cli -m /tmp/ggufs/q05.gguf -no-cnv -st --simple-io -n 4 -p "hi" -v 2>&1 | grep n_threads
0.00.001.021 I cmn  common_param: system_info: n_threads = 12 (n_threads_batch = 12) / 16 | ...
```

Root cause, read directly from `common/common.cpp` in the built checkout:
`common_cpu_get_num_physical_cores()` queries `hw.perflevel0.physicalcpu`
(Apple's **performance**-core count) first on `__APPLE__`, falling back to
`hw.physicalcpu` (all cores) only if that sysctl is unavailable. On this M4
Max (12 P-cores + 4 E-cores), `hw.perflevel0.physicalcpu` = 12, so that is
what a real no-flags invocation uses — confirmed by the log line above, and
matching `llama-bench`'s own separately-observed default of `-t 12`. This
harness's "default configuration" measurement (deliverable (b)) passes **no**
`-t`/`-tb` flag at all rather than hard-coding `-t 16`, so it measures
whatever this real resolution path produces, not an assumption about it.

## 7. Split-phase config (`-t`/`-tb`, no patch)

`GGML_KLEIDIAI_SME` is still a single process-global env var — a `llama-cli`
process cannot use SME2 for decode and NEON for prefill simultaneously, only
different **thread counts** per phase via `-t <decode_threads> -tb
<prefill_threads>`. This harness therefore measures the split-phase config
under **both** available `GGML_KLEIDIAI_SME` states (never guesses which is
better) and reports both, so a reader can see directly which single global
state gets closer to the per-phase optima.

## 8. Reproduce

```
python3 tools/crossover.py --threads 1,2,4,8,16 --sme-modes on,off --reps 5 \
  --per-call-timeout 180 --out-dir results/crossover
```

Reduced axis (if the full sweep is too slow on a given machine):

```
python3 tools/crossover.py --threads 1,2,8 --reps 3 --out-dir results/crossover
```

Any reduction from the full `{1,2,4,8,16} x {on,off} x {decode,prefill}`
grid must be stated explicitly in the accompanying write-up, listing exactly
which cells were skipped — never interpolated. See `--help` for every knob.
