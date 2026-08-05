<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Polygraph contributors (formerly Arm Dispatch Ledger) -->

# The claims registry -- a reusable pattern for a project that has shipped a wrong number

This repo has, twice, shipped a wrong number: a fabricated **"+57.3%"** win (produced by
comparing a baseline and a patched config that were measured in different, unevenly
contended time windows -- see `results/REMEASURE-2026-08-04-QUIET.md` for the full
retraction), and a stale duplicate Devpost submission file that no longer matched the
corrected numbers elsewhere in the repo. Both were **promise-based failures**: nothing
*mechanically* stopped a wrong, stale, or drifted number from being displayed. A person
had to remember to update every place a number appears, every time it changed, forever.

`tools/check_claims.py` is the structural fix, and this file is its single source of
truth. It is deliberately a **reusable pattern**, not a one-off script: any project that
(a) measures things, (b) writes the results into more than one document, and (c) has
ever shipped a stale or wrong number can adopt the same three checks with the same
stdlib-only tool.

## What the checker does

`python3 tools/check_claims.py` runs three checks against the current working tree:

1. **Retraction guard.** This project's own previously-retracted figures (57.3, 71.6,
   45.5, 4.4x, 198.9, 2257.5, 1145.0 -- the fabricated "+57.3%" episode and its
   downstream numbers) must never appear anywhere in the repo's prose *unless* the
   occurrence is inside a section that itself carries retraction language
   (retracted/superseded/historical), or has an explicit, reasoned exemption below for a
   genuinely coincidental digit collision with an unrelated, currently-valid number.
2. **Live claim registry.** Every numeric performance claim (a ratio like `3.43x`, a
   throughput figure like `321.0 tok/s`, a signed/approximate percentage like `+57.3%` or
   `~12%`, a dispatch hit count) found by grepping `README.md`, `docs/*.md` and `site/`
   must resolve to either (a) a hand-curated entry below whose cited `results/` source
   file (and, for a handful of the highest-stakes numbers, an exact JSON path) actually
   contains that value, or (b) any raw numeric leaf value committed anywhere under
   `results/**/*.json` (so a number nobody ever actually measured cannot be typed in).
3. **Cross-file agreement.** A corollary of (2): because every live number must resolve
   to one canonical, source-backed value, two files that print *different* numbers for
   "the same" cell cannot both pass -- the wrong one simply has no source that backs it.
   This is how the tool catches silent drift, not just outright fabrication.

Exit code 0 means every scanned claim is registered/backed and no retracted figure leaks
unmarked. Exit code 1 prints a grouped, actionable report (file, line, matched text,
reason) and is wired into CI as `.github/workflows/claims.yml` on every push and PR.

## Scope: what counts as a "claim"

The checker does not treat every digit in the repo as a claim needing registration --
that would flag thread counts, core counts, file sizes, dates, and issue numbers as
"unregistered performance claims" and drown the real signal in noise. It specifically
targets the numeric shapes that carry a performance/measurement claim: a ratio
(`N.NNx`/`N.NN×`), a throughput figure (`N.N tok/s`, including `A -> B tok/s` and
`A +/- B tok/s` forms), a signed or approximate percentage (`+N%`, `-N%`, `~N%`, or an
unsigned percentage immediately preceded by "by"/"within"), and a dispatch hit count
(`N hits`, or a bare number inside a markdown table whose header row says "hits" or
"tok/s" -- some of this repo's tables put the unit word only in the header, not each
cell). Everything else (thread counts, dates, core counts, GFLOP thresholds inside
backtick code spans, `#NNNN` issue references) is deliberately out of scope.

## Why percentages are always hand-registered

A percentage is always *derived* (computed from two other numbers), never a raw
measurement leaf sitting in a JSON file. If percentages were allowed through the
blanket "any number appearing anywhere in `results/**/*.json`" fallback, a `12%` could
pass only because *some unrelated* JSON field -- a thread count, an `n` of 12 reps --
happens to equal 12. That would be a meaningless pass, not a verification. So every
percentage claim below is either verified by a literal substring match against its
cited source file, or (better, where the two base numbers are already known) by a
`compute` block that recomputes the ratio/percentage from those two numbers and checks
it against the declared value -- real arithmetic verification, not `eval()` on an
arbitrary string (the tool supports exactly three fixed operations: `ratio`,
`percent_drop`, `percent_rise`).

## Why headline numbers are hand-registered even though Tier 2 might also pass them

During development, several of this project's own highest-stakes numbers (`93.6`,
`321.0`, `1230.3`, `2198.1`, `1.79`) were found to pass the blanket JSON-backing
fallback **by coincidence** -- they happen to round, at one decimal place, to some
unrelated leaf value elsewhere in the multi-hundred-number `results/**/*.json` corpus,
not to the real source of those figures (`results/REMEASURE-2026-08-04-QUIET.md`, which
is markdown, not JSON -- that measurement session's raw per-rep data was never
committed as JSON). A 1-decimal rounding bucket is coarse enough that a fabricated
number has a real chance of coincidentally landing on *some* real leaf somewhere in a
large corpus. That is precisely the failure mode this tool exists to close, so every
cross-file or headline claim is hand-entered below regardless of whether Tier 2 would
also have passed it -- a coincidental match is never the *only* thing standing behind a
number that actually matters.

## How to add a claim

Add an entry to the `claims` array in the registry block below:

```json
{
  "id": "short-kebab-case-id",
  "value_text": "3.43",
  "source_file": "results/YOUR-FILE.md",
  "note": "what this number is, one line"
}
```

Optional fields: `aliases` (other literal spellings that should resolve to the same
claim, e.g. `"3.43x"`, `"3.43×"`, `"1,230.3"`), `source_json_path` + `source_json_file`
(for a claim that should be checked against an exact JSON path, not just "somewhere in
this file"), and `compute` (`{"op": "ratio"|"percent_drop"|"percent_rise", "numerator":
N, "denominator": M}` -- verifies `value_text` against real arithmetic on two already-
known numbers instead of a substring match).

## How retraction exemptions and non-claim exemptions work

- `retraction_exemptions`: for a number that happens to share digits with a retracted
  figure but is a genuinely different, currently-valid measurement (not the retracted
  claim under a new coat of paint). Requires `file`, `match_substring` (an exact
  substring of the offending line -- if the line changes, the exemption stops matching
  and the check goes back to enforcing, which is intentional), and `reason`.
- `non_claim_exemptions`: for a number the extractor's pattern-matching catches (e.g. "N
  hits") that is not actually a claim about this project's own results -- an anecdote
  about someone else's methodology mistake, a qualitative "these two tools roughly
  agree" statement with no single clean source. Same shape as a retraction exemption.

## Resolved findings (fixed during hostile-verification pass, 2026-08-04)

The checker surfaced real, pre-existing drift on its first run against the tree. That
drift has since been fixed by correcting the prose to match the currently-committed
`results/dispatch-ledger-darwin-arm64.json` (the JSON was not touched; it remains the
source of truth). Recorded here for the audit trail rather than deleted silently:

- **`docs/UPSTREAM-ISSUE.md`, threads=8 prefill hit count** -- stated `1547 / 13692`;
  the currently-committed `results/dispatch-ledger-darwin-arm64.json` for that exact
  config (`threads=8`, `workload=prefill_long`) says `sme2=1538, dotprod=13702` --
  matching `README.md`'s own citation of the same cell exactly. `docs/UPSTREAM-ISSUE.md`
  disagreed with both; this was a plain transcription slip. **Fixed:** corrected to
  `1538 / 13702`.
- **threads=16 decode and threads=4/16 prefill hit counts** -- `README.md`,
  `docs/UPSTREAM-ISSUE.md`, and `results/SUMMARY.md` all agreed with each other on
  `51,214` / `6,712` / `1,403` / `21,509`, but the currently-committed
  `results/dispatch-ledger-darwin-arm64.json` says `51215` / `6711` / `1377` / `21534`
  for those same cells. Most likely explanation: the committed JSON was regenerated in a
  later session than the one the three prose documents transcribed from, and the new
  numbers were never propagated back into the prose. **Fixed:** all three files
  corrected to `51,215` / `6,711` / `1,377` / `21,534`, matching the committed JSON
  exactly. Per `results/GROUND-TRUTH-DISPATCH.md`'s own methodology caveat, these are
  `lldb` stop counts, not deterministic call counts, and only zero-vs-nonzero is the
  load-bearing signal -- so this correction is a consistency fix (prose must match the
  one committed JSON it cites), not a re-measurement.
- **`README.md` / `results/SUMMARY.md`, "1514.1 ± 198.9"** -- the median (1514.1) was
  exactly right; the stdev was off by 0.1 (`results/bench/bench-apple-m4-max.json`'s
  `stddev_ts` for that cell is `198.84947782179356`, which rounds to `198.8`, not
  `198.9`, at one decimal place). **Fixed:** both files corrected to `198.8`. The
  `non_claim_exemptions` entry for `"1514.1 ± 198.9"` below is now dead (the string no
  longer appears anywhere) but is left in place rather than deleted, since it is still
  accurate documentation of why that digit sequence was never the retracted `198.9`
  figure, should the pre-fix text resurface in a future diff.

None of these are fabricated or invented numbers, and none of them are the retracted
"+57.3%" episode resurfacing -- they are small, honest drift between a raw-data
regeneration and the prose that quoted it, which is exactly the kind of thing that goes
unnoticed without a mechanical gate. That the gate found them on its first run, against
a repo that had already been carefully hand-reviewed multiple times, is the argument for
building it.

## New in this update: DGX Spark server lane (2026-08-05)

`results/server/` adds three files -- `server-bench.json`, `server-dispatch.json`, and
`spark-provenance.txt` -- measured on a second machine (NVIDIA DGX Spark, GB10,
Cortex-X925/Cortex-A725, gcc 13.3.0, aarch64). All three are additive; no existing
`results/` artifact was edited (see `results/RENAME-NOTE.md`'s "don't touch prior
evidence" ethos, applied here to a different concern -- a new lane, not a rename). Every
server-lane claim added to the registry below cites one of those three files as
`source_file`, and most cite an exact `source_json_path` into `server-bench.json` or
`server-dispatch.json` rather than a bare substring match, since those two files are JSON,
not prose.

Two things this pass deliberately did **not** register or overstate, found by reading the
committed files directly instead of trusting a session summary of them (per this
project's own no-overclaim discipline):

- **`sve: 0` dispatch calls under concurrent serving load is documented in prose, not
  registered as a machine-checked numeric claim.** `results/server/server-dispatch.json`
  contains exactly two keys (`dotprod`, `i8mm`); `sve` is entirely absent -- zero is the
  correct reading, but it is an absence, not a literal `0` JSON leaf. A registry entry
  with `value_text: "0"` against that file would only "pass" this tool's substring check
  by coincidence (almost any file with enough digits in it contains a lone `"0"`
  somewhere, e.g. inside `"11360"`) -- the exact same reasoning this file already applies
  to percentages under "Why percentages are always hand-registered" above. Registering it
  anyway would be a meaningless pass, so `docs/DEVPOST-SUBMISSION.md` states "SVE is never
  entered" in prose instead.
- **The build's total `kai_` symbol counts are not registered and are shown in
  `[brackets]`.** Only the `kai_run_matmul`-specific counts -- `0` in the default build,
  `10` in the fixed build -- are literally present in
  `results/server/spark-provenance.txt` (`"kai_run_matmul symbols: 0"` /
  `"kai_run_matmul symbols: 10"`, lines 9 and 16). A broader total across all `kai_`
  symbols (packing helpers included, not just the matmul micro-kernels) is not present
  anywhere in the committed provenance capture (`grep -n "36\|149"
  results/server/spark-provenance.txt` returns nothing), so per this project's own hard
  rule -- never state a number not in `results/`, bracket it otherwise -- that broader
  total is not asserted as a verified figure.
- **TTFT p99 does not monotonically improve with concurrency, and does not stay under
  170ms across the whole 1-16 sweep.** `results/server/server-bench.json` shows a spike
  to **0.221s (221ms) at 4 concurrent clients** -- higher than every other row, including
  the 16-client row (0.168s). `docs/DEVPOST-SUBMISSION.md` states the true range
  (89-221ms) and the spike, not a smoothed-over "stays under 170ms" claim.

## New in this update: cross-model scale experiment (2026-08-05)

`results/scale/scale-experiment.json` adds a single new raw-data file, measured on the same DGX
Spark as the server lane above, that sweeps thread count for **two** model sizes
(`Qwen2.5-0.5B-Instruct-Q4_0` and `Qwen2.5-7B-Instruct-Q4_0`) and re-measures the Finding 3
build-defect throughput cost at both sizes. It is additive — no existing `results/` artifact was
edited. Every claim below cites an exact `source_json_path` into that one file rather than a bare
substring match, since it is JSON, not prose. This is a correction as much as an addition: it shows
this project's own headline decode-tuning multiple (3.43x, measured on a 0.5B model on Apple
Silicon) shrinking to 1.33x on a 7B model, and the Finding 3 build-defect cost moving the *opposite*
direction (bigger, not smaller, on the 7B model) — see `README.md`'s "Does the 3.43x decode-tuning
win generalize to a bigger model?" and the throughput-cost table under Finding 3.

Registered below: the 24 thread-sweep medians (6 thread counts × 2 phases × 2 model sizes), the 8
Finding-3 throughput-cost medians (2 builds × 2 phases × 2 model sizes), and the 6 derived ratios
(`4.56x`, `1.33x`, `1.42x`, `0.99x`, `4.57x`, `1.65x`), each `compute`-verified against the two
medians it derives from rather than substring-matched — the same discipline this file already
applies to every other derived ratio in the registry (see "Why percentages are always
hand-registered" above; the same reasoning applies to any *derived* number, ratios included, even
though the tool's Tier 2 JSON-backing fallback does accept ratios). Per-rep standard deviations from
this file are **not** separately registered below (out of this update's stated scope), but every
`X ± Y tok/s` figure printed in `README.md` that carries one is still a real, raw leaf value in
`results/scale/scale-experiment.json` and passes the checker's Tier 2 JSON-backing fallback on its
own merits.

## The registry

<!-- CLAIMS-REGISTRY:BEGIN -->
```json
{
  "schema_version": 1,
  "retracted_figures": [
    "57.3",
    "71.6",
    "45.5",
    "4.4x",
    "198.9",
    "2257.5",
    "1145.0",
    "1,145.0",
    "2,257.5"
  ],
  "retraction_context_keywords": [
    "retract",
    "supersede",
    "superseded",
    "historical",
    "corrected",
    "correction"
  ],
  "retraction_exemptions": [
    {
      "file": "README.md",
      "match_substring": "1514.1 ± 198.9",
      "reason": "coincidental digit collision, not the retracted claim: this is tools/bench.py's own stddev_ts for prefill_long/threads=16/SME-off (results/bench/bench-apple-m4-max.json, stddev_ts=198.84947782179356) -- an unrelated cell to the retracted patch-comparison figure. Note for a future fix (outside this work package's file scope): the JSON value rounds to 198.8 at 1dp, not 198.9 as displayed -- a separate, minor, pre-existing rounding slip in README.md/results/SUMMARY.md, not a retraction issue."
    }
  ],
  "non_claim_exemptions": [
    {
      "file": "docs/UPSTREAM-ISSUE.md",
      "match_substring": "1992 hits",
      "reason": "illustrative anecdote about a methodology pitfall (an unanchored, non-auto-continuing lldb breakpoint double-counting a call), not a claim about this project's own measured results. The correct count for that same run (996) is registered separately and is JSON-backed."
    },
    {
      "file": "README.md",
      "match_substring": "agree within ~2%",
      "reason": "qualitative cross-tool agreement statement (bench.py vs crossover.py overlapping cells), not a precise point-value performance claim with a single clean source."
    },
    {
      "file": "README.md",
      "match_substring": "stayed stable within ~5%",
      "reason": "qualitative rerun-to-rerun stability statement about the kernel microbenchmark (results/bench/kernel-bench-apple-m4-max.md's own caveat section), not a point-value performance claim."
    },
    {
      "file": "docs/DEVPOST-SUBMISSION.md",
      "match_substring": "within 1-2% every time",
      "reason": "qualitative range statement (the 0002 patch reaches its own -t <cap> target within a 1-2% band across the three results/GENERALIZATION.md configs), not a single point-value performance claim with one clean source. The extractor's percentage pattern incidentally matches the range's second bound as '-2%', which is a parsing artifact of the '1-2%' range shape, not a claim needing separate registration."
    }
  ],
  "json_backed_globs": [
    "results/*.json",
    "results/**/*.json"
  ],
  "claims": [
    {
      "id": "remeasure-decode-default",
      "value_text": "93.6",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "decode, llama.cpp default (no -t/-tb), n=7"
    },
    {
      "id": "remeasure-decode-tuned",
      "value_text": "321.0",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "decode, -t 2, n=7"
    },
    {
      "id": "remeasure-decode-ratio",
      "value_text": "3.43",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "321.0 / 93.6 tuning win",
      "aliases": [
        "3.43x",
        "3.43×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 321.0,
        "denominator": 93.6
      }
    },
    {
      "id": "remeasure-prefill-default",
      "value_text": "1230.3",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "prefill, llama.cpp default (no -t/-tb), n=7",
      "aliases": [
        "1,230.3"
      ]
    },
    {
      "id": "remeasure-prefill-tuned",
      "value_text": "2198.1",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "prefill, -t 8, n=7",
      "aliases": [
        "2,198.1"
      ]
    },
    {
      "id": "remeasure-prefill-ratio",
      "value_text": "1.79",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "2198.1 / 1230.3 tuning win",
      "aliases": [
        "1.79x",
        "1.79×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 2198.1,
        "denominator": 1230.3
      }
    },
    {
      "id": "remeasure-decode-stdev-default",
      "value_text": "2.47",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "stdev, decode default, n=7"
    },
    {
      "id": "remeasure-decode-stdev-tuned",
      "value_text": "2.09",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "stdev, decode -t 2, n=7"
    },
    {
      "id": "remeasure-prefill-stdev-default",
      "value_text": "118.52",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "stdev, prefill default, n=7"
    },
    {
      "id": "remeasure-prefill-stdev-tuned",
      "value_text": "72.59",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "stdev, prefill -t 8, n=7"
    },
    {
      "id": "remeasure-patch-decode-default",
      "value_text": "82.5",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "decode, patched+flag, default threads, n=7 -- the measured regression"
    },
    {
      "id": "remeasure-patch-decode-default-ratio",
      "value_text": "0.88",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "82.5 / 93.6",
      "aliases": [
        "0.88x",
        "0.88×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 82.5,
        "denominator": 93.6
      }
    },
    {
      "id": "remeasure-patch-decode-default-stdev",
      "value_text": "4.07",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "stdev, decode patched+flag default, n=7"
    },
    {
      "id": "remeasure-patch-decode-tuned",
      "value_text": "317.5",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "decode, patched+flag, -t 2, n=7 -- statistical tie with 321.0"
    },
    {
      "id": "remeasure-patch-decode-tuned-ratio",
      "value_text": "0.99",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "317.5 / 321.0",
      "aliases": [
        "0.99x",
        "0.99×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 317.5,
        "denominator": 321.0
      }
    },
    {
      "id": "remeasure-patch-decode-tuned-stdev",
      "value_text": "3.58",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "stdev, decode patched+flag -t 2, n=7"
    },
    {
      "id": "remeasure-patch-prefill-default",
      "value_text": "1202.1",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "prefill, patched+flag, default threads, n=7 -- tie",
      "aliases": [
        "1,202.1"
      ]
    },
    {
      "id": "remeasure-patch-prefill-default-ratio",
      "value_text": "0.98",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "1202.1 / 1230.3",
      "aliases": [
        "0.98x",
        "0.98×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 1202.1,
        "denominator": 1230.3
      }
    },
    {
      "id": "remeasure-patch-prefill-default-stdev",
      "value_text": "96.26",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "stdev, prefill patched+flag default, n=7"
    },
    {
      "id": "remeasure-patch-regression-pct",
      "value_text": "12",
      "source_file": "results/REMEASURE-2026-08-04-QUIET.md",
      "note": "~12% slower: 1 - 82.5/93.6",
      "aliases": [
        "~12",
        "+12",
        "-12"
      ],
      "compute": {
        "op": "percent_drop",
        "numerator": 93.6,
        "denominator": 82.5
      }
    },
    {
      "id": "reconcile-decode-t1",
      "value_text": "1.39",
      "source_file": "results/SUMMARY.md",
      "note": "decode threads=1, SME on/off ratio",
      "aliases": [
        "1.39x"
      ]
    },
    {
      "id": "reconcile-decode-t2",
      "value_text": "1.23",
      "source_file": "results/SUMMARY.md",
      "note": "decode threads=2, SME on/off ratio",
      "aliases": [
        "1.23x"
      ]
    },
    {
      "id": "reconcile-prefill-t1",
      "value_text": "2.16",
      "source_file": "results/SUMMARY.md",
      "note": "prefill_long threads=1, SME on/off ratio",
      "aliases": [
        "2.16x"
      ]
    },
    {
      "id": "reconcile-prefill-t2",
      "value_text": "2.02",
      "source_file": "results/SUMMARY.md",
      "note": "prefill_long threads=2, SME on/off ratio",
      "aliases": [
        "2.02x"
      ]
    },
    {
      "id": "reconcile-prefill-t8",
      "value_text": "0.68",
      "source_file": "results/SUMMARY.md",
      "note": "prefill_long threads=8, SME on/off ratio (NEON wins)",
      "aliases": [
        "0.68x"
      ]
    },
    {
      "id": "reconcile-prefill-neon-vs-sme-best",
      "value_text": "1.46",
      "source_file": "results/SUMMARY.md",
      "note": "NEON@8 (2676.4) vs SME2's own best (1830.1 hybrid@8)",
      "aliases": [
        "1.46x"
      ]
    },
    {
      "id": "reconcile-prefill-neon-vs-sme-t2",
      "value_text": "1.64",
      "source_file": "results/SUMMARY.md",
      "note": "NEON@8 (2676.4) vs SME2@2 (1629.1)",
      "aliases": [
        "1.64x"
      ]
    },
    {
      "id": "decomp-default-sme-on",
      "value_text": "48.0",
      "source_file": "README.md",
      "note": "decomposition sweep, default threads(12), SME on"
    },
    {
      "id": "decomp-default-sme-off",
      "value_text": "59.6",
      "source_file": "README.md",
      "note": "decomposition sweep, default threads(12), SME off"
    },
    {
      "id": "decomp-t2-sme-on",
      "value_text": "309.2",
      "source_file": "README.md",
      "note": "decomposition sweep, -t 2, SME on"
    },
    {
      "id": "decomp-t2-sme-off",
      "value_text": "235.7",
      "source_file": "README.md",
      "note": "decomposition sweep, -t 2, SME off"
    },
    {
      "id": "decomp-total-win",
      "value_text": "6.44",
      "source_file": "README.md",
      "note": "total: 309.2/48.0",
      "aliases": [
        "6.44x"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 309.2,
        "denominator": 48.0
      }
    },
    {
      "id": "decomp-thread-tuning-alone",
      "value_text": "3.95",
      "source_file": "README.md",
      "note": "thread tuning alone (SME off throughout): 235.7/59.6",
      "aliases": [
        "3.95x"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 235.7,
        "denominator": 59.6
      }
    },
    {
      "id": "decomp-sme-contrib-t2",
      "value_text": "1.31",
      "source_file": "README.md",
      "note": "SME2's contribution at -t 2: 309.2/235.7",
      "aliases": [
        "1.31x"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 309.2,
        "denominator": 235.7
      }
    },
    {
      "id": "decomp-sme-contrib-default",
      "value_text": "0.81",
      "source_file": "README.md",
      "note": "SME2's contribution at default threads: 48.0/59.6",
      "aliases": [
        "0.81x"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 48.0,
        "denominator": 59.6
      }
    },
    {
      "id": "autodefault-decode-baseline",
      "value_text": "67.8",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "decode, baseline no-flags, n=9, round-robin interleaved"
    },
    {
      "id": "autodefault-decode-patched",
      "value_text": "145.9",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "decode, autodefault no-flags, n=9"
    },
    {
      "id": "autodefault-decode-handtuned",
      "value_text": "146.0",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "decode, baseline -t 2, n=9"
    },
    {
      "id": "autodefault-decode-ratio",
      "value_text": "2.15",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "145.9 / 67.8",
      "aliases": [
        "2.15x",
        "2.15×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 145.9,
        "denominator": 67.8
      }
    },
    {
      "id": "autodefault-prefill-baseline",
      "value_text": "1835.2",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "prefill, baseline no-flags, n=9"
    },
    {
      "id": "autodefault-prefill-patched",
      "value_text": "1779.8",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "prefill, autodefault no-flags, n=9 -- unchanged within noise"
    },
    {
      "id": "autodefault-prefill-patched-pct",
      "value_text": "3.0",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "prefill: 1 - 1779.8/1835.2 (a drop, reported as -3.0%)",
      "aliases": [
        "-3.0",
        "-3"
      ],
      "compute": {
        "op": "percent_drop",
        "numerator": 1835.2,
        "denominator": 1779.8
      }
    },
    {
      "id": "autodefault-prefill-naive-t2",
      "value_text": "975.6",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "prefill, baseline -t 2 (naive workaround), n=9 -- the 47% collapse"
    },
    {
      "id": "autodefault-prefill-collapse-pct",
      "value_text": "47",
      "source_file": "results/AUTODEFAULTS.md",
      "note": "prefill collapse: 1 - 975.6/1835.2",
      "compute": {
        "op": "percent_drop",
        "numerator": 1835.2,
        "denominator": 975.6
      }
    },
    {
      "id": "kernel-accel-ratio-n512",
      "value_text": "3.2",
      "source_file": "results/bench/kernel-bench-apple-m4-max.md",
      "note": "N=512 Accelerate/SME2, canonical run",
      "aliases": [
        "3.2x",
        "~3.2x"
      ]
    },
    {
      "id": "kernel-accel-ratio-n1024",
      "value_text": "6.7",
      "source_file": "results/bench/kernel-bench-apple-m4-max.md",
      "note": "N=1024 Accelerate/SME2",
      "aliases": [
        "6.7x"
      ]
    },
    {
      "id": "kernel-accel-ratio-n2048",
      "value_text": "18.4",
      "source_file": "results/bench/kernel-bench-apple-m4-max.md",
      "note": "N=2048 Accelerate/SME2",
      "aliases": [
        "18.4x"
      ]
    },
    {
      "id": "kernel-accel-ratio-n512-rerunB",
      "value_text": "5.1",
      "source_file": "results/bench/kernel-bench-apple-m4-max.md",
      "note": "N=512 Accelerate/SME2, rerun B (noise spread)",
      "aliases": [
        "5.1x",
        "~5.1x"
      ]
    },
    {
      "id": "generalization-1_5b-decode-t4",
      "value_text": "122.1",
      "source_file": "results/GENERALIZATION.md",
      "note": "1.5B/Q4_0 decode, threads=4 -- the true per-model optimum, not the SME-cap default"
    },
    {
      "id": "generalization-1_5b-decode-t2",
      "value_text": "103.9",
      "source_file": "results/GENERALIZATION.md",
      "note": "1.5B/Q4_0 decode, threads=2 -- the SME thread cap the 0002 patch auto-selects"
    },
    {
      "id": "generalization-1_5b-thread-cap-miss-pct",
      "value_text": "17.5",
      "source_file": "results/GENERALIZATION.md",
      "note": "1.5B/Q4_0: threads=4 (122.1) beats the SME-cap default threads=2 (103.9) by this much",
      "aliases": [
        "~17.5"
      ],
      "compute": {
        "op": "percent_rise",
        "numerator": 103.9,
        "denominator": 122.1
      }
    },
    {
      "id": "server-tps-parallel1",
      "value_text": "14.9",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[0].agg_tps_median",
      "note": "aggregate tok/s, llama-server -cb, parallel=1/clients=1/threads=20"
    },
    {
      "id": "server-tps-parallel4",
      "value_text": "56.6",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[1].agg_tps_median",
      "note": "aggregate tok/s, parallel=4/clients=4/threads=20"
    },
    {
      "id": "server-tps-parallel8-t20",
      "value_text": "271.8",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[2].agg_tps_median",
      "note": "aggregate tok/s, parallel=8/clients=8/threads=20"
    },
    {
      "id": "server-tps-parallel8-t4",
      "value_text": "264.8",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[3].agg_tps_median",
      "note": "aggregate tok/s, parallel=8/clients=8/threads=4 -- dropping threads 20->4 costs almost nothing"
    },
    {
      "id": "server-tps-parallel16",
      "value_text": "440.4",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[4].agg_tps_median",
      "note": "aggregate tok/s, parallel=16/clients=16/threads=20"
    },
    {
      "id": "server-tps-scaling-ratio",
      "value_text": "29.6",
      "source_file": "results/server/server-bench.json",
      "note": "aggregate scaling from 1 to 16 concurrent clients: 440.4/14.9",
      "aliases": [
        "29.6x",
        "29.6×",
        "~29.6",
        "~29.6x"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 440.4,
        "denominator": 14.9
      }
    },
    {
      "id": "server-ttft-p50-parallel1",
      "value_text": "0.089",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[0].ttft_p50",
      "note": "TTFT p50 (s), parallel=1"
    },
    {
      "id": "server-ttft-p99-parallel1",
      "value_text": "0.089",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[0].ttft_p99",
      "note": "TTFT p99 (s), parallel=1 -- identical to p50 at this concurrency"
    },
    {
      "id": "server-ttft-p50-parallel4",
      "value_text": "0.092",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[1].ttft_p50",
      "note": "TTFT p50 (s), parallel=4"
    },
    {
      "id": "server-ttft-p99-parallel4",
      "value_text": "0.221",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[1].ttft_p99",
      "note": "TTFT p99 (s), parallel=4 -- the sweep's actual peak (221ms), not the 16-client row"
    },
    {
      "id": "server-ttft-p50-parallel8-t20",
      "value_text": "0.062",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[2].ttft_p50",
      "note": "TTFT p50 (s), parallel=8/threads=20"
    },
    {
      "id": "server-ttft-p99-parallel8-t20",
      "value_text": "0.117",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[2].ttft_p99",
      "note": "TTFT p99 (s), parallel=8/threads=20"
    },
    {
      "id": "server-ttft-p50-parallel8-t4",
      "value_text": "0.062",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[3].ttft_p50",
      "note": "TTFT p50 (s), parallel=8/threads=4 -- same p50 as threads=20 at this concurrency"
    },
    {
      "id": "server-ttft-p99-parallel8-t4",
      "value_text": "0.094",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[3].ttft_p99",
      "note": "TTFT p99 (s), parallel=8/threads=4 -- improves vs threads=20's 0.117 despite fewer threads"
    },
    {
      "id": "server-ttft-p50-parallel16",
      "value_text": "0.12",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[4].ttft_p50",
      "note": "TTFT p50 (s), parallel=16"
    },
    {
      "id": "server-ttft-p99-parallel16",
      "value_text": "0.168",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[4].ttft_p99",
      "note": "TTFT p99 (s), parallel=16"
    },
    {
      "id": "server-rss-parallel1",
      "value_text": "724",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[0].peak_rss_mib",
      "note": "peak RSS (MiB), parallel=1"
    },
    {
      "id": "server-rss-parallel4",
      "value_text": "761",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[1].peak_rss_mib",
      "note": "peak RSS (MiB), parallel=4"
    },
    {
      "id": "server-rss-parallel8",
      "value_text": "809",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[2].peak_rss_mib",
      "note": "peak RSS (MiB), parallel=8, threads=20 -- identical at threads=4 ([3].peak_rss_mib is also 809)"
    },
    {
      "id": "server-rss-parallel16",
      "value_text": "901",
      "source_file": "results/server/server-bench.json",
      "source_json_path": "[4].peak_rss_mib",
      "note": "peak RSS (MiB), parallel=16"
    },
    {
      "id": "server-dispatch-i8mm",
      "value_text": "364444",
      "source_file": "results/server/server-dispatch.json",
      "source_json_path": "i8mm",
      "note": "kai_run_matmul I8MM breakpoint hits, gdb attached to llama-server, 8 concurrent clients, continuous batching",
      "aliases": [
        "364,444"
      ]
    },
    {
      "id": "server-dispatch-dotprod",
      "value_text": "11360",
      "source_file": "results/server/server-dispatch.json",
      "source_json_path": "dotprod",
      "note": "kai_run_matmul DOTPROD breakpoint hits, same run as server-dispatch-i8mm",
      "aliases": [
        "11,360"
      ]
    },
    {
      "id": "server-kai-run-matmul-symbols-broken",
      "value_text": "0",
      "source_file": "results/server/spark-provenance.txt",
      "note": "kai_run_matmul symbol count, default build (-DGGML_CPU_KLEIDIAI=ON alone -- the GGML_NATIVE feature probe is silently rejected by gcc 13.3 on this box)"
    },
    {
      "id": "server-kai-run-matmul-symbols-fixed",
      "value_text": "10",
      "source_file": "results/server/spark-provenance.txt",
      "note": "kai_run_matmul symbol count, fixed build (+ -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+sve2+i8mm+bf16+dotprod)"
    },
    {
      "id": "server-sve-cnt",
      "value_text": "16",
      "source_file": "results/server/spark-provenance.txt",
      "note": "SVE_CNT reported by the fixed build's system_info banner -- 128-bit SVE, below the 256-bit gate kleidiai.cpp:209 checks, so I8MM is selected over SVE"
    },
    {
      "id": "scale-7b-t1-prefill",
      "value_text": "28.89",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[0].prefill_median",
      "note": "7B, threads=1, prefill median"
    },
    {
      "id": "scale-7b-t1-decode",
      "value_text": "7.26",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[0].decode_median",
      "note": "7B, threads=1, decode median"
    },
    {
      "id": "scale-7b-t2-prefill",
      "value_text": "55.41",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[1].prefill_median",
      "note": "7B, threads=2, prefill median"
    },
    {
      "id": "scale-7b-t2-decode",
      "value_text": "12.95",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[1].decode_median",
      "note": "7B, threads=2, decode median"
    },
    {
      "id": "scale-7b-t4-prefill",
      "value_text": "104.42",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[2].prefill_median",
      "note": "7B, threads=4, prefill median"
    },
    {
      "id": "scale-7b-t4-decode",
      "value_text": "17.76",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[2].decode_median",
      "note": "7B, threads=4, decode median"
    },
    {
      "id": "scale-7b-t8-prefill",
      "value_text": "179.45",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[3].prefill_median",
      "note": "7B, threads=8, prefill median"
    },
    {
      "id": "scale-7b-t8-decode",
      "value_text": "24.45",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[3].decode_median",
      "note": "7B, threads=8, decode median -- decode peak for this model"
    },
    {
      "id": "scale-7b-t16-prefill",
      "value_text": "212.41",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[4].prefill_median",
      "note": "7B, threads=16, prefill median"
    },
    {
      "id": "scale-7b-t16-decode",
      "value_text": "21.01",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[4].decode_median",
      "note": "7B, threads=16, decode median"
    },
    {
      "id": "scale-7b-t20-prefill",
      "value_text": "218.39",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[5].prefill_median",
      "note": "7B, threads=20, prefill median -- prefill peak for this model, and the box's default thread count"
    },
    {
      "id": "scale-7b-t20-decode",
      "value_text": "18.03",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_7b_thread_sweep[5].decode_median",
      "note": "7B, threads=20, decode median"
    },
    {
      "id": "scale-05b-t1-prefill",
      "value_text": "318.95",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[0].prefill_median",
      "note": "0.5B, threads=1, prefill median"
    },
    {
      "id": "scale-05b-t1-decode",
      "value_text": "85.41",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[0].decode_median",
      "note": "0.5B, threads=1, decode median"
    },
    {
      "id": "scale-05b-t2-prefill",
      "value_text": "556.71",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[1].prefill_median",
      "note": "0.5B, threads=2, prefill median"
    },
    {
      "id": "scale-05b-t2-decode",
      "value_text": "138.60",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[1].decode_median",
      "note": "0.5B, threads=2, decode median"
    },
    {
      "id": "scale-05b-t4-prefill",
      "value_text": "915.65",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[2].prefill_median",
      "note": "0.5B, threads=4, prefill median"
    },
    {
      "id": "scale-05b-t4-decode",
      "value_text": "177.61",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[2].decode_median",
      "note": "0.5B, threads=4, decode median"
    },
    {
      "id": "scale-05b-t8-prefill",
      "value_text": "1457.68",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[3].prefill_median",
      "note": "0.5B, threads=8, prefill median -- both phases peak for this model",
      "aliases": [
        "1,457.68"
      ]
    },
    {
      "id": "scale-05b-t8-decode",
      "value_text": "190.32",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[3].decode_median",
      "note": "0.5B, threads=8, decode median -- both phases peak for this model"
    },
    {
      "id": "scale-05b-t16-prefill",
      "value_text": "1047.66",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[4].prefill_median",
      "note": "0.5B, threads=16, prefill median",
      "aliases": [
        "1,047.66"
      ]
    },
    {
      "id": "scale-05b-t16-decode",
      "value_text": "120.05",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[4].decode_median",
      "note": "0.5B, threads=16, decode median"
    },
    {
      "id": "scale-05b-t20-prefill",
      "value_text": "965.98",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[5].prefill_median",
      "note": "0.5B, threads=20, prefill median -- the box's default thread count"
    },
    {
      "id": "scale-05b-t20-decode",
      "value_text": "45.49",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "A_05b_thread_sweep[5].decode_median",
      "note": "0.5B, threads=20, decode median"
    },
    {
      "id": "scale-finding3-05b-broken-prefill",
      "value_text": "657.00",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[0].prefill_median",
      "note": "0.5B, broken default KleidiAI build, default threads, prefill median"
    },
    {
      "id": "scale-finding3-05b-broken-decode",
      "value_text": "42.18",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[0].decode_median",
      "note": "0.5B, broken default KleidiAI build, default threads, decode median"
    },
    {
      "id": "scale-finding3-05b-fixed-prefill",
      "value_text": "933.63",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[1].prefill_median",
      "note": "0.5B, fixed KleidiAI build (GGML_NATIVE=OFF + explicit arch), default threads, prefill median"
    },
    {
      "id": "scale-finding3-05b-fixed-decode",
      "value_text": "41.70",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[1].decode_median",
      "note": "0.5B, fixed KleidiAI build, default threads, decode median -- effectively unchanged vs the broken build's 42.18"
    },
    {
      "id": "scale-finding3-7b-broken-prefill",
      "value_text": "48.64",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[2].prefill_median",
      "note": "7B, broken default KleidiAI build, default threads, prefill median"
    },
    {
      "id": "scale-finding3-7b-broken-decode",
      "value_text": "11.17",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[2].decode_median",
      "note": "7B, broken default KleidiAI build, default threads, decode median"
    },
    {
      "id": "scale-finding3-7b-fixed-prefill",
      "value_text": "222.14",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[3].prefill_median",
      "note": "7B, fixed KleidiAI build, default threads, prefill median"
    },
    {
      "id": "scale-finding3-7b-fixed-decode",
      "value_text": "18.45",
      "source_file": "results/scale/scale-experiment.json",
      "source_json_path": "B_finding3_cost[3].decode_median",
      "note": "7B, fixed KleidiAI build, default threads, decode median"
    },
    {
      "id": "scale-decode-collapse-05b",
      "value_text": "4.56",
      "source_file": "results/scale/scale-experiment.json",
      "note": "0.5B decode thread-tuning win: peak (190.32, t=8) / default-threads fixed-build figure (41.70)",
      "aliases": [
        "4.56x",
        "4.56×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 190.32,
        "denominator": 41.70
      }
    },
    {
      "id": "scale-decode-collapse-7b",
      "value_text": "1.33",
      "source_file": "results/scale/scale-experiment.json",
      "note": "7B decode thread-tuning win: peak (24.45, t=8) / default-threads fixed-build figure (18.45) -- the same class of win as scale-decode-collapse-05b, collapsed by model size",
      "aliases": [
        "1.33x",
        "1.33×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 24.45,
        "denominator": 18.45
      }
    },
    {
      "id": "scale-finding3-prefill-05b-ratio",
      "value_text": "1.42",
      "source_file": "results/scale/scale-experiment.json",
      "note": "0.5B Finding 3 prefill cost: fixed (933.63) / broken (657.00)",
      "aliases": [
        "1.42x",
        "1.42×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 933.63,
        "denominator": 657.00
      }
    },
    {
      "id": "scale-finding3-decode-05b-ratio",
      "value_text": "0.99",
      "source_file": "results/scale/scale-experiment.json",
      "note": "0.5B Finding 3 decode cost: fixed (41.70) / broken (42.18) -- no effect at this model size",
      "aliases": [
        "0.99x",
        "0.99×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 41.70,
        "denominator": 42.18
      }
    },
    {
      "id": "scale-finding3-prefill-7b-ratio",
      "value_text": "4.57",
      "source_file": "results/scale/scale-experiment.json",
      "note": "7B Finding 3 prefill cost: fixed (222.14) / broken (48.64) -- the largest speedup this project has measured from fixing a single build defect",
      "aliases": [
        "4.57x",
        "4.57×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 222.14,
        "denominator": 48.64
      }
    },
    {
      "id": "scale-finding3-decode-7b-ratio",
      "value_text": "1.65",
      "source_file": "results/scale/scale-experiment.json",
      "note": "7B Finding 3 decode cost: fixed (18.45) / broken (11.17)",
      "aliases": [
        "1.65x",
        "1.65×"
      ],
      "compute": {
        "op": "ratio",
        "numerator": 18.45,
        "denominator": 11.17
      }
    }
  ]
}
```
<!-- CLAIMS-REGISTRY:END -->

### DGX Spark — KleidiAI matmul symbol counts (`results/server/kai-symbols.txt`)

| claim | value | source |
|---|---|---|
| default build, total `kai_` symbols | 36 | `results/server/kai-symbols.txt` |
| default build, `kai_run_matmul` symbols | 0 | `results/server/kai-symbols.txt` |
| fixed build, total `kai_` symbols | 149 | `results/server/kai-symbols.txt` |
| fixed build, `kai_run_matmul` symbols | 10 | `results/server/kai-symbols.txt` |
| fixed build, dotprod family | 6 | `results/server/kai-symbols.txt` |
| fixed build, i8mm family | 2 | `results/server/kai-symbols.txt` |
| fixed build, sve family | 2 | `results/server/kai-symbols.txt` |

**Counting method matters.** A symbol such as
`kai_run_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm` contains *both* `neon` and
`i8mm`. Counting substring occurrences therefore double-counts and produces family totals that
exceed the symbol count — an earlier draft of this lane reported "dotprod 7 / i8mm 3 / neon 8 /
sve 2", which sums to 20 against 10 symbols and was rejected for exactly that reason. The table
above assigns each symbol to exactly one family, most-specific token first.

**The load-bearing consequence:** the fixed build compiles in **2 SVE matmul kernels**, and the
concurrent-load dispatch trace (`results/server/server-dispatch.json`) records **zero** SVE calls.
The kernels exist, the silicon has SVE2, and the dispatcher still never selects them — Finding 2,
observed end to end on Cortex-X925.

### 7B broken-vs-fixed wall-clock race (`results/scale/race-7b-broken-vs-fixed.json`)

| claim | value | source |
|---|---|---|
| broken build, wall clock (median of 5) | 16.181 s | `results/scale/race-7b-broken-vs-fixed.json` |
| fixed build, wall clock (median of 5) | 12.108 s | `results/scale/race-7b-broken-vs-fixed.json` |
| wall-clock ratio | 1.34x | derived |

**Why this is 1.34x and not 4.57x.** End-to-end wall clock at 150 tokens includes loading a 4.4 GB
model, which the build fix does not affect. The 4.57x lives in *prefill throughput* specifically
(`results/scale/scale-experiment.json`). The two figures measure different things and must not be
substituted for one another.

**A discarded first attempt is worth recording.** An initial n=3 capture with no cache warmup gave
walls of `[21.54, 23.42, 15.96]` (broken) and `[18.87, 12.21, 11.99]` (fixed) — a median ratio of
1.76x, but a range spanning 1.24x-1.80x depending on which reps were compared. The spread was cold
page cache on a 4.4 GB file, not architecture. Re-running with a discarded warmup per binary and
n=5 tightened it to `[16.18, 16.86, 18.01, 16.00, 16.16]` vs `[12.16, 12.15, 12.11, 11.79, 11.96]`
and gave the honest 1.34x. **The 1.76x figure was never published.**

### PMU cross-check attempt and L3 probe cost, 2026-08-05 (`results/pmu/pmu-crosscheck.json`)

Arm test box: Cortex-X925 + Cortex-A725, 20 cores, Armv9.2, bare metal, kernel 6.17.0-1021-nvidia,
gcc 13.3.0, gdb 15.1, perf 6.17.13.

| claim | value | source |
|---|---|---|
| `perf_event_paranoid` | 4 | `pmu_availability.perf_event_paranoid` |
| PMU events enumerated by `armv8_pmuv3_0` | 78 | `pmu_availability...kernel_exposed_event_list` |
| of those, SVE/SME/ASE instruction-class events | 0 | `sve_sme_ase_specific_events_present: false` |
| plain run, median wall clock (n=5, interleaved) | 1.2056 s | `overhead_measurement.plain_median_sec` |
| under L3 gdb probe, median wall clock (n=5, interleaved) | 4.379 s | `overhead_measurement.l3_gdb_median_sec` |
| L3 median overhead | 3.63x | `overhead_measurement.overhead_multiplier_median` |
| L3 dispatch hits, every one of 5 probed runs | 15,936 | `overhead_measurement.note` |
| SVE-family L3 hits across all 10 swept configs | 0 | `comparison.fixed_build_ten_symbol.l3_full_sweep` |

**What this does NOT establish.** The PMU-vs-L3 numeric cross-validation was **not performed** —
it could not be, for the two independent reasons above. No claim of agreement or disagreement
between the two methods is made anywhere in this repo. The `3.63x` overhead figure is from a
single configuration (threads=4, decode_short, 15,936 hits) and has not been shown to generalise.
The PMU unavailability is **one machine**, and is not evidence about Arm PMU availability in
general.
