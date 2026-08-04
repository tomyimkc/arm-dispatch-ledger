<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors -->

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
    }
  ]
}
```
<!-- CLAIMS-REGISTRY:END -->
