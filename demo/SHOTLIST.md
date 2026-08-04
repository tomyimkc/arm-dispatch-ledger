<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Polygraph contributors -->

# Demo shot list — Polygraph (target: under 3:00 total)

**Rebuilt 2026-08-04 around `results/REMEASURE-2026-08-04-QUIET.md`**, the authoritative,
round-robin-interleaved, 7-reps-per-config re-measurement of this project's throughput claims.
An earlier cut of this demo told only the pre-patch diagnosis story and quoted throughput
numbers from `results/crossover/`, which was measured on a heavily contended machine with
baseline and patched configs run in *different, non-interleaved* time windows — the exact
setup that manufactures a fake speedup. **Those numbers (57.3%, 71.6, 45.5, 4.4x, 198.9,
2257.5, 1145.0) are retracted.** If you see any of them in an older recording or draft, it is
stale — do not re-use it.

**The corrected arc, in one line:** tuning (`-t 2` for decode, `-t 8` for prefill) is a real,
reproducible, zero-code-change win — 3.43x and 1.79x. This project also wrote a dispatch patch
to *fix* the underlying gate; that patch is a genuine, symbol-level dispatch change that is
**not** a throughput win — it measures ~12% **slower** at default thread count. We are not
hiding that. We measured it, we are publishing the negative result, and we filed both findings
upstream: [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547).

**How this file was produced:** every beat below is a direct transcript of what
`demo/demo.sh` actually prints and runs, timed by running it for real (`NO_COLOR=1
./demo/demo.sh`, default `PAUSE=2.5`, **49s** measured total wall time on the reference
machine — see `demo/demo.sh`'s own final line, "Total demo.sh wall time"). It is not an
aspirational plan — narration lines are copy-pasted from the script's own `narrate`/`fact`
calls, and the beat boundaries below are chosen to land close to the target arc: hook
(0:00–0:20), proof (0:20–0:50), cost (0:50–1:40), root cause (1:40–2:20), honest ending
(2:20–3:00).

**The rule, unchanged from the old shot list:** every number and every terminal output shown
on screen must be the real output of the command shown, or a number already committed to
`results/` and clearly cited — no invented numbers, no unlabeled speed-ups, and **no CI green
check that has not actually run**. This demo does not show any CI run; if a future edit adds
one, apply the same rule the old shot list used: don't claim a result that doesn't exist.

No copyrighted music. If you want a bed track, use only royalty-free/CC0 audio you have the
rights to (e.g. YouTube Audio Library tracks marked safe for use), kept low under the
voice-over. Silence is also a completely acceptable choice.

---

## Before recording

1. `cd polygraph` and confirm `demo/demo.sh` finds your build:
   ```bash
   LLAMA_CLI=... LLAMA_BENCH=... MODEL_PATH=... ./demo/demo.sh   # or rely on the built-in
                                                                  # /tmp candidates -- see demo/README.md
   ```
2. Optionally build the patched binary too (Beat 5 degrades gracefully to citation-only
   without it, but the live warning/enabled-line moment is worth having) — see
   `demo/README.md`, "Optional: building the patched binary".
3. Do one full silent dry run first (`PAUSE=0.5 ./demo/demo.sh`) so the model is warm in the
   OS page cache and you've seen the real output once. **Do not record the first-ever run** —
   a cold model load adds a few unpredictable seconds to Beat 1 that a warm run doesn't have.
4. Terminal: 1080p or higher, **18–20pt monospace font**, high-contrast theme (light-on-dark
   or dark-on-light, just make sure red/green/yellow/magenta stay legible — the script uses
   all four). Single full-width pane; no split screen needed, `demo.sh` already sequences
   everything.
5. Default `PAUSE=2.5` is tuned to give you room to read each narration line aloud before the
   next command fires. If you talk faster or slower, adjust: `PAUSE=3.5 ./demo/demo.sh` for a
   slower read, `PAUSE=1.5` for a tighter cut.
6. `RUN_LIVE_L3` stays `0` (default) for the recorded take — see Beat 2 below for why, and use
   it only for a rehearsal/authenticity check, not the take (it adds ~90s per sweep, up to
   three sweeps if triggered in both Beat 2 and Beat 5).

---

## Beat-by-beat (as printed by `demo/demo.sh`, top to bottom)

### Title (0:00–0:07, ~7s)

**Screen:** ASCII banner reading "Polygraph" + the one-line framing + the resolved config
(`LLAMA_CLI=...`, `LLAMA_BENCH=...`, `MODEL_PATH=...`, `LLAMA_CLI_PATCHED=...`,
`DEFAULT_THREADS=...`).

**Say:** *"Polygraph — a tool that checks whether software is telling the truth. Does the SME2
kernel llama.cpp advertises actually run? Let's check. Apache-2.0, built for the Arm Create: AI
Optimization Challenge."*

---

### Beat 1 / 5 — The hook (0:07–0:20, ~13s, LIVE)

**Screen:** `BEAT 1 / 5 -- The hook: the banner says SME2 is running`, then live
`llama-cli ... --verbose`, filtered to 5 lines: `system_info: ... SME = 1 | SME2 = 1 | ... |
KLEIDIAI = 1`, `kleidiai: primary q4 kernel feature SME2` (×3 for q4/q8/f32), `kleidiai: SME2
enabled (runtime-detected SME cores=2)`.

**Say:** *"llama.cpp's own startup log, at verbose, claims the accelerated SME2 kernel is
selected for every matmul family. Watch the log, live."* … *"Claim: SME equals 1, SME2 equals
1, KLEIDIAI equals 1 — 'primary q4 kernel feature SME2.' Selected for every op, according to
the log. Let's check what actually runs."*

---

### Beat 2 / 5 — The proof (0:20–0:50, ~30s, mostly LIVE)

**Screen, part (a), cited (~5s):** the committed ground-truth row, highlighted in red:
`threads=8 workload=decode_short advertised=SME2 executed=dotprod hits: SME2=0 /
NEON-dotprod=31871 verdict: SILENT_FALLBACK`, plus the exact reproduction command printed
above it.

**Say:** *"A timing-only benchmark cannot see past that log line. We put a real lldb
breakpoint on the SME2 kernel's own entry points and count hits."*

**Why this half is cited, not live, by default:** the real lldb sweep at 8 threads is ~90 real
seconds of stop/continue round trips (see `results/GROUND-TRUTH-DISPATCH.md`'s own timing
note) — recording that live would blow the budget on its own. `demo.sh` prints the exact
command to reproduce it yourself; set `RUN_LIVE_L3=1` to include it for a longer cut.

**Screen, part (b), LIVE (~25–30s measured — this run: 24.9s):** the real
`tools/verify_dispatch.py --threads 2` sweep, scrolling in real time, landing on `VERDICT:
SME2_DISPATCHED`, `hits (adv/other) 5826/0`.

**Say (while it runs):** *"Why 2? kleidiai.cpp hardcodes a thread cap by chip name — 2, on
this M4 Max. At or below the cap, SME2 genuinely dispatches. Watch it happen, live."* …
*(let it finish — the real breakpoint hits scrolling is the point, this is the one beat in
the video that is unmistakably not staged)* … *"The banner said SME2. The debugger says zero —
until one flag changes it."*

**This is the load-bearing "not staged" proof of the whole video** — if you need to cut time
elsewhere, do not cut this beat.

---

### Beat 3 / 5 — The cost (0:50–1:40, ~50s, LIVE + cited)

**Screen:** four real, fast `llama-bench` invocations, live (`-r 3`, ~1s each): decode default
(12 threads, no flags), decode `-t 2`, prefill default, prefill `-t 8` — each prints its own
real ASCII table with `t/s`. Then the cited, authoritative summary table:

```
phase    config                      median tok/s
decode   default (no flags, 12 thr)     93.6
decode   -t 2                        321.0  <- 3.43x
prefill  default (no flags)             1230.3
prefill  -t 8                        2198.1  <- 1.79x
```

**Say:** *"Re-measured on a quiet machine, round-robin, interleaved, so contention hits every
config equally — seven reps per config, median plus standard deviation, never a bare mean."*
… (over the four live bench runs) *"Quick live sanity check first — real commands, real output,
right now."* … *"Three point four three x decode, one point seven nine x prefill. Zero code
changes. Both flags already ship in llama.cpp today — and the banner never tells you to use
them."*

**Note the honesty beat inside this beat:** the script explicitly narrates that the four quick
live runs (3 reps, not interleaved against their counterpart) are illustrative, not the
authoritative number — the cited table from `results/REMEASURE-2026-08-04-QUIET.md` is what
this project stands behind. Keep that framing in the voice-over; don't let the live numbers
be mistaken for the rigorous ones.

---

### Beat 4 / 5 — The root cause (1:40–2:20, ~40s, LIVE)

**Screen:** two live `sed -n` reads of the actual `kleidiai.cpp` source (llama.cpp @ `dbadb68`):
lines 147–169 (the hardcoded per-chip `ModelSMCU` thread-cap table: `{"M4 Ultra",2}, {"M4
Max",2}, {"M4 Pro",2}, {"M4",1}`), then lines 1094–1113 (the dispatch decision:
`too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128)`, collapsing to the non-SME
slot above the cap).

**Say:** *"Why does SME2 disappear above 2 threads for decode? Two real source lines, llama.cpp
at dbadb68, in kleidiai dot cpp."* … *(over the first block)* *"A hardcoded, brand-string-keyed
thread cap for known M4 variants."* … *(over the second block)* *"That cap becomes the dispatch
decision here."* … *"ne11 equals 1 for every decode step — one token at a time — so
too-small-for-hybrid is always true above the cap. Decode falls all the way back to NEON, on
every thread count above 2, on every prompt. Prefill doesn't have this problem, which is
exactly why the fix differs by phase."*

---

### Beat 5 / 5 — The honest ending (2:20–3:00, ~40s, LIVE + cited)

**Screen, live (a):** patched `llama-cli`, flag **unset**, default threads — the real one-shot
warning line: `kleidiai: SME not used for a GEMV op (e.g. token generation) because the thread
count (8) exceeds sme_thread_cap (2); use -t <= 2 for generation, or set
GGML_KLEIDIAI_PHASE_AWARE=1 (experimental) to run SME capped at sme_thread_cap alongside NEON
on the remaining threads`.

**Screen, live (b):** same binary, flag **set** — `kleidiai: phase-aware dispatch enabled
(GGML_KLEIDIAI_PHASE_AWARE=1, experimental): ...`.

**Screen, live (c):** a short `python3` one-liner that reads the two committed dispatch-ledger
JSON files (flag off / flag on) straight off disk and prints the real symbol-level hit counts:
`decode threads=4: flag off sme2=0 (SILENT_FALLBACK) -> flag on sme2=3072
(SME2_HYBRID_DISPATCH)`, and the same for `threads=8` (`0 -> 2354`).

**Screen, cited:** the throughput verdict from `results/REMEASURE-2026-08-04-QUIET.md`:
`decode, default threads: 93.6 -> 82.5 tok/s = 0.88x, ~12% SLOWER`; `decode, -t 2 (patch inert
here): 321.0 -> 317.5 tok/s = 0.99x, tie`; `prefill, default threads: 1230.3 -> 1202.1 tok/s =
0.98x, tie`.

**Say:** *"So we patched it — an opt-in flag that lets decode into the existing hybrid SME plus
NEON split instead of collapsing to NEON-only."* … *(over the two live llama-cli checks)*
*"Two live checks against the patched binary: flag unset gives you a plain warning — that half
costs nothing on its own. Flag set switches the experimental dispatch change on."* … *(over the
Python read)* *"Symbol-level proof the dispatch change is real: decode at four threads goes
from zero SME2 kernel calls to three thousand and seventy-two, in the identical binary."* …
*"But it is not a throughput win — it measures about twelve percent slower at default thread
count. So we are not proposing it as a fix. We measured it, we are publishing the negative
result, and we filed both findings upstream — issue 26547. Most demos can't afford to be this
honest. We can, because the tuning win next to it is real."*

**This is the differentiator beat** — keep it in even if you have to trim elsewhere. It is the
one part of this video most hackathon submissions structurally cannot include.

---

### Close (2:58–3:00, ~2s buffer, text only)

**Screen:** final banner — *"Polygraph"*.

**Say:** *"The banner said SME2. The debugger proved otherwise. Tuning is a real, free, three
point four three x, one point seven nine x win. Our own fix attempt was not — and we're saying
so. Apache-2.0. Filed upstream: issue 26547."*

---

## Total runtime budget

| Beat | Duration | Cumulative | Live? |
|---|---:|---:|---|
| Title | 0:07 | 0:07 | text only |
| 1 — The hook | 0:13 | 0:20 | live (~1–2s cmd) |
| 2 — The proof | 0:30 | 0:50 | cited (~5s) + **live (~25–30s cmd)** |
| 3 — The cost | 0:50 | 1:40 | live (4× ~1s cmd) + cited |
| 4 — The root cause | 0:40 | 2:20 | live (2× instant `sed`) |
| 5 — The honest ending | 0:38 | 2:58 | live (2× ~1s cmd + instant parse) + cited |
| Close | 0:02 | 3:00 | text only |

Measured end-to-end `demo.sh` wall time at the default `PAUSE=2.5` was **49s** on the reference
machine (see `demo/demo.sh`'s own final line, "Total demo.sh wall time") — well under this
table's 3:00, because this table budgets narration/talking time *on top of* the script's own
pacing, the same convention the previous version of this file used. If a take runs long in
editing, cut from Beat 3 (drop one of the four live `llama-bench` calls, keep the cited table)
first, then Beat 4 (show only the second `sed` block, narrate the first from memory). **Do not
cut Beat 2's live `-t 2` sweep** (the load-bearing "not staged" proof) **or Beat 5** (the
differentiator) — those two are what separates this submission from a pure diagnosis, and from
every other submission that can't admit a negative result.
