<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors -->

# Demo shot list — Arm Dispatch Ledger (target: under 3:00 total)

**This file replaces `docs/VIDEO.md`.** That file's shot list was written before
`demo/demo.sh` existed and assumes a different flag set and beat structure; once
this file lands, `docs/VIDEO.md` should be deleted or redirected here by whoever
owns `docs/` — this package (`demo/`) does not edit files outside `demo/`.

**How this file was produced:** every beat below is a direct transcript of what
`demo/demo.sh` actually prints and runs, timed by running it for real. It is not
an aspirational plan — narration lines are copy-pasted from the script's own
`narrate`/`fact` calls, and durations are measured, not estimated.

**The rule, unchanged from the old shot list:** every number and every terminal
output shown on screen must be the real output of the command shown — no
invented numbers, no unlabeled speed-ups, and **no CI green check that has not
actually run**. If `verify-free-arm64.yml` has not been triggered on real GitHub
Actions infrastructure by recording time, do not show or claim a passing run —
narrate the "fork and run it yourself, free" framing instead (this is true
regardless of whether it has been triggered yet).

No copyrighted music. If you want a bed track, use only royalty-free/CC0 audio
you have the rights to (e.g. YouTube Audio Library tracks marked safe for use),
kept low under the voice-over. Silence is also a completely acceptable choice.

---

## Before recording

1. `cd arm-dispatch-ledger` and confirm `demo/demo.sh` finds your build:
   ```bash
   LLAMA_CLI=... MODEL_PATH=... ./demo/demo.sh   # or rely on the built-in
                                                  # /tmp candidates -- see demo/README.md
   ```
2. Do one full silent dry run first (`PAUSE=0.5 ./demo/demo.sh`) so the model is
   warm in the OS page cache and you've seen the real output once. **Do not
   record the first-ever run** — a cold model load adds a few unpredictable
   seconds to Beat 1 that a warm run doesn't have.
3. Terminal: 1080p or higher, **18–20pt monospace font**, high-contrast theme
   (light-on-dark or dark-on-light, just make sure red/green/yellow stay
   legible — the script uses all three). Single full-width pane; no split
   screen needed, `demo.sh` already sequences everything.
4. Default `PAUSE=2.5` is tuned to give you room to read each narration line
   aloud before the next command fires. If you talk faster or slower, adjust:
   `PAUSE=3.5 ./demo/demo.sh` for a slower read, `PAUSE=1.5` for a tighter cut.
5. `RUN_LIVE_L3` stays `0` (default) for the recorded take — see Beat 2 below
   for why, and use it only for a rehearsal/authenticity check, not the take.

---

## Beat-by-beat (as printed by `demo/demo.sh`, top to bottom)

### Beat 0 — Title (0:00–0:07, ~7s)

**Screen:** ASCII banner + repo one-liner + the resolved config
(`LLAMA_CLI=...`, `MODEL_PATH=...`, `PAUSE=...`).

**Say:** *"Arm Dispatch Ledger — does the SME2 kernel llama.cpp advertises
actually run? Apache-2.0, built for the Arm Create: AI Optimization
Challenge."*

---

### Beat 1 — The claim (0:07–0:16, ~9s)

**Screen:** live `llama-cli ... --verbose`, filtered to 5 lines:
`system_info: ... SME = 1 | SME2 = 1 | ... KLEIDIAI = 1`, `kleidiai: primary q4
kernel feature SME2`, `kleidiai: SME2 enabled (...)`.

**Say (over the log lines):** *"llama.cpp's own startup log, at verbose,
claims the accelerated SME2 kernel is selected for every matmul family. Watch
the log, live."* … *"Claim: SME2 = 1, and 'primary q4 kernel feature SME2' —
selected for every op."*

---

### Beat 2 — The lie (0:16–0:26, ~10s)

**Screen:** the cited, committed dispatch-ledger row (not a live run by
default — see note below), highlighted in red:
`threads=8 workload=decode_short advertised=SME2 executed=dotprod hits:
SME2=0 / NEON-dotprod=31871 verdict: SILENT_FALLBACK`, plus the exact
reproduction command printed above it.

**Say:** *"A timing-only benchmark cannot see past that log line. We put a
real lldb breakpoint on the SME2 kernel's own entry points and count hits."*
… *"The banner said SME2. The debugger says zero. Every single time, above
the cap."*

**Why this beat is cited, not live, by default:** the real lldb sweep at 8
threads is ~90 real seconds of stop/continue round trips (measured; see
`results/GROUND-TRUTH-DISPATCH.md`'s own timing note) — recording that live
would blow the 3-minute budget on its own. The number shown is the actual
committed ledger (`results/dispatch-ledger-darwin-arm64.json`,
`results/GROUND-TRUTH-DISPATCH.md`), independently re-confirmed live in the
same session this script was built in (0/31870 vs. the committed 0/31871 —
agrees within normal run-to-run noise). `demo.sh` prints the exact command to
reproduce it yourself. If you want the fully-live version for authenticity in
a longer cut, run `RUN_LIVE_L3=1 ./demo/demo.sh` and accept the extra ~90s.

---

### Beat 3 — Same probe at `-t 2`: thousands of hits (0:26–1:00, ~34s, LIVE)

**Screen:** the real `tools/verify_dispatch.py --threads 2` run, scrolling in
real time (~27–32s measured), landing on
`VERDICT: SME2_DISPATCHED`, `hits (adv/other) 5826/0`.

**Say (while it runs):** *"Why 2? kleidiai.cpp hardcodes a thread cap by chip
name — 2, on this M4 Max. At or below the cap, SME2 genuinely dispatches.
Watch it happen."* … *(let it finish, ~30s of real breakpoint hits scrolling
is the point — this is the one beat in the video that is unmistakably not
staged)* … *"Same binary, same model, one flag changed: -t 8 to -t 2. Zero
hits to thousands."*

**This is the one genuinely slow-but-live beat.** If you need to cut time
elsewhere, do not cut this one — it is the load-bearing "not staged" proof of
the whole video. Cut from Beat 6 or the close instead.

---

### Beat 4 — The measured consequence (1:00–1:15, ~15s)

**Screen:** the printed comparison table (decode SME2@2=327.6 vs. NEON@8=155.3;
prefill_long NEON@8=2676.4 vs. SME2-hybrid@8=1830.1), sourced from
`results/bench/bench-apple-m4-max.md`.

**Say:** *"Five reps per cell, interleaved, warmup-discarded, median plus or
minus standard deviation — never a bare mean."* … *"Decode: SME2 at 2 threads
wins outright, every thread count measured. Prefill: plain NEON at 8 threads
beats SME2's own best cell by one point four six times. Not the flattering
story."*

---

### Beat 5 — The fix, and the gap it can't close (1:15–1:50, ~35s, LIVE)

**Screen:** (a) `llama-cli --help` filtered to the `-tb, --threads-batch`
line — proving the flag exists; (b) a live, fast (~5–10s) `llama-cli -t 2 -tb
8` run with a short prompt; (c) the printed architectural conclusion in red/
yellow: the true combined optimum (327.6 decode + 2676.4 prefill) vs. the
best one process can actually do today (327.6 decode + 1830.1 prefill).

**Say:** *"Decode wants SME2 plus few threads. Prefill wants NEON plus many
threads. llama-cli already has separate flags for exactly this."* … *"So the
thread split is expressible today"* — run the live command — *(the printed
prompt/gen rate on this one dry run is noisy, single-shot, chat-templated;
narrate past it, don't dwell on the number)* … *"But GGML_KLEIDIAI_SME — the
switch between the SME2 and NEON kernel families — is read once per process,
not per op. It cannot vary between -t and -tb within one running process.
One process cannot get SME2-for-decode and NEON-forced-for-prefill at the
same time. That per-phase kernel-family selection is the gap — filed
upstream."*

**This is the beat that makes the submission an optimization finding, not
just a diagnosis** — keep it in even if you have to trim elsewhere.

---

### Beat 6 — Proof the silicon isn't the limiter (1:50–2:00, ~10s, LIVE)

**Screen:** live `kernel_test` (bit-exact/near-bit-exact checks, `ALL CHECKS
PASSED`) then live `kernel_bench` (NEON / SME2-packed / Accelerate GFLOP/s
columns) — both finish in a few seconds combined.

**Say:** *"Hand-written NEON, SME2, and SVE2 kernels, correctness-tested
bit-exact against a scalar reference, benchmarked against the strongest fair
baseline — Apple Accelerate. No strawman comparison."* … *"SME2 kernels are
correct and fast. The bug is in llama.cpp's dispatcher, not the silicon."*

---

### Beat 7 — Reusable, agentic, given back (2:00–2:15, ~15s, LIVE)

**Screen:** live `mcp/server.py --selftest`, filtered to the 4 tool names
(`detect_arm_features`, `verify_dispatch`, `recommend_config`,
`explain_finding`); then the upstream issue line.

**Say:** *"Every tool above is reusable for the next Arm dispatch bug, not
just this one — including an MCP server any agent can query directly."* …
*"And the finding itself was filed upstream, not kept for the submission:
ggml-org/llama.cpp issue 26547 — both findings, exact source lines,
reproduction steps, an offer to patch."*

---

### Close (2:15–2:25, ~10s)

**Screen:** title card / final banner.

**Say:** *"The banner said SME2. The debugger proved otherwise. We measured
the real cost, found the fix llama.cpp already half-supports, and named the
exact capability — per-phase kernel-family dispatch — it is still missing.
Apache-2.0. Kernels, harness, and MCP server: all reusable for the next one."*

---

## Total runtime budget

| Beat | Duration | Cumulative | Live? |
|---|---:|---:|---|
| 0 — Title | 0:07 | 0:07 | text only |
| 1 — The claim | 0:09 | 0:16 | live (~1–2s cmd) |
| 2 — The lie | 0:10 | 0:26 | cited (RUN_LIVE_L3=1 for live, +90s) |
| 3 — `-t 2` proof | 0:34 | 1:00 | **live (~27–32s cmd)** |
| 4 — Measured cost | 0:15 | 1:15 | cited (instant) |
| 5 — The fix + the gap | 0:35 | 1:50 | live (~5–10s cmd) |
| 6 — Silicon proof | 0:10 | 2:00 | live (~4s cmd) |
| 7 — Reusable + upstream | 0:15 | 2:15 | live (~2–3s cmd) |
| Close | 0:10 | 2:25 | text only |

35 seconds of slack under the 3:00 cap for narration pacing, breath, and a
hard cut. Measured end-to-end `demo.sh` wall time at the default `PAUSE=2.5`
was **103s** on the reference machine (see `demo/demo.sh`'s own final line,
"Total demo.sh wall time") — the difference between that and this table's
2:25 is the narration/talking time this table budgets *on top of* the
script's own pacing. If you need to cut further: trim Beat 6 first (drop
`kernel_bench`'s output to just the fp32 row), then Beat 0's title card. Do
**not** cut Beat 3 (the live proof) or Beat 5 (the fix + the gap) — those two
are what separates this submission from a pure diagnosis.
