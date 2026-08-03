# Demo video shot list — Arm Dispatch Ledger (target: under 3:00 total)

Rule for this video, matching the repo's own anti-overclaim discipline: every number
and every terminal output shown on screen must be the real output of the command shown
— no invented numbers, no unlabeled speed-ups. Where footage is sped up for pacing
(dispatch verification genuinely takes ~30-60s per thread-count config), put a small
on-screen caption saying so (e.g. "sped up 4x — real dispatch check"). Record at 1080p+,
large terminal font (18-20pt), high-contrast theme so text is readable at video
resolution.

**Before recording:** run `./scripts/setup.sh` once so the build/model cache is warm —
every command below should execute in the timings given, not include a cold build.
Have three terminal panes/tabs ready: (1) llama-cli banner + lldb sweep, (2) kernel
correctness/bench, (3) MCP server. If, by recording time, `verify-free-arm64.yml` has
actually been run for real on GitHub Actions (not just lint-validated), also have that
Actions run tab open — **only show it if it is a real, completed run**; if it hasn't
run yet, use the fallback line in Shot 7 instead of claiming a result that doesn't
exist.

---

## Shot 1 — The reveal (0:00–0:20)

**Goal:** the banner says SME2, the breakpoint says zero. This must land in the first
20 seconds.

**Screen:** split terminal. Left: run llama-cli at 8 threads (the default on this
machine) and freeze-frame the banner line. Right: the same run under the dispatch
verifier, showing the SME2 kernel breakpoint hit count.

**On-screen commands** (left pane, run first, let banner print, then Ctrl-C or let the
short generation finish — this is fast, ~2-3s):

```bash
./bin/llama-cli -m q05.gguf -p "The capital of France is" -n 8 -no-cnv -st --simple-io -t 8
```

Highlight (zoom/box overlay) this exact substring in the output as it prints:

```
SME = 1 | SME2 = 1 | KLEIDIAI = 1
```

**Right pane** (this is the part that should be pre-run and played back — a live 8-
thread decode dispatch check takes about 30-60s; caption it "sped up" or cut straight
to the finished table):

```bash
python3 tools/verify_dispatch.py --binary ./bin/llama-cli --model q05.gguf \
  --threads 8 --workloads decode_short --l3-debugger lldb
```

Freeze on the output row, highlighted in red:

```
threads=8  workload=decode_short  hits=0  verdict=SILENT_FALLBACK
```

**Narration (voice-over, ~15s):** *"This banner says SME2 is running. It's lying. We
put a breakpoint on the actual SME2 kernel — zero hits. This is Arm Dispatch Ledger:
the tool that catches the gap between what a binary claims and what it actually
dispatches."*

**On-screen title card, last 3s of this shot:** "Arm Dispatch Ledger"

---

## Shot 2 — What's actually happening (0:20–0:45)

**Goal:** explain the bug in plain terms with the real source lines on screen.

**Screen:** editor/terminal `cat`/syntax-highlighted view of the two root-cause
snippets (keep it to ~6 lines each, not a full scroll of the file):

```bash
sed -n '148,169p' /path/to/llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
```

then

```bash
sed -n '1094,1113p' /path/to/llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
```

Overlay two callout labels pointing at the code: "hardcoded per-chip thread cap" and
"collapses to NEON above the cap."

**Narration (~20s):** *"KleidiAI caps SME2 at 2 threads on an Apple M4 — hardcoded, by
chip name, in the source. Ask llama.cpp to use its default thread count — 8 or 16 cores
— and every single-token decode step silently falls back to NEON. The log never says
so. We found this with a debugger, not a benchmark — a timing-only test can't see it."*

---

## Shot 3 — The honest sweep (0:45–1:20, 35s)

**Goal:** show the full thread × workload table, and the un-flattering half of the
finding (NEON wins prefill).

**Screen:** `cat` or pretty-print the measured table from `results/SUMMARY.md` §2 and
§3 — either scroll a rendered Markdown preview, or print a condensed version with a
small script. Suggested on-screen command (real, deterministic, no wait):

```bash
column -t -s'|' results/bench/bench-apple-m4-max.md | less -S
```

Show, specifically, these two rows on screen (build a simple text overlay/table if the
raw file is too dense to read at video resolution):

```
decode:        SME2@2 = 327.6 tok/s   NEON@8 = 155.3 tok/s   -> SME2 wins 2.1x
prefill_long:  SME2@8 = 1830.1 tok/s  NEON@8 = 2676.4 tok/s  -> NEON wins 1.46x
```

**Narration (~25s):** *"So does the cap actually cost anything? We measured it —
five reps per cell, interleaved, median and stddev reported, never a bare mean. For
decode, SME2 wins outright at every thread count we tried. But for prefill, once NEON
is allowed its own best thread count instead of being compared at a misleading extreme,
plain NEON wins — by one and a half times. That's not the flattering story a demo
would pick. It's the one the numbers actually show."*

---

## Shot 4 — Proving the silicon isn't the limiter (1:20–1:45, 25s)

**Goal:** the hand-written kernel library — correctness plus an honest baseline
(Accelerate still wins, no strawman).

**On-screen commands** (real, fast — kernel_test/kernel_bench run in seconds):

```bash
cd kernels/build && ./kernel_test && ctest
./kernel_bench
```

Highlight the printed table columns: NEON / SME2(packed) / Accelerate GFLOP/s.

**Narration (~20s):** *"We also hand-wrote SME2 kernels from scratch — bit-exact
against a scalar reference — to prove the M4's silicon genuinely supports this. But
we're not going to tell you it's 19 times faster than naive NEON, because that's a
strawman. Apple's own Accelerate library still beats our hand-written kernel by roughly
three to eighteen times, depending on matrix size. The honest gap is elsewhere: Accelerate has no integer
GEMM at all — that's the real, unclaimed territory."*

---

## Shot 5 — The agentic piece: MCP server (1:45–2:10, 25s)

**Goal:** show an agent asking "did this actually dispatch?" live through MCP, matching
the challenge's explicit call-out of agentic MCP workloads.

**On-screen commands:**

```bash
python3 mcp/server.py --selftest
```

Then, ideally, a short clip of the tool being invoked from an actual MCP client (Claude
Code / Claude Desktop) asking a natural-language question like *"is SME2 actually
running for this model at 8 threads?"* and getting back the `verify_dispatch` tool's
JSON verdict rendered in the client UI. If a live client isn't available at recording
time, the `--selftest` JSON output alone is sufficient — do not stage a fake client UI.

**Narration (~20s):** *"This isn't just a script — it's an MCP server. Any agent can
call `verify_dispatch` and get a grounded answer instead of trusting a banner. That's
the piece built specifically for agentic, multi-model workloads on Arm."*

---

## Shot 6 — Judge-reproducible, zero cost (2:10–2:35, 25s)

**Goal:** show that anyone — no owned Arm hardware required — can reproduce this.

**Screen:** the `.github/workflows/verify-free-arm64.yml` file (brief scroll, highlight
`runs-on: ubuntu-24.04-arm` and the header comment explaining it's the free,
judge-reproducible lane), **and only if it has actually been run for real by recording
time**, cut to the real, completed GitHub Actions run and its job summary showing
`results/LEDGER.md`.

**Fallback if the workflow has not yet been run on real GitHub infrastructure at
recording time:** do not show a fabricated green check. Instead, narrate: *"This
workflow runs the entire pipeline on GitHub's own free Arm64 runners — fork the repo
and click Run to reproduce every number in this video yourself, no Arm hardware
required."* — true regardless of whether it has been triggered yet.

**Narration (~20s):** *"Every result here is reproducible by anyone, for free, on
GitHub's hosted Arm64 runners — fork this repo and re-run the whole pipeline
yourself."*

---

## Shot 7 — Giving back (2:35–2:50, 15s)

**Screen:** `docs/UPSTREAM-ISSUE.md` open, scrolled to the title and root-cause section.

**Narration (~12s):** *"And we're filing this upstream — a one-line log warning is the
actual right-sized fix, not just a finding to keep for ourselves."*

---

## Shot 8 — Close (2:50–2:58, 8s)

**Screen:** title card — repo name, Apache-2.0, one line: "The kernels, the harness,
and the MCP server are all reusable — for this bug, and the next one."

**Narration (~6s):** *"Arm Dispatch Ledger. Apache-2.0. Link in the submission."*

**No audio/music track needed** — if background music is desired, use only royalty-free
or CC0 tracks you have rights to (e.g. YouTube Audio Library "no copyright" tracks
marked safe for use), kept low under the voice-over. Do not use any commercially
licensed or unlicensed music.

---

## Total runtime budget

| Shot | Duration | Cumulative |
|---|---:|---:|
| 1 — Reveal | 0:20 | 0:20 |
| 2 — Root cause | 0:25 | 0:45 |
| 3 — Honest sweep | 0:35 | 1:20 |
| 4 — Silicon proof | 0:25 | 1:45 |
| 5 — MCP server | 0:25 | 2:10 |
| 6 — Zero-cost repro | 0:25 | 2:35 |
| 7 — Upstream issue | 0:15 | 2:50 |
| 8 — Close | 0:08 | 2:58 |

2 seconds of slack under the 3:00 cap for a hard cut. If any shot runs long in editing,
cut from Shot 3 (the sweep) or Shot 6 (the workflow file scroll) first — Shots 1 and 2
are the reveal and must not be trimmed.
