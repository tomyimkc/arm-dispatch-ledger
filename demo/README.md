<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors -->

# Recording the submission video

This directory makes recording the &lt;3-minute submission video mechanical:
`demo.sh` runs the whole story, narrated by its own on-screen text, timed and
idempotent. You provide the terminal, the microphone, and (optionally) your
own spoken narration matching `SHOTLIST.md`.

- **`demo.sh`** — run this. It is the actual recording script. Five beats:
  the hook (the banner's claim), the proof (a debugger, not a benchmark),
  the cost (real `llama-bench` numbers), the root cause (the actual source
  lines), and the honest ending (we patched it, it didn't help, we published
  that anyway).
- **`SHOTLIST.md`** — what to say over each beat, and the timing budget.
  Read it once before recording so the pacing isn't a surprise.

**This demo was rebuilt 2026-08-04** around
[`results/REMEASURE-2026-08-04-QUIET.md`](../results/REMEASURE-2026-08-04-QUIET.md), the
authoritative, round-robin-interleaved re-measurement of this project's throughput claims. If
you have an older recording or draft quoting "+57.3%" or a 4.4x decode speedup, those numbers
are **retracted** — re-record from this version of the script.

---

## 1. One-time setup

```bash
cd arm-dispatch-ledger
./scripts/setup.sh          # builds llama.cpp (-DGGML_CPU_KLEIDIAI=ON),
                             # fetches the demo GGUF, builds kernels/
```

If you already have a working build elsewhere (this repo's own dev sessions
used `/tmp/llama.cpp` + `/tmp/ggufs/q05.gguf` directly, outside
`scripts/setup.sh`'s cache), `demo.sh` looks for both locations automatically.
Override explicitly if neither matches:

```bash
LLAMA_CLI=/path/to/llama-cli LLAMA_BENCH=/path/to/llama-bench \
MODEL_PATH=/path/to/model.gguf ./demo/demo.sh
```

`demo.sh` degrades gracefully if a binary or model is missing — it prints
the exact setup command and skips just that beat, so a broken environment
never blocks recording the beats that *do* work. Run it once, unrecorded,
before you record anything — this warms the model in the OS page cache and
lets you confirm every beat resolves, so the take you actually record doesn't
have a surprise `[skip]` line in it.

```bash
PAUSE=0.5 ./demo/demo.sh       # fast dry run, confirm everything resolves
```

### Optional: building the patched binary (for Beat 5's live moment)

Beat 5 ("the honest ending") is more compelling live: it runs the *actual*
`GGML_KLEIDIAI_PHASE_AWARE=1` patch and shows its real warning/enabled log
lines. This is optional — without it, Beat 5 still runs, citing the same
evidence from `results/`. To get the live version:

```bash
# Fresh clone at (or near) llama.cpp @ dbadb68, separate from the baseline build
git clone https://github.com/ggml-org/llama.cpp /tmp/llama-phase-aware
cd /tmp/llama-phase-aware
git apply /path/to/arm-dispatch-ledger/patches/0001-kleidiai-phase-aware-dispatch.patch
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --target ggml-cpu llama-cli llama-bench -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
```

`demo.sh` looks for this at `/tmp/llama-phase-aware/build/bin/{llama-cli,llama-bench}` (and the
matching `$TMPDIR/arm-dispatch-ledger-cache/...` path) automatically; override with
`LLAMA_CLI_PATCHED` / `LLAMA_BENCH_PATCHED` if you built it elsewhere. See
`patches/README.md` for the full patch rationale and what it does and does not claim.

---

## 2. Terminal setup for recording

| Setting | Recommendation |
|---|---|
| Resolution | 1080p or higher |
| Font | Monospace, **18–20pt** — has to stay legible after video compression |
| Theme | High-contrast (light-on-dark or dark-on-light). `demo.sh` uses red / green / yellow / cyan / magenta — pick a theme where all five stay readable, and don't force `NO_COLOR=1` for the actual recording (the colour *is* the "highlight this line" mechanism the shot list asks for) |
| Window | Single full-width pane. `demo.sh` sequences everything itself — no split screen needed |
| Shell prompt | Minimal (a bare `$` or hide it entirely) so it doesn't compete with `demo.sh`'s own `$ <command>` echo lines |

Quick terminal size check before recording:

```bash
tput cols; tput lines     # aim for at least 120x40 so long command lines
                           # (verify_dispatch.py's, especially) don't wrap
```

---

## 3. Recording

Standard screen recording (macOS: QuickTime Player → New Screen Recording,
or any capture tool that records terminal + microphone):

```bash
./demo/demo.sh
```

Talk over each beat as it prints, following `SHOTLIST.md`'s "Say:" lines —
they're read directly from `demo.sh`'s own narration text, so what you say
matches what's on screen. Default `PAUSE=2.5` seconds gives you a beat between
each command to read the narration aloud before the next one fires; adjust to
your own reading speed:

```bash
PAUSE=3.5 ./demo/demo.sh      # more room to talk
PAUSE=1.5 ./demo/demo.sh      # tighter cut
```

If a take goes wrong partway through, just re-run `./demo/demo.sh` — it is
idempotent (writes only to a scratch temp directory, never into the repo) and
safe to re-run any number of times back to back.

**Do not set `RUN_LIVE_L3=1` for the recorded take** unless you are
deliberately doing a longer cut — it adds a real ~90-second live `lldb` sweep
to Beat 2 (and, if the patched binary is present, up to two more in Beat 5)
that the default path already covers by citing the committed,
independently-reproducible ledgers (see `SHOTLIST.md`'s Beat 2 note for why
that's the honest tradeoff, not a shortcut).

**A real caveat, observed while building this script, not a hypothetical —
and the exact reason `results/REMEASURE-2026-08-04-QUIET.md` exists:** total
wall time, and any *quick* live `llama-bench` numbers Beat 3 shows you, can
vary noticeably run to run on a shared machine. An earlier version of this
project's throughput claims were built on measurements taken while this
16-core host had a 1-minute load average of 66–147 from unrelated concurrent
agent sessions, with baseline and patched configs measured in different,
non-interleaved time windows — that combination manufactured a fake speedup.
Beat 3's live sanity check is deliberately narrated as illustrative, not
authoritative, for exactly this reason; the cited table next to it is the
one from the round-robin-interleaved, 7-reps-per-config re-measurement.

**Before recording, run `ps aux | grep -iE "llama|lldb"` (or `top`) and
confirm nothing else heavy is running on the box.** If something is, either
wait for it to finish or coordinate with whoever owns it — don't kill a
process you don't recognize on a shared machine. Do one throwaway
`./demo/demo.sh` pass immediately before the take you'll keep; if that pass
finishes well under budget with nothing else competing for CPU, the recorded
take will too. Measured on a quiet reference run: **49s** total `demo.sh`
wall time at the default `PAUSE=2.5` (see `SHOTLIST.md`'s runtime table for
how that maps onto the 3:00 video budget).

---

## 4. Alternative: an asciinema cast

If you'd rather ship a terminal-native recording (embeddable, scrubbable,
copy-pasteable by a judge) instead of or alongside a screen-captured video:

```bash
# Install (not required for demo.sh itself -- only for this recording method)
brew install asciinema        # macOS
# or: pip install asciinema   # anywhere with Python

# Record
asciinema rec arm-dispatch-ledger-demo.cast -c "./demo/demo.sh"

# Play it back locally to check it before sharing
asciinema play arm-dispatch-ledger-demo.cast

# Optionally publish for a shareable link
asciinema upload arm-dispatch-ledger-demo.cast
```

Notes specific to this script:

- `demo.sh` detects a non-interactive/non-color-capable target and degrades
  colour automatically; `asciinema rec` runs it in a real pty, so colours and
  the `clear` at the start work exactly as they do in a normal terminal.
- The cast file is plain JSON/asciicast text — safe to commit or attach
  separately from the main video; it is **not** a substitute for the required
  video submission unless the challenge rules explicitly allow it. Check the
  Devpost submission form before relying on it as your only video artifact.
- Narrate live while recording (asciinema does not capture audio) and either
  layer your voice-over back over the `.cast` playback when producing the
  final video, or record audio separately using `SHOTLIST.md`'s timings as
  your script.

---

## 5. After recording

- Watch the full take back once before submitting. Confirm Beat 2 (the live
  `-t 2` dispatch probe) actually shows hit counts scrolling, not a frozen
  terminal — that beat is the single most important "this is not staged"
  moment in the video.
- Confirm Beat 5 (the honest ending) actually shows the patched binary's real
  log lines if `LLAMA_CLI_PATCHED` was found, or the cited fallback if it
  wasn't — either is fine, but don't let editing accidentally cut the
  "~12% SLOWER" line. That admission is the differentiator; don't trim it out
  for pacing.
- Confirm no beat shows a fabricated or stale result. In particular: **do not
  show a CI green check that has not actually been run** for real on GitHub
  Actions infrastructure, and do not narrate or caption 57.3%, 71.6, 45.5,
  4.4x, 198.9, 2257.5, or 1145.0 as a current number — those are retracted
  (see the header of `demo/demo.sh` and `results/REMEASURE-2026-08-04-QUIET.md`).
- Trim narration gaps in post if you talked faster than the default `PAUSE`;
  do not trim the live command output itself (that would misrepresent how
  long the real dispatch probe actually took).
