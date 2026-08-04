<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors -->

# Recording the submission video

This directory makes recording the &lt;3-minute submission video mechanical:
`demo.sh` runs the whole story, narrated by its own on-screen text, timed and
idempotent. You provide the terminal, the microphone, and (optionally) your
own spoken narration matching `SHOTLIST.md`.

- **`demo.sh`** — run this. It is the actual recording script.
- **`SHOTLIST.md`** — what to say over each beat, and the timing budget.
  Read it once before recording so the pacing isn't a surprise.

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
LLAMA_CLI=/path/to/llama-cli MODEL_PATH=/path/to/model.gguf ./demo/demo.sh
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

---

## 2. Terminal setup for recording

| Setting | Recommendation |
|---|---|
| Resolution | 1080p or higher |
| Font | Monospace, **18–20pt** — has to stay legible after video compression |
| Theme | High-contrast (light-on-dark or dark-on-light). `demo.sh` uses red / green / yellow / cyan / magenta — pick a theme where all five stay readable, and don't force `NO_COLOR=1` for the actual recording (the colour *is* the "highlight this line" mechanism the old shot list asked for manually) |
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
to Beat 2 that the default path already covers by citing the committed,
independently-reproducible ledger (see `SHOTLIST.md`'s Beat 2 note for why
that's the honest tradeoff, not a shortcut).

**A real caveat, observed while building this script, not a hypothetical:**
total wall time varied noticeably run to run on the same machine — as low as
~103s in a clean run, and once as high as ~199s. Tracing the slow run down:
`ps aux | grep llama` showed a second `llama-bench` process, not started by
this script, pinned at 1000%+ CPU at the same time — almost certainly a
**different concurrent session** on this shared dev machine (this repo is
routinely worked on by several agents/sessions at once; see the repo's own
`spark-cluster-ops` guidance on shared-machine contention), not thermal
throttling from this script itself. Beat 5's single-shot `llama-cli`
dry-run prompt-processing rate is the number that swung most under that
contention — anywhere from roughly 2 t/s to over 1000 t/s across different
runs — which is exactly why that number is narrated as "ignore this, it's
noise" rather than cited as evidence.

**Before recording, run `ps aux | grep -iE "llama|lldb"` (or `top`) and
confirm nothing else heavy is running on the box.** If something is, either
wait for it to finish or coordinate with whoever owns it — don't kill a
process you don't recognize on a shared machine. Do one throwaway
`./demo/demo.sh` pass immediately before the take you'll keep; if that pass
finishes well under budget with nothing else competing for CPU, the recorded
take will too.

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

- Watch the full take back once before submitting. Confirm Beat 3 (the live
  `-t 2` dispatch probe) actually shows hit counts scrolling, not a frozen
  terminal — that beat is the single most important "this is not staged"
  moment in the video.
- Confirm no beat shows a fabricated or stale result. In particular: **do not
  show a CI green check that has not actually been run for real** on GitHub
  Actions infrastructure. If `verify-free-arm64.yml` hasn't been triggered by
  recording time, `SHOTLIST.md` has the honest fallback narration for that —
  use it instead of implying a run exists.
- Trim narration gaps in post if you talked faster than the default `PAUSE`;
  do not trim the live command output itself (that would misrepresent how
  long the real dispatch probe actually took).
