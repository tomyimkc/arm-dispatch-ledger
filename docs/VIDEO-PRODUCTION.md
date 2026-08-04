# Submission video — production record

**Deliverable:** `polygraph-final.mp4` — 1920×1080, H.264/AAC, 30 fps,
**103.06 s (1:43)**, ~59 MB. Comfortably inside the contest's 3-minute cap.

Not committed to this repository: a 59 MB binary would dominate a repo whose entire point is
small, auditable, text-based evidence. It is uploaded to YouTube and linked from the Devpost
submission instead.

## Why this page exists

Every number spoken or shown in the video traces to a file in `results/`. This page is the
mapping, so a judge can check the video against the evidence without watching frame-by-frame —
and so the video cannot silently drift from the repo if a number is later corrected. It is the
same discipline `tools/check_claims.py` enforces for the prose.

## Who it is for

**General public first, contest judges second.** The one sentence a viewer should be able to
repeat a week later is: *"He built Polygraph — a tool that checks whether software is telling the
truth."*

An earlier cut opened mid-investigation and never said what the project was or why anyone should
care — it simplified the vocabulary but kept a structure aimed at engineers. A later cut fixed the
vocabulary and named the product, but still opened cold on the 13 s side-by-side race before the
viewer had any of that context. This cut fixes the position, not just the words: the presenter now
opens by naming Polygraph and saying in one line what it does, and the race itself moves from a
cold open to a mid-video payoff.

## Structure — 6 beats x 15 s + a 13 s mid-video race = 1:43

**Re-cut from a version that opened cold on the 13 s race before the viewer knew what the project
even was.** The presenter now opens by naming Polygraph and saying in one line what it does. The
race itself is unchanged — same recording, same numbers — but it no longer plays first: it now
lands as a mid-video payoff, right after beat 04 sets it up ("same laptop, same question — here's
the difference") and right before beat 05 turns that same honest lens on the project's own
numbers.

| # | Timing | Said in plain English | Shown on the panel | Source |
|---|---|---|---|---|
| 01 — intro | 0:00–0:15 | "Hi, I'm Tom, and I built Polygraph. It's a tool that checks whether software is telling the truth about your computer's hardware. I pointed it at the AI assistant on my own laptop, and what it found genuinely surprised me." | "a lie detector for software" · "did that actually run?" · "works on programs you didn't write" | — (framing beat, no measured figure shown) |
| 02 — finding | 0:15–0:30 | "That assistant runs entirely offline. Nothing is sent to a company. It reported that it was using my laptop's AI chip, the part built to make this fast. Polygraph watched that chip and counted. It never ran. Not once." | reported: enabled · actually ran: **0 times** · instead: **31,871** slow-path calls | `results/dispatch-ledger-darwin-arm64.json` |
| 03 — how it works | 0:30–0:45 | "Here's why nobody caught this. Almost every test measures how long something took, and the time looks perfectly normal either way. Polygraph doesn't measure time. It measures what actually ran, down to the individual chip instructions." | benchmarks measure *how long*; this measures *what actually ran*; works on software you didn't write | `tools/verify_dispatch.py` |
| 04 — the fix | 0:45–1:00 | "Two hidden rules inside the code were quietly ruling it out. So I wrote a fix that picks the right setting by itself. No settings for you to learn. Same laptop, same question — here's the difference." | "two hidden rules quietly ruled the chip out" · "the fix picks the setting by itself" · "watch: same laptop, before and after" | `results/AUTODEFAULTS.md` |
| — race | 1:00–1:13 (13 s, no narration) | *(silent — beat 04's line carries it)* | identical answer · before **3.42 s** · after **1.72 s (1.99x)** | `results/video/race-capture.json` |
| 05 — honesty | 1:13–1:28 | "Then I turned Polygraph on my own claim. Most of that speed-up came from something people already knew about. The chip itself added about thirty percent. I published both numbers, including the one that makes my result look smaller." | total **3.43x** · already-known trick **3.95x of it — not mine** · the chip itself **1.31x — the real part** | `results/REMEASURE-2026-08-04-QUIET.md` |
| 06 — close | 1:28–1:43 | "The fix is written and sent to the people who maintain that software. And Polygraph is free. Point it at anything that claims to use your hardware, and it will tell you whether that's actually true." | "sent upstream · llama.cpp #26547" · "free and open source" · "point it at anything that claims to use your chip" | `results/AUTODEFAULTS.md` |

Jargon kept off the soundtrack entirely: `ne11`, `kai_run_matmul`, KleidiAI, SME2, GEMV, dispatch,
thread cap, tokens/sec. Each has a plain stand-in ("your laptop's AI chip", "chip instructions",
"hidden rule", "how fast it types").

## The mid-video race is measured, not staged

`tools/capture_race.py` streams both binaries' stdout and timestamps **every character as it
actually arrives**. `results/video/race-capture.json` is that recording; the video's typing speed
is a replay of it. Nothing is simulated, slowed, or sped up.

Fairness controls:

- Same prompt, same `--seed 7`, same model, same context size → **byte-identical output**, so the
  only visible difference is speed.
- Runs alternated A,B,A,B,A,B so machine contention hit both sides equally.
- Three reps each; the **median** run was kept.

**Disclosure — one outlier.** The patched build's three runs were **4.225 s, 1.721 s, 1.723 s**;
the baseline's were 3.238 s, 3.424 s, 3.671 s. The patched first run was slower than baseline —
almost certainly a cold page-cache effect on first execution of a freshly built binary. The median
(1.723 s) is what the video replays, and reps 2 and 3 agree to within 2 ms. All six raw
wall-clock times are committed in `race-capture.json` under `all_walls`, not just the ones that
flatter the result.

**Why the race shows 1.99x and not 2.15x.** `results/AUTODEFAULTS.md` measures *decode* throughput
in isolation (2.15x). The race is end-to-end wall clock, which also includes model load and prompt
processing — work this patch does not touch. A short answer would dilute the difference further; an
earlier 60-token capture showed only 1.23x for exactly that reason. The 260-token answer in the
video is both the honest figure and the realistic case, because decode dominates once the assistant
writes more than a sentence.

## Display-time cleanup

`llama.cpp` prints a timing footer and an `Exiting...` line after the answer. That is real output
and `race-capture.json` keeps it verbatim, but it reads as debug noise to a general viewer, so it
is trimmed **at display time only** (`ColdOpen.tsx`, `visibleLen`). Character timings are untouched.

## Pipeline

Fully scripted, no manual editing, no screen recording, no on-camera reshoots.

```
reference portrait (on disk)
  └─> base presenter frames, 16:9, speaker on a known third, clean negative space opposite
        └─> xAI grok-imagine-video-1.5  (image → lip-synced 15 s talking clip, 720p)
              └─> Remotion 4.0.503 composition (React/TSX), 1920×1080 @30
                    ├─ presenter clip full-bleed
                    ├─ side-matched scrim so the panel never occludes the speaker
                    ├─ data cards animated in (spring), sourced from story.json
                    └─ persistent footer + progress bar
                          └─> H.264 / AAC, CRF 18, yuv420p
```

Working tree: `/private/tmp/armledger-video-20260804/`
- `gen/scenes.json` — the single source of truth: per-beat speech, expression, panel cards, base frame
- `gen/generate.py` — submits and polls the clip generations, downloads results
- `remotion/Polygraph.tsx` — the composition
- `remotion/story.json` — generated from `scenes.json`; what the renderer consumes

### One deliberate deviation from the reference pipeline

The pipeline this was modelled on fed the input image to the API through a **public cloudflared
tunnel** and had the finished clip **pushed back** to a local upload server. That exposes a local
file server to the internet for the duration of the run.

Neither is necessary. The API accepts the input image as a `data:image/png;base64,…` URI, and the
finished clip is downloadable from the polled `.video.url`. So this run kept everything local —
no ingress, no public exposure, no ephemeral server. `gen/generate.py` documents this in its
module docstring.

## Captions

Burned-in captions plus a sidecar `.srt` (36 cues) ship with the video.

**The display text is authored, not transcribed.** `faster-whisper` was run over each clip, but
only its *timings* are used — the ASR mangles exactly the vocabulary this video is about
("KleidiAI matmul" → "Clyde AI Matmul", "llama.cpp" → "Lama .CPP", "hardcoded" → "hardcaid"),
and the spoken script also contains pronunciation spellings ("llama dot c p p") that must never
appear on screen. Cue text is written out correctly and distributed across each clip's measured
speech span, weighted by character length.

Cues within a scene swap instantly rather than cross-fading; fading every boundary made the
caption blink out between lines of the same sentence.

## Keeping the panel off the speaker's face

The clip generator drifts the subject away from where the base frame placed it — scene 02's
speaker came back centred at 0.41–0.59 of frame width instead of on the right third, so the data
panel landed across his face.

This is solved by measurement, not by eye. `layout.json` is generated by scanning each clip's
column luminance over the upper 62% of frame height (the background is uniform dark navy, so the
subject separates cleanly), then solving for the smallest pan/zoom and widest panel that still
clears the speaker's head by ≥ 0.055 of frame width. Only scene 02 needed a real pan (+0.10,
1.21× zoom); the rest were fixed by narrowing the panel from 0.44 to 0.40.

Measured clearance on the final render, all six scenes:

| scene | side | head span | clearance |
|---|---|---|---:|
| 01 | LEFT | 0.260–0.455 | +0.095 |
| 02 | RIGHT | 0.519–0.728 | +0.069 |
| 03 | LEFT | 0.200–0.464 | +0.086 |
| 04 | RIGHT | 0.565–0.737 | +0.115 |
| 05 | LEFT | 0.234–0.465 | +0.085 |
| 06 | RIGHT | 0.545–0.771 | +0.095 |

## Reproducing

Requires an xAI API key at `~/.grok/auth.json`, Node ≥ 20, and ffmpeg.

```bash
cd /private/tmp/armledger-video-20260804
python3 gen/generate.py                 # skips clips already downloaded
cd remotion && npx remotion render index.ts Polygraph \
    out/polygraph-final.mp4 --codec h264 --audio-codec aac --crf 18 --pixel-format yuv420p
```

Clip generation is idempotent — existing `gen/output/*-raw.mp4` files are skipped, so a re-run
only fills gaps.

## Compliance with the contest's video rules

| Rule | Status |
|---|---|
| Under 3 minutes | 1:43 ✅ |
| Publicly visible on YouTube / Vimeo / Youku | to upload before submitting |
| Shows the project functioning on the device it was built for | on-screen figures are the measured M4 Max results; the `lldb` and `llama-bench` runs behind them are reproducible via `demo/demo.sh` |
| No third-party trademarks | none used |
| No copyrighted music | **no music track at all** — presenter audio only |
| Captions | burned in, plus a sidecar `.srt` for the YouTube upload |

The presenter is the project author. The likeness is generated from his own reference portrait,
with his consent, for his own submission.
