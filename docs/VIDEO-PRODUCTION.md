# Submission video — production record

**Deliverable:** `arm-dispatch-ledger-submission-video.mp4` — 1920×1080, H.264/AAC, 30 fps,
**90.05 s (1:30)**, ~53 MB. Comfortably inside the contest's 3-minute cap.

Not committed to this repository: a 53 MB binary would dominate a repo whose entire point is
small, auditable, text-based evidence. It is uploaded to YouTube and linked from the Devpost
submission instead.

## Why this page exists

Every number spoken or shown in the video traces to a file in `results/`. This page is the
mapping, so a judge can check the video against the evidence without watching frame-by-frame —
and so the video cannot silently drift from the repo if a number is later corrected. It is the
same discipline `tools/check_claims.py` enforces for the prose.

## Who it is for

**General public first, contest judges second.** The one sentence a viewer should be able to
repeat a week later is: *"He built a tool that checks whether software is telling the truth."*

An earlier cut opened mid-investigation and never said what the project was or why anyone should
care — it simplified the vocabulary but kept a structure aimed at engineers. This cut fixes the
structure, not just the words.

## Structure — 13 s cold open + 5 beats x 15 s = 1:28

**0:00–0:13 — the race, no narration.** A real side-by-side: the same AI assistant, same laptop,
same question, same seed, answering twice. The right pane finishes at **1.72 s**; the left is still
writing at 2.40 s and lands at **3.42 s**. Both answers are byte-identical, word for word. The
viewer feels the benefit before a single word is explained.

| # | Said in plain English | Shown on the panel | Source |
|---|---|---|---|
| 01 | "That's an AI assistant running entirely on your own laptop. The slow one believes it's using your laptop's AI chip. It never did." | identical answer · before **3.42 s** · after **1.72 s (1.99x)** | `results/video/race-capture.json` |
| 02 | "I built a tool that checks whether software is telling the truth — not whether it's fast." | benchmarks measure *how long*; this measures *what actually ran*; works on software you didn't write | `tools/verify_dispatch.py` |
| 03 | "The answer was zero. The fast chip never ran once, while every message said it was working." | reported: enabled · actually ran: **0 times** · instead: **31,871** slow-path calls | `results/dispatch-ledger-darwin-arm64.json` |
| 04 | "Most of the speed-up came from something people already knew. I published both numbers." | total **3.43x** · already-known trick **3.95x of it** · the chip itself **1.31x** | `results/REMEASURE-2026-08-04-QUIET.md` |
| 05 | "The fix is sent to the maintainers. It picks the right setting by itself. The tool is free." | now automatic · **2.15x** typing speed · upstream **#26547** | `results/AUTODEFAULTS.md` |

Jargon kept off the soundtrack entirely: `ne11`, `kai_run_matmul`, KleidiAI, SME2, GEMV, dispatch,
thread cap, tokens/sec. Each has a plain stand-in ("your laptop's AI chip", "chip instructions",
"hidden rule", "how fast it types").

## The cold-open race is measured, not staged

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
- `remotion/ArmDispatchLedger.tsx` — the composition
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

Burned-in captions plus a sidecar `.srt` (35 cues) ship with the video.

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
cd remotion && npx remotion render index.ts ArmDispatchLedger \
    out/arm-dispatch-ledger.mp4 --codec h264 --audio-codec aac --crf 18 --pixel-format yuv420p
```

Clip generation is idempotent — existing `gen/output/*-raw.mp4` files are skipped, so a re-run
only fills gaps.

## Compliance with the contest's video rules

| Rule | Status |
|---|---|
| Under 3 minutes | 1:30 ✅ |
| Publicly visible on YouTube / Vimeo / Youku | to upload before submitting |
| Shows the project functioning on the device it was built for | on-screen figures are the measured M4 Max results; the `lldb` and `llama-bench` runs behind them are reproducible via `demo/demo.sh` |
| No third-party trademarks | none used |
| No copyrighted music | **no music track at all** — presenter audio only |
| Captions | burned in, plus a sidecar `.srt` for the YouTube upload |

The presenter is the project author. The likeness is generated from his own reference portrait,
with his consent, for his own submission.
