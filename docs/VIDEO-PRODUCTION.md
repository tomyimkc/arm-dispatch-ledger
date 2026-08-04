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

## Structure — 6 beats × 15.04 s

| # | Beat | Claim shown on screen | Source |
|---|---|---|---|
| 01 | The claim | banner `SME = 1 \| SME2 = 1 \| KLEIDIAI = 1`; log `primary q4 kernel feature SME2`; **zero SME2 kernels executed** | `results/GROUND-TRUTH-DISPATCH.md` |
| 02 | The proof | decode @12 threads: **0 SME2 / 31,871 NEON**; @2 threads: **5,826 SME2**; 18 `kai_run_matmul` symbols | `results/dispatch-ledger-darwin-arm64.json` |
| 03 | Root cause | thread cap hardcoded (`M4 Max = 2`); hybrid needs `ne11 >= 128`; decode is always `ne11 == 1` | `kleidiai.cpp` @ `dbadb68`, cited in `docs/FINDINGS.md` |
| 04 | Honest decomposition | total **3.43×** (93.6 → 321.0 t/s); **3.95×** from thread tuning alone with SME2 OFF; **1.31×** from SME2 itself | `results/REMEASURE-2026-08-04-QUIET.md` |
| 05 | The fix | decode **67.8 → 145.9 t/s (2.15×)** zero flags; matches hand-tuned ceiling 146.0 (0.999×); prefill 1835.2 → 1779.8 unchanged | `results/AUTODEFAULTS.md` |
| 06 | Why the patch, not the flag | naive `-t 2` collapses prefill **1835.2 → 975.6 (−47%)**; patch does not; upstream `#26547` | `results/AUTODEFAULTS.md` |

Beat 04 is deliberate. The video states on camera that most of the raw tuning win is a
well-known Apple Silicon effect and **not** this project's discovery, and that SME2's own
contribution is the smaller 1.31×. A submission video is the easiest place in the world to quietly
drop that caveat; it is stated instead.

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
