<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Polygraph contributors -->

# Recording the submission video

**This file is a pointer, not a shot list.** The real, maintained shot list — beat-by-beat
screen/say/timing, produced by actually running `demo/demo.sh` — lives at
**[`demo/SHOTLIST.md`](../demo/SHOTLIST.md)**. Read that file, not this one, before recording.

For step-by-step recording mechanics (terminal setup, one-time build, how to record, an
asciinema alternative, and what to check after recording), see
**[`demo/README.md`](../demo/README.md)**.

## Why this file still exists

`docs/` and `demo/` are owned by different parts of this project's workflow, and some external
links (the Devpost submission, older commit messages) may still point at `docs/VIDEO.md`. Rather
than break those links or let this file drift into a second, stale copy of the shot list, it
stays as a one-hop redirect.

## The load-bearing facts, if you read nothing else

- The demo's numbers come from **`results/REMEASURE-2026-08-04-QUIET.md`** — the authoritative,
  round-robin-interleaved re-measurement. An earlier draft's numbers (57.3%, 71.6, 45.5, 4.4x,
  198.9, 2257.5, 1145.0) are **retracted**; do not use them in narration, captions, or the video
  description.
- The honest arc is: tuning (`-t 2` decode, `-t 8` prefill) is a real 3.43x / 1.79x win, zero
  code changes. This project's own dispatch patch is a proven-but-not-helpful negative result
  (~12% slower at default threads) — publish that as a strength, not something to hide.
- **Never show a CI green check that has not actually run** on real infrastructure — narrate
  the honest fallback instead if something hasn't been triggered yet by recording time.
- No copyrighted music. Silence or royalty-free/CC0 audio only.

Everything else — narration lines, screen contents, per-beat timing, what to cut if you run
long — is in `demo/SHOTLIST.md`.
