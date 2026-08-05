#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
"""Render docs/media/catch-a-liar.gif from a REAL captured `make demo` run.

The frames are typed out from a literal terminal capture -- nothing here is authored, mocked up,
or re-timed to look better than it is. Regenerate with:

    script -q /tmp/demo_raw.txt make demo
    python3 tools/render_demo_gif.py /tmp/demo_raw.txt docs/media/catch-a-liar.gif

Requires Pillow. This is a documentation tool, not part of the verifier, so it is the one place
in the project allowed a non-stdlib dependency -- `tools/polygraph` itself stays stdlib-only.
"""
from __future__ import annotations

import pathlib
import re
import sys

from PIL import Image, ImageDraw, ImageFont

# Terminal palette. Kept close to the project's dashboard so the GIF and the site look related.
BG = (7, 13, 22)
FG = (232, 240, 251)
DIM = (143, 164, 191)
CYAN = (49, 200, 240)
GREEN = (55, 217, 154)
RED = (242, 97, 95)
AMBER = (240, 176, 58)

FONT_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
FONT_SIZE = 19
PAD = 22
LINE_H = 25
COLS = 108

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")


def load_font():
    for p in FONT_PATHS:
        if pathlib.Path(p).exists():
            try:
                return ImageFont.truetype(p, FONT_SIZE)
            except OSError:
                continue
    return ImageFont.load_default()


def colour_for(line: str):
    """Colour by meaning, not by guessing: only mark what the tool itself asserts."""
    s = line.strip()
    if s.startswith("MISMATCH") or "exit 1" in s or "MISMATCH)" in s:
        return RED
    if s.startswith("MATCH") or "exit 0" in s or "MATCH)" in s:
        return GREEN
    if s.startswith("$") or s.startswith("=="):
        return CYAN
    if s.startswith("#") or s.startswith("  L1") or s.startswith("  L2") or s.startswith("  L3"):
        return DIM
    if s.startswith("verdict:"):
        return AMBER
    return FG


def main() -> int:
    src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo_raw.txt")
    dst = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "docs/media/catch-a-liar.gif")
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = src.read_text(errors="replace")
    lines = []
    for ln in raw.splitlines():
        ln = ANSI.sub("", ln).rstrip()
        if ln.startswith("Script started") or ln.startswith("Script done"):
            continue
        lines.append(ln[:COLS])
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    font = load_font()
    w = PAD * 2 + int(font.getlength("M") * COLS)
    h = PAD * 2 + LINE_H * len(lines)

    frames, durations = [], []
    for shown in range(1, len(lines) + 1):
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines[:shown]):
            d.text((PAD, PAD + i * LINE_H), ln, font=font, fill=colour_for(ln))
        # Cursor on the newest line.
        cur = lines[shown - 1]
        d.text((PAD + font.getlength(cur), PAD + (shown - 1) * LINE_H), "█", font=font, fill=CYAN)
        frames.append(img)
        # Linger on the two verdict lines -- they are the whole point of the demo.
        s = cur.strip()
        durations.append(1400 if (s.startswith("MISMATCH") or s.startswith("MATCH")) else 130)

    final = Image.new("RGB", (w, h), BG)
    fd = ImageDraw.Draw(final)
    for i, ln in enumerate(lines):
        fd.text((PAD, PAD + i * LINE_H), ln, font=font, fill=colour_for(ln))
    frames.append(final)
    durations.append(3500)

    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)
    kb = dst.stat().st_size / 1024
    print(f"wrote {dst}  ({len(frames)} frames, {kb:.0f} KB, {w}x{h})")
    if kb > 4096:
        print("WARNING: over 4 MB; GitHub may not autoplay it inline.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
