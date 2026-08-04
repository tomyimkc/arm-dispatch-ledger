#!/usr/bin/env python3
"""Capture real per-token arrival timings from both binaries for Polygraph's mid-video race.

Runs the SAME prompt with the SAME seed on the stock llama.cpp build and on the build carrying
patches/0002 (SME2-aware thread default), streaming stdout and timestamping every token as it
actually arrives. Runs alternate A,B,A,B,... so machine contention hits both equally, and the
median-duration run of each is kept.

Output: race.json — the literal measured timings the video replays. Nothing is simulated; the
video's typing speed IS this data.
"""
from __future__ import annotations

import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

BASE = "/tmp/llama.cpp/build/bin/llama-cli"
PATCHED = "/tmp/llama-autodefaults/build/bin/llama-cli"
MODEL = "/tmp/ggufs/q05.gguf"
PROMPT = "Explain what a black hole is, and why not even light can escape one."
NPRED = 260
SEED = 7
REPS = 3


def run_once(binary: str, env_extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [
        binary, "-m", MODEL, "-p", PROMPT, "-n", str(NPRED),
        "-no-cnv", "-st", "--simple-io", "--seed", str(SEED), "-c", "1024",
    ]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, bufsize=1, env=env)
    events, text = [], []
    assert proc.stdout is not None
    while True:
        ch = proc.stdout.read(1)
        if ch == "":
            break
        events.append(round(time.perf_counter() - t0, 4))
        text.append(ch)
    proc.wait(timeout=120)
    return {"chars": "".join(text), "t": events, "wall": round(time.perf_counter() - t0, 3)}


def main() -> int:
    for b in (BASE, PATCHED):
        if not pathlib.Path(b).exists():
            print(f"missing binary: {b}", file=sys.stderr)
            return 1

    runs = {"baseline": [], "patched": []}
    for r in range(REPS):
        runs["baseline"].append(run_once(BASE))
        runs["patched"].append(run_once(PATCHED, {"GGML_KLEIDIAI_AUTO_THREADS": "1"}))
        print(f"  rep {r+1}: baseline {runs['baseline'][-1]['wall']:.2f}s  "
              f"patched {runs['patched'][-1]['wall']:.2f}s", flush=True)

    out = {}
    for k, rs in runs.items():
        med = statistics.median(x["wall"] for x in rs)
        pick = min(rs, key=lambda x: abs(x["wall"] - med))
        # strip the echoed prompt: keep only what the model generated
        body = pick["chars"]
        idx = body.find(PROMPT)
        start = idx + len(PROMPT) if idx >= 0 else 0
        out[k] = {
            "wall": pick["wall"],
            "text": body[start:].lstrip(),
            "t": [round(x - pick["t"][start], 4) for x in pick["t"][start:]] if len(pick["t"]) > start else [],
            "all_walls": [x["wall"] for x in rs],
        }
        print(f"{k}: median {med:.2f}s (kept {pick['wall']:.2f}s), {len(out[k]['text'])} chars")

    b, p = out["baseline"]["wall"], out["patched"]["wall"]
    out["meta"] = {
        "prompt": PROMPT, "n_predict": NPRED, "seed": SEED, "reps": REPS,
        "model": "Qwen2.5-0.5B-Instruct-Q4_0",
        "speedup_wall": round(b / p, 2) if p else None,
        "note": "Real streamed token arrival times. Runs alternated A,B,A,B so contention hit both equally.",
    }
    print(f"\nwall-clock speedup: {out['meta']['speedup_wall']}x  ({b:.2f}s -> {p:.2f}s)")
    dest = pathlib.Path(__file__).resolve().parent.parent / "remotion" / "race.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
