#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Dispatch Ledger contributors
# SPDX-License-Identifier: Apache-2.0
"""plot_results.py -- render bench.py's JSON output as figures.

Reads a `results/bench-<platform>.json` file produced by `tools/bench.py` and
writes PNGs next to it (or to --out-dir):

  * one grouped-bar figure per phase (decode / prefill_short / prefill_long):
    tok/s vs threads, SME-on vs SME-off, with error bars from the
    min/max/stddev bench.py already computed (never re-derives a mean itself);
  * one dispatch-annotation table image (or, if matplotlib is unavailable,
    a markdown table) showing which kernel family actually fired for every
    cell, so a chart can never be shown without its dispatch label next to it.

This script does not run any benchmark itself and does not invent any data
point absent from the input JSON -- a (phase, threads, sme) cell with no
"agg" (i.e. not measured, or measurement failed) is rendered as a visibly
empty gap in the chart, never interpolated or zero-filled.

USAGE
-----
    python3 tools/plot_results.py results/bench-<platform>.json
    python3 tools/plot_results.py results/bench-<platform>.json --out-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # headless -- this runs on CI/SSH boxes with no display
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def load_results(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _dispatch_short_label(row: Dict[str, Any]) -> str:
    d = row.get("dispatch") or {}
    if not d.get("verified", False):
        return "?"
    if d.get("hybrid"):
        return "HYBRID"
    if d.get("sme_fires") and d.get("neon_fires"):
        return "SME2+NEON"
    if d.get("sme_fires"):
        return "SME2"
    if d.get("neon_fires"):
        return "NEON"
    return "?"


def plot_phase(phase: str, rows: List[Dict[str, Any]], out_path: Path, quant: str) -> bool:
    """One grouped-bar chart for a given phase: threads on the x-axis,
    SME-on vs SME-off as two bar series, median tok/s as height, min/max as
    the error-bar whiskers. Returns True if a figure was written."""
    phase_rows = [r for r in rows if r.get("phase") == phase and r.get("quant") == quant and not r.get("not_available")]
    if not phase_rows:
        return False

    threads_sorted = sorted({r["threads"] for r in phase_rows})
    if not threads_sorted:
        return False

    def cell(th: int, sme: str) -> Optional[Dict[str, Any]]:
        for r in phase_rows:
            if r["threads"] == th and r["sme_mode"] == sme:
                return r
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.35
    x = range(len(threads_sorted))

    for offset, sme, color, label in ((-width / 2, "on", "#2b6cb0", "SME on (auto)"),
                                       (width / 2, "off", "#c05621", "SME off (NEON only)")):
        medians = []
        err_lo = []
        err_hi = []
        annotations = []
        for th in threads_sorted:
            r = cell(th, sme)
            agg = (r or {}).get("agg")
            if agg is None:
                medians.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
                annotations.append("no data")
                continue
            med = agg["median_ts"]
            medians.append(med)
            err_lo.append(max(0.0, med - agg["min_ts"]))
            err_hi.append(max(0.0, agg["max_ts"] - med))
            annotations.append(_dispatch_short_label(r))

        xs = [xi + offset for xi in x]
        bars = ax.bar(xs, medians, width=width, yerr=[err_lo, err_hi],
                       capsize=3, color=color, label=label, alpha=0.9)
        for bar, ann, med in zip(bars, annotations, medians):
            if med <= 0:
                continue
            ax.annotate(ann, (bar.get_x() + bar.get_width() / 2, med),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, rotation=0)

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(t) for t in threads_sorted])
    ax.set_xlabel("threads (-t)")
    ax.set_ylabel("tok/s (median, whiskers = min/max)")
    ax.set_title(f"{phase} -- {quant} -- dispatch label shown above each bar")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def render_dispatch_table_md(rows: List[Dict[str, Any]]) -> str:
    lines = ["| phase | quant | threads | SME | dispatch |", "|---|---|---:|---|---|"]
    for r in rows:
        if r.get("not_available"):
            lines.append(f"| {r['phase']} | {r['quant']} | {r['threads']} | {r['sme_mode']} | _not available_ |")
            continue
        d = r.get("dispatch") or {}
        if not d.get("verified", False):
            label = f"unverified ({d.get('reason', 'unknown')})"
        elif d.get("hybrid"):
            label = f"HYBRID (SME2 x{d.get('sme_hits','?')} + NEON x{d.get('neon_hits','?')})"
        else:
            label = _dispatch_short_label(r)
        lines.append(f"| {r['phase']} | {r['quant']} | {r['threads']} | {r['sme_mode']} | {label} |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_json", type=Path, help="path to a bench-<platform>.json file produced by tools/bench.py")
    ap.add_argument("--out-dir", type=Path, default=None, help="directory for output figures (default: alongside the input JSON)")
    args = ap.parse_args(argv)

    data = load_results(args.results_json)
    rows = data.get("rows", [])
    meta = data.get("meta", {})
    plat = meta.get("platform", args.results_json.stem.replace("bench-", ""))

    out_dir = args.out_dir or args.results_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    quants = sorted({r["quant"] for r in rows})
    phases = sorted({r["phase"] for r in rows})

    if not HAVE_MPL:
        print("[plot_results.py] matplotlib not available -- writing a markdown dispatch "
              "table only, no PNG figures. Install matplotlib to get charts.", file=sys.stderr)
        table_path = out_dir / f"bench-{plat}-dispatch-table.md"
        table_path.write_text(render_dispatch_table_md(rows))
        print(f"[plot_results.py] wrote {table_path}")
        return 0

    wrote_any = False
    for quant in quants:
        for phase in phases:
            out_path = out_dir / f"bench-{plat}-{phase}-{quant}.png"
            if plot_phase(phase, rows, out_path, quant):
                print(f"[plot_results.py] wrote {out_path}")
                wrote_any = True
            else:
                print(f"[plot_results.py] no measured data for phase={phase} quant={quant}, skipped", file=sys.stderr)

    table_path = out_dir / f"bench-{plat}-dispatch-table.md"
    table_path.write_text(render_dispatch_table_md(rows))
    print(f"[plot_results.py] wrote {table_path}")

    if not wrote_any:
        print("[plot_results.py] WARNING: no figures were produced (no measured cells found in input)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
