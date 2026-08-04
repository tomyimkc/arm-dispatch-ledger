#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Dispatch Ledger contributors
# SPDX-License-Identifier: Apache-2.0
"""crossover.py -- per-phase (prefill vs. decode) crossover measurement harness.

THE QUESTION THIS ANSWERS
--------------------------
`tools/bench.py` already showed *that* SME2 and NEON trade wins depending on
phase (see `results/SUMMARY.md` section 3-4). This script is the dedicated,
narrower instrument that produces the actual BASELINE the optimization claim
in this submission rests on: for each phase (prefill, decode), across the
full `{1,2,4,8,16}` thread sweep and both `GGML_KLEIDIAI_SME` states, what is
the real per-phase optimum (thread count + kernel family), what does the
llama.cpp DEFAULT configuration actually give a user who passes no flags at
all, what is the best SPLIT-PHASE config expressible TODAY with llama.cpp's
existing `-t`/`-tb` flags (different thread counts per phase, but still one
process-global `GGML_KLEIDIAI_SME` setting), and what is the THEORETICAL best
(best prefill cell + best decode cell, independently) that a phase-aware
kernel-family patch should aim to approach but which is **not** expressible
today because `GGML_KLEIDIAI_SME` is a single process-global env var.

WHY JSON, NOT CSV
------------------
A previous ad-hoc attempt in this project failed to parse `llama-bench`'s CSV
output. Before writing any parser here, this script's development ran
`llama-bench -o json` once and inspected the raw output directly (see
tools/crossover.md, "Output format note"). `-o json` returns a single,
uniformly-shaped JSON array with one object per (n_prompt,n_gen,threads,...)
combination, `samples_ts`/`samples_ns` arrays already excluding the internal
warmup pass -- there is no header/units row to misalign, unlike CSV. This
script therefore always invokes `llama-bench -o json` and never touches CSV.

METHODOLOGY (see tools/crossover.md for the full write-up)
------------------------------------------------------------
- Axes: threads in {1,2,4,8,16} x GGML_KLEIDIAI_SME in {unset ("on"), "0"
  ("off")} x phase in {prefill, decode} = 20 cells.
- >=5 repetitions per cell (default 5), each repetition its OWN fresh
  `llama-bench -r 1` process invocation (never `-r 5` in one process for the
  main sweep) so that the 20 cells can be genuinely INTERLEAVED across
  repetitions (round-robin: cell1, cell2, ..., cell20, cell1, cell2, ...)
  instead of measuring all reps of one cell back-to-back -- this is what
  prevents thermal drift over the run from masquerading as a threads/SME
  effect. Within each (phase, threads) pair the two SME states are measured
  back-to-back (on, off) so the most decision-relevant comparison is also the
  most thermally-adjacent one.
- Every repetition still gets `llama-bench`'s own internal warmup pass (not
  `--no-warmup`); `samples_ts`/`samples_ns` in `-o json` output already
  exclude that warmup sample, so "discard warmup" is satisfied by construction
  and never double-counted.
- Reports median/stddev/min/max per cell -- never a bare mean.
- `pmset -g therm` captured before and after the full run and persisted
  verbatim so a reviewer can check for thermal throttling independently.
- The split-phase config (`llama-cli -t <decode_best_threads> -tb
  <prefill_best_threads>`) is measured too, since `-t`/`-tb` are genuinely
  separate flags in llama.cpp today. Because `GGML_KLEIDIAI_SME` is still a
  single process-global setting, this script measures the split-phase config
  under BOTH available SME states (never guesses which is better) and reports
  both, interleaved with each other the same way.
- Never invents or interpolates a cell: a failed measurement is recorded as
  `null`/omitted with the error preserved, not silently dropped or estimated
  from neighbors.

USAGE
-----
    python3 tools/crossover.py --reps 5
    python3 tools/crossover.py --threads 1,2,8 --reps 3   # reduced axis, if the full sweep is too slow

See --help for every knob.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LLAMA_BIN_DIR = Path("/tmp/llama.cpp/build/bin")
DEFAULT_MODEL = Path("/tmp/ggufs/q05.gguf")
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "crossover"

FULL_THREADS = [1, 2, 4, 8, 16]
SME_MODES = ["on", "off"]  # "on" = GGML_KLEIDIAI_SME unset (auto-detect; the real-world default).
                            # "off" = GGML_KLEIDIAI_SME=0 (forced NEON regardless of thread count).

# Phase token counts. Matches tools/bench.py's "decode" / "prefill_long"
# definitions exactly so numbers from the two harnesses are comparable:
#   decode:  n_prompt=0,   n_gen=DECODE_N_GEN   -> ne11 == 1 every step, always
#            below the SME2 hybrid-dispatch gate (ne11 >= 128); see
#            results/GROUND-TRUTH-DISPATCH.md.
#   prefill: n_prompt=PREFILL_N_PROMPT, n_gen=0 -> ne11 == PREFILL_N_PROMPT >=
#            128, i.e. in the regime where the SME2 "hybrid" rescue path can
#            engage even when threads > sme_thread_cap.
DECODE_N_GEN = 32
PREFILL_N_PROMPT = 256

# Word count for the synthetic prompt text file used ONLY for the llama-cli
# split-phase runs (llama-bench's own -p flag needs no text file; it
# generates synthetic prompt tokens internally). 260 repetitions of "word "
# was measured with `llama-tokenize` in this session to tokenize to 262
# tokens on this model's tokenizer -- close to PREFILL_N_PROMPT and safely
# above the >=128 hybrid-dispatch gate. The actual count is re-verified at
# runtime (see tokenize_prompt_file()) and recorded in the output, never
# assumed.
SPLIT_PHASE_PROMPT_WORDS = 260

BRACKET_RE = re.compile(
    r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]"
)


# --------------------------------------------------------------------------
# Small process/subprocess helpers (independent copies of tools/bench.py's
# conventions -- this module deliberately does not import tools/bench.py so
# it stays runnable/self-contained even if that sibling file is mid-edit by
# a concurrent work package).
# --------------------------------------------------------------------------

def run(cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing stdout/stderr as text, stdin detached
    (never inherited -- an inherited non-tty stdin can make some llama.cpp
    tools hang instead of running to completion). Never raises on non-zero
    exit; callers inspect .returncode so one failed cell cannot abort the
    whole sweep."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        return subprocess.run(
            cmd,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return subprocess.CompletedProcess(cmd, returncode=124, stdout=out, stderr=err + "\n[crossover.py] TIMEOUT")


def build_subprocess_env(mode: str) -> Dict[str, str]:
    """Full subprocess environment for a given SME mode.

    "on"  -> GGML_KLEIDIAI_SME left UNSET (explicitly removed from the
             inherited environment in case the calling shell exports it).
             Real-world default: llama.cpp auto-detects SME cores via
             detect_num_smcus() and enables SME/SME2 if it finds any.
    "off" -> GGML_KLEIDIAI_SME=0, forces SME off entirely regardless of
             detection (confirmed by reading kleidiai.cpp's env parsing and
             empirically, see results/GROUND-TRUTH-DISPATCH.md).
    """
    base = os.environ.copy()
    base.pop("GGML_KLEIDIAI_SME", None)
    if mode == "off":
        base["GGML_KLEIDIAI_SME"] = "0"
    elif mode == "on":
        pass
    else:
        raise ValueError(f"unknown sme mode {mode!r}")
    return base


def cpu_brand() -> str:
    if platform.system() == "Darwin":
        cp = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if cp.returncode == 0:
            return cp.stdout.strip()
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "unknown"


def platform_slug() -> str:
    brand = cpu_brand().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", brand).strip("-")
    return slug or platform.machine().lower() or "unknown-platform"


def thermal_snapshot() -> Dict[str, str]:
    if platform.system() == "Darwin":
        cp = run(["pmset", "-g", "therm"], timeout=10)
        return {
            "source": "pmset -g therm",
            "output": cp.stdout.strip() if cp.returncode == 0 else f"[error: {cp.stderr.strip()}]",
        }
    return {"source": "unavailable", "output": f"thermal snapshot not implemented for {platform.system()}"}


def default_llama_n_threads_note() -> str:
    """Ground truth for what llama.cpp's ACTUAL default thread count is on
    this machine, established empirically in this session (not assumed):
    `common_cpu_get_num_physical_cores()` in llama.cpp's common/common.cpp
    queries `hw.perflevel0.physicalcpu` (Apple performance-core count) first,
    falling back to `hw.physicalcpu` only if that sysctl is unavailable. On
    this M4 Max that resolves to 12 (the P-core count), NOT the 16 total
    physical cores (`hw.physicalcpu` alone) -- confirmed by running
    `llama-cli ... -v` and reading its own
    `system_info: n_threads = 12 (n_threads_batch = 12) / 16` log line. See
    tools/crossover.md for the exact command."""
    return (
        "llama.cpp's true no-flags default n_threads on this machine is 12 "
        "(hw.perflevel0.physicalcpu, the P-core count), not the total 16 "
        "physical cores -- verified via `llama-cli -v` this session."
    )


# --------------------------------------------------------------------------
# llama-bench invocation + JSON parsing
# --------------------------------------------------------------------------

class CrossoverError(RuntimeError):
    pass


def find_binary(bin_dir: Path, name: str) -> Path:
    candidate = bin_dir / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    raise CrossoverError(f"{name} not found (or not executable) at {candidate}. Build llama.cpp first.")


def run_llama_bench_once(
    llama_bench: Path,
    model: Path,
    n_prompt: int,
    n_gen: int,
    threads: Optional[int],
    sme_mode: str,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """Invoke llama-bench for exactly one repetition (-r 1) of one
    (n_prompt, n_gen, threads) combination under one SME env, parse its JSON
    output, and return the single result entry. `threads=None` omits -t
    entirely so llama-bench (and llama.cpp) falls back to its own internal
    default thread count -- used for the "default configuration" measurement."""
    cmd = [str(llama_bench), "-m", str(model), "-p", str(n_prompt), "-n", str(n_gen), "-r", "1", "-o", "json"]
    if threads is not None:
        cmd += ["-t", str(threads)]
    env = build_subprocess_env(sme_mode)
    cp = run(cmd, env=env, timeout=timeout)
    if cp.returncode != 0:
        raise CrossoverError(
            f"llama-bench exited {cp.returncode} (threads={threads} sme={sme_mode} "
            f"n_prompt={n_prompt} n_gen={n_gen})\nstdout={cp.stdout[-1500:]}\nstderr={cp.stderr[-1500:]}"
        )
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError as e:
        raise CrossoverError(f"could not parse llama-bench JSON output: {e}\nraw stdout: {cp.stdout[:1500]}")
    if not isinstance(data, list) or not data:
        raise CrossoverError(f"unexpected llama-bench JSON shape: {cp.stdout[:500]}")
    return data[0]


# --------------------------------------------------------------------------
# llama-cli split-phase invocation (-t / -tb are separate flags; llama-bench
# has -t but NOT -tb, so this is the one place this script falls back to
# llama-cli instead of llama-bench, exactly as the brief anticipates).
# --------------------------------------------------------------------------

def build_split_phase_prompt_file(path: Path, words: int = SPLIT_PHASE_PROMPT_WORDS) -> Path:
    path.write_text("word " * words)
    return path


def tokenize_prompt_file(llama_tokenize: Optional[Path], model: Path, prompt_file: Path, timeout: float = 30.0) -> Optional[int]:
    """Best-effort real token count for the split-phase prompt file, via the
    project's own `llama-tokenize` binary. Returns None (never a guess) if
    the binary is missing or the call fails."""
    if llama_tokenize is None:
        return None
    cp = run([str(llama_tokenize), "-m", str(model), "-f", str(prompt_file)], timeout=timeout)
    if cp.returncode != 0:
        return None
    lines = [l for l in cp.stdout.splitlines() if "->" in l]
    return len(lines) if lines else None


def run_llama_cli_split_once(
    llama_cli: Path,
    model: Path,
    prompt_file: Path,
    threads: int,
    threads_batch: int,
    n_gen: int,
    sme_mode: str,
    timeout: float = 90.0,
) -> Dict[str, float]:
    """Invoke llama-cli once with -t <threads> -tb <threads_batch>, parse the
    `[ Prompt: X t/s | Generation: Y t/s ]` summary line it prints on
    completion. Requires -no-cnv -st --simple-io so it processes the prompt
    file once and exits instead of hanging on non-TTY stdin (see the work
    package brief)."""
    cmd = [
        str(llama_cli), "-m", str(model), "-no-cnv", "-st", "--simple-io",
        "-t", str(threads), "-tb", str(threads_batch), "-n", str(n_gen), "-f", str(prompt_file),
    ]
    env = build_subprocess_env(sme_mode)
    cp = run(cmd, env=env, timeout=timeout)
    combined = cp.stdout + "\n" + cp.stderr
    m = BRACKET_RE.search(combined)
    if not m:
        raise CrossoverError(
            f"llama-cli split-phase run produced no '[ Prompt: .. | Generation: .. ]' summary "
            f"(returncode={cp.returncode}, threads={threads}, threads_batch={threads_batch}, sme={sme_mode})\n"
            f"tail of output: {combined[-1500:]}"
        )
    return {"prompt_ts": float(m.group(1)), "gen_ts": float(m.group(2))}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(samples: List[float]) -> Optional[Dict[str, Any]]:
    """median/stddev/min/max (never a bare mean) across raw per-repetition
    tok/s samples, or None if there are zero valid samples."""
    if not samples:
        return None
    return {
        "n": len(samples),
        "median_ts": statistics.median(samples),
        "stddev_ts": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ts": min(samples),
        "max_ts": max(samples),
        "samples_ts": samples,
    }


def cell_key(phase: str, threads: int, sme_mode: str) -> str:
    return f"{phase}|t={threads}|sme={sme_mode}"


def load_average_snapshot() -> Dict[str, Any]:
    """1/5/15-minute load average (os.getloadavg(), POSIX only) plus a
    human-readable `uptime` line. Captured before/after the run for the same
    reason as the thermal snapshot: this machine is shared with other
    concurrent agent sessions (see tools/crossover.md, "Contention note"), and
    a reviewer should be able to see directly whether this run overlapped
    with heavy unrelated CPU load rather than take a clean-room result on
    faith."""
    try:
        one, five, fifteen = os.getloadavg()
        loadavg = {"1min": one, "5min": five, "15min": fifteen}
    except (OSError, AttributeError):
        loadavg = None
    cp = run(["uptime"], timeout=10)
    return {"loadavg": loadavg, "uptime_line": cp.stdout.strip() if cp.returncode == 0 else None}


def call_with_retries(fn, *args, retries: int = 2, backoff_s: float = 3.0, on_retry=None, **kwargs):
    """Call fn(*args, **kwargs); on a CrossoverError caused by a subprocess
    TIMEOUT (returncode 124), retry up to `retries` more times with a short
    backoff before giving up. This exists specifically because this machine
    is shared with other concurrent agent sessions -- a transient CPU-
    contention spike can starve a single llama-bench/llama-cli call past its
    timeout even though the same call reliably succeeds a few seconds later.
    Retrying re-runs the IDENTICAL real subprocess command; it never
    fabricates, estimates, or reuses a stale sample. Non-timeout errors
    (bad JSON, non-zero exit for a real reason, etc.) are NOT retried -- they
    propagate immediately, since retrying those would not change the
    outcome."""
    last_exc: Optional[CrossoverError] = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except CrossoverError as e:
            last_exc = e
            is_timeout = "exited 124" in str(e) or "TIMEOUT" in str(e)
            if not is_timeout or attempt == retries:
                raise
            if on_retry:
                on_retry(attempt, e)
            time.sleep(backoff_s)
    raise last_exc  # pragma: no cover -- unreachable, satisfies type checkers


# --------------------------------------------------------------------------
# Main sweep
# --------------------------------------------------------------------------

def run_main_sweep(
    llama_bench: Path,
    model: Path,
    threads_list: List[int],
    sme_modes: List[str],
    reps: int,
    per_call_timeout: float,
    quiet: bool,
    decode_n_gen: int = DECODE_N_GEN,
    prefill_n_prompt: int = PREFILL_N_PROMPT,
    retry_log: Optional[List[Dict[str, Any]]] = None,
    retries: int = 2,
) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
    """Interleaved (round-robin) measurement of the full
    phase x threads x sme_mode grid. Returns (samples_by_cell, errors)."""
    phases = [
        ("decode", 0, decode_n_gen),
        ("prefill", prefill_n_prompt, 0),
    ]
    # Fixed cell order, repeated identically every round: for each phase, for
    # each thread count, the two SME states back-to-back (the most
    # decision-relevant, most thermally-adjacent comparison), rotating across
    # thread counts and phases within a round so no single cell is
    # systematically favoured by warm-up/thermal position across the run.
    cells: List[Tuple[str, int, int, int, str]] = []
    for phase_name, n_prompt, n_gen in phases:
        for threads in threads_list:
            for sme_mode in sme_modes:
                cells.append((phase_name, n_prompt, n_gen, threads, sme_mode))

    samples: Dict[str, List[float]] = {cell_key(p, t, s): [] for (p, _, _, t, s) in cells}
    errors: List[Dict[str, Any]] = []
    total_calls = len(cells) * reps
    call_i = 0
    for round_i in range(reps):
        for (phase_name, n_prompt, n_gen, threads, sme_mode) in cells:
            call_i += 1
            if not quiet:
                print(
                    f"[crossover.py] round {round_i + 1}/{reps} call {call_i}/{total_calls}: "
                    f"phase={phase_name} threads={threads} sme={sme_mode}",
                    file=sys.stderr,
                )
            try:
                def _on_retry(attempt, e, _p=phase_name, _t=threads, _s=sme_mode):
                    print(f"[crossover.py]   retry {attempt + 1} after timeout for phase={_p} threads={_t} sme={_s} "
                          f"(shared-machine contention, see tools/crossover.md)", file=sys.stderr)
                    if retry_log is not None:
                        retry_log.append({"context": "main_sweep", "phase": _p, "threads": _t, "sme_mode": _s, "attempt": attempt + 1})
                result = call_with_retries(
                    run_llama_bench_once, llama_bench, model, n_prompt, n_gen, threads, sme_mode,
                    timeout=per_call_timeout, retries=retries, on_retry=_on_retry,
                )
                ts_values = result.get("samples_ts") or []
                if not ts_values:
                    raise CrossoverError(f"llama-bench returned zero samples_ts: {result}")
                samples[cell_key(phase_name, threads, sme_mode)].extend(float(v) for v in ts_values)
            except CrossoverError as e:
                print(f"[crossover.py] WARNING: measurement failed for {phase_name} t={threads} sme={sme_mode}: {e}", file=sys.stderr)
                errors.append({"phase": phase_name, "threads": threads, "sme_mode": sme_mode, "round": round_i, "error": str(e)})
    return samples, errors


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def render_markdown(meta: Dict[str, Any], sweep_rows: List[Dict[str, Any]], default_rows: List[Dict[str, Any]],
                     split_rows: List[Dict[str, Any]], optima: Dict[str, Any], theoretical: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Crossover harness results -- {meta['platform']}")
    lines.append("")
    lines.append(f"- Generated: {meta['generated_at']}")
    lines.append(f"- CPU: {meta['cpu_brand']}")
    lines.append(f"- llama.cpp bin dir: `{meta['llama_bin_dir']}`  (commit: {meta.get('llama_build_commit', 'unknown')})")
    lines.append(f"- Model: `{meta['model']}`")
    lines.append(f"- Reps per cell: {meta['reps']} (interleaved round-robin across all {meta['n_cells']} cells, "
                 f"never all reps of one cell back-to-back)")
    lines.append(f"- {default_llama_n_threads_note()}")
    lines.append("- Full methodology: see `tools/crossover.md`. Median/stddev/min/max only, computed across "
                 "independently-launched, interleaved `llama-bench -r 1` process invocations -- never a bare mean.")
    lines.append("")

    therm_before = meta.get("thermal", {}).get("before", {}).get("output", "")
    therm_after = meta.get("thermal", {}).get("after", {}).get("output", "")
    lines.append("## Thermal context (`pmset -g therm`)")
    lines.append("")
    lines.append("Before:")
    lines.append("```")
    lines.append(therm_before or "(unavailable)")
    lines.append("```")
    lines.append("After:")
    lines.append("```")
    lines.append(therm_after or "(unavailable)")
    lines.append("```")
    lines.append("")

    load_before = meta.get("load_average", {}).get("before", {})
    load_after = meta.get("load_average", {}).get("after", {})
    n_retries = meta.get("n_retries_used", 0)
    lines.append("## Contention note (shared machine)")
    lines.append("")
    lines.append(
        f"- Load average before: `{load_before.get('uptime_line', '(unavailable)')}`"
    )
    lines.append(
        f"- Load average after: `{load_after.get('uptime_line', '(unavailable)')}`"
    )
    lines.append(
        f"- {n_retries} of this run's llama-bench/llama-cli calls needed a timeout-retry "
        f"(see `retry_log` in the JSON output for exactly which)."
    )
    if (load_before.get("loadavg") or {}).get("1min", 0) > 16 or (load_after.get("loadavg") or {}).get("1min", 0) > 16:
        lines.append(
            "- **This machine's 1-minute load average exceeded its physical core count (16) during this "
            "run** -- this host is shared with other concurrent, unrelated agent sessions (observed: "
            "several `python -m contest_bench...` multiprocessing workers and other Claude Code sessions "
            "competing for the same cores). Absolute tok/s numbers below may be measurably suppressed "
            "relative to a quiet machine, and cell-to-cell variance (stddev) may be inflated by contention "
            "bursts, not just this workload's own thread-count/kernel-family behaviour. Every call that hit "
            "its timeout was retried (never estimated or interpolated) against the same real binary; a cell "
            "is reported only from calls that actually completed. Relative comparisons (which thread count / "
            "kernel family wins within a phase) are expected to be more robust to this than absolute "
            "magnitudes, since contention affects all configurations of a given call, not selectively -- but "
            "this has not been independently verified by re-running on a quiet machine, so treat this as a "
            "caveat, not a correction factor."
        )
    lines.append("")

    lines.append("## (a) Per-phase optimum: full sweep, threads x SME x phase")
    lines.append("")
    lines.append("| phase | threads | SME | median tok/s | stddev | min | max | n |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for r in sweep_rows:
        agg = r.get("agg")
        if agg is None:
            lines.append(f"| {r['phase']} | {r['threads']} | {r['sme_mode']} | _[measurement failed]_ | | | | |")
            continue
        star = " **<-- optimum**" if r.get("is_optimum") else ""
        lines.append(
            f"| {r['phase']} | {r['threads']} | {r['sme_mode']} | {agg['median_ts']:.1f}{star} | "
            f"{agg['stddev_ts']:.2f} | {agg['min_ts']:.1f} | {agg['max_ts']:.1f} | {agg['n']} |"
        )
    lines.append("")
    for phase_name in ("decode", "prefill"):
        opt = optima.get(phase_name)
        if opt:
            lines.append(
                f"- **{phase_name} optimum:** threads={opt['threads']}, SME={opt['sme_mode']} "
                f"({'SME2/hybrid dispatch region' if opt['sme_mode'] == 'on' else 'NEON forced'}), "
                f"median {opt['median_ts']:.1f} tok/s."
            )
    lines.append("")
    lines.append("Cells not listed/measured must not be treated as zero, equal-to-neighbor, or interpolated.")
    lines.append("")

    lines.append("## (b) llama.cpp DEFAULT configuration (no -t/-tb flags, SME unset)")
    lines.append("")
    lines.append("| phase | median tok/s | stddev | min | max | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in default_rows:
        agg = r.get("agg")
        if agg is None:
            lines.append(f"| {r['phase']} | _[measurement failed]_ | | | | |")
            continue
        lines.append(f"| {r['phase']} | {agg['median_ts']:.1f} | {agg['stddev_ts']:.2f} | {agg['min_ts']:.1f} | {agg['max_ts']:.1f} | {agg['n']} |")
    lines.append("")
    lines.append("This is what a user who runs llama-cli/llama-bench with zero thread flags actually gets today.")
    lines.append("")

    lines.append("## (c) Best hand-tuned split-phase config (llama-cli -t/-tb), TODAY, no patch")
    lines.append("")
    if optima.get("decode") and optima.get("prefill"):
        lines.append(
            f"`-t {optima['decode']['threads']}` (this sweep's decode optimum thread count) "
            f"`-tb {optima['prefill']['threads']}` (this sweep's prefill optimum thread count). "
            f"`GGML_KLEIDIAI_SME` is still a single process-global setting, so both available states were measured."
        )
    lines.append("")
    lines.append("| GGML_KLEIDIAI_SME | prompt (prefill) tok/s | median | stddev | min | max | n | generation (decode) tok/s | median | stddev | min | max | n |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in split_rows:
        pa, ga = r.get("prompt_agg"), r.get("gen_agg")
        if pa is None or ga is None:
            lines.append(f"| {r['sme_mode']} | _[measurement failed]_ | | | | | | | | | | |")
            continue
        lines.append(
            f"| {r['sme_mode']} | | {pa['median_ts']:.1f} | {pa['stddev_ts']:.2f} | {pa['min_ts']:.1f} | {pa['max_ts']:.1f} | {pa['n']} | "
            f"| {ga['median_ts']:.1f} | {ga['stddev_ts']:.2f} | {ga['min_ts']:.1f} | {ga['max_ts']:.1f} | {ga['n']} |"
        )
    lines.append("")

    lines.append("## (d) THEORETICAL best -- best prefill cell + best decode cell (NOT YET ACHIEVABLE TODAY)")
    lines.append("")
    if theoretical.get("achievable_today") is True:
        lines.append(
            "In this run the decode optimum and the prefill optimum happen to share the same "
            "`GGML_KLEIDIAI_SME` state, so this theoretical target IS reachable today with the split-phase "
            "config in section (c) above -- re-check this if the axes/thread counts change."
        )
    else:
        lines.append(
            f"[NOT YET ACHIEVABLE] decode wants threads={theoretical.get('decode_threads')}, "
            f"SME={theoretical.get('decode_sme')} (median {theoretical.get('decode_ts', 0):.1f} tok/s); "
            f"prefill wants threads={theoretical.get('prefill_threads')}, "
            f"SME={theoretical.get('prefill_sme')} (median {theoretical.get('prefill_ts', 0):.1f} tok/s). "
            "These require DIFFERENT `GGML_KLEIDIAI_SME` process-global states simultaneously, which "
            "llama.cpp cannot express today (`GGML_KLEIDIAI_SME` is read once at process start; there is no "
            "per-call or per-phase override). This pairing is the TARGET a phase-aware kernel-family-selection "
            "patch should approach, not a number this harness (or any unpatched llama.cpp invocation) can "
            "itself produce."
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str, allowed: List[str]) -> List[str]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    for v in vals:
        if v not in allowed:
            raise argparse.ArgumentTypeError(f"unknown value {v!r}, expected one of {allowed}")
    return vals


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-bin-dir", type=Path, default=DEFAULT_LLAMA_BIN_DIR)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=parse_int_list, default=FULL_THREADS,
                    help=f"comma-separated thread counts (default: {FULL_THREADS})")
    ap.add_argument("--sme-modes", type=lambda s: parse_str_list(s, SME_MODES), default=SME_MODES)
    ap.add_argument("--reps", type=int, default=5, help="repetitions per cell (default 5, minimum recommended 5)")
    ap.add_argument("--decode-n-gen", type=int, default=DECODE_N_GEN)
    ap.add_argument("--prefill-n-prompt", type=int, default=PREFILL_N_PROMPT)
    ap.add_argument("--per-call-timeout", type=float, default=180.0,
                    help="per llama-bench/llama-cli call timeout in seconds (default 180 -- generous "
                         "because this machine is shared with other concurrent agent sessions; each "
                         "call is also retried on a timeout, see --retries / call_with_retries())")
    ap.add_argument("--retries", type=int, default=2,
                    help="retries per call on a timeout before giving up on that repetition (default 2; "
                         "lower this + --per-call-timeout to bound worst-case wall time per cell when "
                         "the shared machine is heavily contended -- see tools/crossover.md section 5)")
    ap.add_argument("--platform", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--skip-split-phase", action="store_true", help="skip the llama-cli -t/-tb split-phase measurement")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        llama_bench = find_binary(args.llama_bin_dir, "llama-bench")
        llama_cli = find_binary(args.llama_bin_dir, "llama-cli")
    except CrossoverError as e:
        print(f"[crossover.py] FATAL: {e}", file=sys.stderr)
        return 2

    try:
        llama_tokenize = find_binary(args.llama_bin_dir, "llama-tokenize")
    except CrossoverError:
        llama_tokenize = None

    if not args.model.is_file():
        print(f"[crossover.py] FATAL: model not found at {args.model}", file=sys.stderr)
        return 2

    plat = args.platform or platform_slug()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    retry_log: List[Dict[str, Any]] = []
    started_at = datetime.datetime.now(datetime.timezone.utc)
    t0 = time.monotonic()
    thermal_before = thermal_snapshot()
    load_before = load_average_snapshot()
    print(f"[crossover.py] load average before: {load_before}", file=sys.stderr)

    print(f"[crossover.py] starting main sweep: threads={args.threads} sme_modes={args.sme_modes} reps={args.reps} "
          f"({len(args.threads) * len(args.sme_modes) * 2} cells, {len(args.threads) * len(args.sme_modes) * 2 * args.reps} llama-bench calls)",
          file=sys.stderr)
    samples, sweep_errors = run_main_sweep(
        llama_bench, args.model, args.threads, args.sme_modes, args.reps, args.per_call_timeout, args.quiet,
        decode_n_gen=args.decode_n_gen, prefill_n_prompt=args.prefill_n_prompt, retry_log=retry_log,
        retries=args.retries,
    )

    sweep_rows: List[Dict[str, Any]] = []
    optima: Dict[str, Any] = {}
    for phase_name in ("decode", "prefill"):
        best_for_phase: Optional[Dict[str, Any]] = None
        for threads in args.threads:
            for sme_mode in args.sme_modes:
                key = cell_key(phase_name, threads, sme_mode)
                agg = aggregate(samples.get(key, []))
                row = {"phase": phase_name, "threads": threads, "sme_mode": sme_mode, "agg": agg}
                sweep_rows.append(row)
                if agg is not None:
                    if best_for_phase is None or agg["median_ts"] > best_for_phase["median_ts"]:
                        best_for_phase = {"threads": threads, "sme_mode": sme_mode, "median_ts": agg["median_ts"]}
        if best_for_phase:
            optima[phase_name] = best_for_phase
    for row in sweep_rows:
        opt = optima.get(row["phase"])
        row["is_optimum"] = bool(opt and row["agg"] is not None and row["threads"] == opt["threads"] and row["sme_mode"] == opt["sme_mode"])

    print(f"[crossover.py] main sweep done in {time.monotonic() - t0:.1f}s; optima: {optima}", file=sys.stderr)

    # (b) default configuration: no -t/-tb flags at all, SME unset.
    print("[crossover.py] measuring (b) default configuration (no thread flags, SME unset)...", file=sys.stderr)
    default_samples: Dict[str, List[float]] = {"decode": [], "prefill": []}
    default_errors: List[Dict[str, Any]] = []
    default_phase_specs = [("decode", 0, args.decode_n_gen), ("prefill", args.prefill_n_prompt, 0)]
    for round_i in range(args.reps):
        for phase_name, n_prompt, n_gen in default_phase_specs:
            try:
                def _on_retry_default(a, e, _p=phase_name):
                    print(f"[crossover.py]   retry {a + 1} after timeout for default {_p}", file=sys.stderr)
                    retry_log.append({"context": "default_config", "phase": _p, "attempt": a + 1})
                result = call_with_retries(
                    run_llama_bench_once, llama_bench, args.model, n_prompt, n_gen, None, "on",
                    timeout=args.per_call_timeout, retries=args.retries, on_retry=_on_retry_default,
                )
                default_samples[phase_name].extend(float(v) for v in (result.get("samples_ts") or []))
            except CrossoverError as e:
                print(f"[crossover.py] WARNING: default-config measurement failed for {phase_name}: {e}", file=sys.stderr)
                default_errors.append({"phase": phase_name, "round": round_i, "error": str(e)})
    default_rows = [{"phase": p, "agg": aggregate(default_samples[p])} for p in ("decode", "prefill")]

    # (c) split-phase config via llama-cli.
    split_rows: List[Dict[str, Any]] = []
    theoretical: Dict[str, Any] = {}
    if optima.get("decode") and optima.get("prefill"):
        theoretical = {
            "decode_threads": optima["decode"]["threads"], "decode_sme": optima["decode"]["sme_mode"], "decode_ts": optima["decode"]["median_ts"],
            "prefill_threads": optima["prefill"]["threads"], "prefill_sme": optima["prefill"]["sme_mode"], "prefill_ts": optima["prefill"]["median_ts"],
            "achievable_today": optima["decode"]["sme_mode"] == optima["prefill"]["sme_mode"],
        }
    if not args.skip_split_phase and optima.get("decode") and optima.get("prefill"):
        prompt_file = args.out_dir / f"_split-phase-prompt-{plat}.txt"
        build_split_phase_prompt_file(prompt_file)
        real_token_count = tokenize_prompt_file(llama_tokenize, args.model, prompt_file)
        print(f"[crossover.py] split-phase prompt file: {prompt_file} ({SPLIT_PHASE_PROMPT_WORDS} words, "
              f"real tokenized length = {real_token_count if real_token_count is not None else '[not independently verified]'})",
              file=sys.stderr)
        split_samples: Dict[str, Dict[str, List[float]]] = {m: {"prompt_ts": [], "gen_ts": []} for m in SME_MODES}
        split_errors: List[Dict[str, Any]] = []
        decode_t = optima["decode"]["threads"]
        prefill_t = optima["prefill"]["threads"]
        # One untimed warmup call per SME state first (discarded), then
        # `reps` timed, interleaved (on, off, on, off, ...) repetitions.
        def _on_retry_split(a, e, _s=None):
            print(f"[crossover.py]   retry {a + 1} after timeout for split-phase run", file=sys.stderr)
            retry_log.append({"context": "split_phase", "attempt": a + 1})
        for sme_mode in SME_MODES:
            try:
                call_with_retries(run_llama_cli_split_once, llama_cli, args.model, prompt_file, decode_t, prefill_t,
                                   args.decode_n_gen, sme_mode, timeout=args.per_call_timeout, retries=args.retries, on_retry=_on_retry_split)
            except CrossoverError as e:
                print(f"[crossover.py] WARNING: split-phase warmup failed for sme={sme_mode}: {e}", file=sys.stderr)
        for round_i in range(args.reps):
            for sme_mode in SME_MODES:
                try:
                    r = call_with_retries(run_llama_cli_split_once, llama_cli, args.model, prompt_file, decode_t, prefill_t,
                                           args.decode_n_gen, sme_mode, timeout=args.per_call_timeout, retries=args.retries, on_retry=_on_retry_split)
                    split_samples[sme_mode]["prompt_ts"].append(r["prompt_ts"])
                    split_samples[sme_mode]["gen_ts"].append(r["gen_ts"])
                except CrossoverError as e:
                    print(f"[crossover.py] WARNING: split-phase run failed (round={round_i}, sme={sme_mode}): {e}", file=sys.stderr)
                    split_errors.append({"sme_mode": sme_mode, "round": round_i, "error": str(e)})
        for sme_mode in SME_MODES:
            split_rows.append({
                "sme_mode": sme_mode,
                "decode_threads": decode_t,
                "prefill_threads": prefill_t,
                "prompt_agg": aggregate(split_samples[sme_mode]["prompt_ts"]),
                "gen_agg": aggregate(split_samples[sme_mode]["gen_ts"]),
            })
        try:
            prompt_file.unlink()
        except OSError:
            pass
    else:
        split_errors = []
        real_token_count = None

    thermal_after = thermal_snapshot()
    load_after = load_average_snapshot()
    print(f"[crossover.py] load average after: {load_after}", file=sys.stderr)
    finished_at = datetime.datetime.now(datetime.timezone.utc)
    elapsed_s = time.monotonic() - t0

    # llama.cpp build commit, best-effort (one cheap extra call; not stored per-row above).
    llama_build_commit = "unknown"
    try:
        probe = run_llama_bench_once(llama_bench, args.model, 0, 1, 1, "on", timeout=30.0)
        llama_build_commit = probe.get("build_commit", "unknown")
    except CrossoverError:
        pass

    meta = {
        "platform": plat,
        "cpu_brand": cpu_brand(),
        "generated_at": finished_at.isoformat(),
        "started_at": started_at.isoformat(),
        "elapsed_s": round(elapsed_s, 1),
        "llama_bin_dir": str(args.llama_bin_dir),
        "llama_build_commit": llama_build_commit,
        "model": str(args.model),
        "threads_axis": args.threads,
        "sme_modes_axis": args.sme_modes,
        "reps": args.reps,
        "decode_n_gen": args.decode_n_gen,
        "prefill_n_prompt": args.prefill_n_prompt,
        "n_cells": len(args.threads) * len(args.sme_modes) * 2,
        "default_thread_note": default_llama_n_threads_note(),
        "split_phase_prompt_words": SPLIT_PHASE_PROMPT_WORDS,
        "split_phase_prompt_real_token_count": real_token_count,
        "thermal": {"before": thermal_before, "after": thermal_after},
        "load_average": {"before": load_before, "after": load_after},
        "n_retries_used": len(retry_log),
    }

    out = {
        "meta": meta,
        "sweep_rows": sweep_rows,
        "sweep_errors": sweep_errors,
        "optima": optima,
        "default_rows": default_rows,
        "default_errors": default_errors,
        "split_phase_rows": split_rows,
        "split_phase_errors": split_errors,
        "theoretical_best": theoretical,
        "retry_log": retry_log,
    }

    json_path = args.out_dir / f"crossover-{plat}.json"
    json_path.write_text(json.dumps(out, indent=2, sort_keys=False))

    md = render_markdown(meta, sweep_rows, default_rows, split_rows, optima, theoretical)
    md_path = args.out_dir / f"crossover-{plat}.md"
    md_path.write_text(md)

    print(f"[crossover.py] wrote {json_path}", file=sys.stderr)
    print(f"[crossover.py] wrote {md_path}", file=sys.stderr)
    print(f"[crossover.py] total elapsed: {elapsed_s:.1f}s", file=sys.stderr)
    n_errors = len(sweep_errors) + len(default_errors) + len(split_errors)
    if n_errors:
        print(f"[crossover.py] WARNING: {n_errors} measurement(s) failed; see *_errors in the JSON output.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
