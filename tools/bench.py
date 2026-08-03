#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Arm Dispatch Ledger contributors
# SPDX-License-Identifier: Apache-2.0
"""bench.py -- SME2-vs-NEON dispatch benchmark harness for llama.cpp / KleidiAI.

THE QUESTION THIS ANSWERS
--------------------------
On Apple Silicon, llama.cpp (built with -DGGML_CPU_KLEIDIAI=ON) can execute a
quantized matmul through one of two kernel families:

  * SME2, capped at `sme_thread_cap` threads (2 on M4 Max/Ultra/Pro) UNLESS a
    "hybrid" rescue path also engages for large batches (see
    results/GROUND-TRUTH-DISPATCH.md for the exact dispatch rule read from
    ggml/src/ggml-cpu/kleidiai/kleidiai.cpp).
  * NEON (dotprod/i8mm), unconstrained in thread count.

Nobody has published which one wins, at which thread count, in which phase
(prefill vs decode). That is the headline measurement this script produces.

Full methodology, anti-"you faked it" measures, and the traps this harness
specifically avoids are documented in tools/protocol.md -- read that first if
a number here looks surprising. This module intentionally duplicates none of
that prose; only executable decisions live here.

USAGE
-----
    python3 tools/bench.py --threads 1,2,8 --reps 5

See --help for every knob. Never reports a bare mean (median/stddev/min/max
only), never blocks all repetitions of one configuration before another
(interleaved A,B,C,A,B,C,... order), never reports a tok/s number without a
dispatch label, and never invents a number for a configuration it did not
actually run (missing cells are simply absent from the output, or explicitly
marked "not available").
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import platform
import re
import shutil
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
DEFAULT_MODEL_Q4_0 = Path("/tmp/ggufs/q05.gguf")
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# Reduced sweep, per the project brief: if the full sweep is slow, run this
# subset for real and mark the rest as not-yet-measured. Callers can widen it
# with --threads.
DEFAULT_THREADS = [1, 2, 8]
FULL_THREADS = [1, 2, 4, 8, 16]

SME_MODES = ["on", "off"]  # "on" == env var unset (real-world default / auto-detect)

# Phase definitions: (n_prompt, n_gen). Decode uses n_prompt=0 so llama-bench's
# "prompt" test is skipped entirely and only the "generation" test runs, and
# vice versa for the prefill phases. This is how prefill and decode tok/s end
# up measured SEPARATELY, as required -- they have different arithmetic
# intensity (prefill is a batched GEMM; decode is a sequence of GEMVs with
# ne11 == 1 forever, which is why SME2's hybrid rescue path can never help it;
# see results/GROUND-TRUTH-DISPATCH.md).
def default_phases(decode_tokens: int, prefill_short_tokens: int, prefill_long_tokens: int) -> Dict[str, Tuple[int, int]]:
    return {
        # ne11 == 1 every step: below the hybrid gate (ne11 >= 128) always.
        "decode": (0, decode_tokens),
        # ne11 == prefill_short_tokens < 128: below the hybrid gate.
        "prefill_short": (prefill_short_tokens, 0),
        # ne11 == prefill_long_tokens >= 128: above the hybrid gate, so SME2
        # can dispatch even when threads > sme_thread_cap (hybrid mode).
        "prefill_long": (prefill_long_tokens, 0),
    }


KAI_SME_REGEX = r"kai_run_matmul.*_(sme|sme2)_"
KAI_NEON_REGEX = r"kai_run_matmul.*_neon_(dotprod|i8mm)"


# --------------------------------------------------------------------------
# Small process/subprocess helpers
# --------------------------------------------------------------------------

def run(cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run a subprocess, capturing stdout/stderr as text. Never raises on a
    non-zero exit -- callers inspect .returncode themselves so a single failed
    configuration cannot take down the whole sweep."""
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
            # Explicitly detach stdin (rather than inheriting whatever the
            # parent's stdin happens to be). Found the hard way: when bench.py
            # itself runs under a backgrounded/non-tty shell, an inherited
            # stdin can make `lldb -b ... -- <target>` block indefinitely on
            # process launch instead of running to completion or hitting a
            # breakpoint -- a tooling hang, not a dispatch fact. See
            # tools/protocol.md section 6.7.
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        # NOTE: even with text=True, subprocess.run() can hand back *bytes*
        # for e.stdout/e.stderr on a timeout (a documented CPython quirk --
        # the partial buffer captured before the decode step). Normalize to
        # str here so every caller can treat cp.stdout/.stderr uniformly.
        out = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        cp = subprocess.CompletedProcess(cmd, returncode=124, stdout=out, stderr=err + "\n[bench.py] TIMEOUT")
        return cp


def build_subprocess_env(mode: str) -> Dict[str, str]:
    """Return the full subprocess environment for a given SME mode.

    "on"  -> GGML_KLEIDIAI_SME is left UNSET (explicitly removed from the
             inherited environment, in case the calling shell exports it).
             This is the real-world default: llama.cpp auto-detects SME cores
             via detect_num_smcus() (a hardcoded Apple brand-string table) and
             enables SME/SME2 if it finds any. This is what a user gets who
             never read the source.
    "off" -> GGML_KLEIDIAI_SME=0, which forces SME off entirely regardless of
             detection, confirmed by reading kleidiai.cpp's env parsing (env=0
             -> sme_cores = 0) and confirmed empirically (see protocol.md).
    """
    base = os.environ.copy()
    base.pop("GGML_KLEIDIAI_SME", None)
    if mode == "off":
        base["GGML_KLEIDIAI_SME"] = "0"
    elif mode == "on":
        pass  # left unset -> runtime auto-detection, the real default
    else:
        raise ValueError(f"unknown sme mode {mode!r}")
    return base


# --------------------------------------------------------------------------
# Platform / thermal context
# --------------------------------------------------------------------------

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
    """Best-effort thermal/throttle context. macOS: `pmset -g therm`. Anything
    else: explicitly note it was not captured rather than guessing."""
    if platform.system() == "Darwin":
        cp = run(["pmset", "-g", "therm"], timeout=10)
        return {
            "source": "pmset -g therm",
            "output": cp.stdout.strip() if cp.returncode == 0 else f"[error: {cp.stderr.strip()}]",
        }
    return {"source": "unavailable", "output": f"thermal snapshot not implemented for {platform.system()}"}


def lldb_preflight_hint() -> Optional[str]:
    """Best-effort, non-fatal early warning for a specific real failure mode
    hit while preparing this project (see tools/protocol.md section 6 item 9):
    on macOS, `lldb` requires the host's "Developer Mode" to be enabled to
    actually attach/instrument a target. When it is disabled, `lldb` does NOT
    fail fast -- it hangs while trying to instrument the process (observed:
    ~18-19s of apparent activity, every breakpoint still at hit count 0,
    versus <1s for the identical command with no debugger at all) until this
    script's own per-call timeout kills it. That produces a full sweep's
    worth of `unverified (fast-tier lldb timed out ...)` rows, discovered
    only after waiting through every single timeout.

    This function checks the one fact that predicted that outcome ahead of
    time in this project's own investigation (`DevToolsSecurity -status`) and
    returns a one-line, printable hint if it looks like dispatch verification
    is likely to hang -- or None if the check is inconclusive/unavailable,
    which is deliberately NOT treated as "verification will work"; it is
    only ever used to print an early, actionable hint, never to change
    control flow or invent a dispatch result.
    """
    if platform.system() != "Darwin" or not shutil.which("DevToolsSecurity"):
        return None
    cp = run(["DevToolsSecurity", "-status"], timeout=10)
    if cp.returncode != 0:
        return None
    if "disabled" in cp.stdout.lower():
        return (
            "macOS Developer Mode is currently DISABLED (`DevToolsSecurity -status`). "
            "The built-in lldb dispatch verifier will likely HANG (not fail fast) on every "
            "configuration until its per-call timeout -- see tools/protocol.md section 6 "
            "item 9. Options: `sudo DevToolsSecurity -enable` (may require a restart; a "
            "system-wide security setting, so get sign-off before flipping it on someone "
            "else's machine), or pass --dispatch-ledger-json with dispatch facts captured "
            "on a host/session where lldb does work, or pass --skip-dispatch-verify to "
            "proceed with tok/s-only rows explicitly labeled unverified."
        )
    return None


# --------------------------------------------------------------------------
# llama-bench invocation + JSON parsing
# --------------------------------------------------------------------------

class LlamaBenchError(RuntimeError):
    pass


def find_llama_bench(bin_dir: Path) -> Path:
    candidate = bin_dir / "llama-bench"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    raise LlamaBenchError(
        f"llama-bench not found (or not executable) at {candidate}. "
        f"Build llama.cpp first (see tools/protocol.md section 8)."
    )


def run_llama_bench_once(
    llama_bench: Path,
    model: Path,
    n_prompt: int,
    n_gen: int,
    threads: int,
    sme_mode: str,
    reps: int = 1,
    warmup: bool = True,
    timeout: float = 60.0,
) -> List[Dict[str, Any]]:
    """Invoke llama-bench for exactly one (n_prompt, n_gen) pair, parse its
    JSON output, and return the list of result entries (usually length 1,
    since n_prompt==0 or n_gen==0 makes llama-bench skip the other test)."""
    cmd = [
        str(llama_bench),
        "-m", str(model),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(threads),
        "-r", str(reps),
        "-o", "json",
    ]
    if not warmup:
        cmd.append("--no-warmup")
    env = build_subprocess_env(sme_mode)
    cp = run(cmd, env=env, timeout=timeout)
    if cp.returncode != 0:
        raise LlamaBenchError(
            f"llama-bench exited {cp.returncode} for threads={threads} sme={sme_mode} "
            f"n_prompt={n_prompt} n_gen={n_gen}\nstdout={cp.stdout[-2000:]}\nstderr={cp.stderr[-2000:]}"
        )
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError as e:
        raise LlamaBenchError(f"could not parse llama-bench JSON output: {e}\nraw stdout: {cp.stdout[:2000]}")
    if not isinstance(data, list) or not data:
        raise LlamaBenchError(f"unexpected llama-bench JSON shape: {cp.stdout[:500]}")
    return data


# --------------------------------------------------------------------------
# Dispatch verification (see tools/protocol.md section 5.4)
# --------------------------------------------------------------------------

def _lldb_available() -> bool:
    return shutil.which("lldb") is not None


def _write_lldb_script(path: Path, auto_continue: bool) -> None:
    """Write the lldb batch-mode command file. See protocol.md section 5.4
    for why there are two tiers (fast first-hit vs thorough auto-continue)."""
    if auto_continue:
        script = f"""breakpoint set -r "{KAI_SME_REGEX}"
breakpoint command add 1
continue
DONE
breakpoint set -r "{KAI_NEON_REGEX}"
breakpoint command add 2
continue
DONE
run
breakpoint list
quit
"""
    else:
        script = f"""breakpoint set -r "{KAI_SME_REGEX}"
breakpoint set -r "{KAI_NEON_REGEX}"
run
breakpoint list
quit
"""
    path.write_text(script)


_HIT_RE = re.compile(r"^\s*([12])\.\d+:.*hit count = (\d+)", re.MULTILINE)


def _parse_lldb_breakpoint_list(text: str) -> Tuple[int, int]:
    """Sum hit counts for breakpoint group 1 (SME family) and group 2 (NEON
    family) out of an `lldb` `breakpoint list` transcript."""
    sme_hits = 0
    neon_hits = 0
    for group, count in _HIT_RE.findall(text):
        if group == "1":
            sme_hits += int(count)
        else:
            neon_hits += int(count)
    return sme_hits, neon_hits


def verify_dispatch_builtin(
    llama_bench: Path,
    model: Path,
    n_prompt: int,
    n_gen: int,
    threads: int,
    sme_mode: str,
    fast_timeout: float = 15.0,
    thorough_timeout: float = 25.0,
    allow_thorough: bool = True,
    scratch_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify, at the symbol level, which kernel family actually ran for this
    configuration. Returns a dict always containing at least
    {"verified": bool, "method": str}; on success also
    {"sme_fires": bool, "neon_fires": bool}, and when the thorough tier
    completed, {"sme_hits": int, "neon_hits": int, "hybrid": bool}.

    Never raises -- a debugger failure degrades to {"verified": False, ...},
    it never crashes the sweep and never fabricates a dispatch label.
    """
    if not _lldb_available():
        return {"verified": False, "reason": "lldb not found on PATH", "method": "none"}

    scratch_dir = scratch_dir or Path("/tmp")
    env = build_subprocess_env(sme_mode)

    # --- Fast tier: always run. First breakpoint hit, then stop looking. ---
    fast_script = scratch_dir / f"bench_lldb_fast_{os.getpid()}.txt"
    _write_lldb_script(fast_script, auto_continue=False)
    cmd = [
        "lldb", "-b", "-s", str(fast_script), "--",
        str(llama_bench), "-m", str(model),
        "-p", str(n_prompt), "-n", str(n_gen),
        "-t", str(threads), "-r", "1", "--no-warmup", "-o", "json",
        # NOTE: llama-bench's -o only accepts csv|json|jsonl|md|sql. An
        # earlier version of this script passed "-o none" to suppress noisy
        # output, which llama-bench silently accepts as an argument but then
        # runs ZERO benchmark iterations for (confirmed by hand: `-o none`
        # exits 0 with empty stdout/stderr, and a paired lldb run shows every
        # breakpoint at hit count 0). That produced a false "neither kernel
        # fired" dispatch label for every row. `-o json` is parsed by
        # llama-bench and forces the real benchmark loop to run; we discard
        # its JSON here (this is a dispatch check, not a timing measurement)
        # and only look at lldb's own "breakpoint list" transcript.
    ]
    cp = run(cmd, env=env, timeout=fast_timeout)
    fast_sme_hits, fast_neon_hits = _parse_lldb_breakpoint_list(cp.stdout)
    try:
        fast_script.unlink(missing_ok=True)
    except OSError:
        pass

    if cp.returncode == 124:
        return {"verified": False, "reason": f"fast-tier lldb timed out after {fast_timeout}s", "method": "lldb-fast"}

    result: Dict[str, Any] = {
        "verified": True,
        "method": "lldb-fast",
        "sme_fires": fast_sme_hits > 0,
        "neon_fires": fast_neon_hits > 0,
        "note": "fast tier reports which family hit FIRST / had already hit at the moment "
                "of the first stop -- boolean signal only, not a call count (see protocol.md 5.4).",
    }

    # --- Thorough tier: best-effort, bounded by a timeout. Prefill only by
    # default -- empirically, decode-phase auto-continue can stall for
    # minutes even at n_gen=2 (see protocol.md 6.7). We still allow the
    # caller to force it (allow_thorough) for completeness/experimentation,
    # but the sweep driver below never does so for the decode phase.
    if allow_thorough:
        thorough_script = scratch_dir / f"bench_lldb_thorough_{os.getpid()}.txt"
        _write_lldb_script(thorough_script, auto_continue=True)
        cp2 = run(cmd[:2] + ["-s", str(thorough_script)] + cmd[4:], env=env, timeout=thorough_timeout)
        try:
            thorough_script.unlink(missing_ok=True)
        except OSError:
            pass
        if cp2.returncode == 124:
            result["thorough_timed_out"] = True
        else:
            sme_hits, neon_hits = _parse_lldb_breakpoint_list(cp2.stdout)
            result["sme_hits"] = sme_hits
            result["neon_hits"] = neon_hits
            result["hybrid"] = sme_hits > 0 and neon_hits > 0
            # thorough tier supersedes the fast tier's boolean once we have counts
            result["sme_fires"] = sme_hits > 0
            result["neon_fires"] = neon_hits > 0
    return result


def load_dispatch_ledger(path: Path) -> Dict[Tuple, Dict[str, Any]]:
    """Load a precomputed dispatch ledger: a JSON list of records, each with
    at least {"phase", "quant", "threads", "sme_mode", "sme_fires",
    "neon_fires"} (optionally "sme_hits"/"neon_hits"/"hybrid"/"method"/
    "note"). Keyed the same way as the live verifiers, this lets a dispatch
    verification pass done once (e.g. via direct-shell `lldb`, on a machine/
    session where that is reliable) be reused across any number of bench.py
    invocations, instead of re-running a debugger inline on every sweep.

    This exists because live, in-process `lldb` invocation was found to be
    fragile in some sandboxed/nested-subprocess execution contexts during
    this project's own development (see tools/protocol.md section 6.9) --
    the ledger path is the documented workaround, not a shortcut around
    verification: every record must still trace back to a real `lldb` (or
    equivalent) observation, never a guess.
    """
    with open(path) as f:
        records = json.load(f)
    ledger: Dict[Tuple, Dict[str, Any]] = {}
    for rec in records:
        key = (rec["phase"], rec["quant"], int(rec["threads"]), rec["sme_mode"])
        entry = dict(rec)
        entry.setdefault("verified", True)
        entry.setdefault("method", "ledger:" + str(path))
        ledger[key] = entry
    return ledger


def verify_dispatch_external(
    verify_cmd: str,
    llama_bench: Path,
    model: Path,
    n_prompt: int,
    n_gen: int,
    threads: int,
    sme_mode: str,
    phase: str,
    timeout: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """Call a sibling verify_dispatch tool if the caller supplied one (see
    protocol.md section 5.4 for the JSON contract). Returns None (never
    raises) if the tool is missing, fails, or does not speak the contract --
    callers must fall back to verify_dispatch_builtin in that case."""
    args = verify_cmd.split() + [
        "--llama-bin", str(llama_bench),
        "--model", str(model),
        "--threads", str(threads),
        "--sme", sme_mode,
        "--phase", phase,
        "--n-prompt", str(n_prompt),
        "--n-gen", str(n_gen),
    ]
    cp = run(args, timeout=timeout)
    if cp.returncode != 0:
        return None
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "sme_fires" not in data:
        return None
    data.setdefault("method", "external:" + verify_cmd)
    data.setdefault("verified", True)
    return data


# --------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------

def discover_quants(model_q4_0: Optional[Path], model_q8_0: Optional[Path]) -> Dict[str, Optional[Path]]:
    quants: Dict[str, Optional[Path]] = {}
    quants["Q4_0"] = model_q4_0 if (model_q4_0 and model_q4_0.is_file()) else None
    quants["Q8_0"] = model_q8_0 if (model_q8_0 and model_q8_0.is_file()) else None
    return quants


def build_config_grid(
    threads: List[int],
    sme_modes: List[str],
    phases: Dict[str, Tuple[int, int]],
    quants: Dict[str, Optional[Path]],
) -> List[Dict[str, Any]]:
    """Fixed iteration order -- this literally IS the interleaving pattern
    (A,B,C,A,B,C,... at the round level in run_sweep()). Quants with no
    available model file are still listed so the output/markdown can say
    "[not available]" explicitly instead of silently omitting the axis."""
    configs = []
    for phase_name, (n_prompt, n_gen) in phases.items():
        for quant_name, model_path in quants.items():
            for th in threads:
                for sme in sme_modes:
                    configs.append({
                        "phase": phase_name,
                        "n_prompt": n_prompt,
                        "n_gen": n_gen,
                        "quant": quant_name,
                        "model": str(model_path) if model_path else None,
                        "threads": th,
                        "sme_mode": sme,
                    })
    return configs


def run_sweep(
    llama_bench: Path,
    configs: List[Dict[str, Any]],
    reps: int,
    per_call_timeout: float,
    progress: bool = True,
) -> Dict[Tuple, List[Dict[str, Any]]]:
    """Interleaved measurement: round-robins through every config once per
    round, `reps` rounds total, so no configuration's samples are all
    clustered early or late in wall-clock time (see protocol.md section 5.1).
    Returns {config_key: [raw llama-bench JSON entries, one per round]}.
    """
    samples: Dict[Tuple, List[Dict[str, Any]]] = {}
    available = [c for c in configs if c["model"] is not None]
    unavailable = [c for c in configs if c["model"] is None]
    for c in unavailable:
        key = _config_key(c)
        samples[key] = []  # explicitly empty: "not available", never fabricated

    total_calls = len(available) * reps
    call_no = 0
    for round_idx in range(reps):
        for c in available:
            call_no += 1
            key = _config_key(c)
            if progress:
                print(
                    f"[bench.py] round {round_idx + 1}/{reps}  call {call_no}/{total_calls}  "
                    f"phase={c['phase']} quant={c['quant']} threads={c['threads']} sme={c['sme_mode']}",
                    file=sys.stderr,
                )
            try:
                entries = run_llama_bench_once(
                    llama_bench=llama_bench,
                    model=Path(c["model"]),
                    n_prompt=c["n_prompt"],
                    n_gen=c["n_gen"],
                    threads=c["threads"],
                    sme_mode=c["sme_mode"],
                    reps=1,
                    warmup=True,  # every call gets its own discarded warmup
                    timeout=per_call_timeout,
                )
                samples.setdefault(key, []).extend(entries)
            except LlamaBenchError as e:
                print(f"[bench.py] WARNING: measurement failed for {key}: {e}", file=sys.stderr)
                samples.setdefault(key, [])
    return samples


def _config_key(c: Dict[str, Any]) -> Tuple:
    return (c["phase"], c["quant"], c["threads"], c["sme_mode"])


# --------------------------------------------------------------------------
# Aggregation + reporting
# --------------------------------------------------------------------------

def aggregate(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """entries: list of llama-bench JSON result objects, one per round (each
    with reps=1, so samples_ts has exactly one element). Returns
    median/stddev/min/max tok/s across rounds, or None if no samples."""
    ts_values: List[float] = []
    for e in entries:
        vals = e.get("samples_ts") or []
        ts_values.extend(float(v) for v in vals)
    if not ts_values:
        return None
    return {
        "n": len(ts_values),
        "median_ts": statistics.median(ts_values),
        "stddev_ts": statistics.stdev(ts_values) if len(ts_values) > 1 else 0.0,
        "min_ts": min(ts_values),
        "max_ts": max(ts_values),
        "samples_ts": ts_values,
    }


def render_markdown(meta: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append(f"# Bench results -- {meta['platform']}")
    lines.append("")
    lines.append(f"- Generated: {meta['generated_at']}")
    lines.append(f"- CPU: {meta['cpu_brand']}")
    lines.append(f"- llama.cpp bin dir: `{meta['llama_bin_dir']}`")
    lines.append(f"- Reps per cell: {meta['reps']}")
    lines.append("- Full methodology: see `tools/protocol.md`. Median/stddev/min/max are computed "
                 "across independently-warmed-up, interleaved process invocations -- never a bare mean.")
    lines.append("")
    therm_before = meta.get("thermal", {}).get("before", {}).get("output", "")
    therm_after = meta.get("thermal", {}).get("after", {}).get("output", "")
    if therm_before or therm_after:
        lines.append("## Thermal context")
        lines.append("")
        lines.append("Before:")
        lines.append("```")
        lines.append(therm_before or "(unavailable)")
        lines.append("```")
        lines.append("After:")
        lines.append("```")
        lines.append(therm_after or "(unavailable)")
        lines.append("```")
        if therm_before != therm_after:
            lines.append("")
            lines.append("**Note:** thermal snapshot changed between the start and end of the sweep. "
                          "See per-row notes if a specific configuration ran after a state change.")
        lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| phase | quant | threads | SME | median tok/s | stddev | min | max | n | dispatch |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if r.get("not_available"):
            lines.append(
                f"| {r['phase']} | {r['quant']} | {r['threads']} | {r['sme_mode']} | "
                f"_[not available]_ | | | | | model file not present |"
            )
            continue
        agg = r.get("agg")
        if agg is None:
            lines.append(
                f"| {r['phase']} | {r['quant']} | {r['threads']} | {r['sme_mode']} | "
                f"_[measurement failed]_ | | | | | see stderr log |"
            )
            continue
        d = r.get("dispatch", {})
        dispatch_label = _dispatch_label(d)
        lines.append(
            f"| {r['phase']} | {r['quant']} | {r['threads']} | {r['sme_mode']} | "
            f"{agg['median_ts']:.1f} | {agg['stddev_ts']:.2f} | {agg['min_ts']:.1f} | "
            f"{agg['max_ts']:.1f} | {agg['n']} | {dispatch_label} |"
        )

    lines.append("")
    lines.append("Threads/quants not listed above were not measured in this run and must not be "
                  "treated as zero, equal-to-neighbor, or interpolated -- see `tools/protocol.md` section 7.")
    lines.append("")
    return "\n".join(lines)


def _dispatch_label(d: Dict[str, Any]) -> str:
    if not d:
        return "unverified"
    if not d.get("verified", False):
        return f"unverified ({d.get('reason', 'unknown')})"
    sme = d.get("sme_fires")
    neon = d.get("neon_fires")
    if d.get("hybrid"):
        return f"HYBRID: SME2 x{d.get('sme_hits','?')} + NEON x{d.get('neon_hits','?')}"
    if sme and neon:
        return "SME2+NEON both fired (fast-tier snapshot)"
    if sme:
        return "SME2 fired" + ("" if "sme_hits" not in d else f" (x{d['sme_hits']})")
    if neon:
        return "NEON fired" + ("" if "neon_hits" not in d else f" (x{d['neon_hits']})")
    return "neither fired (unexpected -- inspect manually)"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_threads_arg(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_sme_modes_arg(s: str) -> List[str]:
    modes = [x.strip() for x in s.split(",") if x.strip()]
    for m in modes:
        if m not in SME_MODES:
            raise argparse.ArgumentTypeError(f"unknown sme mode {m!r}, expected one of {SME_MODES}")
    return modes


def parse_phases_arg(s: str, decode_tokens: int, prefill_short_tokens: int, prefill_long_tokens: int) -> Dict[str, Tuple[int, int]]:
    all_phases = default_phases(decode_tokens, prefill_short_tokens, prefill_long_tokens)
    chosen = [x.strip() for x in s.split(",") if x.strip()]
    for p in chosen:
        if p not in all_phases:
            raise argparse.ArgumentTypeError(f"unknown phase {p!r}, expected one of {list(all_phases)}")
    return {p: all_phases[p] for p in chosen}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--llama-bin-dir", type=Path, default=DEFAULT_LLAMA_BIN_DIR,
                    help=f"directory containing llama-bench (default: {DEFAULT_LLAMA_BIN_DIR})")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_Q4_0,
                    help=f"Q4_0 .gguf path (default: {DEFAULT_MODEL_Q4_0})")
    ap.add_argument("--model-q8", type=Path, default=None,
                    help="Q8_0 .gguf path, if available (default: none -> marked [not available])")
    ap.add_argument("--threads", type=parse_threads_arg, default=DEFAULT_THREADS,
                    help=f"comma-separated thread counts (default: {DEFAULT_THREADS}; "
                         f"pass 1,2,4,8,16 for the full sweep)")
    ap.add_argument("--sme-modes", type=parse_sme_modes_arg, default=SME_MODES,
                    help=f"comma-separated SME modes (default: {SME_MODES})")
    ap.add_argument("--phases", type=str, default="decode,prefill_short,prefill_long",
                    help="comma-separated phases: decode,prefill_short,prefill_long")
    ap.add_argument("--decode-tokens", type=int, default=32, help="n_gen for the decode phase")
    ap.add_argument("--prefill-short-tokens", type=int, default=64,
                    help="n_prompt for prefill_short (kept below the ne11>=128 hybrid-dispatch gate)")
    ap.add_argument("--prefill-long-tokens", type=int, default=256,
                    help="n_prompt for prefill_long (kept above the ne11>=128 hybrid-dispatch gate)")
    ap.add_argument("--reps", type=int, default=5, help="repetitions per configuration (default: 5, minimum recommended: 5)")
    ap.add_argument("--per-call-timeout", type=float, default=60.0, help="timeout (s) for each llama-bench call")
    ap.add_argument("--platform", type=str, default=None, help="platform slug for output filenames (default: auto-detected from CPU brand)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="directory to write bench-<platform>.json/.md into")
    ap.add_argument("--skip-dispatch-verify", action="store_true", help="skip lldb/external dispatch verification entirely (rows will be 'unverified')")
    ap.add_argument("--dispatch-fast-timeout", type=float, default=15.0)
    ap.add_argument("--dispatch-thorough-timeout", type=float, default=25.0)
    ap.add_argument("--verify-dispatch-cmd", type=str, default=None,
                    help="optional external verifier command, e.g. 'python3 tools/verify_dispatch.py' "
                         "(see tools/protocol.md section 5.4 for the JSON contract). Falls back to the "
                         "built-in lldb verifier if this is unset, missing, or fails.")
    ap.add_argument("--dispatch-ledger-json", type=Path, default=None,
                    help="path to a precomputed dispatch ledger (JSON list of records with "
                         "phase/quant/threads/sme_mode/sme_fires/neon_fires). Checked BEFORE "
                         "--verify-dispatch-cmd and the built-in lldb verifier for each cell; a cell "
                         "missing from the ledger still falls through to those. See tools/protocol.md "
                         "section 6.9 for why this exists.")
    ap.add_argument("--quiet", action="store_true", help="suppress progress lines on stderr")
    args = ap.parse_args(argv)

    try:
        llama_bench = find_llama_bench(args.llama_bin_dir)
    except LlamaBenchError as e:
        print(f"[bench.py] FATAL: {e}", file=sys.stderr)
        return 2

    phases = parse_phases_arg(args.phases, args.decode_tokens, args.prefill_short_tokens, args.prefill_long_tokens)
    quants = discover_quants(args.model, args.model_q8)
    if quants["Q4_0"] is None:
        print(f"[bench.py] FATAL: Q4_0 model not found at {args.model}", file=sys.stderr)
        return 2
    if quants["Q8_0"] is None:
        print("[bench.py] NOTE: Q8_0 model not supplied/found -- marked [not available] in output, "
              "not fabricated. See tools/protocol.md section 3.", file=sys.stderr)

    plat = args.platform or platform_slug()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_config_grid(args.threads, args.sme_modes, phases, quants)

    meta: Dict[str, Any] = {
        "platform": plat,
        "cpu_brand": cpu_brand(),
        "os": f"{platform.system()} {platform.release()}",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "llama_bin_dir": str(args.llama_bin_dir),
        "model_q4_0": str(args.model),
        "model_q8_0": str(args.model_q8) if args.model_q8 else None,
        "threads_swept": args.threads,
        "sme_modes_swept": args.sme_modes,
        "phases_swept": phases,
        "reps": args.reps,
        "thermal": {"before": thermal_snapshot()},
        "protocol": "see tools/protocol.md",
        "candidateOnly": True,
        "canClaimAGI": False,
    }

    print(f"[bench.py] {len(configs)} configurations x {args.reps} reps "
          f"({sum(1 for c in configs if c['model'])} measurable, "
          f"{sum(1 for c in configs if not c['model'])} not-available) -- starting interleaved sweep",
          file=sys.stderr)

    t0 = time.time()
    raw_samples = run_sweep(
        llama_bench=llama_bench,
        configs=configs,
        reps=args.reps,
        per_call_timeout=args.per_call_timeout,
        progress=not args.quiet,
    )
    elapsed = time.time() - t0
    meta["thermal"]["after"] = thermal_snapshot()
    meta["sweep_wall_seconds"] = round(elapsed, 1)
    print(f"[bench.py] sweep done in {elapsed:.1f}s, verifying dispatch per unique configuration...", file=sys.stderr)

    # Dispatch verification -- once per unique (phase, threads, sme, quant),
    # not once per repetition (dispatch is deterministic given those inputs).
    ledger = load_dispatch_ledger(args.dispatch_ledger_json) if args.dispatch_ledger_json else {}
    if args.dispatch_ledger_json:
        print(f"[bench.py] loaded {len(ledger)} precomputed dispatch records from {args.dispatch_ledger_json}", file=sys.stderr)

    # Early, non-fatal warning for a real failure mode this project hit directly
    # (tools/protocol.md section 6 item 9): if any measurable configuration is
    # NOT already covered by the ledger, the built-in lldb verifier is about to
    # run for it -- and on a host with Developer Mode disabled, lldb hangs
    # (rather than failing fast) until the per-call timeout, once per
    # configuration. Print the hint once, up front, instead of letting a user
    # discover it 15-25 seconds at a time.
    if not args.skip_dispatch_verify:
        measurable_keys = {_config_key(c) for c in configs if c["model"] is not None}
        if measurable_keys - set(ledger.keys()):
            hint = lldb_preflight_hint()
            if hint:
                print(f"[bench.py] WARNING: {hint}", file=sys.stderr)

    dispatch_cache: Dict[Tuple, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    seen_keys = set()
    for c in configs:
        key = _config_key(c)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        if c["model"] is None:
            rows.append({**c, "not_available": True})
            continue

        entries = raw_samples.get(key, [])
        agg = aggregate(entries)

        dispatch: Dict[str, Any]
        if args.skip_dispatch_verify:
            dispatch = {"verified": False, "reason": "--skip-dispatch-verify passed"}
        elif key in ledger:
            dispatch = ledger[key]
        else:
            dispatch = None
            if args.verify_dispatch_cmd:
                dispatch = verify_dispatch_external(
                    args.verify_dispatch_cmd, llama_bench, Path(c["model"]),
                    c["n_prompt"], c["n_gen"], c["threads"], c["sme_mode"], c["phase"],
                )
            if dispatch is None:
                # Never run the thorough (auto-continue) tier on the decode
                # phase -- it can stall for minutes under multi-threaded
                # contention (see protocol.md section 6.7). Fast tier still
                # runs for decode, giving a boolean fires/does-not-fire.
                allow_thorough = c["phase"] != "decode"
                dispatch = verify_dispatch_builtin(
                    llama_bench, Path(c["model"]), c["n_prompt"], c["n_gen"],
                    c["threads"], c["sme_mode"],
                    fast_timeout=args.dispatch_fast_timeout,
                    thorough_timeout=args.dispatch_thorough_timeout,
                    allow_thorough=allow_thorough,
                )
        dispatch_cache[key] = dispatch

        rows.append({**c, "agg": agg, "dispatch": dispatch})
        print(f"[bench.py] dispatch[{key}] = {_dispatch_label(dispatch)}", file=sys.stderr)

    out_json_path = args.out_dir / f"bench-{plat}.json"
    out_md_path = args.out_dir / f"bench-{plat}.md"

    payload = {"meta": meta, "rows": copy.deepcopy(rows)}
    out_json_path.write_text(json.dumps(payload, indent=2, default=str))

    md = render_markdown(meta, rows)
    out_md_path.write_text(md)

    print(f"[bench.py] wrote {out_json_path}", file=sys.stderr)
    print(f"[bench.py] wrote {out_md_path}", file=sys.stderr)
    print("", file=sys.stderr)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
