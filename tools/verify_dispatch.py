#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
"""verify_dispatch.py -- the symbol-level KleidiAI dispatch verifier.

THE QUESTION THIS TOOL ANSWERS
-------------------------------
"Did the accelerated kernel ACTUALLY execute?" -- a question a timing-only
benchmark structurally cannot answer, because llama.cpp's own banner and log
("SME = 1 | SME2 = 1 | KLEIDIAI = 1", "kleidiai: primary q4 kernel feature
SME2") are compile-time / selection-time signals, not proof of what the CPU
actually dispatched at inference time.

Two verified findings on Apple Silicon (KleidiAI, llama.cpp @ dbadb68) motivate
this tool:

  1. `detect_num_smcus()` hardcodes a thread cap per Apple CPU brand string
     (2 on an M4 Max). SME2 dispatches when `n_threads <= sme_thread_cap`, OR
     via a *hybrid* path when the batch size being multiplied (`ne11`) is
     large enough (prefill). At small `ne11` (decode, `ne11 == 1`) and
     `n_threads > sme_thread_cap`, the kernel chain silently collapses to a
     NEON kernel -- while the log keeps claiming "SME2 enabled" verbatim.
  2. The SVE kernel family requires `ggml_cpu_get_sve_cnt() == QK8_0` (256-bit
     vectors); any CPU with a narrower SVE2 implementation (e.g. Arm's own
     Cortex-X925 in the DGX Spark, at 128-bit) can never select it, even
     though the core genuinely implements SVE2/i8mm/bf16.

This tool does not trust either the banner or the selection log. It gathers
THREE independent, increasingly decisive layers of evidence for every
(thread-count, workload) configuration in a sweep, and reports all three:

  L1 STATIC   -- do the accelerated-kernel symbols/instructions exist in the
                 built binary/library at all? (nm/otool on macOS,
                 nm/objdump on Linux)
  L2 SELECT   -- what does llama.cpp's own --verbose log say it *chose*?
                 (parses "kleidiai: primary q4 kernel feature X", "SME2
                 enabled", and the buffer-type fallback warning)
  L3 DISPATCH -- the decisive layer: attach a debugger, set an
                 auto-continuing regex breakpoint on every
                 `kai_run_matmul_*` entry point, run the REAL workload, and
                 count hits per symbol. This is the only layer that can
                 distinguish "selected but never called" from "actually ran".

Usage (macOS, lldb; see also `--help`):

    python3 verify_dispatch.py \\
        --binary /tmp/llama.cpp/build/bin/llama-cli \\
        --model  /tmp/ggufs/q05.gguf \\
        --threads 1,2,4,8,16 \\
        --assert

Exits non-zero under --assert if any swept configuration shows the
advertised accelerated kernel family (from L2) NEVER dispatching (from L3)
while a different family did -- i.e. a genuine silent fallback -- so this
can gate CI.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
LLDB_TEMPLATE_PATH = os.path.join(TOOLS_DIR, "dispatch_probe.lldb")
GDB_TEMPLATE_PATH = os.path.join(TOOLS_DIR, "dispatch_probe.gdb")

# Default breakpoint / symbol-scan target. Anchored at the start of the
# symbol name on purpose: see dispatch_probe.lldb's header comment for why
# an unanchored pattern silently double-counts (it also matches the
# compiler-generated template thunk that tail-calls into the real symbol).
DEFAULT_DISPATCH_REGEX = r"^kai_run_matmul"

DEFAULT_THREADS = [1, 2, 4, 8, 16]

# A short prompt with a small generation budget is decode-dominated: almost
# all matmuls happen one token at a time (ne11 == 1 in ggml terms). A long
# prompt is prefill-dominated: the first forward pass multiplies a large
# batch (ne11 == prompt length) in one shot. KleidiAI's hybrid SME2 path
# only engages when ne11 is large, so sweeping BOTH is what actually
# reveals the real dispatch rule -- sweeping threads alone with a single
# short prompt (as an earlier draft of this investigation did) is
# incomplete: see results/GROUND-TRUTH-DISPATCH.md, "Correction to the
# earlier draft".
_LONG_PROMPT_WORDS = (
    "The quick brown fox jumps over the lazy dog. The history of artificial "
    "intelligence began in the mid twentieth century, when researchers "
    "started exploring how machines could simulate human reasoning and "
    "learning processes. Since then the field has expanded dramatically, "
    "encompassing machine learning, deep neural networks, natural language "
    "processing, computer vision, and robotics. Modern systems can now "
    "translate languages, recognize images, generate text, and even compose "
    "music with remarkable fluency. However, many open questions remain "
    "about how these systems generalize, reason, and represent knowledge "
    "internally. Researchers continue to study architectures, training "
    "methods, and evaluation benchmarks in order to better understand the "
    "capabilities and limitations of contemporary models. "
) * 6

DEFAULT_WORKLOADS = [
    {
        "name": "decode_short",
        "prompt": "Hello.",
        "n_predict": 4,
        "description": (
            "short prompt, generation dominates -- exercises the decode-time "
            "(ne11 == 1) dispatch path"
        ),
    },
    {
        "name": "prefill_long",
        "prompt": _LONG_PROMPT_WORDS.strip(),
        "n_predict": 1,
        "description": (
            "long synthetic prompt (~380 words), 1 generated token -- "
            "exercises the prefill-time (large ne11) dispatch path"
        ),
    },
]

# Env vars kleidiai.cpp itself reads (see ggml/src/ggml-cpu/kleidiai/kleidiai.cpp).
# Recorded (not set) by default so the ledger shows what actually influenced
# the run; --env can override/set them for deliberate what-if probes.
KLEIDIAI_ENV_VARS = [
    "GGML_KLEIDIAI_SME",
    "GGML_TOTAL_THREADS",
    "GGML_KLEIDIAI_CHUNK_MULTIPLIER",
    "OMP_NUM_THREADS",
]

# Family classification is intentionally substring-based and ORDER SENSITIVE:
# more specific tokens (sme2, sve2) are checked before their more general
# substrings (sme, sve) so "..._sme2_mopa" is never misclassified as "sme".
_FAMILY_ORDER = ["sme2", "sme", "sve2", "sve", "i8mm", "dotprod", "neon"]


def classify_symbol_family(symbol_name: str) -> str:
    """Classify a kai_run_matmul_* (or similar) symbol into a kernel family.

    Based on KleidiAI's own naming convention, e.g.:
      kai_run_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa -> "sme2"
      kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod -> "dotprod"
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm      -> "i8mm"
    Returns "other" if no known token is found (e.g. a completely different
    naming scheme was linked in).
    """
    n = symbol_name.lower()
    for family in _FAMILY_ORDER:
        if family in n:
            return family
    return "other"


# L1 static-scan mnemonic sets. These are TEXT patterns matched against
# disassembly listings (otool -tv on macOS, objdump -d on Linux) -- a
# best-effort lexical scan, not a real instruction decoder. It answers "does
# this ISA extension's instruction encoding appear anywhere in this binary",
# which is a compile-time/link-time fact, layers below what actually runs.
SME_MNEMONICS = ["smstart", "smstop", "fmopa", "bfmopa", "smopa", "sumopa", "usmopa", "ummopa"]
# SVE/SVE2 Z-register operand syntax (e.g. "z3.s", "z12.d") is a strong,
# fairly unambiguous textual signature of SVE(2) code in AArch64 disassembly.
SVE_REGISTER_RE = re.compile(r"\bz\d{1,2}\.[bhsd]\b")
I8MM_MNEMONICS = ["smmla", "ummla", "usmmla", "sudot", "usdot"]
DOTPROD_MNEMONICS = ["sdot", "udot"]


# --------------------------------------------------------------------------
# Small dataclasses for structured results
# --------------------------------------------------------------------------

@dataclass
class L1Result:
    lib_path: Optional[str]
    tool_chain: str  # "macho" (nm -gU + otool -tv) or "elf" (nm -D + objdump -d)
    available: bool
    error: Optional[str] = None
    kai_symbol_count: int = 0
    kai_run_matmul_symbol_count: int = 0
    kai_symbols_by_family: dict = field(default_factory=dict)
    isa_mnemonic_counts: dict = field(default_factory=dict)


@dataclass
class L2Result:
    available: bool
    command: list
    returncode: Optional[int] = None
    timed_out: bool = False
    error: Optional[str] = None
    primary_kernel_feature: dict = field(default_factory=dict)  # {"q4": "SME2", ...}
    sme_enabled: Optional[bool] = None
    sme_variant: Optional[str] = None
    sme_detail: Optional[str] = None  # e.g. "runtime-detected SME cores=2"
    buffer_type_fallback_seen: bool = False
    unsupported_tensor_types: list = field(default_factory=list)
    model_file_type: Optional[str] = None  # e.g. "Q4_0", from llama.cpp's own model-card line
    system_info_line: Optional[str] = None
    raw_kleidiai_lines: list = field(default_factory=list)
    wall_time_sec: float = 0.0


@dataclass
class L3Result:
    debugger: str  # "lldb", "gdb", or "none"
    available: bool
    command: list = field(default_factory=list)
    completed: bool = False
    timed_out: bool = False
    error: Optional[str] = None
    dispatch_regex: str = DEFAULT_DISPATCH_REGEX
    hits_by_symbol: dict = field(default_factory=dict)
    hits_by_family: dict = field(default_factory=dict)
    total_hits: int = 0
    kernel_family_executed: str = "none"
    wall_time_sec: float = 0.0
    # gdb lane only: proves the probe was actually instrumented, so a zero hit
    # count can be read as "the kernel did not run" rather than "we never looked".
    breakpoints_requested: Optional[int] = None
    breakpoints_created: Optional[int] = None


# --------------------------------------------------------------------------
# Subprocess helper with a real wall-clock timeout that kills the whole
# process group (a debugger that hangs waiting on its inferior must not
# leak an orphaned llama-cli still spinning on 16 threads).
# --------------------------------------------------------------------------

def _run_with_timeout(cmd, timeout, env=None):
    """Run cmd, capturing merged stdout+stderr, honoring a hard timeout.

    Returns (returncode, output_text, timed_out, wall_time_sec).
    On timeout the entire process group is SIGKILLed so no debugger or
    inferior process is left running.
    """
    start = time.monotonic()
    kwargs = dict(
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # start_new_session puts the child (and anything it forks, e.g. the
    # inferior lldb/gdb launches) in its own process group so we can kill
    # the whole tree on timeout instead of just the direct child.
    kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kwargs)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        rc = proc.returncode
    wall = time.monotonic() - start
    return rc, out or "", timed_out, wall


# --------------------------------------------------------------------------
# Platform / hardware identification
# --------------------------------------------------------------------------

def platform_id() -> str:
    return f"{platform.system().lower()}-{platform.machine().lower()}"


def detect_cpu_brand() -> str:
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
        return "[unknown Apple CPU -- sysctl machdep.cpu.brand_string failed]"
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                text = fh.read()
            m = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
            if m:
                return m.group(1).strip()
            m = re.search(r"^Model\s*:\s*(.+)$", text, re.MULTILINE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        return "[unknown Linux CPU -- /proc/cpuinfo model name/Model not found]"
    return f"[unsupported platform for cpu_brand: {system}]"


def detect_isa_features() -> dict:
    """Best-effort ISA feature detection. macOS uses sysctl hw.optional.arm.*
    (exact, since it's what the kernel/CPU itself reports). Linux uses
    /proc/cpuinfo "Features" flags, which does NOT include SVE vector width
    -- that requires a runtime prctl(PR_SVE_GET_VL) call this script does
    not make, so it is reported as an explicit placeholder rather than
    guessed.
    """
    system = platform.system()
    features: dict = {}
    if system == "Darwin":
        keys = [
            "hw.optional.arm.FEAT_SME",
            "hw.optional.arm.FEAT_SME2",
            "hw.optional.arm.FEAT_SME2p1",
            "hw.optional.arm.FEAT_SME_F64F64",
            "hw.optional.arm.FEAT_SME_I16I64",
            "hw.optional.arm.FEAT_I8MM",
            "hw.optional.arm.FEAT_BF16",
            "hw.optional.arm.FEAT_DotProd",
            "hw.optional.arm.sme_max_svl_b",
        ]
        for k in keys:
            try:
                out = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True, timeout=5)
                if out.returncode == 0:
                    val = out.stdout.strip()
                    features[k.split(".")[-1]] = int(val) if val.lstrip("-").isdigit() else val
            except Exception:
                continue
        # Apple ships SME2 WITHOUT non-streaming SVE -- FEAT_SVE is not a
        # published sysctl key on Apple Silicon at all (its absence IS the
        # finding), so record that explicitly rather than silently omitting it.
        features["FEAT_SVE"] = "absent (not exposed on Apple Silicon; SME2 ships without non-streaming SVE)"
        return features
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                text = fh.read()
            m = re.search(r"^Features\s*:\s*(.+)$", text, re.MULTILINE)
            flags = set(m.group(1).split()) if m else set()
        except Exception:
            flags = set()
        for flag in ["sve", "sve2", "i8mm", "bf16", "asimddp", "sme", "sme2"]:
            features[flag] = flag in flags
        features["sve_vector_length_bits"] = (
            "[not measured -- requires prctl(PR_SVE_GET_VL); not queried by this script]"
        )
        return features
    return {"_note": f"ISA feature detection not implemented for platform: {system}"}


# --------------------------------------------------------------------------
# Locating the CPU backend shared library that actually holds the kai_*
# symbols (llama-cli itself just links against it; the symbols live in
# libggml-cpu.{dylib,so}).
# --------------------------------------------------------------------------

def find_cpu_backend_lib(binary_path: str, lib_dir: Optional[str], lib_name: Optional[str]) -> Optional[str]:
    if lib_name and os.path.isfile(lib_name):
        return os.path.abspath(lib_name)
    search_dir = lib_dir or os.path.dirname(os.path.abspath(binary_path))
    system = platform.system()
    patterns = (
        ["libggml-cpu*.dylib"] if system == "Darwin" else ["libggml-cpu*.so*"]
    )
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(search_dir, pattern)))
        # Prefer the unversioned symlink if present (e.g. libggml-cpu.dylib
        # over libggml-cpu.0.18.0.dylib) purely for a cleaner reported path;
        # functionally identical, it's the same file.
        for m in matches:
            if re.search(r"libggml-cpu\.(dylib|so)$", m):
                return os.path.abspath(m)
        if matches:
            return os.path.abspath(matches[0])
    return None


# --------------------------------------------------------------------------
# L1 STATIC -- does the accelerated kernel exist in the built artifact at all?
# --------------------------------------------------------------------------

def run_l1_static(lib_path: Optional[str], dispatch_regex: str) -> L1Result:
    system = platform.system()
    tool_chain = "macho" if system == "Darwin" else "elf"
    if not lib_path or not os.path.isfile(lib_path):
        return L1Result(lib_path=lib_path, tool_chain=tool_chain, available=False,
                         error="CPU backend library not found (pass --lib-dir/--lib-name explicitly)")

    dispatch_re = re.compile(dispatch_regex)
    families: dict = {}
    kai_count = 0
    kai_matmul_count = 0

    if system == "Darwin":
        nm_bin, otool_bin = shutil.which("nm"), shutil.which("otool")
        if not nm_bin or not otool_bin:
            return L1Result(lib_path=lib_path, tool_chain=tool_chain, available=False,
                             error=f"missing tool(s): nm={bool(nm_bin)} otool={bool(otool_bin)}")
        try:
            nm_out = subprocess.run([nm_bin, "-gU", lib_path], capture_output=True, text=True, timeout=30)
        except Exception as e:
            return L1Result(lib_path=lib_path, tool_chain=tool_chain, available=False, error=f"nm failed: {e}")
        for line in nm_out.stdout.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            sym = parts[-1].lstrip("_")
            if not sym.startswith("kai_"):
                continue
            kai_count += 1
            if dispatch_re.match(sym):
                kai_matmul_count += 1
                fam = classify_symbol_family(sym)
                families[fam] = families.get(fam, 0) + 1

        mnemonic_counts = {}
        try:
            otool_out = subprocess.run([otool_bin, "-tv", lib_path], capture_output=True, text=True, timeout=60)
            text = otool_out.stdout.lower()
            for mnemonic in SME_MNEMONICS:
                c = len(re.findall(r"\b" + re.escape(mnemonic) + r"\b", text))
                if c:
                    mnemonic_counts[mnemonic] = c
        except Exception as e:
            mnemonic_counts["_error"] = str(e)

        return L1Result(
            lib_path=lib_path, tool_chain=tool_chain, available=True,
            kai_symbol_count=kai_count, kai_run_matmul_symbol_count=kai_matmul_count,
            kai_symbols_by_family=families, isa_mnemonic_counts=mnemonic_counts,
        )

    # Linux / ELF path.
    nm_bin, objdump_bin = shutil.which("nm"), shutil.which("objdump")
    if not nm_bin or not objdump_bin:
        return L1Result(lib_path=lib_path, tool_chain=tool_chain, available=False,
                         error=f"missing tool(s): nm={bool(nm_bin)} objdump={bool(objdump_bin)}")
    try:
        nm_out = subprocess.run([nm_bin, "-D", lib_path], capture_output=True, text=True, timeout=30)
    except Exception as e:
        return L1Result(lib_path=lib_path, tool_chain=tool_chain, available=False, error=f"nm failed: {e}")
    for line in nm_out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        sym = parts[-1]
        if not sym.startswith("kai_"):
            continue
        kai_count += 1
        if dispatch_re.match(sym):
            kai_matmul_count += 1
            fam = classify_symbol_family(sym)
            families[fam] = families.get(fam, 0) + 1

    mnemonic_counts = {}
    try:
        objdump_out = subprocess.run([objdump_bin, "-d", lib_path], capture_output=True, text=True, timeout=120)
        text = objdump_out.stdout.lower()
        sve_hits = len(SVE_REGISTER_RE.findall(text))
        if sve_hits:
            mnemonic_counts["sve_z_register_operands"] = sve_hits
        for mnemonic in I8MM_MNEMONICS:
            c = len(re.findall(r"\b" + re.escape(mnemonic) + r"\b", text))
            if c:
                mnemonic_counts[mnemonic] = c
        for mnemonic in DOTPROD_MNEMONICS:
            c = len(re.findall(r"\b" + re.escape(mnemonic) + r"\b", text))
            if c:
                mnemonic_counts[mnemonic] = c
    except Exception as e:
        mnemonic_counts["_error"] = str(e)

    return L1Result(
        lib_path=lib_path, tool_chain=tool_chain, available=True,
        kai_symbol_count=kai_count, kai_run_matmul_symbol_count=kai_matmul_count,
        kai_symbols_by_family=families, isa_mnemonic_counts=mnemonic_counts,
    )


# --------------------------------------------------------------------------
# L2 SELECT -- parse llama.cpp's own --verbose log for what it *chose*.
# --------------------------------------------------------------------------

_RE_PRIMARY_FEATURE = re.compile(r"kleidiai: primary (q4|q8|f32) kernel feature (\S+)")
_RE_SME_ENABLED = re.compile(r"kleidiai: SME(2?) enabled \(([^)]*)\)")
_RE_SME_DISABLED = re.compile(r"kleidiai: SME disabled")
_RE_BUFFER_FALLBACK = re.compile(r"cannot be used with preferred buffer type (\S+)")
_RE_UNSUPPORTED_TENSOR = re.compile(r"kleidiai: no kernel for tensor type (\S+), not accelerated")
_RE_SYSTEM_INFO = re.compile(r"^.*system_info:.*$", re.MULTILINE)
# The model's overall quant ("Q4_0" etc). This is llama.cpp's OWN model-card
# summary line, printed unconditionally (no --verbose needed) both in the
# `print_info: file type   = Q4_0` (info-log) form and the plain banner
# `ftype      : Q4_0` form shown above the interactive-mode help text. Using
# this instead of guessing from the model filename matters here: this
# project's model file is named q05.gguf, which does not encode the quant at
# all -- so a filename-based guess would be a fabricated fact this tool never
# actually observed.
_RE_FILE_TYPE = re.compile(r"(?:print_info: file type\s*=\s*|^ftype\s*:\s*)(\S+)", re.MULTILINE)


def run_l2_select(binary: str, model: str, threads: int, prompt: str, n_predict: int,
                   timeout: float, env: dict) -> L2Result:
    cmd = [
        binary, "-m", model, "-p", prompt, "-n", str(n_predict),
        "-no-cnv", "-st", "--simple-io", "-t", str(threads), "--verbose",
    ]
    rc, out, timed_out, wall = _run_with_timeout(cmd, timeout, env=env)
    result = L2Result(available=True, command=cmd, returncode=rc, timed_out=timed_out, wall_time_sec=wall)
    if timed_out:
        result.error = f"L2 probe exceeded {timeout}s timeout"
        return result

    kleidiai_lines = [ln for ln in out.splitlines() if "kleidiai" in ln.lower()]
    result.raw_kleidiai_lines = kleidiai_lines[:40]

    for m in _RE_PRIMARY_FEATURE.finditer(out):
        result.primary_kernel_feature[m.group(1)] = m.group(2)

    m = _RE_SME_ENABLED.search(out)
    if m:
        result.sme_enabled = True
        result.sme_variant = "SME2" if m.group(1) == "2" else "SME"
        result.sme_detail = m.group(2)
    elif _RE_SME_DISABLED.search(out):
        result.sme_enabled = False

    result.buffer_type_fallback_seen = bool(_RE_BUFFER_FALLBACK.search(out))
    result.unsupported_tensor_types = sorted(set(_RE_UNSUPPORTED_TENSOR.findall(out)))

    ftype_match = _RE_FILE_TYPE.search(out)
    if ftype_match:
        result.model_file_type = ftype_match.group(1)

    sysinfo = _RE_SYSTEM_INFO.search(out)
    if sysinfo:
        result.system_info_line = sysinfo.group(0).strip()

    if rc != 0:
        result.error = f"llama-cli exited {rc} (see raw_kleidiai_lines / rerun manually for full log)"

    return result


# --------------------------------------------------------------------------
# L3 DISPATCH -- the decisive layer. lldb (macOS) and gdb (Linux) drivers.
# --------------------------------------------------------------------------

_RE_LLDB_LOCATION = re.compile(
    r"^\s*\d+\.\d+:\s+where\s*=\s*[^`]*`(?P<sym>.+?),\s*address\s*=\s*0x[0-9a-fA-F]+,.*?hit count\s*=\s*(?P<hits>\d+)"
)
_RE_LLDB_SUMMARY = re.compile(
    r"^(?P<id>\d+):\s+regex\s*=\s*'(?P<regex>[^']*)',\s*locations\s*=\s*(?P<locs>\d+),"
    r"\s*resolved\s*=\s*(?P<resolved>\d+),\s*hit count\s*=\s*(?P<total>\d+)"
)


def _render_template(template_path: str, substitutions: dict) -> str:
    with open(template_path) as fh:
        text = fh.read()
    for key, val in substitutions.items():
        text = text.replace(f"__{key}__", val)
    return text


def run_l3_lldb(binary: str, model: str, threads: int, prompt: str, n_predict: int,
                 dispatch_regex: str, timeout: float, env: dict) -> L3Result:
    result = L3Result(debugger="lldb", available=True, dispatch_regex=dispatch_regex)
    lldb_bin = shutil.which("lldb")
    if not lldb_bin:
        result.available = False
        result.error = "lldb not found on PATH"
        return result

    script_text = _render_template(LLDB_TEMPLATE_PATH, {"DISPATCH_REGEX": dispatch_regex})
    with tempfile.NamedTemporaryFile("w", suffix=".lldb", delete=False) as fh:
        fh.write(script_text)
        script_path = fh.name

    target_args = [
        "-m", model, "-p", prompt, "-n", str(n_predict),
        "-no-cnv", "-st", "--simple-io", "-t", str(threads),
    ]
    cmd = [lldb_bin, "-b", "-s", script_path, "--", binary] + target_args
    result.command = cmd
    try:
        rc, out, timed_out, wall = _run_with_timeout(cmd, timeout, env=env)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    result.wall_time_sec = wall
    result.timed_out = timed_out
    if timed_out:
        result.error = f"L3 lldb probe exceeded {timeout}s timeout (killed process group)"
        result.completed = False
        return result

    hits_by_symbol = {}
    reported_total = None
    for line in out.splitlines():
        m = _RE_LLDB_LOCATION.match(line)
        if m:
            hits_by_symbol[m.group("sym")] = int(m.group("hits"))
            continue
        m = _RE_LLDB_SUMMARY.match(line.strip())
        if m:
            reported_total = int(m.group("total"))

    total = sum(hits_by_symbol.values())
    if reported_total is not None and reported_total != total:
        result.error = (
            f"per-location hit sum ({total}) != lldb's own breakpoint summary total "
            f"({reported_total}); reporting the per-location sum, but this mismatch "
            f"itself is worth investigating"
        )

    hits_by_family: dict = {}
    for sym, hits in hits_by_symbol.items():
        fam = classify_symbol_family(sym)
        hits_by_family[fam] = hits_by_family.get(fam, 0) + hits

    result.completed = True
    result.hits_by_symbol = hits_by_symbol
    result.hits_by_family = hits_by_family
    result.total_hits = total
    result.kernel_family_executed = (
        max(hits_by_family, key=hits_by_family.get) if hits_by_family and total > 0 else "none"
    )
    return result


def enumerate_dispatch_symbols(lib_path: Optional[str], dispatch_regex: str) -> list:
    """Return the concrete kernel-entry symbol names matching `dispatch_regex`.

    GDB needs explicit names rather than a regex: ggml dlopen's the CPU backend after
    process start, and `rbreak` -- unlike `break` -- does not create pending
    breakpoints, so a regex evaluated before `run` matches nothing and silently
    instruments nothing. See tools/dispatch_probe.gdb's header for the full story.
    """
    if not lib_path or not os.path.isfile(lib_path):
        return []
    nm_bin = shutil.which("nm")
    if not nm_bin:
        return []
    is_darwin = platform.system() == "Darwin"
    nm_args = [nm_bin, "-gU", lib_path] if is_darwin else [nm_bin, "-D", lib_path]
    try:
        out = subprocess.run(nm_args, capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    dispatch_re = re.compile(dispatch_regex)
    names = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        # Mach-O prefixes global symbols with an underscore; ELF does not.
        sym = parts[-1].lstrip("_") if is_darwin else parts[-1]
        if dispatch_re.match(sym):
            names.add(sym)
    return sorted(names)


def run_l3_gdb(binary: str, model: str, threads: int, prompt: str, n_predict: int,
               dispatch_regex: str, timeout: float, env: dict,
               lib_path: Optional[str] = None) -> L3Result:
    """GDB counterpart of run_l3_lldb.

    Sets one pending breakpoint per concrete kernel symbol (enumerated via nm) rather
    than a pre-`run` regex, and counts hits with a gdb.Breakpoint subclass whose
    stop() returns False so the inferior is never halted. Validated against a
    ground-truth dlopen harness on aarch64 Linux; see tools/dispatch_probe.gdb.
    """
    result = L3Result(debugger="gdb", available=True, dispatch_regex=dispatch_regex)
    gdb_bin = shutil.which("gdb")
    if not gdb_bin:
        result.available = False
        result.error = "gdb not found on PATH"
        return result

    symbols = enumerate_dispatch_symbols(lib_path, dispatch_regex)
    if not symbols:
        # Fail loudly. An uninstrumented probe returns zero hits, which is
        # indistinguishable from a genuine no-dispatch result -- exactly the
        # conflation this project exists to eliminate.
        result.available = False
        result.error = (
            "no symbols matching %r found in %r; refusing to run an uninstrumented "
            "probe that would report a misleading zero" % (dispatch_regex, lib_path)
        )
        return result

    script_text = _render_template(GDB_TEMPLATE_PATH, {"DP_SYMBOLS": repr(symbols)})
    with tempfile.NamedTemporaryFile("w", suffix=".gdb", delete=False) as fh:
        fh.write(script_text)
        script_path = fh.name

    target_args = [
        "-m", model, "-p", prompt, "-n", str(n_predict),
        "-no-cnv", "-st", "--simple-io", "-t", str(threads),
    ]
    cmd = [gdb_bin, "-q", "-batch", "-x", script_path, "--args", binary] + target_args
    result.command = cmd
    try:
        rc, out, timed_out, wall = _run_with_timeout(cmd, timeout, env=env)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    result.wall_time_sec = wall
    result.timed_out = timed_out
    if timed_out:
        result.error = f"L3 gdb probe exceeded {timeout}s timeout (killed process group)"
        result.completed = False
        return result

    hits_by_symbol = {}
    in_block = False
    breakpoints_created = None
    for line in out.splitlines():
        if line.startswith("DP_BREAKPOINTS_CREATED "):
            try:
                breakpoints_created = int(line.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if line.strip() == "DISPATCH_PROBE_RESULT_BEGIN":
            in_block = True
            continue
        if line.strip() == "DISPATCH_PROBE_RESULT_END":
            in_block = False
            continue
        if in_block and line.startswith("DP_HIT "):
            _, name, count = line.split(" ", 2)
            hits_by_symbol[name] = int(count)

    result.breakpoints_requested = len(symbols)
    result.breakpoints_created = breakpoints_created

    # If gdb instrumented nothing, a zero hit count means "the probe was broken",
    # not "the kernel did not run". Never let those two report identically -- that
    # confusion is precisely the bug this tool found in llama.cpp, and the bug that
    # this tool's own v1 gdb path shipped with.
    if not breakpoints_created:
        result.error = (
            "gdb created 0 of %d requested breakpoints -- probe was not instrumented, "
            "so a zero hit count here is meaningless" % len(symbols)
        )
        result.completed = False
        return result

    total = sum(hits_by_symbol.values())
    hits_by_family: dict = {}
    for sym, hits in hits_by_symbol.items():
        fam = classify_symbol_family(sym)
        hits_by_family[fam] = hits_by_family.get(fam, 0) + hits

    result.completed = True
    result.hits_by_symbol = hits_by_symbol
    result.hits_by_family = hits_by_family
    result.total_hits = total
    result.kernel_family_executed = (
        max(hits_by_family, key=hits_by_family.get) if hits_by_family and total > 0 else "none"
    )
    return result


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

def compute_verdict(advertised_family: Optional[str], l3: L3Result) -> str:
    """Classify advertised (L2) vs executed (L3) into one verdict string.

    This is the product: it is the ONE thing a reader should be able to
    scan down a column of ledger rows and immediately understand.
    """
    if not l3.available:
        return "L3_UNAVAILABLE"
    if l3.timed_out:
        return "L3_TIMEOUT"
    if not l3.completed:
        return "L3_ERROR"

    total = l3.total_hits
    if total == 0:
        return "NO_DISPATCH_OBSERVED"

    adv = (advertised_family or "").lower()
    adv_hits = l3.hits_by_family.get(adv, 0)
    other_hits = total - adv_hits

    if not adv or adv in ("none", "unknown"):
        dominant = l3.kernel_family_executed.upper()
        return f"{dominant}_EXECUTED_NO_ADVERTISED_FEATURE"
    if adv_hits > 0 and other_hits == 0:
        return f"{adv.upper()}_DISPATCHED"
    if adv_hits > 0 and other_hits > 0:
        return f"{adv.upper()}_HYBRID_DISPATCH"
    if adv_hits == 0 and other_hits > 0:
        return "SILENT_FALLBACK"
    return "UNKNOWN"


# Verdicts that mean "the advertised acceleration did not actually / fully
# dispatch" -- what --assert gates on by default.
DEFAULT_ASSERT_FAIL_VERDICTS = {"SILENT_FALLBACK", "NO_DISPATCH_OBSERVED"}


# --------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------

def build_config_record(platform_meta: dict, model: str, quant: str, threads: int,
                         workload: dict, env_record: dict,
                         l1: L1Result, l2: L2Result, l3: L3Result) -> dict:
    advertised = l2.primary_kernel_feature.get("q4") if l2.available else None
    verdict = compute_verdict(advertised, l3)
    return {
        "platform": platform_meta["platform"],
        "cpu_brand": platform_meta["cpu_brand"],
        "isa_features": platform_meta["isa_features"],
        "model": model,
        "quant": quant,
        "threads": threads,
        "workload": workload["name"],
        "workload_description": workload["description"],
        "n_predict": workload["n_predict"],
        "prompt_preview": (workload["prompt"][:60] + "...") if len(workload["prompt"]) > 60 else workload["prompt"],
        "env": env_record,
        "l1": asdict(l1),
        "l2": asdict(l2),
        "l3": {
            "debugger": l3.debugger,
            "available": l3.available,
            "completed": l3.completed,
            "timed_out": l3.timed_out,
            "error": l3.error,
            "dispatch_regex": l3.dispatch_regex,
            "wall_time_sec": round(l3.wall_time_sec, 2),
            "kernel_family_executed": l3.kernel_family_executed,
            "hits_by_symbol": l3.hits_by_symbol,
            "hits_by_family": l3.hits_by_family,
            "total_hits": l3.total_hits,
            # gdb lane: proof the probe was instrumented at all, so a reader can tell
            # "0 hits because the kernel did not run" from "0 hits because we never
            # set a breakpoint". null on the lldb lane, which resolves regex
            # breakpoints itself and has no equivalent counter.
            "breakpoints_requested": l3.breakpoints_requested,
            "breakpoints_created": l3.breakpoints_created,
        },
        "advertised_family": advertised,
        "verdict": verdict,
    }


def render_markdown_table(configs: list) -> str:
    lines = [
        "| threads | workload | advertised (L2) | executed (L3) | hits (adv/other) | verdict |",
        "|---:|---|---|---|---|---|",
    ]
    for c in configs:
        l3 = c["l3"]
        adv = c["advertised_family"] or "-"
        adv_hits = l3["hits_by_family"].get((c["advertised_family"] or "").lower(), 0)
        other_hits = l3["total_hits"] - adv_hits
        lines.append(
            f"| {c['threads']} | {c['workload']} | {adv.upper() if adv != '-' else '-'} | "
            f"{l3['kernel_family_executed']} | {adv_hits}/{other_hits} | **{c['verdict']}** |"
        )
    return "\n".join(lines)


def guess_quant_from_l2(configs_l2: list) -> str:
    """Derive the model's overall quant from llama.cpp's OWN model-card line
    (`print_info: file type = Q4_0` / `ftype : Q4_0`), never from the model
    filename -- the filename in this project (q05.gguf) does not encode the
    quant, so guessing from it would be fabricating a fact this tool never
    actually observed. Falls back to reporting the (different!) set of
    tensor types KleidiAI explicitly declined to accelerate, which is
    useful context but is NOT the overall model quant, if the model-card
    line was somehow never seen (e.g. every L2 run errored out).
    """
    file_types = {l2.model_file_type for l2 in configs_l2 if l2.model_file_type}
    if file_types:
        return "/".join(sorted(file_types))
    seen = set()
    for l2 in configs_l2:
        for t in l2.unsupported_tensor_types:
            seen.add(t)
    if seen:
        return ("[model file type not observed; tensor types KleidiAI declined to "
                 "accelerate: " + "/".join(sorted(seen)) + " -- pass --quant to override]")
    return "[unknown -- not observed in any L2 log line; pass --quant to override]"


def parse_env_overrides(pairs: list) -> dict:
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise argparse.ArgumentTypeError(f"--env expects KEY=VALUE, got: {p}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Symbol-level verifier: did the accelerated (KleidiAI) kernel ACTUALLY dispatch?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--binary", required=True, help="path to llama-cli (or similar) executable")
    ap.add_argument("--model", required=True, help="path to the .gguf model")
    ap.add_argument("--lib-dir", default=None, help="dir to search for the CPU backend shared lib (default: dirname(--binary))")
    ap.add_argument("--lib-name", default=None, help="explicit path to the CPU backend shared lib, overrides --lib-dir search")
    ap.add_argument("--threads", default=",".join(str(t) for t in DEFAULT_THREADS),
                     help="comma-separated thread counts to sweep (default: %(default)s)")
    ap.add_argument("--workloads", default="all",
                     help="comma-separated workload names to run, or 'all' (default). "
                          f"Built-in workloads: {', '.join(w['name'] for w in DEFAULT_WORKLOADS)}")
    ap.add_argument("--dispatch-regex", default=DEFAULT_DISPATCH_REGEX,
                     help="anchored function-name regex for the L3 breakpoint (default: %(default)s)")
    ap.add_argument("--quant", default=None, help="override the quant label in the ledger (default: inferred from L2 logs)")
    ap.add_argument("--l2-timeout", type=float, default=60.0, help="per-run L2 (SELECT) timeout, seconds")
    ap.add_argument("--l3-timeout", type=float, default=240.0, help="per-run L3 (DISPATCH) timeout, seconds")
    ap.add_argument("--l3-debugger", choices=["auto", "lldb", "gdb", "none"], default="auto",
                     help="which debugger drives L3 (default: auto -> lldb on Darwin, gdb on Linux, none otherwise)")
    ap.add_argument("--skip-l3", action="store_true", help="skip L3 entirely (fast L1+L2-only smoke test)")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable; applied to both L2 and L3 subprocess environments")
    ap.add_argument("--out", default=None, help="output ledger JSON path (default: results/dispatch-ledger-<platform>.json)")
    ap.add_argument("--assert", dest="do_assert", action="store_true",
                     help="exit non-zero if any config's verdict is in " + ", ".join(sorted(DEFAULT_ASSERT_FAIL_VERDICTS)))
    args = ap.parse_args(argv)

    binary = os.path.abspath(args.binary)
    model = os.path.abspath(args.model)
    if not os.path.isfile(binary):
        print(f"ERROR: --binary not found: {binary}", file=sys.stderr)
        return 2
    if not os.path.isfile(model):
        print(f"ERROR: --model not found: {model}", file=sys.stderr)
        return 2

    threads_list = [int(t.strip()) for t in args.threads.split(",") if t.strip()]

    if args.workloads == "all":
        workloads = DEFAULT_WORKLOADS
    else:
        wanted = {w.strip() for w in args.workloads.split(",") if w.strip()}
        workloads = [w for w in DEFAULT_WORKLOADS if w["name"] in wanted]
        if not workloads:
            print(f"ERROR: no matching workloads for --workloads={args.workloads!r}", file=sys.stderr)
            return 2

    debugger = args.l3_debugger
    if debugger == "auto":
        if args.skip_l3:
            debugger = "none"
        elif platform.system() == "Darwin":
            debugger = "lldb"
        elif shutil.which("gdb"):
            debugger = "gdb"
        else:
            debugger = "none"

    env_overrides = parse_env_overrides(args.env)
    run_env = dict(os.environ)
    run_env.update(env_overrides)

    lib_path = find_cpu_backend_lib(binary, args.lib_dir, args.lib_name)
    l1 = run_l1_static(lib_path, args.dispatch_regex)

    platform_meta = {
        "platform": platform_id(),
        "cpu_brand": detect_cpu_brand(),
        "isa_features": detect_isa_features(),
    }
    env_record = {
        "requested_env_overrides": env_overrides,
        "kleidiai_related_env": {k: run_env.get(k) for k in KLEIDIAI_ENV_VARS},
    }

    configs = []
    l2_results_for_quant_guess = []

    print(f"# verify_dispatch.py sweep -- {platform_meta['platform']} / {platform_meta['cpu_brand']}", file=sys.stderr)
    print(f"# binary={binary}", file=sys.stderr)
    print(f"# model={model}", file=sys.stderr)
    print(f"# cpu backend lib={lib_path}", file=sys.stderr)
    print(f"# threads={threads_list} workloads={[w['name'] for w in workloads]} l3_debugger={debugger}", file=sys.stderr)

    for workload in workloads:
        for threads in threads_list:
            print(f"--- threads={threads} workload={workload['name']} ---", file=sys.stderr)
            l2 = run_l2_select(binary, model, threads, workload["prompt"], workload["n_predict"],
                                args.l2_timeout, run_env)
            l2_results_for_quant_guess.append(l2)
            print(f"  L2: primary_kernel_feature={l2.primary_kernel_feature} "
                  f"sme_enabled={l2.sme_enabled} buffer_fallback={l2.buffer_type_fallback_seen}", file=sys.stderr)

            if debugger == "lldb":
                l3 = run_l3_lldb(binary, model, threads, workload["prompt"], workload["n_predict"],
                                  args.dispatch_regex, args.l3_timeout, run_env)
            elif debugger == "gdb":
                l3 = run_l3_gdb(binary, model, threads, workload["prompt"], workload["n_predict"],
                                 args.dispatch_regex, args.l3_timeout, run_env,
                                 lib_path=lib_path)
            else:
                l3 = L3Result(debugger="none", available=False, error="L3 skipped (--skip-l3 or no debugger found)",
                               dispatch_regex=args.dispatch_regex)
            print(f"  L3: debugger={l3.debugger} completed={l3.completed} timed_out={l3.timed_out} "
                  f"kernel_family_executed={l3.kernel_family_executed} total_hits={l3.total_hits} "
                  f"({l3.wall_time_sec:.1f}s)", file=sys.stderr)

            quant = args.quant or "[pending -- resolved after full sweep]"
            record = build_config_record(platform_meta, model, quant, threads, workload, env_record, l1, l2, l3)
            print(f"  VERDICT: {record['verdict']}", file=sys.stderr)
            configs.append(record)

    quant_final = args.quant or guess_quant_from_l2(l2_results_for_quant_guess)
    for record in configs:
        if record["quant"] == "[pending -- resolved after full sweep]":
            record["quant"] = quant_final

    ledger = {
        "schema_version": 1,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "verify_dispatch.py",
        "generator_argv": sys.argv,
        "binary": binary,
        "cpu_backend_lib": lib_path,
        "model": model,
        "configs": configs,
    }

    out_path = args.out or os.path.join(REPO_ROOT, "results", f"dispatch-ledger-{platform_meta['platform']}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(ledger, fh, indent=2, sort_keys=False)
    print(f"\nWrote ledger: {out_path}", file=sys.stderr)

    print("\n" + render_markdown_table(configs))

    if args.do_assert:
        failing = [c for c in configs if c["verdict"] in DEFAULT_ASSERT_FAIL_VERDICTS]
        if failing:
            print(f"\nASSERT FAILED: {len(failing)}/{len(configs)} configs show the advertised "
                  f"accelerated kernel NOT actually dispatching:", file=sys.stderr)
            for c in failing:
                print(f"  threads={c['threads']} workload={c['workload']} "
                      f"advertised={c['advertised_family']} verdict={c['verdict']}", file=sys.stderr)
            return 1
        print(f"\nASSERT OK: all {len(configs)} configs show the advertised kernel dispatching "
              f"(or, where fallback is legitimate, no mismatch detected).", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
