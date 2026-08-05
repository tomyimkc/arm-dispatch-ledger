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
import hashlib
import json
import os
import platform
import re
import shlex
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


def classify_symbol_family(symbol_name: str, family_order: Optional[list] = None) -> str:
    """Classify a kai_run_matmul_* (or similar) symbol into a kernel family.

    Based on KleidiAI's own naming convention, e.g.:
      kai_run_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa -> "sme2"
      kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod -> "dotprod"
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm      -> "i8mm"
    Returns "other" if no known token is found (e.g. a completely different
    naming scheme was linked in).

    `family_order` lets a declarative target (tools/targets/*.json, see
    tools/polygraph) supply its OWN ordered token list for a symbol naming
    convention that isn't KleidiAI's -- e.g. "gemm"/"gemv" for ggml's repack
    dispatch templates (see tools/targets/llama-cpp-cpu-variants.json). Order
    still matters for the same reason the default list is ordered: put more
    specific tokens before substrings of themselves. Defaults to the original
    KleidiAI-derived _FAMILY_ORDER so every existing call site (which passes
    only `symbol_name`) is unaffected.
    """
    n = symbol_name.lower()
    for family in (family_order if family_order is not None else _FAMILY_ORDER):
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

def nm_symbol_lines(nm_bin: str, path: str, is_macho: bool, timeout: int = 30) -> list[str]:
    """Return nm output lines for `path`, covering both ELF symbol tables.

    Neither ELF nm mode is sufficient alone, and picking one silently loses symbols:

    * ``nm -D`` reads only ``.dynsym``. A normal (non-``-rdynamic``) executable exports
      nothing there, so scanning a *binary* rather than a shared library finds zero symbols
      -- which this tool then correctly, and uselessly, reports as "refusing to run an
      uninstrumented probe".
    * ``nm -gU`` reads ``.symtab``, which a **stripped** shared library does not have --
      and production libraries are routinely stripped, which is exactly the case this
      project cares about.

    So run both and union them. Being wrong about which symbols exist is the one failure
    this tool cannot afford: a missed symbol becomes "0 dispatches", which reads as a
    finding rather than as a blind spot.

    Mach-O has a single symbol table and ``-gU`` covers it; ``-D`` is not meaningful there.
    """
    modes = [["-gU"]] if is_macho else [["-gU"], ["-D"]]
    seen, lines = set(), []
    for mode in modes:
        try:
            out = subprocess.run([nm_bin] + mode + [path], capture_output=True,
                                 text=True, timeout=timeout)
        except (subprocess.SubprocessError, OSError):
            continue
        for ln in out.stdout.splitlines():
            if ln and ln not in seen:
                seen.add(ln)
                lines.append(ln)
    return lines


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
            nm_lines = nm_symbol_lines(nm_bin, lib_path, is_macho=(platform.system() == "Darwin"))
        except Exception as e:
            return L1Result(lib_path=lib_path, tool_chain=tool_chain, available=False, error=f"nm failed: {e}")
        for line in nm_lines:
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
        # -gU (extern-only, defined-only), NOT -D (dynamic-symbol-table-only): GNU nm's -D
        # only lists what the dynamic linker would resolve, which is a strict subset of a
        # shared library's real exported symbols and is EMPTY for a plain (non-`-rdynamic`)
        # executable -- a scan_target the tool explicitly supports scanning directly (see
        # resolve_target_artifacts()'s docstring). -gU reads the regular symbol table
        # instead, filtered to external+defined, which finds the same symbols -D does for a
        # real dlopen'd .so (verified byte-identical against tests/l3_gdb_groundtruth's
        # ground-truth libkai_fake.so) while also working for a plain executable. This is
        # the same flag string already used for Mach-O below -- GNU nm's -g/-U happen to
        # mean the same thing as macOS nm's.
        nm_lines = nm_symbol_lines(nm_bin, lib_path, is_macho=(platform.system() == "Darwin"))
    except Exception as e:
        return L1Result(lib_path=lib_path, tool_chain=tool_chain, available=False, error=f"nm failed: {e}")
    for line in nm_lines:
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


def run_l3_lldb_argv(binary: str, target_args: list, dispatch_regex: str, timeout: float, env: dict,
                      family_order: Optional[list] = None) -> L3Result:
    """Argv-generic core of run_l3_lldb(). `target_args` is whatever argv the target
    binary needs (llama-cli's `-m/-p/-n/...` flags, or any other CLI shape) -- this
    function has no llama.cpp-specific knowledge itself, which is what lets
    tools/polygraph drive arbitrary binaries via a declarative target's
    workload.arg_template (see tools/targets/*.json) through the exact same code
    path verify_dispatch.py's own sweep uses. run_l3_lldb() below is now a thin
    wrapper that builds llama.cpp's fixed argv and calls this. `family_order`
    overrides classify_symbol_family()'s default KleidiAI token list for targets
    with a different symbol naming convention (default: None, i.e. unchanged).
    """
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

    cmd = [lldb_bin, "-b", "-s", script_path, "--", binary] + list(target_args)
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
        fam = classify_symbol_family(sym, family_order)
        hits_by_family[fam] = hits_by_family.get(fam, 0) + hits

    result.completed = True
    result.hits_by_symbol = hits_by_symbol
    result.hits_by_family = hits_by_family
    result.total_hits = total
    result.kernel_family_executed = (
        max(hits_by_family, key=hits_by_family.get) if hits_by_family and total > 0 else "none"
    )
    return result


def run_l3_lldb(binary: str, model: str, threads: int, prompt: str, n_predict: int,
                 dispatch_regex: str, timeout: float, env: dict) -> L3Result:
    """llama.cpp-shaped wrapper kept byte-identical for the existing CLI and for
    tests/l3_lldb_groundtruth/run_test.sh, which calls this exact signature
    directly. Builds llama-cli's fixed argv and delegates to run_l3_lldb_argv()."""
    target_args = [
        "-m", model, "-p", prompt, "-n", str(n_predict),
        "-no-cnv", "-st", "--simple-io", "-t", str(threads),
    ]
    return run_l3_lldb_argv(binary, target_args, dispatch_regex, timeout, env)


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
    # -gU on both platforms (extern-only, defined-only): NOT ELF's -D (dynamic-symbol-table
    # only), which is empty for a plain executable whose dispatch symbols were never
    # exported for dynamic linking -- see scan_symbols_generic()'s identical fix for the
    # full rationale and the ground-truth verification against a real dlopen'd .so.
    try:
        nm_lines = nm_symbol_lines(nm_bin, lib_path, is_macho=(platform.system() == "Darwin"))
    except Exception:
        return []
    dispatch_re = re.compile(dispatch_regex)
    names = set()
    for line in nm_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        # Mach-O prefixes global symbols with an underscore; ELF does not.
        sym = parts[-1].lstrip("_") if is_darwin else parts[-1]
        if dispatch_re.match(sym):
            names.add(sym)
    return sorted(names)


def run_l3_gdb_argv(binary: str, target_args: list, dispatch_regex: str, timeout: float, env: dict,
                     lib_path: Optional[str] = None, family_order: Optional[list] = None) -> L3Result:
    """Argv-generic core of run_l3_gdb() -- the gdb counterpart of run_l3_lldb_argv().

    Sets one pending breakpoint per concrete kernel symbol (enumerated via nm) rather
    than a pre-`run` regex, and counts hits with a gdb.Breakpoint subclass whose
    stop() returns False so the inferior is never halted. Validated against a
    ground-truth dlopen harness on aarch64 Linux; see tools/dispatch_probe.gdb.
    `target_args` is generic argv (see run_l3_lldb_argv's docstring); run_l3_gdb()
    below is now a thin wrapper that builds llama.cpp's fixed argv and calls this.
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

    cmd = [gdb_bin, "-q", "-batch", "-x", script_path, "--args", binary] + list(target_args)
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
        fam = classify_symbol_family(sym, family_order)
        hits_by_family[fam] = hits_by_family.get(fam, 0) + hits

    result.completed = True
    result.hits_by_symbol = hits_by_symbol
    result.hits_by_family = hits_by_family
    result.total_hits = total
    result.kernel_family_executed = (
        max(hits_by_family, key=hits_by_family.get) if hits_by_family and total > 0 else "none"
    )
    return result


def run_l3_gdb(binary: str, model: str, threads: int, prompt: str, n_predict: int,
               dispatch_regex: str, timeout: float, env: dict,
               lib_path: Optional[str] = None) -> L3Result:
    """llama.cpp-shaped wrapper kept byte-identical for the existing CLI. Builds
    llama-cli's fixed argv and delegates to run_l3_gdb_argv()."""
    target_args = [
        "-m", model, "-p", prompt, "-n", str(n_predict),
        "-no-cnv", "-st", "--simple-io", "-t", str(threads),
    ]
    return run_l3_gdb_argv(binary, target_args, dispatch_regex, timeout, env, lib_path=lib_path)


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
# GENERIC TARGET-DEFINITION SUPPORT (tools/polygraph, tools/targets/*.json)
# --------------------------------------------------------------------------
# Everything above this line is llama.cpp/KleidiAI-specific and predates
# tools/polygraph; nothing above was changed in behavior by what follows (see
# tests/ for the byte-identical-output check). Everything below is generic:
# driven by a declarative target JSON instead of hardcoded flags/regexes, so
# the same L1/L2/L3 machinery above can be pointed at a different binary
# (ollama, whisper.cpp, onnxruntime, pytorch, an ad-hoc --binary/--symbols
# pair, ...) without a code change -- only a new JSON file under
# tools/targets/. tools/polygraph is the CLI that drives this; run
# `tools/polygraph explain <target>` for a human-readable dump of one
# target's schema, or read tools/targets/llama-cpp-kleidiai.json (the
# reference target -- it encodes this file's OWN original KleidiAI behavior
# declaratively and is asserted to reproduce it; see tests/target_definition/).

TARGETS_DIR = os.path.join(TOOLS_DIR, "targets")


def list_target_names() -> list:
    """Every tools/targets/*.json target name, sorted. Used by `polygraph list`."""
    if not os.path.isdir(TARGETS_DIR):
        return []
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(TARGETS_DIR, "*.json"))
    )


def load_target(name_or_path: str) -> dict:
    """Load one target definition, by built-in name (tools/targets/<name>.json)
    or an explicit path to any JSON file. Also resolves `aliases` declared by a
    built-in target's own JSON (e.g. README.md's illustrative `polygraph check
    kleidiai` example resolves to llama-cpp-kleidiai via its "aliases" list).
    Raises FileNotFoundError with a clear, actionable message on no match.
    """
    candidates = [name_or_path, os.path.join(TARGETS_DIR, f"{name_or_path}.json")]
    if not os.path.isfile(name_or_path):
        for path in sorted(glob.glob(os.path.join(TARGETS_DIR, "*.json"))):
            try:
                with open(path) as fh:
                    doc = json.load(fh)
            except (OSError, json.JSONDecodeError):
                continue
            if name_or_path in (doc.get("aliases") or []):
                candidates.append(path)
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as fh:
                target = json.load(fh)
            target["_path"] = os.path.abspath(path)
            return target
    raise FileNotFoundError(
        f"no target named {name_or_path!r} (looked in {TARGETS_DIR}); "
        f"run `polygraph list` to see built-in targets, or pass a path to your own JSON"
    )


def find_artifact_by_globs(search_dir: str, patterns: list) -> Optional[str]:
    """Generic sibling of find_cpu_backend_lib(): the first glob pattern with any
    match wins, preferring an unversioned symlink (foo.dylib/foo.so over
    foo.1.2.3.dylib) purely for a cleaner reported path -- same file either way.
    Supports recursive '**' patterns, since not every target's build layout puts
    its shared lib next to the binary (e.g. onnxruntime's installed package dir).
    """
    for pattern in patterns or []:
        matches = sorted(glob.glob(os.path.join(search_dir, pattern), recursive=True))
        if not matches:
            continue
        for m in matches:
            if re.search(r"\.(dylib|so)$", m):
                return os.path.abspath(m)
        return os.path.abspath(matches[0])
    return None


def resolve_target_artifacts(target: dict, binary: str, lib_dir: Optional[str],
                              lib_name: Optional[str]) -> dict:
    """Resolve {'binary', 'lib', 'platform_key'} for a target given an explicit
    --binary. `lib` is None if the target declares no lib_globs for this
    platform (or none matched) -- callers fall back to scanning the binary
    itself, which is correct for a target whose fast-path symbols live in the
    main executable rather than a separate shared library.
    """
    plat_key = "darwin" if platform.system() == "Darwin" else "linux"
    plat_cfg = (target.get("platforms") or {}).get(plat_key, {})
    binary_abs = os.path.abspath(binary)
    if lib_name:
        lib = os.path.abspath(lib_name) if os.path.isfile(lib_name) else None
    else:
        search_dir = lib_dir or os.path.dirname(binary_abs)
        lib = find_artifact_by_globs(search_dir, plat_cfg.get("lib_globs") or [])
    return {"binary": binary_abs, "lib": lib, "platform_key": plat_key}


def detect_artifact_format(path: str) -> str:
    """Sniff whether `path` is a Mach-O or ELF binary from its magic bytes,
    independent of the HOST platform -- what lets scan_symbols_generic()
    correctly nm-scan a foreign-platform artifact (e.g. inspecting a Linux ELF
    .so from a macOS host with llvm-nm, which understands both formats fine --
    it's the *flags* (-gU vs -D) and the underscore-stripping convention that
    are platform-of-the-ARTIFACT-specific, not platform-of-the-host). Falls
    back to the host platform's own convention if the file can't be read or
    matches neither known magic, so behavior for a same-platform artifact --
    the common case, and the only case run_l1_static()/enumerate_dispatch_
    symbols() ever handle -- is unchanged.
    """
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError:
        return "macho" if platform.system() == "Darwin" else "elf"
    if magic == b"\x7fELF":
        return "elf"
    if magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
        return "macho"
    return "macho" if platform.system() == "Darwin" else "elf"


def scan_symbols_generic(artifact_path: Optional[str], symbol_prefix: Optional[str],
                          dispatch_regex: str, family_order: Optional[list]) -> L1Result:
    """Generic L1 static scan: like run_l1_static() but with a configurable
    (optional) symbol-name prefix filter instead of the hardcoded 'kai_' prefix,
    and a configurable family-classification order. Exported-symbol scan (nm)
    only -- unlike run_l1_static() this does NOT also run the ISA-mnemonic
    disassembly scan (otool -tv / objdump -d): that scan's mnemonic sets
    (SME_MNEMONICS etc.) are themselves KleidiAI/Arm-SME-specific and would be
    actively misleading applied to an unrelated target (an x86 binary, or a
    target whose 'fast path' isn't a distinct ISA extension at all -- e.g.
    onnxruntime's execution-provider graph partitioning). isa_mnemonic_counts is
    always {} here; L1Result keeps the field only for schema compatibility with
    run_l1_static()'s output. Artifact FORMAT is detected from the file itself
    (detect_artifact_format), not assumed from the host OS -- see its docstring.
    """
    if not artifact_path or not os.path.isfile(artifact_path):
        fallback_chain = "macho" if platform.system() == "Darwin" else "elf"
        return L1Result(lib_path=artifact_path, tool_chain=fallback_chain, available=False,
                         error="artifact not found (pass --binary/--lib-dir/--lib-name explicitly)")
    tool_chain = detect_artifact_format(artifact_path)
    nm_bin = shutil.which("nm")
    if not nm_bin:
        return L1Result(lib_path=artifact_path, tool_chain=tool_chain, available=False,
                         error="missing tool: nm")
    dispatch_re = re.compile(dispatch_regex)
    # -gU (extern-only, defined-only) on both formats -- NOT ELF's -D (dynamic-symbol-table
    # only). -D is empty for a plain executable (this function's whole "scan_target: binary"
    # case -- see resolve_target_artifacts()'s docstring) whose symbols were never exported
    # for dynamic linking, e.g. examples/catch-a-liar's fixture: `nm -D` finds nothing for
    # fast_path_sum() because it's a normal non-`-rdynamic` PIE executable, while `nm -gU`
    # finds it in the regular symbol table -- and finds the exact same symbols `-D` did for
    # a real dlopen'd .so (verified byte-identical against tests/l3_gdb_groundtruth's
    # ground-truth libkai_fake.so). See run_l1_static()'s identical fix for more detail.
    try:
        nm_lines = nm_symbol_lines(nm_bin, artifact_path, is_macho=(tool_chain == "macho"))
    except Exception as e:
        return L1Result(lib_path=artifact_path, tool_chain=tool_chain, available=False, error=f"nm failed: {e}")

    total_count = 0
    matched_count = 0
    families: dict = {}
    for line in nm_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        sym = parts[-1].lstrip("_") if tool_chain == "macho" else parts[-1]
        if symbol_prefix and not sym.startswith(symbol_prefix):
            continue
        total_count += 1
        if dispatch_re.match(sym):
            matched_count += 1
            fam = classify_symbol_family(sym, family_order)
            families[fam] = families.get(fam, 0) + 1

    return L1Result(
        lib_path=artifact_path, tool_chain=tool_chain, available=True,
        kai_symbol_count=total_count, kai_run_matmul_symbol_count=matched_count,
        kai_symbols_by_family=families, isa_mnemonic_counts={},
    )


def run_l2_workload(cmd: list, timeout: float, env: dict, patterns: list) -> dict:
    """Generic L2: run `cmd`, regex-match its combined stdout+stderr against a
    target's declared patterns. Each pattern is {'label', 'regex', 'group'
    (default 0, meaning 'matched at all' -> True, no captured value)}. Returns a
    plain dict (not an L2Result dataclass -- L2Result's fields are shaped for
    llama.cpp's specific log lines and don't generalize; run_l2_select() above
    remains the KleidiAI-specific reader used by the byte-identical old CLI).
    """
    rc, out, timed_out, wall = _run_with_timeout(cmd, timeout, env=env)
    result = {
        "available": True, "command": cmd, "returncode": rc, "timed_out": timed_out,
        "error": None, "wall_time_sec": wall, "matches": {}, "raw_matched_lines": [],
    }
    if timed_out:
        result["error"] = f"L2 probe exceeded {timeout}s timeout"
        return result
    matched_lines = []
    for pat in patterns or []:
        label = pat.get("label", pat.get("regex"))
        try:
            regex = re.compile(pat["regex"])
        except re.error as e:
            result["matches"][label] = f"[bad regex: {e}]"
            continue
        m = regex.search(out)
        if not m:
            continue
        group = pat.get("group", 0)
        try:
            value = m.group(group) if group else True
        except (IndexError, re.error):
            value = True
        result["matches"][label] = value
        matched_lines.append(m.group(0))
    result["raw_matched_lines"] = matched_lines[:40]
    if rc != 0 and not result["matches"]:
        result["error"] = f"command exited {rc} with no configured L2 pattern matching (see raw output)"
    return result


def cli_exit_code(advertised_family: Optional[str], l3: Optional[L3Result]) -> int:
    """Map an L3 dispatch result to tools/polygraph's contractual exit codes:
      0 - advertised capability matches what executed
      1 - mismatch: something claimed acceleration that did not run
      2 - undetermined (missing debugger, no permission, binary not found -- or
          nothing was ever advertised, so there is no claim to confirm/refute)
    Never returns 0 on missing/failed evidence -- the CLI contract's "never
    silently 0" rule, enforced here independently of compute_verdict()'s string
    (which conflates a couple of these cases under one label for the ledger's
    markdown table -- fine for a human skimming rows, not precise enough for a
    scripted exit code).
    """
    if l3 is None or not l3.available or l3.timed_out or not l3.completed:
        return 2
    adv = (advertised_family or "").strip().lower()
    if not adv or adv in ("none", "unknown"):
        return 2
    if l3.total_hits == 0:
        return 1
    if l3.hits_by_family.get(adv, 0) > 0:
        return 0
    return 1


def render_l2_only_verdict(l2_matches: dict, fallback_indicator_label: Optional[str]) -> str:
    """Verdict for a target with l3.enabled=false (see e.g.
    tools/targets/onnxruntime.json): there is no debugger-hit-count layer to
    compare a claim against, so the verdict is derived straight from whether
    the target's declared 'fallback indicator' L2 pattern matched -- e.g.
    onnxruntime's own verbose log literally saying a node was placed on
    CPUExecutionProvider after CoreMLExecutionProvider was requested first.
    """
    if fallback_indicator_label and fallback_indicator_label in l2_matches:
        return "SILENT_FALLBACK"
    if l2_matches:
        return "L2_ADVERTISED_NO_FALLBACK_DETECTED"
    return "L2_NO_SIGNAL"


def l2_only_exit_code(verdict: str) -> int:
    if verdict == "SILENT_FALLBACK":
        return 1
    if verdict == "L2_ADVERTISED_NO_FALLBACK_DETECTED":
        return 0
    return 2


def render_headline(verdict: str, target_name: str, advertised_family: Optional[str],
                     l3: Optional[L3Result], l1_matched: int, l1_total: int) -> str:
    """One human-readable line a non-expert can read and understand -- the CLI
    contract's 'human output must lead with a one-line verdict' requirement."""
    adv = (advertised_family or "").upper()
    if verdict in ("L3_UNAVAILABLE", "L3_ERROR"):
        detail = (l3.error if l3 else None) or "L3 (dispatch) could not run"
        return f"UNDETERMINED: {target_name} -- {detail}"
    if verdict == "L3_TIMEOUT":
        return f"UNDETERMINED: {target_name} -- L3 (dispatch) probe timed out before the workload finished"
    if verdict == "SILENT_FALLBACK":
        return (f"MISMATCH: {target_name} advertises {adv} but 0 of "
                f"{l3.total_hits if l3 else 0} dispatched kernel hits were {adv}")
    if verdict == "NO_DISPATCH_OBSERVED":
        if adv:
            return f"MISMATCH: {target_name} advertises {adv} but 0 kernel hits were observed at all"
        return f"UNDETERMINED: {target_name} -- 0 kernel hits observed and nothing was advertised to check against"
    if verdict.endswith("_HYBRID_DISPATCH"):
        adv_hits = l3.hits_by_family.get(adv.lower(), 0) if l3 else 0
        return (f"MATCH (hybrid): {target_name} advertises {adv} and it dispatched "
                f"({adv_hits}/{l3.total_hits if l3 else 0} hits were {adv}, the rest a different family)")
    if verdict.endswith("_DISPATCHED_ADHOC"):
        return f"MATCH: {target_name} -- {l3.total_hits if l3 else 0} dispatch hits observed for --symbols"
    if verdict.endswith("_DISPATCHED"):
        return f"MATCH: {target_name} advertises {adv} and it dispatched ({l3.total_hits if l3 else 0} hits, all {adv})"
    if verdict.endswith("_EXECUTED_NO_ADVERTISED_FEATURE"):
        dominant = verdict.split("_EXECUTED_")[0]
        return (f"UNDETERMINED: {target_name} -- {dominant} kernels dispatched "
                f"({l3.total_hits if l3 else 0} hits) but this target has no selection-log claim to check them against")
    if verdict == "L2_ADVERTISED_NO_FALLBACK_DETECTED":
        return f"MATCH: {target_name} -- no fallback indicator matched in the L2 (selection) log"
    if verdict == "L2_NO_SIGNAL":
        return f"UNDETERMINED: {target_name} -- no L2 (selection) pattern matched; nothing to verify"
    if verdict == "L1_ONLY":
        return f"L1 ONLY ({target_name}): {l1_matched}/{l1_total} matching symbols present (checked no further)"
    return f"UNKNOWN verdict ({verdict}) for {target_name}"


def build_workload_command(binary: str, workload: dict, params: dict) -> list:
    """Render a target's workload.arg_template against `params` (target defaults
    merged with caller overrides) into a concrete argv, prefixed with `binary`.
    Raises KeyError (naming the missing placeholder) if a required substitution
    is missing -- fails loudly rather than shipping a literal unsubstituted
    '{model}' into a subprocess argv."""
    template = (workload or {}).get("arg_template") or []
    args = []
    for token in template:
        try:
            args.append(token.format(**params))
        except KeyError as e:
            raise KeyError(f"workload.arg_template needs {e}; pass it via --model/--param") from e
    return [binary] + args


def run_target_check(target: dict, binary: str, lib_dir: Optional[str], lib_name: Optional[str],
                      params: dict, max_level: int, l3_debugger: str,
                      l2_timeout: float, l3_timeout: float, env_overrides: dict,
                      raw_run_cmd: Optional[list] = None, implicit_claim: bool = False) -> dict:
    """Run L1 (+L2 +L3, up to max_level) for one declarative target against one
    real --binary, and return a single JSON-serializable result dict. This is
    tools/polygraph's `check` subcommand's entire implementation -- it lives
    here (not in tools/polygraph) so both the CLI and tests/target_definition/
    can call the exact same code, mirroring how
    tests/l3_lldb_groundtruth/run_test.sh already drives run_l3_lldb() directly
    rather than reimplementing it.

    `raw_run_cmd`, when given (ad-hoc `--run "CMD"` mode), is used verbatim as
    the workload argv instead of rendering target['workload']['arg_template'].
    `implicit_claim`, when True (ad-hoc mode's default, OR when a target's own
    JSON sets its top-level "implicit_claim": true -- see e.g.
    tools/targets/catch-a-liar.json, whose whole point is "a matching symbol
    exists" being the entire claim, with no separate named-family selection
    log to compare against), treats "the user chose this
    --symbols regex" itself as the claim being checked -- there is no separate
    L2 selection-log claim to compare against, so pass/fail becomes "did any
    matching symbol dispatch at all", not "did the *advertised* family match".
    """
    name = target.get("name", "unknown-target")
    implicit_claim = implicit_claim or bool(target.get("implicit_claim", False))
    artifacts = resolve_target_artifacts(target, binary, lib_dir, lib_name)
    l1_cfg = target.get("l1") or {}
    l2_cfg = target.get("l2") or {}
    l3_cfg = target.get("l3") or {}
    workload_cfg = target.get("workload") or {}

    run_env = dict(os.environ)
    run_env.update(env_overrides or {})

    result = {
        "target": name,
        "tested_preset": bool(target.get("tested", False)),
        "platform": platform_id(),
        "cpu_brand": detect_cpu_brand(),
        "binary": artifacts["binary"],
        "lib": artifacts["lib"],
        "level_requested": max_level,
        "level_reached": 0,
        "l1": None, "l2": None, "l3": None,
        "advertised": None,
        "verdict": None,
        "headline": None,
        "exit_code": 2,
    }

    # ---- L1 (static) ----
    l1 = None
    if l1_cfg.get("enabled", True):
        scan_target = artifacts["lib"] if l1_cfg.get("scan_target", "lib") == "lib" else artifacts["binary"]
        l1 = scan_symbols_generic(
            scan_target, l1_cfg.get("symbol_prefix"), l1_cfg.get("dispatch_regex", "."),
            l1_cfg.get("family_order"),
        )
        result["l1"] = {
            "available": l1.available, "error": l1.error, "artifact": l1.lib_path,
            "tool_chain": l1.tool_chain, "total_symbol_count": l1.kai_symbol_count,
            "matched_symbol_count": l1.kai_run_matmul_symbol_count,
            "symbols_by_family": l1.kai_symbols_by_family,
        }
        result["level_reached"] = 1
    else:
        result["l1"] = {"available": False, "error": "L1 disabled for this target"}

    l1_matched = l1.kai_run_matmul_symbol_count if l1 else 0
    l1_total = l1.kai_symbol_count if l1 else 0

    if max_level < 2 or not l2_cfg.get("enabled", True):
        result["verdict"] = "L1_ONLY"
        result["headline"] = render_headline("L1_ONLY", name, None, None, l1_matched, l1_total)
        result["exit_code"] = 2
        return result

    # ---- L2 (selection) ----
    workload_params = dict(workload_cfg.get("defaults") or {})
    workload_params.update(params or {})
    if raw_run_cmd is not None:
        cmd = list(raw_run_cmd)
    else:
        try:
            cmd = build_workload_command(artifacts["binary"], workload_cfg, workload_params)
        except KeyError as e:
            result["l2"] = {"available": False, "error": str(e)}
            result["verdict"] = "L1_ONLY"
            result["headline"] = f"UNDETERMINED: {name} -- {e}"
            result["exit_code"] = 2
            return result

    l2 = run_l2_workload(cmd, l2_timeout, run_env, l2_cfg.get("patterns"))
    result["l2"] = l2
    result["level_reached"] = 2

    advertised_label = l2_cfg.get("advertised_family_label")
    advertised = l2["matches"].get(advertised_label) if advertised_label else None
    if not isinstance(advertised, str):
        # A pattern with group=0 (presence-only) matches as Python True, not a
        # family-name string -- compute_verdict()/cli_exit_code() both call
        # .lower() on this, which would raise AttributeError on a bool. Treat
        # anything that isn't already a real string as "nothing advertised"
        # rather than crash on a target JSON author's honest mistake (using a
        # presence-only pattern as advertised_family_label).
        advertised = None
    result["advertised"] = advertised

    l3_enabled = l3_cfg.get("enabled", True)

    if max_level < 3 or not l3_enabled:
        if l3_enabled:
            # Capped by --level, not by target design: say so, don't guess 0.
            result["l3"] = {"available": False, "error": f"L3 skipped (--level {max_level})"}
            result["verdict"] = "L2_ONLY"
            result["exit_code"] = 2
            result["headline"] = f"UNDETERMINED: {name} -- L3 (dispatch) not attempted (--level {max_level})"
        else:
            fallback_label = l2_cfg.get("fallback_indicator_label")
            verdict = render_l2_only_verdict(l2["matches"], fallback_label)
            result["l3"] = {"available": False, "error": "L3 not defined for this target -- see l3.note"}
            result["verdict"] = verdict
            result["exit_code"] = l2_only_exit_code(verdict)
            if verdict == "SILENT_FALLBACK":
                fallback_value = l2["matches"].get(fallback_label)
                detail = f" ({fallback_value})" if fallback_value not in (None, True) else ""
                result["headline"] = (
                    f"MISMATCH: {name} -- L2 (selection) log shows its own fallback "
                    f"indicator{detail} ('{fallback_label}') -- something was not fully "
                    f"accelerated despite being requested"
                )
            elif verdict == "L2_ADVERTISED_NO_FALLBACK_DETECTED":
                result["headline"] = (
                    f"MATCH: {name} -- L2 (selection) log ran and its fallback indicator "
                    f"('{fallback_label}') did not match; no fallback detected"
                )
            else:
                result["headline"] = (
                    f"UNDETERMINED: {name} -- L2 (selection) ran but matched no configured pattern"
                )
        return result

    # ---- L3 (dispatch) ----
    debugger = l3_debugger
    if debugger == "auto":
        if platform.system() == "Darwin":
            debugger = "lldb"
        elif shutil.which("gdb"):
            debugger = "gdb"
        else:
            debugger = "none"

    l3_scan_target = artifacts["lib"] if l3_cfg.get("scan_target", "lib") == "lib" else artifacts["binary"]
    l3_regex = l3_cfg.get("dispatch_regex") or l1_cfg.get("dispatch_regex", ".")
    l3_family_order = l3_cfg.get("family_order", l1_cfg.get("family_order"))
    target_args = cmd[1:]  # cmd[0] is always the resolved binary itself

    if debugger in ("lldb", "gdb"):
        # Fail closed uniformly for BOTH debuggers. run_l3_gdb_argv() already
        # refuses an uninstrumented probe internally; lldb's regex breakpoint
        # does not (0 resolved locations still "succeeds"), so this check is
        # what makes an lldb-driven GENERIC target fail closed too -- see
        # tools/targets/whisper-cpp.json, whose real installed binary has 0
        # symbols matching its dispatch_regex, for why this matters in practice.
        candidate_syms = enumerate_dispatch_symbols(l3_scan_target, l3_regex)
        if not candidate_syms:
            l3 = L3Result(debugger=debugger, available=False, dispatch_regex=l3_regex,
                           error=(f"no symbols matching {l3_regex!r} found in {l3_scan_target!r}; "
                                  f"refusing to run an uninstrumented probe that would report a "
                                  f"misleading zero (L1 independently found {l1_matched} matching "
                                  f"symbols there too)"))
        elif debugger == "lldb":
            l3 = run_l3_lldb_argv(artifacts["binary"], target_args, l3_regex, l3_timeout, run_env,
                                   family_order=l3_family_order)
        else:
            l3 = run_l3_gdb_argv(artifacts["binary"], target_args, l3_regex, l3_timeout, run_env,
                                  lib_path=l3_scan_target, family_order=l3_family_order)
    else:
        l3 = L3Result(debugger="none", available=False, dispatch_regex=l3_regex,
                       error="L3 skipped: no debugger found (install lldb via Xcode CLT, or gdb on Linux)")

    result["l3"] = {
        "debugger": l3.debugger, "available": l3.available, "completed": l3.completed,
        "timed_out": l3.timed_out, "error": l3.error, "dispatch_regex": l3.dispatch_regex,
        "wall_time_sec": round(l3.wall_time_sec, 2), "command": l3.command,
        "kernel_family_executed": l3.kernel_family_executed,
        "hits_by_symbol": l3.hits_by_symbol, "hits_by_family": l3.hits_by_family,
        "total_hits": l3.total_hits,
        "breakpoints_requested": l3.breakpoints_requested, "breakpoints_created": l3.breakpoints_created,
    }
    result["level_reached"] = 3

    verdict = compute_verdict(advertised, l3)
    exit_code = cli_exit_code(advertised, l3)
    headline = None
    if implicit_claim and not advertised and l3.available and not l3.timed_out and l3.completed:
        # Ad-hoc mode: the user's own --symbols regex IS the claim ("this looks
        # like a fast path"); there is no separate L2 log naming which family
        # should have fired, so pass/fail is simply "did anything matching fire".
        # Headlines are built here directly (not via render_headline(), whose
        # NO_DISPATCH_OBSERVED branch assumes a named advertised_family and
        # would otherwise print a confusing "UNDETERMINED" next to exit 1).
        if l3.total_hits > 0:
            verdict = f"{l3.kernel_family_executed.upper()}_DISPATCHED_ADHOC"
            exit_code = 0
            headline = (f"MATCH: {l3.total_hits} hit(s) observed for --symbols "
                        f"{l3.dispatch_regex!r} (dominant family: {l3.kernel_family_executed})")
        else:
            verdict = "NO_DISPATCH_OBSERVED"
            exit_code = 1
            headline = (f"MISMATCH: {l1_matched} symbol(s) matching --symbols {l3.dispatch_regex!r} "
                        f"exist in the binary but 0 ever dispatched")
    result["verdict"] = verdict
    result["exit_code"] = exit_code
    result["headline"] = headline or render_headline(verdict, name, advertised, l3, l1_matched, l1_total)
    return result


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

    model_digest = hashlib.sha256()
    with open(model, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            model_digest.update(chunk)

    ledger = {
        "schema_version": 1,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tool": "verify_dispatch.py",
        "generator_argv": sys.argv,
        "binary": binary,
        "cpu_backend_lib": lib_path,
        "model": model,
        # Identity of the exact bytes measured, not just their path. A cached model file
        # was found hash-mismatched against scripts/models.txt on 2026-08-06, and no
        # ledger written before then could prove which bytes it had measured -- see the
        # dated addendum in results/GROUND-TRUTH-DISPATCH.md.
        "model_sha256": model_digest.hexdigest(),
        "model_bytes": os.path.getsize(model),
        "configs": configs,
    }

    out_path = args.out or os.path.join(REPO_ROOT, "results", f"dispatch-ledger-{platform_meta['platform']}.json")
    # Guard against silently destroying committed evidence. This default path points at a
    # tracked file under results/, so *any* casual run -- including someone poking at the
    # tool with a throwaway binary -- used to overwrite a real, dated measurement with
    # whatever it just produced. That happened: a run against /bin/ls replaced the
    # 2026-08-03 Apple M4 Max ledger, and only the claims gate caught it. results/ is an
    # append-only evidence record in this project, so refuse rather than clobber; an
    # explicit --out is the way to say you meant it.
    if not args.out and os.path.exists(out_path):
        print(f"\nERROR: refusing to overwrite existing evidence at {out_path}", file=sys.stderr)
        print("       results/ is an append-only record in this project. Pass an explicit "
              "--out PATH (e.g. --out /tmp/ledger.json) to write elsewhere, or --out with "
              "this same path if you genuinely intend to replace it.", file=sys.stderr)
        return 2
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
