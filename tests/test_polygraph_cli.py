#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
"""Black-box tests of the tools/polygraph CLI contract: subcommand surface,
exit-code contract (0 match / 1 mismatch / 2 undetermined, never a silent 0),
and -- when a C compiler and a debugger are present -- the full catch-a-liar
end-to-end flow that `make demo` presents to judges.

Stdlib-only. Run from the repo root:

    python3 -m unittest discover -s tests -p 'test_*.py' -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POLYGRAPH = REPO_ROOT / "tools" / "polygraph"
EXAMPLE = REPO_ROOT / "examples" / "catch-a-liar" / "liar.c"


def run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(POLYGRAPH), *args],
        capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT),
    )


class TestCliSurface(unittest.TestCase):
    def test_version_flag(self):
        p = run_cli("--version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("polygraph", p.stdout)

    def test_list_shows_targets_with_tested_markers(self):
        p = run_cli("list")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("catch-a-liar", p.stdout)
        self.assertIn("llama-cpp-kleidiai", p.stdout)
        self.assertIn("[tested]", p.stdout)

    def test_explain_renders_a_target(self):
        p = run_cli("explain", "catch-a-liar")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("target: catch-a-liar", p.stdout)

    def test_check_without_arguments_is_undetermined(self):
        p = run_cli("check")
        self.assertEqual(p.returncode, 2)
        self.assertIn("--binary", p.stderr)

    def test_unknown_target_is_undetermined_not_a_crash(self):
        p = run_cli("check", "no-such-target-zzz")
        self.assertEqual(p.returncode, 2)
        self.assertIn("no target named", p.stderr)

    def test_adhoc_requires_symbols(self):
        p = run_cli("check", "--binary", "/bin/ls")
        self.assertEqual(p.returncode, 2)


def _debugger_available() -> bool:
    return bool(shutil.which("lldb") or shutil.which("gdb"))


@unittest.skipUnless(shutil.which("cc") or shutil.which("clang") or shutil.which("gcc"),
                     "no C compiler available")
class TestCatchALiarEndToEnd(unittest.TestCase):
    """The judge-facing flow, asserted instead of demonstrated: two builds of
    liar.c, identical banners, one lying -- the liar must exit 1, the honest
    build must exit 0. Skips (not fails) when no debugger is usable here."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="catch-a-liar-test-"))
        cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
        src = str(EXAMPLE)
        cls.liar = str(cls.tmp / "liar")
        cls.honest = str(cls.tmp / "honest")
        for out, extra in ((cls.liar, []), (cls.honest, ["-DACTUALLY_FAST"])):
            r = subprocess.run([cc, "-O0", "-g", "-o", out, src, *extra],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise unittest.SkipTest(f"fixture compile failed: {r.stderr}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _check(self, binary: str) -> subprocess.CompletedProcess:
        return run_cli(
            "check", "--binary", binary,
            "--symbols", "^_?fast_path_sum$",
            "--run", binary, "--level", "3",
            timeout=300,
        )

    def test_both_builds_print_the_same_banner(self):
        """The premise of the whole demo: the banner is identical and proves
        nothing."""
        out_liar = subprocess.run([self.liar], capture_output=True, text=True, timeout=30)
        out_honest = subprocess.run([self.honest], capture_output=True, text=True, timeout=30)
        self.assertIn("using fast path: yes", out_liar.stdout)
        self.assertEqual(out_liar.stdout, out_honest.stdout)

    def test_liar_is_mismatch_and_honest_is_match(self):
        if not _debugger_available():
            self.skipTest("no debugger (lldb/gdb) on PATH")
        p_liar = self._check(self.liar)
        if p_liar.returncode == 2:
            # The debugger exists but cannot attach in this environment
            # (e.g. unpermitted macOS developer mode): undetermined is the
            # contractually correct answer, but it means the end-to-end
            # assertion cannot run here.
            self.skipTest(f"debugger not usable here: {p_liar.stdout[-300:]}")
        self.assertEqual(p_liar.returncode, 1, p_liar.stdout + p_liar.stderr)
        self.assertIn("MISMATCH", p_liar.stdout)

        p_honest = self._check(self.honest)
        self.assertEqual(p_honest.returncode, 0, p_honest.stdout + p_honest.stderr)
        self.assertIn("MATCH", p_honest.stdout)


if __name__ == "__main__":
    unittest.main()
