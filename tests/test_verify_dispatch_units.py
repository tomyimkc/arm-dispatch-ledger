#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
"""Unit tests for the pure-logic surface of tools/verify_dispatch.py and the
tools/targets/*.json schema invariants tools/polygraph relies on.

Stdlib-only. Run from the repo root:

    python3 -m unittest discover -s tests -p 'test_*.py' -v

Deliberately covers the contractual surface only (verdict derivation, exit-code
mapping, target loading/schema, workload templating, artifact format sniffing)
-- the debugger-driven L2/L3 lanes are integration-tested by
tests/l3_lldb_groundtruth/, tests/l3_gdb_groundtruth/ and
tests/target_definition/ instead.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass processing looks the module up here
    spec.loader.exec_module(mod)
    return mod


vd = _load_module("verify_dispatch_under_test", REPO_ROOT / "tools" / "verify_dispatch.py")


def _l3(**overrides) -> "vd.L3Result":
    base = dict(
        debugger="lldb",
        available=True,
        completed=True,
        timed_out=False,
        total_hits=0,
        hits_by_family={},
        kernel_family_executed="none",
    )
    base.update(overrides)
    return vd.L3Result(**base)


class TestClassifySymbolFamily(unittest.TestCase):
    def test_kleidiai_naming_convention(self):
        self.assertEqual(
            vd.classify_symbol_family(
                "kai_run_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme2_mopa"),
            "sme2")
        self.assertEqual(
            vd.classify_symbol_family(
                "kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod"),
            "dotprod")
        self.assertEqual(
            vd.classify_symbol_family(
                "kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm"),
            "i8mm")

    def test_sme2_wins_over_sme_substring(self):
        self.assertEqual(vd.classify_symbol_family("kai_run_matmul_x_sme2_mopa"), "sme2")
        self.assertEqual(vd.classify_symbol_family("kai_run_matmul_x_sve2_mopa"), "sve2")

    def test_custom_family_order(self):
        self.assertEqual(
            vd.classify_symbol_family("repack_gemm_tiled", family_order=["gemm", "gemv"]),
            "gemm")

    def test_unknown_scheme_is_other(self):
        self.assertEqual(vd.classify_symbol_family("totally_unrelated_symbol"), "other")


class TestComputeVerdict(unittest.TestCase):
    def test_l3_failure_modes(self):
        self.assertEqual(vd.compute_verdict("sme2", _l3(available=False)), "L3_UNAVAILABLE")
        self.assertEqual(vd.compute_verdict("sme2", _l3(timed_out=True)), "L3_TIMEOUT")
        self.assertEqual(vd.compute_verdict("sme2", _l3(completed=False)), "L3_ERROR")

    def test_zero_hits_is_no_dispatch(self):
        self.assertEqual(vd.compute_verdict("sme2", _l3()), "NO_DISPATCH_OBSERVED")

    def test_advertised_family_dispatched(self):
        l3 = _l3(total_hits=996, hits_by_family={"sme2": 996},
                 kernel_family_executed="sme2")
        self.assertEqual(vd.compute_verdict("SME2", l3), "SME2_DISPATCHED")

    def test_hybrid_dispatch(self):
        l3 = _l3(total_hits=15240, hits_by_family={"sme2": 1538, "dotprod": 13702})
        self.assertEqual(vd.compute_verdict("sme2", l3), "SME2_HYBRID_DISPATCH")

    def test_silent_fallback(self):
        l3 = _l3(total_hits=31871, hits_by_family={"dotprod": 31871},
                 kernel_family_executed="dotprod")
        self.assertEqual(vd.compute_verdict("sme2", l3), "SILENT_FALLBACK")

    def test_no_advertised_feature(self):
        l3 = _l3(total_hits=100, hits_by_family={"dotprod": 100},
                 kernel_family_executed="dotprod")
        self.assertEqual(
            vd.compute_verdict(None, l3), "DOTPROD_EXECUTED_NO_ADVERTISED_FEATURE")

    def test_assert_fail_set_is_the_mismatch_pair(self):
        self.assertEqual(
            vd.DEFAULT_ASSERT_FAIL_VERDICTS, {"SILENT_FALLBACK", "NO_DISPATCH_OBSERVED"})


class TestExitCodeContract(unittest.TestCase):
    """0 = advertised matches executed; 1 = mismatch; 2 = undetermined. Never a
    silent 0 on missing/failed evidence."""

    def test_missing_or_failed_evidence_is_never_zero(self):
        self.assertEqual(vd.cli_exit_code("sme2", None), 2)
        self.assertEqual(vd.cli_exit_code("sme2", _l3(available=False)), 2)
        self.assertEqual(vd.cli_exit_code("sme2", _l3(timed_out=True)), 2)
        self.assertEqual(vd.cli_exit_code("sme2", _l3(completed=False)), 2)

    def test_nothing_advertised_is_undetermined(self):
        l3 = _l3(total_hits=100, hits_by_family={"dotprod": 100})
        self.assertEqual(vd.cli_exit_code(None, l3), 2)
        self.assertEqual(vd.cli_exit_code("none", l3), 2)
        self.assertEqual(vd.cli_exit_code("", l3), 2)

    def test_zero_hits_with_a_claim_is_mismatch(self):
        self.assertEqual(vd.cli_exit_code("sme2", _l3()), 1)

    def test_advertised_hits_present_is_match(self):
        l3 = _l3(total_hits=3888, hits_by_family={"i8mm": 3888})
        self.assertEqual(vd.cli_exit_code("I8MM", l3), 0)

    def test_only_other_family_hits_is_mismatch(self):
        l3 = _l3(total_hits=31871, hits_by_family={"dotprod": 31871})
        self.assertEqual(vd.cli_exit_code("sme2", l3), 1)

    def test_l2_only_verdict_and_exit_codes(self):
        self.assertEqual(
            vd.render_l2_only_verdict({"cpu_fallback_seen": "1"}, "cpu_fallback_seen"),
            "SILENT_FALLBACK")
        self.assertEqual(
            vd.render_l2_only_verdict({"coreml_capability": "2"}, "cpu_fallback_seen"),
            "L2_ADVERTISED_NO_FALLBACK_DETECTED")
        self.assertEqual(vd.render_l2_only_verdict({}, "cpu_fallback_seen"), "L2_NO_SIGNAL")
        self.assertEqual(vd.l2_only_exit_code("SILENT_FALLBACK"), 1)
        self.assertEqual(vd.l2_only_exit_code("L2_ADVERTISED_NO_FALLBACK_DETECTED"), 0)
        self.assertEqual(vd.l2_only_exit_code("L2_NO_SIGNAL"), 2)


class TestWorkloadTemplating(unittest.TestCase):
    def test_substitution(self):
        cmd = vd.build_workload_command(
            "/bin/llama-cli",
            {"arg_template": ["-m", "{model}", "-t", "{threads}"]},
            {"model": "m.gguf", "threads": 4},
        )
        self.assertEqual(cmd, ["/bin/llama-cli", "-m", "m.gguf", "-t", "4"])

    def test_missing_placeholder_fails_loudly(self):
        with self.assertRaises(KeyError):
            vd.build_workload_command(
                "/bin/llama-cli", {"arg_template": ["-m", "{model}"]}, {})


class TestTargetLoading(unittest.TestCase):
    def test_alias_resolves(self):
        t = vd.load_target("kleidiai")
        self.assertEqual(t["name"], "llama-cpp-kleidiai")

    def test_unknown_target_raises_actionable_error(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            vd.load_target("no-such-target-zzz")
        self.assertIn("polygraph list", str(ctx.exception))

    def test_list_includes_committed_presets(self):
        names = vd.list_target_names()
        for expected in ("catch-a-liar", "llama-cpp-kleidiai",
                         "llama-cpp-kleidiai-cuda-ngl0-baseline",
                         "llama-cpp-kleidiai-cuda-ngl0-nohost",
                         "llama-cpp-kleidiai-cuda-ngl0-devnone",
                         "whisper-cpp", "onnxruntime"):
            self.assertIn(expected, names)


class TestTargetSchemaInvariants(unittest.TestCase):
    """Every committed target JSON must satisfy the schema the CLI depends on.
    A target that claims tested:true must carry at least one dated
    verified_against receipt -- the receipt is what suppresses the CLI's
    'marked untested' warning."""

    def setUp(self):
        self.targets_dir = REPO_ROOT / "tools" / "targets"
        self.paths = sorted(self.targets_dir.glob("*.json"))
        self.assertTrue(self.paths, "no targets committed")

    def test_all_targets_load_and_are_schema_sound(self):
        seen_names = set()
        seen_aliases = set()
        for path in self.paths:
            t = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(t.get("name"), path.stem, f"{path}: name != filename")
            self.assertNotIn(t["name"], seen_names, f"duplicate name {t['name']}")
            seen_names.add(t["name"])
            for alias in t.get("aliases", []):
                self.assertNotIn(alias, seen_aliases, f"duplicate alias {alias}")
                seen_aliases.add(alias)
            self.assertIsInstance(t.get("tested"), bool, f"{path}: tested must be bool")
            self.assertIn("l1", t, f"{path}: missing l1 section")
            self.assertIn("workload", t, f"{path}: missing workload section")
            if t["tested"]:
                receipts = t.get("verified_against")
                self.assertIsInstance(receipts, list, f"{path}: tested but no receipts")
                self.assertTrue(receipts, f"{path}: tested with empty receipts")
                for r in receipts:
                    self.assertIn("date", r, f"{path}: receipt missing date")
                    self.assertIn("result", r, f"{path}: receipt missing result")


class TestArtifactFormatSniffing(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="artifact-format-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_elf_magic(self):
        p = self.tmp / "libfake.so"
        p.write_bytes(b"\x7fELF" + b"\x00" * 12)
        self.assertEqual(vd.detect_artifact_format(str(p)), "elf")

    def test_macho_magic(self):
        for magic in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"):
            p = self.tmp / f"fake_{magic.hex()}.dylib"
            p.write_bytes(magic + b"\x00" * 12)
            self.assertEqual(vd.detect_artifact_format(str(p)), "macho")


class TestArtifactGlobResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="artifact-globs-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_matching_pattern_wins_and_unversioned_preferred(self):
        (self.tmp / "libfoo.1.2.3.dylib").write_bytes(b"x")
        (self.tmp / "libfoo.dylib").write_bytes(b"x")
        got = vd.find_artifact_by_globs(str(self.tmp), ["nomatch*", "libfoo*.dylib"])
        self.assertEqual(os.path.basename(got), "libfoo.dylib")

    def test_recursive_glob(self):
        nested = self.tmp / "pkg" / "capi"
        nested.mkdir(parents=True)
        (nested / "libonnxruntime.dylib").write_bytes(b"x")
        got = vd.find_artifact_by_globs(str(self.tmp), ["**/libonnxruntime*.dylib"])
        self.assertIsNotNone(got)
        self.assertEqual(os.path.basename(got), "libonnxruntime.dylib")


class TestHeadlineRendering(unittest.TestCase):
    def test_mismatch_and_match_language(self):
        l3 = _l3(total_hits=31871, hits_by_family={"dotprod": 31871})
        line = vd.render_headline("SILENT_FALLBACK", "t", "sme2", l3, 10, 149)
        self.assertTrue(line.startswith("MISMATCH:"), line)
        l3b = _l3(total_hits=996, hits_by_family={"sme2": 996})
        line = vd.render_headline("SME2_DISPATCHED", "t", "sme2", l3b, 10, 149)
        self.assertTrue(line.startswith("MATCH:"), line)
        line = vd.render_headline("L3_UNAVAILABLE", "t", "sme2",
                                  _l3(available=False, error="no debugger"), 0, 0)
        self.assertTrue(line.startswith("UNDETERMINED:"), line)


if __name__ == "__main__":
    unittest.main()
