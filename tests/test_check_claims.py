#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
"""Unit + integration tests for tools/check_claims.py (the no-overclaim gate).

Stdlib-only, like the tool under test. Run from the repo root:

    python3 -m unittest discover -s tests -p 'test_*.py' -v

The end-to-end cases build throwaway fixture repos under the platform temp
dir -- a minimal docs/CLAIMS.md registry, a README.md carrying claims, and a
results/ corpus -- then assert the gate's exit code on clean trees and on
trees seeded with exactly one defect class each.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
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


cc = _load_module("check_claims_under_test", REPO_ROOT / "tools" / "check_claims.py")


REGISTRY_TEMPLATE = {
    "schema_version": 1,
    "retracted_figures": ["57.3"],
    "retraction_context_keywords": [
        "retract", "supersede", "superseded", "historical", "corrected", "correction"
    ],
    "retraction_exemptions": [],
    "non_claim_exemptions": [],
    "json_backed_globs": ["results/*.json", "results/**/*.json"],
    "claims": [
        {
            "id": "demo-ratio",
            "value_text": "2.50",
            "source_file": "results/notes.md",
            "compute": {"op": "ratio", "numerator": 10.0, "denominator": 4.0},
        }
    ],
}


def _write_registry(root: Path, registry: dict) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    block = json.dumps(registry, indent=2)
    (docs / "CLAIMS.md").write_text(
        "# Fixture claims registry\n\n"
        "<!-- CLAIMS-REGISTRY:BEGIN -->\n```json\n" + block + "\n```\n"
        "<!-- CLAIMS-REGISTRY:END -->\n",
        encoding="utf-8",
    )


class FixtureRepo:
    """A throwaway repo tree with a clean baseline: one registered ratio claim."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="claims-fixture-"))
        _write_registry(self.root, REGISTRY_TEMPLATE)
        (self.root / "README.md").write_text(
            "# Fixture\n\nWe measured a 2.50x win, round-robin interleaved.\n",
            encoding="utf-8",
        )
        results = self.root / "results"
        results.mkdir(exist_ok=True)
        (results / "notes.md").write_text("baseline 10.0 tok/s, tuned 4.0 tok/s\n")
        (results / "data.json").write_text(
            json.dumps({"median": 10.0, "tuned": 4.0})
        )

    def run_gate(self) -> tuple:
        """Returns (exit_code, combined_output)."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cc.main(["--root", str(self.root)])
        return rc, out.getvalue()

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class TestSmallHelpers(unittest.TestCase):
    def test_normalize_number_strips_thousands_separators(self):
        self.assertEqual(cc.normalize_number("1,230.3"), "1230.3")
        self.assertEqual(cc.normalize_number(" 51,214 "), "51214")

    def test_decimals_in(self):
        self.assertEqual(cc.decimals_in("198.9"), 1)
        self.assertEqual(cc.decimals_in("51,214"), 0)
        self.assertEqual(cc.decimals_in("0.875"), 3)

    def test_parse_float(self):
        self.assertEqual(cc.parse_float("1,230.3"), 1230.3)

    def test_resolve_json_path_dotted_and_bracketed(self):
        data = {"rows": [{"v": 1}, {"v": 2}]}
        self.assertEqual(cc.resolve_json_path(data, "rows[1].v"), 2)

    def test_iter_numeric_leaves_walks_nested_structures(self):
        leaves = set(cc.iter_numeric_leaves({"a": 1.5, "b": [2, {"c": 3.25}], "d": "x"}))
        self.assertEqual(leaves, {1.5, 2, 3.25})


class TestVerifyCompute(unittest.TestCase):
    def test_ratio_passes(self):
        self.assertIsNone(cc.verify_compute(
            "2.50", {"op": "ratio", "numerator": 10.0, "denominator": 4.0}))

    def test_ratio_mismatch_reported(self):
        err = cc.verify_compute(
            "2.60", {"op": "ratio", "numerator": 10.0, "denominator": 4.0})
        self.assertIsNotNone(err)

    def test_percent_drop(self):
        # 1 - 82.5/93.6 = 0.11859... -> 12 at 0dp
        self.assertIsNone(cc.verify_compute(
            "12", {"op": "percent_drop", "numerator": 93.6, "denominator": 82.5}))

    def test_percent_rise(self):
        self.assertIsNone(cc.verify_compute(
            "25.0", {"op": "percent_rise", "numerator": 100.0, "denominator": 125.0}))

    def test_unknown_op_rejected(self):
        self.assertIsNotNone(cc.verify_compute(
            "1", {"op": "eval_anything", "numerator": 1, "denominator": 1}))

    def test_missing_fields_rejected(self):
        self.assertIsNotNone(cc.verify_compute("1", {"op": "ratio"}))


class TestRetractionPatterns(unittest.TestCase):
    def test_boundary_guards_against_digit_collision(self):
        [(lit, pat)] = cc.build_retracted_patterns(["57.3"])
        self.assertTrue(pat.search("the fabricated +57.3% win"))
        self.assertIsNone(pat.search("57.34 is a different number"))
        self.assertIsNone(pat.search("157.3 is a different number"))
        self.assertIsNone(pat.search("57.3, with a trailing comma, is excluded"))


class TestClaimExtraction(unittest.TestCase):
    def _claims_from(self, text: str) -> list:
        tmp = Path(tempfile.mkdtemp(prefix="extract-fixture-"))
        try:
            f = tmp / "README.md"
            f.write_text(text, encoding="utf-8")
            return cc.extract_live_claims(tmp, [f])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_ratio_shapes(self):
        kinds = {(c.kind, cc.normalize_number(c.value_text))
                 for c in self._claims_from("a 3.43x win and a 0.88× tie\n")}
        self.assertEqual(kinds, {("ratio", "3.43"), ("ratio", "0.88")})

    def test_throughput_shapes(self):
        claims = self._claims_from("93.6 -> 321.0 tok/s and 1514.1 ± 198.9 tok/s\n")
        values = {cc.normalize_number(c.value_text) for c in claims if c.kind == "toks"}
        self.assertEqual(values, {"93.6", "321.0", "1514.1", "198.9"})

    def test_percentage_context_rules(self):
        claims = self._claims_from(
            "+57.3% patched, ~12% slower, collapsed by 47%, load hit 236%\n")
        values = {c.value_text for c in claims if c.kind == "pct"}
        # signed/approximate always count; unsigned only with a by/within context;
        # bare "236%" is deliberately out of scope
        self.assertEqual(values, {"+57.3", "~12", "47"})

    def test_inline_hit_counts(self):
        claims = self._claims_from("0 SME2 hits vs 31,871 NEON hits\n")
        values = {cc.normalize_number(c.value_text) for c in claims if c.kind == "hits"}
        self.assertEqual(values, {"0", "31871"})

    def test_table_unit_claims_and_noise_stripping(self):
        text = (
            "| config | hits |\n"
            "|---|---|\n"
            "| `ne11>=128` #26547 | 996 / 0 |\n"
        )
        claims = self._claims_from(text)
        table_values = {cc.normalize_number(c.value_text)
                        for c in claims if getattr(c, "table_derived", False)}
        # code span and issue ref are stripped; both bare cell numbers are claims
        self.assertEqual(table_values, {"996", "0"})

    def test_table_without_unit_header_yields_nothing(self):
        text = "| config | threads |\n|---|---|\n| a | 8 |\n"
        self.assertEqual([c for c in self._claims_from(text)], [])


class TestJsonBacking(unittest.TestCase):
    def test_rounding_bucket_is_exact_at_claim_decimals(self):
        # Mirrors the real README incident: 198.84947... rounds to 198.8, not 198.9
        leaves = {198.84947782179356}
        self.assertTrue(cc.is_json_backed("198.8", leaves))
        self.assertFalse(cc.is_json_backed("198.9", leaves))


class TestGateEndToEnd(unittest.TestCase):
    def setUp(self):
        self.fx = FixtureRepo()

    def tearDown(self):
        self.fx.cleanup()

    def test_clean_tree_passes(self):
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 0, out)

    def test_unregistered_ratio_fails(self):
        readme = self.fx.root / "README.md"
        readme.write_text(readme.read_text() + "\nAlso a 9.99x win.\n")
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 1)
        self.assertIn("unregistered-claim", out)

    def test_unmarked_retracted_figure_fails(self):
        readme = self.fx.root / "README.md"
        readme.write_text(readme.read_text() + "\nWe saw 57.3 improvement once.\n")
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 1)
        self.assertIn("retracted-figure-unmarked", out)

    def test_retraction_context_on_same_line_passes(self):
        readme = self.fx.root / "README.md"
        readme.write_text(readme.read_text() + "\nThe retracted 57.3 figure is banned.\n")
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 0, out)

    def test_percentage_never_passes_via_json_backing(self):
        # Seed a JSON leaf equal to the percentage's digits; a pct claim must
        # STILL fail without an explicit registry entry.
        (self.fx.root / "results" / "data.json").write_text(
            json.dumps({"median": 10.0, "tuned": 4.0, "reps": 12}))
        readme = self.fx.root / "README.md"
        readme.write_text(readme.read_text() + "\nPrefill collapses by 12% here.\n")
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 1)
        self.assertIn("unregistered-claim", out)

    def test_bare_table_number_needs_json_backing(self):
        readme = self.fx.root / "README.md"
        readme.write_text(
            readme.read_text() + "\n| config | hits |\n|---|---|\n| a | 12345 |\n")
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 1)
        # ...and the identical table passes once the number is a committed leaf
        (self.fx.root / "results" / "data.json").write_text(
            json.dumps({"median": 10.0, "tuned": 4.0, "hits": 12345}))
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 0, out)

    def test_compute_block_mismatch_fails(self):
        registry = json.loads(json.dumps(REGISTRY_TEMPLATE))
        registry["claims"][0]["compute"]["numerator"] = 11.0  # 11/4 != 2.50
        _write_registry(self.fx.root, registry)
        rc, out = self.fx.run_gate()
        self.assertEqual(rc, 1)
        self.assertIn("registry-compute-mismatch", out)

    def test_missing_registry_doc_is_fatal(self):
        (self.fx.root / "docs" / "CLAIMS.md").unlink()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cc.main(["--root", str(self.fx.root)])
        self.assertEqual(rc, 1)
        self.assertIn("FATAL", err.getvalue())


class TestRealRepoGate(unittest.TestCase):
    """The actual repository tree must pass its own gate."""

    def test_gate_green_on_this_checkout(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cc.main(["--root", str(REPO_ROOT)])
        self.assertEqual(rc, 0, out.getvalue())


if __name__ == "__main__":
    unittest.main()
