#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
"""Tests for the dependency-free MCP stdio server (mcp/server.py): handshake,
tool inventory, honest degradation of feature detection, and the repo's own
--selftest entry point.

Stdlib-only. Run from the repo root:

    python3 -m unittest discover -s tests -p 'test_*.py' -v
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER = REPO_ROOT / "mcp" / "server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("polygraph_mcp_server_under_test", SERVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


server = _load_server()


class TestHandshakeAndInventory(unittest.TestCase):
    def test_initialize_reports_name_and_version(self):
        resp = server._handle_initialize(
            {"protocolVersion": "2024-11-05", "capabilities": {},
             "clientInfo": {"name": "unit-test", "version": "0"}})
        self.assertIn("serverInfo", resp)
        self.assertTrue(resp["serverInfo"]["name"])
        self.assertTrue(resp["serverInfo"]["version"])

    def test_tool_inventory_is_the_documented_four(self):
        resp = server._handle_tools_list({})
        names = {t["name"] for t in resp["tools"]}
        self.assertEqual(
            names, {"detect_arm_features", "verify_dispatch",
                    "recommend_config", "explain_finding"})
        for t in resp["tools"]:
            self.assertIn("description", t)
            self.assertIn("inputSchema", t)


class TestHonestDegradation(unittest.TestCase):
    def test_detect_arm_features_returns_structured_result_anywhere(self):
        result = server.detect_arm_features({})
        self.assertIsInstance(result, dict)
        self.assertTrue(result, "empty result -- the tool must always say something")
        json.dumps(result)  # must be JSON-serializable to cross stdio

    def test_explain_finding_handles_unknown_id_without_raising(self):
        result = server.explain_finding({"id": "no-such-finding-zzz"})
        self.assertIsInstance(result, dict)


class TestSelftestEntryPoint(unittest.TestCase):
    def test_repo_selftest_runs_clean(self):
        p = subprocess.run(
            [sys.executable, str(SERVER), "--selftest"],
            capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("serverInfo", p.stdout)
        self.assertIn("detect_arm_features", p.stdout)


if __name__ == "__main__":
    unittest.main()
