# Changelog

Notable changes to the Polygraph **tooling**. Measurement results are not
versioned here — they live in `results/` (additive, never edited) and
`docs/CLAIMS.md`'s registry.

## [1.0.0] — 2026-08-06

### Added

- Unit-test suite for the toolchain itself (`tests/test_check_claims.py`,
  `tests/test_verify_dispatch_units.py`, `tests/test_polygraph_cli.py`,
  `tests/test_mcp_server.py`): claims-gate logic end-to-end on fixture trees
  (retraction guard, claim extraction, compute blocks, JSON-backing rules),
  verdict derivation, the exit-code contract, target-schema invariants, CLI
  surface, MCP handshake + selftest. Runs on every push/PR via
  `.github/workflows/unit-tests.yml` (free hosted Arm64 + Apple Silicon lanes).
- `polygraph --version` and `POLYGRAPH_VERSION`; this changelog;
  `CONTRIBUTING.md`.
- Restored the three `#26334`-reproduction presets
  (`tools/targets/llama-cpp-kleidiai-cuda-ngl0-{baseline,nohost,devnone}.json`)
  with `verified_against` receipts pointing at the committed 15-run capture, so
  `results/upstream/FINDING-4-CUDA-HOST-BUFFER.md`'s Reproduce section works
  verbatim.

### Fixed

- `tools/check_claims.py`: the ratio extractor's trailing `\b` silently never
  matched the unicode `×` form, so every `N.NN×` figure in scanned prose was
  invisible to the gate — the exact drift class it exists to catch. Fixed
  (`(?!\w)`); the now-extracted figures were audited and registered. Found by
  the new unit suite on its first run.
- `tools/verify_dispatch.py`: `find_artifact_by_globs()` now actually prefers
  the unversioned library name (`libfoo.dylib` over `libfoo.1.2.3.dylib`), as
  its docstring always claimed.
- Submission-copy (adversarial review): accurate L1/L2/L3 split for Findings 3
  and 4; headline finding now links the correct upstream issue; catch-a-liar
  line count corrected; Neoverse-N2 lane credited with what it actually runs;
  shipped generalization presets (whisper.cpp, ONNX Runtime) surfaced.
