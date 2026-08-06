# Contributing to Polygraph

Polygraph checks whether the accelerated code path a program claims to use actually
executed. Because the whole product is *evidence*, contributions follow stricter
rules than a normal tool repo. Read `README.md` and `docs/CLAIMS.md` before
opening a PR.

## The hard rules

1. **Never edit prior evidence.** Files under `results/` are verbatim records of
   runs as they happened, warnings and all. New measurements are additive files;
   corrections get a new file plus a note, not a rewrite (see
   `results/RENAME-NOTE.md` and `results/upstream/NOTE-TARGET-VALIDATION-WARNING.md`
   for the pattern).
2. **Every numeric claim resolves or it does not merge.** `python3
   tools/check_claims.py` must exit 0. Any new ratio, throughput figure,
   percentage, or dispatch hit count in `README.md`, `docs/*.md`, or `site/`
   needs a registry entry in `docs/CLAIMS.md` (ideally with a `compute` block)
   or a committed leaf under `results/**/*.json`. Retracted figures are banned
   from reappearing unmarked.
3. **Never touch the exit-code contract.** `0` match / `1` mismatch / `2`
   undetermined, never a silent `0`. If you change behavior here, update
   `tests/test_polygraph_cli.py` and `tests/test_verify_dispatch_units.py` in
   the same commit.
4. **Stdlib-only tooling.** No runtime dependencies beyond a debugger already on
   the box (`lldb` on macOS, `gdb` on Linux). That is a judged feature, not an
   accident. Dev-time tooling (linters, etc.) must never become a runtime
   requirement.
5. **Upstream posts are human-written.** `llama.cpp`'s CONTRIBUTING rules forbid
   AI-written issues/PRs/comments/commit messages. Anything destined for
   `ggml-org/llama.cpp` stays in `docs/issues/` as a data pack + writing guide
   for the human maintainer of this repo to author by hand.

## Before you push

```bash
python3 tools/check_claims.py                      # claims gate: must exit 0
python3 -m unittest discover -s tests -p 'test_*.py' -v   # unit suite
make demo                                          # judge-facing flow, exits 0
```

CI runs the claims gate, the unit suite (`unit-tests.yml`), and the evidence
lanes on every push and PR.

## Adding a target preset

Copy `tools/targets/llama-cpp-kleidiai.json` as the reference schema. Rules the
test suite enforces (`tests/test_verify_dispatch_units.py`,
`TestTargetSchemaInvariants`):

- filename stem == `name`; `aliases` unique across all targets
- `tested` is a bool; **`tested: true` requires at least one dated
  `verified_against` receipt** (date + artifact + result) — the receipt is what
  suppresses the CLI's "marked untested" warning
- ship honest `caveats` for what was *not* verified

Run your target for real before committing it; a preset that has never been
executed against a real artifact must stay `tested: false` (see
`tools/targets/pytorch-cpu.json` for the honest pattern).

## Adding a measurement

Round-robin-interleave every config you compare (`A,B,C,…,A,B,C,…`), report
median ± population stdev with rep counts, record external load, and commit the
raw JSON under `results/`. The methodology notes in
`results/REMEASURE-2026-08-04-QUIET.md` are the house standard — they exist
because this repo once shipped a fabricated speedup measured the wrong way.

## License headers

New files carry SPDX headers (`Apache-2.0`) matching the surrounding code.
