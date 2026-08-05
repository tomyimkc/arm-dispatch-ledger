# `examples/catch-a-liar/`

The whole `polygraph` idea, in 30 lines of C, with no Arm hardware and no model download.

- **[`liar.c`](liar.c)** — two implementations of the same function, `fast_path_sum()` (an O(1)
  closed form) and `fallback_sum()` (an O(n) loop), plus a `main()` that always prints
  `using fast path: yes`. Compiled one way (the default), that print statement is a lie:
  `fast_path_sum()` is never called. Compiled with `-DACTUALLY_FAST`, it's true. This is exactly
  the real bug this project found in `llama.cpp`'s KleidiAI backend (see the repo root
  [`README.md`](../../README.md)) — a startup banner or log line that doesn't track what the
  hardware actually dispatched — shrunk to something anyone can build and check in under two
  minutes.
- **[`target.json`](target.json)** / **[`target-honest.json`](target-honest.json)** — the
  `polygraph` target presets for the lying and honest builds respectively, including the expected
  exit code and verdict for each.
- **[`demo.sh`](demo.sh)** — builds both binaries, runs `tools/polygraph check` against each, and
  prints a PASS/FAIL summary against the expectations in the two target JSON files above. Run via
  `make demo` from the repo root; see [`docs/QUICKSTART.md`](../../docs/QUICKSTART.md) for the
  full walkthrough and current status.

Both directions are shipped on purpose: a detector that always says "mismatch" is worthless, so
`build/honest` — the negative control — has to come back clean, not just `build/liar`.
