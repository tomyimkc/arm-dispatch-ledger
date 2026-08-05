# Quickstart

Two minutes, no Arm hardware, no model download: watch `polygraph check` catch a real silent
fallback, then point it at your own binary.

## Status (2026-08-05)

`examples/catch-a-liar/` and this guide are built against the CLI contract (`tools/polygraph
list|check|explain`, documented in full below). `tools/polygraph` itself is a separate,
concurrent work package for this same submission and may not exist yet in the checkout you're
reading this from. `make demo` tells you plainly which state you're in:

- **`tools/polygraph` present** — the full demo runs end to end; the exit codes and verdicts
  below are real, produced by that run.
- **`tools/polygraph` missing** — the script still compiles both real binaries (so you can see
  they behave identically at the banner line), then stops and prints the exact `polygraph check`
  commands to re-run once the tool lands, exiting `2`.

## 2-minute demo

```sh
git clone https://github.com/tomyimkc/polygraph.git
cd polygraph
make demo
```

What it does:

1. Compiles [`examples/catch-a-liar/liar.c`](../examples/catch-a-liar/liar.c) two ways —
   `build/liar` and `build/honest`. Read that file's header first; it's ~30 lines. Both binaries
   print the identical banner line `using fast path: yes` — that line is a print-time claim, not
   proof, the same way `llama.cpp`'s `KLEIDIAI = 1` banner is.
2. Runs `tools/polygraph check` against each, with a debugger attached to `fast_path_sum()`.
3. Prints a one-line verdict per binary and exits accordingly:

| binary | banner claims | `fast_path_sum()` actually called | expected verdict | expected exit code |
|---|---|---|---|---|
| `build/liar` | yes | no | `MISMATCH: ...` (see `examples/catch-a-liar/target.json`) | `1` |
| `build/honest` | yes | yes | no mismatch (see `examples/catch-a-liar/target-honest.json`) | `0` |

Both directions are checked on purpose — a detector that always says "mismatch" is worthless.

Run either check by hand:

```sh
tools/polygraph check \
  --binary  examples/catch-a-liar/build/liar \
  --symbols '^fast_path_sum$' \
  --run     examples/catch-a-liar/build/liar \
  --level   3
echo "exit code: $?"
```

```sh
tools/polygraph check \
  --binary  examples/catch-a-liar/build/honest \
  --symbols '^fast_path_sum$' \
  --run     examples/catch-a-liar/build/honest \
  --level   3
echo "exit code: $?"
```

## Run it on your own binary

No preset needed — the ad-hoc form takes a binary, a symbol pattern for the accelerated code
path, and the command that actually runs it:

```sh
tools/polygraph check \
  --binary  /path/to/your/binary \
  --symbols '^your_fast_kernel_prefix_' \
  --run     "/path/to/your/binary --your --normal --flags"
```

Useful flags:

```sh
tools/polygraph check --binary B --symbols R --run "CMD" --json     # machine-readable result on stdout, nothing else on stdout
tools/polygraph check --binary B --symbols R --run "CMD" --level 2  # static scan + selection log only; skip the debugger (L3)
tools/polygraph check --binary B --symbols R --run "CMD" --quiet    # suppress non-essential output
```

Exit codes (contractual — a CI job can gate on these directly):

| code | meaning |
|---|---|
| `0` | advertised capability matches what executed |
| `1` | **mismatch** — something claimed acceleration that did not run |
| `2` | undetermined — missing debugger, no permission, binary not found. Never silently `0`. |

List and inspect the built-in targets (including this repo's own KleidiAI checks and
`catch-a-liar`):

```sh
tools/polygraph list
tools/polygraph explain catch-a-liar
```

## Run it in CI

Full walkthrough: [`docs/CI.md`](CI.md).

Minimal pattern: gdb ships on GitHub's hosted Linux runners and lldb ships on hosted macOS
runners, so a `polygraph check` gate needs no self-hosted Arm hardware.

```yaml
- name: polygraph check
  run: |
    tools/polygraph check \
      --binary  build/your-binary \
      --symbols '^your_fast_kernel_prefix_' \
      --run     "build/your-binary --your --flags" \
      --json > polygraph-result.json
```

`--json` on stdout plus the exit codes above are the two things a CI step needs: parse the JSON
for detail, gate the job on the exit code. See [`docs/CI.md`](CI.md) for the full walkthrough,
including this project's own judge-reproducible lane,
[`.github/workflows/verify-free-arm64.yml`](../.github/workflows/verify-free-arm64.yml), which
runs the same class of check against the real `llama.cpp`/KleidiAI case on GitHub's free hosted
`ubuntu-24.04-arm` runner.
