<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: 2026 Polygraph contributors -->

# L3 lldb ground-truth test

Proves `tools/dispatch_probe.lldb` (driven through `tools/verify_dispatch.py`'s
`run_l3_lldb()` -- the real, unmodified code path, not a reimplementation) recovers an
**exactly known** per-symbol call count, on macOS/Darwin. This is the lldb counterpart of
[`tests/l3_gdb_groundtruth/`](../l3_gdb_groundtruth/), written because 100% of this repo's
SME2 dispatch evidence comes from the lldb probe and it had never been checked against a
call count anyone actually knew was true. See `run_test.sh`'s header comment for the full
rationale, including a real bug the fixture itself found while being written.

## Run it locally (one-liner)

```bash
bash tests/l3_lldb_groundtruth/run_test.sh
```

Optionally pass a call count (default `7`; the sme2-family symbol is always called
`+4` more times than that, to also exercise `kernel_family_executed`):

```bash
bash tests/l3_lldb_groundtruth/run_test.sh 20
```

Requires macOS with Xcode Command Line Tools (`clang`, `lldb`) on `PATH`; skips
gracefully (exit `0`, with a clear `[skip]` message) on any other OS or if `lldb`/`cc`
is missing. Exits non-zero the moment a recovered count, family classification, or
`kernel_family_executed` verdict does not match ground truth exactly.

## What it builds

- `libkai_fake.c` -- a synthetic `libkai_fake.dylib` with two `kai_run_matmul_*`-named,
  `noinline` symbols (one `..._neon_dotprod`, one `..._sme2_mopa`) called a known,
  *different* number of times each.
- `main.c` -- a fake `llama-cli`-shaped binary (`fake_llama`) that accepts exactly the
  argv shape `run_l3_lldb()` builds (`-m ... -p ... -n ... -no-cnv -st --simple-io
  -t ...`), `dlopen`s the fake dylib **after** process start (mirroring how ggml loads
  its CPU backend), and calls the workload the requested number of times.

CI wiring: `.github/workflows/verify-macos-arm64.yml` (the self-hosted, `workflow_dispatch`
-only macOS lane -- see that file's own header for why it isn't `push`-triggered).
