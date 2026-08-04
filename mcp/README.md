# polygraph MCP server

A dependency-free MCP (Model Context Protocol) server that turns this
project's two verified findings about llama.cpp's KleidiAI CPU backend into
callable tools for an agentic client. Arm Create Track 2 explicitly calls
out "agentic multi-model workloads with MCP servers" -- this is that piece:
an agent can ask *this machine, right now* whether an SME2/SVE kernel is
actually being dispatched, instead of trusting a compile-time banner.

`mcp/server.py` implements the MCP stdio JSON-RPC 2.0 transport directly
against the Python standard library. **No `pip install` is required** --
see `requirements.txt` for why, and for the optional official-SDK swap-in
path if you'd rather use that instead.

## Tools

| Tool | What it does |
|---|---|
| `detect_arm_features()` | Live ISA feature detection for the host running the server: SME/SME2/SVE/SVE2/I8MM/BF16/DotProd, streaming vector length, core counts, brand string. `sysctl`-based on macOS (verified on an Apple M4 Max); `/proc/cpuinfo` + `SMIDR_EL1` sysfs based on Linux (implemented per spec, **not yet exercised on real Linux/Spark hardware** -- it says so in its own output). |
| `verify_dispatch(binary, model, threads, ...)` | Runs the actual L1 (compile-time banner) / L2 (selection-time log) / L3 (dispatch-time `lldb` breakpoint hit-count) check against a llama.cpp-family binary + GGUF model, and returns a verdict: was the SME2 kernel *actually executed*, or did it silently fall back? |
| `recommend_config(model, quant, workload)` | Reads `results/*.json` / `*.jsonl` (this project's measured ledger) if entries exist; otherwise returns an explicit "not yet measured" architectural recommendation grounded in Finding 1's root cause. **Never invents a throughput number.** |
| `explain_finding(id)` | Returns the root-cause writeup + exact `kleidiai.cpp` source line excerpts for Finding 1 (`id="1"`) or Finding 2 (`id="2"`), plus the hit-count evidence measured while this server was built. |

Every one of these degrades honestly instead of guessing: missing `lldb` ->
`L3_dispatch_time.available: false` with an explanation, not a fabricated
hit count; empty `results/` -> `recommend_config` says so explicitly instead
of inventing a number; an unrecognized `explain_finding` id returns the list
of valid ids instead of silently picking one.

## Add it to your client

**Claude Code** (CLI, run from the repo root):

```bash
claude mcp add polygraph -- python3 "$(pwd)/mcp/server.py"
```

**Claude Desktop / Cursor** (edit `claude_desktop_config.json` or
`~/.cursor/mcp.json` / project `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "polygraph": {
      "command": "python3",
      "args": ["/absolute/path/to/polygraph/mcp/server.py"]
    }
  }
}
```

Replace the path with the absolute path to this repo's `mcp/server.py` on
your machine. No `env` block, no extra dependencies -- `python3` on `PATH`
is the only requirement.

## Self-test (no MCP client needed)

```bash
python3 mcp/server.py --selftest
```

This runs `initialize` -> `notifications/initialized` -> `tools/list` ->
three `tools/call`s (`detect_arm_features`, `explain_finding`,
`recommend_config`) through the *exact* stdio JSON-RPC codepath the real
transport uses, entirely in-process, and prints the newline-delimited JSON
responses. Useful for a judge who wants to see it work without configuring
a client at all.

## Manual smoke test (real subprocess, real stdin pipe)

This is exactly how it was verified while building it -- pipe requests into
a real `python3 mcp/server.py` subprocess over stdin and read the responses
from stdout:

```bash
cat > /tmp/req.jsonl <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual-test","version":"0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"verify_dispatch","arguments":{"binary":"/path/to/llama-cli","model":"/path/to/model.gguf","threads":4,"n_predict":4}}}
EOF
python3 mcp/server.py < /tmp/req.jsonl
```

Real output captured against `llama-cli` @ llama.cpp `dbadb68`
(`-DGGML_CPU_KLEIDIAI=ON`) with `Qwen2.5-0.5B-Instruct-Q4_0.gguf`, on this
project's Apple M4 Max, `threads=4` (i.e. *above* this chip's SME thread
cap of 2 -- see `explain_finding("1")`):

```json
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"{\n  \"tiers\": {\n    \"L1_compile_time_banner\": {\"raw\": \"system_info: n_threads = 4 ... | SME = 1 | SME2 = 1 | KLEIDIAI = 1 | ...\"},\n    \"L2_selection_time_log\": {\"lines\": [\"kleidiai: primary q4 kernel feature SME2\", \"kleidiai: SME2 enabled (runtime-detected SME cores=2)\"]},\n    \"L3_dispatch_time\": {\"available\": true, \"locations\": 18, \"resolved\": 18, \"total_hit_count\": 0, \"sme_dispatched\": false}\n  },\n  \"verdict\": \"SME2 NOT DISPATCHED (0 hits) -- silent fallback to NEON/DotProd/I8MM kernels, DESPITE L1/L2 above both still reporting SME2 as compiled-in and selected. This is Finding 1.\"\n}"}],"isError":false}}
```

L1 and L2 both still say SME2 is on. L3 -- the only tier that ran a real
breakpoint against the kernel body -- says it never executed once. That
gap, reproduced live through this tool, *is* Finding 1.

## Graceful degradation, by design

- **No `results/` ledger yet?** `recommend_config` returns `"ledger_status":
  "no matching entries..."` plus an explicitly-labeled architectural
  recommendation (bracketed `[NOT YET MEASURED]` placeholders where a real
  throughput number would go) -- never a made-up tokens/sec figure.
- **No `lldb` on `PATH` (e.g. a stripped-down Linux CI image)?**
  `verify_dispatch` still returns L1/L2 and marks `L3_dispatch_time.available:
  false` with a note, and the overall verdict becomes `INCONCLUSIVE` rather
  than a false "not dispatched".
- **Running on Linux?** `detect_arm_features` uses a best-effort
  `/proc/cpuinfo` + `SMIDR_EL1`-sysfs path and labels its own output
  `"verified_on_this_session": false` with an explicit caveat, because this
  server was built and exercised only on macOS/Apple Silicon in this
  development session -- it was never run against the DGX Spark or a
  Neoverse-N2 CI runner. Run `detect_arm_features` there yourself to
  confirm before trusting those fields.

## Files in this directory

- `server.py` -- the server (stdlib-only, Apache-2.0 SPDX header).
- `requirements.txt` -- intentionally empty; documents why.
- `README.md` -- this file.
