<!-- SPDX-License-Identifier: Apache-2.0 -->

# CI: adopt the dispatch check in one line

`tools/polygraph` (see the repo README for what it measures and why) ships as a reusable
composite GitHub Action so any project can wire "does my advertised accelerated code path
actually execute?" into its own CI, without vendoring this repo's source.

## Adopt it

```yaml
- uses: tomyimkc/polygraph/.github/actions/polygraph-check@main
  with:
    target: kleidiai
    binary: build/bin/llama-cli      # optional, preset auto-detects
    fail-on: mismatch                 # mismatch | never
```

That is the whole integration. The action installs `gdb` if this is a Linux runner and it is
missing (macOS runners use `lldb`, which ships with Xcode Command Line Tools), runs
`tools/polygraph check --json` for you, writes a readable table to the job's Step Summary, and
exits with the tool's own contractual exit code.

Ad-hoc mode (no built-in preset — check any binary against any symbol regex and workload) works
the same way:

```yaml
- uses: tomyimkc/polygraph/.github/actions/polygraph-check@main
  with:
    binary: build/my-app
    symbols: '^my_accelerated_kernel'
    run: build/my-app --benchmark
```

`.github/workflows/demo-action.yml` in this repo runs both a passing and a failing case of
exactly this ad-hoc mode, live, against two tiny synthetic binaries — open the Actions tab and
look at the `demo-action` workflow to see it catch a real mismatch.

## Inputs

| input | required | default | meaning |
|---|---|---|---|
| `target` | one of `target` or (`binary`+`symbols`+`run`) | — | a built-in preset name (see `tools/polygraph list`), passed as the CLI's positional `<target>` |
| `binary` | see above | — | path to the binary under test. Optional with `target` (the preset auto-detects it); required in ad-hoc mode |
| `symbols` | ad-hoc mode only | — | anchored regex of the accelerated-kernel symbol(s) to look for |
| `run` | ad-hoc mode only | — | shell command that exercises the binary |
| `level` | no | `3` | max verification level to attempt: `1` (static symbols), `2` (runtime selection log), `3` (real debugger dispatch count) |
| `fail-on` | no | `mismatch` | `mismatch`: the action's step fails whenever `tools/polygraph check` exits non-zero. `never`: always exit 0; read the outputs yourself |

## Outputs

| output | meaning |
|---|---|
| `verdict` | the tool's verdict string from its `--json` result (e.g. `MATCH`, `MISMATCH`, `NO_DISPATCH_OBSERVED`, `SILENT_FALLBACK`), or `TOOL_NOT_AVAILABLE` if `tools/polygraph` could not be found at all |
| `level-reached` | highest verification level actually reached — may be lower than requested if a level gracefully degraded |
| `json-path` | runner-local path to the raw `--json` result file; upload it as a workflow artifact yourself if you want to keep it |

## What the exit codes mean

`tools/polygraph check` (and this action, by default) uses three exit codes, always — it never
silently reports success when it could not actually tell:

| exit code | meaning |
|---|---|
| `0` | **match** — the advertised capability matches what executed |
| `1` | **mismatch** — something claimed acceleration that did not run |
| `2` | **undetermined** — missing debugger, no permission, binary not found. Never silently `0` |

With the default `fail-on: mismatch`, this action's step fails on *either* `1` or `2` — an
undetermined result is not treated as a pass. Set `fail-on: never` if you want to inspect the
`verdict` output yourself instead of failing the build.

## How the action finds `tools/polygraph`

No separate checkout step, no extra network call, and no version skew: `$GITHUB_ACTION_PATH`
always points at the action's own directory at exactly the ref/commit it was resolved from —
GitHub Actions downloads the **whole** source repository for a subdirectory-referenced action
before running it. Three levels up from `.github/actions/polygraph-check` is always this
repository's root, so the `tools/polygraph` found there is always the same version this
`action.yml` shipped with. If `tools/polygraph` is missing entirely, that's an
environment/integration problem, not a verdict — the action fails loudly regardless of
`fail-on`.

## Existing workflows in this repo

None of this repo's self-hosted lanes (`verify-macos-arm64.yml`, `verify-spark-aarch64.yml`)
trigger on `pull_request` or `pull_request_target` — both are `workflow_dispatch`-only,
confirmed by reading every file under `.github/workflows/` for this doc. The free,
judge-reproducible lane (`verify-free-arm64.yml`) and the claims/pages lanes run on
GitHub-hosted runners only. `demo-action.yml` follows the same rule: GitHub-hosted
(`ubuntu-24.04-arm`, override with the `POLYGRAPH_DEMO_RUNNER` repository variable) only, and not
triggered on `pull_request` either — it deliberately runs one job that is expected to fail (see
that file's header comment), and keeping that out of every contributor's PR checks list is a
usability choice, not a security one.
