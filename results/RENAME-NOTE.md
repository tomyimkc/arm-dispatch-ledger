# Why the files in this directory still say "arm-dispatch-ledger"

This project was renamed **Polygraph** on 2026-08-04. The GitHub repository moved from
`github.com/tomyimkc/arm-dispatch-ledger` to `github.com/tomyimkc/polygraph` (the old URL
301-redirects; same account, same commit history, nothing about the underlying evidence changed).

Everything else in this repository that is living prose — the README, the docs under `docs/`
(other than the verbatim historical record in `docs/UPSTREAM-ISSUE.md`), the site, the tools, the
scripts — has been or is being updated to say Polygraph.

**The files in this directory are the one deliberate exception.**

## What is left untouched, and why

Every `*.log`, `*.tsv`, and tool-generated `*.json` under `results/` — the dispatch ledgers
(`dispatch-ledger-darwin-arm64*.json`), the bench and crossover output (`results/bench/`,
`results/crossover/`), `results/video/race-capture.json`, `results/headline.json`, and every raw
log under `results/logs/` — was generated **before 2026-08-04**, while this repository was still
named and pathed as `arm-dispatch-ledger`. Where those files record a local filesystem path
(`/path/to/arm-dispatch-ledger/...`), a binary name, a `system_info:` banner line, or a working
directory, that path or name is exactly what existed on disk at the moment the tool actually ran
and captured it.

Those files are left **byte-identical** to how the tools that produced them wrote them. Nothing
in `results/` that is a run artifact has been touched by this rename.

## Why this matters here specifically

This project's entire thesis is that a tool's claims about what it did must be checkable against
what actually happened — that is what `tools/verify_dispatch.py` exists to prove about
`llama.cpp`'s own dispatch banner, and it is the same standard this repository holds itself to.
Editing a run artifact after the fact — even to fix something as cosmetically minor as a renamed
directory in a logged path — would mean the committed evidence no longer matches the run it
claims to be a record of. That is precisely the kind of quiet evidence-editing this project exists
to catch when other software does it. Doing it to our own evidence, even in the name of a tidy
rebrand, would be indefensible by this project's own standard.

Concretely: if you open `results/dispatch-ledger-darwin-arm64.json` and see a path or log line
referencing `arm-dispatch-ledger`, that is not stale branding to be "fixed" — it is an accurate,
unmodified record of a real `lldb`-attached run that happened under that name and that path, on
2026-08-03/04. Rewriting it to say `polygraph` would misrepresent that the run happened under a
name and path it did not actually run under.

## What to do instead

- **Citing a number from these files?** Cite it exactly as recorded; the numbers themselves
  (tok/s, hit counts, verdicts) are unaffected by the rename and remain accurate.
- **Citing a path or repo name from these files in new prose?** Use `polygraph` /
  `github.com/tomyimkc/polygraph` in whatever you are writing now — only the historical artifact
  itself keeps the old name.
- **Regenerating a report from these artifacts after 2026-08-04?** The regenerated report is new
  prose and should say Polygraph; the underlying `results/*.json`/`*.tsv`/`*.log` inputs it reads
  from stay exactly as they are.

No run artifact under `results/` will be edited to reflect this rename. This note exists so a
reader who notices the old name in a raw log or ledger file understands why, rather than assuming
it was missed.
