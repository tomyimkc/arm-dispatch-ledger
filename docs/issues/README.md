# Ready-to-file upstream reports

Each report here is split into the exact two inputs `gh issue create` wants, so filing is one
command with no copy-paste editing:

```bash
gh issue create --repo ggml-org/llama.cpp \
  --title "$(cat docs/issues/finding3-title.txt)" \
  --body-file docs/issues/finding3-body.md
```

`docs/UPSTREAM-ISSUE-FINDING3.md` remains the human-readable version of the same report — it keeps
the framing notes and filing checklist that belong to us, not to the issue tracker. The two files
here are its title line and its body, verbatim, with our internal header stripped.

**Once a report is filed, do not edit these files.** They become the record of what was actually
sent. Corrections belong in a comment on the filed issue, the same rule the rest of this repo
applies to `results/`.

| file | status | filed as |
|---|---|---|
| `finding3-*` | **not yet filed** | — |
| (Finding 1/2, see `docs/UPSTREAM-ISSUE.md`) | filed | ggml-org/llama.cpp#26547 |
