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
| `finding3-*` | **filed 2026-08-05** | [ggml-org/llama.cpp#26630](https://github.com/ggml-org/llama.cpp/issues/26630) |
| (Finding 1/2, see `docs/UPSTREAM-ISSUE.md`) | filed 2026-08-04 | [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547) |

**`finding3-title.txt` and `finding3-body.md` are now frozen.** They are the record of what was
actually sent to #26630. Corrections go in a comment on that issue, never in these files — the
same rule this repo applies to `results/`.
