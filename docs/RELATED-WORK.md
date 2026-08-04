# Related work

**Scholarly hygiene, not apology.** An unacknowledged overlap a judge discovers on their own is far
more damaging to a submission's credibility than one the submission cites first. This page says
plainly what prior and concurrent work exists in this challenge track, what it found before we did,
and what this project adds on top of it. Nothing below changes any throughput number or dispatch
finding elsewhere in this repo — it only adds attribution this repo was missing.

---

## Finding 2's mechanism was published first — `luongs3/arm-dispatch-audit`

[`luongs3/arm-dispatch-audit`](https://github.com/luongs3/arm-dispatch-audit) (Apache-2.0) was
created **2026-08-01T18:01:43Z** — confirmed via `gh api repos/luongs3/arm-dispatch-audit` — which is
**two days before** this repository (`tomyimkc/arm-dispatch-ledger`, created **2026-08-03T23:19:09Z**,
same API). Its README headline:

> "Your Neoverse says SVE2. Your matmul runs neon_i8mm. An ISA dispatch audit for Arm —
> reproducible on a free CI runner."

Their README (fetched via the GitHub API on 2026-08-04) cites the identical
`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:209` line, the identical `QK8_0 == 32` (256-bit)
exact-width SVE gate (`#define QK8_0 32`, `ggml_cpu_get_sve_cnt() == QK8_0`), the same
"advertised vs. dispatched" framing (their probe, `src/isa-probe.c`, exits `2` for exactly that
case), and the same free `ubuntu-24.04-arm` GitHub-runner methodology this project's Finding 2 also
uses.

**They published this mechanism first.** This project's Finding 2 (`docs/FINDINGS.md`, README's
"Finding 2 — the SVE kernel family is architecturally unreachable below 256-bit vectors") was
derived independently, from source, before this project was aware their repository existed — but
independent derivation is not priority, and **we are not claiming priority on Finding 2.** Credit for
publishing the SVE 256-bit exact-width dispatch gate first goes to `luongs3/arm-dispatch-audit`.

**Verified, not assumed, that Finding 1 is not also scooped:** grepping their fetched README for
`sme_thread_cap`, `SMCU`, `ne11`, and `thread cap` returns **zero matches** on any of the four terms.
None of Finding 1's vocabulary — the SME2 decode thread-gating on Apple Silicon — appears in their
repository. Their work is Linux/Neoverse-N2-focused and does not mention Apple Silicon, SME, or SME2
at all. As far as this project could verify on 2026-08-04, **Finding 1 is still original to this
repository.**

### What this project adds beyond `arm-dispatch-audit`

| This project | `arm-dispatch-audit` |
|---|---|
| **Finding 1** — SME2 decode thread-gating on Apple Silicon (`sme_thread_cap`, the `ModelSMCU` brand-string table, the `ne11 >= 128` hybrid rescue path) | Not present — no Apple Silicon / SME coverage at all |
| **L3 debugger-level dispatch proof with true call counts** — `lldb`/`gdb` breakpoint on every `kai_run_matmul_*` entry point, auto-continuing so the hit count is a real per-symbol call count (`tools/verify_dispatch.py`) | A static/selection-level advertised-vs-dispatched probe (`src/isa-probe.c`); does not attach a debugger to count real kernel invocations |
| **Upstream filing** — [ggml-org/llama.cpp#26547](https://github.com/ggml-org/llama.cpp/issues/26547), both findings reported to the maintainers with reproduction commands and an offer to send a patch | No upstream filing found in their repository |
| **Cross-platform dispatch ledger** — one JSON/Markdown schema spanning Apple Silicon (SME2) and Arm Linux (SVE2/NEON/i8mm), published to a live dashboard (`site/`, `.github/workflows/pages.yml`) | Single-platform (Arm Linux / Neoverse-N2), no cross-platform ledger or dashboard |
| **Patch experiments** — a real, measured, opt-in `llama.cpp` patch (`patches/0001-kleidiai-phase-aware-dispatch.patch`) attempting to fix the thread-gating gap, with an honest negative result reported (`docs/FINDINGS.md`) | No patch attempt found in their repository — diagnosis only |

Both projects independently landed on the identical `kleidiai.cpp:209` line and the identical
`QK8_0`-equality reasoning for Finding 2's mechanism — which, if anything, is corroborating evidence
that the finding itself is correct, arrived at twice by unrelated methods (their static ISA probe;
this project's source read plus `tools/verify_dispatch.py`'s L1 tier).

---

## Other Track 2 (Cloud AI) entries we are aware of

Checked live via `gh api repos/<owner>/<repo>` on 2026-08-04; all five repositories below currently
exist and are active submissions in the same track.

| Repo | What it does | How this project differs |
|---|---|---|
| [`yannan000/kleidibench`](https://github.com/yannan000/kleidibench) | A free-CI harness that answers "which GGUF quant + KleidiAI build flag is fastest for my model, and what does it cost per million tokens" — a serving-decision report (quant, RAM, quality cost, $/Mtok), not a dispatch-correctness check. | Orthogonal, not competing: kleidibench assumes the advertised kernel actually dispatches and measures throughput/cost across quants. This project's core claim is that the advertised kernel silently *doesn't* dispatch for a common workload (decode) — exactly the kind of gap a throughput-only benchmark like kleidibench has no way to detect on its own. |
| [`Sombra-1/arm-agent-optimizer`](https://github.com/Sombra-1/arm-agent-optimizer) ("AArchTune") | Searches `llama.cpp` runtime configurations (threads, batching, parallel slots, mmap, prompt caching) and recommends one only if it is both faster and passes workload-correctness/quality gates; its strongest reported result is correctly recommending *no* candidate out of 132 tried. | A config-space search with a correctness gate, not a dispatch-verification tool — it does not attach a debugger to confirm which kernel family actually executed for a winning config. This project's L1/L2/L3 method is the kind of check a search like AArchTune's would need in order to know *why* a given thread count wins, not just *that* it does. |
| [`QasimKhan5x/VerifyLane`](https://github.com/QasimKhan5x/VerifyLane) ("SurgeDesk + ArmProof") | An application-level demo (banking-support triage) plus `ArmProof`, a fail-closed CI release gate that approves an Arm deployment only when evidence shows the quality contract holds, the serving objective improves, the Arm acceleration path executes, and the run reproduces from pinned artifacts. | End-to-end product-plus-release-gate framing on a different model and workload (Phi-4 Mini INT4 on AWS Graviton4 via ONNX Runtime GenAI). Does not touch `llama.cpp`/KleidiAI dispatch internals or Apple Silicon SME2 at all. |
| [`StephenSook/gravitonkv`](https://github.com/StephenSook/gravitonkv) ("GravitonKV") | A reproducible KV-cache-quantization tradeoff study on AWS Graviton4 (`llama.cpp` + KleidiAI as the fixed baseline): sweeps `--cache-type-k/-v` precision across models and context lengths, measuring the prefill/decode/memory/quality three-way trade, plus a PMU-level mechanism pass. | A different lever entirely — KV-cache precision, not matmul kernel dispatch. It takes KleidiAI's dispatch behavior as a given fixed baseline rather than auditing whether the advertised kernel actually runs, which is exactly this project's question. |
| [`agrovr/ParetoPilot`](https://github.com/agrovr/ParetoPilot) | Compares Arm64 `llama.cpp` inference configurations (quant, thread count, concurrency) against published, provenance-checked benchmark archives and recommends a configuration for a stated goal (latency vs. throughput vs. memory). | A configuration-comparison and provenance-archival tool operating on reported throughput numbers as given. It does not independently verify, at the dispatch level, whether the kernel family a configuration *reports* using is the one that actually executed — again, this project's question, not theirs. |

None of the five above overlaps with either of this project's two findings; they are listed here for
completeness and honest positioning within the track, not because any of them scooped anything.

---

## What this page does and doesn't claim

- It does **not** claim priority on Finding 2's mechanism. `luongs3/arm-dispatch-audit` published it
  first, and that is stated above without qualification.
- It **does** claim Finding 1 (SME2 decode thread-gating on Apple Silicon) is, as far as this project
  could verify on 2026-08-04, original to this repository — a claim made falsifiable by the exact
  grep command described above, reproducible by anyone.
- It does not attempt to rank this project against the other five Track 2 entries listed — only to
  describe what each does, honestly, so a reviewer does not have to reconstruct the landscape
  themselves.
