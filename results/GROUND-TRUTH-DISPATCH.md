# Ground truth: when does KleidiAI actually dispatch SME2?

> **Authoritative.** Measured on this machine 2026-08-04. Any statement in the README, docs, or
> submission copy that contradicts this file is wrong and must be corrected to match it.
> An earlier draft of this finding was **incomplete** — see "Correction" below.

## Environment

| | |
|---|---|
| Host | Apple M4 Max, macOS 27, 16 cores (12P/4E) |
| ISA | `FEAT_SME=1`, `FEAT_SME2=1`, `sme_max_svl_b=64` (512-bit SVL), `FEAT_I8MM=1`, `FEAT_BF16=1`; **`FEAT_SVE` absent** |
| llama.cpp | `dbadb68`, built `-DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF` |
| Model | `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (337 MB, Apache-2.0, license verified live) |
| Method | `lldb` regex breakpoint on `kai_run_matmul.*sme` (**18 locations resolved**), count `stop reason = breakpoint` |

## The measurement

```
GGML_KLEIDIAI_SME=2 lldb -b -s probe.lldb -- ./bin/llama-cli \
    -m q05.gguf -p "<PROMPT>" -n 4 -no-cnv -st --simple-io -t <THREADS>
```

| threads | short prompt (~4 tok) | long prompt (~400 tok) |
|---:|:---:|:---:|
| 1  | fires | — |
| 2  | **fires** | **fires** |
| 4  | **zero** | — |
| 8  | **zero** | **fires** |
| 16 | **zero** | **fires** |

## Correction to the earlier draft

The first draft of this finding said *"SME2 is silently thread-gated; it never fires above 2
threads."* **That is not correct.** The real rule has two independent paths, and the earlier test
only exercised one because it used a 4-token prompt.

## The actual rule, read from source

`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp` (~L1094–1112):

```c
const int  sme_cap_limit = ctx.sme_thread_cap;
const bool use_hybrid    = sme_cap_limit > 0 && runtime_count > 1 && nth_total > sme_cap_limit;

size_t min_cols_per_thread = std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);

const bool hybrid_enabled = use_hybrid && !too_small_for_hybrid;

if (!hybrid_enabled) {
    ...
    } else if (runtime_count > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
        chosen_slot = 1;          // <-- collapses to the NON-SME (NEON) slot
    }
}
```

SME2 dispatches if **either**:
1. `n_threads <= sme_thread_cap` — the single-slot SME path; or
2. **hybrid mode**: `n_threads > sme_thread_cap` **AND** `ne11 >= 128` **AND** `ne01/n_threads >= 2`.

Otherwise the kernel chain collapses to slot 1, the NEON kernel
(`kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod` for Q4_0).

`sme_thread_cap` comes from `detect_num_smcus()`, which on Apple is a **hardcoded brand-string
table**: `{"M4 Ultra",2}, {"M4 Max",2}, {"M4 Pro",2}, {"M4",1}`. So the cap is 2 on this machine.
An M5/M6 or an unlisted brand string falls through the table.

## Why this matters — the load-bearing consequence

`ne11` is the number of columns of `src1`, i.e. **the batch size / number of tokens processed in one
matmul**.

- **Prefill** of a long prompt: `ne11` is large → `ne11 >= 128` holds → hybrid engages → **SME2 runs.**
- **Decode** (autoregressive token generation): `ne11 == 1`, always → `too_small_for_hybrid` is
  always true → **SME2 never runs at default thread counts**, on any prompt.

So on Apple Silicon, **SME2 accelerates prefill but is structurally unreachable for token-by-token
decode** unless you drop to `-t 2` — which costs you 14 of 16 cores. That is a real, quantifiable
engineering trade-off, and it is the thing worth measuring.

Throughout, `system_info:` prints `SME = 1 | SME2 = 1 | KLEIDIAI = 1` and the log prints
`kleidiai: primary q4 kernel feature SME2` — **in every one of the zero-dispatch rows above.**
Both are compile-time / selection-time signals. Neither reflects dispatch.

## Second finding — the SVE width gate — **CONFIRMED ON HARDWARE 2026-08-04**

Same file, ~L209:

```c
((ggml_cpu_has_sve() && ggml_cpu_get_sve_cnt() == QK8_0) ? CPU_FEATURE_SVE : CPU_FEATURE_NONE)
```

`QK8_0 == 32` bytes == 256-bit. Any core implementing SVE2 at 128-bit can never set
`CPU_FEATURE_SVE`, so the SVE kernel family is unreachable there — despite the core genuinely
having SVE2, i8mm and bf16.

**Status upgraded from "derived from source" to MEASURED.** Confirmed on GitHub's free
`ubuntu-24.04-arm` runner (**Neoverse-N2**, 4 cores), CI run
[`30862916023`](https://github.com/tomyimkc/arm-dispatch-ledger/actions/runs/30862916023) — a
platform *any judge can re-run at zero cost*, which is stronger evidence than the DGX Spark would
have been.

Evidence chain from that run:
- `/proc/cpuinfo` Features advertises **`sve sve2 sveaes svebitperm svesha3 svesm4 svei8mm svebf16
  i8mm bf16`** — the hardware genuinely has SVE2.
- L1 static scan of `libggml-cpu.so` finds SVE kernels present and compiled in:
  `kai_symbols_by_family = {dotprod: 6, i8mm: 2, sve: 2}`, plus 26,629 SVE z-register operands.
- L2 selection nonetheless reports `primary q4/q8 kernel feature = I8MM`, **never SVE**.
- L3 dispatch confirms execution is `i8mm` / `dotprod`; the SVE family is never entered.

So on Neoverse-N2 the SVE kernels are compiled in, the silicon supports SVE2, and the dispatcher
still cannot select them — exactly as the `== QK8_0` gate predicts.

The DGX Spark (Cortex-X925, also 128-bit SVE2) is predicted to behave identically but **has not been
run**: its self-hosted runner is registered to a different repository. Keep that specific claim
labelled `[not measured on Spark]`.

## Third observation — Neoverse-N2 decode also advertises one kernel and runs another

From the same CI run, `dispatch-ledger-Linux-aarch64.json`:

| threads | workload | advertised (L2) | executed (L3) | hits |
|---:|---|---|---|---:|
| 1 | decode_short | I8MM | **dotprod** | 1,014 |
| 2 | decode_short | I8MM | **dotprod** | 8,112 |
| 4 | decode_short | I8MM | **dotprod** | 16,224 |
| 1 | prefill_long | I8MM | i8mm | 672 |
| 2 | prefill_long | I8MM | i8mm | 5,376 |
| 4 | prefill_long | I8MM | i8mm | 10,752 |

Decode advertises I8MM and executes dotprod; prefill advertises I8MM and executes I8MM. This is the
**same class of advertised-vs-executed divergence as Finding 1, on completely different silicon and
a completely different kernel family** — evidence that the gap between what KleidiAI reports and
what it runs is systemic, not an Apple-specific quirk.

This is an *observation*, not yet a root-caused finding: the likely mechanism is the GEMV-vs-GEMM
kernel split (decode is `ne11 == 1`, i.e. a GEMV, and the dotprod GEMV kernel may simply be the
selected variant for that shape) rather than a bug. **Do not present it as a defect until that is
read out of the source.** It is reported here because the ledger measured it.

## Methodology caveat — do not overstate the hit counts

The counts above are **lldb stop events, not kernel invocation counts.** With `lldb -b` and no
auto-continue the process stops at the breakpoint and the script ends, so the absolute number is not
a call count. **Only `zero` vs `non-zero` is meaningful here.** To get true call counts the verifier
must attach a breakpoint command that auto-continues, e.g.:

```
break set -r "kai_run_matmul.*sme"
break command add -s python 1
> import __main__; __main__.n = getattr(__main__,'n',0)+1
> return False
DONE
```

Report `dispatched: true/false` plus a separately-derived call count; never present a stop count as
a call count.

## Prior-art check (done 2026-08-04)

- `sme_thread_cap` appears **nowhere** in llama.cpp's docs, and returns nothing on web search.
- llama.cpp *did* add a one-shot `GGML_LOG_WARN` for the **weight-type** silent fallback
  (non-`Q4_0`/`Q8_0` tensors) in PR #25701 — precedent for exactly the fix proposed for this gate,
  which strengthens the upstream ask rather than pre-empting it.
- Issue #22182 independently notes `ggml_cpu_has_sme()` is compile-time, so the banner can mislead.

Nothing found that documents the thread-cap / `ne11 >= 128` dispatch rule.

---

## Addendum — 2026-08-05: replication sweep, probe fail-closed check, and probe cost

Added below the original run rather than edited into it, per this repo's rule that a results page
records what was measured when it was measured.

**The SVE width-gate result replicates across the full sweep.** The 2026-08-04 finding above was
confirmed on a single configuration. It has now been re-run on the DGX Spark across **all ten**
configurations — threads 1/2/4/8/16 x {decode_short, prefill_long}. The SVE kernel family is
compiled in (2 symbols) and took **0 hits in all ten**. Executed family was `dotprod` for decode
and `i8mm` for prefill, while L2 advertised `I8MM` throughout — recorded as
`I8MM_HYBRID_DISPATCH`. Total hits ranged 660 to 51,216 and scaled with thread count.

**The probe's fail-closed behaviour was verified live.** Pointed at the zero-symbol default build,
the L3 probe refused to run rather than returning a zero:

> `no symbols matching '^kai_run_matmul' found in .../libggml-cpu.so; refusing to run an
> uninstrumented probe that would report a misleading zero`

This matters because an earlier version of this probe *did* silently report zero hits, which
looked exactly like a clean negative result. That bug is why `tests/l3_gdb_groundtruth/` exists.

**The probe's cost was measured** — round-robin interleaved, 5 reps per arm, threads=4,
decode_short: plain median **1.2056 s** vs. under-probe median **4.379 s**, a **3.63x** median
overhead. The dispatch count was identical (15,936) on all five probed runs, so the count is
deterministic even though the wall clock is not (spread 2.68 s under gdb vs 0.21 s plain).

**The Arm PMU could not be used as an independent check on this machine.** Two independent
reasons — `perf_event_paranoid = 4`, and a kernel PMU driver that enumerates 78 events with no
SVE/SME/ASE instruction-class counter among them. The intended PMU-vs-L3 cross-validation is
therefore **unperformed, not passed**. Full detail and provenance:
[`results/pmu/pmu-crosscheck.json`](pmu/pmu-crosscheck.json).

---

## Addendum 2026-08-06 — which bytes did this page measure? A provenance gap, disclosed

**Trigger.** On 2026-08-06 the cached baseline `/tmp/ggufs/q05.gguf` failed a check against
`scripts/models.txt`: the file on disk is sha256
`c8cd5f37dd1235fb010c45316d4ff8af875e1a4e0ff368b4bf6cacb9053d4919`, 352,972,352 bytes; the
manifest lists `7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed`.

**What checking upstream established.** Hugging Face's tree API reports the official
`Qwen/Qwen2.5-0.5B-Instruct-GGUF` q4_0 blob as `7671c0c3…`, **428,730,208 bytes**, with the
repo unchanged since 2024-09-20. So no ~337 MB file with the manifest's hash ever existed:
the manifest row pairs the HF file's real hash (captured by a real Aug-4 download — of a
428.7 MB file) with the size and role of the local baseline. Those were never the same file,
and the manifest's claim that every committed Finding 1/2 measurement used the hash-listed
file is wrong as written.

**What the measured baseline actually was.** The ledger this page narrates
(`dispatch-ledger-darwin-arm64.json`, generated 2026-08-03T22:26Z) records only a model
*path*, not a hash — as did every ledger produced before this addendum. The strongest
available identification: Finding 4's independently committed evidence
(`upstream/llamacpp-26334-cuda-host-buffer.json`, Arm test box, 2026-08-05) records its
`q05.gguf` as sha256 `c8cd5f37…` — the same bytes as the file on this machine today, whose
mtime predates the Aug-3 run. On that evidence the working baseline across both machines was
the `c8cd5f37…` file (352,972,352 bytes), whose original source is **unrecorded** — it did
not come from the manifest's HF URL, whose file has been 428,730,208 bytes for its entire
history. This is identification, not proof: no darwin ledger recorded a hash at measurement
time, which is exactly the gap being disclosed.

**Corroboration by re-measurement (2026-08-06).** A fresh sweep on this machine using the
true HF file (sha `7671c0c3…` verified before use) at llama.cpp `a035a8887` — changing
**both** the model bytes **and** the commit relative to this page (`q05.gguf` @ `dbadb68`) —
reproduces **all 10 dispatch verdicts identically**, with per-cell L3 totals uniformly
+1.8–2.2% (ratios 1.0181–1.0218; e.g. 4-thread decode 15,936 → 16,224). Raw evidence:
[`upstream/pr26076/`](upstream/pr26076/). The count shift is jointly attributable to the two
changed variables and is not decomposed here. What this corroborates is the load-bearing
content of this page: the SME2 dispatch **rule** (cap-gated decode, hybrid prefill, kernel
family selection) is invariant across both model files and both commits. Absolute hit counts
were never the claim.

**Fix going forward.** As of the commit adding this note, `tools/verify_dispatch.py` writes
`model_sha256` and `model_bytes` into every ledger it produces, so "which bytes did this
measure" is answerable from the artifact alone. Per this repo's rules no existing results
file was edited; this note is appended.

---

## Addendum 2026-08-06 (II) — a tensor-level check: same architecture/vocab/license, not confirmed same weights

**Why this addendum, given the one above already disclosed the gap.** The addendum above
already avoids claiming the two files hold identical weights ("whose original source is
unrecorded"; "those were never the same file") — it needs no correction. This addendum exists
because a later review pass proposed a stronger, specific claim not yet checked at the time
the addendum above was written: that the two files are "the same … weights … differing only
in how llama.cpp's Q4_0 quantizer was configured." That claim does not survive a direct
tensor-value check and is recorded as false here so it is not repeated.

**What was checked.** Both files' 290 shared tensors were parsed with `gguf-py` and compared
by actual value, not just name/shape/type. 72 of 290 are byte-identical (`output_norm.weight`
and almost all `attn_*.bias` tensors). The other 218 — including every one of the 24 blocks'
`attn_norm.weight` and `ffn_norm.weight` tensors — differ. Those norm tensors are stored as
plain F32 and are exempt from Q4_0 quantization (GGUF's `file_type=2` spec excludes 1D
tensors) and are not touched by imatrix or tied/untied-output logic in llama.cpp's quantizer
(`src/llama-quant.cpp`) or in Qwen2's own HF→GGUF converter — both leave an F32 tensor as an
untouched byte copy when source and target type match. This session independently re-checked
three of those tensors directly: `blk.0.attn_norm.weight`, `blk.5.ffn_norm.weight`, and
`blk.23.attn_norm.weight` have correlations of 0.89, 0.95, and 0.91 respectively between the
two files (not the ~1.0 identical floats would give), while `output_norm.weight` in the same
two files is bit-identical (correlation 1.0) — confirming the parsing/comparison itself is
sound, not a systematic bug. A broader pass across all 48 `attn_norm`/`ffn_norm` tensors
(reported by the review that triggered this addendum, not independently re-run in full here)
found correlations ranging 0.625–0.99 and a linear best fit (`b = m·a + c`) with R² as low as
0.39 — well short of the ~1.0 a lossless rescale would produce.

**What this does and does not establish.** Since no quantizer-configuration mechanism can
explain a difference in tensors no quantizer touches, "same weights, different quantizer
build" is not a supportable description of the relationship between the 352,972,352-byte file
(sha `c8cd5f37…`) and the 428,730,208-byte HF file (sha `7671c0c3…`). What the two files do
share, directly verified: identical architecture (`qwen2`, 24 blocks, `embedding_length` 896),
byte-identical tokenizer/vocab (151,936 tokens, merges, token_type), and self-declared
Apache-2.0 license metadata. Whether the 352,972,352-byte file is a different trained
checkpoint of the same architecture, or something else, is **not established** here and
remains open — its download provenance is still unrecorded (see the addendum above). Nothing
here changes the corroboration-by-re-measurement finding above (the SME2 dispatch rule
reproducing across both files): that result concerns which *kernel* the code dispatches to,
not whether the two files hold identical trained parameters. Per this repo's rules no
existing results file was edited; this note is appended.

---

## Addendum — 2026-08-06 (III): the baseline model is the faithful one. Earlier addenda understated this.

Appended, not edited, per this page's append-only rule. This corrects the *emphasis* of addenda I
and II, which described the historical baseline's provenance as an open gap without establishing
what the file actually is.

It has now been established, and the answer runs the other way.

`model.safetensors` was range-fetched directly from `Qwen/Qwen2.5-0.5B-Instruct` — the source
weights repository, not the GGUF one — its header parsed, and three F32 layer-norm tensors
converted BF16→F32 and compared element by element against the same tensors read out of both local
GGUF files. Layer norms are stored as F32 and are never touched by the Q4_0 quantizer, so any two
faithful conversions of the same checkpoint must agree on them bit for bit.

| | `blk.0.attn_norm` | `blk.5.ffn_norm` | `blk.20.attn_norm` |
|---|---|---|---|
| **353 MB baseline** (`c8cd5f37…`) — produced every Finding 1/2 measurement | **896/896 exact**, max diff **0** | **896/896 exact**, max diff **0** | **896/896 exact**, max diff **0** |
| 428 MB file named by `scripts/models.txt` (`7671c0c3…`) | 0/896, max diff 0.48 | 0/896, max diff 1.90 | 0/896, max diff 4.24 |

**The file this project measured against is a bit-exact conversion of the canonical upstream
weights.** The file the manifest pointed at is not. The manifest was wrong in the direction
opposite to the one addendum I implied.

Two consequences worth stating plainly:

- **The Apache-2.0 attribution is sound on the strongest available basis.** The weights are
  demonstrably the upstream Qwen weights, not merely a file that claims to be.
- **The measurements never needed re-running.** Addendum II left open whether the committed
  Finding 1/2 numbers should be re-measured against a verifiable file. They were already taken
  against the more faithful of the two.

What remains genuinely unrecorded is the **download URL** the baseline came from. It is identified
here by content, not by origin — which is the stronger of the two, but it is not the same as a
provenance chain, and this page does not claim otherwise.

Not established, and not claimed: why the published `Qwen2.5-0.5B-Instruct-GGUF` q4_0 blob
diverges from the weights it is named for. That is an observation about a third-party artifact.
Noted only, without inference: its own metadata self-reports `general.size_label = "630M"` for an
architecture both files agree is 24 blocks at embedding length 896.

Evidence, with the full method and every literal count:
[`results/provenance/baseline-model-identity-2026-08-06.json`](provenance/baseline-model-identity-2026-08-06.json).
