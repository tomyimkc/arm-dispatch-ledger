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
