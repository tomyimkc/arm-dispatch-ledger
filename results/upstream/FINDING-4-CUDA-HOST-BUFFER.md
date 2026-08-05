# Finding 4 — a CUDA build silently switches KleidiAI off, and the banner still says it is on

**Measured 2026-08-05.** Evidence: [`llamacpp-26334-cuda-host-buffer.json`](llamacpp-26334-cuda-host-buffer.json)
(15 runs, 3 arms, 5 round-robin-interleaved reps each).

## What happens

Build `llama.cpp` with **both** `-DGGML_CPU_KLEIDIAI=ON` and `-DGGML_CUDA=ON`, then run CPU-only
inference with `-ngl 0`. Every KleidiAI kernel is compiled into the binary. The startup banner
prints `KLEIDIAI = 1`. The selection log prints `kleidiai: primary q4/q8 kernel feature I8MM`.

**Not one KleidiAI kernel executes.**

| arm | command | KleidiAI dispatch hits | polygraph exit |
|---|---|---|---|
| default | `llama-cli -ngl 0 …` | **0** | `1` (mismatch) |
| `--no-host` | `llama-cli -ngl 0 --no-host …` | **7,968** | `0` (match) |
| `-dev none` | `llama-cli -ngl 0 -dev none …` | **7,968** | `0` (match) |

5/5 interleaved reps per arm, zero variation between reps. Hit counts are real non-halting `gdb`
breakpoints on all 10 `kai_run_matmul_*` symbols — not inferred from logs.

## Why it is invisible

`make_cpu_buft_list()` (`src/llama-model.cpp`) builds the CPU buffer-type priority list in a fixed
order: **ACCEL → GPU host → CPU extra → CPU**. KleidiAI registers as a *CPU extra* buffer type.
A CUDA **host** (pinned-memory) buffer type sits ahead of it and is added whenever a CUDA device
appears in the model's `devices` list — and `-ngl 0` does **not** remove CUDA from `devices`; it
only sets how many layers get offloaded. `select_weight_buft()` returns the *first* buffer type
that supports the op, so KleidiAI's entry is never even tried.

The one runtime line that would reveal this —
`cannot be used with preferred buffer type CUDA_Host, using CPU instead` (291 tensors) — is
logged at `LLAMA_LOG_DEBUG`. **At default verbosity it does not print at all.**

So all three of the signals a normal user has agree with each other and are all wrong:

| signal | buggy run | fixed run |
|---|---|---|
| `KLEIDIAI = 1` banner | identical | identical |
| `kleidiai: primary q4/q8 kernel feature I8MM` | identical | identical |
| `nm` symbol count (L1) | 10/149 | 10/149 |
| **actual kernel dispatches (L3)** | **0** | **7,968** |

This is the cleanest example the project has of why L3 exists. L1 and L2 both say "fine". Only
counting real execution finds it.

## Why this one matters more than Findings 1–3

Finding 3 needs three simultaneous non-default conditions, including a compiler older than the
CPU. **This needs none of that.** It needs two build flags people deliberately turn on together
and one runtime flag (`-ngl 0`) that is the normal way to ask for CPU inference. On a Grace,
Jetson, or DGX Spark class machine — Arm CPU next to an NVIDIA GPU — that combination is the
obvious thing to build.

## Provenance and honest limits

The mechanism was reported upstream as
[ggml-org/llama.cpp#26334](https://github.com/ggml-org/llama.cpp/issues/26334) by `izard`, who
diagnosed the buffer-ordering cause by reading the source. A contributor confirmed the two CLI
workarounds and the reporter closed it the next day. **We are not claiming discovery of the
mechanism** — we are contributing the thing the thread never had: a measured reproduction showing
the kernel count actually goes to zero and actually comes back.

What this does **not** establish:

- **No performance claim.** We measured *whether* the kernel ran, not how much slower the fallback
  is. The reporter's qualitative "~2x slower" is theirs, not ours; we ran no timing sweep.
- **Not the reporter's machine.** They disclosed only "Linux" and a version hash. We reproduced the
  code mechanism on hardware where it builds, not their exact environment.
- **Not a bug verdict.** Whether host-buffer-before-extra-bufts is a deliberate trade-off or an
  oversight is upstream's call. This page shows the observable consequence of the current ordering.
- **The issue is closed**, resolved by workaround rather than a code change. The honest framing is
  "the mechanism still reproduces at HEAD `dbadb68ee`", not "this was never fixed".
- This build deliberately used this project's own `GGML_NATIVE=OFF` + `GGML_CPU_ARM_ARCH`
  workaround so that KleidiAI symbols were genuinely compiled in — otherwise Finding 3's
  zero-symbol failure would have been a confound and the zero hits would have meant nothing.

## Reproduce it

```bash
cmake -S . -B build-cuda-kleidiai -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CPU_KLEIDIAI=ON -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH="armv9.2-a+sve2+i8mm+bf16+dotprod"
cmake --build build-cuda-kleidiai -j

# 0 hits, exit 1
tools/polygraph check llama-cpp-kleidiai \
  --binary build-cuda-kleidiai/bin/llama-cli --model model.gguf

# 7,968 hits, exit 0 -- same binary, same model, one extra flag
tools/polygraph check llama-cpp-kleidiai \
  --binary build-cuda-kleidiai/bin/llama-cli --model model.gguf --param extra=--no-host
```
