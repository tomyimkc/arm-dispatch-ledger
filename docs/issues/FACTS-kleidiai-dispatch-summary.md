# FACT SHEET — KleidiAI dispatch-summary series (for a HUMAN to write posts from)

**Rule that governs this file's use:** llama.cpp CONTRIBUTING.md — "It is strictly prohibited
to use AI to write your posts for you (bug reports, feature requests, pull request
descriptions, Github discussions, responding to humans, ...)". This file is verified raw
material: numbers, quotes, links, and required elements. Compose the issue and PR text in
your own words. Do not copy sentences from earlier AI drafts. Quoting log output, code, and
the numbers below verbatim is fine — they are data, not prose.

Every fact below was verified against primary sources on 2026-08-06 by two independent
adversarial review rounds; the two errors those rounds caught are already corrected here.

## The problem (what the issue must convey)

- Nothing at runtime reports whether a KleidiAI micro-kernel actually executed:
  - `KLEIDIAI = 1` in system info = compile-time flag (`ggml/src/ggml-cpu/ggml-cpu.cpp:632`).
  - `kleidiai: primary q4 kernel feature SME2` = init-time selection
    (`ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`, init_kleidiai_context).
  - `load_tensors: CPU_KLEIDIAI model buffer size` = allocation.
- Measured defect (Finding 4): `-DGGML_CPU_KLEIDIAI=ON -DGGML_CUDA=ON`, run `-ngl 0` on an
  Arm CPU + NVIDIA GPU machine → **0** micro-kernel invocations vs **7,968** with
  `--no-host`; banner, selection log, and symbol table byte-identical across the two runs.
  Evidence: https://github.com/tomyimkc/polygraph/blob/main/results/upstream/FINDING-4-CUDA-HOST-BUFFER.md
- Mechanism (CREDIT UPSTREAM #26334 — first identified from source there, by user izard):
  - Weight buffer-type priority comment, verbatim: `// CPU: ACCEL -> GPU host -> CPU extra -> CPU`
    (`src/llama-model.cpp:895`, above `make_cpu_buft_list()`).
  - **`select_weight_buft()`** (`src/llama-model-loader.cpp:1046`, called from create_tensor
    ~line 1198) returns the FIRST buffer type that supports the op — first-match-wins.
    (NOT `select_buft()` in llama-model.cpp — that one is for control-vector tensors only.
    An earlier draft got this wrong; a maintainer will check.)
  - `-ngl 0` does not remove CUDA from the device list (`llama_prepare_model_devices()`
    enumerates every GPU device regardless of n_gpu_layers).
  - The only runtime signal is DEBUG-level, verbatim from the Finding-4 run:
    `cannot be used with preferred buffer type CUDA_Host, using CPU instead`
    (`src/llama-model-loader.cpp:1320`; never prints at default verbosity).
    (NOT "...CPU_KLEIDIAI" — that variant is from a different, non-CUDA context.)
  - The `no_host` field's own comment: `// bypass host buffer allowing extra buffers to be used`
    (`common/common.h:572`, `include/llama.h:338`).
- Related, with one-line glosses:
  - #26547 — SME2 dispatch silently thread-gated; SVE family unreachable below 256-bit (own filing).
  - #26630 — documented KleidiAI build line compiles zero kernels while banner says KLEIDIAI = 1 (own filing).
  - #25701 — merged warn-once for unsupported quant types; its review established the wording
    distinction: a KleidiAI decline "does not necessarily imply a generic CPU fallback".
- There is NO KleidiAI-specific coverage in `tests/` today (verified: zero matches).
- **No single CI job builds CUDA and KleidiAI together** — so the Finding-4 configuration is
  not CI-testable upstream, which is why the proposed test is the CPU-only
  selected-implies-executed proxy. Verified job-by-job on the PR-head tree 2026-08-06:
  KleidiAI is enabled by exactly two jobs, `cpu-arm64-graviton4-kleidiai`
  (`.github/workflows/build-self-hosted.yml`) and `server-kleidiai`
  (`.github/workflows/server-self-hosted.yml`), both on the Arm runner
  `ah-ubuntu_22_04-c8g_8x`, neither setting `GGML_CUDA=ON`. CUDA is enabled by four jobs
  (`build-cuda-ubuntu.yml`/`cuda` on ubuntu-24.04, `build-cuda-windows.yml`/`cuda` and
  `release.yml`/`windows-cuda` on windows-2022, `server-self-hosted.yml`/`server-cuda` on a
  self-hosted Linux+NVIDIA runner), none setting a KleidiAI flag. (Note: three workflow
  FILES mention both strings; the separation is at job level, so cite jobs, not files.)
- The synthetic-model test path cannot cover this: `llama_model_init_from_user` hard-sets
  `use_extra_bufts = false` (`src/llama.cpp:441`).

## The series (what the PR must describe)

Two commits on local branch `proto-kleidiai-dispatch-summary` (in /tmp/llama-26076-pr,
also as /tmp/kleidiai-dispatch-summary-series.patch, 349 lines):

1. `kleidiai: count micro-kernel dispatch, add opt-in GGML_KLEIDIAI_SUMMARY report`
   - kleidiai.cpp +93, kleidiai.h +6 (= 99 lines under ggml/, no new files there), docs/build.md +8.
   - One relaxed atomic add at each of the THREE `run_kernel_ex` call sites, keyed by
     op path (f32 / fp16 / qx_gemm / qx_gemv) × kernel family (SME2/SME/SVE/I8MM/DOTPROD/OTHER).
   - Counting always on; `GGML_KLEIDIAI_SUMMARY=1` (parsed with existing `parse_uint_env`)
     registers an atexit handler printing via GGML_LOG_INFO.
   - Internal accessor `ggml_backend_cpu_kleidiai_dispatch_total()` (kleidiai.h only; no public API).
2. `tests: add fail-closed KleidiAI dispatch-execution test`
   - tests/test-kleidiai-dispatch.cpp (the only new file, 146 lines) + CMake registration
     guarded `if (GGML_CPU_KLEIDIAI AND NOT GGML_BACKEND_DL)`; default `main` ctest label →
     runs on the existing `cpu-arm64-graviton4-kleidiai` CI job
     (`.github/workflows/build-self-hosted.yml:350`, runs `ctest -L main` via ci/run.sh).
   - Q4_0 weight allocated into the KleidiAI buffer type via
     `ggml_backend_alloc_ctx_tensors_from_buft`; MUL_MAT at batch 1 (gemv) and 8 (gemm);
     asserts the counter advanced; SKIPS visibly if the buffer type declines the weight.
- Total: 258 insertions, 0 deletions, 5 files.
- Overhead statement: one relaxed fetch_add per micro-kernel invocation; no other work
  unless the env var is set.

## Validation numbers (all committed / reproducible)

- Cross-validation vs independent non-halting lldb breakpoint counting on `kai_run_matmul_*`
  (same binary, model sha256 7671c0c3…, workload `-p "Hello." -n 4 -no-cnv -st --simple-io`),
  Apple M4 Max, PR #26076 head 7d83248 + series, AppleClang 21.0.0.21000323:
  - t=2: summary total **5952** (qx_gemm/SME2=2883 + qx_gemv/SME2=3069) vs L3 ledger **5952** — exact.
  - t=8: summary total **32448** (gemm/DOTPROD=15840, gemv/I8MM=192, gemv/DOTPROD=16416) vs
    L3 ledger **32445** — differs by 3. HONEST WORDING: a small gap plausibly from
    thread-scheduling nondeterminism between independently-invoked runs; repeated-trial
    variance was NOT collected, so do not call it a "±3 run-to-run band".
  - Record (verbatim outputs + commands):
    https://github.com/tomyimkc/polygraph/blob/main/results/upstream/pr26076/dispatch-summary-crossvalidation.txt
- Example summary block (verbatim tool output, safe to paste as a code block):
  ```
  kleidiai: dispatch summary: selected q4=SME2 q8=SME2 f32=SME2
  kleidiai: dispatch summary: executed qx_gemm/SME2=2883
  kleidiai: dispatch summary: executed qx_gemv/SME2=3069
  kleidiai: dispatch summary: executed total=5952
  ```
- Test result on M4 Max: 4 invocations per case, both cases OK; failure path exercised by
  temporarily inverting the assertion (exit 1 with diagnostic).
- llama-cli filters INFO by default: the summary needs `--verbose` (same as the existing
  selection lines). The raw test binary prints without it.
- Linux/aarch64, dated 2026-08-06, an Arm CPU + NVIDIA GPU machine (Cortex-X925, gcc 13.3;
  refer to it publicly only that way) — **the fail-closed case, demonstrated on a real
  defective build**: the documented build line (`cmake -B build -DCMAKE_BUILD_TYPE=Release
  -DGGML_CPU_KLEIDIAI=ON`, PR #26076 head 7d83248 + series) reproduces the Finding-3 class:
  runtime detection correctly reports the CPU features (`no compatible q4 kernels found for
  CPU features mask 3` — mask 3 = dotprod+i8mm) while the compile-time gates produced an
  empty kernel table. The new instrumentation reported it exactly as designed, verbatim:
  - `kleidiai: dispatch summary: selected q4=none q8=none f32=none`
  - `kleidiai: dispatch summary: executed total=0`
  - test: `0 KleidiAI kernel invocations - FAIL` on both batch shapes, exit 1 with
    "the KleidiAI buffer type accepted the weight but executed zero micro-kernels".
  Model was the faithful baseline (sha256 `c8cd5f37…`, 352,972,352 bytes).
  ADDITIONAL FINDING exposed by the test (fragments — compose your own sentence if you use
  it): on the PR head; buffer type ACCEPTS + packs a Q4_0 weight; kernel table empty;
  fallback at compute time; only signals = a WARN (`no runtime kernel slot available for
  supported op`) and the zero counter.
  HONEST FRAMING RULE: this run validates the fail-closed behavior, NOT "the test passes on
  Linux" — the pass case needs a corrected-arch build (see next bullet).
- Linux/aarch64 pass case, dated 2026-08-06 (corrected build, the documented Finding-3 fix
  flags `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9.2-a+sve2+i8mm+bf16+dotprod`): **test
  PASSES** (`primary q4/q8 kernel feature I8MM`, 8 invocations per batch shape, exit 0);
  summaries: t=1 total=996 (gemm/I8MM=486 + gemv/I8MM=510), t=4 total=**15,936**
  (gemm/I8MM=7776 + gemv/I8MM=8160). `f32=none` is expected — the f32 kernels are
  SME-family and this CPU has no SME.
  **THE HEADLINE CROSS-CHECK:** 15,936 at t=4 equals, exactly, the committed 2026-08-05
  gdb-based L3 count on this same machine/model/workload ("15,936 hits on every one of the
  five probed runs", docs/PRIOR-ART-AND-ALTERNATIVES.md) — measured with a different tool
  on a different llama.cpp commit. Three independent counting mechanisms (lldb on macOS,
  gdb on Linux, the in-tree counter) now agree to the hit across two platforms and two
  commits. Committed evidence:
  https://github.com/tomyimkc/polygraph/blob/main/results/upstream/pr26076/dispatch-summary-linux-validation.txt
- CUDA/Finding-4 live arm: still DEFERRED — a GPU compute process auto-starts at boot on
  that box, and the run gate correctly refuses to contend for the device.

## Process requirements (llama.cpp side — verified verbatim 2026-08-06)

- CONTRIBUTING.md: features must begin with an ISSUE, not a PR; AI-written posts prohibited
  (rule 5); AI-generated CODE allowed with: explicit disclosure of HOW AI was used,
  comprehensive manual review, ability to explain every line without AI.
- AGENTS.md prohibited list: AI-written PR descriptions, COMMIT MESSAGES, reviewer
  responses; automated commits or PR submissions. → The human must rewrite the two commit
  messages in their own words before pushing anywhere, and must submit personally.
- PR template REQUIRED section (do not delete):
  `## Requirements` containing (a) "I have read and agree with the contributing guidelines"
  and (b) `AI usage disclosure: YES/NO`. Disclosure must be YES, human-worded, and accurate
  about the code being AI-assisted.
- New contributors: limit 1 open PR; no trivial fixes.
- Issue-first: post the feature issue, let interest accumulate; open the PR only after
  engagement (or a maintainer invitation).

## Link preconditions

- FINDING-4 link: live now.
- dispatch-summary-crossvalidation.txt: must be pushed to polygraph main before any post
  links it (status at writing: being pushed).
