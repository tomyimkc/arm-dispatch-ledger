# Line-by-line walkthrough: kleidiai dispatch counters + opt-in summary

Study material only. Not for posting. Written so a human contributor can defend
every line of `proto-kleidiai-dispatch-summary` to a llama.cpp maintainer without
further AI help, per llama.cpp's `AGENTS.md` requirement.

Source series: `/tmp/kleidiai-dispatch-summary-series.patch` (2 commits — counters
+ opt-in summary + docs, and a fail-closed test). Files referenced below live in
`/tmp/llama-26076-pr/` on the `proto-kleidiai-dispatch-summary` branch.

---

## 1. Orientation (10 lines)

KleidiAI is Arm's library of hand-tuned micro-kernels for matmul-shaped ops on
Arm CPUs (dotprod / I8MM / SVE / SME / SME2). llama.cpp integrates it as a CPU
**extra buffer type** (`ggml_backend_cpu_kleidiai_buffer_type`,
`kleidiai.cpp:1971`) — a second CPU buffer type, alongside the default one, that
a weight tensor can be allocated into. When a weight lands in that buffer type,
`alloc_buffer` (`kleidiai.cpp:1812`) and `set_tensor` (`kleidiai.cpp:1795`)
repack it into KleidiAI's blocked/interleaved layout at load time, once, not
per inference call. At graph-build time, `extra_buffer_type::supports_op`
(`kleidiai.cpp:1890`) and `get_tensor_traits` (`kleidiai.cpp:1933`) decide
whether a given `MUL_MAT`/`GET_ROWS` op on that weight can be accelerated.
Compute then runs through exactly one of three paths on `class tensor_traits`
(`kleidiai.cpp:733`): `compute_forward_f32` (plain F32 weights), a fp16 path
(`compute_forward_fp16`, mixed F16 weight / F32 activation), and a quantized
path (`compute_forward_qx`, Q4_0/Q8_0 weights, itself branching gemv vs gemm by
batch size). Which concrete micro-kernel a path uses is decided once at
process start in `init_kleidiai_context()` (`kleidiai.cpp:378`), which detects
CPU features and calls `ggml_kleidiai_select_kernels_q4_0/_q8_0/_f32`, storing
the result in the global `ctx.kernels_q4` / `ctx.kernels_q8` / `ctx.kernels_f32`
(struct at `kleidiai.cpp:61`). The only three places a KleidiAI micro-kernel
function pointer, `kernel->run_kernel_ex` (declared `kernels.h:41`), is
actually invoked and does real work are: `kleidiai.cpp:1000` (f32 path),
`kleidiai.cpp:1162` (fp16 path), and `kleidiai.cpp:1453` inside the `run_chunk`
lambda (qx path). Everything upstream of those three calls — feature
detection, kernel selection, buffer-type routing, LHS/RHS packing, thread/chunk
partitioning — decides *whether and how* a kernel will run; only these three
call sites are where a kernel *runs*. That is exactly where this patch hooks in.

---

## 2. Hunk-by-hunk annotation

### Hunk 2.1 — `docs/build.md`, new paragraph after the `GGML_KLEIDIAI_SME` note

```diff
+Note that the buffer-size line and the kernel-selection log report what was
+selected, not what ran. To see how many KleidiAI micro-kernels actually
+executed, set GGML_KLEIDIAI_SUMMARY=1 and run with `--verbose`; a
+per-kernel-family execution summary is printed at process exit:
+```
+kleidiai: dispatch summary: selected q4=SME2 q8=SME2 f32=SME2
+kleidiai: dispatch summary: executed qx_gemv/SME2=3069
+kleidiai: dispatch summary: executed total=5952
+```
+An `executed total=0` under a non-`none` selected line means the selected
+kernels never ran ...
```

What it does: adds one paragraph to the existing KleidiAI doc section,
immediately after the `GGML_KLEIDIAI_SME` bullet list and before the
"higher priority backends" paragraph (`docs/build.md:631`). It documents the
new env var, the `--verbose` requirement, and gives a literal sample of the
three log-line shapes the code below actually emits.

Why it is there: the existing doc already tells the reader to check for the
`CPU_KLEIDIAI model buffer size = ...` line and the "primary ... kernel
feature ..." init log as proof KleidiAI is "being used" (`docs/build.md:619-625`).
Both are *selection* signals, not *execution* signals (see the test's rationale
in §3). Leaving that claim undisputed in the doc while adding an execution
counter elsewhere in the code would be actively misleading — a reader could
reasonably conclude the existing log lines already prove kernels ran. This
hunk closes that gap in the same document, right next to the claim it
qualifies.

What would break if removed: nothing at compile or run time — this is prose
only. But the counters and summary hook would ship with no discoverability
path; a user hitting the "acceleration claimed, nothing accelerated" failure
mode this feature exists to catch (see the test file's comment,
`test-kleidiai-dispatch.cpp:4-13`, citing ggml-org/llama.cpp #26547 / #26630)
would have no documented way to find `GGML_KLEIDIAI_SUMMARY`.

---

### Hunk 2.2 — `kleidiai.cpp`, new block after `cpu_feature_to_string` (`kleidiai.cpp:95-170`)

This is the largest hunk; it introduces the whole counting subsystem. Walking
it top to bottom:

```cpp
enum kleidiai_dispatch_path {
    KLEIDIAI_DISPATCH_PATH_F32 = 0,
    KLEIDIAI_DISPATCH_PATH_FP16,
    KLEIDIAI_DISPATCH_PATH_QX_GEMM,
    KLEIDIAI_DISPATCH_PATH_QX_GEMV,
    KLEIDIAI_DISPATCH_PATH_COUNT,
};
```
(`kleidiai.cpp:101-107`) One enumerator per compute path in `tensor_traits`
(§1) — `compute_forward_f32`, `compute_forward_fp16`, and the qx path split
into its two shapes (gemm/prefill vs gemv/decode, matching the `is_gemv` split
already present at `kleidiai.cpp:1186` and `:1034`). `KLEIDIAI_DISPATCH_PATH_COUNT`
is the usual "number of enumerators" sentinel used to size arrays, not a real path.

```cpp
static const int KLEIDIAI_DISPATCH_FAMILY_COUNT = 6;
static const char * kleidiai_dispatch_path_names[...] = { "f32", "fp16", "qx_gemm", "qx_gemv" };
static const char * kleidiai_dispatch_family_names[...] = { "SME2", "SME", "SVE", "I8MM", "DOTPROD", "OTHER" };
```
(`kleidiai.cpp:109-117`) Six kernel families: the five named `cpu_feature`
values (`kernels.h:9-16`) plus one catch-all bucket ("OTHER", index 5) for
`CPU_FEATURE_NONE`/anything unrecognized. The path/family name tables exist
purely so the printer (below) can emit human-readable strings instead of raw
enum ints — they are indexed by the enum values above, so their order must
track the enum order exactly (a maintainer should check this by eye: the
`OTHER` string sits at index 5, matching `kleidiai_dispatch_family_index`'s
fallthrough `return 5`, §below).

```cpp
static std::atomic<uint64_t> kleidiai_dispatch_counts[KLEIDIAI_DISPATCH_PATH_COUNT][KLEIDIAI_DISPATCH_FAMILY_COUNT] = {};
```
(`kleidiai.cpp:119`) A 4x6 grid of atomic counters — **why keyed path x
family**: a single flat total would answer "did anything run" but not "which
of the three call sites ran, using which kernel family" — the two questions
this feature exists to answer (contrast the init-time log, which only reports
the single globally-selected family per op type, not what actually executed
per call site). Keying by both dimensions lets `kleidiai: dispatch summary:
executed qx_gemv/SME2=3069` distinguish "the gemv path ran on SME2" from "the
gemm path ran on I8MM fallback," which a single counter cannot.
`std::atomic` because `compute_forward_*` runs on every thread in the CPU
thread pool concurrently — this is genuinely a shared mutable counter under
concurrent writers, so a plain `uint64_t` would be a data race (UB, and in
practice lost updates under `-fsanitize=thread` or just contended cache lines).

```cpp
static int kleidiai_dispatch_family_index(cpu_feature f) {
    // Mirrors cpu_feature_to_string's precedence.
    if ((f & CPU_FEATURE_SME2) == CPU_FEATURE_SME2) { return 0; }
    else if ((f & CPU_FEATURE_SME) == CPU_FEATURE_SME) { return 1; }
    else if ((f & CPU_FEATURE_SVE) == CPU_FEATURE_SVE) { return 2; }
    else if ((f & CPU_FEATURE_I8MM) == CPU_FEATURE_I8MM) { return 3; }
    else if ((f & CPU_FEATURE_DOTPROD) == CPU_FEATURE_DOTPROD) { return 4; }
    return 5;
}
```
(`kleidiai.cpp:121-135`) **Why it mirrors `cpu_feature_to_string`'s precedence
rather than inventing a new mapping**: `cpu_feature` (`kernels.h:9-16`) is a
bitmask (`DOTPROD=1, I8MM=2, SVE=4, SME=8, SME2=16`), and a real CPU's
`required_cpu` value can have several bits set at once (e.g. a CPU with SME2
support also reports DOTPROD and I8MM, since SME2 implies them in practice).
`cpu_feature_to_string` (`kleidiai.cpp:75-93`, unchanged by this patch) already
encodes the project's answer to "which single label do we print for a
multi-bit mask" — most-capable-first: SME2 > SME > SVE > I8MM > DOTPROD >
unknown. If this new function picked a different precedence (say, checked
DOTPROD first), the per-family counter breakdown in the summary could disagree
with the family name already printed at init by `cpu_feature_to_string`
(`kleidiai.cpp:481` etc.) for the *same* `required_cpu` value on the *same*
kernel struct — two different "which family is this kernel" answers coexisting
in the same log output would be confusing at best, actively contradictory at
worst. Mirroring the existing precedence keeps one project-wide answer to that
question. The one intentional difference: `cpu_feature_to_string` returns a
string ("UNKNOWN") for the no-match case; this function returns an index (5,
mapping to "OTHER" in the family-name table) because it needs an array index,
not a string, but the fallback semantics are the same "none of the known
single-feature bits matched" case.

```cpp
static inline void kleidiai_dispatch_count_add(kleidiai_dispatch_path path, cpu_feature f) {
    kleidiai_dispatch_counts[path][kleidiai_dispatch_family_index(f)].fetch_add(1, std::memory_order_relaxed);
}
```
(`kleidiai.cpp:137-139`) The single increment primitive every call site below
uses. **Why `std::memory_order_relaxed` is sufficient**: relaxed only
guarantees the increment itself is atomic (no lost updates, no torn reads);
it does *not* guarantee any ordering relative to other memory operations on
other threads. That's fine here because these counters are pure statistics —
nothing downstream ever makes a correctness decision conditioned on "counter A
was incremented before counter B" or uses the counter value to publish/consume
other data (no producer-consumer handoff through this atomic). The values are
only ever read in two places, both single-threaded and both far removed from
the hot compute loop: `kleidiai_print_dispatch_summary` (called from `atexit`,
long after all worker threads have joined) and the test's
`ggml_backend_cpu_kleidiai_dispatch_total()` (called between
`ggml_backend_graph_compute` calls, i.e. also outside the threaded region — see
§3). Neither reader needs to observe *when* relative to other writes an
increment happened, only the final summed value. A stronger order
(`acq_rel`/`seq_cst`) would add a real cost here: this increment sits on the
hot path, once per micro-kernel invocation across every thread in the pool, so
a fence-carrying RMW would add cross-core synchronization overhead for a
property (ordering) nothing here needs.

```cpp
static void kleidiai_print_dispatch_summary(void) {
    GGML_LOG_INFO("kleidiai: dispatch summary: selected q4=%s q8=%s f32=%s\n",
                  ctx.kernels_q4  ? cpu_feature_to_string(ctx.kernels_q4->required_cpu)  : "none",
                  ...);
    uint64_t total = 0;
    for (int p = 0; p < KLEIDIAI_DISPATCH_PATH_COUNT; ++p) {
        for (int f = 0; f < KLEIDIAI_DISPATCH_FAMILY_COUNT; ++f) {
            const uint64_t n = kleidiai_dispatch_counts[p][f].load(std::memory_order_relaxed);
            if (n > 0) { GGML_LOG_INFO("kleidiai: dispatch summary: executed %s/%s=%llu\n", ...); }
            total += n;
        }
    }
    GGML_LOG_INFO("kleidiai: dispatch summary: executed total=%llu\n", (unsigned long long) total);
}
```
(`kleidiai.cpp:141-160`) The printer. First line reports what was *selected*
at init (reads `ctx.kernels_q4/q8/f32` directly, the same globals the existing
init-time log at `kleidiai.cpp:481` etc. already reads — this is deliberately
the same "selected" vocabulary, not a new one). Then it walks the 4x6 grid and
prints one line per non-zero `(path, family)` cell — zero cells are skipped so
the summary output length is proportional to how many distinct
path/family combinations actually fired, not a fixed 24 lines every run — plus
a grand total line always printed (even if 0, so "nothing ran" is a visible
`executed total=0` rather than silence). `%llu` with an explicit
`(unsigned long long)` cast, not `%lu` — `uint64_t` is `unsigned long` on some
ABIs and `unsigned long long` on others, and `%llu` + the cast is the
portable pairing across platforms without relying on `PRIu64` inside a
`GGML_LOG_INFO` call (which is a `printf`-style varargs macro, so the format
string and args must agree exactly on every target, not just the one being
compiled on right now).

```cpp
uint64_t ggml_backend_cpu_kleidiai_dispatch_total(void) {
    uint64_t total = 0;
    for (...) for (...) total += kleidiai_dispatch_counts[p][f].load(std::memory_order_relaxed);
    return total;
}
```
(`kleidiai.cpp:162-170`) The public accessor — same summation as the printer's
running total, factored out so the test (§3) can read a live total without
triggering the full summary print or depending on `atexit`/env-var state at
all. **Why declared in `kleidiai.h`, not a public header**: this returns an
internal implementation counter with no defined semantics for a library
consumer beyond "diagnostics" — it is not part of ggml's or llama.cpp's public
API contract (nothing in `include/` declares it), so exposing it there would
imply a stability promise the maintainers haven't made. `kleidiai.h`
(`kleidiai.h:15,19`) already declares exactly one other function the same way
— `ggml_backend_cpu_buffer_type()` itself — establishing the precedent that
this header is for "cross-TU-but-still-internal" declarations shared between
`kleidiai.cpp` and its test, not the project's public surface.

**What would break if this whole hunk were removed**: nothing else in the
patch would compile — `kleidiai_dispatch_count_add` is called from all three
`compute_forward_*` methods (hunks 2.4-2.6), `kleidiai_print_dispatch_summary`
is registered by the `atexit` call in hunk 2.3, and
`ggml_backend_cpu_kleidiai_dispatch_total` is declared in the header (hunk 2.7)
and called by the test (hunk 3). This hunk is the shared foundation all the
others build on.

---

### Hunk 2.3 — `kleidiai.cpp`, inside `init_kleidiai_context()` (`kleidiai.cpp:389`, `:415-421`)

```diff
+        const char *env_summary     = getenv("GGML_KLEIDIAI_SUMMARY");
```
Added alongside the three existing `getenv` calls for `GGML_KLEIDIAI_SME`,
`GGML_TOTAL_THREADS`, `GGML_KLEIDIAI_CHUNK_MULTIPLIER` (`kleidiai.cpp:386-389`)
— same pattern, same place, no special-casing.

```diff
+        if (env_summary) {
+            bool ok = false;
+            int v = parse_uint_env(env_summary, "GGML_KLEIDIAI_SUMMARY", &ok);
+            if (ok && v > 0) {
+                atexit(kleidiai_print_dispatch_summary);
+            }
+        }
```
(`kleidiai.cpp:415-421`) Placed after the `GGML_KLEIDIAI_CHUNK_MULTIPLIER`
block and before the SME-cores block, inside the
`if (!initialized) { initialized = true; ... }` guard (`kleidiai.cpp:382-383`)
that makes `init_kleidiai_context()` idempotent under
`ggml_critical_section_start/end` (`kleidiai.cpp:379,515`).

**Why `parse_uint_env` is reused rather than a bespoke parser**: it already
exists (`kleidiai.cpp:360-376`) and already implements exactly the semantics
needed here — parse a base-10 integer with `strtol`, reject non-numeric or
trailing-garbage input with a `GGML_LOG_WARN`, reject negative or
overflowing values, and report success/failure through an out-parameter. Every
other env var in this function (`GGML_KLEIDIAI_SME`, `GGML_TOTAL_THREADS`,
`GGML_KLEIDIAI_CHUNK_MULTIPLIER`) already goes through it. Writing a second,
slightly different parser for `GGML_KLEIDIAI_SUMMARY` would (a) duplicate ~15
lines for no behavioral gain, and (b) risk a subtly different error message or
edge-case behavior (e.g. what happens on `GGML_KLEIDIAI_SUMMARY=""`) that a
reader would have to notice and reconcile against the other three. Reuse keeps
"how do KleidiAI env vars get parsed" a single answer.

**Why `atexit`, registered once inside this initialized-guard, matching the
existing env-var block, rather than a destructor**: there is no object whose
lifetime naturally ends at "process exit, after the summary should print" —
the counters are file-scope `static` atomics with static storage duration and
no owning object; there is nothing to attach a destructor *to*.
`init_kleidiai_context()` runs lazily on first use (called from
`ggml_backend_cpu_kleidiai_buffer_type()`, `kleidiai.cpp:1986`) and is already
idempotent (the `initialized` guard exists precisely so one-time setup work —
parsing env vars, selecting kernels, logging the selection — runs exactly
once no matter how many times the buffer-type accessor is called). Registering
the `atexit` handler inside that same guard, in the same function, next to the
other env-var handling, means: it registers exactly once (an `atexit`
registered on every call would queue duplicate handlers — every prior env var
in this function is parsed exactly once for the same reason), it happens at
the same natural "one-time init" point as everything else here, and it needs
no new lifetime machinery. A destructor would require inventing an object
whose sole purpose is to own that destructor, is riskier at static-init/exit
order (a global object's destructor runs at *its own* static-storage-duration
teardown point, which is less predictable relative to other globals than
`atexit`'s LIFO-relative-to-registration-time ordering — see the review-risk
discussion in §4), and has no precedent in this file, whereas `atexit` for
opt-in exit-time reporting has a direct precedent in-tree
(`ggml/src/ggml-et/ggml-et.cpp:185`, `std::atexit(ggml_et_driver_cleanup)`,
registered conditionally for opt-in profiling — same shape: env-var-gated,
registered once, reads accumulated state, logs it).

**What would break if removed**: `GGML_KLEIDIAI_SUMMARY=1` would have no
effect — the counters (hunk 2.2) would still increment (counting is
unconditional, see next paragraph), but `kleidiai_print_dispatch_summary`
would never run, so the feature would be silently inert. Note this hunk does
*not* gate the counting itself, only the printing — `kleidiai_dispatch_count_add`
calls exist unconditionally in the three compute paths regardless of whether
`GGML_KLEIDIAI_SUMMARY` was ever set. **Why counting is always on while
printing is opt-in** (pre-empting "why not gate everything" and "why not
always print"): counting is a single relaxed atomic increment per micro-kernel
invocation — cheap enough that gating it behind a runtime branch would add a
branch-predictor-visible check on the hot path for a saving that is smaller
than the branch itself, and it would mean the *unset* (default) behavior and
the *set* behavior of `GGML_KLEIDIAI_SUMMARY` differ in more than "does it
print" — they'd differ in whether the counters even exist as a live signal,
which is exactly the thing the fail-closed test in §3 depends on being true
unconditionally (the test never sets `GGML_KLEIDIAI_SUMMARY` at all — it reads
`ggml_backend_cpu_kleidiai_dispatch_total()` directly). Conversely, always
printing would mean every `llama-cli` invocation gets multi-line stderr output
whether or not the user asked for it — the summary is a diagnostic aimed at
"why isn't KleidiAI accelerating my model," not something every run needs, so
the print itself is the appropriate thing to gate, not the (cheap,
always-useful-for-tests) counting underneath it.

---

### Hunk 2.4 — `kleidiai.cpp`, inside `compute_forward_f32` (`kleidiai.cpp:1009`)

```diff
                                           sizeof(float),
                                           -FLT_MAX,
                                           FLT_MAX);
+
+                    kleidiai_dispatch_count_add(KLEIDIAI_DISPATCH_PATH_F32, kernels->required_cpu);
                }
```
Sits immediately after the `kernel->run_kernel_ex(...)` call at
`kleidiai.cpp:1000-1007`, still inside the `if (n_to_process > 0)` block
(`kleidiai.cpp:991`) and the `while ((size_t) current_col < n)` chunk loop
(`kleidiai.cpp:987`). `kernels` here is the local pulled from
`kleidiai_primary_kernel_f32()` at `kleidiai.cpp:895` — the single kernel
struct this whole function uses (the f32 path has no hybrid/fallback chain,
unlike qx).

**Why the count-add line sits exactly after the `run_kernel_ex` call**: one
call to `run_kernel_ex` is one micro-kernel invocation on one chunk of
columns, by definition of what this function's chunk loop does — the loop
claims a chunk (`ggml_threadpool_chunk_add`, `kleidiai.cpp:986,1012`), and if
the chunk is non-empty, runs the kernel on exactly that chunk once. Placing
the increment right after the call (rather than, say, once per thread outside
the loop) means "one invocation = one count" holds exactly — the counter
value after a run directly answers "how many kernel invocations happened,"
which is what the test in §3 asserts is `> 0`. Placing it before the call
would count invocations that might not have happened yet if `run_kernel_ex`
itself somehow didn't reach the point of doing work (it doesn't have an early
return in this signature, but "count after, not before" is also just the more
defensible convention for "did the thing happen" counters in general).

**What would break if removed**: the f32 path (plain-F32-weight matmuls
through the KleidiAI buffer type) would never contribute to
`ggml_backend_cpu_kleidiai_dispatch_total()` or the summary's
`f32/<family>=N` lines — an f32-weight model accelerated correctly by
KleidiAI would report `executed total=0` for its f32 traffic, which is exactly
the false-negative failure mode this whole feature exists to prevent, just
inverted (false "it's not running" instead of false "it is running").

---

### Hunk 2.5 — `kleidiai.cpp`, inside `compute_forward_fp16` (`kleidiai.cpp:1164`)

```diff
                    kernel->run_kernel_ex(m, n_to_process, k, 0, lhs_ptr, rhs_ptr, dst_ptr, dst_stride, sizeof(float), -FLT_MAX, FLT_MAX);
+
+                    kleidiai_dispatch_count_add(KLEIDIAI_DISPATCH_PATH_FP16, kernels->required_cpu);
                }
```
Same shape as 2.4: immediately after the sole `run_kernel_ex` call in this
function (`kleidiai.cpp:1162`), inside `if (ith < num_threads_n)`
(`kleidiai.cpp:1146`), inside the per-batch loop (`kleidiai.cpp:1081`).
`kernels` is from `ggml_kleidiai_select_kernels(ctx.features, dst)`
(`kleidiai.cpp:1029`) — note this path re-selects per-op via the generic
selector (unlike the f32 path's fixed `kleidiai_primary_kernel_f32()`),
because fp16 weights can be `MUL_MAT`'d against gemv- or gemm-shaped
activations and `kernel`/`lhs_info` are chosen via `is_gemv` right below it
(`kleidiai.cpp:1034-1036`) — but there is still exactly one `kernels` value in
play for this call, so one count-add site suffices (contrast qx below, which
has two paths precisely because it can run two *different* kernel structs
concurrently under hybrid dispatch).

**What would break if removed**: F16-weight matmuls (the "another CPU backend
may still take it" fallback path referenced in the comment at
`kleidiai.cpp:1938-1941`, when KleidiAI *does* claim an F16 tensor via
`get_tensor_traits`) would never register in the summary — same
false-"not running" failure mode as 2.4, scoped to the fp16 path.

---

### Hunk 2.6 — `kleidiai.cpp`, inside `compute_forward_qx`'s `run_chunk` lambda (`kleidiai.cpp:1462-1463`)

```diff
             slot.kernel->run_kernel_ex(m, cols, k, slot.rhs_bl,
                                        lhs_ptr,
                                        rhs_ptr,
                                        dst_ptr,
                                        dst_stride,
                                        sizeof(float),
                                        -FLT_MAX,
                                        FLT_MAX);
+
+            kleidiai_dispatch_count_add(is_gemv ? KLEIDIAI_DISPATCH_PATH_QX_GEMV : KLEIDIAI_DISPATCH_PATH_QX_GEMM,
+                                        slot.kernels->required_cpu);
         };
```
This is the most involved of the three sites and the one most likely to draw
maintainer questions, because `compute_forward_qx` supports **hybrid
dispatch**: up to `GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2` (`kleidiai.cpp:56`)
kernel "slots" can be active *simultaneously*, each bound to a different
thread range, when the CPU has an SME-family kernel available but not enough
threads are worth dedicating to it (`kleidiai_collect_kernel_chain`,
`kleidiai.cpp:685-690`, builds a primary+fallback chain; the hybrid-vs-single
decision is `use_hybrid`/`hybrid_enabled`, `kleidiai.cpp:1299-1310`; per-slot
thread ranges are assigned at `kleidiai.cpp:1341-1387`).

Walking why each specific detail is the way it is:

- **`is_gemv ? QX_GEMV : QX_GEMM`**: `is_gemv` is computed once, near the top
  of `compute_forward_qx`, as `src1->ne[1] == 1` (`kleidiai.cpp:1186`) — batch
  size 1 (decode-shaped: one new token, one activation row) uses the `gemv`
  kernel variant; anything wider (prefill-shaped) uses `gemm`. This mirrors
  the kernel-struct selection immediately below it in the same function
  (`kinfo = is_gemv ? &kernels->gemv : &kernels->gemm`, `kleidiai.cpp:1235`) —
  the count-add uses the same boolean the function already uses to pick which
  kernel variant runs, so the label always matches which kernel actually
  executed for this call. This is also why the test drives both batch size 1
  and batch size 8 (§3) — each `is_gemv` branch is otherwise untested by the
  other batch size.

- **`slot.kernels->required_cpu`, not `ctx.kernels_q4->required_cpu`**: this
  is the detail that most needs explaining, because it looks at first glance
  like it should just reuse the "primary" family the way hunks 2.4/2.5 use
  their single `kernels` local. It cannot, because under `hybrid_enabled`
  (`kleidiai.cpp:1310`) two different `runtime_slot`s can be live at once —
  e.g. one slot bound to `ctx.kernels_q4` (SME2, `is_sme_family`,
  `kleidiai.cpp:1284-1289`) restricted to `ctx.sme_thread_cap` threads
  (`kleidiai.cpp:1298,1342-1346`), and a second slot bound to the fallback
  kernel (non-SME family, `kleidiai.cpp:1291-1296`) taking the remaining
  threads (`kleidiai.cpp:1359-1369`). `run_chunk` (`kleidiai.cpp:1445`) takes
  `runtime_slot & slot` as a parameter and is invoked once per chunk from
  inside the per-thread chunk loop (`kleidiai.cpp:1517-1533`), where `slot =
  runtime[local_slot]` (`kleidiai.cpp:1517`) — `local_slot` is *this thread's*
  assigned slot, resolved from `ith_total` against each slot's
  `[thread_begin, thread_end)` range (`kleidiai.cpp:1389-1400`). So a thread
  running in the fallback (non-SME) slot's range executes `slot.kernel`
  belonging to the fallback kernel struct, and `slot.kernels->required_cpu`
  correctly reports that fallback family — while `ctx.kernels_q4->required_cpu`
  (what the init-time "primary q4 kernel feature" log printed, and what the
  summary's own `selected q4=...` line reports) would still say SME2, because
  that global never changes after init. Under hybrid dispatch these two
  answers ("what was selected as primary" vs "what this specific chunk
  actually ran on") are legitimately different, and the whole point of this
  patch is to report the second one. Using `ctx.kernels_q4->...` here would
  silently misattribute every fallback-slot chunk to the primary family,
  making the per-family breakdown wrong exactly in the hybrid case — the one
  case where the breakdown is most useful.

- **Why the increment sits inside the `run_chunk` lambda rather than at each
  of its call sites**: `run_chunk` has exactly one call site in this function
  (`kleidiai.cpp:1529`, inside the `while (current_chunk < nchunk)` loop,
  itself inside the per-batch loop, `kleidiai.cpp:1466`) — the lambda exists
  to be that per-chunk unit of work, dynamically claimed by threads via
  `ggml_threadpool_chunk_add(params->threadpool, 1)` (`kleidiai.cpp:1532`).
  Putting the count-add inside the lambda, right after its `run_kernel_ex`
  call, keeps the same "one invocation = one count" invariant as 2.4/2.5 —
  each call to `run_chunk` is one call to `run_kernel_ex` on one chunk, so one
  increment — without needing to duplicate the increment at multiple call
  sites (there's only one, but the lambda is also the natural place because
  it's the scope that has both `slot` and `is_gemv` in hand together).

**What would break if removed**: the qx path — the one carrying essentially
all real quantized-weight (Q4_0/Q8_0) inference traffic, i.e. the common case
for a quantized GGUF model — would never register any counts at all. Given
this is the dominant path in practice, removing just this hunk while keeping
2.4/2.5 would leave the summary reporting f32/fp16 traffic (comparatively
rare) while showing `executed total=0` for essentially every normal quantized
inference run — the worst-case version of the false-negative failure mode,
on the most common configuration.

---

### Hunk 2.7 — `kleidiai.h`

```diff
+#include <stdint.h>
...
+// Total number of KleidiAI micro-kernel invocations since process start.
+// Always counted; intended for tests and diagnostics (see GGML_KLEIDIAI_SUMMARY).
+uint64_t ggml_backend_cpu_kleidiai_dispatch_total(void);
```
(`kleidiai.h:9,17-19`) `#include <stdint.h>` is added because this header
previously had no fixed-width integer usage (its only prior declaration,
`ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void)`, uses
a type from `ggml-alloc.h`); the new declaration returns `uint64_t`, so the
header must pull in the standard header that defines it rather than relying
on it being transitively included — this header is meant to be self-contained
(it's wrapped in its own `#pragma once` and `extern "C"` guard,
`kleidiai.h:5,11-13,21-23`, so it should compile standalone).

The declaration itself is a straight forward-declaration of the function
defined in hunk 2.2 (`kleidiai.cpp:162-170`), inside the same
`extern "C" { ... }` block (`kleidiai.h:11-23`) as the existing buffer-type
accessor, so it gets C linkage — required because the test (§3) declares it
`extern "C"` independently (not by including this header at all, since it's
not a public header — see §3) and the two declarations must agree on linkage
or the linker won't resolve the symbol the test expects.

**What would break if removed**: `kleidiai.cpp` would still compile
(`ggml_backend_cpu_kleidiai_dispatch_total`'s definition doesn't need a
forward declaration in the same TU it's defined in — C++ allows a function
to be called and defined in the same file with only the definition visible
below prior uses, and it's not called from earlier in `kleidiai.cpp` anyway),
but nothing outside `kleidiai.cpp` — specifically the test — would have any
declared way to reference it; the test's own local `extern "C"` declaration
(§3) would still let it link (a matching `extern "C"` declaration elsewhere in
the program is sufficient for the linker regardless of whether it came from a
shared header), but there would be no single source-of-truth declaration for
future callers to find, and the deliberate choice of "internal header, not
public header" would have no header to express it in.

---

### Hunk 2.8 — `tests/CMakeLists.txt`

```diff
+    if (GGML_CPU_KLEIDIAI AND NOT GGML_BACKEND_DL)
+        llama_build_and_test(test-kleidiai-dispatch.cpp)
+        target_link_libraries(test-kleidiai-dispatch PRIVATE ggml-cpu)
+    endif()
```
(`tests/CMakeLists.txt:199-202`) Inserted right after
`llama_build_and_test(test-llama-archs.cpp)` (`tests/CMakeLists.txt:197`) and
before the `MODEL_DIR` setup for the model-downloading tests further down —
i.e. grouped with the other lightweight, no-model-download tests, not the
model-dependent ones below it.

See §3 for the full "why `GGML_CPU_KLEIDIAI AND NOT GGML_BACKEND_DL`"
reasoning (it's really a test-file-level design decision expressed here as a
build guard) and "why `target_link_libraries(... PRIVATE ggml-cpu)`" (the
`llama_build_and_test` helper, `tests/CMakeLists.txt:12-16`, only links
`llama` and `llama-common` by default — it does not link `ggml-cpu` directly,
so a test that calls a `ggml-cpu`-internal symbol needs this explicit extra
line; no other test in this file currently needs a raw `ggml-cpu` symbol, which
is why the helper doesn't do this by default).

**What would break if removed**: the test file (hunk 3, a new file) would
simply never be built or run by any config — CMake only discovers a test
target through an explicit `llama_build_and_test`/`add_executable`
invocation, and this is the only one in the tree for
`test-kleidiai-dispatch.cpp`.

---

## 3. Test annotation (`tests/test-kleidiai-dispatch.cpp`)

**Why the weight is allocated via
`ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, ggml_backend_cpu_kleidiai_buffer_type())`**
(`test-kleidiai-dispatch.cpp:108`): this is the one call in the whole test that
decides whether KleidiAI is in the loop at all. `ggml_backend_cpu_kleidiai_buffer_type()`
returns the extra buffer type described in §1; allocating the weight tensor
*into* that buffer type (rather than the default CPU buffer type) is exactly
what makes `op->src[0]->buffer->buft == ggml_backend_cpu_kleidiai_buffer_type()`
true later at graph-build time, which is the precondition
`extra_buffer_type::supports_op` (`kleidiai.cpp:1893-1896`) and
`get_tensor_traits` (`kleidiai.cpp:1935`) both check before routing the op to
`tensor_traits::compute_forward` at all. A weight allocated into the ordinary
CPU buffer type would run through the CPU backend's own (non-KleidiAI)
`MUL_MAT` implementation instead — same numerical result, zero KleidiAI
dispatch counter movement, and the test would be testing nothing. Buffer-type
placement, not tensor *type*, is what routes compute through KleidiAI's
`tensor_traits` — this is the load-bearing mechanism the test exercises, and
it's also exactly the mechanism the referenced field defects
(ggml-org/llama.cpp #26547 / #26630, cited in the test's header comment,
`test-kleidiai-dispatch.cpp:6-7`) broke: a weight that *looked* like it should
get KleidiAI's buffer type (right quant type, right CPU) sometimes didn't,
because of buffer-type/backend priority interactions elsewhere (the "CUDA-host-
buffer priority variant" the comment mentions), and every existing signal
(build flag, init log) still claimed acceleration.

**Why decline-to-allocate is a SKIP, not a FAIL**
(`test-kleidiai-dispatch.cpp:109-115`): `ggml_backend_alloc_ctx_tensors_from_buft`
returns `nullptr` when the buffer type's `alloc_buffer` declines the request —
in KleidiAI's case, this happens when no kernel is compatible with this CPU's
detected features (`ctx.kernels_q4 == nullptr`, e.g. running on an x86 CI
runner or an Arm CPU with none of dotprod/i8mm/sve/sme). That is a legitimate,
expected outcome on unsupported hardware — it is not the failure this test is
built to catch. Treating it as `FAIL` would make the test red on every
non-Arm or old-Arm CI runner regardless of whether the actual contract (accept
implies execute) holds, which would either block CI everywhere the feature
isn't even applicable, or train reviewers to ignore this test's failures as
noise — both worse than a clean, visible `SKIP`. The comment at
`test-kleidiai-dispatch.cpp:110-111` is explicit about why this matters: "a
visible outcome, not a silent conflation" — silently treating decline as pass
would be the opposite failure mode (a green check that proves nothing),
exactly mirroring what the summary's `executed total=0` line (hunk 2.2) is
for: making an absence-of-execution outcome visible instead of ambiguous.

**Why batch 1 and batch 8** (`test-kleidiai-dispatch.cpp:124`,
`const int64_t batch_sizes[2] = { 1, 8 }`): batch size 1 forces `is_gemv` true
in `compute_forward_qx` (`src1->ne[1] == 1`, `kleidiai.cpp:1186`) — the
decode-shaped case, one token at a time, the dominant case during normal
autoregressive generation. Batch size 8 forces `is_gemv` false — the
gemm/prefill-shaped case, processing a chunk of tokens at once (prompt
processing, batched decoding). These are the only two branches of `is_gemv`
in the qx path, and they route to genuinely different kernel structs
(`kernels->gemv` vs `kernels->gemm`, `kleidiai.cpp:1235`) and different count
buckets (`QX_GEMV` vs `QX_GEMM`, hunk 2.6). Testing only one batch size would
leave the other branch — and its count-add call — completely unverified by
this test. 8 (rather than, say, 2) is simply comfortably inside gemm territory
without needing to reason about small-batch edge cases; the test isn't
asserting anything about *which* number counts as "gemm-shaped," only that
`is_gemv` comes out false for it.

**Why the CMake guard is `GGML_CPU_KLEIDIAI AND NOT GGML_BACKEND_DL`, not just
`GGML_CPU_KLEIDIAI`**: `GGML_CPU_KLEIDIAI` gates whether the KleidiAI backend
is compiled in at all (needed regardless — no point building this test
without it, the two `extern "C"` symbols wouldn't exist). `GGML_BACKEND_DL`
(`ggml/CMakeLists.txt:86`, "build backends as dynamic libraries") is a
*separate* axis: when it's on, `ggml-cpu` and friends are built as loadable
plugins (`.so`/`.dylib`/`.dll`) discovered and `dlopen`'d at runtime, not
linked at build time (`ggml/src/CMakeLists.txt:217-218,266-270`, backends get
`GGML_BACKEND_DL` compiled in and are registered via the dynamic-loading
path rather than direct linkage). The test calls
`ggml_backend_cpu_kleidiai_buffer_type()` and
`ggml_backend_cpu_kleidiai_dispatch_total()` as ordinary extern C functions,
resolved by the *linker* at build time (`target_link_libraries(...
PRIVATE ggml-cpu)`, hunk 2.8) — under `GGML_BACKEND_DL`, those symbols live
inside a plugin that isn't a link-time dependency of the test binary at all
(the whole point of `GGML_BACKEND_DL` is that the host program doesn't need
to link backend-specific symbols; it discovers them through the backend
registry at runtime instead). So under `GGML_BACKEND_DL`, the direct
`target_link_libraries(test-kleidiai-dispatch PRIVATE ggml-cpu)` line would
either fail to link (if `ggml-cpu` isn't built as a conventional static/object
library target at all in that mode) or, worse, silently link against a
KleidiAI implementation the rest of the running system isn't actually using
(since a `GGML_BACKEND_DL` deployment resolves the real backend by scanning
`GGML_BACKEND_DIR` at runtime, not via this build's own `ggml-cpu` target) —
either way, the test would not be testing the code path a `GGML_BACKEND_DL`
deployment actually exercises. Excluding it there is the honest choice, not a
missed opportunity to cover it.

**Why the accessor is declared locally `extern "C"` in the test, not via
`#include "kleidiai.h"`**: the two-line comment at
`test-kleidiai-dispatch.cpp:26-27` states this directly — "declared locally
here to keep the hook out of the public API." `kleidiai.h` lives under
`ggml/src/ggml-cpu/kleidiai/` (an internal source directory, not
`ggml/include/`), so it isn't on this test's normal include path and
including it would require adding a new include-directory dependency from
`tests/` into a CPU-backend-internal source directory — a layering violation
(tests depending on a specific backend's internal header layout) for the sake
of two forward declarations. Re-declaring the two functions locally with
matching `extern "C"` signatures (`test-kleidiai-dispatch.cpp:28-29`) achieves
the same linkage without that dependency, at the cost of the declarations
needing to be kept in sync by hand if the real signatures ever change — an
explicit, deliberate tradeoff the comment names, not an oversight.

---

## 4. Known review-risk points (honestly)

**"Why not gate counting too, the same way printing is gated?"** Already
argued in hunk 2.3's writeup, restated here as the risk framing: a maintainer
who is cost-conscious about the hot compute path may ask why every
`run_kernel_ex` call pays for an atomic increment even when
`GGML_KLEIDIAI_SUMMARY` is never set. The honest answer is a tradeoff, not a
proof: one `fetch_add(relaxed)` is small next to the cost of an actual
micro-kernel invocation (a `run_kernel_ex` call processes a chunk of a matmul
— thousands of FLOPs at minimum), so it is very likely amortized to
negligible, but this patch does not include a microbenchmark demonstrating
that on real hardware. Gating it would also mean the disabled (default)
build's behavior differs from the enabled build's behavior in more than "does
it print" — the counters the test depends on would no longer exist
unconditionally, coupling the test's correctness to an unrelated env var
being set during test runs, which the test currently does not do. If a
maintainer wants a benchmark before accepting the always-on cost, that is a
legitimate ask this patch does not yet answer with data.

**The `atexit`-in-a-library concern.** Registering process-exit hooks from
inside a library (as opposed to the top-level application) is a known sharp
edge in C/C++: `atexit` handlers run in LIFO order relative to *registration*
time, not relative to any particular teardown priority, and a handler that
touches another translation unit's state risks running after that state's own
static-storage-duration destructor has already fired (a static/global
destructor order problem, distinct from but related to the classic
"static init order fiasco"). Precedent in this tree softens but doesn't
eliminate the concern: `ggml/src/ggml-et/ggml-et.cpp:185` already does
`std::atexit(ggml_et_driver_cleanup)` for opt-in profiling, and
`ggml/src/ggml-zdnn/ggml-zdnn.cpp:618` does `atexit(ggml_zdnn_cleanup)`
unconditionally — so the pattern is accepted in this codebase, not a novel
risk this patch introduces. What this handler actually touches at exit time
is narrow: it reads the file-scope atomics (POD, static storage duration, no
constructor/destructor to race) and calls `GGML_LOG_INFO`, which goes through
`g_logger_state` (`ggml/src/ggml.c:284`) — a plain C struct (function pointer
+ `void*`, no C++ destructor) — to whatever callback is registered, typically
`common_log_default_callback` (`common/log.cpp:454`), which forwards to
`common_log_main()`'s singleton. That singleton is a function-local static,
so its own destructor is itself registered via `atexit` at first use — and
because `init_kleidiai_context()` runs lazily on first KleidiAI buffer-type
access (during model load, i.e. well after `llama-cli`'s own startup logging
has already run and already constructed that singleton), this patch's
`atexit(kleidiai_print_dispatch_summary)` registers *after* the logger
singleton's own atexit-registered destructor — meaning, by LIFO order, this
handler runs *before* the logger is torn down. That is a real argument for
safety in the `llama-cli`/typical-host case, but it is an argument about
*program structure* (logging happens before model load), not a guarantee the
type system or the code enforces — an embedding application that constructs
the logger lazily, only after loading a KleidiAI-backed model, or that tears
down logging explicitly before process exit some other way, could still hit
trouble. Worth stating plainly to a reviewer as "this is safe in the shipped
CLI's actual startup order, argued from that order, not proven independent of
it."

**`GGML_LOG` at exit-time callback validity, more generally.** Separate from
the ordering argument above: `ggml_log_internal` (`ggml/src/ggml.c:286-310`)
formats into a stack buffer and calls the currently-registered callback
directly — no heap allocation of new global state, no thread creation, no
dependency on anything that would be unsafe specifically because it's running
inside an `atexit` handler rather than "normal" code. If a host application
overrides the log callback via `ggml_log_set` (`ggml/src/ggml.c:7999-8001`)
with something that itself isn't atexit-safe (e.g. writes to a file handle
already closed by another exit hook, or posts to a UI thread that has already
shut down), that's a hazard, but it's a pre-existing hazard of any
`GGML_LOG_*` call made late in shutdown, not something this patch introduces
uniquely — every other `atexit`-registered logger in this tree
(`ggml-et`, `ggml-zdnn`) has the identical exposure.

**"The summary requires `--verbose` under `llama-cli` — is that documented
and is it the right default?"** Confirmed by tracing the actual log-level
plumbing: `GGML_LOG_INFO` calls (as opposed to the `common/log.h` `LOG_INF`
macro, which is a different call path used by `common/` code) route through
`common_log_default_callback` (`common/log.cpp:454-458`), which maps
`GGML_LOG_LEVEL_INFO` to verbosity `LOG_LEVEL_TRACE = 4`
(`common_get_verbosity`, `common/log.cpp:441-452`) and only logs when
`verbosity <= common_log_verbosity_thold`. The default threshold is
`LOG_DEFAULT_LLAMA = LOG_LEVEL_INFO = 3` (`common/log.h:27,32`), so a raw
`GGML_LOG_INFO` call is filtered out by default (`4 <= 3` is false) — it only
shows once `-v`/`--verbose`/`--log-verbose` sets the threshold to `INT_MAX`
(`common/arg.cpp:3773-3780`). This matches the doc hunk's claim exactly (hunk
2.1) and is not a guess: every existing KleidiAI init-time log (the "primary
q4 kernel feature ..." lines, `kleidiai.cpp:481` etc.) already has this same
property today, using the same `GGML_LOG_INFO` call, so the summary's
`--verbose` requirement is consistent with existing KleidiAI logging
behavior, not a new inconsistency this patch introduces. Whether `--verbose`
is the *right* default for this specific diagnostic (versus, say, always
showing at plain `GGML_LOG_WARN` when `executed total=0` contradicts a
non-`none` `selected` line) is a legitimate design question a maintainer could
raise — this patch chose the conservative, consistent-with-existing-code
answer, not a maximally-discoverable one.

---

## 5. Closing self-quiz

A maintainer might reasonably ask any of the following. Answers live in the
sections above — this list is deliberately answer-free so it's useful as a
defend-it-yourself check before review, not a script to read from.

1. What exactly does "selected" mean in the summary output, and where does
   that line's data come from?
2. What happens if `GGML_KLEIDIAI_SUMMARY=abc` (not a valid integer)?
3. Why can the hybrid qx path report a different kernel family in the
   execution summary than the family the init-time log printed for the same
   op type?
4. What does `executed total=0` under `selected=SME2` prove, and what does it
   *not* prove?
5. What is the runtime overhead of this patch when `GGML_KLEIDIAI_SUMMARY` is
   never set?
6. Why does the test not assert an exact dispatch count for either batch
   size?
7. Why is `kleidiai_dispatch_family_index`'s precedence order not independent
   of `cpu_feature_to_string`'s — what would go wrong if someone "simplified"
   it to check `DOTPROD` first?
8. Why is the accessor function declared in `kleidiai.h` instead of a header
   under `ggml/include/`?
9. Under what build configuration does `tests/test-kleidiai-dispatch.cpp` not
   get built at all, and why is that the correct call rather than a coverage
   gap to fix?
10. Why does the qx path's count-add site read `slot.kernels->required_cpu`
    while the f32 and fp16 paths each read a single, function-scoped
    `kernels->required_cpu`?
11. Why is the `atexit` call placed inside `init_kleidiai_context()`'s
    `initialized` guard specifically, rather than, say, at the top of
    `ggml_backend_cpu_kleidiai_buffer_type()`?
12. If a Q4_0 weight tensor is allocated with plain
    `ggml_backend_cpu_buffer_type()` instead of
    `ggml_backend_cpu_kleidiai_buffer_type()`, does `compute_forward_qx` ever
    run for it, and does the dispatch counter move?
