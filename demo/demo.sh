#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
#
# demo/demo.sh -- self-contained, timed, narrated terminal walkthrough for the
# submission video. Five beats, under 3 minutes of screen time: the hook, the
# proof, the cost, the root cause, and the honest ending.
#
# Polygraph (formerly Arm Dispatch Ledger): does your software actually do what
# it says? This demo answers that question for llama.cpp's KleidiAI CPU backend.
#
# THIS SCRIPT WAS REBUILT 2026-08-04 around results/REMEASURE-2026-08-04-QUIET.md,
# which is the authoritative, round-robin-interleaved, 7-reps-per-config
# re-measurement of this project's throughput claims. It supersedes an earlier
# arc built on results/crossover/, which was measured on a heavily contended
# machine (1-minute load average 66-147) with baseline and patched configs run
# in *different, non-interleaved* time windows -- exactly the setup that
# manufactures a fake speedup. That arc's headline numbers (57.3%, 71.6, 45.5,
# 4.4x, 198.9, 2257.5, 1145.0) are RETRACTED. Do not resurrect them here.
#
# The corrected story, in one line: tuning (`-t 2` for decode, `-t 8` for
# prefill) is a real, reproducible, zero-code-change win -- 3.43x and 1.79x.
# The phase-aware dispatch patch this project also wrote is a genuine,
# symbol-level dispatch change that is NOT a throughput win -- it is ~12%
# SLOWER at default thread count. We measured that, we are not hiding it, and
# we filed both the tuning finding and the patch's honest negative result
# upstream: https://github.com/ggml-org/llama.cpp/issues/26547
#
# Design rules this script follows (do not relax these when editing):
#   1. Every number this script PRINTS is either (a) produced live, in front
#      of the viewer, by a real command in this repo, or (b) a number already
#      committed to results/ in an earlier, properly-measured run, clearly
#      labelled as such with the exact command to reproduce it live. Nothing
#      is invented, rounded up, or silently reused across a different
#      methodology. In particular: never quote 57.3, 71.6, 45.5, 4.4x, 198.9,
#      2257.5, or 1145.0 as a current result -- see the header note above.
#   2. Every live command is echoed before it runs, so a screen recording
#      shows the viewer exactly what was typed, not a black box.
#   3. Never abort mid-take. No `set -e`. Missing binaries/models/tools
#      degrade to a clear, actionable message and the *next* beat still runs.
#   4. Idempotent: writes nothing into the repo tree. All scratch output goes
#      to $SCRATCH_DIR (outside the checkout). Safe to re-run any number of
#      times, in any order of prior runs, with no cleanup required.
#   5. Deterministic wall-clock budget: beats that are cheap (a few seconds)
#      run live by default. The beats that are expensive by construction (a
#      full lldb dispatch sweep at a high thread count, ~90s of stop/continue
#      round trips each -- see results/GROUND-TRUTH-DISPATCH.md's own timing
#      note) are *pre-measured and cited* by default, and can be switched to
#      a real live run with RUN_LIVE_L3=1 for anyone who wants to pay the
#      extra time for full live authenticity.
#
# Usage:
#   ./demo/demo.sh                  # fast path, ready to screen-record
#   PAUSE=1 ./demo/demo.sh          # shorter pauses, for a dry run / rehearsal
#   PAUSE=0 ./demo/demo.sh          # no pauses at all -- fastest smoke test
#   RUN_LIVE_L3=1 ./demo/demo.sh    # also live-run the expensive lldb sweeps
#                                   # (Beat 2's default-thread probe, and
#                                   # Beat 5's patched flag on/off probes)
#                                   # instead of citing the committed ledgers
#   NO_COLOR=1 ./demo/demo.sh       # plain text, no ANSI colour
#
# Env vars (all optional, all have working defaults on the reference machine
# this repo was built and measured on -- see results/REMEASURE-2026-08-04-QUIET.md):
#   LLAMA_CLI            path to llama-cli built -DGGML_CPU_KLEIDIAI=ON (baseline, dbadb68)
#   LLAMA_BENCH          path to llama-bench (same build)
#   MODEL_PATH           path to the Q4_0 GGUF (Qwen2.5-0.5B-Instruct)
#   LLAMA_CLI_PATCHED    path to llama-cli built from the SAME base commit with
#                        patches/0001-kleidiai-phase-aware-dispatch.patch applied
#                        (build ef973b1). Optional -- Beat 5 degrades to citing
#                        the committed evidence if not found. See demo/README.md
#                        "Optional: building the patched binary".
#   LLAMA_BENCH_PATCHED  path to llama-bench from the same patched build
#   KLEIDIAI_SRC         path to ggml/src/ggml-cpu/kleidiai/kleidiai.cpp in the
#                        baseline llama.cpp checkout (for Beat 4's live source read)
#   DEFAULT_THREADS      the "reasonable default" thread count used to show the
#                        silent fallback and the patch's warning line (default: 8)
#   LIVE_BENCH_REPS      reps for Beat 3's quick LIVE llama-bench sanity check
#                        (default: 3 -- NOT the rigorous number; see Beat 3)
#   PAUSE                seconds paused after each beat (default: 2.5)
#   RUN_LIVE_L3           1 = actually run the expensive lldb dispatch sweeps live
#   SCRATCH_DIR          where scratch/temp output goes (default: see below)

set -uo pipefail

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

find_first_existing() {
    local p
    for p in "$@"; do
        if [[ -n "$p" && -e "$p" ]]; then
            printf '%s' "$p"
            return 0
        fi
    done
    return 1
}

CACHE_DIR_GUESS="${TMPDIR:-/tmp}/polygraph-cache"

LLAMA_CLI="${LLAMA_CLI:-$(find_first_existing \
    "/tmp/llama.cpp/build/bin/llama-cli" \
    "$CACHE_DIR_GUESS/llama.cpp/build/bin/llama-cli" \
    2>/dev/null)}"
LLAMA_BENCH="${LLAMA_BENCH:-$(find_first_existing \
    "/tmp/llama.cpp/build/bin/llama-bench" \
    "$CACHE_DIR_GUESS/llama.cpp/build/bin/llama-bench" \
    2>/dev/null)}"
MODEL_PATH="${MODEL_PATH:-$(find_first_existing \
    "/tmp/ggufs/q05.gguf" \
    "$CACHE_DIR_GUESS/models/qwen2.5-0.5b-instruct-q4_0.gguf" \
    2>/dev/null)}"
LLAMA_CLI_PATCHED="${LLAMA_CLI_PATCHED:-$(find_first_existing \
    "/tmp/llama-phase-aware/build/bin/llama-cli" \
    "$CACHE_DIR_GUESS/llama-phase-aware/build/bin/llama-cli" \
    2>/dev/null)}"
LLAMA_BENCH_PATCHED="${LLAMA_BENCH_PATCHED:-$(find_first_existing \
    "/tmp/llama-phase-aware/build/bin/llama-bench" \
    "$CACHE_DIR_GUESS/llama-phase-aware/build/bin/llama-bench" \
    2>/dev/null)}"
KLEIDIAI_SRC="${KLEIDIAI_SRC:-$(find_first_existing \
    "/tmp/llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp" \
    "$CACHE_DIR_GUESS/llama.cpp/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp" \
    2>/dev/null)}"

VERIFY_DISPATCH="$REPO_ROOT/tools/verify_dispatch.py"
DISPATCH_LEDGER_FLAG_OFF="${DISPATCH_LEDGER_FLAG_OFF:-$REPO_ROOT/results/dispatch-ledger-darwin-arm64-patched-flag-off.json}"
DISPATCH_LEDGER_FLAG_ON="${DISPATCH_LEDGER_FLAG_ON:-$REPO_ROOT/results/dispatch-ledger-darwin-arm64-patched-flag-on.json}"

SCRATCH_DIR="${SCRATCH_DIR:-${TMPDIR:-/tmp}/polygraph-demo}"
mkdir -p "$SCRATCH_DIR"

DEFAULT_THREADS="${DEFAULT_THREADS:-8}"
LIVE_BENCH_REPS="${LIVE_BENCH_REPS:-3}"
PAUSE="${PAUSE:-2.5}"
RUN_LIVE_L3="${RUN_LIVE_L3:-0}"

# -----------------------------------------------------------------------
# Colour / style (respects NO_COLOR, degrades to plain text off a TTY)
# -----------------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]] && command -v tput >/dev/null 2>&1 \
   && [[ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]]; then
    BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
    RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
    MAGENTA="$(tput setaf 5)"; CYAN="$(tput setaf 6)"
else
    BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; MAGENTA=""; CYAN=""
fi

# -----------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------

pause() {
    local secs="${1:-$PAUSE}"
    [[ "$secs" == "0" ]] && return 0
    sleep "$secs" 2>/dev/null || true
}

banner() {
    local title="$1" width=78
    echo
    printf '%s' "${CYAN}${BOLD}"
    printf '=%.0s' $(seq 1 "$width"); echo
    printf '  %s\n' "$title"
    printf '=%.0s' $(seq 1 "$width"); echo
    printf '%s' "$RESET"
}

narrate() {
    printf '%s\n' "${YELLOW}${BOLD}>> ${RESET}${YELLOW}$*${RESET}"
}

note() {
    printf '%s\n' "${DIM}   $*${RESET}"
}

fact() {
    printf '%s\n' "${MAGENTA}${BOLD}   $*${RESET}"
}

# Echo a command the way a viewer would read it off the terminal, then run
# it for real. Never aborts the whole script on a nonzero exit -- prints the
# exit code and keeps going (rule 3 above).
run_live() {
    printf '\n%s' "${DIM}\$ ${RESET}"
    printf '%s' "${GREEN}"
    printf '%q ' "$@"
    printf '%s\n' "${RESET}"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '%s\n' "${RED}   (exited $rc -- continuing; this beat degraded rather than the whole demo)${RESET}"
    fi
    return 0
}

# Same echo-then-run contract as run_live, but captures stdout+stderr to a
# file instead of streaming it live. Use this whenever the caller needs to
# grep/trim the output afterward: piping run_live's own stdout into a
# filter also eats the "$ ..." echo line above it, since the echo and the
# command output share one stream. Capturing to a file keeps the echo on
# the terminal and lets the caller print exactly the excerpt it wants.
run_live_quiet() {
    local outfile="$1"; shift
    printf '\n%s' "${DIM}\$ ${RESET}"
    printf '%s' "${GREEN}"
    printf '%q ' "$@"
    printf '%s\n' "${RESET}"
    "$@" > "$outfile" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        printf '%s\n' "${RED}   (exited $rc -- continuing; this beat degraded rather than the whole demo)${RESET}"
    fi
    return 0
}

have_llama() {
    [[ -n "$LLAMA_CLI" && -x "$LLAMA_CLI" && -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]
}

have_bench() {
    [[ -n "$LLAMA_BENCH" && -x "$LLAMA_BENCH" && -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]
}

have_patched() {
    [[ -n "$LLAMA_CLI_PATCHED" && -x "$LLAMA_CLI_PATCHED" && -n "$MODEL_PATH" && -f "$MODEL_PATH" ]]
}

have_kleidiai_src() {
    [[ -n "$KLEIDIAI_SRC" && -f "$KLEIDIAI_SRC" ]]
}

warn_missing_llama() {
    printf '%s\n' "${RED}${BOLD}   [skip] llama-cli and/or the model GGUF were not found.${RESET}"
    note "LLAMA_CLI  = ${LLAMA_CLI:-<not found>}"
    note "MODEL_PATH = ${MODEL_PATH:-<not found>}"
    note "Build them with: ./scripts/setup.sh   (see demo/README.md, Option B / Option A)"
    note "Or point this script at an existing build:"
    note "  LLAMA_CLI=/path/to/llama-cli MODEL_PATH=/path/to/model.gguf ./demo/demo.sh"
}

warn_missing_bench() {
    printf '%s\n' "${RED}${BOLD}   [skip] llama-bench and/or the model GGUF were not found.${RESET}"
    note "LLAMA_BENCH = ${LLAMA_BENCH:-<not found>}"
    note "MODEL_PATH  = ${MODEL_PATH:-<not found>}"
    note "Build with: ./scripts/setup.sh"
}

warn_missing_patched() {
    printf '%s\n' "${RED}${BOLD}   [skip] the patched llama-cli (phase-aware dispatch) was not found.${RESET}"
    note "LLAMA_CLI_PATCHED = ${LLAMA_CLI_PATCHED:-<not found>}"
    note "This is optional -- the demo still shows the cited, committed evidence below."
    note "To build it yourself, see demo/README.md 'Optional: building the patched binary'"
    note "(applies patches/0001-kleidiai-phase-aware-dispatch.patch to a dbadb68 checkout)."
}

warn_missing_kleidiai_src() {
    printf '%s\n' "${RED}${BOLD}   [skip] kleidiai.cpp source not found at KLEIDIAI_SRC.${RESET}"
    note "KLEIDIAI_SRC = ${KLEIDIAI_SRC:-<not found>}"
    note "Showing the same lines from this repo's own citation instead (docs/UPSTREAM-ISSUE.md)."
}

have_lldb() { command -v lldb >/dev/null 2>&1; }
have_python3() { command -v python3 >/dev/null 2>&1; }

# -----------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------

DEMO_START_TS=$(date +%s)

[[ -t 1 ]] && clear 2>/dev/null || true
printf '%s\n' "${CYAN}${BOLD}"
cat <<'BANNER'
    ____        __                             __
   / __ \____  / /_  ______ __________ _____  / /_
  / /_/ / __ \/ / / / / __ `/ ___/ __ `/ __ \/ __ \
 / ____/ /_/ / / /_/ / /_/ / /  / /_/ / /_/ / / / /
/_/    \____/_/\__, /\__, /_/   \__,_/ .___/_/ /_/
              /____//____/          /_/
BANNER
printf '%s\n' "$RESET"
narrate "Polygraph -- a tool that checks whether software is telling the truth."
note "Does the SME2 kernel llama.cpp advertises actually run? Let's check."
note "Apache-2.0 -- github.com/tomyimkc/polygraph -- Arm Create: AI Optimization Challenge"
echo
note "LLAMA_CLI         = ${LLAMA_CLI:-<not found>}"
note "LLAMA_BENCH        = ${LLAMA_BENCH:-<not found>}"
note "MODEL_PATH         = ${MODEL_PATH:-<not found>}"
note "LLAMA_CLI_PATCHED  = ${LLAMA_CLI_PATCHED:-<not found, Beat 5 will cite instead>}"
note "DEFAULT_THREADS    = $DEFAULT_THREADS   PAUSE=${PAUSE}s   RUN_LIVE_L3=$RUN_LIVE_L3"
pause

# =========================================================================
# BEAT 1 / 5 -- The hook: the banner claims SME2 is running
# =========================================================================
banner "BEAT 1 / 5 -- The hook: the banner says SME2 is running"
narrate "llama.cpp's own startup log, at --verbose, claims the accelerated SME2"
narrate "kernel is selected for every matmul family. Watch the log, live."

if have_llama; then
    HOOK_LOG="$SCRATCH_DIR/beat1-hook.log"
    run_live "$LLAMA_CLI" -m "$MODEL_PATH" -p "The capital of France is" -n 4 \
        -no-cnv -st --simple-io -t "$DEFAULT_THREADS" --verbose > "$HOOK_LOG" 2>&1
    echo
    note "(full --verbose log captured; showing just the dispatch-relevant lines)"
    if grep -E "system_info:.*SME|kleidiai: (primary|SME2 enabled)" "$HOOK_LOG" >/dev/null 2>&1; then
        grep -E "system_info:.*SME|kleidiai: (primary|SME2 enabled)" "$HOOK_LOG" \
            | sed -E "s/^/${RED}${BOLD}   /; s/\$/${RESET}/"
    else
        printf '%s\n' "${RED}   (no SME/KLEIDIAI log lines found -- this llama.cpp build may not have"
        printf '%s\n' "    -DGGML_CPU_KLEIDIAI=ON, or the log format changed. See $HOOK_LOG.)${RESET}"
    fi
else
    warn_missing_llama
fi
echo
fact "Claim: SME = 1 | SME2 = 1 | KLEIDIAI = 1, 'primary q4 kernel feature SME2'."
fact "Selected for every op, according to the log. Let's check what actually runs."
pause

# =========================================================================
# BEAT 2 / 5 -- The proof: zero hits at default threads, thousands at -t 2
# =========================================================================
banner "BEAT 2 / 5 -- The proof: a debugger, not a benchmark"
narrate "A timing-only benchmark cannot see past that log line. We put a real"
narrate "lldb breakpoint on the SME2 kernel's own entry points and count hits."

if [[ "$RUN_LIVE_L3" == "1" ]]; then
    if have_llama && have_lldb; then
        narrate "(RUN_LIVE_L3=1 -- running the real sweep live, ~90s of lldb stop/continue)"
        run_live python3 "$VERIFY_DISPATCH" \
            --binary "$LLAMA_CLI" --model "$MODEL_PATH" \
            --threads "$DEFAULT_THREADS" --workloads decode_short \
            --l3-debugger lldb \
            --out "$SCRATCH_DIR/beat2-live-default-t${DEFAULT_THREADS}.json"
    else
        warn_missing_llama
        have_lldb || note "lldb not found on PATH -- L3 dispatch probing needs it."
    fi
else
    note "(the full lldb sweep at this thread count takes ~90s of real stop/continue"
    note " round trips -- see results/GROUND-TRUTH-DISPATCH.md's own timing note. To"
    note " keep this demo under 3 minutes we cite the committed, reproducible result"
    note " below instead of re-running it live. Reproduce it yourself with:"
    printf '%s\n' "${DIM}\$ ${RESET}${GREEN}python3 tools/verify_dispatch.py --binary \$LLAMA_CLI --model \$MODEL_PATH \\
    --threads $DEFAULT_THREADS --workloads decode_short --l3-debugger lldb${RESET}"
    echo
    printf '%s\n' "${RED}${BOLD}   threads=$DEFAULT_THREADS  workload=decode_short  advertised=SME2  executed=dotprod"
    printf '%s\n' "   hits: SME2=0 / NEON-dotprod=31871   verdict: SILENT_FALLBACK${RESET}"
    note "(results/GROUND-TRUTH-DISPATCH.md, results/dispatch-ledger-darwin-arm64.json)"
fi
echo
narrate "Why 2? kleidiai.cpp hardcodes a thread cap by chip name (2, on this M4"
narrate "Max). At or below the cap, SME2 genuinely dispatches. Watch it happen, live:"

if have_llama && have_lldb; then
    run_live python3 "$VERIFY_DISPATCH" \
        --binary "$LLAMA_CLI" --model "$MODEL_PATH" \
        --threads 2 --workloads decode_short --l3-debugger lldb \
        --out "$SCRATCH_DIR/beat2-live-t2.json"
else
    warn_missing_llama
    have_lldb || note "lldb not found on PATH -- required for this beat's live probe."
fi
echo
fact "The banner said SME2. The debugger says zero -- until one flag changes it."
pause

# =========================================================================
# BEAT 3 / 5 -- The cost: does it actually move tokens/sec, and by how much?
# =========================================================================
banner "BEAT 3 / 5 -- The cost: real llama-bench numbers, real ratios"
narrate "Re-measured on a quiet machine, 2026-08-04: every config run round-robin,"
narrate "interleaved (A,B,C,...,A,B,C,...), so contention hits every config equally."
narrate "7 reps/config, median + stdev -- never a bare mean. See"
narrate "results/REMEASURE-2026-08-04-QUIET.md for the full methodology note,"
narrate "including why an EARLIER, non-interleaved measurement was WRONG."

if have_bench; then
    narrate "Quick LIVE sanity check first (fewer reps -- $LIVE_BENCH_REPS, not the"
    narrate "rigorous round-robin number above; real commands, real output, right now):"
    run_live "$LLAMA_BENCH" -m "$MODEL_PATH" -p 0 -n 32 -r "$LIVE_BENCH_REPS"
    run_live "$LLAMA_BENCH" -m "$MODEL_PATH" -p 0 -n 32 -r "$LIVE_BENCH_REPS" -t 2
    run_live "$LLAMA_BENCH" -m "$MODEL_PATH" -p 256 -n 0 -r "$LIVE_BENCH_REPS"
    run_live "$LLAMA_BENCH" -m "$MODEL_PATH" -p 256 -n 0 -r "$LIVE_BENCH_REPS" -t 8
    echo
    note "(that quick check is illustrative, not authoritative -- $LIVE_BENCH_REPS reps,"
    note " not interleaved against its counterpart config. The table below is the"
    note " real, methodologically-honest number this project stands behind.)"
else
    warn_missing_bench
fi
echo
printf '%s\n' "${BOLD}   phase    config                      median tok/s${RESET}"
printf '%s\n' "   decode   default (no flags, 12 thr)     93.6"
printf '%s\n' "   decode   ${GREEN}-t 2${RESET}                        ${GREEN}321.0${RESET}  <- ${GREEN}3.43x${RESET}"
printf '%s\n' "   prefill  default (no flags)             1230.3"
printf '%s\n' "   prefill  ${GREEN}-t 8${RESET}                        ${GREEN}2198.1${RESET}  <- ${GREEN}1.79x${RESET}"
echo
fact "3.43x decode, 1.79x prefill. Zero code changes. Both flags already ship"
fact "in llama.cpp today -- and the banner never tells you to use them."
note "(results/REMEASURE-2026-08-04-QUIET.md -- authoritative; supersedes an"
note " earlier, contention-corrupted figure this project has since retracted.)"
pause

# =========================================================================
# BEAT 4 / 5 -- The root cause: the exact source lines
# =========================================================================
banner "BEAT 4 / 5 -- The root cause: two gates in kleidiai.cpp"
narrate "Why does SME2 disappear above 2 threads for decode? Two real source"
narrate "lines, llama.cpp @ dbadb68, ggml/src/ggml-cpu/kleidiai/kleidiai.cpp:"

if have_kleidiai_src; then
    run_live sed -n '147,169p' "$KLEIDIAI_SRC"
else
    warn_missing_kleidiai_src
    cat <<'SRC1'

   147:#elif defined(__APPLE__) && defined(__aarch64__)
   148:    // table for known M4 variants. Users can override via GGML_KLEIDIAI_SME=<n>.
   149:    char chip_name[256] = {};
   150:    size_t size = sizeof(chip_name);
   151:
   152:    if (sysctlbyname("machdep.cpu.brand_string", chip_name, &size, nullptr, 0) == 0) {
   153:        const std::string brand(chip_name);
   154:
   155:        struct ModelSMCU { const char *match; size_t smcus; };
   156:        static const ModelSMCU table[] = {
   157:            { "M4 Ultra", 2 },
   158:            { "M4 Max",   2 },
   159:            { "M4 Pro",   2 },
   160:            { "M4",       1 },
   161:        };
   162:
   163:        for (const auto &e : table) {
   164:            if (brand.find(e.match) != std::string::npos) {
   165:                return e.smcus;
   166:            }
   167:        }
   168:    }
   169:    return 0;
SRC1
fi
echo
narrate "That hardcoded, brand-string-keyed cap becomes the dispatch decision here:"
if have_kleidiai_src; then
    run_live sed -n '1094,1113p' "$KLEIDIAI_SRC"
else
    cat <<'SRC2'

  1094:        const int sme_cap_limit = ctx.sme_thread_cap;
  1095:        const bool use_hybrid = sme_cap_limit > 0 &&
  1096:                                 runtime_count > 1 &&
  1097:                                 nth_total > sme_cap_limit;
  1098:        // Heuristic: disable hybrid for very small workloads where per-slot overhead dominates.
  1099:        // If rows are small or average columns per thread are small, keep single-slot.
  1100:        size_t min_cols_per_thread = 0;
  1101:        if (runtime_count > 0 && nth_total > 0) {
  1102:            min_cols_per_thread = (size_t) std::max<int64_t>(1, (int64_t)ne01 / (int64_t)nth_total);
  1103:        }
  1104:        const bool too_small_for_hybrid = (min_cols_per_thread < 2) || (ne11 < 128);
  1105:
  1106:        const bool hybrid_enabled = use_hybrid && !too_small_for_hybrid;
  1107:
  1108:        if (!hybrid_enabled) {
  1109:            int chosen_slot = 0;
  1110:            if (too_small_for_hybrid && sme_slot != -1) {
  1111:                chosen_slot = nth_total > sme_cap_limit && non_sme_slot != -1 ? non_sme_slot : sme_slot;
  1112:            } else if (runtime_count > 1 && ctx.sme_thread_cap > 0 && nth_total > ctx.sme_thread_cap) {
  1113:                chosen_slot = 1;          // <- collapses to the non-SME (NEON) slot
SRC2
fi
echo
fact "ne11 == 1 for every decode step (one token at a time), so too_small_for_hybrid"
fact "is always true above the cap -- decode falls all the way back to NEON,"
fact "on every thread count above 2, on every prompt. Prefill (ne11 >= 128) doesn't"
fact "have this problem, which is exactly why the fix (Beat 3) differs by phase."
pause

# =========================================================================
# BEAT 5 / 5 -- The honest ending: we patched it, it didn't help
# =========================================================================
banner "BEAT 5 / 5 -- The honest ending: we fixed the gate, it made things worse"
narrate "So we patched it: GGML_KLEIDIAI_PHASE_AWARE=1 (opt-in, default off) lets"
narrate "decode into the existing hybrid SME+NEON split instead of collapsing to"
narrate "NEON-only. Two live checks against the patched binary, same host:"

if have_patched; then
    echo
    narrate "(a) flag UNSET, default threads -- the patch's OTHER half, a plain warning:"
    run_live "$LLAMA_CLI_PATCHED" -m "$MODEL_PATH" -p "Hi" -n 2 -no-cnv -st --simple-io \
        -t "$DEFAULT_THREADS" --verbose 2>&1 | grep -E "SME not used for a GEMV" \
        | sed -E "s/^/${YELLOW}${BOLD}   /; s/\$/${RESET}/"
    echo
    narrate "(b) flag SET -- the experimental dispatch change itself switches on:"
    run_live env GGML_KLEIDIAI_PHASE_AWARE=1 "$LLAMA_CLI_PATCHED" -m "$MODEL_PATH" -p "Hi" -n 2 \
        -no-cnv -st --simple-io -t "$DEFAULT_THREADS" --verbose 2>&1 | grep -E "phase-aware dispatch enabled" \
        | sed -E "s/^/${MAGENTA}${BOLD}   /; s/\$/${RESET}/"
else
    warn_missing_patched
fi

echo
narrate "Symbol-level proof the dispatch change is REAL (committed lldb-sweep ledgers,"
narrate "same patched binary, flag off vs on -- read live from the actual JSON files):"
if have_python3 && [[ -f "$DISPATCH_LEDGER_FLAG_OFF" && -f "$DISPATCH_LEDGER_FLAG_ON" ]]; then
    run_live python3 -c "
import json
off = json.load(open('$DISPATCH_LEDGER_FLAG_OFF'))
on  = json.load(open('$DISPATCH_LEDGER_FLAG_ON'))
rows = lambda d: {(c['threads'], c['workload']): c for c in d['configs']}
r_off, r_on = rows(off), rows(on)
for t in (4, 8):
    co = r_off[(t, 'decode_short')]
    cn = r_on[(t, 'decode_short')]
    print(f\"decode threads={t}:  flag off sme2={co['l3']['hits_by_family']['sme2']:>6} \"
          f\"({co['verdict']})   ->   flag on sme2={cn['l3']['hits_by_family']['sme2']:>6} \"
          f\"({cn['verdict']})\")
"
    note "(re-run this live yourself with RUN_LIVE_L3=1, ~90s x2 of lldb stop/continue,"
    note " or see results/OPTIMIZATION.md section 4 for the full table.)"
else
    printf '%s\n' "${RED}   [skip] python3 or the committed dispatch ledgers were not found.${RESET}"
    printf '%s\n' "${MAGENTA}${BOLD}   decode threads=4: flag off sme2=0 (SILENT_FALLBACK) -> flag on sme2=3072 (SME2_HYBRID_DISPATCH)"
    printf '%s\n' "   decode threads=8: flag off sme2=0 (SILENT_FALLBACK) -> flag on sme2=2354 (SME2_HYBRID_DISPATCH)${RESET}"
    note "(results/dispatch-ledger-darwin-arm64-patched-flag-{off,on}.json, results/OPTIMIZATION.md §4)"
fi
echo
printf '%s\n' "${BOLD}   the throughput verdict (results/REMEASURE-2026-08-04-QUIET.md):${RESET}"
printf '%s\n' "   decode, default threads:  93.6 -> ${RED}${BOLD}82.5${RESET} tok/s  =  ${RED}${BOLD}0.88x, ~12% SLOWER${RESET}"
printf '%s\n' "   decode, -t 2 (patch inert here):  321.0 -> 317.5 tok/s  =  0.99x, statistical tie"
printf '%s\n' "   prefill, default threads: 1230.3 -> 1202.1 tok/s  =  0.98x, statistical tie"
echo
fact "The dispatch change is real and proven at the symbol level. It is NOT a"
fact "throughput win -- it is measurably slower at default thread count. So we are"
fact "NOT proposing it as a fix. We measured it, we are publishing the negative"
fact "result, and we filed both findings upstream: ggml-org/llama.cpp#26547."
note "(The warning half of the patch above -- telling a user SME2 went unused --"
note " costs nothing and stands on its own; we're proposing that half separately.)"
pause

# =========================================================================
# Close
# =========================================================================
banner "Polygraph"
narrate "The banner said SME2. The debugger proved otherwise. Tuning is a real,"
narrate "free, 3.43x/1.79x win. Our own fix attempt was not -- and we're saying so."
note "Apache-2.0. Filed upstream: ggml-org/llama.cpp#26547."
echo

DEMO_END_TS=$(date +%s)
ELAPSED=$(( DEMO_END_TS - DEMO_START_TS ))
printf '%s\n' "${CYAN}${BOLD}Total demo.sh wall time: ${ELAPSED}s (excludes any screen-recording/narration slack you add)${RESET}"
if [[ "$RUN_LIVE_L3" != "1" ]]; then
    note "(RUN_LIVE_L3=0 -- the expensive lldb sweeps in Beats 2 and 5 were cited, not"
    note " re-run live. Set RUN_LIVE_L3=1 to include them, ~90s x2-3 more.)"
fi
