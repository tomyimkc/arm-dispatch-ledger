#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
#
# demo/demo.sh -- self-contained, timed, narrated terminal walkthrough for the
# submission video. Tells the whole story -- claim, lie, proof, cost, fix, and
# the gap the fix can't close -- in well under 3 minutes of screen time.
#
# Design rules this script follows (do not relax these when editing):
#   1. Every number this script PRINTS is either (a) produced live, in front
#      of the viewer, by a real command in this repo, or (b) a number already
#      committed to results/ in an earlier, properly-measured run (5 reps,
#      interleaved, median -- see results/SUMMARY.md), clearly labelled as
#      such with the exact command to reproduce it live. Nothing is invented,
#      rounded up, or silently reused across a different methodology.
#   2. Every live command is echoed before it runs, so a screen recording
#      shows the viewer exactly what was typed, not a black box.
#   3. Never abort mid-take. No `set -e`. Missing binaries/models/tools
#      degrade to a clear, actionable message and the *next* beat still runs.
#   4. Idempotent: writes nothing into the repo tree. All scratch output goes
#      to $SCRATCH_DIR (outside the checkout). Safe to re-run any number of
#      times, in any order of prior runs, with no cleanup required.
#   5. Deterministic wall-clock budget: beats that are cheap (a few seconds)
#      run live by default. The one beat that is expensive by construction
#      (a full-sweep dispatch probe at a high thread count, ~90s of lldb
#      stop/continue round trips -- see results/GROUND-TRUTH-DISPATCH.md's
#      own timing note) is *pre-measured and cited* by default, exactly the
#      way results/SUMMARY.md and docs/VIDEO.md already treat that same cost,
#      and can be switched to a real live run with RUN_LIVE_L3=1 for anyone
#      who wants to pay the extra ~90s for full live authenticity.
#
# Usage:
#   ./demo/demo.sh                  # fast path, ~110-140s, ready to screen-record
#   PAUSE=1 ./demo/demo.sh          # shorter pauses, for a dry run / rehearsal
#   PAUSE=0 ./demo/demo.sh          # no pauses at all -- fastest smoke test
#   RUN_LIVE_L3=1 ./demo/demo.sh    # also live-run the expensive default-thread
#                                   # dispatch probe (~90s extra) instead of
#                                   # citing the committed ledger for it
#   NO_COLOR=1 ./demo/demo.sh       # plain text, no ANSI colour
#
# Env vars (all optional, all have working defaults on the reference machine
# this repo was built and measured on -- see results/SUMMARY.md):
#   LLAMA_CLI        path to llama-cli built -DGGML_CPU_KLEIDIAI=ON
#   LLAMA_BENCH      path to llama-bench (same build)
#   MODEL_PATH       path to the Q4_0 GGUF (Qwen2.5-0.5B-Instruct)
#   DEFAULT_THREADS  the "reasonable default" thread count used to show the
#                    silent fallback (default: 8 -- see README's own framing)
#   PAUSE            seconds paused after each beat (default: 2.5)
#   RUN_LIVE_L3      1 = actually run the expensive zero-hit probe live
#   SCRATCH_DIR      where scratch/temp output goes (default: see below)

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

CACHE_DIR_GUESS="${TMPDIR:-/tmp}/arm-dispatch-ledger-cache"

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

KERNEL_TEST="$REPO_ROOT/kernels/build/kernel_test"
KERNEL_BENCH="$REPO_ROOT/kernels/build/kernel_bench"
MCP_SERVER="$REPO_ROOT/mcp/server.py"
VERIFY_DISPATCH="$REPO_ROOT/tools/verify_dispatch.py"

SCRATCH_DIR="${SCRATCH_DIR:-${TMPDIR:-/tmp}/arm-dispatch-ledger-demo}"
mkdir -p "$SCRATCH_DIR"

DEFAULT_THREADS="${DEFAULT_THREADS:-8}"
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

warn_missing_llama() {
    printf '%s\n' "${RED}${BOLD}   [skip] llama-cli and/or the model GGUF were not found.${RESET}"
    note "LLAMA_CLI  = ${LLAMA_CLI:-<not found>}"
    note "MODEL_PATH = ${MODEL_PATH:-<not found>}"
    note "Build them with: ./scripts/setup.sh   (see README.md, Option B / Option A)"
    note "Or point this script at an existing build:"
    note "  LLAMA_CLI=/path/to/llama-cli MODEL_PATH=/path/to/model.gguf ./demo/demo.sh"
}

have_lldb() { command -v lldb >/dev/null 2>&1; }

# -----------------------------------------------------------------------
# Start
# -----------------------------------------------------------------------

DEMO_START_TS=$(date +%s)

[[ -t 1 ]] && clear 2>/dev/null || true
printf '%s\n' "${CYAN}${BOLD}"
cat <<'BANNER'
   _____                   ____  _                 __       __
  / ___/  ____ ___         / __ \(_)________  ____ _/ /______/ /
  \__ \  / __ `__ \       / / / / / ___/ __ \/ __ `/ __/ ___/ /_
 ___/ / / / / / / /      / /_/ / (__  ) /_/ / /_/ / /_/ /__/ __/
/____(_)_/ /_/ /_/      /_____/_/____/ .___/\__,_/\__/\___/_/
                                       /_/    __             __
                        __        __         / /   ___  ____/ /___ ____  _____
                       / /   ___ / /__  ____ / /   / _ \/ __  / __ `/ _ \/ ___/
                      / /___/ -_) __/ / __// /___/  __/ /_/ / /_/ /  __/ /
                     /_____/\__/\__/_/     /_____/\___/\__,_/\__, /\___/_/
                                                             /____/
BANNER
printf '%s\n' "$RESET"
narrate "Arm Dispatch Ledger -- does the SME2 kernel llama.cpp advertises actually run?"
note "Apache-2.0 -- github.com/tomyimkc/arm-dispatch-ledger -- Arm Create: AI Optimization Challenge"
echo
note "LLAMA_CLI        = ${LLAMA_CLI:-<not found>}"
note "LLAMA_BENCH       = ${LLAMA_BENCH:-<not found>}"
note "MODEL_PATH        = ${MODEL_PATH:-<not found>}"
note "DEFAULT_THREADS   = $DEFAULT_THREADS   PAUSE=${PAUSE}s   RUN_LIVE_L3=$RUN_LIVE_L3"
pause

# =========================================================================
# BEAT 1 -- the claim
# =========================================================================
banner "BEAT 1 / 7 -- The claim: the banner says SME2 is running"
narrate "llama.cpp's own startup log, at --verbose, claims the accelerated SME2"
narrate "kernel is selected for every matmul family. Watch the log, live."

if have_llama; then
    CLAIM_LOG="$SCRATCH_DIR/beat1-claim.log"
    run_live "$LLAMA_CLI" -m "$MODEL_PATH" -p "The capital of France is" -n 4 \
        -no-cnv -st --simple-io -t "$DEFAULT_THREADS" --verbose > "$CLAIM_LOG" 2>&1
    echo
    note "(full --verbose log captured; showing just the dispatch-relevant lines)"
    if grep -E "system_info:.*SME|kleidiai: (primary|SME2 enabled)" "$CLAIM_LOG" >/dev/null 2>&1; then
        grep -E "system_info:.*SME|kleidiai: (primary|SME2 enabled)" "$CLAIM_LOG" \
            | sed -E "s/^/${RED}${BOLD}   /; s/\$/${RESET}/"
    else
        printf '%s\n' "${RED}   (no SME/KLEIDIAI log lines found -- this llama.cpp build may not have"
        printf '%s\n' "    -DGGML_CPU_KLEIDIAI=ON, or the log format changed. See $CLAIM_LOG.)${RESET}"
    fi
else
    warn_missing_llama
fi
echo
fact "Claim: SME2 = 1, and 'primary q4 kernel feature SME2' -- selected for every op."
pause

# =========================================================================
# BEAT 2 -- the lie (default threads never dispatch it)
# =========================================================================
banner "BEAT 2 / 7 -- The lie: at a normal thread count, it never fires"
narrate "A timing-only benchmark cannot see past that log line. We put a real"
narrate "lldb breakpoint on the SME2 kernel's own entry points and count hits."

if [[ "$RUN_LIVE_L3" == "1" ]]; then
    if have_llama && have_lldb; then
        narrate "(RUN_LIVE_L3=1 -- running the real sweep live, ~90s of lldb stop/continue)"
        run_live python3 "$VERIFY_DISPATCH" \
            --binary "$LLAMA_CLI" --model "$MODEL_PATH" \
            --threads "$DEFAULT_THREADS" --workloads decode_short \
            --l3-debugger lldb \
            --out "$SCRATCH_DIR/beat2-live-t${DEFAULT_THREADS}.json"
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
    note "(results/GROUND-TRUTH-DISPATCH.md, results/dispatch-ledger-darwin-arm64.json --"
    note " independently re-confirmed this session: a fresh run at -t 8 reproduced"
    note " 0 SME2 / 31870 dotprod hits, within run-to-run noise of the committed 31871)"
fi
echo
fact "The banner said SME2. The debugger says zero. Every single time, above the cap."
pause

# =========================================================================
# BEAT 3 -- the same probe at -t 2 (live -- this one is fast)
# =========================================================================
banner "BEAT 3 / 7 -- The same probe, at -t 2: thousands of hits"
narrate "Why 2? kleidiai.cpp hardcodes a thread cap by chip name (2, on this M4"
narrate "Max). At or below the cap, SME2 genuinely dispatches. Watch it happen."

if have_llama && have_lldb; then
    run_live python3 "$VERIFY_DISPATCH" \
        --binary "$LLAMA_CLI" --model "$MODEL_PATH" \
        --threads 2 --workloads decode_short --l3-debugger lldb \
        --out "$SCRATCH_DIR/beat3-live-t2.json"
else
    warn_missing_llama
    have_lldb || note "lldb not found on PATH -- required for this beat's live probe."
fi
echo
fact "Same binary, same model, one flag changed: -t 8 -> -t 2. Zero hits -> thousands."
pause

# =========================================================================
# BEAT 4 -- the measured throughput consequence
# =========================================================================
banner "BEAT 4 / 7 -- Does it actually cost tokens/sec? (yes -- and it's not symmetric)"
narrate "5 reps/cell, interleaved, warmup-discarded, median +/- stddev -- never a"
narrate "bare mean. Real numbers from results/bench/bench-apple-m4-max.md:"
echo
printf '%s\n' "${BOLD}   phase          config          tok/s${RESET}"
printf '%s\n' "   decode         SME2 @ 2 thr    ${GREEN}327.6${RESET}  <- fastest decode config measured"
printf '%s\n' "   decode         NEON @ 8 thr    155.3  (SILENT_FALLBACK config, from Beat 2)"
printf '%s\n' "   prefill_long   NEON @ 8 thr    ${GREEN}2676.4${RESET} <- fastest prefill config measured"
printf '%s\n' "   prefill_long   SME2-hybrid @ 8 1830.1  (what the default env actually gets at 8 threads)"
echo
fact "Decode: SME2@2 wins outright, every thread count measured (up to 2.1x)."
fact "Prefill: plain NEON@8 beats SME2's own best cell by 1.46x. Not the flattering story."
note "(results/SUMMARY.md section 4 has the full reconciliation, including the"
note " unflattering half -- we are not hiding the case where SME2 loses.)"
pause

# =========================================================================
# BEAT 5 -- the fix, and the gap it can't close
# =========================================================================
banner "BEAT 5 / 7 -- The fix: llama.cpp already has a per-phase thread split"
narrate "decode wants SME2 + few threads. prefill wants NEON + many threads."
narrate "llama-cli already has separate flags for exactly this:"
if [[ -n "$LLAMA_CLI" && -x "$LLAMA_CLI" ]]; then
    HELP_LOG="$SCRATCH_DIR/beat5-help.log"
    run_live_quiet "$HELP_LOG" "$LLAMA_CLI" --help
    grep -A1 -- "-tb,   --threads-batch" "$HELP_LOG" | sed -E "s/^/   /" \
        || note "(-tb/--threads-batch not found in this build's --help; see $HELP_LOG)"
else
    note "(llama-cli --help unavailable -- see: -t/--threads for decode, -tb/--threads-batch for prefill)"
fi
echo
narrate "So the thread split is expressible today. Proving it runs, live, in one process:"
narrate "(ignore the printed prompt/gen rate below -- one unwarmed, chat-templated dry"
narrate "run is noise; Beat 4's numbers above are the rigorous, comparable ones)"
if have_llama; then
    run_live "$LLAMA_CLI" -m "$MODEL_PATH" -p "The capital of France is" -n 8 \
        -no-cnv -st --simple-io -t 2 -tb 8
    note "(single dry-run, short prompt -- illustrative that -t 2 -tb 8 runs today;"
    note " NOT the rigorous 5-rep methodology behind the Beat 4 numbers above)"
else
    warn_missing_llama
fi
echo
narrate "But GGML_KLEIDIAI_SME -- the switch between the SME2 and NEON kernel"
narrate "families -- is read ONCE per process (kleidiai.cpp, ctx.sme_thread_cap"
narrate "at context init), not per op. It cannot vary between -t and -tb within"
narrate "one running process. That is the missing capability:"
echo
printf '%s\n' "${RED}${BOLD}   decode@SME2(2thr)=327.6 + prefill@NEON-forced(8thr)=2676.4${RESET}  <- the true combined optimum"
printf '%s\n' "   ${YELLOW}decode@SME2(2thr)=327.6 + prefill@hybrid(8thr, default env)=1830.1${RESET}  <- best ONE PROCESS can do today"
note "(prefill's own env-forced ceiling, 2676.4, is 1.46x its own hybrid ceiling, 1830.1 --"
note " both numbers already measured in results/bench/, restated here, not remeasured)"
fact "One process cannot get SME2-for-decode and NEON-forced-for-prefill at the same time."
fact "That per-phase KERNEL-FAMILY selection is the gap -- filed upstream, see Beat 7."
pause

# =========================================================================
# BEAT 6 -- proof the silicon isn't the limiter
# =========================================================================
banner "BEAT 6 / 7 -- Proof this isn't an SME2-is-slow story"
narrate "Hand-written NEON/SME2/SVE2 kernels, correctness-tested bit-exact"
narrate "against a scalar reference, benchmarked against the strongest fair"
narrate "baseline (Apple Accelerate) -- no strawman comparison."
if [[ -x "$KERNEL_TEST" ]]; then
    run_live "$KERNEL_TEST"
else
    printf '%s\n' "${RED}   [skip] kernels/build/kernel_test not found.${RESET}"
    note "Build with: cd kernels && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -- -j"
fi
if [[ -x "$KERNEL_BENCH" ]]; then
    run_live "$KERNEL_BENCH"
else
    printf '%s\n' "${RED}   [skip] kernels/build/kernel_bench not found (build steps above).${RESET}"
fi
echo
fact "SME2 kernels are correct and fast. The bug is in llama.cpp's dispatcher, not the silicon."
pause

# =========================================================================
# BEAT 7 -- reusable artifacts, agentic MCP, giving back
# =========================================================================
banner "BEAT 7 / 7 -- Reusable, agentic, and given back upstream"
narrate "Every tool above is reusable for the next Arm dispatch bug, not just"
narrate "this one -- including an MCP server any agent can query directly:"
if [[ -f "$MCP_SERVER" ]] && command -v python3 >/dev/null 2>&1; then
    MCP_LOG="$SCRATCH_DIR/beat7-mcp.log"
    run_live_quiet "$MCP_LOG" python3 "$MCP_SERVER" --selftest
    grep -o '"name":"[a-zA-Z_]*"' "$MCP_LOG" | sort -u | sed -E "s/^/   /"
    note "4 callable MCP tools: detect_arm_features, verify_dispatch, recommend_config, explain_finding"
else
    printf '%s\n' "${RED}   [skip] mcp/server.py or python3 not found.${RESET}"
fi
echo
narrate "And the finding itself was filed upstream, not kept for the submission:"
fact "ggml-org/llama.cpp#26547 -- both findings, exact source lines, reproduction steps, offer to patch."
pause

# =========================================================================
# Close
# =========================================================================
banner "Arm Dispatch Ledger"
narrate "The banner said SME2. The debugger proved otherwise. We measured the real"
narrate "cost, found the fix llama.cpp already half-supports, and named the exact"
narrate "capability -- per-phase kernel-family dispatch -- it is still missing."
note "Apache-2.0. Kernels, harness, and MCP server: all reusable for the next one."
echo

DEMO_END_TS=$(date +%s)
ELAPSED=$(( DEMO_END_TS - DEMO_START_TS ))
printf '%s\n' "${CYAN}${BOLD}Total demo.sh wall time: ${ELAPSED}s (excludes any screen-recording/narration slack you add)${RESET}"
if [[ "$RUN_LIVE_L3" != "1" ]]; then
    note "(RUN_LIVE_L3=0 -- Beat 2's expensive probe was cited, not re-run live. Set RUN_LIVE_L3=1 to include it, ~90s more.)"
fi
