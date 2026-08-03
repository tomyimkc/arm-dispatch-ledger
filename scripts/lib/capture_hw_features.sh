#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/capture_hw_features.sh — record the CPU feature set of the
# runner this lane executed on, plus the llama-cli backend-selection banner
# (the "SME = 1 | SME2 = 1 | KLEIDIAI = 1" line etc). This captures
# compile-time / selection-time facts only — it is deliberately NOT the
# dispatch-time verification (that is tools/verify_dispatch's job; see the
# CROSS-PACKAGE CONTRACT in scripts/common.sh). Recording both side by side
# is what makes Finding 1 legible from the artifact alone: the banner can
# say KLEIDIAI=1 while the thread-sweep hit-count trace (from
# verify_dispatch) shows 0 dispatches at the runner's default thread count.
#
# Output: results/hw-features-<slug>.txt, where <slug> is derived from
# `uname -m` + a short OS/CPU identifier so the three CI lanes never clobber
# each other's file when results/ is merged into one artifact.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

STAGE="capture_hw_features"

os_name="$(uname -s)"
arch_name="$(uname -m)"
slug="${os_name}-${arch_name}"
out_file="$RESULTS_DIR/hw-features-${slug}.txt"

{
    echo "# Hardware feature capture"
    echo "captured_at_utc: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "uname -a: $(uname -a)"
    echo "nproc: $(detect_nproc)"
    echo

    if [[ "$os_name" == "Darwin" ]]; then
        echo "## sysctl (Apple Silicon feature flags)"
        echo "cpu.brand_string: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
        for key in \
            hw.optional.arm.FEAT_SME \
            hw.optional.arm.FEAT_SME2 \
            hw.optional.arm.FEAT_SME_F64F64 \
            hw.optional.arm.FEAT_SME_I16I64 \
            hw.optional.arm.sme_max_svl_b \
            hw.optional.arm.FEAT_SVE \
            hw.optional.arm.FEAT_I8MM \
            hw.optional.arm.FEAT_BF16 \
            hw.optional.arm.FEAT_DotProd; do
            val="$(sysctl -n "$key" 2>/dev/null || echo 'absent')"
            echo "$key: $val"
        done
    elif [[ "$os_name" == "Linux" ]]; then
        echo "## /proc/cpuinfo Features line (first core)"
        grep -m1 '^Features' /proc/cpuinfo 2>/dev/null || echo "no /proc/cpuinfo Features line found"
        echo
        echo "## /proc/cpuinfo model name / CPU part (first core)"
        grep -m1 -E '^(model name|CPU part|CPU implementer)' /proc/cpuinfo 2>/dev/null || true
        # SVE vector length, if the kernel + hardware expose it (aarch64 only).
        if command -v lscpu >/dev/null 2>&1; then
            echo
            echo "## lscpu"
            lscpu 2>/dev/null || true
        fi
    else
        echo "## unrecognized OS ($os_name); no feature probe implemented"
    fi

    echo
    echo "## llama-cli backend-selection banner"
    if [[ -x "$LLAMA_CLI" && -f "$MODEL_PATH" ]]; then
        # The system_info / kleidiai selection lines ("SME = 1 | SME2 = 1 |
        # KLEIDIAI = 1", "kleidiai: primary q4 kernel feature SME2") are only
        # emitted at verbose (-v) log level, not in default output — verified
        # by hand on this M4 Max. -p must be a real, non-empty prompt with
        # -n >= 1: an empty prompt (or -n 0) leaves llama-cli sitting in its
        # interactive read loop forever even with -no-cnv, which is exactly
        # the "hangs forever on non-TTY stdin" trap noted in this project's
        # findings. </dev/null is a second line of defence against that.
        "$LLAMA_CLI" -m "$MODEL_PATH" -p "Hi" -n 1 -no-cnv -st --simple-io -t 1 -v \
            </dev/null 2>&1 | grep -iE 'SME|SVE|KLEIDIAI|system_info|CPU :' || \
            echo "(no matching banner lines found in llama-cli output — see full log if needed)"
    else
        echo "llama-cli and/or model GGUF not present yet ($LLAMA_CLI , $MODEL_PATH) — run scripts/lib/build_llamacpp.sh and scripts/lib/fetch_model.sh first"
    fi
} >"$out_file" 2>&1 || true

record_stage "$STAGE" OK "wrote $out_file"
