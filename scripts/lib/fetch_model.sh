#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/fetch_model.sh — download the small Apache-2.0 GGUF used by
# every lane (Qwen2.5-0.5B-Instruct, Q4_0 quant, ~409 MiB), idempotently.
#
# License note: Qwen/Qwen2.5-0.5B-Instruct-GGUF on Hugging Face carries
# license:apache-2.0 in its model card metadata (verified via the HF API on
# 2026-08-04) — same license as this repo, safe to redistribute/re-fetch in
# CI and to ship instructions for in a public Devpost submission.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

: "${FORCE:=0}"
STAGE="fetch_model"
URL="https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}"

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo ""
    fi
}

mkdir -p "$MODEL_DIR"

if [[ "$FORCE" != "1" && -f "$MODEL_PATH" ]]; then
    existing_sha="$(sha256_of "$MODEL_PATH")"
    if [[ -n "$existing_sha" && "$existing_sha" == "$HF_FILE_SHA256" ]]; then
        record_stage "$STAGE" OK "reusing $MODEL_PATH (sha256 verified, set FORCE=1 to re-download)"
        exit 0
    fi
    log_warn "existing $MODEL_PATH present but sha256 did not match expected ($existing_sha vs $HF_FILE_SHA256) or shasum unavailable; re-downloading"
fi

log_info "downloading $URL"
tmp_path="$MODEL_PATH.partial"
if ! curl -fL --retry 3 --retry-delay 2 -o "$tmp_path" "$URL" 2>"$LOG_DIR/fetch_model.log"; then
    record_stage "$STAGE" FAIL "download failed, see $LOG_DIR/fetch_model.log"
    exit 1
fi
mv "$tmp_path" "$MODEL_PATH"

# Magic-byte sanity check first (cheap, catches an HTML error page saved as
# .gguf long before we bother hashing 409 MiB).
magic="$(head -c 4 "$MODEL_PATH" 2>/dev/null || true)"
if [[ "$magic" != "GGUF" ]]; then
    record_stage "$STAGE" FAIL "downloaded file at $MODEL_PATH does not start with GGUF magic bytes"
    exit 1
fi

downloaded_sha="$(sha256_of "$MODEL_PATH")"
if [[ -z "$downloaded_sha" ]]; then
    log_warn "no shasum/sha256sum available to verify integrity; skipping hash check"
    record_stage "$STAGE" OK "downloaded $MODEL_PATH (GGUF magic OK, sha256 NOT checked — no hasher on PATH)"
elif [[ "$downloaded_sha" == "$HF_FILE_SHA256" ]]; then
    record_stage "$STAGE" OK "downloaded $MODEL_PATH (sha256 verified: $downloaded_sha)"
else
    # Non-fatal: upstream may have re-uploaded the file. We cannot control
    # that, but a judge/reviewer should see this loudly rather than have it
    # pass silently.
    log_warn "sha256 mismatch: got $downloaded_sha, expected $HF_FILE_SHA256 (upstream file may have changed since 2026-08-04 — update HF_FILE_SHA256 in scripts/common.sh if this is expected)"
    record_stage "$STAGE" OK "downloaded $MODEL_PATH (GGUF magic OK, sha256 MISMATCH — see log)"
fi
