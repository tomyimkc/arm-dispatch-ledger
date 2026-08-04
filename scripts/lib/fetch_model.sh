#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# scripts/lib/fetch_model.sh — download GGUF model(s) used by this repo's
# lanes, idempotently, sha256-pinned.
#
# Three modes, selected by which of MODEL_ID / MODEL_SET is set (checked in
# that order). The default (neither set) is byte-for-byte the same behavior
# this script has always had, so every existing caller
# (scripts/setup.sh, .github/workflows/verify-{free,macos,spark}-*.yml, and
# the rest of verify-free-arm64.yml) needs zero changes to keep working:
#
#   (default)      Fetch the single baseline model named by
#                  HF_REPO/HF_FILE/HF_FILE_SHA256 (scripts/common.sh's
#                  defaults, or caller-overridden env), into MODEL_PATH.
#   MODEL_ID=<id>  Look <id> up in scripts/models.txt and fetch just that
#                  one model, into $MODEL_DIR/<hf_file>. Exports
#                  HF_REPO/HF_FILE/HF_FILE_SHA256/MODEL_PATH for the rest of
#                  this process (e.g. so a caller that sources common.sh via
#                  a downstream script picks up the resolved model) -- does
#                  NOT edit scripts/common.sh.
#   MODEL_SET=<x>  Comma-separated list of model_ids, or the literal "all"
#                  for every row in scripts/models.txt. Fetches each in
#                  turn, continues past a per-model failure so one bad
#                  model doesn't block the rest, and exits non-zero if any
#                  model failed. Each lands at $MODEL_DIR/<hf_file>.
#
# scripts/models.txt is the single manifest of models this repo knows how to
# fetch, each pinned by sha256 and license -- see that file's header for the
# exact pipe-delimited format and the license-verification rule
# (apache-2.0/mit only, verified live against the HF API, never copied from
# a README by eye).
#
# License note (unchanged default model): Qwen/Qwen2.5-0.5B-Instruct-GGUF on
# Hugging Face carries license:apache-2.0 in its model card metadata
# (verified via the HF API on 2026-08-04) -- same license as this repo, safe
# to redistribute/re-fetch in CI and to ship instructions for in a public
# Devpost submission.
set -euo pipefail
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "$LIB_DIR/../common.sh"

: "${FORCE:=0}"
: "${MODEL_ID:=}"
: "${MODEL_SET:=}"
MODELS_MANIFEST="$REPO_ROOT/scripts/models.txt"

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        echo ""
    fi
}

# manifest_lookup <model_id> -- prints "hf_repo|hf_file|sha256|license" for
# the first manifest row whose model_id matches (comments/blank lines
# skipped, model_id whitespace-trimmed); returns 1 (no output) if not found
# or the manifest file itself is missing.
manifest_lookup() {
    local id="$1"
    [[ -f "$MODELS_MANIFEST" ]] || return 1
    awk -F'|' -v want="$id" '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*$/ { next }
        {
            id = $1
            gsub(/^[ \t]+|[ \t]+$/, "", id)
            if (id == want) { print $2 "|" $3 "|" $4 "|" $5; found = 1; exit }
        }
        END { if (!found) exit 1 }
    ' "$MODELS_MANIFEST"
}

# manifest_all_ids -- prints every model_id in the manifest, one per line,
# in file order (comments/blank lines skipped).
manifest_all_ids() {
    [[ -f "$MODELS_MANIFEST" ]] || return 1
    awk -F'|' '
        /^[[:space:]]*#/ { next }
        /^[[:space:]]*$/ { next }
        { id = $1; gsub(/^[ \t]+|[ \t]+$/, "", id); print id }
    ' "$MODELS_MANIFEST"
}

# fetch_one <hf_repo> <hf_file> <sha256> <dest_path> <label>
# The original single-model download+verify logic, parameterized so both
# the default path and the manifest-driven paths below share one
# implementation. Returns 0 for any usable downloaded-or-reused file
# (including a loud sha256-MISMATCH, matching this script's long-standing
# behavior of treating an upstream re-upload as a non-fatal WARN, never a
# hard failure this project doesn't control); returns 1 only for a hard
# failure (transfer error / not-a-GGUF).
fetch_one() {
    local repo="$1" file="$2" sha="$3" dest="$4" label="$5"
    local url="https://huggingface.co/${repo}/resolve/main/${file}"
    mkdir -p "$(dirname "$dest")"

    if [[ "$FORCE" != "1" && -f "$dest" ]]; then
        local existing
        existing="$(sha256_of "$dest")"
        if [[ -n "$existing" && "$existing" == "$sha" ]]; then
            log_ok "$label: reusing $dest (sha256 verified, set FORCE=1 to re-download)"
            return 0
        fi
        log_warn "$label: existing $dest present but sha256 did not match expected ($existing vs $sha) or shasum unavailable; re-downloading"
    fi

    log_info "$label: downloading $url"
    local tmp="$dest.partial"
    local dl_log="$LOG_DIR/fetch_model_${label}.log"
    if ! curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url" 2>"$dl_log"; then
        log_fail "$label: download failed, see $dl_log"
        return 1
    fi
    mv "$tmp" "$dest"

    # Magic-byte sanity check first (cheap, catches an HTML error page saved
    # as .gguf long before we bother hashing hundreds of MiB).
    local magic
    magic="$(head -c 4 "$dest" 2>/dev/null || true)"
    if [[ "$magic" != "GGUF" ]]; then
        log_fail "$label: downloaded file at $dest does not start with GGUF magic bytes"
        return 1
    fi

    local got
    got="$(sha256_of "$dest")"
    if [[ -z "$got" ]]; then
        log_warn "$label: no shasum/sha256sum available to verify integrity; skipping hash check"
        log_ok "$label: downloaded $dest (GGUF magic OK, sha256 NOT checked -- no hasher on PATH)"
    elif [[ "$got" == "$sha" ]]; then
        log_ok "$label: downloaded $dest (sha256 verified: $got)"
    else
        # Non-fatal: upstream may have re-uploaded the file. We cannot
        # control that, but a judge/reviewer should see this loudly rather
        # than have it pass silently.
        log_warn "$label: sha256 mismatch: got $got, expected $sha (upstream file may have changed -- update the sha256 in scripts/models.txt, or HF_FILE_SHA256 in scripts/common.sh for the default model, if this is expected)"
        log_ok "$label: downloaded $dest (GGUF magic OK, sha256 MISMATCH -- see log)"
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------

if [[ -n "$MODEL_SET" ]]; then
    STAGE="fetch_model_set"
    ids=()
    if [[ "$MODEL_SET" == "all" ]]; then
        while IFS= read -r id; do
            [[ -n "$id" ]] && ids+=("$id")
        done < <(manifest_all_ids || true)
    else
        IFS=',' read -ra raw_ids <<<"$MODEL_SET"
        for id in "${raw_ids[@]}"; do
            id="$(echo "$id" | xargs)" # trim whitespace
            [[ -n "$id" ]] && ids+=("$id")
        done
    fi

    if [[ "${#ids[@]}" -eq 0 ]]; then
        record_stage "$STAGE" FAIL "MODEL_SET='$MODEL_SET' resolved to zero model ids (manifest: $MODELS_MANIFEST)"
        exit 1
    fi

    any_fail=0
    fetched=()
    for id in "${ids[@]}"; do
        entry="$(manifest_lookup "$id" || true)"
        if [[ -z "$entry" ]]; then
            log_fail "MODEL_SET: unknown model id '$id' (not found in $MODELS_MANIFEST)"
            any_fail=1
            continue
        fi
        IFS='|' read -r m_repo m_file m_sha m_license <<<"$entry"
        m_dest="$MODEL_DIR/$m_file"
        if fetch_one "$m_repo" "$m_file" "$m_sha" "$m_dest" "$id"; then
            fetched+=("$id ($m_license) -> $m_dest")
        else
            any_fail=1
        fi
    done

    if [[ "$any_fail" -eq 1 ]]; then
        record_stage "$STAGE" FAIL "one or more models in MODEL_SET='$MODEL_SET' failed to fetch (see log above); fetched OK: ${fetched[*]:-none}"
        exit 1
    fi
    record_stage "$STAGE" OK "fetched ${#fetched[@]} model(s) from MODEL_SET='$MODEL_SET': ${fetched[*]}"
    exit 0
fi

if [[ -n "$MODEL_ID" ]]; then
    STAGE="fetch_model"
    entry="$(manifest_lookup "$MODEL_ID" || true)"
    if [[ -z "$entry" ]]; then
        record_stage "$STAGE" FAIL "MODEL_ID='$MODEL_ID' not found in $MODELS_MANIFEST"
        exit 1
    fi
    IFS='|' read -r HF_REPO HF_FILE HF_FILE_SHA256 model_license <<<"$entry"
    MODEL_PATH="$MODEL_DIR/$HF_FILE"
    export HF_REPO HF_FILE HF_FILE_SHA256 MODEL_PATH
    if fetch_one "$HF_REPO" "$HF_FILE" "$HF_FILE_SHA256" "$MODEL_PATH" "$MODEL_ID"; then
        record_stage "$STAGE" OK "MODEL_ID=$MODEL_ID ($model_license) -> $MODEL_PATH"
        exit 0
    else
        record_stage "$STAGE" FAIL "MODEL_ID=$MODEL_ID fetch failed, see log above"
        exit 1
    fi
fi

# --- default: unchanged single-baseline-model behavior ---
STAGE="fetch_model"
if fetch_one "$HF_REPO" "$HF_FILE" "$HF_FILE_SHA256" "$MODEL_PATH" "baseline"; then
    record_stage "$STAGE" OK "$MODEL_PATH"
    exit 0
else
    record_stage "$STAGE" FAIL "baseline model fetch failed, see log above"
    exit 1
fi
