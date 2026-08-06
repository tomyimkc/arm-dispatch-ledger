#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Polygraph contributors
"""check_claims.py -- the consistency gate for this repo's numeric claims.

This repo has twice shipped a wrong number: a fabricated "+57.3%" win (produced by
comparing two configs measured in different, unevenly-contended time windows -- see
`results/REMEASURE-2026-08-04-QUIET.md`), and a stale duplicate Devpost file. Both were
promise-based failures: nothing *mechanically* stopped a wrong or stale number from
being displayed. This tool is the structural fix.

It is stdlib-only (no third-party deps -- runs anywhere Python 3.8+ runs, including a
bare `ubuntu-latest` CI image) and does three independent things:

  1. RETRACTION GUARD -- fails if any of this project's own previously-retracted figures
     (the fabricated "+57.3%" episode and its downstream numbers) appear anywhere in the
     repo's prose without being inside a section that itself carries retraction language
     (retracted/superseded/historical), or an explicit, reasoned registry exemption for a
     *coincidental* digit collision with an unrelated, currently-valid number.

  2. LIVE CLAIM REGISTRY -- greps README.md, docs/*.md and site/ for numeric performance
     claims (ratios like "3.43x", throughput figures like "321.0 tok/s", signed/approximate
     percentages like "+57.3%" or "~12%", and dispatch hit counts) and fails if a claim
     found in prose is not backed by either (a) a hand-curated entry in the registry
     (`docs/CLAIMS.md`) whose cited results/ source file (and, where declared, JSON path)
     actually contains that value, or (b) any raw numeric leaf value committed anywhere
     under `results/**/*.json` (round-trip verification against the measurement JSON
     itself, so a number that was never actually measured cannot be typed in).

  3. CROSS-FILE AGREEMENT -- a natural corollary of (2): since every live number must
     resolve to one canonical, source-backed value, two files that print *different*
     numbers for "the same" cell cannot both pass (the wrong one has no source that backs
     it) -- this is how the tool catches drift, not just outright fabrication.

Exit code 0 = every live claim is registered/backed and no retracted figure leaked
unmarked. Exit code 1 = a clear, actionable report of what failed and why.

Usage:
    python3 tools/check_claims.py [--root PATH] [-v]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

CLAIMS_DOC = "docs/CLAIMS.md"
REGISTRY_BEGIN = "<!-- CLAIMS-REGISTRY:BEGIN -->"
REGISTRY_END = "<!-- CLAIMS-REGISTRY:END -->"

# Files scanned for the retraction guard (check 1) -- everything a reader or judge might
# actually open, plus the results/ prose that narrates the correction itself.
RETRACTION_SCAN_GLOBS = [
    "README.md",
    "docs/*.md",
    "results/*.md",
    "site/*.html",
    "site/*.js",
]

# Files scanned for the live-claim registry (checks 2/3) -- deliberately narrower, and
# matches this work package's brief literally: README.md, docs/*.md and site/.
REGISTRY_SCAN_GLOBS = [
    "README.md",
    "docs/*.md",
    "site/*.html",
    "site/*.js",
]

# docs/CLAIMS.md is the registry itself (it necessarily *contains* the retracted-figure
# literals and example numbers as data, not as claims) -- never scan it as a target.
SELF_EXCLUDE = {CLAIMS_DOC}

# results/ JSON is the raw-measurement corpus. Any number appearing anywhere as a leaf
# value in any of these files is treated as "backed by committed raw data" for check 2.
JSON_BACKED_GLOBS = [
    "results/*.json",
    "results/**/*.json",
]


# --------------------------------------------------------------------------------------
# Registry loading
# --------------------------------------------------------------------------------------


class RegistryError(RuntimeError):
    pass


def load_registry(root: Path) -> dict:
    claims_path = root / CLAIMS_DOC
    if not claims_path.is_file():
        raise RegistryError(f"registry doc not found: {claims_path}")
    text = claims_path.read_text(encoding="utf-8")
    try:
        start = text.index(REGISTRY_BEGIN) + len(REGISTRY_BEGIN)
        end = text.index(REGISTRY_END, start)
    except ValueError as e:
        raise RegistryError(
            f"{CLAIMS_DOC} is missing the {REGISTRY_BEGIN} / {REGISTRY_END} "
            "machine-readable block"
        ) from e
    block = text[start:end]
    # The registry lives inside a fenced ```json code block between the markers.
    fence_start = block.index("```json") + len("```json")
    fence_end = block.rindex("```")
    raw_json = block[fence_start:fence_end]
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise RegistryError(f"{CLAIMS_DOC} registry JSON is not valid: {e}") from e
    for key in ("retracted_figures", "claims"):
        if key not in data:
            raise RegistryError(f"{CLAIMS_DOC} registry is missing required key: {key}")
    return data


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def normalize_number(text: str) -> str:
    """Strip thousands-separator commas so '1,230.3' and '1230.3' compare equal."""
    return text.replace(",", "").strip()


def decimals_in(text: str) -> int:
    text = normalize_number(text)
    return len(text.split(".", 1)[1]) if "." in text else 0


def parse_float(text: str) -> float:
    return float(normalize_number(text))


def resolve_json_path(data, path: str):
    """Resolve a small dotted/bracket JSON path, e.g. 'rows[2].agg.median_ts'."""
    cur = data
    token = ""
    i = 0
    tokens: list = []
    while i < len(path):
        c = path[i]
        if c == ".":
            if token:
                tokens.append(token)
                token = ""
        elif c == "[":
            if token:
                tokens.append(token)
                token = ""
            j = path.index("]", i)
            tokens.append(int(path[i + 1 : j]))
            i = j
        else:
            token += c
        i += 1
    if token:
        tokens.append(token)
    for t in tokens:
        cur = cur[t]
    return cur


def iter_numeric_leaves(obj) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_numeric_leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_numeric_leaves(v)


def expand_globs(root: Path, patterns: list) -> list:
    out: list = []
    for pat in patterns:
        for p in sorted(glob.glob(str(root / pat), recursive=True)):
            path = Path(p)
            if not path.is_file() or path in out:
                continue
            if str(path.relative_to(root)) in SELF_EXCLUDE:
                continue
            out.append(path)
    return out


# --------------------------------------------------------------------------------------
# Check 1 -- retraction guard
# --------------------------------------------------------------------------------------


@dataclass
class Finding:
    check: str
    file: str
    line: int
    text: str
    reason: str


HEADING_RE = re.compile(r"^(#{1,6})\s+\S")


def _headings(lines: list) -> list:
    return [(i, len(m.group(1))) for i, line in enumerate(lines) if (m := HEADING_RE.match(line))]


def _enclosing_heading(lines: list, line_idx: int, headings: list) -> Optional[tuple]:
    """The nearest heading at or before line_idx, i.e. the most specific heading whose
    own preamble or descendants could contain this line. None if line_idx precedes the
    first heading in the file."""
    enclosing = None
    for h_idx, h_level in headings:
        if h_idx > line_idx:
            break
        enclosing = (h_idx, h_level)
    return enclosing


def _own_preamble_bound(lines: list, h_idx: int, headings: list) -> int:
    """End index of a heading's *own* preamble: the very next heading of ANY level (not
    the old "next heading of level <= its own" rule, which -- for a heading with no
    same-or-shallower sibling anywhere later in the file, e.g. the sole H1 title every
    file in this repo has -- degenerates to "the rest of the document", silently
    granting every later subsection's content the same retraction-context scope as
    content sitting directly in that H1's own, unrelated intro paragraphs. Caught by an
    adversarial self-test: see docs/CLAIMS.md and the git history of this function."""
    for h2_idx, _ in headings:
        if h2_idx > h_idx:
            return h2_idx
    return len(lines)


def _first_block(lines: list, start: int, end: int) -> list:
    """The first contiguous non-blank block starting at or after `start`, bounded by
    `end`: skip leading blank lines, then take lines up to (not including) the next
    blank line or `end`, whichever comes first."""
    i = start
    while i < end and not lines[i].strip():
        i += 1
    j = i
    while j < end and lines[j].strip():
        j += 1
    return lines[i:j]


def _blocks(lines: list, start: int, end: int) -> list:
    """Split [start, end) into contiguous non-blank blocks (paragraphs/blockquotes/
    tables), each a (block_start, block_end) pair, in document order."""
    out = []
    i = start
    while i < end:
        while i < end and not lines[i].strip():
            i += 1
        if i >= end:
            break
        j = i
        while j < end and lines[j].strip():
            j += 1
        out.append((i, j))
        i = j
    return out


def _heading_and_first_block(lines: list, h_idx: int, headings: list) -> list:
    """A heading's own text is checked directly (a heading can itself say "corrected"),
    plus the first block of its own preamble (one paragraph/blockquote/table -- not the
    whole preamble, and never a descendant subsection's content). Used for the PARENT
    check only: an explicit disclaimer placed directly under a "## Section" heading, and
    only the first block of that preamble -- a scope-declaring disclaimer, not a general
    license for anything anywhere under that heading (see results/OPTIMIZATION.md
    section 2's SUPERSEDED blockquote, which explicitly states it governs "each table"
    below it)."""
    bound = _own_preamble_bound(lines, h_idx, headings)
    return [lines[h_idx]] + _first_block(lines, h_idx + 1, bound)


def _own_context_block(lines: list, h_idx: int, line_idx: int, headings: list) -> list:
    """The match's own heading text, plus the block line_idx sits in, plus the blocks
    immediately before and after it (bounded by the heading's own preamble -- never a
    descendant subsection). Adjacent-block, not whole-section: this catches a same-
    section explanatory note placed directly after the table it annotates (e.g.
    results/SUMMARY.md's "(Note: the stddev value `198.9` ... is unrelated to the
    retracted ...)" note, one block below the table it explains) without granting a
    free pass to unrelated content several blocks away in a long, multi-topic preamble
    that happens to contain an unrelated sentence using a retraction keyword (verified
    against a synthetic adversarial case -- see git history of this function)."""
    bound = _own_preamble_bound(lines, h_idx, headings)
    blocks = _blocks(lines, h_idx + 1, bound)
    own_idx = None
    for i, (b_start, b_end) in enumerate(blocks):
        if b_start <= line_idx < b_end:
            own_idx = i
            break
    out = [lines[h_idx]]
    if own_idx is None:
        return out
    for i in (own_idx - 1, own_idx, own_idx + 1):
        if 0 <= i < len(blocks):
            b_start, b_end = blocks[i]
            out.extend(lines[b_start:b_end])
    return out


def _retraction_context_text(lines: list, line_idx: int) -> str:
    """Text checked for retraction-context keywords: the nearest enclosing heading's own
    text plus the block containing line_idx and its immediate neighbors (adjacent-block,
    see _own_context_block), PLUS -- one level up only -- the immediate parent heading's
    own text and its own preamble's *first* block only (a scope-declaring disclaimer,
    see _heading_and_first_block). Neither check ever includes a heading's whole
    multi-paragraph preamble or reaches into a nested subsection: both were tried and
    both proved too permissive against a synthetic adversarial case (a fabricated ratio
    and an unmarked retracted figure dropped into the middle of README.md's long,
    multi-topic intro, which contains an unrelated retraction sentence several
    paragraphs above -- correctly rejected by the design below, incorrectly accepted by
    two earlier, broader versions of this function; see its git history)."""
    headings = _headings(lines)
    enclosing = _enclosing_heading(lines, line_idx, headings)
    if enclosing is None:
        # line_idx precedes the file's first heading entirely -- no heading text to
        # check; fall back to just the block line_idx sits in.
        bound = headings[0][0] if headings else len(lines)
        for b_start, b_end in _blocks(lines, 0, bound):
            if b_start <= line_idx < b_end:
                return "\n".join(lines[b_start:b_end])
        return ""
    h_idx, h_level = enclosing
    parts = [_own_context_block(lines, h_idx, line_idx, headings)]
    parent = None
    for other_idx, other_level in headings:
        if other_idx >= h_idx:
            break
        if other_level < h_level:
            parent = (other_idx, other_level)
    if parent:
        parts.append(_heading_and_first_block(lines, parent[0], headings))
    return "\n".join("\n".join(p) for p in parts)


def build_retracted_patterns(retracted_figures: list) -> list:
    patterns = []
    for lit in retracted_figures:
        escaped = re.escape(lit)
        patterns.append((lit, re.compile(rf"(?<![\d.,]){escaped}(?![\d.,])")))
    return patterns


def check_retraction_guard(
    root: Path, registry: dict, verbose: bool
) -> tuple:
    """Returns (findings, retracted_mentions) where retracted_mentions is a set of
    (file_rel, line_no) for every retracted-figure occurrence found anywhere in the
    registry-enforced scan (used to suppress double-reporting in check 2)."""
    findings: list = []
    retracted_mentions: set = set()

    retracted_figures = registry["retracted_figures"]
    context_keywords = [k.lower() for k in registry.get("retraction_context_keywords", [])]
    exemptions = registry.get("retraction_exemptions", [])
    patterns = build_retracted_patterns(retracted_figures)

    files = expand_globs(root, RETRACTION_SCAN_GLOBS)
    for path in files:
        rel = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for lit, pat in patterns:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                line_idx = line_no - 1
                matched_line = lines[line_idx] if line_idx < len(lines) else ""
                retracted_mentions.add((rel, line_no))

                # 1) same-line self-marking (cheapest, most common case).
                if any(kw in matched_line.lower() for kw in context_keywords):
                    if verbose:
                        print(f"  [retraction ok/same-line] {rel}:{line_no} '{lit}'")
                    continue

                # 2) innermost heading-bounded section (+ immediate parent's own
                #    preamble -- see _retraction_context_text) carries retraction
                #    language.
                section_text = _retraction_context_text(lines, line_idx).lower()
                if any(kw in section_text for kw in context_keywords):
                    if verbose:
                        print(f"  [retraction ok/section] {rel}:{line_no} '{lit}'")
                    continue

                # 3) explicit, reasoned registry exemption (coincidental digit collision
                #    with an unrelated, currently-valid number -- NOT the retracted claim).
                exempted = False
                for ex in exemptions:
                    if ex.get("file") == rel and ex.get("match_substring", "") in matched_line:
                        exempted = True
                        if not ex.get("reason"):
                            findings.append(
                                Finding(
                                    "retraction-exemption-missing-reason",
                                    rel,
                                    line_no,
                                    matched_line.strip(),
                                    "retraction_exemptions entry has no 'reason' field",
                                )
                            )
                        elif verbose:
                            print(f"  [retraction ok/exemption] {rel}:{line_no} '{lit}'")
                        break
                if exempted:
                    continue

                findings.append(
                    Finding(
                        "retracted-figure-unmarked",
                        rel,
                        line_no,
                        matched_line.strip(),
                        f"'{lit}' is a retracted figure and this occurrence is not inside a "
                        "block marked retracted/superseded/historical, and has no registry "
                        "exemption",
                    )
                )
    return findings, retracted_mentions


# --------------------------------------------------------------------------------------
# Check 2/3 -- live claim registry + cross-file agreement
# --------------------------------------------------------------------------------------

# Ratio multiplier: "3.43x", "0.88×". Trailing `\b` would silently drop the unicode
# "×" form (a non-word character has no word-boundary after it), which would leave
# every "N.NN×" figure in scanned prose unextracted and therefore unpoliced -- the
# exact drift class this gate exists to catch. `(?!\w)` is equivalent for ascii "x"
# and correct for "×".
RATIO_RE = re.compile(r"(?<![\d.,])(\d+(?:,\d{3})*\.\d+)\s*[x×](?!\w)")

# Throughput figures: "321.0 tok/s", "93.6 -> 321.0 tok/s", "1514.1 +/- 198.9 tok/s"
TOKS_RE = re.compile(
    r"(?P<a>\d+(?:,\d{3})*\.\d+)"
    r"(?:\s*(?:→|->)\s*(?P<b>\d+(?:,\d{3})*\.\d+))?"
    r"(?:\s*(?:±|\+/-)\s*(?P<pm>\d+(?:,\d{3})*\.\d+))?"
    r"\s*tok/s"
)

# Signed or approximate percentages: "+57.3%", "-3.0%", "~12%", or an unsigned percentage
# whose preceding word is "by"/"within" ("collapses prefill by 47%"). Deliberately excludes
# other bare/unsigned percentages (e.g. external CPU-load "236%") -- see docs/CLAIMS.md
# "Scope" -- those aren't performance claims about this project's own results.
PCT_RE = re.compile(r"(?:(?P<sign>[+\-~])\s*)?(?P<num>\d+(?:\.\d+)?)\s*%")
PCT_CONTEXT_RE = re.compile(r"\b(by|within)\s*$", re.IGNORECASE)

# Dispatch hit counts stated inline in prose: "996 lldb hits", "51,214 NEON hits".
HITS_INLINE_RE = re.compile(r"(?<![\d.,])(\d+(?:,\d{3})*)\s+(?:[A-Za-z2/_+-]+\s+){0,2}hits\b")

NUMBER_TOKEN_RE = re.compile(r"(?<![\d.,])\d+(?:,\d{3})*(?:\.\d+)?(?![\d.,])")

# Stripped out of a table row before token-scanning a "hits"/"tok/s" table: inline code
# spans (e.g. `ne11>=128`, a threshold constant, not a measurement) and GitHub issue/PR
# references (e.g. "#26547", an identifier, not a measurement).
CODE_SPAN_RE = re.compile(r"`[^`]*`")
ISSUE_REF_RE = re.compile(r"#\d+")


@dataclass
class LiveClaim:
    file: str
    line: int
    kind: str
    value_text: str
    context: str
    # Only claims pulled out of a "hits"/"tok/s"-headed table cell (extract_table_unit_
    # claims) are eligible for the blanket Tier-2 JSON-backing fallback: those cells are
    # semantically *transcriptions* of a results/*.json row. A bare decimal floating in
    # prose (a ratio, a derived percentage, a headline throughput mention) must always be
    # explicitly registered -- Tier 2 alone is too coarse (1-decimal rounding buckets
    # collide with unrelated JSON leaves often enough to rubber-stamp a wrong number; see
    # docs/CLAIMS.md "Why prose numbers require Tier 1").
    table_derived: bool = False


def extract_table_unit_claims(rel: str, lines: list) -> list:
    """Some markdown tables put the unit word only in the header, not each cell (e.g. a
    'hits (SME2 / other)' column holding bare '996 / 0' cells, or a 'decode tok/s' column
    holding a bare '48.0' cell). Walk contiguous '|'-led blocks and, for any block whose
    header row contains 'hits' or 'tok/s', pull every number out of each data row (after
    stripping inline code spans and GitHub issue refs, which are not measurements)."""
    claims: list = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].lstrip().startswith("|"):
            block_start = i
            j = i
            while j < n and lines[j].lstrip().startswith("|"):
                j += 1
            header = lines[block_start].lower()
            kind = "hits" if "hits" in header else ("toks" if "tok/s" in header else None)
            if kind:
                # rows: header, separator (---), then data rows
                for r in range(block_start + 2, j):
                    cleaned = ISSUE_REF_RE.sub("", CODE_SPAN_RE.sub("", lines[r]))
                    for m in NUMBER_TOKEN_RE.finditer(cleaned):
                        claims.append(
                            LiveClaim(rel, r + 1, kind, m.group(0), lines[r].strip(), table_derived=True)
                        )
            i = j
        else:
            i += 1
    return claims


def extract_live_claims(root: Path, files: list) -> list:
    claims: list = []
    for path in files:
        rel = str(path.relative_to(root))
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        for lineno, line in enumerate(lines, start=1):
            for m in RATIO_RE.finditer(line):
                claims.append(LiveClaim(rel, lineno, "ratio", m.group(1), line.strip()))
            for m in TOKS_RE.finditer(line):
                for g in ("a", "b", "pm"):
                    v = m.group(g)
                    if v:
                        claims.append(LiveClaim(rel, lineno, "toks", v, line.strip()))
            for m in PCT_RE.finditer(line):
                sign = m.group("sign")
                prefix = line[: m.start()]
                if sign in ("+", "-", "~"):
                    claims.append(
                        LiveClaim(rel, lineno, "pct", sign + m.group("num"), line.strip())
                    )
                elif PCT_CONTEXT_RE.search(prefix):
                    claims.append(
                        LiveClaim(rel, lineno, "pct", m.group("num"), line.strip())
                    )
            for m in HITS_INLINE_RE.finditer(line):
                claims.append(LiveClaim(rel, lineno, "hits", m.group(1), line.strip()))

        claims.extend(extract_table_unit_claims(rel, lines))
    return claims


def collect_json_backed_numbers(root: Path, registry: dict) -> set:
    numbers: set = set()
    for path in expand_globs(root, registry.get("json_backed_globs", JSON_BACKED_GLOBS)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        numbers.update(iter_numeric_leaves(data))
    return numbers


def is_json_backed(value_text: str, json_numbers: set) -> bool:
    v = parse_float(value_text.lstrip("+-~"))
    dp = decimals_in(value_text.lstrip("+-~"))
    for jv in json_numbers:
        if round(jv, dp) == round(v, dp):
            return True
    return False


def verify_compute(value_text: str, compute: dict) -> Optional[str]:
    """Arithmetic self-check for a derived claim (a ratio or percentage computed from two
    other numbers already printed in the same source, e.g. a decomposition table). Returns
    None if the claim's stated value matches the computation, else an error string.
    Deliberately NOT eval() -- op is one of a small fixed set, never an arbitrary string."""
    op = compute.get("op")
    a = compute.get("numerator")
    b = compute.get("denominator")
    if a is None or b is None or op is None:
        return f"compute block missing op/numerator/denominator: {compute}"
    if op == "ratio":
        result = a / b
    elif op == "percent_drop":
        result = (1 - b / a) * 100
    elif op == "percent_rise":
        result = (b / a - 1) * 100
    else:
        return f"unknown compute op: {op}"
    dp = decimals_in(value_text.lstrip("+-~"))
    declared = parse_float(value_text.lstrip("+-~"))
    if round(result, dp) != round(declared, dp):
        return f"{op}({a}, {b}) = {result:.4f}, registry claims {value_text}"
    return None


def build_claims_index(root: Path, registry: dict) -> tuple:
    """Validate each hand-curated claim against its cited source, and return a dict
    mapping normalized value_text -> claim entry for the registry-membership check."""
    findings: list = []
    index: dict = {}
    for claim in registry["claims"]:
        cid = claim.get("id", "<no id>")
        value_text = claim.get("value_text")
        source_file = claim.get("source_file")
        if not value_text or not source_file:
            findings.append(
                Finding("registry-schema", CLAIMS_DOC, 0, cid, "claim missing value_text/source_file")
            )
            continue
        src_path = root / source_file
        if not src_path.is_file():
            findings.append(
                Finding("registry-source-missing", CLAIMS_DOC, 0, cid, f"source_file not found: {source_file}")
            )
            continue
        src_text = src_path.read_text(encoding="utf-8")
        compute = claim.get("compute")
        json_path = claim.get("source_json_path")
        if compute:
            err = verify_compute(value_text, compute)
            if err:
                findings.append(Finding("registry-compute-mismatch", CLAIMS_DOC, 0, cid, err))
        elif json_path:
            json_file = claim.get("source_json_file", source_file)
            try:
                data = json.loads((root / json_file).read_text(encoding="utf-8"))
                resolved = resolve_json_path(data, json_path)
                dp = decimals_in(value_text)
                if round(float(resolved), dp) != round(parse_float(value_text), dp):
                    findings.append(
                        Finding(
                            "registry-value-mismatch",
                            CLAIMS_DOC,
                            0,
                            cid,
                            f"{json_file}#{json_path} = {resolved}, registry says {value_text}",
                        )
                    )
            except Exception as e:  # noqa: BLE001 - report, don't crash the gate
                findings.append(
                    Finding("registry-json-path-error", CLAIMS_DOC, 0, cid, f"{json_path}: {e}")
                )
        else:
            if normalize_number(value_text).lstrip("+-~") not in normalize_number(src_text):
                findings.append(
                    Finding(
                        "registry-value-not-in-source",
                        CLAIMS_DOC,
                        0,
                        cid,
                        f"'{value_text}' not found verbatim in {source_file}",
                    )
                )
        for alias in [value_text] + list(claim.get("aliases", [])):
            key = normalize_number(alias).lstrip("+-~")
            index[key] = claim
    return index, findings


def check_live_claims(
    root: Path,
    registry: dict,
    claims_index: dict,
    json_numbers: set,
    retracted_mentions: set,
    verbose: bool,
) -> list:
    findings: list = []
    non_claim = registry.get("non_claim_exemptions", [])
    files = expand_globs(root, REGISTRY_SCAN_GLOBS)
    live = extract_live_claims(root, files)

    for lc in live:
        if (lc.file, lc.line) in retracted_mentions:
            # Already adjudicated (pass or fail) by the retraction guard.
            continue

        key = normalize_number(lc.value_text).lstrip("+-~")

        exempted = False
        for ex in non_claim:
            if ex.get("file") == lc.file and ex.get("match_substring", "") in lc.context:
                exempted = True
                break
        if exempted:
            if verbose:
                print(f"  [non-claim exempt] {lc.file}:{lc.line} '{lc.value_text}' ({lc.kind})")
            continue

        if key in claims_index:
            if verbose:
                print(f"  [registered] {lc.file}:{lc.line} '{lc.value_text}' ({lc.kind})")
            continue

        # Percentages are always *derived* (never a raw JSON leaf), so a numeric
        # coincidence against an unrelated JSON value (e.g. a thread count) would be a
        # meaningless pass -- percentages must be explicitly registered (substring- or
        # compute-verified), never accepted via blanket JSON-backing. Ratio/throughput
        # figures may fall back to Tier 2, but every cross-file or headline number is
        # *also* hand-entered in the registry regardless of whether Tier 2 would already
        # pass it (see docs/CLAIMS.md "Why headline numbers are registered anyway") --
        # so a 1-decimal rounding coincidence against an unrelated JSON leaf is never the
        # only thing standing behind a number that actually matters.
        if lc.kind != "pct" and is_json_backed(lc.value_text, json_numbers):
            if verbose:
                print(f"  [json-backed] {lc.file}:{lc.line} '{lc.value_text}' ({lc.kind})")
            continue

        findings.append(
            Finding(
                "unregistered-claim",
                lc.file,
                lc.line,
                lc.context,
                f"'{lc.value_text}' ({lc.kind}) is not in docs/CLAIMS.md's registry and does "
                "not match any numeric value committed under results/**/*.json",
            )
        )
    return findings


# --------------------------------------------------------------------------------------
# Check 4 -- manifest hash/byte-size consistency
#
# Origin: the 2026-08-06 model-manifest provenance incident, where scripts/models.txt
# paired one file's real sha256 with a different file's byte size and historical role.
# This check is narrow and deliberately conservative -- it does NOT try to police every
# size mention in prose (docs/CLAIMS.md already documents that file sizes are out of
# scope for check 2/3, and rightly so: "337 MB" vs "336.66 MiB" is a legitimate rounding
# choice, not a defect). It only catches the one thing that IS unambiguous: an exact,
# comma-grouped byte count stated close to a specific sha256 in scripts/models.txt, where
# a committed results/**/*.json ledger (written by tools/verify_dispatch.py, which embeds
# model_sha256 + model_bytes since the same incident) records a *different* byte count
# for that same hash.
# --------------------------------------------------------------------------------------

MANIFEST_PATH = "scripts/models.txt"
HASH_RE = re.compile(r"\b[0-9a-f]{64}\b")
EXACT_BYTE_COUNT_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})+)\s*-?\s*bytes?\b")
# A hash/byte-count pair is only trusted when they sit within this many characters of
# each other AND are each other's nearest match -- see tools/check_claims.py's own test
# invocation (or the incident writeup) for why: on the real manifest text, genuine
# same-sentence pairings land under ~150 chars apart, while unrelated hash/byte-count
# mentions elsewhere in the same file land at 400+ chars -- a wide, safe margin.
MANIFEST_PAIR_MAX_DISTANCE = 150


def collect_manifest_hash_byte_claims(root: Path) -> dict:
    """{sha256: (byte_count, line_no)} for unambiguous hash/byte-count pairs found in
    scripts/models.txt. A pair is trusted only when the hash and the byte count are
    mutual nearest neighbors (by character offset) within MANIFEST_PAIR_MAX_DISTANCE --
    this deliberately drops any hash or byte count that has no close, unambiguous
    partner (e.g. a row whose notes only give an approximate "(676 MB)" size) rather
    than guess.
    """
    path = root / MANIFEST_PATH
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    hashes = [(m.start(), m.group(0)) for m in HASH_RE.finditer(text)]
    counts = [(m.start(), int(m.group(1).replace(",", ""))) for m in EXACT_BYTE_COUNT_RE.finditer(text)]
    if not hashes or not counts:
        return {}

    claims: dict = {}
    for hpos, h in hashes:
        bpos, n = min(counts, key=lambda c: abs(c[0] - hpos))
        if abs(bpos - hpos) > MANIFEST_PAIR_MAX_DISTANCE:
            continue
        # Mutual-nearest check: this byte count's own nearest hash must be this hash,
        # otherwise it belongs to some other mention and pairing it here would guess.
        nearest_hash_pos, _ = min(hashes, key=lambda hh: abs(hh[0] - bpos))
        if nearest_hash_pos != hpos:
            continue
        line_no = text.count("\n", 0, hpos) + 1
        claims[h] = (n, line_no)
    return claims


def collect_ledger_hash_bytes(root: Path) -> dict:
    """{sha256: (byte_count, rel_path)} from every committed results/**/*.json object
    that carries both model_sha256 and model_bytes (only ledgers written since the
    2026-08-06 fix to tools/verify_dispatch.py have these keys; older ledgers are
    silently skipped, not treated as a contradiction).
    """
    out: dict = {}
    for jf in sorted((root / "results").rglob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("model_sha256"), str) and isinstance(
            data.get("model_bytes"), int
        ):
            out[data["model_sha256"]] = (data["model_bytes"], str(jf.relative_to(root)))
    return out


def check_manifest_hash_byte_consistency(root: Path, verbose: bool) -> list:
    findings: list = []
    manifest_claims = collect_manifest_hash_byte_claims(root)
    ledger_truth = collect_ledger_hash_bytes(root)
    for h, (claimed_bytes, line_no) in sorted(manifest_claims.items(), key=lambda kv: kv[1][1]):
        if h not in ledger_truth:
            if verbose:
                print(f"  [no ledger evidence] {MANIFEST_PATH}:{line_no} sha256 {h[:12]}... "
                      f"claimed {claimed_bytes:,} bytes (nothing committed under results/ to check against)")
            continue
        real_bytes, ledger_file = ledger_truth[h]
        if real_bytes == claimed_bytes:
            if verbose:
                print(f"  [consistent] {MANIFEST_PATH}:{line_no} sha256 {h[:12]}... "
                      f"{claimed_bytes:,} bytes matches {ledger_file}")
            continue
        findings.append(
            Finding(
                "manifest-hash-byte-mismatch",
                MANIFEST_PATH,
                line_no,
                f"sha256 {h} stated as {claimed_bytes:,} bytes",
                f"{ledger_file} independently records the same sha256 as {real_bytes:,} bytes -- "
                "a manifest claim contradicts committed measurement evidence for its own hash "
                "(this is the exact defect class the 2026-08-06 model-manifest incident found).",
            )
        )
    return findings


# --------------------------------------------------------------------------------------
# Report + main
# --------------------------------------------------------------------------------------


def print_report(findings: list) -> None:
    by_check: dict = {}
    for f in findings:
        by_check.setdefault(f.check, []).append(f)
    for check, items in by_check.items():
        print(f"\n=== {check} ({len(items)}) ===")
        for f in items:
            loc = f"{f.file}:{f.line}" if f.line else f.file
            print(f"  {loc}")
            print(f"    text:   {f.text[:160]}")
            print(f"    reason: {f.reason}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each pass, not just failures")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()

    try:
        registry = load_registry(root)
    except RegistryError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 1

    all_findings: list = []

    if args.verbose:
        print("== Check 1: retraction guard ==")
    retraction_findings, retracted_mentions = check_retraction_guard(root, registry, args.verbose)
    all_findings.extend(retraction_findings)

    if args.verbose:
        print("\n== Building claims index from docs/CLAIMS.md ==")
    claims_index, registry_findings = build_claims_index(root, registry)
    all_findings.extend(registry_findings)

    if args.verbose:
        print("\n== Collecting JSON-backed numbers from results/**/*.json ==")
    json_numbers = collect_json_backed_numbers(root, registry)
    if args.verbose:
        print(f"  {len(json_numbers)} numeric leaves loaded")

    if args.verbose:
        print("\n== Check 2/3: live claim registry + cross-file agreement ==")
    live_findings = check_live_claims(
        root, registry, claims_index, json_numbers, retracted_mentions, args.verbose
    )
    all_findings.extend(live_findings)

    if args.verbose:
        print("\n== Check 4: manifest hash/byte-size consistency ==")
    manifest_findings = check_manifest_hash_byte_consistency(root, args.verbose)
    all_findings.extend(manifest_findings)

    if all_findings:
        print_report(all_findings)
        print(f"\nFAIL: {len(all_findings)} claim-consistency finding(s). See docs/CLAIMS.md.")
        return 1

    print("OK: every scanned numeric claim is registered/JSON-backed; no retracted figure "
          "appears unmarked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
