// SPDX-License-Identifier: Apache-2.0
//
// Polygraph dashboard. (Previously published as Arm Dispatch Ledger.)
//
// Renders ONLY data/manifest.json + the files it points at, both written by
// .github/workflows/pages.yml at publish time from the committed results/
// directory. No cross-origin fetches. If a file is missing (e.g. only the
// Apple M4 Max ledger has been published so far, no Linux one yet), the
// affected section renders an honest empty state instead of guessing.
//
// No build step, no framework, no chart library: this file talks to the DOM
// and draws SVG by hand.

(function () {
  "use strict";

  // ----------------------------------------------------------------------
  // Theme toggle (prefers-color-scheme drives the default; this just lets a
  // reader override it, persisted in localStorage).
  // ----------------------------------------------------------------------
  function initTheme() {
    var root = document.documentElement;
    var btn = document.getElementById("theme-toggle");
    var stored = null;
    try {
      stored = window.localStorage.getItem("adl-theme");
    } catch (e) {
      /* localStorage unavailable (private browsing etc.) -- fall back to OS preference only */
    }
    if (stored === "light" || stored === "dark") {
      root.setAttribute("data-theme", stored);
    }
    function currentIsDark() {
      var explicit = root.getAttribute("data-theme");
      if (explicit === "dark") return true;
      if (explicit === "light") return false;
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    function sync() {
      var dark = currentIsDark();
      if (btn) {
        btn.setAttribute("aria-pressed", String(dark));
        btn.querySelector(".theme-toggle-icon").textContent = dark ? "☾" : "☀";
      }
    }
    if (btn) {
      btn.addEventListener("click", function () {
        var next = currentIsDark() ? "light" : "dark";
        root.setAttribute("data-theme", next);
        try {
          window.localStorage.setItem("adl-theme", next);
        } catch (e) {
          /* ignore */
        }
        sync();
      });
    }
    sync();
  }

  // ----------------------------------------------------------------------
  // Small helpers
  // ----------------------------------------------------------------------
  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "text") {
          node.textContent = attrs[k];
        } else if (k === "html") {
          // Only used for a handful of trusted, hardcoded strings below
          // (never with fetched data) -- see call sites.
          node.innerHTML = attrs[k];
        } else {
          node.setAttribute(k, attrs[k]);
        }
      });
    }
    (children || []).forEach(function (c) {
      if (c) node.appendChild(c);
    });
    return node;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        node.setAttribute(k, attrs[k]);
      });
    }
    return node;
  }

  function fmt(n, digits) {
    if (n === null || n === undefined || Number.isNaN(n)) return "–";
    var d = digits === undefined ? 1 : digits;
    return Number(n).toLocaleString(undefined, {
      minimumFractionDigits: d,
      maximumFractionDigits: d,
    });
  }

  // Formats a ratio (e.g. 2.1519...) as "2.15x". Every ratio the headline
  // band shows is computed here from raw tok/s values in headline.json --
  // never stored pre-computed, so the page cannot drift from that file.
  function fmtRatio(n, digits) {
    if (n === null || n === undefined || Number.isNaN(n) || !isFinite(n)) return "–";
    var d = digits === undefined ? 2 : digits;
    return Number(n).toFixed(d) + "x";
  }

  function fetchJSON(path) {
    return fetch(path, { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status + " for " + path);
        return r.json();
      })
      .catch(function (err) {
        console.warn("[polygraph] could not load", path, err.message);
        return null;
      });
  }

  // ----------------------------------------------------------------------
  // Manifest
  // ----------------------------------------------------------------------
  var EMPTY_MANIFEST = {
    dispatch_ledgers: [],
    bench: [],
    crossover: [],
    figures: [],
    other_json: [],
  };

  function loadManifest() {
    return fetchJSON("data/manifest.json").then(function (m) {
      return m || EMPTY_MANIFEST;
    });
  }

  // ----------------------------------------------------------------------
  // Verdict classification -- shared between the hero table and the summary
  // strip so the counts always agree with the rows a reader can see.
  // ----------------------------------------------------------------------
  function classifyVerdict(verdict) {
    var v = (verdict || "").toUpperCase();
    if (v.indexOf("HYBRID") !== -1) {
      return { key: "hybrid", label: "HYBRID", icon: "◐", cls: "hybrid" };
    }
    if (v.indexOf("FALLBACK") !== -1 || v.indexOf("MISMATCH") !== -1) {
      return { key: "mismatch", label: "MISMATCH", icon: "✗", cls: "mismatch" };
    }
    if (v.indexOf("DISPATCH") !== -1 || v.indexOf("MATCH") !== -1) {
      return { key: "match", label: "MATCH", icon: "✓", cls: "match" };
    }
    return { key: "unknown", label: v || "UNKNOWN", icon: "?", cls: "unknown" };
  }

  function badge(verdictRaw) {
    var c = classifyVerdict(verdictRaw);
    return el(
      "span",
      { class: "badge badge-" + c.cls, title: verdictRaw || "unknown" },
      [
        el("span", { "aria-hidden": "true", text: c.icon }),
        el("span", { text: c.label + (verdictRaw && c.label !== verdictRaw ? " (" + verdictRaw + ")" : "") }),
      ]
    );
  }

  // Parses "... n_threads = 1 (n_threads_batch = 1) / 16 | CPU : ..." to
  // recover the physical/logical core count llama.cpp itself detected --
  // i.e. the thread count a user gets with no -t flag at all. Derived from
  // the platform's own log line, never hardcoded.
  function defaultThreadsFromSystemInfo(line) {
    if (!line) return null;
    var m = /\/\s*(\d+)\s*\|/.exec(line);
    return m ? parseInt(m[1], 10) : null;
  }

  // ----------------------------------------------------------------------
  // Headline band -- results/headline.json -> the hero results band above
  // the dispatch table. Only raw tok/s numbers live in that file; every
  // ratio/percentage below is computed here, so a future re-measurement
  // only has to update headline.json's raw numbers and every displayed
  // ratio recomputes itself.
  // ----------------------------------------------------------------------
  function loadHeadline() {
    return fetchJSON("data/results/headline.json");
  }

  // One labelled group of horizontal bars, scaled to the group's own max
  // value (decode and prefill are different orders of magnitude, so each
  // group gets its own scale). Every bar pairs its colour with an icon AND
  // a text note -- never colour alone -- per this project's accessibility
  // rule.
  function buildHeadlineBarGroup(title, items) {
    var max = 0;
    items.forEach(function (it) {
      if (typeof it.value === "number" && it.value > max) max = it.value;
    });
    if (max <= 0) max = 1;

    var group = el("div", { class: "headline-bar-group" }, [
      el("p", { class: "headline-bar-group-title", text: title }),
    ]);

    items.forEach(function (it) {
      var pct = Math.max(2, (it.value / max) * 100);
      var row = el("div", { class: "headline-bar-row headline-bar-" + it.cls }, [
        el("span", { class: "headline-bar-label", text: it.label }),
        el("span", { class: "headline-bar-track" }, [
          el("span", { class: "headline-bar-fill", style: "width:" + pct.toFixed(1) + "%" }),
        ]),
        el("span", { class: "headline-bar-value" }, [
          el("span", { class: "headline-bar-icon", "aria-hidden": "true", text: it.icon }),
          el("span", { text: " " + fmt(it.value, 1) + " tok/s" + (it.note ? " — " + it.note : "") }),
        ]),
      ]);
      group.appendChild(row);
    });
    return group;
  }

  function renderHeadline(data) {
    var band = document.getElementById("headline-band");
    var empty = document.getElementById("headline-empty");
    var grid = document.getElementById("headline-grid");
    var contrast = band ? band.querySelector(".headline-contrast") : null;
    var honesty = document.getElementById("headline-honesty");

    var auto = (data && data.autodefaults) || {};
    var dec = auto.decode || {};
    var pre = auto.prefill || {};
    var decomp = (data && data.decomposition) || {};

    var haveDecode = typeof dec.baseline_no_flags === "number" && typeof dec.autodefault_no_flags === "number";
    var havePrefill = typeof pre.baseline_no_flags === "number" && typeof pre.autodefault_no_flags === "number";
    var haveNaive = havePrefill && typeof pre.naive_t2_workaround === "number";
    var haveDecomp =
      typeof decomp.default_threads_sme_on === "number" &&
      typeof decomp.default_threads_sme_off === "number" &&
      typeof decomp.t2_sme_on === "number" &&
      typeof decomp.t2_sme_off === "number";

    if (!data || (!haveDecode && !havePrefill)) {
      if (empty) empty.hidden = false;
      if (grid) grid.hidden = true;
      if (contrast) contrast.hidden = true;
      if (honesty) honesty.hidden = true;
      return;
    }
    if (empty) empty.hidden = true;

    // ---- Decode card ----
    if (haveDecode) {
      var decodeRatio = dec.autodefault_no_flags / dec.baseline_no_flags;
      document.getElementById("headline-decode-ratio").textContent = fmtRatio(decodeRatio, 2) + " faster decode, zero flags";
      document.getElementById("headline-decode-detail").textContent =
        fmt(dec.baseline_no_flags, 1) + " → " + fmt(dec.autodefault_no_flags, 1) + " tok/s (round-robin interleaved, n=9).";
      var decodeNote = document.getElementById("headline-decode-note");
      if (typeof dec.hand_tuned_t2 === "number") {
        var vsHandTuned = dec.autodefault_no_flags / dec.hand_tuned_t2;
        decodeNote.textContent =
          "Matches the hand-tuned -t 2 ceiling (" +
          fmt(dec.hand_tuned_t2, 1) +
          " tok/s) within noise (" +
          fmtRatio(vsHandTuned, 3) +
          ") — automatically, with no flags passed.";
      }
    }

    // ---- Prefill card ----
    if (havePrefill) {
      var prefillRatio = pre.autodefault_no_flags / pre.baseline_no_flags;
      document.getElementById("headline-prefill-ratio").textContent = "Prefill unchanged (" + fmtRatio(prefillRatio, 3) + ")";
      document.getElementById("headline-prefill-detail").textContent =
        fmt(pre.baseline_no_flags, 1) + " → " + fmt(pre.autodefault_no_flags, 1) + " tok/s — within measurement noise.";
      document.getElementById("headline-prefill-note").textContent =
        "The patch only changes generation (decode) threads; --threads-batch / prefill is left alone.";
    }

    // ---- Contrast bars: the decode win, and why the naive fix breaks prefill ----
    var barsContainer = document.getElementById("headline-bars");
    var introEl = document.getElementById("headline-contrast-intro");
    if (barsContainer) barsContainer.textContent = "";

    if (haveDecode) {
      var decodeItems = [
        { label: "Baseline, no flags", value: dec.baseline_no_flags, cls: "neutral", icon: "–" },
      ];
      if (typeof dec.hand_tuned_t2 === "number") {
        decodeItems.push({ label: "Hand-tuned -t 2 (breaks prefill)", value: dec.hand_tuned_t2, cls: "bad", icon: "⚠" });
      }
      decodeItems.push({ label: "Autodefault patch, zero flags", value: dec.autodefault_no_flags, cls: "good", icon: "✓" });
      barsContainer.appendChild(buildHeadlineBarGroup("Decode throughput (tok/s)", decodeItems));
    }

    if (haveNaive) {
      var prefillItems = [
        { label: "Baseline, no flags", value: pre.baseline_no_flags, cls: "neutral", icon: "–" },
        { label: "Autodefault patch, zero flags", value: pre.autodefault_no_flags, cls: "good", icon: "✓", note: "unchanged" },
        { label: "Naive workaround: pass -t 2, no -tb", value: pre.naive_t2_workaround, cls: "bad", icon: "✗", note: "collapses" },
      ];
      barsContainer.appendChild(buildHeadlineBarGroup("Prefill throughput (tok/s)", prefillItems));

      var naiveRatio = pre.naive_t2_workaround / pre.baseline_no_flags;
      var collapsePct = Math.round((1 - naiveRatio) * 100);
      var prefillRatioForIntro = pre.autodefault_no_flags / pre.baseline_no_flags;
      if (introEl) {
        introEl.textContent =
          "Passing -t 2 alone reaches the same decode ceiling, but stock llama.cpp's --threads-batch " +
          "silently inherits --threads — so that \"obvious\" fix also collapses prefill by " +
          collapsePct +
          "% (" +
          fmt(pre.baseline_no_flags, 1) +
          " → " +
          fmt(pre.naive_t2_workaround, 1) +
          " tok/s, " +
          fmtRatio(naiveRatio, 3) +
          "). This patch is phase-aware — it changes only generation threads — so prefill stays at " +
          fmtRatio(prefillRatioForIntro, 3) +
          ", unchanged within noise.";
      }
    } else if (introEl) {
      introEl.textContent = "Naive-workaround comparison data unavailable.";
    }

    // ---- Honesty check: how much is SME2 vs. generic thread tuning ----
    var honestyText = document.getElementById("headline-honesty-text");
    if (haveDecomp && honestyText) {
      var total = decomp.t2_sme_on / decomp.default_threads_sme_on;
      var threadTuningAlone = decomp.t2_sme_off / decomp.default_threads_sme_off;
      var sme2AtT2 = decomp.t2_sme_on / decomp.t2_sme_off;
      var sme2AtDefault = decomp.default_threads_sme_on / decomp.default_threads_sme_off;
      honestyText.textContent =
        "Of the " +
        fmtRatio(total, 2) +
        " you get pulling both levers together (fewer threads + SME2), " +
        fmtRatio(threadTuningAlone, 2) +
        " is generic thread-oversubscription avoidance alone (SME2 forced off throughout — a well-known " +
        "Apple Silicon effect, not this project's discovery). SME2's own contribution at the tuned thread " +
        "count is a smaller, real " +
        fmtRatio(sme2AtT2, 2) +
        ". At the untuned default, SME2 actively hurts (" +
        fmtRatio(sme2AtDefault, 2) +
        ").";
    } else if (honestyText) {
      honestyText.textContent = "Decomposition data unavailable.";
    }
  }

  // ----------------------------------------------------------------------
  // Dispatch ledgers -> hero table + summary strip
  // ----------------------------------------------------------------------
  function loadDispatchLedgers(paths) {
    if (!paths.length) return Promise.resolve([]);
    return Promise.all(
      paths.map(function (p) {
        return fetchJSON("data/" + p).then(function (doc) {
          return doc ? { path: p, doc: doc } : null;
        });
      })
    ).then(function (results) {
      return results.filter(Boolean);
    });
  }

  function flattenConfigs(ledgers) {
    var rows = [];
    ledgers.forEach(function (ledger) {
      var configs = (ledger.doc && ledger.doc.configs) || [];
      configs.forEach(function (c) {
        rows.push(c);
      });
    });
    return rows;
  }

  function renderHeroTable(configs) {
    var tbody = document.getElementById("hero-table-body");
    var empty = document.getElementById("hero-table-empty");
    tbody.textContent = "";

    if (!configs.length) {
      empty.hidden = false;
      tbody.appendChild(el("tr", {}, [el("td", { colspan: "7", text: "No data." })]));
      return { total: 0, match: 0, mismatch: 0, hybrid: 0, unknown: 0, defaultThreadsByPlatform: {} };
    }
    empty.hidden = true;

    var sorted = configs.slice().sort(function (a, b) {
      var pa = a.platform || "",
        pb = b.platform || "";
      if (pa !== pb) return pa < pb ? -1 : 1;
      var wa = a.workload || "",
        wb = b.workload || "";
      if (wa !== wb) return wa < wb ? -1 : 1;
      return (a.threads || 0) - (b.threads || 0);
    });

    var counts = { match: 0, mismatch: 0, hybrid: 0, unknown: 0 };
    var defaultThreadsByPlatform = {};
    var frag = document.createDocumentFragment();

    sorted.forEach(function (c) {
      var l2 = c.l2 || {};
      var l3 = c.l3 || {};
      var advertised = c.advertised_family || l2.sme_variant || (l2.primary_kernel_feature && l2.primary_kernel_feature.q4) || "–";
      var executed = l3.kernel_family_executed || "–";
      var hitsByFamily = l3.hits_by_family || {};
      var totalHits = typeof l3.total_hits === "number" ? l3.total_hits : Object.values(hitsByFamily).reduce(function (a, b) { return a + b; }, 0);
      var advKey = String(advertised).toLowerCase();
      var advHits = hitsByFamily[advKey] || 0;
      var otherHits = totalHits - advHits;
      var verdict = c.verdict || "";
      var cls = classifyVerdict(verdict);
      counts[cls.key] = (counts[cls.key] || 0) + 1;

      var dt = defaultThreadsFromSystemInfo(l2.system_info_line);
      if (dt && c.platform) {
        defaultThreadsByPlatform[c.platform] = dt;
      }

      var tr = el("tr", { class: "row-" + cls.cls }, [
        el("td", { text: c.platform || "–" }),
        el("td", { text: String(c.threads !== undefined ? c.threads : "–") }),
        el("td", { text: c.workload || "–" }),
        el("td", { text: String(advertised).toUpperCase() }),
        el("td", { text: String(executed).toUpperCase() }),
        el("td", { text: fmt(advHits, 0) + " / " + fmt(otherHits, 0) }),
        el("td", {}, [badge(verdict)]),
      ]);
      frag.appendChild(tr);
    });

    tbody.appendChild(frag);

    return {
      total: sorted.length,
      match: counts.match || 0,
      mismatch: counts.mismatch || 0,
      hybrid: counts.hybrid || 0,
      unknown: counts.unknown || 0,
      defaultThreadsByPlatform: defaultThreadsByPlatform,
    };
  }

  function renderSummaryStrip(stats) {
    document.getElementById("stat-total").textContent = stats.total ? String(stats.total) : "–";
    document.getElementById("stat-mismatch").textContent = stats.total ? String(stats.mismatch) : "–";
    document.getElementById("stat-hybrid").textContent = stats.total ? String(stats.hybrid) : "–";
    document.getElementById("stat-match").textContent = stats.total ? String(stats.match) : "–";

    var headline = document.getElementById("headline-sentence");
    if (!stats.total) {
      headline.textContent =
        "No dispatch-ledger JSON has been published to this site yet -- see results/dispatch-ledger-*.json in the repository.";
      return;
    }
    var pct = Math.round((stats.mismatch / stats.total) * 100);
    var sentence =
      stats.mismatch > 0
        ? stats.mismatch +
          " of " +
          stats.total +
          " verified configurations (" +
          pct +
          "%) advertise a kernel family that a debugger breakpoint proves never executed. " +
          stats.hybrid +
          " more only partially dispatch it (hybrid)."
        : "All " + stats.total + " verified configurations executed the kernel family they advertised.";
    headline.textContent = sentence;
  }

  // ----------------------------------------------------------------------
  // Bench JSON -> throughput crossover charts
  // ----------------------------------------------------------------------
  function loadBenchFiles(paths) {
    if (!paths.length) return Promise.resolve([]);
    return Promise.all(
      paths.map(function (p) {
        return fetchJSON("data/" + p).then(function (doc) {
          return doc ? { path: p, doc: doc } : null;
        });
      })
    ).then(function (results) {
      return results.filter(Boolean);
    });
  }

  // A bench-shaped dataset's row array: tools/bench.py writes it as
  // top-level "rows"; the sibling tools/crossover.py writes the equivalent
  // per-cell sweep as top-level "sweep_rows" (same row shape: phase,
  // threads, sme_mode, agg.median_ts). Accept either so results/crossover/
  // renders with the same chart, not just a raw-JSON link, the moment it
  // shows up.
  function rowsOf(doc) {
    return doc.rows || doc.sweep_rows || [];
  }

  // Groups a bench dataset's rows by phase, returning, for each phase, a
  // sorted-by-threads array of { threads, on, off } median tok/s (Q4_0 only
  // -- the only quant measured in this environment; rows with
  // not_available:true, e.g. Q8_0, are skipped rather than shown as zero).
  function seriesByPhase(doc) {
    var byPhase = {};
    rowsOf(doc).forEach(function (r) {
      if (r.not_available) return;
      if (r.quant && r.quant !== "Q4_0") return;
      if (!r.agg || typeof r.agg.median_ts !== "number") return;
      var phase = r.phase || "unknown";
      byPhase[phase] = byPhase[phase] || {};
      var t = r.threads;
      byPhase[phase][t] = byPhase[phase][t] || { threads: t, on: null, off: null };
      if (r.sme_mode === "on") byPhase[phase][t].on = r.agg.median_ts;
      else if (r.sme_mode === "off") byPhase[phase][t].off = r.agg.median_ts;
    });
    var out = {};
    Object.keys(byPhase).forEach(function (phase) {
      out[phase] = Object.values(byPhase[phase]).sort(function (a, b) {
        return a.threads - b.threads;
      });
    });
    return out;
  }

  var PHASE_LABELS = {
    decode: "Decode (token generation)",
    prefill: "Prefill",
    prefill_short: "Prefill — short prompt",
    prefill_long: "Prefill — long prompt",
  };

  function buildChartSvg(rows) {
    var W = 420,
      H = 240,
      padL = 52,
      padR = 14,
      padT = 14,
      padB = 34;
    var innerW = W - padL - padR;
    var innerH = H - padT - padB;

    var maxVal = 0;
    rows.forEach(function (r) {
      if (typeof r.on === "number") maxVal = Math.max(maxVal, r.on);
      if (typeof r.off === "number") maxVal = Math.max(maxVal, r.off);
    });
    maxVal = maxVal > 0 ? maxVal * 1.12 : 1;

    var n = rows.length;
    function x(i) {
      return n <= 1 ? padL + innerW / 2 : padL + (innerW * i) / (n - 1);
    }
    function y(v) {
      return padT + innerH - (innerH * v) / maxVal;
    }

    var svg = svgEl("svg", {
      viewBox: "0 0 " + W + " " + H,
      role: "img",
      "aria-label": "Line chart of tokens per second by thread count, SME2 path versus NEON forced",
    });

    // Gridlines + y labels (4 bands).
    var gridCount = 4;
    for (var g = 0; g <= gridCount; g++) {
      var gv = (maxVal * g) / gridCount;
      var gy = y(gv);
      svg.appendChild(
        svgEl("line", { x1: padL, x2: W - padR, y1: gy, y2: gy, stroke: "var(--grid-line)", "stroke-width": "1" })
      );
      var label = svgEl("text", {
        x: padL - 6,
        y: gy + 3,
        "text-anchor": "end",
        "font-size": "9",
        fill: "var(--text-muted)",
      });
      label.textContent = fmt(gv, 0);
      svg.appendChild(label);
    }

    // Axis baseline.
    svg.appendChild(
      svgEl("line", { x1: padL, x2: W - padR, y1: padT + innerH, y2: padT + innerH, stroke: "var(--border)", "stroke-width": "1" })
    );

    // X ticks (ordinal thread counts).
    rows.forEach(function (r, i) {
      var xi = x(i);
      svg.appendChild(
        svgEl("line", { x1: xi, x2: xi, y1: padT + innerH, y2: padT + innerH + 4, stroke: "var(--border)", "stroke-width": "1" })
      );
      var t = svgEl("text", {
        x: xi,
        y: padT + innerH + 16,
        "text-anchor": "middle",
        "font-size": "10",
        fill: "var(--text-muted)",
      });
      t.textContent = r.threads + "t";
      svg.appendChild(t);
    });

    function buildLine(key, colorVar, dash, marker) {
      var pts = [];
      rows.forEach(function (r, i) {
        if (typeof r[key] === "number") pts.push([x(i), y(r[key]), r[key]]);
      });
      if (pts.length < 1) return;
      if (pts.length > 1) {
        var d = pts.map(function (p, i) { return (i === 0 ? "M" : "L") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(" ");
        svg.appendChild(
          svgEl("path", {
            d: d,
            fill: "none",
            stroke: colorVar,
            "stroke-width": "2.25",
            "stroke-dasharray": dash || "",
          })
        );
      }
      pts.forEach(function (p) {
        var m;
        if (marker === "square") {
          m = svgEl("rect", { x: p[0] - 3.5, y: p[1] - 3.5, width: 7, height: 7, fill: colorVar });
        } else {
          m = svgEl("circle", { cx: p[0], cy: p[1], r: 4, fill: colorVar });
        }
        var title = svgEl("title", {});
        title.textContent = fmt(p[2], 1) + " tok/s";
        m.appendChild(title);
        svg.appendChild(m);
      });
    }

    buildLine("off", "var(--line-off)", "5 4", "square");
    buildLine("on", "var(--line-on)", "", "circle");

    return svg;
  }

  function buildChartTable(rows) {
    var table = el("table", { class: "chart-table" });
    table.appendChild(
      el("thead", {}, [
        el("tr", {}, [
          el("th", { text: "Threads" }),
          el("th", { text: "SME2 path (tok/s)" }),
          el("th", { text: "NEON forced (tok/s)" }),
        ]),
      ])
    );
    var tbody = el("tbody");
    rows.forEach(function (r) {
      tbody.appendChild(
        el("tr", {}, [
          el("td", { text: String(r.threads) }),
          el("td", { text: fmt(r.on, 1) }),
          el("td", { text: fmt(r.off, 1) }),
        ])
      );
    });
    table.appendChild(tbody);
    return table;
  }

  // Finds the first (ascending threads) point where NEON-forced ties or
  // beats the SME2 path, purely from the rows given -- never a hardcoded
  // thread count.
  function crossoverCaption(rows) {
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (typeof r.on === "number" && typeof r.off === "number" && r.off >= r.on) {
        return "NEON forced ties or overtakes the SME2 path at " + r.threads + " threads and above, in this data.";
      }
    }
    var hasBoth = rows.some(function (r) { return typeof r.on === "number" && typeof r.off === "number"; });
    return hasBoth
      ? "The SME2 path stays ahead of NEON forced at every thread count measured here."
      : "Not enough paired on/off measurements to determine a crossover.";
  }

  function renderChartCard(container, platformLabel, phase, rows) {
    var hasData = rows.some(function (r) { return typeof r.on === "number" || typeof r.off === "number"; });
    var card = el("div", { class: "chart-card" }, [
      el("h3", { text: PHASE_LABELS[phase] || phase }),
      el("p", { class: "chart-subtitle", text: platformLabel }),
    ]);
    if (!hasData) {
      card.appendChild(el("p", { class: "loading-note", text: "No Q4_0 measurements for this phase in the published bench JSON." }));
      container.appendChild(card);
      return;
    }
    card.appendChild(buildChartSvg(rows));
    card.appendChild(
      el("div", { class: "chart-legend" }, [
        el("span", { class: "legend-item" }, [el("span", { class: "legend-swatch on" }), el("span", { text: "SME2 path (env unset)" })]),
        el("span", { class: "legend-item" }, [el("span", { class: "legend-swatch off" }), el("span", { text: "NEON forced (GGML_KLEIDIAI_SME=0)" })]),
      ])
    );
    card.appendChild(el("p", { class: "chart-caption", text: crossoverCaption(rows) }));
    card.appendChild(buildChartTable(rows));
    container.appendChild(card);
  }

  function platformLabelFor(doc) {
    var meta = doc.meta || {};
    return meta.cpu_brand || meta.platform || "unknown platform";
  }

  function renderCharts(benchDatasets) {
    var container = document.getElementById("charts");
    container.textContent = "";
    if (!benchDatasets.length) {
      container.appendChild(el("p", { class: "chart-empty", text: "No bench JSON has been published to this site yet -- see results/bench/*.json in the repository." }));
      return;
    }

    var phaseOrder = ["decode", "prefill_short", "prefill_long"];
    benchDatasets.forEach(function (ds) {
      var byPhase = seriesByPhase(ds.doc);
      var label = platformLabelFor(ds.doc);
      var phases = Object.keys(byPhase).sort(function (a, b) {
        var ia = phaseOrder.indexOf(a),
          ib = phaseOrder.indexOf(b);
        if (ia === -1) ia = 99;
        if (ib === -1) ib = 99;
        return ia - ib;
      });
      phases.forEach(function (phase) {
        renderChartCard(container, label, phase, byPhase[phase]);
      });
    });

    var q8Skipped = benchDatasets.some(function (ds) {
      return rowsOf(ds.doc).some(function (r) { return r.not_available; });
    });
    if (q8Skipped) {
      container.appendChild(
        el("p", { class: "chart-empty", text: "Note: one or more quantizations in the published bench data are marked not_available (no matching GGUF in the measuring environment) and are omitted above rather than shown as zero." })
      );
    }
  }

  // ----------------------------------------------------------------------
  // crossover/*.json -- schema not fixed by this project yet. Reuse the
  // bench renderer when the shape matches (rows[].phase/threads/sme_mode/
  // agg.median_ts); otherwise link to the raw file rather than dropping it
  // silently.
  // ----------------------------------------------------------------------
  function looksLikeBenchShape(doc) {
    var rows = doc && rowsOf(doc);
    return !!(rows && rows.length && rows[0] && "phase" in rows[0] && "sme_mode" in rows[0]);
  }

  function renderCrossoverExtras(paths) {
    if (!paths.length) return Promise.resolve();
    return loadBenchFiles(paths).then(function (datasets) {
      if (!datasets.length) return;
      var recognized = datasets.filter(function (d) { return looksLikeBenchShape(d.doc); });
      var unrecognized = datasets.filter(function (d) { return !looksLikeBenchShape(d.doc); });

      var section = el("div", { class: "crossover-extra" }, [el("h3", { text: "Additional crossover data" })]);
      var any = false;

      if (recognized.length) {
        any = true;
        var container = el("div", { class: "chart-grid" });
        renderChartsInto(container, recognized);
        section.appendChild(container);
      }
      if (unrecognized.length) {
        any = true;
        var ul = el("ul");
        unrecognized.forEach(function (d) {
          ul.appendChild(
            el("li", {}, [
              el("a", { href: "data/" + d.path, text: d.path }),
              el("span", { text: " — schema not recognized by this dashboard; linked as raw JSON." }),
            ])
          );
        });
        section.appendChild(el("p", { text: "Files with an unrecognized shape (raw links, nothing dropped):" }));
        section.appendChild(ul);
      }
      if (any) {
        document.getElementById("chart-section").appendChild(section);
      }
    });
  }

  function renderChartsInto(container, benchDatasets) {
    var phaseOrder = ["decode", "prefill_short", "prefill_long"];
    benchDatasets.forEach(function (ds) {
      var byPhase = seriesByPhase(ds.doc);
      var label = platformLabelFor(ds.doc) + " (" + ds.path + ")";
      Object.keys(byPhase)
        .sort(function (a, b) {
          var ia = phaseOrder.indexOf(a),
            ib = phaseOrder.indexOf(b);
          if (ia === -1) ia = 99;
          if (ib === -1) ib = 99;
          return ia - ib;
        })
        .forEach(function (phase) {
          renderChartCard(container, label, phase, byPhase[phase]);
        });
    });
  }

  // ----------------------------------------------------------------------
  // Figures gallery (bonus, static matplotlib PNGs)
  // ----------------------------------------------------------------------
  function renderFigures(paths) {
    if (!paths.length) return;
    var section = document.getElementById("figures-section");
    var container = document.getElementById("figures");
    container.textContent = "";
    paths.forEach(function (p) {
      var name = p.split("/").pop();
      container.appendChild(
        el("figure", {}, [
          el("img", { src: "data/" + p, alt: name, loading: "lazy" }),
          el("figcaption", { text: name }),
        ])
      );
    });
    section.hidden = false;
  }

  // ----------------------------------------------------------------------
  // Footer "data captured" timestamp -- the freshest generated_at(_utc)
  // found across every loaded JSON file, not the page build time.
  // ----------------------------------------------------------------------
  function renderGeneratedAt(ledgers, benchDatasets) {
    var stamps = [];
    ledgers.forEach(function (l) {
      if (l.doc && l.doc.generated_at_utc) stamps.push(l.doc.generated_at_utc);
    });
    benchDatasets.forEach(function (d) {
      if (d.doc.meta && d.doc.meta.generated_at) stamps.push(d.doc.meta.generated_at);
    });
    var span = document.getElementById("data-generated-at");
    if (!stamps.length) {
      span.textContent = "unknown (no timestamped JSON published yet)";
      return;
    }
    stamps.sort();
    span.textContent = stamps[stamps.length - 1];
  }

  // ----------------------------------------------------------------------
  // Boot
  // ----------------------------------------------------------------------
  function main() {
    initTheme();
    // headline.json doesn't depend on the manifest -- start it immediately
    // rather than waiting on the manifest round-trip first.
    var headlineP = loadHeadline();
    loadManifest().then(function (manifest) {
      var ledgersP = loadDispatchLedgers(manifest.dispatch_ledgers || []);
      var benchP = loadBenchFiles(manifest.bench || []);

      Promise.all([ledgersP, benchP, headlineP]).then(function (res) {
        var ledgers = res[0];
        var benchDatasets = res[1];
        var headline = res[2];

        renderHeadline(headline);

        var configs = flattenConfigs(ledgers);
        var stats = renderHeroTable(configs);
        renderSummaryStrip(stats);

        renderCharts(benchDatasets);
        renderFigures(manifest.figures || []);
        renderGeneratedAt(ledgers, benchDatasets);
        renderCrossoverExtras(manifest.crossover || []);
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
