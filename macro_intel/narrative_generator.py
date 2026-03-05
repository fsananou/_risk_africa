"""
narrative_generator.py — Weekly macro intelligence note.

Inputs: rules_engine + inference_engine + sector_dependencies outputs.
Five sections:
  1. What changed this week
  2. What it means for the next 3–6 months
  3. Where risks are building
  4. Where opportunities are emerging
  5. What to watch next

Returns dict + to_markdown() for export.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np

from config import LEVEL_ICON


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _pct(v, dec=1) -> str:
    if _nan(v): return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{dec}f}%"


_REGIME_EMOJI = {"crisis": "🔴", "stressed": "🟠", "cautious": "🟡", "benign": "🟢"}
_REGIME_DESC = {
    "crisis":   "CRISIS — Multiple simultaneous stress indicators. Maximum defensive posture warranted.",
    "stressed": "STRESSED — Elevated cross-asset stress. Risk-reward skewed defensively.",
    "cautious": "CAUTIOUS — Underlying vulnerabilities present; selective risk exposure appropriate.",
    "benign":   "BENIGN — Broadly constructive macro regime; watch for regime change signals.",
}


# ── Section builders ──────────────────────────────────────────────────────────

def _section_what_changed(ind: dict, rules: list[dict]) -> dict:
    bullets = []

    def _add(metric, value, suffix="", regime=""):
        s = f"**{metric}**: {value}{suffix}"
        if regime and regime not in ("unknown", "normal", ""):
            s += f" ({regime})"
        bullets.append(s)

    slope   = ind.get("curve_slope", np.nan)
    vix     = ind.get("vix", np.nan)
    dxy     = ind.get("dxy", np.nan)
    brent   = ind.get("brent", np.nan)
    copper  = ind.get("copper", np.nan)
    fao_fpi = ind.get("fao_fpi", np.nan)
    hy      = ind.get("hy_spread", np.nan)
    eu_gas  = ind.get("eu_gas_storage_pct", np.nan)

    if not _nan(slope):
        _add("Yield curve", f"{slope:+.0f} bps", regime=ind.get("curve_regime",""))
    if not _nan(vix):
        _add("VIX", f"{vix:.1f}", regime=ind.get("vix_regime",""))
    if not _nan(dxy):
        _add("DXY", f"{dxy:.1f}", regime=ind.get("dollar_regime",""))
    if not _nan(brent):
        _add("Brent crude", f"${brent:.1f}/bbl", regime=ind.get("oil_regime",""))
    if not _nan(copper):
        cu_1m = ind.get("copper_1m_chg", np.nan)
        _add("Copper", f"{_pct(cu_1m)} 1M", regime=ind.get("copper_regime",""))
    if not _nan(hy):
        _add("US HY spread", f"{hy:.0f} bps", regime=ind.get("hy_regime",""))
    if not _nan(eu_gas):
        _add("EU gas storage", f"{eu_gas:.1f}% full", "(AGSI+)", regime=ind.get("eu_gas_storage_regime",""))
    if not _nan(fao_fpi):
        _add("FAO Food Price Index", f"{fao_fpi:.0f}", "(2014-16=100)", regime=ind.get("fao_fpi_regime",""))

    n_alerts   = sum(1 for r in rules if r["level"] == "alert")
    n_warnings = sum(1 for r in rules if r["level"] == "warning")
    regime     = ind.get("macro_regime", "benign")

    body = (
        f"The current macro configuration registers **{n_alerts} alert(s)** and "
        f"**{n_warnings} warning(s)**. Composite regime: **{regime.upper()}**. Key readings:"
    )
    if not bullets:
        bullets = ["Insufficient real data to generate readings — check API key configuration."]

    return {"title": "What Changed This Week", "body": body, "bullets": bullets}


def _section_medium_term(inferences: list[dict]) -> dict:
    bullets = []
    target = {"medium-term (1-6 months)", "structural (6-18 months)"}
    forward = [i for i in inferences
               if i.get("horizon","") in target and i.get("confidence") in ("high","medium")]
    for inf in forward[:6]:
        icon = inf.get("icon","📌")
        conf = inf.get("confidence","")
        stmt = inf.get("statement","")
        impl = inf.get("implication","")
        srcs = ", ".join(inf.get("data_sources",[]))
        bullets.append(f"{icon} **[{conf.upper()}]** {stmt} — *{impl}* _(sources: {srcs})_")

    if not bullets:
        bullets = ["No high/medium-confidence medium-term signals detected. Macro regime is broadly stable."]

    body = (
        "Forward-looking conditional analysis (inference engine, 3–6 month horizon). "
        "Only rules backed by real data are shown:"
    )
    return {"title": "What It Means for the Next 3–6 Months", "body": body, "bullets": bullets}


def _section_risks(ind: dict, rules: list[dict], inferences: list[dict],
                   propagations: list[dict]) -> dict:
    bullets = []

    # Alerts from rules engine
    for r in [x for x in rules if x["level"] == "alert"]:
        bullets.append(f"🔴 **{r['category']}**: {r['headline']}")

    # Near-term high-confidence inferences
    near = [i for i in inferences
            if i.get("confidence") == "high" and i.get("horizon") == "near-term (1-4 weeks)"]
    for inf in near[:3]:
        bullets.append(f"⚡ **Near-term risk**: {inf.get('statement','')}")

    # Strong cross-sector propagations
    for prop in [p for p in propagations if p.get("strength") == "strong"]:
        bullets.append(f"🔗 **Sector cascade**: {prop['headline']}")

    # Specific EM/Africa
    if ind.get("fx_res_deteriorating") and ind.get("fx_res_worst_country"):
        bullets.append(f"📉 **FX reserve drawdown**: {ind['fx_res_worst_country']} — sovereign stress risk elevated (World Bank)")

    if not bullets:
        bullets = ["No acute risk signals identified. Monitor leading indicators for early warning."]

    body = "Active stress signals and risk accumulation zones from real indicator data:"
    return {"title": "Where Risks Are Building", "body": body, "bullets": bullets}


def _section_opportunities(ind: dict, inferences: list[dict]) -> dict:
    bullets = []

    if ind.get("dollar_regime") == "weakening":
        dxy = ind.get("dxy", np.nan)
        bullets.append(f"🌍 **EM rotation**: Weakening USD (DXY {dxy:.1f if not _nan(dxy) else 'N/A'}) eases EM financial conditions — EM equities and local bonds outperform historically.")

    if ind.get("copper_regime") == "rising":
        bullets.append(f"🔧 **Industrial cycle**: Copper {_pct(ind.get('copper_1m_chg'))} (1M) signals improving PMI — DRC, Chile, Zambia exporters benefit; EV supply chain plays attractive.")

    if ind.get("eu_gas_storage_regime") == "comfortable":
        bullets.append("⚡ **Energy normalisation**: EU gas storage comfortable — energy cost headwinds for European industry easing.")

    if ind.get("curve_regime") == "steep":
        bullets.append(f"🏦 **Steepening curve** ({ind.get('curve_slope', 0):+.0f} bps): Bank NIM expands — financials outperform in steep-curve regimes.")

    if ind.get("vix_regime") == "elevated":
        bullets.append("📊 **Volatility premium**: Elevated VIX = rich options premium for structured protection and vol-selling strategies.")

    if ind.get("oil_regime") == "surging":
        bullets.append("🛢️ **Africa energy exporters**: Oil surge benefits Nigeria, Angola, Algeria — fiscal positions improve, local spreads may tighten.")

    if ind.get("copper_gold_regime") == "risk_on" and ind.get("vix_regime") == "normal":
        bullets.append("🚀 **Risk-on window**: Cu/Gold ratio rising + low VIX = historically favourable entry for EM equity and credit.")

    # Medium-confidence opportunity inferences
    opp = [i for i in inferences
           if i.get("confidence") == "medium"
           and "opportunit" in i.get("implication","").lower()]
    for inf in opp[:2]:
        bullets.append(f"💡 {inf.get('statement','')} — *{inf.get('implication','')}*")

    if not bullets:
        bullets = ["No high-conviction tactical opportunities in current regime. Build dry powder for regime-shift entry."]

    body = "Regime-consistent positioning ideas from real indicator configuration (not investment advice):"
    return {"title": "Where Opportunities Are Emerging", "body": body, "bullets": bullets}


def _section_watch_next(rules: list[dict], inferences: list[dict],
                        propagations: list[dict]) -> dict:
    watch_set: dict[str, int] = {}

    for r in rules:
        priority = {"alert": 0, "warning": 1, "info": 2}.get(r.get("level","info"), 3)
        for item in r.get("watch", []):
            if item not in watch_set or watch_set[item] > priority:
                watch_set[item] = priority

    sorted_items = [item for item, _ in sorted(watch_set.items(), key=lambda x: x[1])]

    # Near-term high-confidence trigger conditions
    for inf in [i for i in inferences
                if i.get("confidence") == "high" and i.get("horizon") == "near-term (1-4 weeks)"]:
        item = f"⚡ Trigger: {inf.get('trigger','')} (near-term)"
        if item not in sorted_items:
            sorted_items.insert(0, item)

    # Sector propagation early warnings
    for prop in [p for p in propagations if p.get("strength") == "strong"]:
        item = f"🔗 Sector chain: {prop.get('from_sector','')} → {prop.get('to_sector','')}"
        if item not in sorted_items:
            sorted_items.append(item)

    # Always-on structural items
    structural = [
        "IMF program status (Ghana, Kenya, Ethiopia, Egypt)",
        "Eurobond maturity wall 2025–2027",
        "FX reserve import cover (<3M = critical)",
        "FAO monthly Food Price Index release",
        "EIA weekly petroleum status report",
        "EU gas storage injection pace (AGSI+)",
    ]
    for item in structural:
        if item not in sorted_items:
            sorted_items.append(item)

    bullets = [f"• {item}" for item in sorted_items[:12]]
    body = "Priority watch list for the coming week (sources: FRED, yfinance, EIA, AGSI+, FAO, World Bank):"
    return {"title": "What to Watch Next", "body": body, "bullets": bullets}


def _section_sector_summary(sectors: dict) -> dict:
    bullets = []
    for name, data in sectors.items():
        status = data.get("status", "no_data")
        headline = data.get("headline", "No data")
        icon = "🔴" if status in ("crisis","contraction") else \
               "🟠" if status in ("stress","slowing","stressed") else \
               "🟢" if status in ("normal","comfortable","expanding") else "⚫"
        bullets.append(f"{icon} **{name}**: {headline}")
        for ph in data.get("placeholders", []):
            bullets.append(f"  ⚫ PLACEHOLDER: {ph}")

    body = "Sector intelligence snapshot (real data where available; PLACEHOLDERs labelled):"
    return {"title": "Sector Intelligence Summary", "body": body, "bullets": bullets}


# ── Main entry point ──────────────────────────────────────────────────────────

def generate(
    ind:          dict,
    rules:        list[dict],
    inferences:   list[dict],
    propagations: list[dict],
    sectors:      dict,
    as_of:        Optional[datetime.date] = None,
) -> dict:
    """Generate the weekly macro intelligence note."""
    as_of     = as_of or datetime.date.today()
    as_of_str = as_of.strftime("%d %B %Y")
    regime    = ind.get("macro_regime", "benign")
    fin_score = ind.get("financial_stress_score", 0)
    geo_score = ind.get("geo_stress_score", 0)

    n_alerts   = sum(1 for r in rules if r["level"] == "alert")
    n_warnings = sum(1 for r in rules if r["level"] == "warning")
    n_high_inf = sum(1 for i in inferences if i.get("confidence") == "high")

    if regime == "crisis":
        headline = f"CRISIS ALERT — {n_alerts} simultaneous stress signals; {n_high_inf} high-confidence forward risks"
    elif regime == "stressed":
        headline = f"Elevated stress — {n_alerts} alerts, {n_warnings} warnings; {n_high_inf} high-confidence forward risks"
    elif regime == "cautious":
        headline = f"Cautious — {n_warnings} warning signals; underlying vulnerabilities warrant monitoring"
    else:
        headline = "Broadly benign — no acute cross-asset stress signals detected"

    sections = [
        _section_what_changed(ind, rules),
        _section_medium_term(inferences),
        _section_risks(ind, rules, inferences, propagations),
        _section_opportunities(ind, inferences),
        _section_watch_next(rules, inferences, propagations),
        _section_sector_summary(sectors),
    ]

    # Collect real data sources
    data_sources = list({s for i in inferences for s in i.get("data_sources",[]) if s})
    data_sources += [s for s in ("FRED", "yfinance") if any(s in r.get("headline","") for r in rules)]

    footer = (
        f"Macro Intelligence Briefing — {as_of_str} | "
        f"Financial stress: {fin_score}/7 | Geo-risk: {geo_score}/4 | "
        "Real data only. Placeholders labelled. Not investment advice."
    )

    return {
        "as_of":        as_of_str,
        "headline":     headline,
        "regime_label": regime.upper(),
        "regime_desc":  _REGIME_DESC.get(regime, ""),
        "regime_emoji": _REGIME_EMOJI.get(regime, "⚪"),
        "sections":     sections,
        "footer":       footer,
        "data_sources": data_sources,
    }


def to_markdown(note: dict) -> str:
    lines = [
        f"# Macro Intelligence Briefing — {note['as_of']}",
        "",
        f"**Regime:** {note['regime_emoji']} {note['regime_label']}",
        f"> {note['regime_desc']}",
        "",
        f"## {note['headline']}",
        "",
    ]
    for sec in note["sections"]:
        lines += [f"### {sec['title']}", "", sec["body"], ""]
        lines += [f"- {b}" for b in sec["bullets"]]
        lines.append("")
    if note.get("data_sources"):
        lines.append(f"*Sources: {', '.join(sorted(set(note['data_sources'])))}*")
        lines.append("")
    lines.append(f"*{note['footer']}*")
    return "\n".join(lines)
