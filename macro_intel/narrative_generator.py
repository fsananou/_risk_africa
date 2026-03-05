"""
narrative_generator.py — Weekly macro intelligence note.

Produces a structured 1-page briefing from rules_engine + inference_engine outputs.
Five sections:
  1. What changed this week
  2. What it means for the next 3–6 months
  3. Where risks are building
  4. Where opportunities are emerging
  5. What to watch next

Usage:
    from narrative_generator import generate
    note = generate(ind, rules, inferences, as_of=None)
    # note is a dict with keys: as_of, headline, regime_label,
    #   sections (list of {title, body, bullets}), footer
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np

from config import LEVEL_COLOR, LEVEL_ICON


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _pct(v, dec=1) -> str:
    if _nan(v):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{dec}f}%"


def _bps(v, dec=0) -> str:
    if _nan(v):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{dec}f} bps"


def _fmt(v, dec=1, suffix="") -> str:
    if _nan(v):
        return "N/A"
    return f"{v:.{dec}f}{suffix}"


def _alerts(rules: list[dict]) -> list[dict]:
    return [r for r in rules if r["level"] == "alert"]


def _warnings(rules: list[dict]) -> list[dict]:
    return [r for r in rules if r["level"] == "warning"]


def _high_conf(inferences: list[dict]) -> list[dict]:
    return [i for i in inferences if i.get("confidence") == "high"]


def _medium_conf(inferences: list[dict]) -> list[dict]:
    return [i for i in inferences if i.get("confidence") == "medium"]


# ── Regime labels ─────────────────────────────────────────────────────────────

_REGIME_DESC = {
    "crisis":   "CRISIS — Multiple simultaneous stress indicators, maximum defensive posture warranted.",
    "stressed": "STRESSED — Elevated cross-asset stress. Risk-reward skewed defensively.",
    "cautious": "CAUTIOUS — Underlying vulnerabilities present; selective risk exposure appropriate.",
    "benign":   "BENIGN — Broadly constructive macro regime; watch for regime change signals.",
}

_REGIME_EMOJI = {
    "crisis": "🔴",
    "stressed": "🟠",
    "cautious": "🟡",
    "benign": "🟢",
}


# ── Section builders ──────────────────────────────────────────────────────────

def _section_what_changed(ind: dict, rules: list[dict]) -> dict:
    """Section 1: Current-state snapshot — what is happening NOW."""
    bullets = []

    # Yield curve
    slope = ind.get("curve_slope", np.nan)
    curve_rg = ind.get("curve_regime", "")
    if not _nan(slope):
        direction = ind.get("curve_direction", "")
        bullets.append(
            f"**Yield curve** {curve_rg} at {slope:+.0f} bps (2Y–10Y)"
            + (f", {direction}" if direction else "")
        )

    # VIX
    vix = ind.get("vix", np.nan)
    vix_rg = ind.get("vix_regime", "")
    if not _nan(vix):
        bullets.append(f"**VIX** {vix:.1f} ({vix_rg} volatility regime)")

    # Dollar
    dxy = ind.get("dxy", np.nan)
    dollar_rg = ind.get("dollar_regime", "")
    if not _nan(dxy):
        bullets.append(f"**DXY** {dxy:.1f} — dollar regime: {dollar_rg}")

    # Oil & Copper
    oil_rg = ind.get("oil_regime", "")
    oil_chg = ind.get("oil_1m_chg", np.nan)
    if not _nan(oil_chg):
        bullets.append(f"**Oil** {_pct(oil_chg)} (1M) — {oil_rg}")

    cu_rg = ind.get("copper_regime", "")
    cu_chg = ind.get("copper_1m_chg", np.nan)
    if not _nan(cu_chg):
        bullets.append(f"**Copper** {_pct(cu_chg)} (1M) — {cu_rg} (industrial signal)")

    # Gold
    gold_rg = ind.get("gold_regime", "")
    gold_chg = ind.get("gold_1m_chg", np.nan)
    if not _nan(gold_chg) and gold_rg in ("rising", "surging"):
        bullets.append(f"**Gold** {_pct(gold_chg)} (1M) — safe-haven demand {gold_rg}")

    # EM spreads
    embi = ind.get("embi", np.nan)
    em_rg = ind.get("em_regime", "")
    if not _nan(embi):
        bullets.append(f"**EMBI spreads** {embi:.0f} bps — EM sovereign regime: {em_rg}")

    # HY spreads
    hy = ind.get("hy_spread", np.nan)
    hy_rg = ind.get("hy_regime", "")
    if not _nan(hy):
        bullets.append(f"**US HY spread** {hy:.0f} bps — credit regime: {hy_rg}")

    # GPR
    gpr = ind.get("gpr", np.nan)
    gpr_rg = ind.get("gpr_regime", "")
    if not _nan(gpr) and gpr_rg in ("elevated", "high"):
        bullets.append(f"**Geopolitical risk** (GPR {gpr:.0f}) — {gpr_rg}")

    # Systemic stress
    if ind.get("systemic_stress_signal"):
        bullets.append("⚠️ **Systemic stress signal active**: gold and USD rising simultaneously")

    # Financial conditions composite
    fin_cond = ind.get("fin_cond_regime", "")
    fci = ind.get("financial_conditions_index", np.nan)
    if fin_cond in ("tightening",) and not _nan(fci):
        bullets.append(f"**Financial conditions** tightening (FCI z-score: {fci:.2f})")

    # Compose body
    n_alerts   = len(_alerts(rules))
    n_warnings = len(_warnings(rules))
    stress_lvl = ind.get("macro_regime", "benign")

    body = (
        f"The current macro configuration registers **{n_alerts} alert(s)** and "
        f"**{n_warnings} warning(s)** across cross-asset indicators. "
        f"The composite regime is classified as **{stress_lvl.upper()}**. "
        "Key readings:"
    )

    return {"title": "What Changed This Week", "body": body, "bullets": bullets}


def _section_medium_term(inferences: list[dict]) -> dict:
    """Section 2: Forward-looking — 3–6 month implications from high/medium confidence rules."""
    bullets = []

    target_horizons = {"medium-term (1-6 months)", "structural (6-18 months)"}
    forward = [
        i for i in inferences
        if i.get("horizon", "") in target_horizons
        and i.get("confidence") in ("high", "medium")
    ]

    for inf in forward[:6]:
        icon = inf.get("icon", "📌")
        conf = inf.get("confidence", "")
        stmt = inf.get("statement", "")
        impl = inf.get("implication", "")
        bullets.append(f"{icon} **[{conf.upper()}]** {stmt} — *{impl}*")

    if not bullets:
        bullets.append(
            "No high/medium-confidence medium-term signals detected. "
            "Current regime is broadly stable — maintain baseline positioning."
        )

    body = (
        "Forward-looking conditional analysis (inference engine, 3–6 month horizon). "
        "Rules fire on current indicator configuration — confidence reflects robustness "
        "of the trigger–outcome relationship historically:"
    )

    return {"title": "What It Means for the Next 3–6 Months", "body": body, "bullets": bullets}


def _section_risks(ind: dict, rules: list[dict], inferences: list[dict]) -> dict:
    """Section 3: Where risks are building."""
    bullets = []

    # Pull alert-level rules
    for r in _alerts(rules):
        bullets.append(f"🔴 **{r['category']}**: {r['headline']}")

    # Pull high-confidence near-term inferences that are risk-oriented
    near = [
        i for i in inferences
        if i.get("confidence") == "high"
        and i.get("horizon") == "near-term (1-4 weeks)"
    ]
    for inf in near[:3]:
        bullets.append(f"⚡ **Near-term risk**: {inf.get('statement', '')}")

    # EM-specific tail risks
    fx_det = ind.get("fx_res_deteriorating", False)
    worst  = ind.get("fx_res_worst_country", "")
    if fx_det:
        bullets.append(
            f"📉 **FX reserve drawdown**: {worst or 'Multiple EM economies'} showing "
            "below-threshold reserve cover — sovereign stress risk elevated."
        )

    # Sanctions / geo compounding
    if ind.get("sanctions_elevated") and ind.get("gpr_regime") in ("elevated", "high"):
        bullets.append(
            "🚫 **Sanctions + geopolitical risk compounding**: elevated GPR with active "
            "sanctions pressure raises supply-chain fragmentation risk."
        )

    # Inverted curve + tightening
    if ind.get("curve_regime") == "inverted" and ind.get("fin_cond_regime") == "tightening":
        bullets.append(
            "📉 **Recession signal**: inverted yield curve + tightening financial conditions "
            "— historically 12–18 month lead to contraction."
        )

    # Breakeven inflation + curve flat/inverted
    if ind.get("inflation_regime") in ("high", "very_high") and ind.get("curve_regime") in ("flat", "inverted"):
        bullets.append(
            "🔥 **Stagflation risk**: elevated inflation expectations + "
            "flat/inverted curve — most difficult environment for policy."
        )

    if not bullets:
        bullets.append("No acute risk signals identified. Monitor leading indicators for early warning.")

    body = (
        "Active stress signals and risk accumulation zones identified by the "
        "rules engine and inference engine. These represent conditions where "
        "the probability of adverse outcomes is materially elevated:"
    )

    return {"title": "Where Risks Are Building", "body": body, "bullets": bullets}


def _section_opportunities(ind: dict, inferences: list[dict]) -> dict:
    """Section 4: Where opportunities are emerging."""
    bullets = []

    # Dollar weakening → EM tailwind
    if ind.get("dollar_regime") == "weakening":
        dxy = ind.get("dxy", np.nan)
        bullets.append(
            f"🌍 **EM rotation opportunity**: weakening USD (DXY {_fmt(dxy)}) eases "
            "financial conditions globally — EM equities and local-currency bonds historically outperform."
        )

    # Copper rising → commodity exporters
    if ind.get("copper_regime") == "rising":
        cu_chg = ind.get("copper_1m_chg", np.nan)
        bullets.append(
            f"🔧 **Industrial commodity cycle**: copper {_pct(cu_chg)} (1M) signals "
            "improving PMI outlook — DRC, Chile, Zambia copper exporters benefit; "
            "EV supply chain plays attractive."
        )

    # Steep curve → banks, value equities
    if ind.get("curve_regime") == "steep":
        slope = ind.get("curve_slope", np.nan)
        bullets.append(
            f"🏦 **Steepening curve** ({_fmt(slope, 0)} bps): bank net interest margins expand "
            "— financials sector and value equities outperform in steep-curve regimes."
        )

    # VIX elevated → vol selling premium
    vix_rg = ind.get("vix_regime", "")
    if vix_rg == "elevated":
        bullets.append(
            "📊 **Volatility premium**: elevated VIX creates attractive options premium "
            "for structured downside protection; mean-reversion plays on VIX futures."
        )

    # Oil surging + Africa producers
    if ind.get("oil_regime") == "surging":
        bullets.append(
            "🛢️ **Africa energy producers**: oil surge benefits Nigeria, Angola, Algeria — "
            "current account and fiscal positions improve; local bond spreads may tighten."
        )

    # Low VIX + copper rising + dollar neutral → risk-on
    if (vix_rg == "normal"
            and ind.get("copper_regime") in ("stable", "rising")
            and ind.get("dollar_regime") in ("neutral", "weakening")):
        bullets.append(
            "🚀 **Risk-on window**: low volatility + constructive copper + neutral/weak dollar "
            "— historically a favorable entry point for EM equity and credit."
        )

    # Gold rising → central bank reserve plays
    if ind.get("gold_regime") in ("rising", "surging"):
        bullets.append(
            "🥇 **Gold / real asset allocation**: sustained gold bid (central bank structural "
            "accumulation + de-dollarization trend) supports precious metals and royalty companies."
        )

    # Pull medium-confidence positive inferences
    opp_inferences = [
        i for i in inferences
        if i.get("confidence") == "medium"
        and "opportunity" in i.get("implication", "").lower()
    ]
    for inf in opp_inferences[:2]:
        bullets.append(f"💡 {inf.get('statement', '')} — *{inf.get('implication', '')}*")

    if not bullets:
        bullets.append(
            "Current regime does not present high-conviction tactical opportunities. "
            "Maintain diversification; build dry powder for regime-shift entry points."
        )

    body = (
        "Macro configuration-driven opportunity identification. "
        "These are regime-consistent positioning ideas — not investment advice:"
    )

    return {"title": "Where Opportunities Are Emerging", "body": body, "bullets": bullets}


def _section_watch_next(rules: list[dict], inferences: list[dict], ind: dict) -> dict:
    """Section 5: What to watch next — consolidated watch list."""
    watch_set: dict[str, int] = {}  # item → priority (lower = higher priority)

    # Collect from rules
    for r in rules:
        level = r.get("level", "info")
        priority = {"alert": 0, "warning": 1, "info": 2}.get(level, 3)
        for item in r.get("watch", []):
            if item not in watch_set or watch_set[item] > priority:
                watch_set[item] = priority

    # Deduplicate and sort
    sorted_items = sorted(watch_set.items(), key=lambda x: x[1])
    top_watch = [item for item, _ in sorted_items[:10]]

    # Add horizon-specific items from inferences
    near_term = [
        i for i in inferences
        if i.get("horizon") == "near-term (1-4 weeks)"
        and i.get("confidence") == "high"
    ]
    for inf in near_term[:2]:
        trigger = inf.get("trigger", "")
        if trigger and trigger not in top_watch:
            top_watch.insert(0, f"⚡ {trigger} (near-term trigger)")

    # Key structural items always worth watching in Africa/EM context
    structural_defaults = [
        "IMF program status updates",
        "Eurobond maturity wall 2025–2027 (Ghana, Kenya, Ethiopia, Egypt)",
        "FX reserve import cover (<3M threshold = critical)",
        "Fed dot plot and rate path revisions",
    ]
    for item in structural_defaults:
        if item not in top_watch:
            top_watch.append(item)

    bullets = [f"• {item}" for item in top_watch[:12]]

    body = (
        "Consolidated watch list for the coming week. "
        "Prioritized by current alert severity and forward-looking inference triggers:"
    )

    return {"title": "What to Watch Next", "body": body, "bullets": bullets}


# ── Main entry point ──────────────────────────────────────────────────────────

def generate(
    ind: dict,
    rules: list[dict],
    inferences: list[dict],
    as_of: Optional[datetime.date] = None,
) -> dict:
    """
    Generate the weekly macro intelligence note.

    Parameters
    ----------
    ind         : indicators dict from indicators.compute_all()
    rules       : output of rules_engine.run(ind)
    inferences  : output of inference_engine.run(ind, rules)
    as_of       : date for the note header (defaults to today)

    Returns
    -------
    dict with keys:
        as_of        : str (ISO date)
        headline     : str
        regime_label : str (CRISIS / STRESSED / CAUTIOUS / BENIGN)
        regime_desc  : str
        regime_emoji : str
        sections     : list[{title, body, bullets}]
        footer       : str
        data_sources : list[str]
    """
    as_of = as_of or datetime.date.today()
    as_of_str = as_of.strftime("%d %B %Y")

    regime = ind.get("macro_regime", "benign")
    fin_score = ind.get("financial_stress_score", 0)
    geo_score = ind.get("geo_stress_score", 0)

    # Headline
    n_alerts   = len(_alerts(rules))
    n_warnings = len(_warnings(rules))
    n_high_inf = len(_high_conf(inferences))

    if regime == "crisis":
        headline = f"CRISIS ALERT — {n_alerts} simultaneous stress signals across cross-asset indicators"
    elif regime == "stressed":
        headline = (
            f"Elevated stress regime — {n_alerts} alerts, {n_warnings} warnings detected; "
            f"{n_high_inf} high-confidence forward risks flagged"
        )
    elif regime == "cautious":
        headline = (
            f"Cautious macro regime — {n_warnings} warning signals; "
            "underlying vulnerabilities warrant active monitoring"
        )
    else:
        headline = "Broadly benign macro environment — no acute cross-asset stress signals"

    # Regime description
    regime_desc = _REGIME_DESC.get(regime, "")
    regime_emoji = _REGIME_EMOJI.get(regime, "⚪")

    # Build sections
    sections = [
        _section_what_changed(ind, rules),
        _section_medium_term(inferences),
        _section_risks(ind, rules, inferences),
        _section_opportunities(ind, inferences),
        _section_watch_next(rules, inferences, ind),
    ]

    # Data source labels
    data_sources = []
    for key in (
        "curve_source", "vix_source", "dxy_source",
        "oil_source", "copper_source", "gpr_source",
        "em_spread_source", "fx_res_source",
    ):
        src = ind.get(key, "")
        if src and src not in data_sources:
            data_sources.append(src)

    footer = (
        f"Macro Intelligence Briefing — {as_of_str} | "
        f"Financial stress score: {fin_score}/7 | "
        f"Geopolitical stress score: {geo_score}/4 | "
        "This note is generated algorithmically. Not investment advice."
    )

    return {
        "as_of":        as_of_str,
        "headline":     headline,
        "regime_label": regime.upper(),
        "regime_desc":  regime_desc,
        "regime_emoji": regime_emoji,
        "sections":     sections,
        "footer":       footer,
        "data_sources": data_sources,
    }


def to_markdown(note: dict) -> str:
    """
    Convert a note dict to a plain Markdown string (for export / clipboard).
    """
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
        lines.append(f"### {sec['title']}")
        lines.append("")
        lines.append(sec["body"])
        lines.append("")
        for b in sec["bullets"]:
            lines.append(f"- {b}")
        lines.append("")

    if note.get("data_sources"):
        lines.append(f"*Data sources: {', '.join(note['data_sources'])}*")
        lines.append("")

    lines.append(f"*{note['footer']}*")
    return "\n".join(lines)
