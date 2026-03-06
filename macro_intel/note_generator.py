"""
note_generator.py — Daily and hourly macro intelligence notes.

Produces two note types from the same indicator dict:
  - daily_note(ind, rules, inferences, propagations, sectors) → dict
  - hourly_note(ind, rules, inferences)                       → dict
  - to_text(note)                                             → str

Notes use ONLY real-data indicators.
PLACEHOLDER indicators are labelled explicitly and skipped from analysis.

Sections (daily):
  1. Macro Regime Update
  2. Geo-Risk Update
  3. EM & Africa Stress Update
  4. Sector Intelligence Update
  5. Market-Implied Update
  6. Rules Engine Insights
  7. Inference Engine Forward-Looking Statements
  8. Sector Dependency Propagation Insights
  9. What to Watch Next (next 6–24 hours)

Hourly note: condensed 4-section snapshot.
"""

from __future__ import annotations

import datetime
from typing import Optional

import numpy as np


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _pct(v, dec=1) -> str:
    if _nan(v):
        return "N/A"
    sign = "+" if float(v) >= 0 else ""
    return f"{sign}{float(v) * 100:.{dec}f}%"


def _val(v, fmt=".1f", suffix="", fallback="N/A") -> str:
    if _nan(v):
        return fallback
    return f"{float(v):{fmt}}{suffix}"


# ── Box drawing helpers ────────────────────────────────────────────────────────

_BOX_W = 80

def _box_line(char="─") -> str:
    return char * _BOX_W


def _box_row(text: str, pad=2) -> str:
    inner = _BOX_W - 2 * pad
    lines = []
    while len(text) > inner:
        lines.append(" " * pad + text[:inner])
        text = text[inner:]
    lines.append(" " * pad + text.ljust(inner))
    return "\n".join(lines)


def _box(title: str, rows: list[str], level: str = "info") -> str:
    """Return a text box with title and rows."""
    icons = {"alert": "🔴", "warning": "🟠", "info": "🔵", "good": "🟢", "neutral": "⚪"}
    icon = icons.get(level, "⚪")
    pad_row = "─" * _BOX_W
    header = f"┌{pad_row}┐"
    footer = f"└{pad_row}┘"
    title_line = f"│ {icon} {title:<{_BOX_W - 4}} │"
    sep = f"│{'─' * _BOX_W}│"
    body_lines = [f"│ {r:<{_BOX_W - 2}} │" for r in rows]
    return "\n".join([header, title_line, sep] + body_lines + [footer])


# ── Section builders ───────────────────────────────────────────────────────────

def _sec_macro(ind: dict) -> dict:
    """Section 1: Macro Regime Update."""
    regime   = ind.get("macro_regime", "unknown")
    fin_sc   = ind.get("financial_stress_score", 0)
    geo_sc   = ind.get("geo_stress_score", 0)
    slope    = ind.get("curve_slope", np.nan)
    curve_rg = ind.get("curve_regime", "unknown")
    vix      = ind.get("vix", np.nan)
    vix_rg   = ind.get("vix_regime", "unknown")
    dxy      = ind.get("dxy", np.nan)
    dol_rg   = ind.get("dollar_regime", "unknown")
    hy       = ind.get("hy_spread", np.nan)
    hy_rg    = ind.get("hy_regime", "unknown")
    nfci     = ind.get("nfci", np.nan)
    fc_rg    = ind.get("fin_cond_regime", "unknown")
    be5      = ind.get("breakeven5y", np.nan)
    be55     = ind.get("breakeven5y5y", np.nan)
    inf_rg   = ind.get("inflation_regime", "unknown")
    tp       = ind.get("term_premium", np.nan)
    move     = ind.get("move_proxy", np.nan)
    corr_rg  = ind.get("correlation_regime", "unknown")
    oecd     = ind.get("oecd_cli_oecdall", np.nan)
    oecd_rg  = ind.get("oecd_cli_regime", "unknown")
    indpro   = ind.get("indpro_yoy", np.nan)
    indpro_rg= ind.get("indpro_regime", "unknown")

    emoji = {"crisis": "🔴", "stressed": "🟠", "cautious": "🟡", "benign": "🟢"}.get(regime, "⚪")

    rows = [
        f"Composite Regime : {emoji} {regime.upper()}  |  Financial stress: {fin_sc}/7  |  Geo-risk: {geo_sc}/4",
        "",
        f"Yield Curve      : {_val(slope, '+.0f', ' bps')}  [{curve_rg.upper()}]  (source: FRED / Treasury Direct)",
        f"VIX              : {_val(vix, '.1f')}  [{vix_rg.upper()}]  (source: Yahoo Finance)",
        f"DXY              : {_val(dxy, '.1f')}  [{dol_rg.upper()}]  (source: Yahoo Finance)",
        f"HY Spread        : {_val(hy, '.0f', ' bps')}  [{hy_rg.upper()}]  (source: FRED)",
        f"NFCI             : {_val(nfci, '.2f')}  [{fc_rg.upper()}]  (source: FRED)",
        f"5Y Breakeven     : {_val(be5, '.2f', '%')}  5Y5Y: {_val(be55, '.2f', '%')}  [{inf_rg.upper()}]  (source: FRED)",
        f"Term Premium     : {_val(tp, '.2f', '%')}  |  MOVE proxy: {_val(move, '.0f')}  (source: FRED / yields)",
        f"Eq-Bond Corr     : {_val(ind.get('eq_bond_corr'), '.2f')}  [{corr_rg.upper()}]  (source: Yahoo Finance)",
        f"OECD CLI         : {_val(oecd, '.1f')}  [{oecd_rg.upper()}]  (source: OECD SDMX)",
        f"US Indpro YoY    : {_val(indpro, '+.1f', '%')}  [{indpro_rg.upper()}]  (source: FRED)",
    ]
    level = {"crisis": "alert", "stressed": "warning", "cautious": "warning", "benign": "good"}.get(regime, "info")
    return {"title": "1. MACRO REGIME UPDATE", "rows": rows, "level": level}


def _sec_georisk(ind: dict) -> dict:
    """Section 2: Geo-Risk Update."""
    brent    = ind.get("brent", np.nan)
    oil_rg   = ind.get("oil_regime", "unknown")
    oil_1m   = ind.get("oil_1m_chg", np.nan)
    oil_inv  = ind.get("us_oil_inventory_dev", np.nan)
    inv_rg   = ind.get("oil_inventory_regime", "unknown")
    eu_gas   = ind.get("eu_gas_storage_pct", np.nan)
    eu_gas_rg= ind.get("eu_gas_storage_regime", "unknown")
    us_gas   = ind.get("us_gas_storage_last", np.nan)
    us_dev   = ind.get("us_gas_storage_dev", np.nan)
    natgas   = ind.get("natgas", np.nan)

    rows = [
        f"Brent Crude      : {_val(brent, '.1f', ' $/bbl')}  1M: {_pct(oil_1m)}  [{oil_rg.upper()}]  (source: Yahoo Finance)",
        f"US Oil Inventory : {_pct(oil_inv)} vs 5Y avg  [{inv_rg.upper()}]  (source: EIA — needs API key)",
        f"EU Gas Storage   : {_val(eu_gas, '.1f', '% full')}  [{eu_gas_rg.upper()}]  (source: AGSI+ / GIE — free)",
        f"US Nat Gas Storage: {_val(us_gas, '.0f', ' Bcf')}  Dev vs 5Y avg: {_pct(us_dev)}  (source: EIA — needs API key)",
        f"Henry Hub (NG=F) : {_val(natgas, '.2f', ' $/MMBtu')}  (source: Yahoo Finance)",
        "",
        "⚫ GPR (Caldara & Iacoviello) : PLACEHOLDER — load gpr_daily.csv to data/ folder to activate",
        "⚫ Container Shipping Rates   : PLACEHOLDER — Freightos Baltic Index (paid subscription)",
        "⚫ Sanctions Intensity Index  : PLACEHOLDER — no free real-time API",
        "⚫ Chokepoint Stress          : PLACEHOLDER — no free real-time API",
    ]

    level = "alert" if eu_gas_rg == "crisis" or oil_rg == "surging" else \
            "warning" if eu_gas_rg == "stress" else "info"
    return {"title": "2. GEO-RISK UPDATE", "rows": rows, "level": level}


def _sec_em_africa(ind: dict) -> dict:
    """Section 3: EM & Africa Stress Update."""
    em_rg    = ind.get("em_regime", "unknown")
    em_sc    = ind.get("em_stress_score", 0)
    eem      = ind.get("eem_1m_chg", np.nan)
    em_eq_rg = ind.get("em_equity_regime", "unknown")
    fx_avg   = ind.get("em_fx_stress_avg", np.nan)
    fx_det   = ind.get("fx_res_deteriorating", False)
    worst    = ind.get("fx_res_worst_country", "")
    usd_vuln = ind.get("usd_debt_vulnerability", "unknown")
    dollar_rg= ind.get("dollar_regime", "unknown")

    em_fx = ind.get("em_fx_1m_depreciation", {})
    fx_rows = []
    for ccy, chg in em_fx.items():
        sign = "⚠️" if chg > 0.03 else ""
        fx_rows.append(f"  USD/{ccy}: {_pct(chg)} 1M  {sign}")

    rows = [
        f"EM Regime        : {em_rg.upper()}  (stress score: {em_sc}/3)  (source: Yahoo Finance + World Bank)",
        f"EM Equity (EEM)  : {_pct(eem)} 1M  [{em_eq_rg.upper()}]  (source: Yahoo Finance)",
        f"EM FX Avg Depr.  : {_pct(fx_avg)}  (source: Yahoo Finance)",
        f"FX Reserves (WB) : {'⚠️ Drawdown detected — ' + worst if fx_det else 'No acute drawdown'}  (source: World Bank)",
        "",
        "EM FX Pairs (1M change — Yahoo Finance):",
    ] + fx_rows + [
        "",
        "Africa Eurobond Maturities 2025–2027: Ghana · Kenya · Ethiopia · Egypt (monitor closely)",
        "⚫ EMBI Sovereign Spreads : PLACEHOLDER — JPMorgan EMBI (Bloomberg required)",
    ]

    level = "alert" if em_rg == "crisis" else "warning" if em_rg in ("stress","elevated") else "info"
    return {"title": "3. EM & AFRICA STRESS UPDATE", "rows": rows, "level": level}


def _sec_sector(sectors: dict) -> dict:
    """Section 4: Sector Intelligence Update."""
    rows = []
    for name, data in sectors.items():
        status   = data.get("status", "no_data")
        headline = data.get("headline", "No data")
        icon     = "🔴" if status in ("crisis", "contraction") else \
                   "🟠" if status in ("stress", "slowing", "stressed") else \
                   "🟢" if status in ("normal", "comfortable", "expanding") else "⚫"
        rows.append(f"{icon} {name:<18}: {headline}")
        for ph in data.get("placeholders", []):
            rows.append(f"   ⚫ PLACEHOLDER: {ph}")
    if not rows:
        rows = ["No sector data available — check API connections."]
    return {"title": "4. SECTOR INTELLIGENCE UPDATE", "rows": rows, "level": "info"}


def _sec_market_implied(ind: dict) -> dict:
    """Section 5: Market-Implied Forward Layer."""
    vix_ts   = ind.get("vix_term_structure", "unknown")
    cu_gold  = ind.get("copper_gold_regime", "unknown")
    sys_str  = ind.get("systemic_stress_signal", False)
    hyg      = ind.get("hyg_1m_chg", np.nan)
    emb      = ind.get("emb_1m_chg", np.nan)
    tip      = ind.get("tip_1m_chg", np.nan)
    xlf      = ind.get("xlf_1m_chg", np.nan)
    xle      = ind.get("xle_1m_chg", np.nan)
    xlb      = ind.get("xlb_1m_chg", np.nan)
    xli      = ind.get("xli_1m_chg", np.nan)
    copper   = ind.get("copper", np.nan)
    gold     = ind.get("gold", np.nan)
    gold_1m  = ind.get("gold_1m_chg", np.nan)
    nickel   = ind.get("nickel", np.nan)
    nickel_1m= ind.get("nickel_1m_chg", np.nan)

    rows = [
        f"VIX Term Structure : {vix_ts.upper()}  (source: Yahoo Finance rolling avg)",
        f"Cu/Gold Regime     : {cu_gold.upper()}  (source: Yahoo Finance)",
        f"Systemic Stress    : {'⚠️ YES — all safe havens bid' if sys_str else 'No'}  (source: Yahoo Finance)",
        "",
        "ETF signals (Yahoo Finance):",
        f"  HYG (HY bonds)   : {_pct(hyg)} 1M",
        f"  EMB (EM bonds)   : {_pct(emb)} 1M",
        f"  TIP (TIPS)       : {_pct(tip)} 1M",
        f"  XLF (Financials) : {_pct(xlf)} 1M",
        f"  XLE (Energy)     : {_pct(xle)} 1M",
        f"  XLB (Materials)  : {_pct(xlb)} 1M",
        f"  XLI (Industrials): {_pct(xli)} 1M",
        "",
        "Commodity signals (Yahoo Finance):",
        f"  Gold             : {_val(gold, '.0f', ' $/oz')}  1M: {_pct(gold_1m)}",
        f"  Copper (HG=F)    : {_val(copper, '.2f', ' $/lb')}",
        f"  Nickel (NI=F)    : {_val(nickel, '.0f', ' $/MT')}  1M: {_pct(nickel_1m)}",
        "",
        "⚫ ICE MOVE Index   : PLACEHOLDER — ICE Data Services (subscription required)",
        "⚫ EMBI Spreads     : PLACEHOLDER — JPMorgan EMBI (Bloomberg required)",
        "⚫ Skew / Vol surf  : PLACEHOLDER — options data not available free",
    ]
    level = "warning" if sys_str else "info"
    return {"title": "5. MARKET-IMPLIED FORWARD LAYER", "rows": rows, "level": level}


def _sec_rules(rules: list[dict]) -> dict:
    """Section 6: Rules Engine Insights."""
    rows = []
    icons = {"alert": "🔴", "warning": "🟠", "info": "🔵"}
    for r in rules:
        icon = icons.get(r.get("level", "info"), "⚪")
        rows.append(f"{icon} [{r.get('level','info').upper()}] {r.get('category','')} — {r.get('headline','')}")
    if not rows:
        rows = ["No rules fired — all indicators normal or unavailable."]
    level = "alert" if any(r.get("level") == "alert" for r in rules) else \
            "warning" if any(r.get("level") == "warning" for r in rules) else "good"
    return {"title": "6. RULES ENGINE INSIGHTS", "rows": rows, "level": level}


def _sec_inference(inferences: list[dict]) -> dict:
    """Section 7: Inference Engine Forward-Looking Statements."""
    rows = []
    conf_icons = {"high": "🔴", "medium": "🟠", "low": "⚪"}
    for inf in inferences:
        conf = inf.get("confidence", "low")
        icon = conf_icons.get(conf, "⚪")
        srcs = ", ".join(inf.get("data_sources", []))
        rows.append(f"{icon} [{conf.upper()}] {inf.get('category','')} · {inf.get('horizon','')}")
        rows.append(f"   {inf.get('statement','')}")
        rows.append(f"   → {inf.get('implication','')}  (sources: {srcs})")
        rows.append("")
    if not rows:
        rows = ["No forward-looking signals — macro regime stable."]
    level = "alert" if any(i.get("confidence") == "high" for i in inferences) else "info"
    return {"title": "7. INFERENCE ENGINE — FORWARD-LOOKING", "rows": rows, "level": level}


def _sec_propagation(propagations: list[dict]) -> dict:
    """Section 8: Sector Dependency Propagation Insights."""
    rows = []
    for prop in propagations:
        icon  = "🔗"
        strength = prop.get("strength", "")
        rows.append(f"{icon} {prop.get('from_sector','')} → {prop.get('to_sector','')}  [{strength.upper()} linkage]")
        rows.append(f"   {prop.get('headline','')}")
        rows.append(f"   Mechanism: {prop.get('mechanism','')}")
        rows.append(f"   Signal: {prop.get('signal','')}")
        rows.append(f"   → {prop.get('implication','')}  (sources: {', '.join(prop.get('data_sources',[]))})")
        rows.append("")
    if not rows:
        rows = ["No active cross-sector propagation chains. Sectors broadly stable."]
    level = "warning" if any(p.get("strength") == "strong" for p in propagations) else "info"
    return {"title": "8. SECTOR DEPENDENCY PROPAGATION", "rows": rows, "level": level}


def _sec_watch_next(rules: list[dict], inferences: list[dict],
                    propagations: list[dict], hourly: bool = False) -> dict:
    """Section 9: What to Watch Next."""
    items: list[str] = []

    # Near-term high-confidence inferences first
    for inf in inferences:
        if inf.get("confidence") == "high" and inf.get("horizon") == "near-term (1-4 weeks)":
            items.append(f"⚡ Trigger: {inf.get('trigger','')} [{inf.get('category','')}]")

    # Alert-level rule watch items
    for r in [x for x in rules if x.get("level") == "alert"]:
        for item in r.get("watch", []):
            if item not in items:
                items.append(f"🔴 {item}")

    # Warning watch items
    for r in [x for x in rules if x.get("level") == "warning"]:
        for item in r.get("watch", []):
            if item not in items:
                items.append(f"🟠 {item}")

    # Sector propagation early warnings
    for prop in [p for p in propagations if p.get("strength") == "strong"]:
        item = f"🔗 Sector chain: {prop.get('from_sector','')} → {prop.get('to_sector','')}"
        if item not in items:
            items.append(item)

    # Always-on items
    always_on = [
        "📅 FAO monthly Food Price Index release",
        "📅 EIA weekly petroleum status report (Wed ~10:30 ET)",
        "📅 EU gas storage injection/withdrawal pace (AGSI+ daily)",
        "📅 FX reserve import cover (Ghana, Kenya, Ethiopia, Egypt)",
        "📅 IMF program review status (Ghana, Kenya, Ethiopia)",
        "📅 Eurobond maturity wall 2025–2027 — spread watch",
    ]
    if hourly:
        always_on = [
            "⏱️ US equity open / Asia close momentum",
            "⏱️ EIA oil/gas intraday news flow",
            "⏱️ Fed speaker watch",
            "⏱️ EM FX intraday moves vs DXY",
        ]

    for item in always_on:
        if item not in items:
            items.append(item)

    label = "next 6 hours" if hourly else "next 6–24 hours"
    rows = [f"Monitor for {label}:"] + [f"  • {item}" for item in items[:12]]
    title = "9. WHAT TO WATCH NEXT" if not hourly else "4. WHAT TO WATCH NEXT (hourly)"
    return {"title": title, "rows": rows, "level": "info"}


# ══════════════════════════════════════════════════════════════════════════════
# Main public functions
# ══════════════════════════════════════════════════════════════════════════════

def daily_note(
    ind:          dict,
    rules:        list[dict],
    inferences:   list[dict],
    propagations: list[dict],
    sectors:      dict,
    as_of:        Optional[datetime.datetime] = None,
) -> dict:
    """
    Generate the full daily macro intelligence note.
    Returns dict with sections list and metadata.
    """
    as_of   = as_of or datetime.datetime.utcnow()
    as_of_s = as_of.strftime("%Y-%m-%d %H:%M UTC")
    regime  = ind.get("macro_regime", "unknown")
    fin_sc  = ind.get("financial_stress_score", 0)
    geo_sc  = ind.get("geo_stress_score", 0)

    sections = [
        _sec_macro(ind),
        _sec_georisk(ind),
        _sec_em_africa(ind),
        _sec_sector(sectors),
        _sec_market_implied(ind),
        _sec_rules(rules),
        _sec_inference(inferences),
        _sec_propagation(propagations),
        _sec_watch_next(rules, inferences, propagations, hourly=False),
    ]

    n_alerts   = sum(1 for r in rules if r.get("level") == "alert")
    n_warnings = sum(1 for r in rules if r.get("level") == "warning")
    n_high_inf = sum(1 for i in inferences if i.get("confidence") == "high")

    headline = (
        f"{'🔴 CRISIS' if regime == 'crisis' else '🟠 STRESSED' if regime == 'stressed' else '🟡 CAUTIOUS' if regime == 'cautious' else '🟢 BENIGN'}"
        f" | Fin stress: {fin_sc}/7 | Geo-risk: {geo_sc}/4"
        f" | {n_alerts} alert(s), {n_warnings} warning(s), {n_high_inf} high-confidence forward signal(s)"
    )

    return {
        "type":     "daily",
        "as_of":    as_of_s,
        "regime":   regime,
        "headline": headline,
        "sections": sections,
        "footer":   (
            f"DAILY INTELLIGENCE NOTE — {as_of_s} | Real data: FRED · Yahoo Finance · "
            "EIA · AGSI+ · FAO · World Bank · IMF · OECD | "
            "PLACEHOLDERs labelled explicitly | Not investment advice."
        ),
    }


def hourly_note(
    ind:        dict,
    rules:      list[dict],
    inferences: list[dict],
    as_of:      Optional[datetime.datetime] = None,
) -> dict:
    """
    Generate a condensed hourly macro snapshot (4 sections).
    """
    as_of   = as_of or datetime.datetime.utcnow()
    as_of_s = as_of.strftime("%Y-%m-%d %H:%M UTC")
    regime  = ind.get("macro_regime", "unknown")

    # Condensed macro block
    slope   = ind.get("curve_slope", np.nan)
    vix     = ind.get("vix", np.nan)
    dxy     = ind.get("dxy", np.nan)
    brent   = ind.get("brent", np.nan)
    eu_gas  = ind.get("eu_gas_storage_pct", np.nan)
    fao_fpi = ind.get("fao_fpi", np.nan)
    hy      = ind.get("hy_spread", np.nan)
    oecd    = ind.get("oecd_cli_oecdall", np.nan)

    macro_rows = [
        f"Regime: {regime.upper()}  |  Fin stress: {ind.get('financial_stress_score',0)}/7  |  Geo: {ind.get('geo_stress_score',0)}/4",
        "",
        f"Curve slope: {_val(slope, '+.0f', ' bps')} [{ind.get('curve_regime','?').upper()}]"
        f"  |  VIX: {_val(vix, '.1f')} [{ind.get('vix_regime','?').upper()}]",
        f"DXY: {_val(dxy, '.1f')} [{ind.get('dollar_regime','?').upper()}]"
        f"  |  HY: {_val(hy, '.0f', ' bps')} [{ind.get('hy_regime','?').upper()}]",
        f"Brent: {_val(brent, '.1f', ' $/bbl')} [{ind.get('oil_regime','?').upper()}]"
        f"  |  EU Gas: {_val(eu_gas, '.1f', '% full')} [{ind.get('eu_gas_storage_regime','?').upper()}]",
        f"FAO FPI: {_val(fao_fpi, '.0f')} [{ind.get('fao_fpi_regime','?').upper()}]"
        f"  |  OECD CLI: {_val(oecd, '.1f')} [{ind.get('oecd_cli_regime','?').upper()}]",
    ]

    # Active alerts only
    alert_rows = []
    for r in [x for x in rules if x.get("level") in ("alert", "warning")][:5]:
        icon = "🔴" if r.get("level") == "alert" else "🟠"
        alert_rows.append(f"{icon} {r.get('category','')} — {r.get('headline','')}")
    if not alert_rows:
        alert_rows = ["🟢 No active alerts or warnings — regime stable."]

    # High-confidence near-term inferences
    fwd_rows = []
    for inf in [i for i in inferences if i.get("confidence") == "high"
                and i.get("horizon") == "near-term (1-4 weeks)"][:4]:
        fwd_rows.append(f"🔴 {inf.get('category','')} — {inf.get('statement','')}")
    if not fwd_rows:
        fwd_rows = ["No near-term high-confidence signals."]

    sections = [
        {"title": "1. SNAPSHOT — REAL-TIME MACRO", "rows": macro_rows,
         "level": {"crisis":"alert","stressed":"warning","cautious":"warning","benign":"good"}.get(regime,"info")},
        {"title": "2. ACTIVE ALERTS (rules engine)", "rows": alert_rows,
         "level": "alert" if any(r.get("level")=="alert" for r in rules) else "warning"},
        {"title": "3. NEAR-TERM FORWARD SIGNALS (inference engine)", "rows": fwd_rows,
         "level": "alert" if fwd_rows and fwd_rows[0].startswith("🔴") else "info"},
        _sec_watch_next(rules, inferences, [], hourly=True),
    ]

    n_alerts = sum(1 for r in rules if r.get("level") == "alert")
    regime_emoji = {"crisis": "🔴", "stressed": "🟠", "cautious": "🟡", "benign": "🟢"}.get(regime, "⚪")
    headline = (
        f"{regime_emoji} {regime.upper()} — Hourly snapshot | "
        f"{n_alerts} alert(s) active"
    )

    return {
        "type":     "hourly",
        "as_of":    as_of_s,
        "regime":   regime,
        "headline": headline,
        "sections": sections,
        "footer":   (
            f"HOURLY INTELLIGENCE NOTE — {as_of_s} | "
            "Real data: FRED · Yahoo Finance · EIA · AGSI+ · FAO · World Bank · OECD | "
            "PLACEHOLDERs labelled | Not investment advice."
        ),
    }


def to_text(note: dict) -> str:
    """Render a note dict to a crisp plain-text / monospace string."""
    lines = [
        _box_line("═"),
        f"  MACRO INTELLIGENCE — {note['type'].upper()} NOTE",
        f"  {note['as_of']}",
        _box_line("═"),
        f"  {note['headline']}",
        _box_line("─"),
        "",
    ]
    for sec in note.get("sections", []):
        lines.append(_box(sec["title"], sec["rows"], sec.get("level", "info")))
        lines.append("")
    lines += [
        _box_line("─"),
        f"  {note.get('footer','')}",
        _box_line("═"),
    ]
    return "\n".join(lines)
