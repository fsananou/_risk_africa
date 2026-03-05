"""
rules_engine.py — Current-state regime insights from real data only.

Rules only fire when the relevant indicator is not 'unknown' or NaN.
Each insight: {level, category, headline, detail, watch}
"""

from __future__ import annotations

import numpy as np
from config import LEVEL_ORDER, THRESH


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _pct(v, dec=1) -> str:
    if _nan(v): return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{dec}f}%"


def run(ind: dict) -> list[dict]:
    """Execute all rules. Returns sorted list: alerts → warnings → info."""
    out: list[dict] = []

    def add(level, category, headline, detail, watch):
        out.append({"level": level, "category": category,
                    "headline": headline, "detail": detail, "watch": watch})

    slope      = ind.get("curve_slope", np.nan)
    curve_rg   = ind.get("curve_regime", "unknown")
    direction  = ind.get("curve_direction", "")
    vix        = ind.get("vix", np.nan)
    vix_rg     = ind.get("vix_regime", "unknown")
    dxy        = ind.get("dxy", np.nan)
    dollar_rg  = ind.get("dollar_regime", "unknown")
    oil_rg     = ind.get("oil_regime", "unknown")
    oil_1m     = ind.get("oil_1m_chg", np.nan)
    brent      = ind.get("brent", np.nan)
    cu_rg      = ind.get("copper_regime", "unknown")
    cu_1m      = ind.get("copper_1m_chg", np.nan)
    gold_rg    = ind.get("gold_regime", "unknown")
    gold_1m    = ind.get("gold_1m_chg", np.nan)
    hy         = ind.get("hy_spread", np.nan)
    hy_rg      = ind.get("hy_regime", "unknown")
    fin_cond   = ind.get("fin_cond_regime", "unknown")
    be5        = ind.get("breakeven5y", np.nan)
    be55       = ind.get("breakeven5y5y", np.nan)
    inf_rg     = ind.get("inflation_regime", "unknown")
    nfci       = ind.get("nfci", np.nan)
    move       = ind.get("move_proxy", np.nan)
    move_rg    = ind.get("move_regime", "unknown")
    corr_rg    = ind.get("correlation_regime", "unknown")
    sys_stress = ind.get("systemic_stress_signal", False)
    fin_score  = ind.get("financial_stress_score", 0)
    em_rg      = ind.get("em_regime", "unknown")
    fx_det     = ind.get("fx_res_deteriorating", False)
    worst      = ind.get("fx_res_worst_country", "")
    eu_gas_rg  = ind.get("eu_gas_storage_regime", "unknown")
    eu_gas_pct = ind.get("eu_gas_storage_pct", np.nan)
    oil_inv_rg = ind.get("oil_inventory_regime", "unknown")
    oil_inv_dev= ind.get("us_oil_inventory_dev", np.nan)
    fao_fpi    = ind.get("fao_fpi", np.nan)
    fao_rg     = ind.get("fao_fpi_regime", "unknown")
    fao_3m     = ind.get("fao_fpi_3m_chg", np.nan)
    agri_rg    = ind.get("agri_stress_regime", "unknown")
    indpro_rg  = ind.get("indpro_regime", "unknown")
    indpro_yoy = ind.get("indpro_yoy", np.nan)
    oecd_rg    = ind.get("oecd_cli_regime", "unknown")
    oecd_v     = ind.get("oecd_cli_oecdall", np.nan)

    # ── 1. Yield curve ────────────────────────────────────────────────────────
    if curve_rg not in ("unknown",) and not _nan(slope):
        if curve_rg == "inverted":
            add("alert", "Rates / Yield Curve",
                f"Yield curve inverted ({slope:+.0f} bps, {direction})",
                f"A {slope:.0f} bps inversion has preceded every US recession since 1960 "
                f"(12–18 month lead). Curve is {direction}. "
                + ("Steepening from inversion often marks rate-cut cycle onset."
                   if direction == "steepening" else "Continued inversion confirms credit tightening."),
                ["Fed Funds futures curve", "Credit spreads (IG/HY)", "Bank lending standards"])
        elif curve_rg == "flat":
            add("warning", "Rates / Yield Curve",
                f"Yield curve flat ({slope:+.0f} bps, {direction})",
                "Flat curve compresses bank margins and signals slowing growth. "
                + ("Steepening from flat often marks rate-cut cycle onset."
                   if direction == "steepening" else "Further flattening elevates recession risk."),
                ["2Y–10Y trend", "Fed guidance", "Bank NIM"])
        elif curve_rg == "steep":
            add("info", "Rates / Yield Curve",
                f"Yield curve steep ({slope:+.0f} bps) — growth or fiscal premium?",
                "Steep curve means robust growth pricing OR rising term premium from fiscal concerns. "
                "Check TIPS real yields vs breakeven inflation.",
                ["5Y5Y breakeven", "TIPS real yields", "Treasury auction demand"])

    # ── 2. VIX ────────────────────────────────────────────────────────────────
    if vix_rg not in ("unknown",) and not _nan(vix):
        if vix_rg == "high":
            add("alert", "Market Stress / VIX",
                f"VIX in high-stress territory ({vix:.1f})",
                "VIX >35 reflects genuine fear. Expect USD safe-haven bid, EM capital outflows, "
                "commodity demand destruction, credit spread widening.",
                ["EM capital flows", "HY/IG spread ratio", "Safe-haven FX"])
        elif vix_rg == "elevated":
            add("warning", "Market Stress / VIX",
                f"Volatility elevated (VIX {vix:.1f})",
                "Above-normal VIX signals hedging demand and tail-risk pricing. "
                "Monitor cross-asset correlations for full risk-off regime transition.",
                ["Options skew", "VIX term structure", "EM positioning"])

    # ── 3. Dollar ─────────────────────────────────────────────────────────────
    if dollar_rg not in ("unknown",) and not _nan(dxy):
        if dollar_rg in ("strong", "very_strong"):
            em_note = " Combined with EM stress, refinancing risk is acute." if em_rg in ("stress","crisis") else ""
            add("alert" if dollar_rg == "very_strong" and em_rg in ("stress","crisis") else "warning",
                "Dollar / EM Capital",
                f"{'Very strong' if dollar_rg == 'very_strong' else 'Strong'} USD (DXY {dxy:.1f})",
                "Strong dollar tightens global financial conditions. EM sovereigns with USD debt "
                "face higher debt-service costs. Commodity prices face headwinds." + em_note,
                ["EM FX reserves", "Eurobond maturity walls", "IMF program requests"])
        elif dollar_rg == "weakening":
            add("info", "Dollar / EM Capital",
                f"USD weakening (DXY {dxy:.1f}) — EM tailwind",
                "Dollar weakness eases global financial conditions. "
                "Rotation into EM equities and local-currency bonds historically follows.",
                ["EM equity ETF inflows", "EM local bond yields", "Commodity prices"])

    # ── 4. Energy / Oil ───────────────────────────────────────────────────────
    if oil_rg not in ("unknown",) and not _nan(oil_1m):
        if oil_rg == "surging":
            add("warning", "Energy / Oil",
                (f"Oil surging ({_pct(oil_1m)} 1M, Brent ${brent:.0f})" if not _nan(brent) else
                 f"Oil surging ({_pct(oil_1m)} 1M)"),
                "Sharp oil increase raises inflation persistence. Net oil importers (India, Turkey, SSA) "
                "face current account and currency pressure.",
                ["CPI prints", "Central bank reaction", "SSA CA balance"])
        elif oil_rg == "crashing":
            add("warning", "Energy / Oil",
                f"Oil crashing ({_pct(oil_1m)} 1M)",
                "Sharp oil decline may signal demand destruction or growth scare. "
                "Oil-exporting EM sovereigns (Nigeria, Angola, Algeria) face fiscal stress.",
                ["OPEC+ meeting", "Global PMI", "Nigeria/Angola fiscal position"])

    if oil_inv_rg not in ("unknown",) and not _nan(oil_inv_dev):
        if oil_inv_rg == "stress":
            add("warning", "Energy / Oil Inventories",
                f"US crude inventories {_pct(oil_inv_dev)} below 5Y average (EIA)",
                "Tight US crude inventories reduce buffer against supply shocks "
                "and support higher spot prices.",
                ["EIA Weekly Petroleum Status", "OPEC+ quota compliance", "Cushing stocks"])
        elif oil_inv_rg == "surplus":
            add("info", "Energy / Oil Inventories",
                f"US crude inventories {_pct(oil_inv_dev)} above 5Y average (EIA)",
                "Elevated inventories provide supply buffer and may cap oil price upside.",
                ["EIA Weekly Petroleum Status", "Refinery utilization"])

    # ── 5. EU Gas Storage (AGSI+) ─────────────────────────────────────────────
    if eu_gas_rg not in ("unknown",) and not _nan(eu_gas_pct):
        if eu_gas_rg == "crisis":
            add("alert", "Energy / EU Gas",
                f"EU gas storage critically low ({eu_gas_pct:.1f}% full) — AGSI+",
                "EU gas storage below 20% is a supply security crisis. "
                "Expect industrial curtailments, energy price spikes, fertilizer production cuts.",
                ["EU emergency storage regulation", "LNG spot imports", "Industrial curtailments"])
        elif eu_gas_rg == "stress":
            add("warning", "Energy / EU Gas",
                f"EU gas storage below normal ({eu_gas_pct:.1f}% full) — AGSI+",
                "Below-normal EU gas storage elevates winter energy security risk "
                "and supports higher gas prices. Fertilizer margins under pressure.",
                ["EU storage refill pace", "LNG import capacity", "Gas-to-coal switching"])
        elif eu_gas_rg == "comfortable":
            add("info", "Energy / EU Gas",
                f"EU gas storage comfortable ({eu_gas_pct:.1f}% full) — AGSI+",
                "Well-filled EU storage reduces near-term energy security risk.",
                ["Injection/withdrawal rate", "Demand trends"])

    # ── 6. Copper ─────────────────────────────────────────────────────────────
    if cu_rg not in ("unknown",) and not _nan(cu_1m):
        if cu_rg == "rising":
            add("info", "Growth Signal / Copper",
                f"Copper {_pct(cu_1m)} (1M) — positive industrial signal",
                "Copper leads industrial activity by 3–6 months. Rising copper signals "
                "improving PMI, infrastructure demand, EV supply chain acceleration.",
                ["China Caixin PMI", "EV production schedules", "LME inventories"])
        elif cu_rg == "falling":
            add("warning", "Growth Signal / Copper",
                f"Copper {_pct(cu_1m)} (1M) — industrial slowdown signal",
                "Falling copper anticipates PMI deterioration by 3–6 months. "
                "EM commodity exporters (DRC, Chile, Zambia) most exposed.",
                ["China NBS/Caixin PMI", "EM commodity-exporter FX", "Mining capex"])

    # ── 7. Gold ───────────────────────────────────────────────────────────────
    if gold_rg not in ("unknown",) and not _nan(gold_1m) and gold_rg in ("rising","surging"):
        add("warning" if not sys_stress else "alert",
            "Gold / Safe Haven",
            f"Gold {_pct(gold_1m)} (1M)"
            + (" — ALL safe havens bid (systemic stress)" if sys_stress else ""),
            ("Gold surging alongside USD = flight to ALL safe havens — systemic stress. "
             if sys_stress else
             "Gold rising may reflect: risk-off fear, falling real rates, or CB accumulation. ") +
            "Distinguish by checking TIPS real yields.",
            ["TIPS real yields", "Central bank gold purchases (WGC)", "EM reserve composition"])

    # ── 8. EM / Africa ────────────────────────────────────────────────────────
    if em_rg not in ("unknown",) and em_rg in ("stress","crisis"):
        fx_note = f" FX reserve drawdowns: {worst} most exposed." if fx_det and worst else ""
        add("alert" if em_rg == "crisis" else "warning",
            "EM / Africa Stress",
            "EM sovereign stress: multiple indicators elevated",
            f"EM stress signals: equity decline, FX depreciation, reserve pressure.{fx_note} "
            "Eurobond maturity walls 2025-2027 for Ghana, Kenya, Ethiopia, Egypt are critical.",
            ["IMF program status", "Eurobond maturities 2025-2027",
             "FX reserve import cover (<3M = critical)", "AfDB/World Bank support"])
    if fx_det:
        add("warning", "EM / FX Reserves",
            f"FX reserve drawdown detected — {worst or 'multiple countries'} (World Bank)",
            "Declining FX reserves reduce import cover and raise sovereign financing risk. "
            "Below 3 months of imports = critical vulnerability.",
            ["Import cover ratio", "IMF Article IV", "Eurobond yields"])

    # ── 9. Financial conditions ───────────────────────────────────────────────
    if fin_cond not in ("unknown",) and fin_cond in ("tightening","crisis"):
        add("alert" if fin_cond == "crisis" else "warning",
            "Financial Conditions",
            f"Financial conditions tightening"
            + (f" (NFCI {nfci:.2f})" if not _nan(nfci) else "")
            + (f", HY spread {hy:.0f} bps" if not _nan(hy) else ""),
            "HY spread widening spreads credit stress to riskier borrowers. "
            "Tighter conditions reduce investment, slow hiring, increase default risk.",
            ["HY default rates (Moody's)", "Bank lending standards", "Private credit"])

    # ── 10. Inflation ─────────────────────────────────────────────────────────
    if inf_rg not in ("unknown",) and inf_rg in ("high","very_high"):
        be_val = be5 if not _nan(be5) else be55
        if not _nan(be_val):
            add("warning" if inf_rg == "high" else "alert",
                "Inflation Expectations",
                f"Inflation breakevens elevated ({be_val:.2f}%) — FRED",
                "Elevated breakevens signal persistent inflation pricing by bond markets. "
                "Combined with inverted/flat curve, signals stagflation.",
                ["Fed/ECB reaction function", "CPI core prints", "Wage data"])

    # ── 11. Agricultural / Food ───────────────────────────────────────────────
    if fao_rg not in ("unknown",) and fao_rg in ("stress","crisis") and not _nan(fao_fpi):
        add("alert" if fao_rg == "crisis" else "warning",
            "Agriculture / Food Prices",
            f"FAO Food Price Index elevated ({fao_fpi:.0f}, 2014-16=100) — FAO",
            "High food prices drive inflation in EM countries where food = 40-60% of CPI. "
            + (f"3M change: {_pct(fao_3m)}. " if not _nan(fao_3m) else "") +
            "Combined with USD strength, triggers EM balance-of-payments stress.",
            ["FAO monthly FPI release", "Wheat/corn futures", "EM food import bills"])

    if agri_rg not in ("unknown",) and agri_rg in ("stress","crisis","warning"):
        wheat_1m = ind.get("wheat_1m_chg", np.nan)
        corn_1m  = ind.get("corn_1m_chg", np.nan)
        add("warning" if agri_rg != "crisis" else "alert",
            "Agriculture / Crop Prices",
            f"Crop price stress: wheat {_pct(wheat_1m)}, corn {_pct(corn_1m)} (1M) — yfinance",
            "Simultaneous wheat and corn price surges signal global supply disruption. "
            "EM food-importing countries most at risk.",
            ["USDA WASDE report", "Black Sea shipping routes", "Fertilizer prices"])

    # ── 12. Industrial production ─────────────────────────────────────────────
    if indpro_rg not in ("unknown",) and indpro_rg in ("contraction","slowing"):
        add("warning" if indpro_rg == "slowing" else "alert",
            "Industrials / Production",
            f"US industrial production {indpro_rg}" +
            (f" ({indpro_yoy:+.1f}% YoY) — FRED" if not _nan(indpro_yoy) else ""),
            "Slowing industrial production leads broad GDP by 1-2 quarters. "
            "Watch manufacturing PMI and new orders for confirmation.",
            ["ISM Manufacturing PMI", "Factory orders", "Capacity utilization"])

    # ── 13. OECD CLI ──────────────────────────────────────────────────────────
    if oecd_rg not in ("unknown",) and oecd_rg == "contracting" and not _nan(oecd_v):
        add("warning", "OECD / Growth",
            f"OECD CLI below trend ({oecd_v:.1f}) — OECD",
            "CLI below 100 signals growth momentum slowing across OECD economies. "
            "Historically precedes GDP deceleration by 3-6 months.",
            ["OECD CLI components", "PMI new orders", "Export orders"])

    # ── 14. Bond volatility ───────────────────────────────────────────────────
    if move_rg == "elevated" and not _nan(move):
        add("warning", "Bond Volatility",
            f"Bond volatility elevated (MOVE proxy {move:.0f}) — computed from FRED/Treasury yields",
            "High bond volatility signals rate path uncertainty — fiscal stress, "
            "inflation surprises, or central bank pivots.",
            ["Treasury auction bid-to-cover", "Fed dot plot dispersion"])

    # ── 15. Cross-asset correlation ───────────────────────────────────────────
    if corr_rg == "stress":
        add("warning", "Cross-Asset Regime",
            "Equity-bond correlation positive — stress regime (yfinance data)",
            "Positive equity-bond correlation marks stagflationary or acute stress regimes "
            "where 60/40 diversification breaks down.",
            ["Real rate trajectory", "Gold as diversifier", "Portfolio risk decomposition"])

    # ── 16. Broad composite ───────────────────────────────────────────────────
    if fin_score >= 5:
        add("alert", "Cross-Asset Regime",
            f"Broad-based financial stress ({fin_score}/7 indicators elevated)",
            "Multiple simultaneous stress indicators historically precede risk-asset drawdowns. "
            "Defensive positioning and safe-haven assets are priorities.",
            ["Cross-asset correlation", "Repo/money market", "Central bank emergency tools"])

    # ── Default: benign ───────────────────────────────────────────────────────
    if not any(i["level"] in ("alert","warning") for i in out):
        add("info", "Regime Assessment",
            "No major stress signals — broadly benign macro regime",
            "Current cross-asset configuration consistent with mid-to-late cycle expansion. "
            "Watch for: yield curve slope changes, DXY trend, copper vs oil divergence.",
            ["Yield curve slope", "DXY direction", "Copper/oil divergence", "OECD CLI"])

    out.sort(key=lambda x: LEVEL_ORDER.get(x["level"], 9))
    return out
