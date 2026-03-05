"""
rules_engine.py — Simple regime-state insights.

Takes the indicators dict → returns list of insight dicts describing CURRENT conditions.
Each insight: level, category, headline, detail, watch.

For FORWARD-LOOKING conditional logic → see inference_engine.py
"""

from __future__ import annotations

import numpy as np

from config import LEVEL_ORDER, THRESH


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _pct(v, dec=1):
    if _nan(v): return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{dec}f}%"


def run(ind: dict) -> list[dict]:
    """
    Execute all rules and return sorted list of insights (alerts → warnings → info).
    """
    out: list[dict] = []

    def add(level, category, headline, detail, watch):
        out.append({"level": level, "category": category,
                    "headline": headline, "detail": detail, "watch": watch})

    slope      = ind.get("curve_slope",  np.nan)
    curve_rg   = ind.get("curve_regime", "normal")
    direction  = ind.get("curve_direction", "")
    vix        = ind.get("vix", np.nan)
    vix_rg     = ind.get("vix_regime", "normal")
    dxy        = ind.get("dxy", np.nan)
    dollar_rg  = ind.get("dollar_regime", "neutral")
    oil_rg     = ind.get("oil_regime", "stable")
    oil_chg    = ind.get("oil_1m_chg", np.nan)
    cu_rg      = ind.get("copper_regime", "stable")
    cu_chg     = ind.get("copper_1m_chg", np.nan)
    gold_rg    = ind.get("gold_regime", "stable")
    gold_chg   = ind.get("gold_1m_chg", np.nan)
    gpr_rg     = ind.get("gpr_regime", "normal")
    gpr        = ind.get("gpr", np.nan)
    ship_rg    = ind.get("shipping_regime", "normal")
    em_rg      = ind.get("em_regime", "normal")
    embi       = ind.get("embi", np.nan)
    africa_sp  = ind.get("africa_spreads", np.nan)
    em_wide    = ind.get("em_spreads_widening", False)
    fx_det     = ind.get("fx_res_deteriorating", False)
    worst      = ind.get("fx_res_worst_country", "")
    fin_cond   = ind.get("fin_cond_regime", "neutral")
    hy         = ind.get("hy_spread", np.nan)
    hy_rg      = ind.get("hy_regime", "normal")
    be5        = ind.get("breakeven5y", np.nan)
    be55       = ind.get("breakeven5y5y", np.nan)
    inf_rg     = ind.get("inflation_regime", "normal")
    tp         = ind.get("term_premium", np.nan)
    corr_rg    = ind.get("correlation_regime", "normal")
    move       = ind.get("move_proxy", np.nan)
    sys_stress = ind.get("systemic_stress_signal", False)
    san_elev   = ind.get("sanctions_elevated", False)
    fin_score  = ind.get("financial_stress_score", 0)
    geo_score  = ind.get("geo_stress_score", 0)

    # ── 1. Yield curve ────────────────────────────────────────────────────────
    if not _nan(slope):
        if curve_rg == "inverted":
            add("alert", "Rates Regime",
                f"Yield curve inverted ({slope:+.0f} bps, {direction})",
                (f"A {slope:.0f} bps inversion has preceded every US recession since 1960 "
                 f"(12–18 month lead). The curve is {direction}, which "
                 + ("may signal early rate-cut pricing — watch Fed guidance."
                    if direction == "steepening" else
                    "confirms ongoing credit tightening.")),
                ["Fed Funds futures curve", "Credit spreads (IG/HY)", "Bank lending standards",
                 "Labor market prints"])
        elif curve_rg == "flat":
            add("warning", "Rates Regime",
                f"Yield curve flat ({slope:+.0f} bps, {direction})",
                ("Flat curve compresses bank margins and signals slowing growth pricing. "
                 + ("Steepening from flat often marks onset of rate-cut cycle." if direction == "steepening"
                    else "Continued flattening elevates recession probability.")),
                ["2Y–10Y spread trend", "Fed meeting minutes", "Bank earnings", "Mortgage volumes"])
        elif curve_rg == "steep":
            add("info", "Rates Regime",
                f"Yield curve steep ({slope:+.0f} bps) — reflation or fiscal premium?",
                ("Steep curve can mean robust growth pricing OR rising term premium from fiscal concerns. "
                 "Distinguish by checking whether TIPS real yields or breakeven inflation drives it."),
                ["5Y5Y breakeven", "TIPS real yields", "Treasury auction demand", "Fiscal deficit"])

    # ── 2. VIX / Volatility ───────────────────────────────────────────────────
    if not _nan(vix):
        if vix_rg == "high":
            add("alert", "Market Stress / VIX",
                f"VIX in high-stress territory ({vix:.1f})",
                ("VIX >35 reflects genuine fear. Expect: USD safe-haven bid, EM capital outflows, "
                 "commodity demand destruction, credit spread widening. "
                 "Risk-reward skewed defensively."),
                ["EM capital flows (IIF weekly)", "HY/IG spread ratio",
                 "Safe-haven FX (JPY, CHF, Gold)", "Commodity spot vs futures"])
        elif vix_rg == "elevated":
            add("warning", "Market Stress / VIX",
                f"Volatility elevated (VIX {vix:.1f})",
                ("Above-normal VIX signals hedging demand and tail-risk pricing. "
                 "Monitor cross-asset correlations — if bonds and equities sell off together, "
                 "a full risk-off regime transition may be underway."),
                ["Options skew", "Put/call ratio", "VIX term structure", "EM positioning"])

    # ── 3. Dollar regime ──────────────────────────────────────────────────────
    if dollar_rg in ("strong", "very_strong") and not _nan(dxy):
        em_note = (" Combined with widening EM spreads, refinancing risk is acute."
                   if em_rg in ("stress", "crisis") else "")
        add("alert" if dollar_rg == "very_strong" and em_rg in ("stress","crisis") else "warning",
            "Dollar / EM Capital",
            f"{'Very strong' if dollar_rg == 'very_strong' else 'Strong'} USD (DXY {dxy:.1f})",
            ("Strong dollar mechanically tightens global financial conditions. "
             "EM sovereigns with USD-denominated debt face higher real debt-service costs. "
             "Commodity prices (USD-priced) face headwinds." + em_note),
            ["EM FX reserves", "Eurobond maturity wall 2025-2027",
             "IMF program requests", "Commodity prices in local currency"])
    elif dollar_rg == "weakening":
        add("info", "Dollar / EM Capital",
            f"USD weakening (DXY {dxy:.1f}) — EM tailwind",
            ("Dollar weakness eases global financial conditions. "
             "Expect rotation into EM equities and local-currency bonds. "
             "Commodity prices (gold, oil, copper) also benefit."),
            ["EM equity ETF inflows (EEM)", "EM local bond yields",
             "Gold and commodity breakouts", "Fed policy trajectory"])

    # ── 4. Geo-risk ───────────────────────────────────────────────────────────
    if gpr_rg in ("elevated", "high"):
        extras = []
        if ship_rg in ("stress", "crisis"): extras.append("shipping disruption")
        if oil_rg == "surging":             extras.append("oil price surge")
        if san_elev:                         extras.append("elevated sanctions")
        level = "alert" if (gpr_rg == "high" or len(extras) >= 2) else "warning"
        add(level, "Geo-Risk / Supply Chains",
            f"Geo-risk {'high' if gpr_rg == 'high' else 'elevated'} (GPR {gpr:.0f})"
            + (f" + {', '.join(extras)}" if extras else ""),
            (f"GPR at {gpr:.0f} (baseline=100) reflects conflict and policy uncertainty. "
             + ("Shipping disruption is transmitting through supply chains. " if ship_rg in ("stress","crisis") else "") +
             ("Oil surge adds stagflationary pressure. " if oil_rg == "surging" else "") +
             "Elevated GPR correlates with lower investment, higher energy prices, "
             "and accelerated defense spending."),
            ["Red Sea/Hormuz shipping rates", "LNG contract premiums",
             "Defense procurement", "FDI redirection flows"])

    # ── 5. Commodity regime ───────────────────────────────────────────────────
    if cu_rg == "rising" and not _nan(cu_chg):
        oil_ctx = (" Oil stability suggests demand-pull not supply inflation." if oil_rg == "stable"
                   else " Rising oil alongside copper = broad reflation.")
        add("info", "Growth Signal (Copper)",
            f"Copper +{cu_chg*100:.1f}% (1M) — positive industrial signal",
            ("Copper leads industrial activity by 3–6 months. Rising copper signals: "
             "improving manufacturing PMI, infrastructure demand, EV supply-chain acceleration." + oil_ctx),
            ["China Caixin PMI", "EV production schedules", "LME copper inventories", "BHP/Freeport guidance"])
    elif cu_rg == "falling" and not _nan(cu_chg):
        add("warning", "Growth Signal (Copper)",
            f"Copper {cu_chg*100:.1f}% (1M) — industrial slowdown signal",
            ("Falling copper anticipates PMI deterioration by 3–6 months. "
             "EM commodity exporters (DRC, Chile, Zambia) most exposed."),
            ["China NBS/Caixin PMI", "EM commodity-exporter FX", "Mining capex", "Container volumes"])

    if oil_rg == "surging" and not _nan(oil_chg):
        add("warning", "Energy / Inflation",
            f"Oil surging (+{oil_chg*100:.1f}% in 1M)",
            ("Sharp oil increase raises inflation persistence. Central banks face harder trade-off. "
             "Net oil importers (India, Turkey, SSA) face CA and currency pressure."),
            ["CPI surprise prints", "Central bank reaction", "SSA/Turkey CA balance", "Oil futures curve"])

    if gold_rg == "surging" and not _nan(gold_chg):
        add("warning" if not sys_stress else "alert",
            "Gold / Safe Haven",
            f"Gold surging (+{gold_chg*100:.1f}% in 1M)"
            + (" — ALL safe havens bid (systemic stress)" if sys_stress else ""),
            ("Gold surging alongside USD = flight to ALL safe havens — systemic stress signal. "
             if sys_stress else
             "Gold rising can reflect: risk-off fear, falling real rates, or structural central bank accumulation. ") +
            "Distinguish by checking TIPS real yields.",
            ["TIPS real yields", "Central bank gold purchases (WGC)", "EM reserve composition",
             "USD correlation breakdown"])

    # ── 6. EM / Africa stress ─────────────────────────────────────────────────
    if em_rg in ("stress", "crisis") or (em_wide and dollar_rg in ("strong", "very_strong")):
        africa_note = (f" Africa composite spread at {africa_sp:.0f} bps." if not _nan(africa_sp) else "")
        fx_note = (f" FX reserve drawdowns detected ({worst} most exposed)." if fx_det else "")
        add("alert" if em_rg == "crisis" else "warning",
            "EM / Africa Stress",
            (f"EM sovereign stress: EMBI {embi:.0f} bps"
             + (" (widening)" if em_wide else "")
             + (f" · Africa {africa_sp:.0f} bps" if not _nan(africa_sp) else "")),
            (f"EMBI at {embi:.0f} bps signals elevated refinancing risk. "
             + ("Strong USD amplifies dollar-denominated debt-service costs. " if dollar_rg in ("strong","very_strong") else "")
             + africa_note + fx_note +
             " Eurobond maturity walls 2025-2027 for Ghana, Kenya, Ethiopia, Egypt are critical."),
            ["IMF program status", "Eurobond maturities 2025-2027",
             "FX reserve import cover (<3M = critical)", "AfDB/World Bank budget support"])

    # ── 7. Financial conditions ───────────────────────────────────────────────
    if fin_cond == "tightening" or hy_rg in ("stress", "crisis"):
        add("warning" if hy_rg != "crisis" else "alert",
            "Financial Conditions",
            f"Financial conditions tightening (HY spread {hy:.0f} bps)" if not _nan(hy) else
            "Financial conditions tightening",
            ("HY spread widening signals credit stress spreading to riskier borrowers. "
             "Tighter conditions reduce investment, slow hiring, increase default risk. "
             "Leveraged loans and private credit portfolios most exposed."),
            ["HY default rates (Moody's)", "Leveraged loan spreads",
             "Bank lending standards", "Private credit NAV marks"])

    # ── 8. Inflation regime ────────────────────────────────────────────────────
    if inf_rg in ("high", "very_high"):
        add("warning" if inf_rg == "high" else "alert",
            "Inflation Expectations",
            f"Inflation breakevens elevated (5Y: {be5:.2f}%, 5Y5Y: {be55:.2f}%)"
            if not (_nan(be5) or _nan(be55)) else "Inflation breakevens elevated",
            ("Elevated breakevens signal persistent inflation pricing by bond markets. "
             "Combined with a flat/inverted curve, this signals a stagflationary mix — "
             "the most difficult environment for central bank policy."),
            ["Fed/ECB reaction function", "CPI core prints", "Wage data",
             "Commodity price trajectory"])

    # ── 9. Bond volatility (MOVE proxy) ───────────────────────────────────────
    if not _nan(move) and ind.get("move_regime") == "elevated":
        add("warning", "Bond Volatility",
            f"Bond volatility elevated (MOVE proxy: {move:.0f})",
            ("High bond volatility signals uncertainty about the rate path — "
             "typically accompanying fiscal stress, inflation surprises, or central bank pivots. "
             "Watch for duration positioning unwinds."),
            ["Treasury auction bid-to-cover", "Pension fund duration hedging",
             "Fed dot plot dispersion", "Breakeven vol"])

    # ── 10. Cross-asset correlation regime ────────────────────────────────────
    if corr_rg == "stress":
        add("warning", "Cross-Asset Regime",
            "Equity-bond correlation positive — stress regime",
            ("Positive equity-bond correlation (both selling off together) historically marks "
             "stagflationary or acute stress regimes where traditional 60/40 diversification breaks down."),
            ["Real rate trajectory", "Inflation premium vs risk premium",
             "Gold as alternative diversifier", "Portfolio risk decomposition"])

    # ── 11. Broad composite stress ────────────────────────────────────────────
    if fin_score >= 5:
        add("alert", "Cross-Asset Regime",
            f"Broad-based financial stress ({fin_score}/7 indicators elevated)",
            ("Multiple simultaneous stress indicators historically precede risk-asset drawdowns of 15-25%. "
             "Defensive positioning, cash, and duration/safe-haven assets are priorities."),
            ["Cross-asset correlation", "Repo/money market stress", "Portfolio margin calls",
             "Central bank emergency tool signals"])

    # ── Default: calm ─────────────────────────────────────────────────────────
    if not any(i["level"] in ("alert", "warning") for i in out):
        add("info", "Regime Assessment",
            "No major stress signals — broadly benign macro regime",
            ("Current cross-asset configuration is consistent with a mid-to-late cycle expansion. "
             "Watch for: yield curve slope changes, DXY trend, copper vs oil divergence, "
             "and any GPR acceleration as early warning signals."),
            ["Yield curve slope trend", "DXY direction", "Copper/oil divergence", "GPR acceleration"])

    out.sort(key=lambda x: LEVEL_ORDER.get(x["level"], 9))
    return out
