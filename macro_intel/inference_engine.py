"""
inference_engine.py — Conditional forward-looking logic (real data only).

Form: IF trigger_condition AND context_condition THEN expected_outcome.
Rules only fire when both trigger and context are based on available real data.

Each inference dict:
    confidence : "high" | "medium" | "low"
    horizon    : "near-term (1-4 weeks)" | "medium-term (1-6 months)" | "structural (6-18 months)"
    category   : str
    trigger    : str   (what fired the rule)
    context    : str   (amplifying condition)
    statement  : str   (the forward-looking claim)
    implication: str   (actionable so-what)
    icon       : str
    data_sources: list[str]
"""

from __future__ import annotations

import numpy as np
from config import CONF_COLOR, HORIZON_ICON, LEVEL_ORDER


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _known(v, *invalid) -> bool:
    """True if v is not NaN and not in the invalid set."""
    return not _nan(v) and v not in invalid


def run(ind: dict, rules: list[dict]) -> list[dict]:
    """
    Fire 25+ conditional forward-looking inferences from real indicator data.
    Returns list sorted by confidence: high → medium → low.
    """
    out: list[dict] = []

    def add(confidence, horizon, category, trigger, context, statement, implication,
            icon="📌", sources=None):
        out.append({
            "confidence":   confidence,
            "horizon":      horizon,
            "category":     category,
            "trigger":      trigger,
            "context":      context,
            "statement":    statement,
            "implication":  implication,
            "icon":         icon,
            "conf_color":   CONF_COLOR.get(confidence, "#7f8c8d"),
            "data_sources": sources or [],
        })

    curve_rg   = ind.get("curve_regime",       "unknown")
    direction  = ind.get("curve_direction",    "")
    slope      = ind.get("curve_slope",         np.nan)
    fin_cond   = ind.get("fin_cond_regime",    "unknown")
    vix_rg     = ind.get("vix_regime",         "unknown")
    dollar_rg  = ind.get("dollar_regime",      "unknown")
    oil_rg     = ind.get("oil_regime",         "unknown")
    oil_inv_rg = ind.get("oil_inventory_regime","unknown")
    cu_rg      = ind.get("copper_regime",      "unknown")
    gold_rg    = ind.get("gold_regime",        "unknown")
    sys_stress = ind.get("systemic_stress_signal", False)
    eu_gas_rg  = ind.get("eu_gas_storage_regime","unknown")
    us_gas_dev = ind.get("us_gas_storage_dev",  np.nan)
    fao_rg     = ind.get("fao_fpi_regime",     "unknown")
    agri_rg    = ind.get("agri_stress_regime", "unknown")
    inf_rg     = ind.get("inflation_regime",   "unknown")
    em_rg      = ind.get("em_regime",          "unknown")
    fx_det     = ind.get("fx_res_deteriorating", False)
    hy_rg      = ind.get("hy_regime",          "unknown")
    indpro_rg  = ind.get("indpro_regime",      "unknown")
    oecd_rg    = ind.get("oecd_cli_regime",    "unknown")
    corr_rg    = ind.get("correlation_regime", "unknown")
    em_eq_rg   = ind.get("em_equity_regime",   "unknown")
    cu_gold_rg = ind.get("copper_gold_regime", "unknown")
    move_rg    = ind.get("move_regime",        "unknown")
    fin_score  = ind.get("financial_stress_score", 0)
    geo_score  = ind.get("geo_stress_score",   0)

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK 1: Rates / Financial Conditions
    # ═══════════════════════════════════════════════════════════════════════════

    # R1 — Inverted curve + tightening = recession risk
    if _known(curve_rg, "unknown") and curve_rg == "inverted" and \
       _known(fin_cond, "unknown") and fin_cond in ("tightening","crisis"):
        add("high", "medium-term (1-6 months)",
            "Recession Risk",
            "Inverted yield curve",
            "Financial conditions tightening simultaneously",
            "Historical recession signal: inverted curve + tight credit = high recession probability in 12-18 months",
            "Reduce risk exposure; favour defensive sectors (utilities, healthcare, staples); increase duration.",
            "🔴", ["FRED", "yfinance"])

    # R2 — Inverted curve steepening = rate cut cycle onset
    if _known(curve_rg, "unknown") and curve_rg == "inverted" and direction == "steepening":
        add("high", "near-term (1-4 weeks)",
            "Rates / Policy",
            "Inverted curve now steepening",
            "Market pricing rate cuts",
            "Bull steepening from inversion historically marks the early phase of a Fed rate-cut cycle",
            "Long duration bonds attractive; financial sector equities historically outperform early easing.",
            "📈", ["FRED", "US Treasury Direct"])

    # R3 — Very high VIX + tightening = maximum risk-off
    if _known(vix_rg, "unknown") and vix_rg == "high" and \
       _known(fin_cond, "unknown") and fin_cond in ("tightening","crisis"):
        add("high", "near-term (1-4 weeks)",
            "Market Stress",
            "VIX in high-stress territory",
            "Financial conditions tightening",
            "Maximum risk-off configuration: expect EM capital outflows, credit spread widening, safe-haven demand",
            "Maximum defensive posture. Cash, gold, USD, short-duration bonds. Avoid EM and HY.",
            "🚨", ["yfinance", "FRED"])

    # R4 — Flat curve + high inflation = stagflation warning
    if _known(curve_rg, "unknown") and curve_rg in ("flat","inverted") and \
       _known(inf_rg, "unknown") and inf_rg in ("high","very_high"):
        add("high", "medium-term (1-6 months)",
            "Stagflation Risk",
            "Flat/inverted yield curve",
            "Elevated inflation breakevens (FRED)",
            "Flat/inverted curve + high inflation expectations = stagflationary mix — hardest environment for policy",
            "Central banks face impossible tradeoff. Short duration, commodities (oil, gold) outperform.",
            "🔥", ["FRED"])

    # R5 — NFCI tightening + HY stress = credit cycle turning
    if _known(hy_rg, "unknown") and hy_rg in ("stress","crisis") and \
       _known(fin_cond, "unknown") and fin_cond in ("tightening","crisis"):
        add("high", "medium-term (1-6 months)",
            "Credit Cycle",
            "HY spreads in stress",
            "NFCI financial conditions tightening",
            "Credit cycle turning: HY stress + tight conditions precede default rate spikes by 6-12 months",
            "Avoid leveraged credit. Favour IG and government bonds. Short HY ETFs (HYG) as hedge.",
            "💳", ["FRED"])

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK 2: Dollar / EM
    # ═══════════════════════════════════════════════════════════════════════════

    # R6 — Very strong USD + EM stress = EM crisis escalation
    if _known(dollar_rg, "unknown") and dollar_rg == "very_strong" and \
       _known(em_rg, "unknown") and em_rg in ("stress","crisis"):
        add("high", "medium-term (1-6 months)",
            "EM / Africa Crisis",
            "Very strong USD (yfinance)",
            "EM sovereign stress signals elevated",
            "Dollar at extreme + EM stress = elevated risk of sovereign debt crisis in frontier markets",
            "Monitor Eurobond maturities 2025-2027 for Ghana, Kenya, Ethiopia, Egypt. IMF programs critical.",
            "🌍", ["yfinance", "World Bank"])

    # R7 — USD strong + FX reserves falling = devaluation risk
    if _known(dollar_rg, "unknown") and dollar_rg in ("strong","very_strong") and fx_det:
        add("high", "near-term (1-4 weeks)",
            "EM / FX Stress",
            "Strong USD",
            "FX reserve drawdown detected (World Bank)",
            "Countries with falling FX reserves under USD strength face forced devaluation risk",
            "Monitor import cover ratios. Countries <3M of import cover face immediate IMF engagement risk.",
            "💱", ["yfinance", "World Bank"])

    # R8 — USD weakening + copper rising = EM risk-on window
    if _known(dollar_rg, "unknown") and dollar_rg == "weakening" and \
       _known(cu_rg, "unknown") and cu_rg == "rising":
        add("medium", "near-term (1-4 weeks)",
            "EM / Capital Flows",
            "USD weakening",
            "Copper rising (industrial demand signal)",
            "Dollar weakness + copper rally = classic EM risk-on window: capital inflows, local currency appreciation",
            "EM equity (EEM) and local currency bonds historically outperform. Africa commodity exporters benefit.",
            "🚀", ["yfinance"])

    # R9 — EM equity crisis + strong dollar = capital flight
    if _known(em_eq_rg, "unknown") and em_eq_rg in ("stress","crisis") and \
       _known(dollar_rg, "unknown") and dollar_rg in ("strong","very_strong"):
        add("high", "near-term (1-4 weeks)",
            "EM / Capital Flows",
            "EM equities in stress/crisis (EEM)",
            "Strong USD amplifying outflows",
            "EM equity stress + dollar strength = accelerating capital flight from EM",
            "EM assets likely to underperform. Monitor IIF weekly capital flow data for confirmation.",
            "📉", ["yfinance"])

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK 3: Energy / Gas Storage
    # ═══════════════════════════════════════════════════════════════════════════

    # R10 — EU gas low + oil surging = energy security crisis
    if _known(eu_gas_rg, "unknown") and eu_gas_rg in ("stress","crisis") and \
       _known(oil_rg, "unknown") and oil_rg == "surging":
        add("high", "near-term (1-4 weeks)",
            "Energy Security",
            "EU gas storage below normal (AGSI+)",
            "Oil surging simultaneously",
            "Dual energy stress: low EU gas storage + oil spike = severe European energy cost shock",
            "EU industrial production cuts likely. Energy-intensive sectors (chemicals, aluminium) most at risk.",
            "⚡", ["AGSI+ (GIE)", "yfinance"])

    # R11 — Low EU gas + winter approaching = price spike
    if _known(eu_gas_rg, "unknown") and eu_gas_rg == "stress":
        add("medium", "medium-term (1-6 months)",
            "Energy / EU Gas",
            "EU gas storage below seasonal norm (AGSI+)",
            "Storage refill pace matters before winter injection season",
            "Below-normal EU gas storage into autumn elevates Q4 gas price spike probability",
            "European energy costs stay elevated. Fertilizer (ammonia, urea) production margins compressed.",
            "🔥", ["AGSI+ (GIE)"])

    # R12 — EU gas crisis = fertilizer production cuts
    if _known(eu_gas_rg, "unknown") and eu_gas_rg == "crisis":
        add("high", "near-term (1-4 weeks)",
            "Energy → Agriculture (Sector Dependency)",
            "EU gas storage in crisis",
            "Natural gas is primary feedstock for nitrogen fertilizers",
            "EU fertilizer (ammonia, urea) plant curtailments imminent → global nitrogen supply crunch",
            "Global food price inflation risk elevated. Monitor FAO FPI. Africa net food importers most exposed.",
            "🌾", ["AGSI+ (GIE)", "FAO"])

    # R13 — Tight US oil inventories + geopolitical oil risk
    if _known(oil_inv_rg, "unknown") and oil_inv_rg == "stress" and \
       _known(oil_rg, "unknown") and oil_rg == "surging":
        add("high", "near-term (1-4 weeks)",
            "Energy / Supply Shock",
            "US crude inventories below 5Y average (EIA)",
            "Oil prices already surging",
            "Thin inventory buffer + oil spike = limited SPR-release capacity for price relief",
            "Oil price likely to remain elevated. Airlines, shipping, and petrochemicals face margin pressure.",
            "🛢️", ["EIA", "yfinance"])

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK 4: Agriculture / Food
    # ═══════════════════════════════════════════════════════════════════════════

    # R14 — FAO FPI crisis + strong USD = EM food inflation emergency
    if _known(fao_rg, "unknown") and fao_rg == "crisis" and \
       _known(dollar_rg, "unknown") and dollar_rg in ("strong","very_strong"):
        add("high", "medium-term (1-6 months)",
            "Agriculture → EM (Sector Dependency)",
            "FAO Food Price Index in crisis (FAO)",
            "Strong USD amplifying import costs for food-importing EMs",
            "Global food crisis conditions: EM food import bills surge in local currency terms → social unrest risk",
            "West Africa, Horn of Africa, Egypt, Sri Lanka most exposed. IMF emergency lending risk elevated.",
            "🌾", ["FAO", "yfinance"])

    # R15 — Crop prices + EU gas crisis = fertilizer-food feedback loop
    if _known(agri_rg, "unknown") and agri_rg in ("stress","crisis") and \
       _known(eu_gas_rg, "unknown") and eu_gas_rg in ("stress","crisis"):
        add("high", "medium-term (1-6 months)",
            "Agriculture / Chemicals → Food (Sector Dependency)",
            "Crop prices in stress (wheat/corn — yfinance)",
            "EU gas storage low → fertilizer supply constrained (AGSI+)",
            "Fertilizer shortage → reduced crop yields next season → food price inflation persistence",
            "2-season feedback loop: fertilizer crunch now → lower yields in 6-18 months → food insecurity.",
            "🌾", ["yfinance", "AGSI+ (GIE)", "FAO"])

    # R16 — FAO FPI rising + EM FX weakening = food import crisis
    if _known(fao_rg, "unknown") and fao_rg in ("stress","crisis") and \
       _known(em_rg, "unknown") and em_rg in ("stress","crisis"):
        add("medium", "medium-term (1-6 months)",
            "Agriculture → EM",
            "FAO Food Price Index elevated (FAO)",
            "EM currencies under pressure",
            "Rising food prices + EM FX depreciation = double squeeze on EM food import bills",
            "Food-insecure countries (Ethiopia, Somalia, Sudan, Yemen) face acute humanitarian risk.",
            "🚨", ["FAO", "yfinance"])

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK 5: Commodities / Industrials
    # ═══════════════════════════════════════════════════════════════════════════

    # R17 — Copper falling + VIX elevated = growth scare
    if _known(cu_rg, "unknown") and cu_rg == "falling" and \
       _known(vix_rg, "unknown") and vix_rg in ("elevated","high"):
        add("high", "medium-term (1-6 months)",
            "Growth / Commodities",
            "Copper falling (yfinance)",
            "VIX elevated — risk-off",
            "Copper decline + elevated VIX = growth scare or recession risk pricing by markets",
            "Industrial EM exporters (DRC, Chile, Zambia) face fiscal deterioration. Reduce cyclical exposure.",
            "📉", ["yfinance"])

    # R18 — Copper rising + OECD CLI expanding = sustained cyclical upswing
    if _known(cu_rg, "unknown") and cu_rg == "rising" and \
       _known(oecd_rg, "unknown") and oecd_rg == "expanding":
        add("medium", "medium-term (1-6 months)",
            "Growth / Cyclicals",
            "Copper rising (yfinance)",
            "OECD CLI above trend (OECD)",
            "Copper rally + OECD leading indicators above trend = self-reinforcing cyclical expansion",
            "Favour cyclicals: industrials, materials, EM equities. Africa commodity exporters benefit.",
            "🚀", ["yfinance", "OECD"])

    # R19 — Oil surging + copper rising = broad commodity cycle
    if _known(oil_rg, "unknown") and oil_rg == "surging" and \
       _known(cu_rg, "unknown") and cu_rg == "rising":
        add("medium", "near-term (1-4 weeks)",
            "Commodities / Inflation",
            "Oil surging (yfinance)",
            "Copper rising simultaneously",
            "Broad commodity cycle: oil + copper rising together signals genuine demand-pull or supply shock",
            "Commodity-exporting EM benefit. Net importers face stagflationary pressure. Watch inflation prints.",
            "🛢️", ["yfinance"])

    # R20 — Copper/Gold risk-off + inverted curve = pre-recession signal
    if _known(cu_gold_rg, "unknown") and cu_gold_rg == "risk_off" and \
       _known(curve_rg, "unknown") and curve_rg in ("inverted","flat"):
        add("high", "medium-term (1-6 months)",
            "Growth / Multi-Asset",
            "Copper/Gold ratio falling (risk-off signal)",
            "Yield curve flat/inverted",
            "Risk-off Cu/Gold ratio + flat/inverted curve = two independent recession signals converging",
            "High-conviction defensive positioning. History suggests 6-12 month window before growth impact.",
            "🔴", ["yfinance", "FRED"])

    # R21 — IP contraction + OECD CLI below trend = global slowdown
    if _known(indpro_rg, "unknown") and indpro_rg == "contraction" and \
       _known(oecd_rg, "unknown") and oecd_rg == "contracting":
        add("high", "medium-term (1-6 months)",
            "Industrials / Global Growth",
            "US industrial production contracting (FRED)",
            "OECD Composite Leading Indicators below trend (OECD)",
            "Synchronized global industrial slowdown: US IP + OECD CLI both contractionary",
            "Global trade volumes likely to decline. Shipping demand softens. EM exporters most exposed.",
            "🏭", ["FRED", "OECD"])

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK 6: Cross-Asset / Safe Haven / Structural
    # ═══════════════════════════════════════════════════════════════════════════

    # R22 — Systemic stress signal (gold + USD both up)
    if sys_stress:
        add("high", "near-term (1-4 weeks)",
            "Systemic Stress",
            "Gold and USD both rising simultaneously (yfinance)",
            "All safe havens bid",
            "Systemic stress signal: flight to ALL safe havens (gold, USD, JPY, CHF) simultaneously",
            "Maximum defensive posture. Avoid risk assets. Monitor for central bank emergency action.",
            "🚨", ["yfinance"])

    # R23 — Equity-bond correlation stress + high VIX = crisis regime
    if _known(corr_rg, "unknown") and corr_rg == "stress" and \
       _known(vix_rg, "unknown") and vix_rg in ("elevated","high"):
        add("high", "near-term (1-4 weeks)",
            "Cross-Asset / Crisis Regime",
            "Equity-bond correlation positive (yfinance)",
            "VIX elevated — diversification failing",
            "Crisis regime: stocks and bonds falling together = traditional diversification broken",
            "Only gold and cash provide diversification in this regime historically. Reduce overall exposure.",
            "⚠️", ["yfinance"])

    # R24 — Bond volatility elevated + inverted curve = rate path uncertainty
    if _known(move_rg, "unknown") and move_rg == "elevated" and \
       _known(curve_rg, "unknown") and curve_rg == "inverted":
        add("medium", "near-term (1-4 weeks)",
            "Rates / Volatility",
            "Bond volatility elevated (computed from FRED yields)",
            "Curve inverted — market disagrees on Fed path",
            "High bond vol + inverted curve = deep uncertainty about rate trajectory",
            "Options on rates are expensive; avoid leveraged duration bets. Wait for clarity before adding duration.",
            "📊", ["FRED"])

    # R25 — Broad financial stress score ≥5 = systemic risk
    if fin_score >= 5:
        add("high", "near-term (1-4 weeks)",
            "Cross-Asset / Systemic",
            f"{fin_score}/7 financial stress indicators elevated",
            "Broad-based simultaneous stress across multiple asset classes",
            "Broad-based financial stress historically precedes 15-25% risk-asset drawdowns",
            "Maximum defensive allocation. Historical median drawdown from this configuration is -20% on equities.",
            "🔴", ["FRED", "yfinance"])

    # R26 — Geo stress elevated + agri stress = food-geopolitics nexus
    if geo_score >= 3 and _known(agri_rg, "unknown") and agri_rg in ("stress","crisis"):
        add("medium", "structural (6-18 months)",
            "Geo-Economics / Food Security",
            f"Geo stress score {geo_score}/4 elevated",
            "Agricultural stress simultaneously elevated",
            "Geopolitical tensions + food stress = food-as-geopolitical-weapon risk; export bans likely",
            "Monitor wheat/rice export restriction policies. Africa food security most vulnerable.",
            "🌍", ["yfinance", "FAO"])

    # R27 — OECD CLI contracting + copper falling = EM export revenue decline
    if _known(oecd_rg, "unknown") and oecd_rg == "contracting" and \
       _known(cu_rg, "unknown") and cu_rg == "falling":
        add("medium", "medium-term (1-6 months)",
            "EM / Commodities",
            "OECD CLI below trend (OECD)",
            "Copper falling (yfinance)",
            "OECD slowdown + falling copper = EM commodity export revenue decline ahead",
            "Africa commodity exporters (DRC, Zambia, Chile) face fiscal deterioration. Budget adjustments needed.",
            "📉", ["OECD", "yfinance"])

    # R28 — FX reserves falling + oil surging + strong USD = triple squeeze
    if fx_det and _known(oil_rg, "unknown") and oil_rg == "surging" and \
       _known(dollar_rg, "unknown") and dollar_rg in ("strong","very_strong"):
        add("high", "near-term (1-4 weeks)",
            "EM / Africa — Triple Squeeze",
            "FX reserves deteriorating (World Bank)",
            "Oil surging + strong USD amplifying pressure",
            "Triple squeeze: falling reserves + higher oil import bills + USD debt service = acute BoP stress",
            "Immediate IMF engagement likely for affected countries. Watch Kenya, Ethiopia, Ghana Eurobond spreads.",
            "🚨", ["World Bank", "yfinance"])

    # Sort by confidence: high → medium → low
    conf_order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda x: (conf_order.get(x["confidence"], 9),
                             LEVEL_ORDER.get(x.get("level","info"), 9)))
    return out
