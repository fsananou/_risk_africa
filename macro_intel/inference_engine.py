"""
inference_engine.py — Conditional forward-looking logic.

Form:  IF trigger_condition AND context_condition THEN expected_outcome.

Each inference returns:
  confidence: high / medium / low
  horizon:    near-term (1-4 weeks) / medium-term (1-6 months) / structural (6-18 months)
  category:   domain
  trigger:    what triggered this rule
  context:    what amplifying context is present
  statement:  the forward-looking claim
  implication: what to watch / position for

32 rules covering: rates, dollar/EM, geo-risk, commodities, capital flows, cross-asset.
"""

from __future__ import annotations

import numpy as np

from config import CONF_COLOR, HORIZON_ICON


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _inf(confidence, horizon, category, trigger, context, statement, implication) -> dict:
    return {
        "confidence":  confidence,
        "horizon":     horizon,
        "category":    category,
        "trigger":     trigger,
        "context":     context,
        "statement":   statement,
        "implication": implication,
        "icon":        HORIZON_ICON.get(horizon, ""),
        "conf_color":  CONF_COLOR.get(confidence, "#333"),
    }


def run(ind: dict, rules: list[dict]) -> list[dict]:
    """
    Run all 32 conditional inference rules.
    Input: indicators dict + rules_engine output (for context).
    Output: list of inference dicts, sorted by confidence (high → medium → low).
    """
    out: list[dict] = []

    # Convenience aliases
    curve_rg  = ind.get("curve_regime", "normal")
    direction = ind.get("curve_direction", "flattening")
    slope     = ind.get("curve_slope", np.nan)
    vix_rg    = ind.get("vix_regime", "normal")
    vix       = ind.get("vix", np.nan)
    dol_rg    = ind.get("dollar_regime", "neutral")
    em_rg     = ind.get("em_regime", "normal")
    em_wide   = ind.get("em_spreads_widening", False)
    fx_det    = ind.get("fx_res_deteriorating", False)
    fin_cond  = ind.get("fin_cond_regime", "neutral")
    hy_rg     = ind.get("hy_regime", "normal")
    gpr_rg    = ind.get("gpr_regime", "normal")
    ship_rg   = ind.get("shipping_regime", "normal")
    oil_rg    = ind.get("oil_regime", "stable")
    oil_chg   = ind.get("oil_1m_chg", np.nan)
    cu_rg     = ind.get("copper_regime", "stable")
    cu_chg    = ind.get("copper_1m_chg", np.nan)
    gold_rg   = ind.get("gold_regime", "stable")
    gold_chg  = ind.get("gold_1m_chg", np.nan)
    inf_rg    = ind.get("inflation_regime", "normal")
    be5       = ind.get("breakeven5y", np.nan)
    be55      = ind.get("breakeven5y5y", np.nan)
    tp        = ind.get("term_premium", np.nan)
    corr_rg   = ind.get("correlation_regime", "normal")
    sys_str   = ind.get("systemic_stress_signal", False)
    san_elev  = ind.get("sanctions_elevated", False)
    usd_vuln  = ind.get("usd_debt_vulnerability", "low")
    embi      = ind.get("embi", np.nan)
    africa_sp = ind.get("africa_spreads", np.nan)
    min_str   = ind.get("minerals_stressed", False)
    fin_score = ind.get("financial_stress_score", 0)
    geo_score = ind.get("geo_stress_score", 0)
    cu_au     = ind.get("copper_gold_regime", "neutral")

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 1 — RATES & FINANCIAL CONDITIONS
    # ══════════════════════════════════════════════════════════════════════════

    # Rule 1: Inverted curve + tightening conditions → recession
    if curve_rg == "inverted" and fin_cond == "tightening":
        out.append(_inf("high", "medium-term (1-6 months)",
            "Rates / Recession Risk",
            "Yield curve inverted",
            "Financial conditions simultaneously tightening",
            "Recession probability is significantly elevated. Historical median lead time from "
            "this dual condition to recession onset is 9–12 months.",
            "Reduce cyclical exposure; favor short duration, cash, defensive sectors (utilities, staples). "
            "Watch ISM Manufacturing, initial jobless claims, and bank lending standards as confirming signals."))

    # Rule 2: Inverted curve → steepening (early recovery signal)
    if curve_rg in ("inverted", "flat") and direction == "steepening" and vix_rg == "normal":
        out.append(_inf("medium", "medium-term (1-6 months)",
            "Rates / Cycle Turn",
            "Yield curve steepening from inverted/flat",
            "VIX is contained — no acute stress",
            "Steepening from inversion historically signals early pricing of Fed rate cuts and "
            "coming end of the tightening cycle. Growth assets typically outperform in this phase "
            "6–12 months forward.",
            "Begin positioning for rate-sensitive assets (small-cap, EM, homebuilders). "
            "Watch for first Fed rate cut and ISM new orders inflection."))

    # Rule 3: Rising 2Y + curve inverting → rate cycle still active
    us2y = ind.get("us2y", np.nan)
    if not _nan(us2y) and curve_rg in ("inverted", "flat") and fin_cond in ("tightening", "neutral"):
        out.append(_inf("medium", "near-term (1-4 weeks)",
            "Rates / Central Bank",
            "2Y yield elevated and curve flat/inverted",
            "Rate cycle still active",
            "Short-end rates are the direct transmission of central bank policy. "
            "Elevated 2Y signals the market expects rates to stay higher for longer — "
            "this mechanically pressures valuations, EM carry trades, and real estate.",
            "Short duration positioning. Watch next FOMC meeting, CPI core, and PCE deflator prints."))

    # Rule 4: Breakevens rising + curve steep → nominal reflation
    if (not _nan(be5) and be5 > 2.5) and curve_rg == "steep":
        out.append(_inf("medium", "medium-term (1-6 months)",
            "Inflation / Reflation",
            "Breakeven inflation rising (5Y: {:.1f}%)".format(be5),
            "Yield curve steep — growth pricing also elevated",
            "Reflation trade is active: markets are simultaneously pricing growth AND inflation. "
            "This regime historically favors: commodities, TIPS, commodity-exporting EMs, energy equities.",
            "Overweight commodities, TIPS, and commodity-linked EM currencies. "
            "Underweight long-duration nominal bonds. Watch for central bank reaction."))

    # Rule 5: HY > crisis + VIX > 30 → credit event imminent
    if hy_rg == "crisis" and vix_rg in ("elevated", "high"):
        out.append(_inf("high", "near-term (1-4 weeks)",
            "Credit / Contagion Risk",
            "HY spreads in crisis territory",
            "VIX simultaneously elevated",
            "The combination of crisis-level credit spreads and elevated equity volatility "
            "historically precedes a financial contagion event (bank stress, hedge fund deleveraging, "
            "or sovereign EM crisis) within 1–4 weeks.",
            "Reduce risk exposure immediately. Favor: Treasuries, gold, USD, yen, Swiss franc. "
            "Monitor: repo market rates, money market fund flows, bank CDS."))

    # Rule 6: Tightening conditions + EM stress → synchronized global slowdown
    if fin_cond == "tightening" and em_rg in ("stress", "crisis"):
        out.append(_inf("high", "medium-term (1-6 months)",
            "Global Macro / Synchronised Stress",
            "Financial conditions tightening",
            "EM sovereign stress simultaneously elevated",
            "Simultaneous DM credit tightening and EM sovereign stress historically produces "
            "a synchronized global growth deceleration within 6–12 months. "
            "This is the 'doom loop' for frontier markets: tighter DM conditions → USD strength "
            "→ EM capital outflow → spread widening → more outflows.",
            "Reduce broad EM exposure. Focus on EM countries with current account surpluses, "
            "adequate reserves, and IMF program access. Watch BIS cross-border lending data."))

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 2 — DOLLAR / EM
    # ══════════════════════════════════════════════════════════════════════════

    # Rule 7: Strong USD + falling EM reserves → EM sovereign stress
    if dol_rg in ("strong", "very_strong") and fx_det:
        out.append(_inf("high", "medium-term (1-6 months)",
            "EM / Sovereign Stress",
            "Strong USD (DXY above threshold)",
            "EM FX reserves simultaneously deteriorating",
            "Strong dollar combined with reserve drawdowns signals EM sovereigns are defending "
            "FX pegs or meeting debt obligations from reserves. "
            "Countries with <3 months import cover face acute crisis risk within 3–6 months.",
            "Monitor: IMF Article IV consultations, Eurobond spread widening, rating agency reviews. "
            "Most exposed: Ethiopia, Ghana, Egypt, Pakistan, Sri Lanka analogues. "
            "Watch for IMF emergency financing requests."))

    # Rule 8: Strong USD + EM spreads widening + VIX elevated → EM triple squeeze
    if dol_rg in ("strong", "very_strong") and em_wide and vix_rg in ("elevated", "high"):
        out.append(_inf("high", "medium-term (1-6 months)",
            "EM / Triple Squeeze",
            "Strong USD + widening EM spreads",
            "VIX elevated (global risk-off)",
            "Three simultaneous EM headwinds: stronger dollar (higher debt service), "
            "spread widening (higher borrowing cost), risk-off (capital outflow). "
            "This is the classic EM crisis configuration — 1997 Asia, 2013 Taper Tantrum, 2018.",
            "Underweight EM equities and local bonds. "
            "If holding EM debt: concentrate in countries with IMF programs, large reserves, "
            "and commodity export buffers. Watch IIF weekly EM flow data."))

    # Rule 9: USD weakening + EM reserves stable → EM recovery
    if dol_rg == "weakening" and not fx_det and em_rg == "normal":
        out.append(_inf("medium", "medium-term (1-6 months)",
            "EM / Recovery Trade",
            "USD weakening below 50-day MA",
            "EM reserves stable and EM spreads not stressed",
            "Dollar weakness with stable EM fundamentals creates a favorable window for EM recovery. "
            "Historically, EM equities and local-currency bonds significantly outperform during "
            "sustained dollar bear markets.",
            "Build EM exposure: MSCI EM equities, local-currency bonds (EMLC), commodity-linked EMs. "
            "Focus on India, Vietnam, Indonesia as structural growth plays."))

    # Rule 10: Africa spreads > 500bps + FX reserves deteriorating → debt restructuring risk
    if (not _nan(africa_sp) and africa_sp > 500) and fx_det:
        out.append(_inf("high", "structural (6-18 months)",
            "Africa / Debt Distress",
            "Africa composite sovereign spread above 500 bps",
            "FX reserves in at least one African country deteriorating",
            "African frontier markets are in debt distress territory. "
            "Countries with large Eurobond maturities (2025-2028), limited reserve buffers, "
            "and high food/energy import dependency face sovereign restructuring risk within 12-18 months.",
            "Monitor: Ghana, Ethiopia, Kenya, Egypt Eurobond maturities. "
            "Watch for IMF staff visits, G20 Common Framework negotiations, "
            "and China bilateral debt restructuring talks. "
            "Debt relief frameworks typically take 12-24 months to execute."))

    # Rule 11: USD debt vulnerability high + EM regime stressed
    if usd_vuln == "high" and em_rg in ("stress", "crisis"):
        out.append(_inf("high", "medium-term (1-6 months)",
            "EM / USD Debt Trap",
            "USD debt vulnerability assessed as high",
            "EM sovereign spreads already stressed",
            "Markets have moved ahead of the fundamental deterioration. "
            "USD-denominated debt at high spreads means rollover risk is now existential "
            "for some sovereigns — forced devaluation or default within 6-12 months for weakest credits.",
            "Assign haircut scenarios to frontier EM debt holdings. "
            "Watch for cross-default clauses in bond documentation. "
            "Follow Paris Club / G20 Common Framework status closely."))

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 3 — GEO-RISK
    # ══════════════════════════════════════════════════════════════════════════

    # Rule 12: GPR high + shipping stress → supply-chain inflation
    if gpr_rg in ("elevated", "high") and ship_rg in ("stress", "crisis"):
        out.append(_inf("high", "near-term (1-4 weeks)",
            "Geo-Risk / Supply Chain",
            "GPR elevated/high",
            "Shipping stress simultaneously elevated",
            "Geopolitical risk is transmitting into physical supply chains. "
            "Supply-chain inflation typically appears in PPI data within 4-8 weeks. "
            "Energy and food are first-order impacts; manufactured goods follow with a 6-8 week lag.",
            "Watch weekly PPI prints, ISM prices paid, and spot LNG/container rates. "
            "Overweight energy supply-chain beneficiaries. Monitor OPEC+ spare capacity."))

    # Rule 13: GPR high + oil surging + shipping disrupted → stagflation
    if gpr_rg in ("elevated", "high") and oil_rg == "surging" and ship_rg in ("stress", "crisis"):
        out.append(_inf("high", "medium-term (1-6 months)",
            "Geo-Risk / Stagflation",
            "GPR high + oil surging",
            "Shipping also disrupted",
            "Classic stagflation configuration: supply-side shock is simultaneously raising costs "
            "(oil, shipping) AND disrupting output (supply chains). "
            "Central banks face their hardest trade-off: fight inflation or support growth. "
            "EMs with high food/energy import dependency are most at risk.",
            "Historical playbook: overweight oil, gold, short-duration TIPS, and commodity-exporting EMs. "
            "Underweight rate-sensitive growth assets. Monitor Fed/ECB communication on stagflation."))

    # Rule 14: Defense budgets rising + GPR high → structural capex cycle
    if gpr_rg in ("elevated", "high") and ind.get("geo_stress_score", 0) >= 2:
        out.append(_inf("medium", "structural (6-18 months)",
            "Structural / Defense Capex",
            "Elevated geopolitical risk",
            "Multiple geo-risk indicators simultaneously elevated",
            "Elevated sustained GPR historically triggers multi-year defense spending cycles. "
            "NATO commitment to 2% GDP, Gulf rearmament, and AUKUS represent $2T+ in committed defense capex. "
            "Industrial policy and defense manufacturing will outperform over 5-10 years.",
            "Build exposure to: defense primes (LMT, RTX, BA, Airbus), "
            "cybersecurity, space, autonomous systems, and critical minerals supply chains. "
            "These are 5-7 year structural positions, not tactical trades."))

    # Rule 15: Sanctions elevated + trade disruption → FDI relocation
    if san_elev and gpr_rg in ("elevated", "high"):
        out.append(_inf("medium", "structural (6-18 months)",
            "Structural / Industrial Relocation",
            "Sanctions intensity elevated",
            "Geopolitical risk also elevated",
            "Sanctions are forcing supply-chain reconfiguration. "
            "Companies are executing 'China+1' or 'Russia+1' strategies, redirecting investment "
            "to geopolitically neutral manufacturing hubs. "
            "Vietnam, India, Mexico, Morocco, Indonesia are the primary beneficiaries.",
            "FDI into Vietnam, India, Mexico manufacturing. "
            "Watch nearshoring plays (Mexico infrastructure, logistics REITs near border). "
            "Monitor China trade data for export substitution evidence."))

    # Rule 16: GPR high + minerals stressed → supply-chain weaponization
    if gpr_rg in ("elevated", "high") and min_str:
        out.append(_inf("high", "structural (6-18 months)",
            "Structural / Critical Minerals",
            "GPR elevated/high",
            "Critical minerals prices surging simultaneously",
            "Geopolitical stress is intersecting with critical minerals supply chains. "
            "Critical minerals (copper, lithium, cobalt, rare earths) are becoming instruments "
            "of geopolitical leverage — China controls 60-80% of processing for most. "
            "This triggers strategic stockpiling and accelerated domestic supply development.",
            "Long copper, lithium royalties, and rare earth miners outside China. "
            "Watch US IRA subsidies, EU Critical Raw Materials Act implementation, "
            "and DRC/Chile/Indonesia policy changes."))

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 4 — COMMODITIES
    # ══════════════════════════════════════════════════════════════════════════

    # Rule 17: Copper rising + demand signals positive → manufacturing bottleneck
    if cu_rg == "rising" and not _nan(cu_chg) and cu_chg > 0.07:
        out.append(_inf("medium", "near-term (1-4 weeks)",
            "Commodities / Industrial",
            f"Copper rising strongly (+{cu_chg*100:.1f}% in 1M)",
            "Price move exceeds threshold suggesting demand-pull not just speculation",
            "Strong copper price movement signals real physical demand tightness. "
            "LME inventories typically respond with a 4-8 week lag. "
            "Watch for supply bottlenecks in electrical equipment, construction, EVs.",
            "Watch LME copper warrant cancellations (leading indicator of delivery demand). "
            "BHP, Freeport, Antofagasta earnings guidance. "
            "China PMI and real estate fixed asset investment data."))

    # Rule 18: Copper falling + VIX elevated → growth scare
    if cu_rg == "falling" and vix_rg in ("elevated", "high"):
        out.append(_inf("high", "near-term (1-4 weeks)",
            "Commodities / Growth Scare",
            "Copper falling (industrial demand signal)",
            "VIX elevated (financial markets also risk-off)",
            "When copper falls AND volatility is elevated simultaneously, "
            "markets are pricing a genuine growth scare — not just a commodities correction. "
            "This configuration has a high hit rate for predicting ISM manufacturing below 50 within 2 months.",
            "Reduce cyclical exposure. Favor defensive sectors and quality. "
            "Watch ISM manufacturing new orders, China trade data, and PMI flash prints."))

    # Rule 19: Oil spike + Middle East tension → inflation rebound
    if oil_rg == "surging" and gpr_rg in ("elevated", "high"):
        out.append(_inf("high", "medium-term (1-6 months)",
            "Commodities / Inflation Rebound",
            "Oil surging",
            "Geopolitical risk elevated (Middle East/Hormuz risk)",
            "Oil supply disruption risk from Middle East conflict. "
            "A sustained $10/bbl oil increase adds ~0.3-0.5% to headline CPI within 3 months. "
            "This would delay Fed/ECB rate cuts and extend financial tightening.",
            "Long energy sector, short rate-sensitive sectors. "
            "Monitor Hormuz shipping data, OPEC+ emergency meeting signals, "
            "and US SPR release announcements."))

    # Rule 20: Oil falling + strong USD → EM commodity exporters squeezed
    if oil_rg == "crashing" and dol_rg in ("strong", "very_strong"):
        out.append(_inf("high", "medium-term (1-6 months)",
            "EM / Commodity Exporter Stress",
            "Oil crashing",
            "Strong USD compounding the revenue shock",
            "Oil falling in USD terms creates a double squeeze for energy-exporting EMs: "
            "lower export revenue AND higher USD borrowing costs. "
            "Most exposed: Nigeria, Angola, Ecuador, Kazakhstan, Algeria.",
            "Short EM oil-exporters' sovereign bonds and currencies. "
            "Watch Nigeria NGN, Angola AOA, Kazakhstan KZT. "
            "Monitor fiscal breakeven oil prices vs current price for each country."))

    # Rule 21: Critical minerals + EV demand → structural supply crunch
    if min_str:
        out.append(_inf("medium", "structural (6-18 months)",
            "Structural / Energy Transition",
            "Critical minerals prices surging",
            "EV and energy transition demand accelerating structurally",
            "Critical minerals are in a structural supply deficit relative to energy transition demand. "
            "Lithium supply response lag is 5-7 years (mine permitting to production). "
            "Cobalt faces Congo supply concentration risk. "
            "This is a multi-year structural bull market with periodic volatility.",
            "Long physical copper, lithium royalties (Lithium Americas, SQM, Albemarle), "
            "cobalt processors. Watch IEA Critical Minerals Report, "
            "US DoE loan guarantees, and Chinese export restriction signals."))

    # Rule 22: Gold rising + real yields falling → monetary credibility
    real_yield = (ind.get("us10y", np.nan) or 0) - (ind.get("breakeven10y", be5) or 0)
    if gold_rg in ("stable", "surging") and not _nan(real_yield) and real_yield < 0:
        out.append(_inf("medium", "medium-term (1-6 months)",
            "Gold / Monetary Policy",
            f"Gold firm while real yields negative ({real_yield:.1f}%)",
            "Negative real rates erode the opportunity cost of holding gold",
            "Negative real yields historically correlate strongly with gold outperformance. "
            "If fiscal deficits remain large and real rates stay negative, "
            "gold faces a structurally favorable environment.",
            "Maintain gold exposure as portfolio insurance. "
            "Watch TIPS real yields (breakeven spread), Fed balance sheet policy, "
            "and central bank gold purchase data (World Gold Council quarterly)."))

    # Rule 23: Copper/gold ratio falling → risk-off dominates
    if ind.get("copper_gold_regime") == "risk_off":
        out.append(_inf("medium", "near-term (1-4 weeks)",
            "Cross-Asset / Risk Sentiment",
            "Copper/gold ratio falling (safety over growth)",
            "Risk-off sentiment dominates cyclical pricing",
            "When gold outperforms copper, markets are pricing safety/fear over growth. "
            "This regime is associated with PMI contraction, EM outflows, "
            "and defensive equity outperformance.",
            "Tilt toward quality, low-beta, defensive sectors. "
            "Reduce exposure to industrial EMs and cyclical commodities."))

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 5 — CAPITAL FLOWS & STRUCTURAL
    # ══════════════════════════════════════════════════════════════════════════

    # Rule 24: FDI patterns + sanctions → nearshoring beneficiaries
    if san_elev:
        out.append(_inf("low", "structural (6-18 months)",
            "Structural / Nearshoring",
            "Sanctions intensity elevated",
            "Global supply chain reconfiguration accelerating",
            "Elevated sanctions are accelerating supply-chain diversification. "
            "Manufacturing FDI is shifting toward geopolitically neutral, cost-competitive hubs. "
            "Vietnam ($38bn FDI 2023), India ($83bn FDI 2023), Mexico ($36bn FDI 2023), "
            "Morocco (auto/aerospace) are the primary beneficiaries.",
            "Overweight infrastructure and logistics plays in Vietnam, India, Mexico. "
            "Watch Foxconn, Apple, Samsung manufacturing announcements. "
            "Monitor US-Mexico nearshoring data (maquiladora output)."))

    # Rule 25: Defense budgets rising + minerals stressed → minerals-defense nexus
    if ind.get("geo_stress_score", 0) >= 3 and min_str:
        out.append(_inf("medium", "structural (6-18 months)",
            "Structural / Minerals Security",
            "High geo-risk AND critical minerals stress",
            "Defense spending rising simultaneously",
            "The minerals-defense nexus is tightening. "
            "Defense systems require rare earths, cobalt, and specialty metals. "
            "Governments are classifying these as strategic assets and funding domestic supply chains. "
            "DRC, Chile, Indonesia are negotiating new extraction terms with Western buyers.",
            "Long rare earth miners outside China, cobalt royalties, and specialty metals ETFs. "
            "Watch US IRA, EU CRMA implementation, and G7 mineral security initiatives."))

    # Rule 26: FDI to SSA declining + spreads widening → external financing dry-up
    if em_rg in ("stress", "crisis") and (not _nan(africa_sp) and africa_sp > 500):
        out.append(_inf("medium", "medium-term (1-6 months)",
            "Africa / External Financing",
            "EM spreads in stress/crisis zone",
            "Africa composite spreads above 500 bps",
            "High spreads effectively shut African frontier markets out of Eurobond markets. "
            "With commercial financing unavailable, governments turn to: "
            "(1) IMF emergency programs, (2) Gulf SWF bilateral loans, "
            "(3) Chinese debt renegotiation, (4) domestic financing (crowding out private credit).",
            "Monitor: AfDB financing approvals, Saudi/UAE bilateral commitments, "
            "IMF disbursement schedules, and domestic T-bill auction results in affected countries."))

    # ══════════════════════════════════════════════════════════════════════════
    # BLOCK 6 — CROSS-ASSET / REGIME
    # ══════════════════════════════════════════════════════════════════════════

    # Rule 27: VIX backwardation + HY stress → near-term correction
    if (ind.get("vix_term_structure") == "backwardation" and
            hy_rg in ("stress", "crisis")):
        out.append(_inf("high", "near-term (1-4 weeks)",
            "Cross-Asset / Near-Term Risk",
            "VIX in backwardation (spot > term structure)",
            "HY credit spreads simultaneously stressed",
            "VIX backwardation signals immediate fear, not just uncertainty. "
            "When this coincides with HY stress, markets are pricing an imminent credit event. "
            "Historical examples: Lehman week, March 2020, Oct 2022.",
            "Reduce gross exposure immediately. Long VIX calls as tail hedge. "
            "Monitor intraday credit default swap markets for any single-name blowups."))

    # Rule 28: Gold + DXY both rising → systemic stress
    if sys_str:
        out.append(_inf("high", "near-term (1-4 weeks)",
            "Cross-Asset / Systemic Stress",
            "Gold and USD simultaneously rising",
            "Markets bidding both safe havens — unusual correlation breakdown",
            "When gold and USD both strengthen, investors are fleeing ALL risk assets "
            "toward ANY safe haven — a systemic stress signal. "
            "This pattern preceded: Lehman Brothers, European debt crisis, COVID market crash. "
            "Magnitude matters: the stronger the co-movement, the more acute the stress.",
            "Maximum defensive positioning. Hold cash (USD/CHF), gold, short-dated Treasuries. "
            "Monitor: repo market spreads, tri-party repo volumes, commercial paper spreads."))

    # Rule 29: TIPS breakevens + nominal yields + real yields all rising → stagflation
    if (not _nan(be55) and be55 > 2.7 and
            not _nan(ind.get("us10y")) and
            ind.get("inflation_regime") in ("high", "very_high")):
        out.append(_inf("medium", "medium-term (1-6 months)",
            "Macro / Stagflation",
            "5Y5Y breakevens above 2.7%",
            "Curve steep + inflation expectations well-anchored above 2.5%",
            "Markets are pricing a medium-term stagflation scenario: "
            "sustained above-target inflation with slowing growth. "
            "This is the worst policy environment for central banks — "
            "raising rates fights inflation but deepens the slowdown; "
            "cutting rates fights the slowdown but entrenches inflation.",
            "Overweight: commodities, TIPS, energy, gold, commodity-exporting EMs. "
            "Underweight: long-duration bonds, rate-sensitive growth equities. "
            "Watch: Fed 5Y forward inflation projection, PCE core persistence."))

    # Rule 30: EM equities underperforming SPX + USD strong → sustained EM lag
    if "eem" in (ind.get("growth_signal", "") or ""):
        pass  # placeholder - check via EEM vs SPX return
    emb_chg  = ind.get("emb_1m_chg", np.nan)
    hyg_chg  = ind.get("hyg_1m_chg", np.nan)
    if (not _nan(emb_chg) and emb_chg < -0.03 and
            dol_rg in ("strong", "very_strong")):
        out.append(_inf("medium", "medium-term (1-6 months)",
            "EM / Sustained Underperformance",
            "EM bonds (EMB) falling meaningfully",
            "USD strong, amplifying EM outflows",
            "EM debt is being sold both on fundamentals (spreads widening) and FX (USD strengthening). "
            "This dual pressure typically sustains for 3-6 months before EM stabilizes. "
            "Frontier markets face the sharpest adjustment.",
            "Reduce EM exposure. If maintaining EM: favor commodity-exporters with reserves > 6M imports. "
            "Watch for any Fed pivot signal as the likely catalyst for EM reversal."))

    # Rule 31: Correlation regime stress → 60/40 breakdown
    if corr_rg == "stress" and vix_rg in ("elevated", "high"):
        out.append(_inf("medium", "near-term (1-4 weeks)",
            "Cross-Asset / Portfolio Regime",
            "Equity-bond correlation positive (both falling)",
            "VIX elevated confirming risk-off",
            "Traditional 60/40 portfolio diversification is broken in this regime. "
            "Both equities and bonds are being sold — this is characteristic of "
            "inflation-driven deleveraging (2022) or liquidity crises (2020, 2008). "
            "Alternative diversifiers are required.",
            "Add: gold, commodities, trend-following (CTA-style), FX carry (defensive). "
            "Reduce: duration, credit, growth equities. "
            "This regime typically lasts 2-6 months before correlation normalizes."))

    # Rule 32: Broad composite — regime change
    if fin_score >= 4 and geo_score >= 2:
        out.append(_inf("high", "medium-term (1-6 months)",
            "Macro Regime / Structural Change",
            f"Financial stress score {fin_score}/7 AND geo-risk score {geo_score}/4",
            "Convergence of financial and geopolitical stress",
            "Financial market stress and geopolitical risk are activating simultaneously. "
            "This convergence historically marks the beginning of a macro regime change — "
            "not a temporary correction but a lasting shift in the risk-return landscape. "
            "The most likely new regime: stagflation with geopolitical fragmentation.",
            "Rethink portfolio construction for the next 3-5 years, not the next quarter. "
            "Themes: defense, energy security, critical minerals, nearshoring, "
            "domestic supply chains, and EM 'geopolitical neutrals' (Vietnam, India, Gulf). "
            "Reduce: globalization-dependent business models, long-duration, EM 'geopolitical losers'."))

    # Sort by confidence: high → medium → low
    order = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda x: order.get(x["confidence"], 9))
    return out
