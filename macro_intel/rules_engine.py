"""
rules_engine.py — turn indicator states into actionable economist insights.

Each rule reads from the indicators dict and appends an insight dict:
  level:    "alert" | "warning" | "info"
  category: short domain label
  headline: one punchy sentence
  detail:   2-4 sentences of analysis with context
  watch:    list of 3-5 items the economist should monitor

Insights are sorted alerts → warnings → info.
"""

from __future__ import annotations

import numpy as np

from config import THRESH, LEVEL_ORDER


def _fmt_pct(v: float | None, decimals: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def run(ind: dict) -> list[dict]:
    """
    Execute all rules and return sorted list of insights.
    Pass output of indicators.compute_all().
    """
    insights: list[dict] = []

    # ── 1. Yield curve / rates regime ─────────────────────────────────────────
    slope     = ind.get("curve_slope", np.nan)
    direction = ind.get("curve_direction", "")
    regime    = ind.get("curve_regime", "normal")

    if not _nan(slope):
        if regime == "inverted":
            insights.append({
                "level": "alert",
                "category": "Rates Regime",
                "headline": f"Yield curve inverted ({slope:+.0f} bps, {direction})",
                "detail": (
                    f"An inversion of {slope:.0f} bps has preceded every US recession since 1960, "
                    f"with a lead time of 12–18 months. The curve is currently {direction}, which "
                    + ("may signal early market pricing of Fed rate cuts — watch FOMC communications carefully."
                       if direction == "steepening" else
                       "signals ongoing credit tightening and falling growth expectations.")
                ),
                "watch": [
                    "Fed Funds futures curve", "Credit spreads (IG, HY)",
                    "Bank lending standards (SLOOS)", "Labor market softening",
                ],
            })
        elif regime == "flat":
            insights.append({
                "level": "warning",
                "category": "Rates Regime",
                "headline": f"Yield curve flat ({slope:+.0f} bps, {direction})",
                "detail": (
                    "A flat curve compresses bank net interest margins and signals slowing growth pricing. "
                    + ("Steepening from flat territory often marks the onset of a rate-cut cycle." if direction == "steepening" else
                       "Continued flattening increases recession risk and drags on financial sector profitability.")
                ),
                "watch": [
                    "2Y-10Y spread trend (daily)", "Fed meeting minutes",
                    "Bank earnings and margin guidance", "Mortgage origination volumes",
                ],
            })
        elif regime == "steep":
            insights.append({
                "level": "info",
                "category": "Rates Regime",
                "headline": f"Yield curve steep ({slope:+.0f} bps) — reflation or fiscal premium?",
                "detail": (
                    f"A steep curve ({slope:.0f} bps) can mean two things: markets pricing robust growth ahead, "
                    "or rising term premium from fiscal concerns (deficit, issuance). "
                    "Distinguish by checking whether breakeven inflation is rising (inflation premium) "
                    "or TIPS real yields are rising (growth/risk premium)."
                ),
                "watch": [
                    "Breakeven inflation (5Y5Y)", "TIPS real yields",
                    "Treasury auction bid-to-cover", "Fiscal deficit projections",
                ],
            })

    # ── 2. Volatility / risk regime ───────────────────────────────────────────
    vix        = ind.get("vix", np.nan)
    vix_regime = ind.get("vix_regime", "normal")

    if not _nan(vix):
        if vix_regime == "high":
            insights.append({
                "level": "alert",
                "category": "Market Stress / VIX",
                "headline": f"VIX in high-stress territory ({vix:.1f})",
                "detail": (
                    f"VIX above {THRESH['vix_high']} reflects genuine market fear, not just hedging. "
                    "In this regime: USD safe-haven bid strengthens, EM sees capital outflows, "
                    "commodity demand contracts, and credit spreads widen. "
                    "Risk-reward is skewed defensive. Watch for liquidity dislocations."
                ),
                "watch": [
                    "EM capital flows (IIF weekly)", "HY/IG credit spread ratio",
                    "USD/JPY and USD/CHF (safe-haven FX)", "Commodity spot vs futures spread",
                ],
            })
        elif vix_regime == "elevated":
            insights.append({
                "level": "warning",
                "category": "Market Stress / VIX",
                "headline": f"Volatility elevated (VIX {vix:.1f})",
                "detail": (
                    "Above-average VIX signals hedging demand and tail-risk pricing. "
                    "Markets are not in crisis but uncertainty is real. "
                    "Monitor cross-asset correlations — if equities, bonds, and EM FX all sell off together, "
                    "a regime shift to full risk-off may be underway."
                ),
                "watch": [
                    "Equity/bond correlation", "VIX term structure (contango vs backwardation)",
                    "Put/call ratio", "Options skew",
                ],
            })

    # ── 3. Dollar regime + EM implications ───────────────────────────────────
    dxy           = ind.get("dxy", np.nan)
    dollar_regime = ind.get("dollar_regime", "neutral")
    em_regime     = ind.get("em_regime", "normal")
    em_widening   = ind.get("em_spreads_widening", False)

    if dollar_regime in ("strong", "very_strong") and not _nan(dxy):
        em_compound = em_regime in ("stress", "crisis")
        insights.append({
            "level": "alert" if (dollar_regime == "very_strong" and em_compound) else "warning",
            "category": "Dollar / EM Capital",
            "headline": f"{'Very strong' if dollar_regime == 'very_strong' else 'Strong'} USD (DXY {dxy:.1f})",
            "detail": (
                f"DXY at {dxy:.1f} tightens global financial conditions mechanically — "
                "every EM sovereign and corporate with USD-denominated debt faces higher real debt-service costs. "
                + ("Combined with widening EM spreads, refinancing risk for frontier markets is acute. " if em_compound else "") +
                "Commodity prices (priced in USD) face a structural headwind. "
                "Net oil importers (India, SSA) face double pressure: stronger USD + higher energy import bills."
            ),
            "watch": [
                "EM FX reserves (months of import cover)", "Eurobond maturity wall (2025-2027)",
                "IMF/World Bank program requests", "Commodity prices in local currency terms",
            ],
        })
    elif dollar_regime == "weakening":
        insights.append({
            "level": "info",
            "category": "Dollar / EM Capital",
            "headline": f"USD weakening (DXY {dxy:.1f}, below 50d MA)",
            "detail": (
                "Dollar weakness eases global financial conditions and is a tailwind for EM assets. "
                "Expect rotation into EM equities and local-currency bonds. "
                "Commodity prices (gold, oil, copper) also benefit from USD softness."
            ),
            "watch": [
                "EM equity ETF inflows (EEM)", "EM local bond yields",
                "Gold and commodity breakouts", "Fed policy trajectory",
            ],
        })

    # ── 4. Geo-risk + shipping ────────────────────────────────────────────────
    gpr         = ind.get("gpr", np.nan)
    gpr_regime  = ind.get("gpr_regime", "normal")
    gpr_chg     = ind.get("gpr_mom_chg", np.nan)
    ship_regime = ind.get("shipping_regime", "normal")
    oil_regime  = ind.get("oil_regime", "stable")

    if gpr_regime in ("elevated", "high"):
        compound_geo = (ship_regime in ("stress", "crisis")) or (oil_regime == "surging")
        level = "alert" if (gpr_regime == "high" or compound_geo) else "warning"
        extras = []
        if ship_regime in ("stress", "crisis"):
            extras.append("shipping disruption amplifying supply-chain costs")
        if oil_regime == "surging":
            extras.append("oil price surge adding stagflationary pressure")
        detail = (
            f"GPR at {gpr:.0f} (baseline = 100) reflects elevated conflict and policy uncertainty. "
            + (f"Compounding factors: {', '.join(extras)}. " if extras else "") +
            "Elevated GPR is empirically linked to: lower investment, higher energy prices, "
            "accelerated defense spending, and industrial supply-chain relocation."
        )
        insights.append({
            "level": level,
            "category": "Geo-Risk / Supply Chains",
            "headline": (
                f"Geo-risk {'high' if gpr_regime == 'high' else 'elevated'} (GPR {gpr:.0f}"
                + (f", {_fmt_pct(gpr_chg)} MoM" if not _nan(gpr_chg) else "")
                + (")"
                   + (" + shipping stress" if ship_regime in ("stress", "crisis") else "")
                   + (" + oil surging" if oil_regime == "surging" else ""))
            ),
            "detail": detail,
            "watch": [
                "Red Sea / Hormuz shipping rates", "LNG contract spot premiums",
                "Defense procurement (US, EU, Gulf)", "FDI redirection flows (Vietnam, India, Morocco)",
            ],
        })

    # ── 5. Commodity regime ───────────────────────────────────────────────────
    copper_regime = ind.get("copper_regime", "stable")
    cu_chg        = ind.get("copper_1m_chg", np.nan)
    oil_chg       = ind.get("oil_1m_chg", np.nan)
    gold_regime   = ind.get("gold_regime", "stable")
    gold_chg      = ind.get("gold_1m_chg", np.nan)

    if copper_regime == "rising" and not _nan(cu_chg):
        oil_context = (
            " Oil stability suggests demand-pull not supply inflation — favorable macro setup."
            if oil_regime == "stable" else
            " Rising oil alongside copper points to broad commodity reflation — watch for inflation persistence."
        )
        insights.append({
            "level": "info",
            "category": "Growth Signal (Copper)",
            "headline": f"Copper +{cu_chg*100:.1f}% (1M) — positive industrial growth signal",
            "detail": (
                "Copper is the best single commodity leading indicator of global industrial activity, "
                "with a ~3-6 month lead on manufacturing PMIs. Rising copper suggests: "
                "improving factory output, infrastructure demand, and EV supply-chain ramp-up."
                + oil_context
            ),
            "watch": [
                "China Caixin Manufacturing PMI", "EV production schedules (BYD, Tesla)",
                "LME copper inventories", "BHP / Freeport earnings guidance",
            ],
        })
    elif copper_regime == "falling" and not _nan(cu_chg):
        insights.append({
            "level": "warning",
            "category": "Growth Signal (Copper)",
            "headline": f"Copper {cu_chg*100:.1f}% (1M) — industrial slowdown signal",
            "detail": (
                "Falling copper anticipates deteriorating factory output by 3-6 months. "
                "Key exposures: DRC (cobalt/copper), Chile (copper), Zambia (copper). "
                "Watch for PMI revisions and China credit impulse data."
            ),
            "watch": [
                "China NBS / Caixin PMI", "EM commodity-exporter FX",
                "Mining capex plans", "Container shipping volumes",
            ],
        })

    if oil_regime == "surging" and not _nan(oil_chg):
        insights.append({
            "level": "warning",
            "category": "Energy / Inflation",
            "headline": f"Oil surging (+{oil_chg*100:.1f}% in 1M)",
            "detail": (
                "Sharp oil price increases raise core inflation persistence and complicate central bank policy. "
                "Net oil importers face the worst trade-off: South Asia, Sub-Saharan Africa, Turkey. "
                "If sustained >3 months, expect current account deterioration in import-dependent EMs "
                "and higher-for-longer rates policy in DMs."
            ),
            "watch": [
                "CPI print surprises (core vs headline)", "Central bank reaction functions",
                "SSA / Turkey current account", "Oil futures curve (backwardation signal)",
            ],
        })

    if gold_regime == "surging" and not _nan(gold_chg):
        insights.append({
            "level": "warning",
            "category": "Gold / Safe Haven",
            "headline": f"Gold surging (+{gold_chg*100:.1f}% in 1M) — stress or de-dollarization?",
            "detail": (
                "Gold surging can reflect two distinct regimes: (1) risk-off / macro fear "
                "(typically accompanied by high VIX and wide credit spreads), or "
                "(2) structural de-dollarization and central bank accumulation (especially BRICS+ and Gulf SWFs). "
                "Distinguish by checking whether real rates are falling (inflation hedge) "
                "or rising (structural demand)."
            ),
            "watch": [
                "Central bank gold purchases (WGC quarterly)", "TIPS real yields",
                "USD correlation (breaking down = structural demand)",
                "EM central bank reserve composition",
            ],
        })

    # ── 6. EM / Africa stress ─────────────────────────────────────────────────
    embi          = ind.get("embi", np.nan)
    africa_spread = ind.get("africa_spreads", np.nan)
    fx_det        = ind.get("fx_res_deteriorating", False)
    worst_ctry    = ind.get("fx_res_worst_country", "")

    if em_regime in ("stress", "crisis") or (em_widening and dollar_regime in ("strong", "very_strong")):
        level = "alert" if em_regime == "crisis" else "warning"
        africa_note = (
            f" Africa composite spread at {africa_spread:.0f} bps — well above investment-grade threshold."
            if not _nan(africa_spread) else ""
        )
        fx_note = (
            f" FX reserve drawdowns detected ({worst_ctry} most exposed) — watch import cover ratios."
            if fx_det else ""
        )
        insights.append({
            "level": level,
            "category": "EM / Africa Stress",
            "headline": (
                f"EM sovereign stress: EMBI {embi:.0f} bps"
                + (" (widening)" if em_widening else "")
                + (f" · Africa {africa_spread:.0f} bps" if not _nan(africa_spread) else "")
            ),
            "detail": (
                f"EMBI at {embi:.0f} bps signals elevated refinancing risk across EM. "
                + ("Combined with strong USD, dollar-denominated debt service costs are rising. " if dollar_regime in ("strong","very_strong") else "") +
                africa_note + fx_note +
                " Eurobond maturity walls (2025-2027) for Ghana, Kenya, Ethiopia, Egypt remain critical pressure points."
            ),
            "watch": [
                "IMF Article IV / program status (Ghana, Ethiopia, Egypt)",
                "Eurobond maturity schedule 2025-2027",
                "FX reserve adequacy (< 3 months import cover = critical)",
                "African Development Bank / World Bank budget support",
            ],
        })

    # ── 7. Financial conditions ───────────────────────────────────────────────
    fin_cond = ind.get("fin_cond_regime", "neutral")
    hyg_chg  = ind.get("hyg_1m_chg", np.nan)

    if fin_cond == "tightening":
        insights.append({
            "level": "warning",
            "category": "Financial Conditions",
            "headline": f"Financial conditions tightening (HY credit {_fmt_pct(hyg_chg)} in 1M)",
            "detail": (
                "High-yield bond spread widening signals credit stress spreading to riskier borrowers. "
                "Tighter financial conditions reduce business investment, slow hiring, "
                "and increase default risk in leveraged loans and private credit portfolios. "
                "Monitor for contagion to IG credit and bank lending conditions."
            ),
            "watch": [
                "HY default rates (Moody's/S&P monthly)", "Leveraged loan spreads",
                "Private credit NAV marks", "Bank lending standards (Fed SLOOS)",
            ],
        })

    # ── 8. Cross-asset composite regime ──────────────────────────────────────
    fin_score = ind.get("financial_stress_score", 0)
    geo_score = ind.get("geo_stress_score", 0)
    total     = ind.get("total_stress_score", 0)

    if fin_score >= 4:
        insights.append({
            "level": "alert",
            "category": "Cross-Asset Regime",
            "headline": f"Broad-based financial stress ({fin_score}/5 indicators elevated)",
            "detail": (
                "Multiple simultaneous financial stress signals historically precede risk-asset drawdowns of 15-25%. "
                "Current configuration: "
                + ", ".join(filter(None, [
                    "inverted curve" if ind.get("curve_regime") == "inverted" else None,
                    f"VIX {ind.get('vix', 0):.0f}" if ind.get("vix_regime") in ("elevated", "high") else None,
                    "USD strength" if ind.get("dollar_regime") in ("strong", "very_strong") else None,
                    "EM stress" if ind.get("em_regime") in ("stress", "crisis") else None,
                    "credit tightening" if ind.get("fin_cond_regime") == "tightening" else None,
                ])) + ". Defensive positioning and liquidity management are priorities."
            ),
            "watch": [
                "Cross-asset correlation breakdown", "Repo / money market stress",
                "Central bank emergency tool activation", "Portfolio margin calls",
            ],
        })
    elif fin_score >= 2 and geo_score >= 2:
        insights.append({
            "level": "warning",
            "category": "Cross-Asset Regime",
            "headline": "Financial + geopolitical stress converging — stagflation risk elevated",
            "detail": (
                "Simultaneous financial market stress and geopolitical tension increases non-linear shock probability. "
                "The most likely macro path from this configuration is stagflation: "
                "elevated inflation (from energy/supply disruption) + slowing growth (from tighter financial conditions). "
                "The hardest policy regime for central banks — and the worst for EM."
            ),
            "watch": [
                "Stagflation scenario probability (inflation breakevens + growth forecasts)",
                "Energy supply diversification timelines", "EM fiscal buffer adequacy",
                "Supply-chain relocation acceleration (nearshoring winners)",
            ],
        })

    # ── If nothing fires, confirm calm ────────────────────────────────────────
    has_alert_or_warn = any(i["level"] in ("alert", "warning") for i in insights)
    if not has_alert_or_warn:
        insights.append({
            "level": "info",
            "category": "Regime Assessment",
            "headline": "No major stress signals — broadly benign macro regime",
            "detail": (
                "Current cross-asset configuration is consistent with a late-cycle expansion: "
                "positive growth signal from copper, manageable volatility, and stable EM spreads. "
                "This is precisely when complacency sets in — maintain watchful positioning."
            ),
            "watch": [
                "Yield curve slope trend (steepening or flattening?)",
                "DXY direction (break above 104 = EM warning)",
                "Copper vs oil divergence (growth vs inflation)",
                "GPR acceleration (geopolitical risk can shift fast)",
            ],
        })

    # Sort: alerts first, then warnings, then info
    insights.sort(key=lambda x: LEVEL_ORDER.get(x["level"], 9))
    return insights
