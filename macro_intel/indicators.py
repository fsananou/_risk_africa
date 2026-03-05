"""
indicators.py — Compute all regime indicators from real data only.

Rules:
  - If data is None or insufficient, the indicator is set to np.nan / None / "unknown".
  - No synthetic fallback values.
  - All indicators are computed from real data passed in.

Input:  raw data from data_fetchers.py
Output: flat dict consumed by all downstream engines and the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import THRESH


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _last(series: pd.Series | None) -> float:
    """Safe last value from a Series."""
    if series is None or series.empty:
        return np.nan
    v = series.dropna()
    return float(v.iloc[-1]) if not v.empty else np.nan


def _chg(series: pd.Series | None, periods: int) -> float:
    """Safe N-period simple return."""
    if series is None or len(series) <= periods:
        return np.nan
    s = series.dropna()
    if len(s) <= periods:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-periods] - 1)


def _yoy(series: pd.Series | None, freq: str = "M") -> float:
    """Year-over-year % change (monthly or weekly series)."""
    periods = 12 if freq == "M" else 52
    return _chg(series, periods)


# ══════════════════════════════════════════════════════════════════════════════
# Domain functions
# ══════════════════════════════════════════════════════════════════════════════

def _macro(mkt: dict | None, yields: pd.DataFrame | None,
           fred: dict | None) -> dict:
    """Yield curve, VIX, dollar, financial conditions, inflation."""
    out = {}

    # ── Yield curve (prefer FRED yields df; fallback to yfinance) ────────────
    slope = np.nan
    if yields is not None and "10Y" in yields.columns and "2Y" in yields.columns:
        last10 = _last(yields["10Y"])
        last2  = _last(yields["2Y"])
        if not (_nan(last10) or _nan(last2)):
            slope = (last10 - last2) * 100  # in bps
    elif mkt is not None:
        t10 = _last(mkt.get("us10y"))
        t2  = _last(mkt.get("us2y"))
        if not (_nan(t10) or _nan(t2)):
            slope = (t10 - t2) * 10  # ^TNX is in % * 10

    out["curve_slope"] = slope
    if not _nan(slope):
        if slope < THRESH["curve_inverted"]:
            out["curve_regime"] = "inverted"
        elif slope < THRESH["curve_flat"]:
            out["curve_regime"] = "flat"
        elif slope > THRESH["curve_steep"]:
            out["curve_regime"] = "steep"
        else:
            out["curve_regime"] = "normal"
        # Direction (last 20 trading days)
        if yields is not None and "10Y" in yields.columns and "2Y" in yields.columns:
            spread_ts = (yields["10Y"] - yields["2Y"]).dropna() * 100
            if len(spread_ts) >= 20:
                direction = "steepening" if spread_ts.iloc[-1] > spread_ts.iloc[-20] else "flattening"
                out["curve_direction"] = direction
    else:
        out["curve_regime"]    = "unknown"
        out["curve_direction"] = "unknown"

    # ── VIX ──────────────────────────────────────────────────────────────────
    vix = _last(mkt.get("vix") if mkt is not None else None)
    out["vix"] = vix
    if not _nan(vix):
        if vix > THRESH["vix_high"]:
            out["vix_regime"] = "high"
        elif vix > THRESH["vix_elevated"]:
            out["vix_regime"] = "elevated"
        else:
            out["vix_regime"] = "normal"
        # VIX term structure: compare VIX (spot) vs VIX3M proxy (not available without sub)
        # Use 20-day rolling avg as backwardation proxy
        vix_s = mkt.get("vix") if mkt is not None else None
        if vix_s is not None and len(vix_s) >= 20:
            vix_avg = float(vix_s.dropna().iloc[-20:].mean())
            out["vix_term_structure"] = "backwardation" if vix > vix_avg else "contango"
        else:
            out["vix_term_structure"] = "unknown"
    else:
        out["vix_regime"] = "unknown"
        out["vix_term_structure"] = "unknown"

    # ── Dollar ───────────────────────────────────────────────────────────────
    dxy = _last(mkt.get("dxy") if mkt is not None else None)
    out["dxy"] = dxy
    if not _nan(dxy):
        dxy_1m = _chg(mkt.get("dxy") if mkt is not None else None, 22)
        if dxy > THRESH["dxy_very_strong"]:
            out["dollar_regime"] = "very_strong"
        elif dxy > THRESH["dxy_strong"]:
            out["dollar_regime"] = "strong"
        elif not _nan(dxy_1m) and dxy_1m < -0.02:
            out["dollar_regime"] = "weakening"
        else:
            out["dollar_regime"] = "neutral"
    else:
        out["dollar_regime"] = "unknown"

    # ── FRED macro (spreads, breakevens, NFCI, term premium) ─────────────────
    hy_spread    = _last(fred.get("hy_spread")    if fred else None)
    ig_spread    = _last(fred.get("ig_spread")    if fred else None)
    nfci         = _last(fred.get("nfci")         if fred else None)
    term_prem    = _last(fred.get("term_premium") if fred else None)
    be5          = _last(fred.get("breakeven5y")  if fred else None)
    be55         = _last(fred.get("breakeven5y5y") if fred else None)
    indpro       = fred.get("indpro_us")          if fred else None

    out["hy_spread"]     = hy_spread
    out["ig_spread"]     = ig_spread
    out["nfci"]          = nfci
    out["term_premium"]  = term_prem
    out["breakeven5y"]   = be5
    out["breakeven5y5y"] = be55

    # HY regime
    if not _nan(hy_spread):
        if hy_spread > THRESH["hy_crisis"]:
            out["hy_regime"] = "crisis"
        elif hy_spread > THRESH["hy_stress"]:
            out["hy_regime"] = "stress"
        else:
            out["hy_regime"] = "normal"
    else:
        out["hy_regime"] = "unknown"

    # Financial conditions regime (NFCI or HY as proxy)
    if not _nan(nfci):
        if nfci > THRESH["nfci_crisis"]:
            out["fin_cond_regime"] = "crisis"
        elif nfci > THRESH["nfci_tight"]:
            out["fin_cond_regime"] = "tightening"
        else:
            out["fin_cond_regime"] = "neutral"
    elif not _nan(hy_spread):
        hy_chg = _chg(fred.get("hy_spread") if fred else None, 22)
        out["fin_cond_regime"] = "tightening" if (not _nan(hy_chg) and hy_chg > 0.1) else "neutral"
    else:
        out["fin_cond_regime"] = "unknown"

    # Inflation regime (from breakevens)
    be = be5 if not _nan(be5) else be55
    if not _nan(be):
        if be > THRESH["breakeven_very_high"]:
            out["inflation_regime"] = "very_high"
        elif be > THRESH["breakeven_high"]:
            out["inflation_regime"] = "high"
        else:
            out["inflation_regime"] = "normal"
    else:
        out["inflation_regime"] = "unknown"

    # MOVE proxy — compute from FRED yield daily changes if available
    if yields is not None and "10Y" in yields.columns:
        y10 = yields["10Y"].dropna()
        if len(y10) >= 22:
            move_p = float(y10.diff().dropna().iloc[-22:].std() * np.sqrt(252) * 100)
            out["move_proxy"] = move_p
            out["move_regime"] = "elevated" if move_p > 100 else "normal"
        else:
            out["move_proxy"] = np.nan
            out["move_regime"] = "unknown"
    else:
        out["move_proxy"] = np.nan
        out["move_regime"] = "unknown"

    # Industrial production momentum
    if indpro is not None and not indpro.empty:
        indpro_yoy = _yoy(indpro, "M") * 100
        out["indpro_yoy"] = indpro_yoy
        out["indpro_regime"] = "contraction" if indpro_yoy < THRESH["indpro_recession"] else \
                               "slowing" if indpro_yoy < 0 else "expanding"
    else:
        out["indpro_yoy"]    = np.nan
        out["indpro_regime"] = "unknown"

    # Cross-asset correlation (equity vs bonds, 60-day rolling)
    if mkt is not None:
        spx = mkt.get("spx")
        tip = mkt.get("tip")
        if spx is not None and tip is not None and len(spx) >= 60 and len(tip) >= 60:
            spx_r = spx.pct_change().dropna()
            tip_r = tip.pct_change().dropna()
            aligned = pd.concat([spx_r, tip_r], axis=1).dropna()
            if len(aligned) >= 60:
                corr = float(aligned.iloc[-60:].corr().iloc[0, 1])
                out["eq_bond_corr"] = corr
                out["correlation_regime"] = "stress" if corr > 0.3 else "normal"
            else:
                out["eq_bond_corr"] = np.nan
                out["correlation_regime"] = "unknown"
        else:
            out["eq_bond_corr"] = np.nan
            out["correlation_regime"] = "unknown"
    else:
        out["eq_bond_corr"] = np.nan
        out["correlation_regime"] = "unknown"

    return out


def _commodities(mkt: dict | None, fred: dict | None,
                 oil_inv: pd.Series | None,
                 eu_gas: pd.DataFrame | None,
                 us_gas: pd.Series | None,
                 fao_fpi: pd.Series | None) -> dict:
    """Oil, copper, gold, gas, agricultural indicators."""
    out = {}

    # ── Oil ───────────────────────────────────────────────────────────────────
    brent = _last(mkt.get("brent") if mkt is not None else None)
    if _nan(brent) and fred:
        brent = _last(fred.get("brent_fred"))
    out["brent"] = brent
    oil_1m = _chg(mkt.get("brent") if mkt is not None else None, 22)
    out["oil_1m_chg"] = oil_1m
    if not _nan(oil_1m):
        if oil_1m > THRESH["oil_surge"]:
            out["oil_regime"] = "surging"
        elif oil_1m < THRESH["oil_crash"]:
            out["oil_regime"] = "crashing"
        else:
            out["oil_regime"] = "stable"
    else:
        out["oil_regime"] = "unknown"

    # US oil inventories (vs 5-year average)
    if oil_inv is not None and not oil_inv.empty and len(oil_inv) >= 260:
        last_inv   = float(oil_inv.iloc[-1])
        avg_5y     = float(oil_inv.iloc[-260:].mean())
        inv_dev    = (last_inv - avg_5y) / avg_5y
        out["us_oil_inventory_last"] = last_inv
        out["us_oil_inventory_dev"]  = inv_dev
        out["oil_inventory_regime"]  = (
            "stress" if inv_dev < THRESH["us_oil_inv_dev_stress"] else
            "surplus" if inv_dev > 0.05 else "normal"
        )
    else:
        out["us_oil_inventory_last"] = np.nan
        out["us_oil_inventory_dev"]  = np.nan
        out["oil_inventory_regime"]  = "unknown"

    # ── Copper ───────────────────────────────────────────────────────────────
    copper = _last(mkt.get("copper") if mkt is not None else None)
    out["copper"] = copper
    cu_1m = _chg(mkt.get("copper") if mkt is not None else None, 22)
    out["copper_1m_chg"] = cu_1m
    if not _nan(cu_1m):
        if cu_1m > THRESH["copper_boom"]:
            out["copper_regime"] = "rising"
        elif cu_1m < THRESH["copper_bust"]:
            out["copper_regime"] = "falling"
        else:
            out["copper_regime"] = "stable"
    else:
        out["copper_regime"] = "unknown"

    # ── Gold ─────────────────────────────────────────────────────────────────
    gold = _last(mkt.get("gold") if mkt is not None else None)
    out["gold"] = gold
    gold_1m = _chg(mkt.get("gold") if mkt is not None else None, 22)
    out["gold_1m_chg"] = gold_1m
    out["gold_regime"] = (
        "surging" if (not _nan(gold_1m) and gold_1m > 0.07) else
        "rising"  if (not _nan(gold_1m) and gold_1m > 0.03) else
        "stable"  if not _nan(gold_1m) else "unknown"
    )

    # Systemic stress: gold and USD both rising
    out["systemic_stress_signal"] = (
        out["gold_regime"] in ("rising","surging") and
        out["dollar_regime"] in ("strong","very_strong")
        if "dollar_regime" in out else False
    )

    # Copper / Gold ratio (growth vs safety)
    if not (_nan(copper) or _nan(gold)) and gold > 0:
        cg = copper / gold * 100
        out["copper_gold_ratio"] = cg
        cg_1m = (
            (float(mkt["copper"].dropna().iloc[-1]) /
             float(mkt["gold"].dropna().iloc[-1])) /
            (float(mkt["copper"].dropna().iloc[-22]) /
             float(mkt["gold"].dropna().iloc[-22])) - 1
            if mkt and len(mkt.get("copper", pd.Series())) >= 22
               and len(mkt.get("gold", pd.Series())) >= 22
            else np.nan
        )
        out["copper_gold_regime"] = (
            "risk_on" if (not _nan(cg_1m) and cg_1m > 0.03) else
            "risk_off" if (not _nan(cg_1m) and cg_1m < -0.03) else "neutral"
        )
    else:
        out["copper_gold_ratio"] = np.nan
        out["copper_gold_regime"] = "unknown"

    # ── Natural gas (US futures) ──────────────────────────────────────────────
    natgas = _last(mkt.get("natgas") if mkt is not None else None)
    out["natgas"] = natgas
    out["natgas_1m_chg"] = _chg(mkt.get("natgas") if mkt is not None else None, 22)

    # ── EU gas storage ────────────────────────────────────────────────────────
    if eu_gas is not None and not eu_gas.empty and "full_pct" in eu_gas.columns:
        pct_full = float(eu_gas["full_pct"].dropna().iloc[-1])
        out["eu_gas_storage_pct"] = pct_full
        out["eu_gas_storage_regime"] = (
            "crisis"       if pct_full < THRESH["eu_gas_storage_crisis"] else
            "stress"       if pct_full < THRESH["eu_gas_storage_low"] else
            "comfortable"  if pct_full > THRESH["eu_gas_storage_high"] else
            "normal"
        )
        # Seasonal: compare to 3-week prior
        if len(eu_gas) >= 21:
            trend = pct_full - float(eu_gas["full_pct"].dropna().iloc[-21])
            out["eu_gas_storage_trend"] = trend  # positive = filling, negative = drawing
        else:
            out["eu_gas_storage_trend"] = np.nan
    else:
        out["eu_gas_storage_pct"]    = np.nan
        out["eu_gas_storage_regime"] = "unknown"
        out["eu_gas_storage_trend"]  = np.nan

    # ── US gas storage ────────────────────────────────────────────────────────
    if us_gas is not None and not us_gas.empty and len(us_gas) >= 52:
        last_us = float(us_gas.iloc[-1])
        avg_5y_us = float(us_gas.iloc[-260:].mean()) if len(us_gas) >= 260 else np.nan
        out["us_gas_storage_last"] = last_us
        out["us_gas_storage_dev"]  = (
            (last_us - avg_5y_us) / avg_5y_us if not _nan(avg_5y_us) and avg_5y_us > 0 else np.nan
        )
    else:
        out["us_gas_storage_last"] = np.nan
        out["us_gas_storage_dev"]  = np.nan

    # ── Agricultural (FAO + FRED futures) ─────────────────────────────────────
    fpi = _last(fao_fpi)
    out["fao_fpi"] = fpi
    if not _nan(fpi):
        out["fao_fpi_regime"] = (
            "crisis" if fpi > THRESH["fao_fpi_crisis"] else
            "stress" if fpi > THRESH["fao_fpi_stress"] else "normal"
        )
        # 3M change
        fao_3m = _chg(fao_fpi, 3)
        out["fao_fpi_3m_chg"] = fao_3m
    else:
        out["fao_fpi_regime"]  = "unknown"
        out["fao_fpi_3m_chg"]  = np.nan

    # Wheat, corn, rice from yfinance futures
    wheat = _last(mkt.get("wheat") if mkt is not None else None)
    corn  = _last(mkt.get("corn")  if mkt is not None else None)
    out["wheat"] = wheat
    out["corn"]  = corn
    out["wheat_1m_chg"] = _chg(mkt.get("wheat") if mkt is not None else None, 22)
    out["corn_1m_chg"]  = _chg(mkt.get("corn")  if mkt is not None else None, 22)

    # Agricultural stress composite (FPI + wheat + corn)
    stress_signals = 0
    if not _nan(fpi) and fpi > THRESH["fao_fpi_stress"]:
        stress_signals += 1
    if not _nan(out.get("wheat_1m_chg")) and out["wheat_1m_chg"] > 0.08:
        stress_signals += 1
    if not _nan(out.get("corn_1m_chg")) and out["corn_1m_chg"] > 0.08:
        stress_signals += 1
    out["agri_stress_score"]  = stress_signals
    out["agri_stress_regime"] = "crisis" if stress_signals >= 3 else \
                                "stress" if stress_signals >= 2 else \
                                "warning" if stress_signals == 1 else "normal"

    # Nickel (critical minerals proxy — available via yfinance)
    nickel = _last(mkt.get("nickel") if mkt is not None else None)
    out["nickel"] = nickel
    out["nickel_1m_chg"] = _chg(mkt.get("nickel") if mkt is not None else None, 22)

    return out


def _em_africa(fx_reserves: pd.DataFrame | None,
               fdi: pd.DataFrame | None,
               ext_debt: pd.DataFrame | None,
               mkt: dict | None) -> dict:
    """EM and Africa stress indicators from real data."""
    out = {}

    # ── FX reserves (World Bank) ──────────────────────────────────────────────
    if fx_reserves is not None and not fx_reserves.empty:
        out["fx_res_available"] = True
        # Most recent values
        latest = fx_reserves.dropna(how="all").iloc[-1] if not fx_reserves.empty else pd.Series()
        out["fx_res_latest"] = latest.to_dict()
        # Deterioration: compare last 2 available years
        if len(fx_reserves) >= 2:
            prev  = fx_reserves.dropna(how="all").iloc[-2]
            chg   = ((latest - prev) / prev.abs()).replace([np.inf, -np.inf], np.nan)
            worst = chg.idxmin() if not chg.empty else ""
            out["fx_res_deteriorating"] = bool((chg < -0.10).any())
            out["fx_res_worst_country"] = str(worst)
        else:
            out["fx_res_deteriorating"] = False
            out["fx_res_worst_country"] = ""
    else:
        out["fx_res_available"]      = False
        out["fx_res_latest"]         = {}
        out["fx_res_deteriorating"]  = False
        out["fx_res_worst_country"]  = ""

    # USD debt vulnerability (external debt / GDP direction from WB)
    if ext_debt is not None and not ext_debt.empty:
        out["ext_debt_available"] = True
        # Most recent row — check if any EM had >40% external debt to GDP (proxy)
        latest_debt = ext_debt.dropna(how="all").iloc[-1]
        out["ext_debt_latest"] = latest_debt.to_dict()
    else:
        out["ext_debt_available"]  = False
        out["ext_debt_latest"]     = {}

    # EM FX performance (from yfinance)
    em_fx = {
        "BRL": mkt.get("usdbrl") if mkt else None,
        "ZAR": mkt.get("usdzar") if mkt else None,
        "TRY": mkt.get("usdtry") if mkt else None,
        "CNH": mkt.get("usdcnh") if mkt else None,
        "INR": mkt.get("usdinr") if mkt else None,
        "MXN": mkt.get("usdmxn") if mkt else None,
    }
    depreciations = {}
    for ccy, s in em_fx.items():
        chg = _chg(s, 22)
        if not _nan(chg):
            depreciations[ccy] = chg  # positive = USD appreciated = EM depreciated
    out["em_fx_1m_depreciation"] = depreciations
    out["em_fx_stress_avg"] = float(np.mean(list(depreciations.values()))) if depreciations else np.nan

    # EM equity stress (EEM 1M return)
    eem_chg = _chg(mkt.get("eem") if mkt is not None else None, 22)
    out["eem_1m_chg"] = eem_chg
    out["em_equity_regime"] = (
        "crisis"  if (not _nan(eem_chg) and eem_chg < -0.15) else
        "stress"  if (not _nan(eem_chg) and eem_chg < -0.08) else
        "normal"  if not _nan(eem_chg) else "unknown"
    )

    # USD vulnerability composite
    dollar_rg = "unknown"  # will be filled from macro block
    score = 0
    if out["fx_res_deteriorating"]:
        score += 1
    if not _nan(out["em_fx_stress_avg"]) and out["em_fx_stress_avg"] > 0.03:
        score += 1
    if not _nan(eem_chg) and eem_chg < -0.05:
        score += 1
    out["em_stress_score"] = score
    out["em_regime"] = "crisis" if score >= 3 else "stress" if score >= 2 else \
                       "elevated" if score >= 1 else "normal"

    return out


def _market_implied(mkt: dict | None, yields: pd.DataFrame | None,
                    fred: dict | None) -> dict:
    """Market-implied indicators: VIX term structure, correlations, HYG."""
    out = {}

    # HYG ETF proxy for financial conditions
    hyg_1m = _chg(mkt.get("hyg") if mkt is not None else None, 22)
    out["hyg_1m_chg"] = hyg_1m
    # EMB (EM bonds) as EM spread proxy
    emb_1m = _chg(mkt.get("emb") if mkt is not None else None, 22)
    out["emb_1m_chg"] = emb_1m
    out["embi_proxy"]  = _last(mkt.get("emb") if mkt is not None else None)

    # TIP (TIPS ETF) vs LQD (IG bonds) — real rate signal
    tip_1m = _chg(mkt.get("tip") if mkt is not None else None, 22)
    lqd_1m = _chg(mkt.get("lqd") if mkt is not None else None, 22)
    out["tip_1m_chg"] = tip_1m
    out["lqd_1m_chg"] = lqd_1m

    # Sector ETF momentum (real equity data)
    for key in ("xlf", "xle", "xlb", "xli"):
        chg = _chg(mkt.get(key) if mkt is not None else None, 22)
        out[f"{key}_1m_chg"] = chg

    return out


def _composites(ind: dict) -> dict:
    """Composite stress scores and macro regime from individual indicators."""
    out = {}

    fin_stress = 0
    if ind.get("vix_regime") in ("elevated", "high"):        fin_stress += 1
    if ind.get("hy_regime") in ("stress", "crisis"):         fin_stress += 1
    if ind.get("fin_cond_regime") in ("tightening","crisis"):fin_stress += 1
    if ind.get("curve_regime") == "inverted":                fin_stress += 1
    if ind.get("inflation_regime") in ("high","very_high"):  fin_stress += 1
    if ind.get("correlation_regime") == "stress":            fin_stress += 1
    if ind.get("move_regime") == "elevated":                 fin_stress += 1
    out["financial_stress_score"] = fin_stress

    geo_stress = 0
    if ind.get("oil_regime") == "surging":                   geo_stress += 1
    if ind.get("eu_gas_storage_regime") in ("stress","crisis"):geo_stress += 1
    if ind.get("oil_inventory_regime") == "stress":          geo_stress += 1
    if ind.get("agri_stress_regime") in ("stress","crisis"): geo_stress += 1
    out["geo_stress_score"] = geo_stress

    total = fin_stress + geo_stress
    out["total_stress_score"] = total
    out["macro_regime"] = (
        "crisis"   if total >= 8 else
        "stressed" if total >= 5 else
        "cautious" if total >= 3 else
        "benign"
    )

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(
    mkt:          dict | None,
    yields:       pd.DataFrame | None,
    fred:         dict | None,
    oil_inv:      pd.Series | None   = None,
    eu_gas:       pd.DataFrame | None = None,
    us_gas:       pd.Series | None   = None,
    fao_fpi:      pd.Series | None   = None,
    fx_reserves:  pd.DataFrame | None = None,
    fdi:          pd.DataFrame | None = None,
    ext_debt:     pd.DataFrame | None = None,
    oecd_cli:     pd.DataFrame | None = None,
) -> dict:
    """
    Compute all regime indicators from real data.
    Any unavailable data results in np.nan / 'unknown' — never synthetic values.
    """
    ind: dict = {}

    ind.update(_macro(mkt, yields, fred))

    # Systemic stress needs dollar_regime from macro block
    comm = _commodities(mkt, fred, oil_inv, eu_gas, us_gas, fao_fpi)
    # Fix systemic_stress_signal reference
    comm["systemic_stress_signal"] = (
        comm.get("gold_regime") in ("rising", "surging") and
        ind.get("dollar_regime") in ("strong", "very_strong")
    )
    ind.update(comm)

    ind.update(_em_africa(fx_reserves, fdi, ext_debt, mkt))
    ind.update(_market_implied(mkt, yields, fred))

    # OECD CLI
    if oecd_cli is not None and not oecd_cli.empty:
        latest_cli = oecd_cli.dropna(how="all").iloc[-1]
        ind["oecd_cli_latest"] = latest_cli.to_dict()
        # OECDALL or USA as headline
        for col in ("OECDALL", "USA", "US"):
            if col in latest_cli.index:
                v = float(latest_cli[col])
                ind["oecd_cli_oecdall"] = v
                ind["oecd_cli_regime"]  = (
                    "expanding"    if v > 100.5 else
                    "contracting"  if v < 99.5  else "neutral"
                )
                break
    else:
        ind["oecd_cli_oecdall"] = np.nan
        ind["oecd_cli_regime"]  = "unknown"

    ind.update(_composites(ind))
    return ind
