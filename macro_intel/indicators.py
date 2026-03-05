"""
indicators.py — Compute all regime indicators from raw data.

Input:  raw DataFrames / Series from data_fetchers.py
Output: flat dict consumed by rules_engine, inference_engine, narrative_generator, and the UI.

Organized in 5 domains:
  MACRO      — yield curve, dollar, financial conditions, inflation
  GEO-RISK   — GPR, shipping, sanctions, cyber
  COMMODITIES— oil, copper tightness, critical minerals
  EM/AFRICA  — spreads, FX reserves adequacy, USD debt vulnerability
  MARKET     — VIX regime, correlation, term premium, breakevens
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import THRESH


# ── Safe helpers ───────────────────────────────────────────────────────────────
def _last(s, default=np.nan):
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    if s is None: return default
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else default


def _prev(s, n=1, default=np.nan):
    if s is None: return default
    s = s.dropna()
    return float(s.iloc[-1 - n]) if len(s) > n else default


def _chg(s, n=22, pct=True):
    """n-period change. n=22 ≈ 1 month trading days."""
    if s is None: return np.nan
    s = s.dropna()
    if len(s) <= n: return np.nan
    return float((s.iloc[-1] / s.iloc[-n] - 1) if pct else (s.iloc[-1] - s.iloc[-n]))


def _roll_std(s, n=22):
    if s is None: return np.nan
    s = s.dropna()
    return float(s.tail(n).std()) if len(s) >= n else np.nan


def _roll_corr(a: pd.Series, b: pd.Series, n=60) -> float:
    try:
        df = pd.concat([a.dropna(), b.dropna()], axis=1).dropna()
        return float(df.tail(n).corr().iloc[0, 1]) if len(df) >= n else np.nan
    except Exception:
        return np.nan


def _zscore(s: pd.Series, window=252) -> float:
    """Z-score of latest value vs rolling window."""
    s = s.dropna()
    if len(s) < window // 2: return np.nan
    mu, sigma = float(s.tail(window).mean()), float(s.tail(window).std())
    return float((s.iloc[-1] - mu) / sigma) if sigma > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MACRO indicators
# ══════════════════════════════════════════════════════════════════════════════

def _macro(ind: dict, mkt: pd.DataFrame, yields: pd.DataFrame, fred: dict):
    # ── Yield curve ──────────────────────────────────────────────────────────
    yld_10 = yields.get("10Y") if isinstance(yields, pd.DataFrame) and "10Y" in yields.columns else None
    yld_2  = yields.get("2Y")  if isinstance(yields, pd.DataFrame) and "2Y"  in yields.columns else None
    yld_3m = yields.get("3M")  if isinstance(yields, pd.DataFrame) and "3M"  in yields.columns else None
    yld_30 = yields.get("30Y") if isinstance(yields, pd.DataFrame) and "30Y" in yields.columns else None

    if yld_10 is not None and yld_2 is not None:
        slope_s = (yld_10 - yld_2) * 100
        ind["curve_slope"]          = _last(slope_s)
        ind["curve_slope_1m"]       = _prev(slope_s.dropna(), n=22)
        ind["curve_slope_series"]   = slope_s
        ind["us10y"]                = _last(yld_10)
        ind["us2y"]                 = _last(yld_2)
        ind["us3m"]                 = _last(yld_3m)
        ind["us30y"]                = _last(yld_30)

        slope = ind["curve_slope"]
        ind["curve_regime"] = (
            "inverted" if slope < THRESH["curve_inverted"] else
            "flat"     if slope < THRESH["curve_flat"]     else
            "steep"    if slope > THRESH["curve_steep"]    else "normal"
        )
        ind["curve_direction"] = (
            "steepening" if not np.isnan(ind["curve_slope_1m"]) and slope > ind["curve_slope_1m"]
            else "flattening"
        )

    # ── Dollar regime ─────────────────────────────────────────────────────────
    if "dxy" in mkt.columns:
        dxy_s  = mkt["dxy"].dropna()
        d      = _last(dxy_s)
        ma50   = float(dxy_s.tail(50).mean()) if len(dxy_s) >= 50 else d
        ma200  = float(dxy_s.tail(200).mean()) if len(dxy_s) >= 200 else d
        ind["dxy"]          = d
        ind["dxy_1m_chg"]   = _chg(dxy_s)
        ind["dxy_zscore"]   = _zscore(dxy_s)
        ind["dollar_regime"] = (
            "very_strong" if d > THRESH["dxy_very_strong"] else
            "strong"      if d > THRESH["dxy_strong"]      else
            "weakening"   if d < ma50 * 0.97               else "neutral"
        )
        ind["dollar_vs_200ma"] = "above" if d > ma200 else "below"

    # ── Financial conditions composite ────────────────────────────────────────
    fci_components = []
    if "hy_spread" in fred:
        hy  = fred["hy_spread"].dropna()
        hy_z = _zscore(hy)
        ind["hy_spread"] = _last(hy)
        ind["hy_1m_chg"] = _chg(hy)
        if not np.isnan(hy_z): fci_components.append(hy_z)
    if "nfci" in fred:
        nf   = fred["nfci"].dropna()
        ind["nfci"]        = _last(nf)
        ind["nfci_zscore"] = _zscore(nf)
        if not np.isnan(ind["nfci_zscore"]): fci_components.append(ind["nfci_zscore"])
    if "vix" in mkt.columns:
        vix_z = _zscore(mkt["vix"].dropna())
        if not np.isnan(vix_z): fci_components.append(vix_z)
    if "hyg" in mkt.columns:
        hyg_chg = _chg(mkt["hyg"])
        ind["hyg_1m_chg"] = hyg_chg
        if not np.isnan(hyg_chg): fci_components.append(-hyg_chg / 0.05)

    ind["fci_composite"] = float(np.mean(fci_components)) if fci_components else np.nan
    ind["fin_cond_regime"] = (
        "tightening" if not np.isnan(ind.get("fci_composite", np.nan)) and ind["fci_composite"] > 0.5 else
        "easing"     if not np.isnan(ind.get("fci_composite", np.nan)) and ind["fci_composite"] < -0.5 else
        "neutral"
    )

    # ── Inflation signals ─────────────────────────────────────────────────────
    if "breakeven5y" in fred:
        be5 = fred["breakeven5y"].dropna()
        ind["breakeven5y"]    = _last(be5)
        ind["breakeven5y_1m"] = _chg(be5)
        be_val = ind["breakeven5y"]
        ind["inflation_regime"] = (
            "very_high" if be_val > THRESH["breakeven_very_high"] else
            "high"      if be_val > THRESH["breakeven_high"]       else "normal"
        )
    if "breakeven5y5y" in fred:
        be55 = fred["breakeven5y5y"].dropna()
        ind["breakeven5y5y"]    = _last(be55)
        ind["breakeven5y5y_1m"] = _chg(be55)

    # ── Term premium ──────────────────────────────────────────────────────────
    if "term_premium" in fred:
        tp = fred["term_premium"].dropna()
        ind["term_premium"]    = _last(tp)
        ind["term_premium_1m"] = _chg(tp)
    elif yld_10 is not None and yld_3m is not None:
        # Simplified proxy: 10Y - (rolling 10Y average of 3M)
        r3m = yld_3m.dropna()
        if len(r3m) >= 252:
            expected_10y = float(r3m.rolling(252).mean().iloc[-1])
            ind["term_premium"] = float((_last(yld_10) or 0) - expected_10y)
        else:
            ind["term_premium"] = np.nan


# ══════════════════════════════════════════════════════════════════════════════
# GEO-RISK indicators
# ══════════════════════════════════════════════════════════════════════════════

def _georisk(ind: dict, gpr: pd.Series, ship: pd.Series,
             sanctions: pd.Series, cyber: pd.Series):
    if not gpr.empty:
        gv = _last(gpr)
        ind["gpr"]         = gv
        ind["gpr_mom_chg"] = _chg(gpr, n=1)
        ind["gpr_3m_chg"]  = _chg(gpr, n=3)
        ind["gpr_regime"]  = (
            "high"     if gv > THRESH["gpr_high"]     else
            "elevated" if gv > THRESH["gpr_elevated"] else "normal"
        )

    if not ship.empty:
        sv = _last(ship)
        ind["shipping"]        = sv
        ind["shipping_1m_chg"] = _chg(ship, n=4)  # weekly data
        ind["shipping_regime"] = (
            "crisis" if sv > THRESH["shipping_crisis"] else
            "stress" if sv > THRESH["shipping_stress"] else "normal"
        )

    if not sanctions.empty:
        sc = _last(sanctions)
        ind["sanctions"]        = sc
        ind["sanctions_1m_chg"] = _chg(sanctions, n=1)
        ind["sanctions_elevated"] = bool(sc > 150)

    if not cyber.empty:
        cy = _last(cyber)
        ind["cyber_risk"]    = cy
        ind["cyber_elevated"] = bool(cy > 150)

    # Composite geo-risk score (0-4)
    ind["geo_stress_score"] = int(sum([
        ind.get("gpr_regime") in ("elevated", "high"),
        ind.get("shipping_regime") in ("stress", "crisis"),
        ind.get("sanctions_elevated", False),
        ind.get("cyber_elevated", False),
    ]))


# ══════════════════════════════════════════════════════════════════════════════
# COMMODITY indicators
# ══════════════════════════════════════════════════════════════════════════════

def _commodities(ind: dict, mkt: pd.DataFrame, minerals: pd.DataFrame):
    # ── Oil ──────────────────────────────────────────────────────────────────
    if "brent" in mkt.columns:
        oil_s = mkt["brent"].dropna()
        oil   = _last(oil_s)
        oil_chg_1m = _chg(oil_s)
        oil_vol    = _roll_std(oil_s.pct_change().dropna()) * np.sqrt(252) * 100  # annualised %
        ind["oil"]          = oil
        ind["oil_1m_chg"]   = oil_chg_1m
        ind["oil_vol"]      = oil_vol
        ind["oil_regime"]   = (
            "surging"  if not np.isnan(oil_chg_1m) and oil_chg_1m >  THRESH["oil_surge"] else
            "crashing" if not np.isnan(oil_chg_1m) and oil_chg_1m <  THRESH["oil_crash"] else "stable"
        )
        # Oil stress index: price + volatility
        oil_stress = 0.0
        if oil > 100:           oil_stress += 1
        if oil > 120:           oil_stress += 1
        if not np.isnan(oil_vol) and oil_vol > 35: oil_stress += 1
        ind["oil_stress_score"] = oil_stress

    # ── Copper tightness ─────────────────────────────────────────────────────
    if "copper" in mkt.columns:
        cu_s     = mkt["copper"].dropna()
        cu       = _last(cu_s)
        cu_chg_1m = _chg(cu_s)
        # LME inventory proxy: falling price with rising copper = tighter (mock score)
        # Real: replace with LME warrants data
        cu_trend_3m = _chg(cu_s, n=63)
        ind["copper"]         = cu
        ind["copper_1m_chg"]  = cu_chg_1m
        ind["copper_3m_chg"]  = cu_trend_3m
        ind["copper_regime"]  = (
            "rising"  if not np.isnan(cu_chg_1m) and cu_chg_1m >  THRESH["copper_boom"] else
            "falling" if not np.isnan(cu_chg_1m) and cu_chg_1m <  THRESH["copper_bust"] else "stable"
        )
        ind["growth_signal"] = (
            "positive" if not np.isnan(cu_chg_1m) and cu_chg_1m >  0.03 else
            "negative" if not np.isnan(cu_chg_1m) and cu_chg_1m < -0.03 else "neutral"
        )
        # Copper/Gold ratio (growth vs safety signal)
        if "gold" in mkt.columns:
            gold = _last(mkt["gold"])
            if gold and gold > 0:
                cu_au_ratio = cu / (gold / 1000)  # copper ($/lb) / (gold ($/oz) / 1000)
                ind["copper_gold_ratio"]     = cu_au_ratio
                ind["copper_gold_1m_chg"]    = _chg(
                    (mkt["copper"] / mkt["gold"] * 1000).dropna())
                ind["copper_gold_regime"] = "risk_on" if cu_au_ratio > 0.18 else "risk_off"

    # ── Gold ─────────────────────────────────────────────────────────────────
    if "gold" in mkt.columns:
        g_s = mkt["gold"].dropna()
        ind["gold"]        = _last(g_s)
        ind["gold_1m_chg"] = _chg(g_s)
        ind["gold_regime"] = (
            "surging" if not np.isnan(ind["gold_1m_chg"]) and ind["gold_1m_chg"] > 0.05 else
            "falling" if not np.isnan(ind["gold_1m_chg"]) and ind["gold_1m_chg"] < -0.04 else "stable"
        )

    # ── Critical minerals composite ───────────────────────────────────────────
    if not minerals.empty:
        minerals_last = minerals.dropna().iloc[-1] if len(minerals.dropna()) > 0 else None
        if minerals_last is not None:
            ind["minerals_composite"] = float(minerals_last.mean())
            ind["minerals_1m_chg"]    = float(minerals.pct_change().dropna().iloc[-1].mean()) \
                if len(minerals.pct_change().dropna()) > 0 else np.nan
            ind["minerals_stressed"]  = bool(
                not np.isnan(ind["minerals_1m_chg"]) and ind["minerals_1m_chg"] > 0.08)


# ══════════════════════════════════════════════════════════════════════════════
# EM / AFRICA indicators
# ══════════════════════════════════════════════════════════════════════════════

def _em_africa(ind: dict, mkt: pd.DataFrame, em_spreads: pd.DataFrame,
               fx_reserves: pd.DataFrame):
    # ── EM sovereign spreads ──────────────────────────────────────────────────
    if not em_spreads.empty and "EMBI Global" in em_spreads.columns:
        embi = em_spreads["EMBI Global"].dropna()
        embi_val = _last(embi)
        ind["embi"]               = embi_val
        ind["embi_prev"]          = _prev(embi)
        ind["embi_1m_chg"]        = _chg(embi, n=1)
        ind["em_spreads_widening"] = bool(embi_val > ind["embi_prev"])
        ind["em_regime"]          = (
            "crisis" if embi_val > THRESH["em_spread_crisis"] else
            "stress" if embi_val > THRESH["em_spread_stress"] else "normal"
        )
    if not em_spreads.empty and "Africa Composite" in em_spreads.columns:
        ind["africa_spreads"] = _last(em_spreads["Africa Composite"].dropna())

    # ── FX reserves adequacy ──────────────────────────────────────────────────
    if not fx_reserves.empty and len(fx_reserves) >= 2:
        chgs = fx_reserves.iloc[-1] / fx_reserves.iloc[-2] - 1
        ind["fx_res_min_chg"]        = float(chgs.min())
        ind["fx_res_worst_country"]  = str(chgs.idxmin())
        ind["fx_res_deteriorating"]  = bool((chgs < -0.05).any())
        ind["fx_res_latest"]         = fx_reserves.iloc[-1].to_dict()

    # ── USD debt vulnerability proxy ──────────────────────────────────────────
    # If FX reserves declining AND dollar regime strong → elevated vulnerability
    dol_rg = ind.get("dollar_regime", "neutral")
    fx_det = ind.get("fx_res_deteriorating", False)
    ind["usd_debt_vulnerability"] = (
        "high"   if dol_rg in ("strong", "very_strong") and fx_det else
        "medium" if dol_rg in ("strong", "very_strong") or fx_det else "low"
    )

    # ── EM FX stress ─────────────────────────────────────────────────────────
    em_fx = [c for c in ["usdbrl", "usdzar", "usdtry", "usdcnh"] if c in mkt.columns]
    if em_fx:
        avg_em_fx_chg = float(np.nanmean([_chg(mkt[c]) for c in em_fx]))
        ind["em_fx_avg_depreciation"] = avg_em_fx_chg
        ind["em_fx_stress"] = bool(avg_em_fx_chg > 0.03)

    # ── EMB ETF (market-implied EM stress) ────────────────────────────────────
    if "emb" in mkt.columns:
        ind["emb_1m_chg"] = _chg(mkt["emb"])


# ══════════════════════════════════════════════════════════════════════════════
# MARKET-IMPLIED indicators
# ══════════════════════════════════════════════════════════════════════════════

def _market_implied(ind: dict, mkt: pd.DataFrame, fred: dict):
    # ── VIX ──────────────────────────────────────────────────────────────────
    if "vix" in mkt.columns:
        vix_s = mkt["vix"].dropna()
        v = _last(vix_s)
        ind["vix"]         = v
        ind["vix_1m_chg"]  = _chg(vix_s)
        ind["vix_ma20"]    = float(vix_s.tail(20).mean()) if len(vix_s) >= 20 else v
        ind["vix_regime"]  = (
            "high"     if v >= THRESH["vix_high"]     else
            "elevated" if v >= THRESH["vix_elevated"] else "normal"
        )
        # VIX term structure proxy: spot VIX vs 20d MA (backwardation if spot > MA)
        ind["vix_term_structure"] = (
            "backwardation" if v > ind.get("vix_ma20", v) * 1.05 else
            "contango"      if v < ind.get("vix_ma20", v) * 0.95 else "flat"
        )

    # ── Cross-asset correlation regime ────────────────────────────────────────
    if "spx" in mkt.columns and "us10y" in mkt.columns:
        spx_r  = mkt["spx"].dropna().pct_change().dropna()
        y10_r  = mkt["us10y"].dropna().diff().dropna()
        corr   = _roll_corr(spx_r, y10_r, n=60)
        ind["equity_bond_corr"] = corr
        # Positive equity-bond correlation = stress regime (both sell off)
        ind["correlation_regime"] = (
            "stress" if not np.isnan(corr) and corr > 0.2 else
            "normal" if not np.isnan(corr) and corr < -0.2 else "mixed"
        )

    if "gold" in mkt.columns and "dxy" in mkt.columns:
        gold_r = mkt["gold"].dropna().pct_change().dropna()
        dxy_r  = mkt["dxy"].dropna().pct_change().dropna()
        ind["gold_dxy_corr"] = _roll_corr(gold_r, dxy_r, n=60)
        # When gold and DXY both rise → flight to ALL safe havens = systemic stress
        gold_chg = _chg(mkt["gold"])
        dxy_chg  = _chg(mkt["dxy"])
        ind["systemic_stress_signal"] = bool(
            not (np.isnan(gold_chg) or np.isnan(dxy_chg)) and gold_chg > 0.03 and dxy_chg > 0.01
        )

    # ── MOVE proxy (bond vol): rolling std of 10Y yield daily changes ─────────
    if "us10y" in mkt.columns:
        y10 = mkt["us10y"].dropna()
        if len(y10) >= 22:
            move_proxy = float(y10.diff().dropna().tail(22).std() * np.sqrt(252) * 100)
            ind["move_proxy"]         = move_proxy
            ind["move_regime"]        = "elevated" if move_proxy > 90 else "normal"

    # ── Credit spread signals ─────────────────────────────────────────────────
    hy = ind.get("hy_spread", np.nan)
    if not np.isnan(hy):
        ind["hy_regime"] = (
            "crisis" if hy > THRESH["hy_crisis"] else
            "stress" if hy > THRESH["hy_stress"] else "normal"
        )


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE stress scores
# ══════════════════════════════════════════════════════════════════════════════

def _composites(ind: dict):
    ind["financial_stress_score"] = int(sum([
        ind.get("vix_regime")       in ("elevated", "high"),
        ind.get("dollar_regime")    in ("strong", "very_strong"),
        ind.get("em_regime")        in ("stress", "crisis"),
        ind.get("curve_regime")     == "inverted",
        ind.get("fin_cond_regime")  == "tightening",
        ind.get("hy_regime", "")    in ("stress", "crisis"),
        ind.get("correlation_regime") == "stress",
    ]))
    ind["total_stress_score"] = (
        ind["financial_stress_score"] + ind.get("geo_stress_score", 0)
    )
    # Overall macro regime label
    total = ind["total_stress_score"]
    ind["macro_regime"] = (
        "crisis"   if total >= 7 else
        "stressed" if total >= 4 else
        "cautious" if total >= 2 else "benign"
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(
    mkt:         pd.DataFrame,
    yields:      pd.DataFrame,
    fred:        dict,
    gpr:         pd.Series,
    ship:        pd.Series,
    sanctions:   pd.Series,
    cyber:       pd.Series,
    em_spreads:  pd.DataFrame,
    fx_reserves: pd.DataFrame,
    minerals:    pd.DataFrame,
) -> dict:
    """
    Compute all indicators from raw data.
    Returns flat dict — all keys documented inline above.
    """
    ind: dict = {}

    _macro(ind, mkt, yields, fred)
    _georisk(ind, gpr, ship, sanctions, cyber)
    _commodities(ind, mkt, minerals)
    _em_africa(ind, mkt, em_spreads, fx_reserves)
    _market_implied(ind, mkt, fred)
    _composites(ind)

    return ind
