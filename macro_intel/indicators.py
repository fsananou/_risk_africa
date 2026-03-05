"""
indicators.py — compute regime indicators from raw data.

Input:  raw DataFrames from data_fetchers.py
Output: flat dict of scalar values, regime labels, and lightweight Series
        consumed by rules_engine.py and the Streamlit charts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import THRESH


# ── Safe helpers ──────────────────────────────────────────────────────────────
def _last(s: pd.Series | pd.DataFrame, default: float = np.nan) -> float:
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else default


def _chg(s: pd.Series, n: int = 22) -> float:
    """n-period % change. n=22 ≈ 1 month of trading days."""
    s = s.dropna()
    if len(s) <= n:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-n] - 1)


def _prev(s: pd.Series, default: float = np.nan) -> float:
    s = s.dropna()
    return float(s.iloc[-2]) if len(s) >= 2 else default


# ── Main function ─────────────────────────────────────────────────────────────
def compute_all(
    mkt: pd.DataFrame,
    yc: pd.DataFrame,
    gpr: pd.Series,
    ship: pd.Series,
    em_spreads: pd.DataFrame,
    fx_res: pd.DataFrame,
) -> dict:
    """
    Compute all indicators and return a flat dict.
    Keys prefixed by domain:  curve_*, vix_*, dollar_*, oil_*, copper_*,
    gpr_*, ship_*, em_*, fx_*, fin_*, cross_*
    """
    ind: dict = {}

    # ── Yield curve ──────────────────────────────────────────────────────────
    if not yc.empty and "10 Yr" in yc.columns and "2 Yr" in yc.columns:
        slope_s = (yc["10 Yr"] - yc["2 Yr"]) * 100  # bps
        ind["curve_slope_series"] = slope_s
        cur = _last(slope_s)
        ind["curve_slope"] = cur
        ind["curve_slope_1m"] = _last(slope_s.dropna().iloc[:-22]) \
            if len(slope_s.dropna()) > 22 else np.nan

        ind["us10y"] = _last(yc["10 Yr"])
        ind["us2y"]  = _last(yc["2 Yr"])
        ind["us3m"]  = _last(yc.get("1 Mo", pd.Series(dtype=float)))
        ind["us30y"] = _last(yc.get("30 Yr", pd.Series(dtype=float)))

        ind["curve_regime"] = (
            "inverted" if cur < THRESH["curve_inverted"] else
            "flat"     if cur < THRESH["curve_flat"]     else
            "steep"    if cur > THRESH["curve_steep"]    else
            "normal"
        )
        slope_1m = ind["curve_slope_1m"]
        ind["curve_direction"] = (
            "steepening" if not np.isnan(slope_1m) and cur > slope_1m else
            "flattening"
        )

    # ── VIX ──────────────────────────────────────────────────────────────────
    if "vix" in mkt.columns:
        v = _last(mkt["vix"])
        ind["vix"] = v
        ind["vix_1m_chg"] = _chg(mkt["vix"])
        ind["vix_regime"] = (
            "high"     if v >= THRESH["vix_high"]     else
            "elevated" if v >= THRESH["vix_elevated"] else
            "normal"
        )

    # ── Dollar ───────────────────────────────────────────────────────────────
    if "dxy" in mkt.columns:
        dxy_s  = mkt["dxy"].dropna()
        d      = _last(dxy_s)
        ma50   = float(dxy_s.tail(50).mean()) if len(dxy_s) >= 50 else d
        ind["dxy"]         = d
        ind["dxy_1m_chg"]  = _chg(dxy_s)
        ind["dollar_regime"] = (
            "very_strong" if d > THRESH["dxy_very_strong"] else
            "strong"      if d > THRESH["dxy_strong"]      else
            "weakening"   if d < ma50 * 0.97               else
            "neutral"
        )

    # ── Oil ──────────────────────────────────────────────────────────────────
    if "brent" in mkt.columns:
        oil_chg = _chg(mkt["brent"])
        ind["oil"]         = _last(mkt["brent"])
        ind["oil_1m_chg"]  = oil_chg
        ind["oil_regime"]  = (
            "surging"  if not np.isnan(oil_chg) and oil_chg >  THRESH["oil_surge"] else
            "crashing" if not np.isnan(oil_chg) and oil_chg <  THRESH["oil_crash"] else
            "stable"
        )

    # ── Copper (leading growth indicator) ────────────────────────────────────
    if "copper" in mkt.columns:
        cu_chg = _chg(mkt["copper"])
        ind["copper"]          = _last(mkt["copper"])
        ind["copper_1m_chg"]   = cu_chg
        ind["copper_regime"]   = (
            "rising"  if not np.isnan(cu_chg) and cu_chg >  THRESH["copper_boom"] else
            "falling" if not np.isnan(cu_chg) and cu_chg <  THRESH["copper_bust"] else
            "stable"
        )
        ind["growth_signal"] = (
            "positive" if not np.isnan(cu_chg) and cu_chg >  0.03 else
            "negative" if not np.isnan(cu_chg) and cu_chg < -0.03 else
            "neutral"
        )

    # ── Gold ─────────────────────────────────────────────────────────────────
    if "gold" in mkt.columns:
        ind["gold"]          = _last(mkt["gold"])
        ind["gold_1m_chg"]   = _chg(mkt["gold"])
        ind["gold_regime"]   = (
            "surging" if not np.isnan(ind["gold_1m_chg"]) and ind["gold_1m_chg"] > 0.05
            else "falling" if not np.isnan(ind["gold_1m_chg"]) and ind["gold_1m_chg"] < -0.05
            else "stable"
        )

    # ── Financial conditions proxy (HYG ETF) ─────────────────────────────────
    if "hyg" in mkt.columns:
        hyg_chg = _chg(mkt["hyg"])
        ind["hyg_1m_chg"] = hyg_chg
        ind["fin_cond_regime"] = (
            "tightening" if not np.isnan(hyg_chg) and hyg_chg < THRESH["fin_cond_tightening"] else
            "easing"     if not np.isnan(hyg_chg) and hyg_chg > THRESH["fin_cond_easing"]     else
            "neutral"
        )

    # ── EM bond ETF (market-implied EM stress) ────────────────────────────────
    if "emb" in mkt.columns:
        ind["emb_1m_chg"] = _chg(mkt["emb"])

    # ── GPR ──────────────────────────────────────────────────────────────────
    if not gpr.empty:
        gv = _last(gpr)
        pv = _prev(gpr)
        ind["gpr"]          = gv
        ind["gpr_mom_chg"]  = (gv / pv - 1) if not np.isnan(pv) and pv > 0 else np.nan
        ind["gpr_regime"]   = (
            "high"     if gv > THRESH["gpr_high"]     else
            "elevated" if gv > THRESH["gpr_elevated"] else
            "normal"
        )

    # ── Shipping ─────────────────────────────────────────────────────────────
    if not ship.empty:
        sv = _last(ship)
        ind["shipping"] = sv
        # 1-month change (weekly data → ~4 periods)
        ship_d = ship.dropna()
        ind["shipping_1m_chg"] = float(sv / ship_d.iloc[-5] - 1) \
            if len(ship_d) >= 5 else np.nan
        ind["shipping_regime"] = (
            "crisis" if sv > THRESH["shipping_crisis"] else
            "stress" if sv > THRESH["shipping_stress"] else
            "normal"
        )

    # ── EM sovereign spreads ──────────────────────────────────────────────────
    if not em_spreads.empty and "EMBI Global" in em_spreads.columns:
        embi     = em_spreads["EMBI Global"].dropna()
        embi_val = _last(embi)
        ind["embi"]              = embi_val
        ind["embi_prev"]         = _prev(embi, embi_val)
        ind["em_spreads_widening"] = bool(embi_val > ind["embi_prev"])
        ind["em_regime"]         = (
            "crisis" if embi_val > THRESH["em_spread_crisis"] else
            "stress" if embi_val > THRESH["em_spread_stress"] else
            "normal"
        )
    if not em_spreads.empty and "Africa Composite" in em_spreads.columns:
        ind["africa_spreads"] = _last(em_spreads["Africa Composite"].dropna())

    # ── FX reserves ──────────────────────────────────────────────────────────
    if not fx_res.empty and len(fx_res) >= 2:
        chgs = fx_res.iloc[-1] / fx_res.iloc[-2] - 1
        ind["fx_res_min_chg"]       = float(chgs.min())
        ind["fx_res_deteriorating"] = bool((chgs < -0.05).any())
        ind["fx_res_worst_country"] = str(chgs.idxmin())

    # ── Cross-asset stress composite ─────────────────────────────────────────
    ind["financial_stress_score"] = int(sum([
        ind.get("vix_regime")        in ("elevated", "high"),
        ind.get("dollar_regime")     in ("strong", "very_strong"),
        ind.get("em_regime")         in ("stress", "crisis"),
        ind.get("curve_regime")      == "inverted",
        ind.get("fin_cond_regime")   == "tightening",
    ]))
    ind["geo_stress_score"] = int(sum([
        ind.get("gpr_regime")      in ("elevated", "high"),
        ind.get("shipping_regime") in ("stress", "crisis"),
        ind.get("oil_regime")      == "surging",
    ]))
    ind["total_stress_score"] = ind["financial_stress_score"] + ind["geo_stress_score"]

    return ind
