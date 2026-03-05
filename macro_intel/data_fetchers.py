"""
data_fetchers.py — fetch market data (yfinance + US Treasury).
Everything unavailable from public APIs is mocked with realistic synthetic series.
All public functions are cached via @st.cache_data.
"""

from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import TICKERS


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rng(start: str, end: str, base: float, vol: float,
         drift: float = 0.0, seed: int = 0, freq: str = "M") -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq=freq)
    n = len(dates)
    r = np.random.normal(drift / (12 if freq == "M" else 52),
                         vol / np.sqrt(12 if freq == "M" else 52), n)
    vals = base * np.cumprod(1 + r)
    return pd.Series(vals, index=dates)


# ── Market prices (yfinance) ──────────────────────────────────────────────────
@st.cache_data(ttl=900)
def get_market_data(period: str = "1y") -> pd.DataFrame:
    """
    Download close prices for all tickers in config.TICKERS.
    Falls back to synthetic data if yfinance is unavailable.
    """
    syms = list(TICKERS.values())
    inv  = {v: k for k, v in TICKERS.items()}
    try:
        raw = yf.download(syms, period=period, auto_adjust=True,
                          progress=False, threads=True)
    except Exception:
        return _mock_market()

    if raw.empty:
        return _mock_market()

    # Extract Close (handles both flat and MultiIndex columns)
    try:
        close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
    except Exception:
        return _mock_market()

    close = close.rename(columns=inv)
    known = set(TICKERS.keys())
    close = close[[c for c in close.columns if c in known]]

    if len(close.columns) < 3:
        return _mock_market()
    return close


def _mock_market() -> pd.DataFrame:
    """Synthetic fallback so the app always runs."""
    dates = pd.date_range(end=datetime.now(), periods=252, freq="B")
    n = len(dates)

    def ts(base, vol, drift=0.0, seed=0):
        np.random.seed(seed)
        r = np.random.normal(drift / 252, vol / np.sqrt(252), n)
        return base * np.cumprod(1 + r)

    return pd.DataFrame({
        "vix":    ts(20,   0.50, 0,    1),
        "us10y":  ts(4.30, 0.25, 0,    2),
        "us5y":   ts(4.10, 0.25, 0,    3),
        "us3m":   ts(5.30, 0.05, 0,    4),
        "us30y":  ts(4.50, 0.20, 0,    5),
        "dxy":    ts(103,  0.05, 0,    6),
        "eurusd": ts(1.09, 0.05, 0,    7),
        "usdjpy": ts(148,  0.06, 0,    8),
        "usdbrl": ts(5.0,  0.10, 0.15, 9),
        "usdzar": ts(18.5, 0.08, 0.05, 10),
        "usdtry": ts(32,   0.12, 0.35, 11),
        "usdcnh": ts(7.2,  0.03, 0,    12),
        "brent":  ts(80,   0.25, 0,    13),
        "wti":    ts(76,   0.25, 0,    14),
        "copper": ts(4.20, 0.18, 0.05, 15),
        "gold":   ts(2050, 0.12, 0.10, 16),
        "natgas": ts(2.80, 0.40, 0,    17),
        "spx":    ts(5000, 0.15, 0.10, 18),
        "eem":    ts(42,   0.18, 0,    19),
        "hyg":    ts(79,   0.08, 0,    20),
        "emb":    ts(90,   0.10, 0,    21),
        "tip":    ts(110,  0.05, 0,    22),
    }, index=dates)


# ── US Treasury yield curve ───────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_yield_curve() -> tuple[pd.Series, pd.DataFrame]:
    """
    Fetch daily Treasury yield curve from Treasury Direct (free, no API key).
    Returns (latest_row: Series, history: DataFrame with DatetimeIndex).
    Falls back to synthetic data.
    """
    today = datetime.now()
    dfs = []
    for year in range(today.year - 1, today.year + 1):
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            f"interest-rates/daily-treasury-rates.csv/{year}/all"
            "?type=daily_treasury_yield_curve&field_tdr_date_value=all&page&_format=csv"
        )
        try:
            df = pd.read_csv(url, parse_dates=["Date"])
            dfs.append(df)
        except Exception:
            pass

    if dfs:
        full = (pd.concat(dfs)
                .drop_duplicates("Date")
                .sort_values("Date")
                .set_index("Date"))
        num_cols = [c for c in full.columns]
        hist = full[num_cols].apply(pd.to_numeric, errors="coerce")
        return hist.iloc[-1], hist

    return _mock_yield_curve()


def _mock_yield_curve() -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.date_range(end=datetime.now(), periods=500, freq="B")
    n = len(dates)
    np.random.seed(99)
    tenors = {
        "1 Mo": 5.30, "2 Mo": 5.28, "3 Mo": 5.25, "4 Mo": 5.20,
        "6 Mo": 5.10, "1 Yr": 4.90, "2 Yr": 4.50, "3 Yr": 4.35,
        "5 Yr": 4.25, "7 Yr": 4.25, "10 Yr": 4.30, "20 Yr": 4.55, "30 Yr": 4.50,
    }
    data = {}
    for tenor, base in tenors.items():
        np.random.seed(hash(tenor) % 1000)
        noise = np.cumsum(np.random.normal(0, 0.012, n))
        data[tenor] = (base + noise).clip(0.1, 9.0)
    hist = pd.DataFrame(data, index=dates)
    return hist.iloc[-1], hist


# ── Geopolitical Risk Index (mock) ────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_gpr() -> pd.Series:
    """
    Caldara & Iacoviello GPR index — synthetic with historically-timed shocks.
    Replace with FRED series 'GPRD' when API key is available.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    np.random.seed(55)
    dates = pd.date_range("2019-01-01", end, freq="M")
    n = len(dates)
    base = 105 + np.cumsum(np.random.normal(0.3, 8, n))
    s = pd.Series(base, index=dates)
    for date_str, delta in [("2020-03", 40), ("2022-02", 65), ("2022-03", 85),
                             ("2023-10", 55), ("2024-04", 35), ("2025-01", 25)]:
        y, m = int(date_str[:4]), int(date_str[5:])
        mask = (s.index.year == y) & (s.index.month == m)
        s[mask] += delta
    return s.clip(60, 350).rename("GPR")


# ── Shipping stress index (mock) ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_shipping_stress() -> pd.Series:
    """
    BDI/container-rate composite — synthetic with Red Sea disruption spike.
    Replace with real Baltic Dry Index or Freightos data if available.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    np.random.seed(77)
    dates = pd.date_range("2019-01-01", end, freq="W")
    n = len(dates)
    base = 1_500 + np.cumsum(np.random.normal(0, 90, n))
    s = pd.Series(base.clip(400, 12_000), index=dates)
    # COVID supply shock
    mask1 = (s.index >= "2021-03-01") & (s.index <= "2022-06-01")
    s[mask1] *= 2.5
    # Red Sea disruption
    mask2 = (s.index >= "2023-11-01") & (s.index <= "2024-07-01")
    s[mask2] *= 1.8
    return s.rename("Shipping Index")


# ── EM sovereign spreads (mock) ───────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_em_spreads() -> pd.DataFrame:
    """EMBI-style spreads (bps). Replace 'EMBI Global' with EMB ETF spread when available."""
    end = datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range("2019-01-01", end, freq="M")
    n = len(dates)
    np.random.seed(33)
    return pd.DataFrame({
        "EMBI Global":      (340 + np.cumsum(np.random.normal(0.3, 10, n))).clip(150, 900),
        "Africa Composite": (460 + np.cumsum(np.random.normal(0.8, 16, n))).clip(200, 1200),
        "LatAm Composite":  (310 + np.cumsum(np.random.normal(0.2, 12, n))).clip(150, 800),
        "Asia EM":          (195 + np.cumsum(np.random.normal(0.0,  8, n))).clip(80,  600),
    }, index=dates)


# ── African FX reserves (mock, $ bn) ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_fx_reserves() -> pd.DataFrame:
    dates = pd.date_range("2017-01-01", datetime.now().strftime("%Y-%m-%d"), freq="Q")
    n = len(dates)
    np.random.seed(22)
    return pd.DataFrame({
        "Nigeria":      (36 + np.cumsum(np.random.normal(-0.2, 1.6, n))).clip(4,  75),
        "South Africa": (52 + np.cumsum(np.random.normal( 0.0, 1.5, n))).clip(20, 85),
        "Kenya":        ( 8 + np.cumsum(np.random.normal(-0.1, 0.4, n))).clip(2,  15),
        "Egypt":        (27 + np.cumsum(np.random.normal(-0.4, 2.0, n))).clip(3,  55),
        "Ethiopia":     ( 2 + np.cumsum(np.random.normal(-0.1, 0.2, n))).clip(0.3, 8),
    }, index=dates)


# ── Critical minerals prices (indexed to 100 in Jan 2020) ────────────────────
@st.cache_data(ttl=3600)
def get_critical_minerals() -> pd.DataFrame:
    end = datetime.now().strftime("%Y-%m-%d")
    return pd.DataFrame({
        "Lithium":     _rng("2020-01-01", end, 100, 0.30, drift= 0.25, seed=11),
        "Cobalt":      _rng("2020-01-01", end, 100, 0.20, drift=-0.10, seed=22),
        "Copper":      _rng("2020-01-01", end, 100, 0.14, drift= 0.05, seed=33),
        "Nickel":      _rng("2020-01-01", end, 100, 0.18, drift= 0.00, seed=44),
        "Rare Earth":  _rng("2020-01-01", end, 100, 0.12, drift= 0.10, seed=55),
    }).dropna()


# ── Defense spending as % GDP ─────────────────────────────────────────────────
@st.cache_data(ttl=86_400)
def get_defense_spending() -> pd.DataFrame:
    years = list(range(2010, 2027))
    return pd.DataFrame({
        "US":          [4.7,4.6,4.3,3.8,3.5,3.3,3.1,3.1,3.2,3.4,3.5,3.5,3.5,3.6,3.4,3.5,3.7],
        "NATO Europe": [1.5,1.4,1.4,1.4,1.4,1.4,1.5,1.5,1.7,1.8,1.8,1.9,2.0,2.1,2.3,2.5,2.7],
        "China":       [1.9,2.0,2.0,2.0,2.1,2.1,2.0,2.0,1.9,1.9,1.7,1.7,1.6,1.6,1.6,1.6,1.7],
        "Russia":      [3.9,3.9,4.4,4.5,5.5,5.3,5.5,3.9,3.9,4.3,4.2,4.1,4.1,5.9,6.8,7.5,8.0],
        "Africa avg":  [1.5,1.5,1.6,1.6,1.5,1.5,1.5,1.6,1.6,1.6,1.5,1.5,1.6,1.7,1.8,2.0,2.1],
    }, index=pd.to_datetime([f"{y}-01-01" for y in years]))


# ── FDI inflows by region ($ bn) ─────────────────────────────────────────────
@st.cache_data(ttl=86_400)
def get_fdi_flows() -> pd.DataFrame:
    years = list(range(2015, 2026))
    n = len(years)
    np.random.seed(88)
    return pd.DataFrame({
        "Sub-Saharan Africa": (45  + np.cumsum(np.random.normal( 0.5, 3,  n))).clip(20),
        "South/SE Asia":      (140 + np.cumsum(np.random.normal( 2.0, 8,  n))).clip(60),
        "LatAm":              (135 + np.cumsum(np.random.normal( 1.0, 10, n))).clip(60),
        "MENA":               (55  + np.cumsum(np.random.normal( 1.0, 4,  n))).clip(20),
        "Emerging Europe":    (60  + np.cumsum(np.random.normal( 0.5, 5,  n))).clip(15),
    }, index=pd.to_datetime([f"{y}-01-01" for y in years]))
