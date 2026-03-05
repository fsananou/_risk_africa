"""
data_fetchers.py — Data ingestion layer.

Priority chain for each data type:
  1. FRED API          (requires free key in env / .streamlit/secrets.toml)
  2. World Bank API    (free, no key)
  3. IMF DataMapper    (free, no key)
  4. US Treasury Direct(free, no key)
  5. yfinance          (free, no key)
  6. Synthetic mock    (always works)

To swap any mock series for a real source: edit ONLY this file.
All functions return (data, source_label) tuples so the UI can flag real vs synthetic data.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from config import (
    FRED_SERIES, IMF_BASE, TICKERS, WB_AFRICA, WB_BASE,
    WB_EM_BROAD, WB_INDICATORS, get_fred_key,
)

_TIMEOUT = 12   # seconds for HTTP calls


# ══════════════════════════════════════════════════════════════════════════════
# Low-level API helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fred(series_id: str, start: str = "2018-01-01") -> pd.Series | None:
    key = get_fred_key()
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": key, "file_type": "json",
                    "observation_start": start, "sort_order": "asc", "limit": 10_000},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        obs = r.json().get("observations", [])
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"]  = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()
        return s if len(s) > 10 else None
    except Exception:
        return None


def _worldbank(indicator: str, countries: list[str],
               years: int = 15) -> pd.DataFrame | None:
    try:
        cstr = ";".join(countries)
        url  = f"{WB_BASE}/country/{cstr}/indicator/{indicator}"
        r    = requests.get(url, params={"format": "json", "per_page": 2000,
                                         "mrv": years}, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            return None
        records = [
            {"country": rec["country"]["value"],
             "year":    int(rec["date"]),
             "value":   rec["value"]}
            for rec in data[1] if rec["value"] is not None
        ]
        if not records:
            return None
        df = pd.DataFrame(records)
        out = df.pivot(index="year", columns="country", values="value").sort_index()
        out.index = pd.to_datetime(out.index.astype(str) + "-01-01")
        return out
    except Exception:
        return None


def _imf(indicator: str, countries: list[str]) -> pd.DataFrame | None:
    try:
        url = f"{IMF_BASE}/{indicator}/{'/'.join(countries)}"
        r   = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        values = r.json().get("values", {}).get(indicator, {})
        if not values:
            return None
        frames = {
            c: {int(y): float(v) for y, v in yd.items() if v is not None}
            for c, yd in values.items()
        }
        df = pd.DataFrame(frames).sort_index()
        df.index = pd.to_datetime(df.index.astype(str) + "-01-01")
        return df if not df.empty else None
    except Exception:
        return None


def _treasury_direct() -> pd.DataFrame | None:
    """US Treasury Direct yield curve CSV (free, no key)."""
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
    if not dfs:
        return None
    full = (pd.concat(dfs).drop_duplicates("Date")
              .sort_values("Date").set_index("Date"))
    return full.apply(pd.to_numeric, errors="coerce") if not full.empty else None


# ══════════════════════════════════════════════════════════════════════════════
# Mock generators  (synthetic but realistic — replace by editing this section)
# ══════════════════════════════════════════════════════════════════════════════

def _rng(start: str, end: str, base: float, vol: float,
         drift: float = 0.0, seed: int = 0, freq: str = "M") -> pd.Series:
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq=freq)
    n = len(dates)
    per = 12 if freq in ("M", "MS") else 52 if freq in ("W",) else 252
    r = np.random.normal(drift / per, vol / per**0.5, n)
    return pd.Series(base * np.cumprod(1 + r), index=dates)


def _mock_yields() -> pd.DataFrame:
    """Synthetic US Treasury yields if all real sources fail."""
    dates = pd.date_range(end=datetime.now(), periods=500, freq="B")
    n = len(dates)
    np.random.seed(99)
    tenors = {"3M": 5.28, "2Y": 4.50, "5Y": 4.25, "10Y": 4.30, "30Y": 4.50}
    data = {}
    for label, base in tenors.items():
        np.random.seed(hash(label) % 1000)
        data[label] = (base + np.cumsum(np.random.normal(0, 0.012, n))).clip(0.05, 9.0)
    return pd.DataFrame(data, index=dates)


def _mock_market() -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(), periods=252, freq="B")
    n = len(dates)
    def ts(base, vol, drift=0.0, seed=0):
        np.random.seed(seed)
        r = np.random.normal(drift / 252, vol / 252**0.5, n)
        return base * np.cumprod(1 + r)
    return pd.DataFrame({
        "vix":    ts(20,   0.50, 0,    1),   "us10y":  ts(4.30, 0.25, 0,    2),
        "us5y":   ts(4.10, 0.25, 0,    3),   "us3m":   ts(5.30, 0.05, 0,    4),
        "us30y":  ts(4.50, 0.20, 0,    5),   "dxy":    ts(103,  0.05, 0,    6),
        "eurusd": ts(1.09, 0.05, 0,    7),   "usdjpy": ts(148,  0.06, 0,    8),
        "usdbrl": ts(5.0,  0.10, 0.15, 9),   "usdzar": ts(18.5, 0.08, 0.05, 10),
        "usdtry": ts(32,   0.12, 0.35, 11),  "usdcnh": ts(7.2,  0.03, 0,    12),
        "usdinr": ts(83,   0.03, 0.05, 13),  "usdmxn": ts(17,   0.08, 0.03, 14),
        "brent":  ts(80,   0.25, 0,    15),  "wti":    ts(76,   0.25, 0,    16),
        "copper": ts(4.20, 0.18, 0.05, 17),  "gold":   ts(2050, 0.12, 0.10, 18),
        "natgas": ts(2.80, 0.40, 0,    19),  "silver": ts(24,   0.20, 0,    20),
        "wheat":  ts(5.50, 0.30, 0,    21),  "corn":   ts(4.80, 0.25, 0,    22),
        "spx":    ts(5000, 0.15, 0.10, 23),  "eem":    ts(42,   0.18, 0,    24),
        "hyg":    ts(79,   0.08, 0,    25),  "emb":    ts(90,   0.10, 0,    26),
        "tip":    ts(110,  0.05, 0,    27),  "lqd":    ts(115,  0.05, 0,    28),
        "iyr":    ts(85,   0.15, 0,    29),  "xlf":    ts(42,   0.18, 0.05, 30),
    }, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
# Public data-loading functions (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=900, show_spinner=False)
def get_market_data(period: str = "1y") -> tuple[pd.DataFrame, str]:
    """yfinance prices for all tickers. Fallback: synthetic."""
    syms = list(TICKERS.values())
    inv  = {v: k for k, v in TICKERS.items()}
    try:
        raw = yf.download(syms, period=period, auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            raise ValueError("empty")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close.rename(columns=inv)
        known = set(TICKERS.keys())
        close = close[[c for c in close.columns if c in known]]
        if len(close.columns) >= 5:
            return close, "yfinance"
    except Exception:
        pass
    return _mock_market(), "Mock"


@st.cache_data(ttl=3600, show_spinner=False)
def get_yields() -> tuple[pd.DataFrame, str]:
    """US Treasury yields: FRED → Treasury Direct → mock."""
    key = get_fred_key()
    if key:
        series = {
            "3M":  FRED_SERIES["us3m"],
            "2Y":  FRED_SERIES["us2y"],
            "5Y":  "DGS5",
            "10Y": FRED_SERIES["us10y"],
            "30Y": FRED_SERIES["us30y"],
        }
        frames = {label: _fred(sid) for label, sid in series.items()}
        valid = {k: v for k, v in frames.items() if v is not None}
        if len(valid) >= 3:
            return pd.DataFrame(valid).dropna(how="all"), "FRED"

    td = _treasury_direct()
    if td is not None:
        rename = {"3 Mo": "3M", "2 Yr": "2Y", "5 Yr": "5Y", "10 Yr": "10Y", "30 Yr": "30Y"}
        td2 = td[[c for c in rename if c in td.columns]].rename(columns=rename)
        if not td2.empty:
            return td2, "Treasury Direct"

    return _mock_yields(), "Mock"


@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_macro() -> tuple[dict[str, pd.Series], str]:
    """
    FRED macro series: breakevens, HY/IG spreads, NFCI, term premium.
    Returns (dict_of_series, source_label).
    """
    key = get_fred_key()
    result: dict[str, pd.Series] = {}
    source = "Mock"

    if key:
        mapping = {
            "breakeven5y":   FRED_SERIES["breakeven5y"],
            "breakeven5y5y": FRED_SERIES["breakeven5y5y"],
            "hy_spread":     FRED_SERIES["hy_spread"],
            "ig_spread":     FRED_SERIES["ig_spread"],
            "nfci":          FRED_SERIES["nfci"],
            "term_premium":  FRED_SERIES["term_premium"],
        }
        for name, sid in mapping.items():
            s = _fred(sid)
            if s is not None:
                result[name] = s
        if len(result) >= 3:
            source = "FRED"

    # Fill missing with synthetic
    end = datetime.now().strftime("%Y-%m-%d")
    if "breakeven5y" not in result:
        np.random.seed(11)
        dates = pd.date_range("2018-01-01", end, freq="B")
        result["breakeven5y"] = pd.Series(
            (2.15 + np.cumsum(np.random.normal(0, 0.008, len(dates)))).clip(0.5, 4.5),
            index=dates)
    if "breakeven5y5y" not in result:
        np.random.seed(12)
        dates = pd.date_range("2018-01-01", end, freq="B")
        result["breakeven5y5y"] = pd.Series(
            (2.30 + np.cumsum(np.random.normal(0, 0.007, len(dates)))).clip(0.8, 4.8),
            index=dates)
    if "hy_spread" not in result:
        np.random.seed(13)
        dates = pd.date_range("2018-01-01", end, freq="B")
        base = 380 + np.cumsum(np.random.normal(0, 5, len(dates)))
        # Add Covid spike + 2022 tightening
        s = pd.Series(base, index=dates)
        s["2020-03-01":"2020-06-30"] += 500
        s["2022-06-01":"2022-12-31"] += 150
        result["hy_spread"] = s.clip(200, 1500)
    if "ig_spread" not in result:
        np.random.seed(14)
        dates = pd.date_range("2018-01-01", end, freq="B")
        result["ig_spread"] = pd.Series(
            (110 + np.cumsum(np.random.normal(0, 2, len(dates)))).clip(50, 400), index=dates)
    if "nfci" not in result:
        np.random.seed(15)
        dates = pd.date_range("2018-01-01", end, freq="W")
        s = pd.Series(np.cumsum(np.random.normal(0, 0.05, len(dates))), index=dates)
        s["2020-03-01":"2020-06-01"] += 2.0
        s["2022-06-01":"2023-01-01"] += 0.6
        result["nfci"] = s
    if "term_premium" not in result:
        np.random.seed(16)
        dates = pd.date_range("2018-01-01", end, freq="B")
        result["term_premium"] = pd.Series(
            (-0.2 + np.cumsum(np.random.normal(0, 0.005, len(dates)))).clip(-1.5, 2.0),
            index=dates)

    return result, source


@st.cache_data(ttl=86_400, show_spinner=False)
def get_worldbank_reserves() -> tuple[pd.DataFrame, str]:
    """African FX reserves (USD). World Bank → mock."""
    df = _worldbank(WB_INDICATORS["fx_reserves"], WB_AFRICA, years=15)
    if df is not None and not df.empty:
        return df / 1e9, "World Bank"  # convert to USD bn

    # Mock fallback
    end = datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range("2010-01-01", end, freq="A-DEC")
    n = len(dates)
    np.random.seed(22)
    data = {
        "Nigeria":      (36 + np.cumsum(np.random.normal(-0.3, 2.0, n))).clip(4,  80),
        "South Africa": (52 + np.cumsum(np.random.normal( 0.1, 1.8, n))).clip(20, 90),
        "Kenya":        ( 8 + np.cumsum(np.random.normal(-0.1, 0.5, n))).clip(2,  15),
        "Egypt":        (27 + np.cumsum(np.random.normal(-0.5, 2.5, n))).clip(3,  60),
        "Ethiopia":     ( 2 + np.cumsum(np.random.normal(-0.1, 0.3, n))).clip(0.3, 8),
        "Ghana":        ( 7 + np.cumsum(np.random.normal(-0.1, 0.6, n))).clip(1,  15),
        "Angola":       (14 + np.cumsum(np.random.normal(-0.2, 1.0, n))).clip(3,  30),
    }
    return pd.DataFrame(data, index=dates), "Mock"


@st.cache_data(ttl=86_400, show_spinner=False)
def get_worldbank_fdi() -> tuple[pd.DataFrame, str]:
    """FDI inflows by region. World Bank → mock."""
    # Try WB for a broad EM set (one country at a time is more reliable)
    regions = {
        "Sub-Saharan Africa": ["NGA", "ZAF", "KEN", "GHA", "ETH", "TZA", "ZMB"],
        "South/SE Asia":      ["IND", "IDN", "VNM", "THA", "MYS", "PHL"],
        "LatAm":              ["BRA", "MEX", "COL", "CHL", "ARG", "PER"],
        "MENA":               ["EGY", "MAR", "TUN", "SAU", "ARE"],
        "Emerging Europe":    ["POL", "CZE", "HUN", "ROU", "UKR"],
    }
    result = {}
    for region, countries in regions.items():
        df = _worldbank(WB_INDICATORS["fdi_inflows"], countries, years=12)
        if df is not None and not df.empty:
            result[region] = df.sum(axis=1) / 1e9  # USD bn
    if len(result) >= 3:
        return pd.DataFrame(result).dropna(how="all"), "World Bank"

    # Mock
    years = list(range(2012, 2026))
    n = len(years)
    np.random.seed(88)
    data = {
        "Sub-Saharan Africa": (45  + np.cumsum(np.random.normal( 0.5, 3,  n))).clip(15),
        "South/SE Asia":      (140 + np.cumsum(np.random.normal( 2.0, 8,  n))).clip(60),
        "LatAm":              (135 + np.cumsum(np.random.normal( 1.0, 10, n))).clip(60),
        "MENA":               (55  + np.cumsum(np.random.normal( 1.0, 4,  n))).clip(20),
        "Emerging Europe":    (60  + np.cumsum(np.random.normal( 0.5, 5,  n))).clip(15),
    }
    idx = pd.to_datetime([f"{y}-01-01" for y in years])
    return pd.DataFrame(data, index=idx), "Mock"


@st.cache_data(ttl=86_400, show_spinner=False)
def get_imf_macro() -> tuple[dict[str, pd.DataFrame], str]:
    """IMF macro indicators. IMF DataMapper → mock."""
    result: dict[str, pd.DataFrame] = {}
    countries = ["US", "CN", "DE", "JP", "IN", "BR", "ZA", "NG", "EG", "TR"]

    for label, ind_id in [("inflation", "PCPIPCH"), ("gov_debt", "GGXWDG_NGDP"),
                           ("ca_gdp", "BCA_NGDPD"), ("gdp_usd", "NGDPD")]:
        df = _imf(ind_id, countries)
        if df is not None and not df.empty:
            result[label] = df

    if len(result) >= 2:
        return result, "IMF DataMapper"

    # Mock fallback
    years = list(range(2015, 2026))
    n = len(years)
    idx = pd.to_datetime([f"{y}-01-01" for y in years])
    np.random.seed(55)
    result["inflation"] = pd.DataFrame({
        "US":  (2.3 + np.cumsum(np.random.normal(0.1, 0.3, n))).clip(0, 12),
        "CN":  (2.1 + np.cumsum(np.random.normal(0.0, 0.2, n))).clip(0, 8),
        "NG":  (12  + np.cumsum(np.random.normal(0.5, 1.5, n))).clip(5, 35),
        "EG":  (10  + np.cumsum(np.random.normal(0.3, 1.2, n))).clip(5, 35),
        "TR":  (11  + np.cumsum(np.random.normal(1.0, 3.0, n))).clip(5, 90),
        "ZA":  (5   + np.cumsum(np.random.normal(0.1, 0.5, n))).clip(2, 12),
    }, index=idx)
    result["gov_debt"] = pd.DataFrame({
        "US": (100 + np.cumsum(np.random.normal(1.5, 1,   n))).clip(80, 140),
        "CN": (47  + np.cumsum(np.random.normal(1.0, 0.8, n))).clip(30, 80),
        "JP": (230 + np.cumsum(np.random.normal(1.0, 0.5, n))).clip(200, 260),
        "NG": (28  + np.cumsum(np.random.normal(0.5, 1.0, n))).clip(10, 60),
        "EG": (85  + np.cumsum(np.random.normal(0.5, 2.0, n))).clip(40, 110),
    }, index=idx)
    return result, "Mock"


@st.cache_data(ttl=3600, show_spinner=False)
def get_gpr() -> tuple[pd.Series, str]:
    """
    Geopolitical Risk Index (Caldara & Iacoviello 2022).
    Real data: download gpr_daily.csv from https://www.matteoiacoviello.com/gpr.htm
    and place in data/ folder — this fetcher will use it automatically.
    Otherwise: synthetic with historically-timed shocks.
    """
    import pathlib
    csv_path = pathlib.Path(__file__).parent / "data" / "gpr_daily.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            s  = df.set_index("date")["GPR"].dropna()
            return s.resample("M").mean(), "GPR (Caldara & Iacoviello)"
        except Exception:
            pass

    # Synthetic
    end = datetime.now().strftime("%Y-%m-%d")
    np.random.seed(55)
    dates = pd.date_range("2018-01-01", end, freq="M")
    n = len(dates)
    base = 105 + np.cumsum(np.random.normal(0.3, 8, n))
    s = pd.Series(base, index=dates)
    for date_str, delta in [
        ("2020-03", 45), ("2022-02", 65), ("2022-03", 88),
        ("2023-10", 58), ("2024-04", 38), ("2025-01", 28),
    ]:
        y, m = int(date_str[:4]), int(date_str[5:])
        s[(s.index.year == y) & (s.index.month == m)] += delta
    return s.clip(55, 360), "Mock (place gpr_daily.csv in data/ to use real)"


@st.cache_data(ttl=3600, show_spinner=False)
def get_shipping() -> tuple[pd.Series, str]:
    """
    Shipping stress index (BDI + container rate composite).
    Real data: replace with Freightos Baltic Daily Index API or CSV.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    np.random.seed(77)
    dates = pd.date_range("2018-01-01", end, freq="W")
    n = len(dates)
    base = 1_400 + np.cumsum(np.random.normal(0, 90, n))
    s = pd.Series(base.clip(400, 15_000), index=dates)
    s["2021-03-01":"2022-06-01"] *= 2.8   # COVID/Suez disruption
    s["2023-11-01":"2024-07-01"] *= 1.9   # Red Sea disruption
    return s, "Mock (replace with Freightos/BDI feed)"


@st.cache_data(ttl=3600, show_spinner=False)
def get_em_spreads() -> tuple[pd.DataFrame, str]:
    """
    EM sovereign spreads (EMBI-style, bps).
    Proxy: EMB ETF Z-spread vs LQD; full EMBI data requires Bloomberg/JPMorgan subscription.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range("2018-01-01", end, freq="M")
    n = len(dates)
    np.random.seed(33)
    base = {
        "EMBI Global":      (340, 0.3, 10),
        "Africa Composite": (470, 0.8, 16),
        "LatAm Composite":  (315, 0.2, 12),
        "Asia EM":          (195, 0.0,  8),
        "Frontier Markets": (600, 1.2, 22),
    }
    data = {}
    for name, (start_val, drift, vol) in base.items():
        np.random.seed(hash(name) % 1000)
        s = pd.Series(
            (start_val + np.cumsum(np.random.normal(drift, vol, n))).clip(100, 1500),
            index=dates)
        # Covid shock
        s["2020-03-01":"2020-09-30"] += 120
        data[name] = s
    return pd.DataFrame(data), "Mock (replace with EMBI Bloomberg data)"


@st.cache_data(ttl=3600, show_spinner=False)
def get_critical_minerals() -> tuple[pd.DataFrame, str]:
    """Critical minerals price index (Jan 2020 = 100). yfinance + synthetic."""
    end = datetime.now().strftime("%Y-%m-%d")
    dates_m = pd.date_range("2019-01-01", end, freq="M")

    # Copper from yfinance
    copper_real = None
    try:
        raw = yf.download("HG=F", start="2019-01-01", auto_adjust=True, progress=False)
        c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        copper_real = (c.resample("M").last().squeeze() if not c.empty else None)
    except Exception:
        pass

    def mineral(base, vol, drift, seed):
        np.random.seed(seed)
        n = len(dates_m)
        r = np.random.normal(drift / 12, vol / 12**0.5, n)
        return base * np.cumprod(1 + r)

    data: dict[str, pd.Series] = {
        "Lithium":    pd.Series(mineral(100, 0.30,  0.22, 11), index=dates_m),
        "Cobalt":     pd.Series(mineral(100, 0.22, -0.08, 22), index=dates_m),
        "Nickel":     pd.Series(mineral(100, 0.20,  0.02, 44), index=dates_m),
        "Rare Earth": pd.Series(mineral(100, 0.14,  0.10, 55), index=dates_m),
    }
    if copper_real is not None:
        # Normalise to 100 at first common date
        base_date = copper_real.index[0]
        data["Copper"] = (copper_real / copper_real.iloc[0] * 100).rename("Copper")
    else:
        data["Copper"] = pd.Series(mineral(100, 0.14, 0.05, 33), index=dates_m)

    src = "yfinance (copper) + Mock (others)"
    return pd.DataFrame(data).dropna(), src


@st.cache_data(ttl=86_400, show_spinner=False)
def get_defense_spending() -> tuple[pd.DataFrame, str]:
    """Defense spending (% GDP). SIPRI/IMF data — synthetic here."""
    years = list(range(2010, 2027))
    return pd.DataFrame({
        "US":          [4.7,4.6,4.3,3.8,3.5,3.3,3.1,3.1,3.2,3.4,3.5,3.5,3.5,3.6,3.4,3.5,3.7],
        "NATO Europe": [1.5,1.4,1.4,1.4,1.4,1.4,1.5,1.5,1.7,1.8,1.8,1.9,2.0,2.1,2.3,2.5,2.8],
        "China":       [1.9,2.0,2.0,2.0,2.1,2.1,2.0,2.0,1.9,1.9,1.7,1.7,1.6,1.6,1.6,1.6,1.7],
        "Russia":      [3.9,3.9,4.4,4.5,5.5,5.3,5.5,3.9,3.9,4.3,4.2,4.1,4.1,5.9,6.8,7.5,8.2],
        "Africa avg":  [1.5,1.5,1.6,1.6,1.5,1.5,1.5,1.6,1.6,1.6,1.5,1.5,1.6,1.7,1.8,2.0,2.1],
        "Gulf (avg)":  [4.5,4.8,5.0,5.2,5.5,5.8,5.6,5.0,5.2,5.4,5.0,5.2,5.4,5.6,5.8,6.2,6.5],
    }, index=pd.to_datetime([f"{y}-01-01" for y in years])), "Mock (SIPRI)"


@st.cache_data(ttl=86_400, show_spinner=False)
def get_sanctions_intensity() -> tuple[pd.Series, str]:
    """Sanctions intensity index (proxy, mock). Real: OFAC/EU sanctions tracker."""
    end = datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range("2018-01-01", end, freq="M")
    n = len(dates)
    np.random.seed(66)
    base = 100 + np.cumsum(np.random.normal(0.5, 3, n))
    s = pd.Series(base, index=dates)
    s["2022-03-01":"2022-12-31"] += 60   # Russia sanctions
    s["2023-10-01":"2024-03-31"] += 25   # Iran/Hamas-related
    return s.clip(80, 400), "Mock (replace with OFAC tracker)"


@st.cache_data(ttl=86_400, show_spinner=False)
def get_cyber_risk() -> tuple[pd.Series, str]:
    """Cyber risk index (mock). Real: ENISA, Rapid7, or Mandiant threat indices."""
    end = datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range("2018-01-01", end, freq="M")
    n = len(dates)
    np.random.seed(44)
    base = 80 + np.cumsum(np.random.normal(0.8, 4, n))
    s = pd.Series(base, index=dates)
    s["2020-12-01":"2021-03-31"] += 45   # SolarWinds
    s["2021-05-01":"2021-07-31"] += 35   # Colonial Pipeline
    s["2022-02-01":"2022-05-31"] += 50   # Ukraine cyber war
    return s.clip(60, 350), "Mock (replace with ENISA/Mandiant)"


def get_all_sources(mkt_s, yields_s, fred_s, wb_res_s, wb_fdi_s,
                    imf_s, gpr_s, ship_s, em_s, min_s) -> dict[str, str]:
    """Collect source labels for display in UI."""
    return {
        "Market prices":   mkt_s,
        "Yields":          yields_s,
        "Macro (FRED)":    fred_s,
        "FX Reserves":     wb_res_s,
        "FDI flows":       wb_fdi_s,
        "Macro (IMF)":     imf_s,
        "GPR":             gpr_s,
        "Shipping":        ship_s,
        "EM Spreads":      em_s,
        "Minerals":        min_s,
    }
