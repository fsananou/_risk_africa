"""
data_fetchers.py — Real API integrations + explicit PLACEHOLDER functions.

Rules:
  - Every public function returns (data, source_label) tuple.
  - Real data: returns (DataFrame/Series, "SourceName").
  - Failed real data: returns (None, "SourceName — FAILED").
  - No live API: returns (None, "PLACEHOLDER — <reason>").
  - NO synthetic or mock data anywhere in this file.

Real sources used:
  FRED (free API key)       — yields, spreads, NFCI, commodities
  Yahoo Finance (no key)    — market prices, ETFs, FX, futures
  World Bank (no key)       — FX reserves, FDI, external debt
  IMF DataMapper (no key)   — inflation, debt, current account
  US Treasury Direct (none) — yield curve fallback
  EIA (free API key)        — US oil inventories, US gas storage
  EU AGSI+ (no key)         — EU gas storage
  FAO (no key)              — Food Price Index
  OECD SDMX (no key)        — Composite Leading Indicators
"""

from __future__ import annotations

import datetime
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    AGSI_EU_URL, EIA_BASE, FAO_FPI_URL, FRED_SERIES,
    IMF_BASE, IMF_INDICATORS, IMF_SDMX_BASE, OECD_CLI_URL,
    TICKERS, WB_AFRICA, WB_BASE,
    WB_CMO_URL, WB_EM_BROAD, WB_INDICATORS, get_eia_key, get_fred_key,
)

_TIMEOUT = 15  # HTTP timeout in seconds


# ══════════════════════════════════════════════════════════════════════════════
# Low-level API helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fred(series_id: str, key: str, start: str = "2018-01-01") -> pd.Series | None:
    """Fetch a single FRED series. Returns None if key missing or request fails."""
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": series_id, "api_key": key,
                    "file_type": "json", "observation_start": start,
                    "sort_order": "asc", "limit": 10_000},
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


def _worldbank(indicator: str, countries: list[str], years: int = 15) -> pd.DataFrame | None:
    try:
        cstr = ";".join(countries)
        r = requests.get(
            f"{WB_BASE}/country/{cstr}/indicator/{indicator}",
            params={"format": "json", "per_page": 2000, "mrv": years},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            return None
        records = [
            {"iso": rec["countryiso3code"], "year": int(rec["date"]), "value": rec["value"]}
            for rec in data[1] if rec["value"] is not None
        ]
        if not records:
            return None
        df = pd.DataFrame(records)
        out = df.pivot(index="year", columns="iso", values="value").sort_index()
        out.index = pd.to_datetime(out.index.astype(str) + "-01-01")
        return out
    except Exception:
        return None


def _imf_sdmx(
    indicators: list[str],
    countries: list[str],
    start: int = 2018,
    end: int = 2030,
) -> dict[str, pd.DataFrame] | None:
    """
    Fetch multiple WEO indicators × countries in ONE SDMX 3.0 call.
    Returns {indicator_id: DataFrame(index=year int, columns=ISO3)}.
    Key format: COUNTRY.INDICATOR.FREQUENCY  (e.g. NGA+KEN.NGDP_RPCH.A)
    """
    country_key   = "+".join(countries)
    indicator_key = "+".join(indicators)
    url = (
        f"{IMF_SDMX_BASE}/data/dataflow/IMF.RES/WEO/9.0.0"
        f"/{country_key}.{indicator_key}.A"
        f"?format=jsondata"
    )
    try:
        r = requests.get(url, timeout=_TIMEOUT + 20,
                         headers={"Accept": "application/json"})
        r.raise_for_status()
        payload = r.json()

        structure = payload["data"]["structures"][0]   # list, not dict
        datasets  = payload["data"]["dataSets"]
        if not datasets:
            return None

        # ── Decode dimension metadata ─────────────────────────────────────────
        series_dims = structure["dimensions"]["series"]
        obs_dims    = structure["dimensions"]["observation"]

        dim_order = [d["id"] for d in series_dims]
        # Build value-list per dimension (time uses "value" key, series use "id")
        dim_vals = {
            d["id"]: [v.get("id", v.get("value", "")) for v in d["values"]]
            for d in series_dims
        }
        # Time dim values use "value" key
        time_vals = [
            v.get("value", v.get("id", ""))
            for v in obs_dims[0]["values"]
        ]

        # Dimension names in WEO: COUNTRY, INDICATOR, FREQUENCY
        area_pos = next((i for i, d in enumerate(dim_order)
                         if d in ("COUNTRY", "REF_AREA") or "AREA" in d), None)
        ind_pos  = next((i for i, d in enumerate(dim_order) if "INDICATOR" in d), None)
        if area_pos is None or ind_pos is None:
            return None

        area_dim = dim_order[area_pos]
        ind_dim  = dim_order[ind_pos]

        # ── Parse series ──────────────────────────────────────────────────────
        raw: dict[str, dict[str, dict[str, float]]] = {ind: {} for ind in indicators}

        for key_str, series_obj in datasets[0].get("series", {}).items():
            parts      = key_str.split(":")
            country_id = dim_vals[area_dim][int(parts[area_pos])]
            ind_id     = dim_vals[ind_dim][int(parts[ind_pos])]
            if ind_id not in raw:
                continue
            if country_id not in raw[ind_id]:
                raw[ind_id][country_id] = {}
            for t_str, obs in series_obj.get("observations", {}).items():
                val = obs[0] if obs else None
                if val is not None:
                    year = time_vals[int(t_str)]
                    try:
                        yr_int = int(year)
                    except (ValueError, TypeError):
                        continue
                    if start <= yr_int <= end:
                        raw[ind_id][country_id][year] = float(val)

        # ── Build DataFrames ──────────────────────────────────────────────────
        result: dict[str, pd.DataFrame] = {}
        for ind_id, country_data in raw.items():
            if not country_data:
                continue
            records = [
                {"country": c, "year": int(yr), "value": v}
                for c, yv in country_data.items()
                for yr, v in yv.items()
            ]
            if not records:
                continue
            df = pd.DataFrame(records).pivot(
                index="year", columns="country", values="value"
            ).sort_index()
            df.index = pd.to_datetime(df.index.astype(str) + "-01-01")
            result[ind_id] = df

        return result if result else None

    except Exception:
        return None


def _imf_datamapper(indicator: str, countries: list[str]) -> pd.DataFrame | None:
    """Fallback: old DataMapper API for a single indicator."""
    try:
        url = f"{IMF_BASE}/{indicator}/{'/'.join(countries)}"
        r = requests.get(url, timeout=_TIMEOUT)
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
    """US Treasury Direct yield curve CSV — free, no key."""
    today = datetime.datetime.now()
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
    full = pd.concat(dfs).drop_duplicates("Date").sort_values("Date").set_index("Date")
    return full.apply(pd.to_numeric, errors="coerce") if not full.empty else None


# ══════════════════════════════════════════════════════════════════════════════
# Market data (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════════════

def get_market_data(period: str = "1y") -> tuple[dict[str, pd.Series] | None, str]:
    """
    Real-time market prices via yfinance.
    Returns dict of {ticker_key: pd.Series}.
    """
    syms = list(TICKERS.values())
    inv  = {v: k for k, v in TICKERS.items()}
    try:
        raw = yf.download(syms, period=period, auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            return None, "yfinance — FAILED (empty)"
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        close = close.rename(columns=inv)
        close = close[[c for c in close.columns if c in TICKERS]]
        if len(close.columns) < 5:
            return None, "yfinance — FAILED (insufficient tickers)"
        result = {col: close[col].dropna() for col in close.columns}
        return result, "Yahoo Finance"
    except Exception as e:
        return None, f"yfinance — FAILED ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
# Yield curve
# ══════════════════════════════════════════════════════════════════════════════

def get_yields(key: str = "") -> tuple[pd.DataFrame | None, str]:
    """US Treasury yields: FRED → Treasury Direct → None."""
    if not key:
        key = get_fred_key() or ""
    if key:
        series = {
            "3M": FRED_SERIES["us3m"], "2Y": FRED_SERIES["us2y"],
            "5Y": FRED_SERIES["us5y"], "10Y": FRED_SERIES["us10y"],
            "30Y": FRED_SERIES["us30y"],
        }
        frames = {label: _fred(sid, key) for label, sid in series.items()}
        valid  = {k: v for k, v in frames.items() if v is not None}
        if len(valid) >= 3:
            return pd.DataFrame(valid).dropna(how="all"), "FRED"

    td = _treasury_direct()
    if td is not None:
        rename = {"3 Mo": "3M", "2 Yr": "2Y", "5 Yr": "5Y",
                  "10 Yr": "10Y", "30 Yr": "30Y"}
        td2 = td[[c for c in rename if c in td.columns]].rename(columns=rename)
        if not td2.empty:
            return td2, "US Treasury Direct"

    return None, "FRED + Treasury Direct — FAILED"


# ══════════════════════════════════════════════════════════════════════════════
# FRED macro series
# ══════════════════════════════════════════════════════════════════════════════

def get_fred_macro(key: str = "") -> tuple[dict[str, pd.Series] | None, str]:
    """
    FRED: breakevens, HY/IG spreads, NFCI, term premium.
    Returns (dict_of_series, source) or (None, ...) if no key.
    """
    if not key:
        key = get_fred_key() or ""
    if not key:
        return None, "FRED — NO API KEY (get free key at fred.stlouisfed.org)"

    mapping = {
        "breakeven5y":   FRED_SERIES["breakeven5y"],
        "breakeven5y5y": FRED_SERIES["breakeven5y5y"],
        "hy_spread":     FRED_SERIES["hy_spread"],
        "ig_spread":     FRED_SERIES["ig_spread"],
        "nfci":          FRED_SERIES["nfci"],
        "term_premium":  FRED_SERIES["term_premium"],
        "indpro_us":     FRED_SERIES["indpro_us"],
        "brent_fred":    FRED_SERIES["brent"],
        "wheat_fred":    FRED_SERIES["wheat"],
        "corn_fred":     FRED_SERIES["corn"],
        "rice_fred":     FRED_SERIES["rice"],
        "baltic_dry":    FRED_SERIES["baltic_dry"],
    }
    result = {}
    for name, sid in mapping.items():
        s = _fred(sid, key)
        if s is not None:
            result[name] = s

    if not result:
        return None, "FRED — FAILED (all series failed)"

    return result, "FRED"


# ══════════════════════════════════════════════════════════════════════════════
# World Bank
# ══════════════════════════════════════════════════════════════════════════════

def get_worldbank_reserves() -> tuple[pd.DataFrame | None, str]:
    """African FX reserves (USD) — World Bank."""
    df = _worldbank(WB_INDICATORS["fx_reserves"], WB_AFRICA, years=15)
    if df is not None and not df.empty:
        return df, "World Bank"
    return None, "World Bank — FAILED"


def get_worldbank_fdi() -> tuple[pd.DataFrame | None, str]:
    """FDI net inflows by region — World Bank."""
    regions = {
        "Sub-Saharan Africa": ["NGA","ZAF","KEN","GHA","ETH","TZA","ZMB"],
        "South/SE Asia":      ["IND","IDN","VNM","THA","MYS"],
        "LatAm":              ["BRA","MEX","COL","CHL","ARG"],
        "MENA":               ["EGY","MAR","TUN","SAU","ARE"],
    }
    result = {}
    for region, countries in regions.items():
        df = _worldbank(WB_INDICATORS["fdi_inflows"], countries, years=12)
        if df is not None and not df.empty:
            result[region] = df.sum(axis=1) / 1e9
    if result:
        return pd.DataFrame(result).dropna(how="all"), "World Bank"
    return None, "World Bank FDI — FAILED"


def get_worldbank_ext_debt() -> tuple[pd.DataFrame | None, str]:
    """External debt stocks — World Bank."""
    df = _worldbank(WB_INDICATORS["ext_debt"], WB_EM_BROAD, years=12)
    if df is not None and not df.empty:
        return df, "World Bank"
    return None, "World Bank External Debt — FAILED"


def get_worldbank_gni_per_capita() -> tuple[pd.DataFrame | None, str]:
    """GNI per capita (current USD) for all Africa countries — World Bank."""
    df = _worldbank(WB_INDICATORS["gni_pc"], WB_AFRICA, years=8)
    if df is not None and not df.empty:
        return df, "World Bank"
    return None, "World Bank GNI/capita — FAILED"


def get_worldbank_debt_service() -> tuple[pd.DataFrame | None, str]:
    """Govt interest payments as % of fiscal revenue for all Africa countries — World Bank."""
    df = _worldbank(WB_INDICATORS["debt_service"], WB_AFRICA, years=8)
    if df is not None and not df.empty:
        return df, "World Bank"
    return None, "World Bank Debt Service — FAILED"


# ══════════════════════════════════════════════════════════════════════════════
# IMF
# ══════════════════════════════════════════════════════════════════════════════

_IMF_AFRICA_ISO3 = [
    # Sub-Saharan Africa
    "AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD","COM",
    "COG","COD","CIV","DJI","GNQ","ERI","SWZ","ETH","GAB","GMB",
    "GHA","GIN","GNB","KEN","LSO","LBR","MDG","MWI","MLI","MRT",
    "MUS","MOZ","NAM","NER","NGA","RWA","STP","SEN","SLE","SOM",
    "ZAF","SSD","SDN","TZA","TGO","UGA","ZMB","ZWE",
    # North Africa
    "DZA","EGY","MAR","TUN","LBY",
]


def get_imf_macro() -> tuple[dict[str, pd.DataFrame] | None, str]:
    """
    IMF WEO: GDP growth, inflation, govt debt, current account — all Africa.
    Primary: SDMX 3.0 API (single batch call, forecasts included).
    Fallback: DataMapper API (one call per indicator).
    """
    indicators = list(IMF_INDICATORS.values())   # ["NGDP_RPCH", "PCPIPCH", ...]
    id_to_label = {v: k for k, v in IMF_INDICATORS.items()}

    # ── 1. SDMX 3.0 batch ────────────────────────────────────────────────────
    raw = _imf_sdmx(indicators, _IMF_AFRICA_ISO3)
    if raw:
        result = {id_to_label.get(k, k): v for k, v in raw.items()}
        return result, "IMF WEO (SDMX 3.0)"

    # ── 2. DataMapper fallback ────────────────────────────────────────────────
    result = {}
    for label, ind_id in IMF_INDICATORS.items():
        df = _imf_datamapper(ind_id, _IMF_AFRICA_ISO3)
        if df is not None:
            result[label] = df
    if result:
        return result, "IMF DataMapper (fallback)"
    return None, "IMF — FAILED (SDMX 3.0 + DataMapper both failed)"


# ══════════════════════════════════════════════════════════════════════════════
# EIA (US Energy Information Administration)
# ══════════════════════════════════════════════════════════════════════════════

def get_eia_oil_inventories(key: str = "") -> tuple[pd.Series | None, str]:
    """
    US crude oil weekly stocks (thousand barrels) — EIA API v2.
    Free API key required: eia.gov/opendata
    """
    if not key:
        key = get_eia_key() or ""
    if not key:
        return None, "EIA — NO API KEY (get free key at eia.gov/opendata)"
    try:
        r = requests.get(
            f"{EIA_BASE}/petroleum/stoc/wstk/data/",
            params={
                "api_key":            key,
                "frequency":          "weekly",
                "data[0]":            "value",
                "facets[product][]":  "EPC0",   # Crude oil
                "facets[duoarea][]":  "NUS",    # US total
                "sort[0][column]":    "period",
                "sort[0][direction]": "asc",
                "offset":             0,
                "length":             260,       # ~5 years weekly
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        if not data:
            return None, "EIA — FAILED (empty response)"
        df = pd.DataFrame(data)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("period")["value"].dropna().sort_index()
        return s if len(s) > 10 else None, "EIA"
    except Exception as e:
        return None, f"EIA — FAILED ({type(e).__name__})"


def get_eia_gas_storage_us(key: str = "") -> tuple[pd.Series | None, str]:
    """
    US natural gas weekly storage (billion cubic feet) — EIA API v2.
    Free API key required.
    """
    if not key:
        key = get_eia_key() or ""
    if not key:
        return None, "EIA — NO API KEY"
    try:
        r = requests.get(
            f"{EIA_BASE}/natural-gas/stor/sum/data/",
            params={
                "api_key":            key,
                "frequency":          "weekly",
                "data[0]":            "value",
                "facets[process][]":  "SAB",    # Total lower 48
                "facets[duoarea][]":  "NUS",
                "sort[0][column]":    "period",
                "sort[0][direction]": "asc",
                "offset":             0,
                "length":             260,
            },
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        if not data:
            return None, "EIA Gas — FAILED (empty)"
        df = pd.DataFrame(data)
        df["period"] = pd.to_datetime(df["period"])
        df["value"]  = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("period")["value"].dropna().sort_index()
        return s if len(s) > 10 else None, "EIA"
    except Exception as e:
        return None, f"EIA Gas — FAILED ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
# EU Gas Storage — AGSI+ (free, no key)
# ══════════════════════════════════════════════════════════════════════════════

def get_eu_gas_storage() -> tuple[pd.DataFrame | None, str]:
    """
    EU aggregate gas storage level and % full — AGSI+ (Gas Infrastructure Europe).
    Free API, no key required.
    """
    try:
        r = requests.get(
            AGSI_EU_URL,
            params={"page": 1, "size": 300},
            headers={"x-key": ""},   # AGSI+ works without key for EU aggregate
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json()
        # AGSI+ returns list of daily records
        data = payload.get("data", [])
        if not data:
            return None, "AGSI+ — FAILED (empty)"
        rows = []
        for rec in data:
            try:
                rows.append({
                    "date":    pd.to_datetime(rec["gasDayStart"]),
                    "full_pct": float(rec["full"]),
                    "trend":    float(rec.get("trend", 0) or 0),
                    "status":   rec.get("status", ""),
                })
            except (KeyError, TypeError, ValueError):
                continue
        if not rows:
            return None, "AGSI+ — FAILED (parse error)"
        df = pd.DataFrame(rows).set_index("date").sort_index()
        return df, "AGSI+ (GIE)"
    except Exception as e:
        return None, f"AGSI+ — FAILED ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
# FAO Food Price Index (free)
# ══════════════════════════════════════════════════════════════════════════════

def get_fao_food_price_index() -> tuple[pd.Series | None, str]:
    """
    FAO Food Price Index (monthly, 2014-2016=100).
    Uses FAOSTAT API — free, no key required.
    Note: API may be slow or intermittently unavailable.
    """
    try:
        r = requests.get(FAO_FPI_URL, timeout=_TIMEOUT + 10)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data", [])
        if not data:
            return None, "FAO — FAILED (empty)"
        rows = []
        for rec in data:
            try:
                year  = int(rec.get("Year", 0))
                month = rec.get("Months", "")
                value = rec.get("Value")
                if value is None:
                    continue
                # Month code like "M01", "M02" etc.
                m_code = month[-2:] if len(month) >= 2 else "01"
                date = pd.to_datetime(f"{year}-{m_code}-01")
                rows.append({"date": date, "fpi": float(value)})
            except (TypeError, ValueError):
                continue
        if not rows:
            return None, "FAO — FAILED (parse)"
        s = pd.DataFrame(rows).set_index("date")["fpi"].sort_index()
        return s if len(s) > 6 else None, "FAO FAOSTAT"
    except Exception as e:
        return None, f"FAO — FAILED ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
# OECD Composite Leading Indicators (free SDMX API)
# ══════════════════════════════════════════════════════════════════════════════

def get_oecd_cli() -> tuple[pd.DataFrame | None, str]:
    """
    OECD Composite Leading Indicators (amplitude-adjusted).
    Free SDMX CSV API — no key required.
    Returns DataFrame(index=DatetimeIndex monthly, columns=country ISO2).
    """
    try:
        from io import StringIO
        r = requests.get(OECD_CLI_URL, timeout=_TIMEOUT + 10)
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text))
        if raw.empty or "OBS_VALUE" not in raw.columns:
            return None, "OECD CLI — FAILED (empty CSV)"

        raw = raw[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].dropna(subset=["OBS_VALUE"])
        raw["date"] = pd.to_datetime(raw["TIME_PERIOD"] + "-01", errors="coerce")
        raw = raw.dropna(subset=["date"])
        raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
        raw = raw.dropna(subset=["OBS_VALUE"])

        df = raw.pivot_table(index="date", columns="REF_AREA", values="OBS_VALUE", aggfunc="last")
        df = df.sort_index()
        if df.empty:
            return None, "OECD CLI — FAILED (no data after pivot)"
        return df, "OECD"
    except Exception as e:
        return None, f"OECD CLI — FAILED ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
# World Bank CMO — Commodity Markets Outlook Excel
# ══════════════════════════════════════════════════════════════════════════════

def _parse_cmo_sheet(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """Detect layout of a WB CMO sheet and extract named time series."""
    if raw.shape[0] < 5 or raw.shape[1] < 3:
        return {}

    # Find first row where col-0 is a 4-digit year (1990–2035)
    year_start = None
    for ri in range(min(20, raw.shape[0])):
        try:
            yr = int(float(raw.iloc[ri, 0]))
            if 1990 <= yr <= 2035:
                year_start = ri
                break
        except (ValueError, TypeError):
            continue
    if year_start is None or year_start == 0:
        return {}

    # Header row: scan upward for last non-empty row before year_start
    header_row = year_start - 1
    for ri in range(year_start - 1, -1, -1):
        vals = raw.iloc[ri].fillna("").astype(str)
        if any(len(v.strip()) > 1 for v in vals.iloc[1:]):
            header_row = ri
            break

    headers = raw.iloc[header_row].fillna("").astype(str).tolist()
    data_block = raw.iloc[year_start:]

    # Build year index
    years_dt, valid_idx = [], []
    for ri in range(len(data_block)):
        try:
            yr = int(float(data_block.iloc[ri, 0]))
            if 1990 <= yr <= 2035:
                years_dt.append(pd.Timestamp(f"{yr}-01-01"))
                valid_idx.append(ri)
        except (ValueError, TypeError):
            continue
    if len(years_dt) < 3:
        return {}

    subset = data_block.iloc[valid_idx]
    result = {}
    for ci in range(1, min(len(headers), subset.shape[1])):
        name = " ".join(str(headers[ci]).split()).strip()
        if not name or name.lower() in ("nan", "none") or len(name) < 2:
            continue
        try:
            vals = pd.to_numeric(subset.iloc[:, ci], errors="coerce")
            s = pd.Series(vals.values, index=years_dt[:len(vals)]).dropna()
            if len(s) >= 3:
                result[name] = s
        except Exception:
            continue
    return result


def get_wb_cmo() -> tuple[dict[str, pd.Series] | None, str]:
    """
    World Bank Commodity Markets Outlook — annual price data / forecasts.
    Fetches the Excel file directly. Free, no API key.
    Returns dict of commodity_name → pd.Series(DatetimeIndex, float).
    """
    import io
    try:
        resp = requests.get(
            WB_CMO_URL, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research/1.0)"},
        )
        resp.raise_for_status()
        xl = pd.ExcelFile(io.BytesIO(resp.content), engine="openpyxl")

        # Prioritise sheets likely to have annual price data
        prio = ["annual", "price", "forecast", "cmo", "data"]
        ordered = sorted(
            xl.sheet_names,
            key=lambda s: next((i for i, kw in enumerate(prio) if kw in s.lower()), 99),
        )
        result = {}
        for sheet in ordered[:6]:
            try:
                raw = xl.parse(sheet, header=None)
                parsed = _parse_cmo_sheet(raw)
                result.update(parsed)
                if len(result) >= 8:
                    break
            except Exception:
                continue

        if result:
            return result, "World Bank CMO (Oct 2025)"
        return None, "WB CMO — FAILED (no parseable data)"
    except Exception as e:
        return None, f"WB CMO — FAILED ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER functions — no real-time free API available
# ══════════════════════════════════════════════════════════════════════════════

def get_shipping_rates() -> tuple[None, str]:
    """
    PLACEHOLDER — Container shipping rates (Freightos Baltic Daily Index).
    Requires paid Freightos API subscription.
    See: https://fbx.freightos.com
    """
    return None, "PLACEHOLDER — Freightos Baltic Index (paid subscription required)"


def get_embi_spreads() -> tuple[None, str]:
    """
    PLACEHOLDER — EM sovereign spreads (JPMorgan EMBI).
    Requires Bloomberg Terminal or paid data subscription.
    Proxy available: use FRED BAMLH0A4HVBB (EM HY) as rough substitute.
    """
    return None, "PLACEHOLDER — JPMorgan EMBI (Bloomberg required)"


def get_move_index() -> tuple[None, str]:
    """
    PLACEHOLDER — ICE BofA MOVE bond volatility index.
    Requires ICE Data Services subscription.
    Proxy: compute from FRED daily yield changes (rolling std).
    """
    return None, "PLACEHOLDER — ICE MOVE Index (subscription required)"


def get_semiconductor_sales() -> tuple[None, str]:
    """
    PLACEHOLDER — Global semiconductor sales (WSTS monthly).
    Requires WSTS membership.
    See: https://www.wsts.org
    """
    return None, "PLACEHOLDER — WSTS Semiconductor Sales (membership required)"


def get_eu_electricity_prices() -> tuple[None, str]:
    """
    PLACEHOLDER — EU electricity spot prices (ENTSO-E).
    ENTSO-E Transparency Platform requires registration and API token.
    See: https://transparency.entsoe.eu
    """
    return None, "PLACEHOLDER — ENTSO-E (registration + API token required)"


def get_fertilizer_prices() -> tuple[None, str]:
    """
    PLACEHOLDER — Fertilizer prices (urea, ammonia, potash).
    World Bank Pink Sheet is Excel-only; no free REST API endpoint.
    See: https://www.worldbank.org/en/research/commodity-markets
    """
    return None, "PLACEHOLDER — World Bank Pink Sheet (Excel only, no REST API)"


def get_lme_inventories() -> tuple[None, str]:
    """
    PLACEHOLDER — LME metal inventory data (copper, nickel, aluminium).
    Requires LME Data subscription.
    See: https://www.lme.com
    """
    return None, "PLACEHOLDER — LME Data (subscription required)"


def get_gpr_index() -> tuple[pd.Series | None, str]:
    """
    Geopolitical Risk Index — Caldara & Iacoviello (2022).

    Attempt order:
    1. Local CSV: macro_intel/data/gpr_daily.csv  (download from matteoiacoviello.com/gpr.htm)
    2. Remote CSV: public GitHub mirror maintained by the authors.
    3. PLACEHOLDER — returns (None, "PLACEHOLDER...").

    CSV expected columns: DATE (or 'date'), GPRD_ALL (or 'gpr_daily').
    Source label: "Caldara & Iacoviello (GPR)".
    """
    import os
    import pathlib

    # ── 1. Try local file first ────────────────────────────────────────────────
    local_paths = [
        pathlib.Path(__file__).parent / "data" / "gpr_daily.csv",
        pathlib.Path(__file__).parent / "data" / "gpr_monthly.csv",
    ]
    for path in local_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
                # Normalise column names
                df.columns = [c.strip().lower() for c in df.columns]
                date_col = next((c for c in df.columns if "date" in c), None)
                val_col  = next((c for c in df.columns
                                 if "gprd_all" in c or "gpr" in c or "gprc" in c), None)
                if date_col and val_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
                    s = df.dropna(subset=[date_col, val_col]).set_index(date_col)[val_col].sort_index()
                    if len(s) > 10:
                        return s, "Caldara & Iacoviello (GPR) — local CSV"
            except Exception:
                pass

    # ── 2. Live fetch from matteoiacoviello.com ────────────────────────────────
    import io
    _HEADERS  = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    _GPR_XLS  = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    _GPR_XLSX = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xlsx"

    def _parse_gpr_df(df: pd.DataFrame):
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), None)
        val_col  = next((c for c in df.columns if "gprd_all" in c or "gprd" in c or "gpr" in c), None)
        if date_col and val_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
            s = df.dropna(subset=[date_col, val_col]).set_index(date_col)[val_col].sort_index()
            if len(s) > 10:
                return s
        return None

    # Try .xls (xlrd engine)
    try:
        resp = requests.get(_GPR_XLS, timeout=_TIMEOUT + 15, headers=_HEADERS)
        resp.raise_for_status()
        s = _parse_gpr_df(pd.read_excel(io.BytesIO(resp.content), engine="xlrd"))
        if s is not None:
            return s, "Caldara & Iacoviello (GPR)"
    except Exception:
        pass

    # Try .xlsx fallback (openpyxl engine)
    try:
        resp = requests.get(_GPR_XLSX, timeout=_TIMEOUT + 15, headers=_HEADERS)
        resp.raise_for_status()
        s = _parse_gpr_df(pd.read_excel(io.BytesIO(resp.content), engine="openpyxl"))
        if s is not None:
            return s, "Caldara & Iacoviello (GPR)"
    except Exception:
        pass

    # ── 3. Fallback ────────────────────────────────────────────────────────────
    return None, "Caldara & Iacoviello GPR — FAILED (fetch error)"


def get_lithium_prices() -> tuple[None, str]:
    """
    PLACEHOLDER — Lithium carbonate/hydroxide spot prices.
    Fastmarkets / S&P Global — commercial data, no free API.
    Proxy: use nickel (NI=F on yfinance) as critical minerals signal.
    """
    return None, "PLACEHOLDER — Fastmarkets lithium (commercial)"


def get_cobalt_prices() -> tuple[None, str]:
    """
    PLACEHOLDER — Cobalt prices (LME / Fastmarkets).
    No free real-time public API.
    """
    return None, "PLACEHOLDER — LME cobalt (subscription required)"


# ══════════════════════════════════════════════════════════════════════════════
# News Feed (RSS — free, no key)
# ══════════════════════════════════════════════════════════════════════════════

# category → list of (name, url)
_NEWS_SOURCES: dict[str, list[tuple[str, str]]] = {
    "Global Markets": [
        ("Reuters Business",  "https://feeds.reuters.com/reuters/businessNews"),
        ("BBC Business",      "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("FT",                "https://www.ft.com/rss/home"),
        ("FT Markets",        "https://www.ft.com/rss/home/markets"),
        ("CNBC World",        "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("Politico Economy",  "https://rss.politico.com/economy.xml"),
        ("Axios Markets",     "https://api.axios.com/feed/rss/finance"),
        ("The Economist",     "https://www.economist.com/finance-and-economics/rss.xml"),
    ],
    "Emerging Markets": [
        ("Reuters EM",        "https://feeds.reuters.com/reuters/emergingMarketsNews"),
        ("Nikkei Asia",       "https://asia.nikkei.com/rss/feed/nar"),
        ("Al-Monitor",        "https://www.al-monitor.com/rss"),
    ],
    "Africa": [
        ("The Africa Report", "https://www.theafricareport.com/feed/"),
        ("African Business",  "https://african.business/feed/"),
        ("AllAfrica Economy", "https://allafrica.com/tools/headlines/rdf/economy/headlines.rdf"),
        ("Reuters Africa",    "https://feeds.reuters.com/reuters/AFRICATopNews"),
        ("Business Day (ZA)", "https://www.businesslive.co.za/rss/bd/"),
    ],
}

_NEWS_COLORS = {
    "Reuters Business":  "#d35400",
    "BBC Business":      "#c0392b",
    "FT":                "#1a3c5e",
    "FT Markets":        "#1a3c5e",
    "CNBC World":        "#2980b9",
    "Politico Economy":  "#0f4c81",
    "Axios Markets":     "#6c3483",
    "The Economist":     "#E3120B",
    "Reuters EM":        "#e67e22",
    "Nikkei Asia":       "#c0392b",
    "Al-Monitor":        "#27ae60",
    "The Africa Report": "#8e44ad",
    "African Business":  "#16a085",
    "AllAfrica Economy": "#2c3e50",
    "Reuters Africa":    "#d35400",
    "Business Day (ZA)": "#1abc9c",
}


def get_news_feed() -> tuple[dict[str, list[dict]] | None, str]:
    """
    Financial news headlines via free RSS feeds.
    Returns dict: {category: [items]} where each item has
    {source, color, title, link, published, summary}.
    """
    import re
    import email.utils
    import feedparser

    _strip = re.compile(r"<[^>]+>")

    result: dict[str, list[dict]] = {}
    fetched: list[str] = []

    for category, sources in _NEWS_SOURCES.items():
        cat_items: list[dict] = []
        seen_titles: set[str] = set()
        for source_name, url in sources:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:7]:
                    title = entry.get("title", "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    pub_raw = entry.get("published", "") or entry.get("updated", "")
                    try:
                        pub_dt = email.utils.parsedate_to_datetime(pub_raw)
                        pub_str = pub_dt.strftime("%d %b %H:%M")
                    except Exception:
                        pub_str = pub_raw[:16]
                    summary = entry.get("summary", "") or entry.get("description", "")
                    summary = _strip.sub(" ", summary).strip()[:240]
                    cat_items.append({
                        "source":    source_name,
                        "color":     _NEWS_COLORS.get(source_name, "#2c3e50"),
                        "title":     title,
                        "link":      entry.get("link", ""),
                        "published": pub_str,
                        "summary":   summary,
                    })
                if feed.entries:
                    fetched.append(source_name)
            except Exception:
                pass
        if cat_items:
            result[category] = cat_items

    if result:
        return result, f"RSS ({', '.join(fetched[:6])}{'…' if len(fetched) > 6 else ''})"
    return None, "News RSS — FAILED"
