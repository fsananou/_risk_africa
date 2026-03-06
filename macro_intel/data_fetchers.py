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
    IMF_BASE, OECD_CLI_URL, TICKERS, WB_AFRICA, WB_BASE,
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


def _imf(indicator: str, countries: list[str]) -> pd.DataFrame | None:
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


# ══════════════════════════════════════════════════════════════════════════════
# IMF
# ══════════════════════════════════════════════════════════════════════════════

def get_imf_macro() -> tuple[dict[str, pd.DataFrame] | None, str]:
    """IMF DataMapper: inflation, govt debt, current account."""
    countries = ["US","CN","DE","JP","IN","BR","ZA","NG","EG","TR","GH","KE"]
    result = {}
    for label, ind_id in [("inflation","PCPIPCH"),("gov_debt","GGXWDG_NGDP"),
                           ("ca_gdp","BCA_NGDPD"),("gdp_usd","NGDPD")]:
        df = _imf(ind_id, countries)
        if df is not None:
            result[label] = df
    if result:
        return result, "IMF DataMapper"
    return None, "IMF DataMapper — FAILED"


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
    Free SDMX-JSON API — no key required.
    """
    try:
        r = requests.get(OECD_CLI_URL, timeout=_TIMEOUT + 10,
                         headers={"Accept": "application/vnd.sdmx.data+json"})
        r.raise_for_status()
        payload = r.json()

        # Parse SDMX-JSON structure
        ds = payload["data"]["dataSets"][0]["series"]
        structure = payload["data"]["structure"]
        dims = structure["dimensions"]["series"]

        # Find country dimension index
        country_dim = next(d for d in dims if d["id"] == "REF_AREA")
        country_idx = dims.index(country_dim)
        country_values = [v["id"] for v in country_dim["values"]]

        # Time periods
        time_dim = structure["dimensions"]["observation"][0]
        periods = [v["id"] for v in time_dim["values"]]
        dates = pd.to_datetime([p + "-01" for p in periods])

        result = {}
        for key_str, series_data in ds.items():
            keys = key_str.split(":")
            if len(keys) <= country_idx:
                continue
            c_idx = int(keys[country_idx])
            if c_idx >= len(country_values):
                continue
            country = country_values[c_idx]
            obs = series_data.get("observations", {})
            vals = {int(t): obs[t][0] for t in obs if obs[t][0] is not None}
            if not vals:
                continue
            s = pd.Series({dates[t]: v for t, v in vals.items() if t < len(dates)})
            result[country] = s

        if not result:
            return None, "OECD CLI — FAILED (no series)"
        df = pd.DataFrame(result).sort_index()
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

    # ── 2. Live fetch from matteoiacoviello.com (XLS) then GitHub CSV mirror ──
    #
    # Primary: official XLS (old Excel format — requires xlrd)
    #   https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls
    # Fallback: GitHub CSV mirror
    _GPR_XLS = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    _GPR_CSV = "https://raw.githubusercontent.com/pjpmarques/World-Modeling-Datasets/master/GPR/gpr_daily.csv"

    def _parse_gpr_df(df: pd.DataFrame, source_label: str):
        df.columns = [c.strip().lower() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), None)
        val_col  = next((c for c in df.columns if "gprd_all" in c or "gprd" in c or "gpr" in c), None)
        if date_col and val_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[val_col]  = pd.to_numeric(df[val_col], errors="coerce")
            s = df.dropna(subset=[date_col, val_col]).set_index(date_col)[val_col].sort_index()
            if len(s) > 10:
                return s, source_label
        return None, None

    # Try XLS (official source — xlrd engine required for .xls)
    try:
        import io
        resp = requests.get(_GPR_XLS, timeout=_TIMEOUT + 10)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), engine="xlrd")
        s, lbl = _parse_gpr_df(df, "Caldara & Iacoviello (GPR) — matteoiacoviello.com")
        if s is not None:
            return s, lbl
    except Exception:
        pass

    # Try CSV mirror
    try:
        resp = requests.get(_GPR_CSV, timeout=_TIMEOUT)
        resp.raise_for_status()
        import io
        df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip")
        s, lbl = _parse_gpr_df(df, "Caldara & Iacoviello (GPR) — GitHub mirror")
        if s is not None:
            return s, lbl
    except Exception:
        pass

    # ── 3. Fallback: PLACEHOLDER ───────────────────────────────────────────────
    return (
        None,
        "PLACEHOLDER — Caldara & Iacoviello GPR: "
        "install xlrd (pip install xlrd) — data fetched from "
        "matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    )


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
