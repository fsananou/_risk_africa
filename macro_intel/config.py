"""
config.py — API keys, series IDs, thresholds, visual constants.

Free API keys required:
  FRED: https://fred.stlouisfed.org/docs/api/api_key.html
  EIA:  https://www.eia.gov/opendata/register.php

Set via .streamlit/secrets.toml:
  [api_keys]
  fred = "your_fred_key"
  eia  = "your_eia_key"

Or environment variables:
  export FRED_API_KEY=your_key
  export EIA_API_KEY=your_key
"""

from __future__ import annotations
import os

try:
    import streamlit as st

    def get_fred_key() -> str | None:
        try:
            return st.secrets.get("api_keys", {}).get("fred") or os.getenv("FRED_API_KEY")
        except Exception:
            return os.getenv("FRED_API_KEY")

    def get_eia_key() -> str | None:
        try:
            return st.secrets.get("api_keys", {}).get("eia") or os.getenv("EIA_API_KEY")
        except Exception:
            return os.getenv("EIA_API_KEY")

except ImportError:
    def get_fred_key() -> str | None:
        return os.getenv("FRED_API_KEY")

    def get_eia_key() -> str | None:
        return os.getenv("EIA_API_KEY")


# ── FRED Series ────────────────────────────────────────────────────────────────
FRED_SERIES = {
    "us2y":          "DGS2",
    "us5y":          "DGS5",
    "us10y":         "DGS10",
    "us30y":         "DGS30",
    "us3m":          "DGS3MO",
    "breakeven5y":   "T5YIE",
    "breakeven10y":  "T10YIE",
    "breakeven5y5y": "T5YIFR",
    "hy_spread":     "BAMLH0A0HYM2",
    "ig_spread":     "BAMLC0A0CM",
    "nfci":          "NFCI",
    "term_premium":  "THREEFYTP10",
    "brent":         "DCOILBRENTEU",
    "copper":        "PCOPPUSDM",
    "wheat":         "PWHEAMTUSDM",
    "corn":          "PMAIZMTUSDM",
    "rice":          "PRICENPQUSDM",
    "indpro_us":     "INDPRO",
}

# ── Yahoo Finance Tickers ──────────────────────────────────────────────────────
TICKERS = {
    "vix":    "^VIX",    "us10y":  "^TNX",    "us30y":  "^TYX",
    "us5y":   "^FVX",    "us3m":   "^IRX",
    "dxy":    "DX-Y.NYB","eurusd": "EURUSD=X","usdjpy": "USDJPY=X",
    "usdbrl": "USDBRL=X","usdzar": "USDZAR=X","usdtry": "USDTRY=X",
    "usdcnh": "USDCNH=X","usdinr": "USDINR=X","usdmxn": "USDMXN=X",
    "brent":  "BZ=F",    "wti":    "CL=F",    "copper": "HG=F",
    "gold":   "GC=F",    "natgas": "NG=F",    "wheat":  "ZW=F",
    "corn":   "ZC=F",    "nickel": "NI=F",
    "spx":    "^GSPC",   "eem":    "EEM",     "hyg":    "HYG",
    "emb":    "EMB",     "tip":    "TIP",     "lqd":    "LQD",
    "xlf":    "XLF",     "xle":    "XLE",     "xlb":    "XLB",
    "xli":    "XLI",
}

# ── World Bank ─────────────────────────────────────────────────────────────────
WB_BASE = "https://api.worldbank.org/v2"
WB_INDICATORS = {
    "fx_reserves": "FI.RES.TOTL.CD",
    "fdi_inflows": "BX.KLT.DINV.CD.WD",
    "ext_debt":    "DT.DOD.DECT.CD",
    "gdp_usd":     "NY.GDP.MKTP.CD",
    "imports":     "NE.IMP.GNFS.CD",
    "indpro":      "NV.IND.MANF.KD.ZG",
}
WB_AFRICA   = ["NGA","ZAF","KEN","EGY","ETH","GHA","TZA","ZMB","MOZ","CIV",
               "DZA","AGO","SDN","UGA","CMR"]
WB_EM_BROAD = ["NGA","ZAF","KEN","EGY","GHA","BRA","MEX","IND","IDN","CHN",
               "TUR","ZMB","ARG","CHL","VNM"]

# ── IMF ───────────────────────────────────────────────────────────────────────
IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"
IMF_INDICATORS = {
    "gdp_usd":   "NGDPD",
    "inflation": "PCPIPCH",
    "ca_gdp":    "BCA_NGDPD",
    "gov_debt":  "GGXWDG_NGDP",
}

# ── EIA (free key: eia.gov/opendata) ──────────────────────────────────────────
EIA_BASE = "https://api.eia.gov/v2"

# ── EU Gas Storage (AGSI+ — free, no key) ────────────────────────────────────
AGSI_EU_URL = "https://agsi.gie.eu/api/data/eu"

# ── FAO Food Price Index (free) ───────────────────────────────────────────────
# Note: FAO FAOSTAT API is rate-limited and may require retries.
FAO_FPI_URL = (
    "https://fenixservices.fao.org/faostat/api/v1/en/data/CP"
    "?area=1%3E%3E3&element=6132&item=23013&type=json"
    "&show_codes=true&show_unit=true&null_values=false&lang=en"
)

# ── OECD CLI (free SDMX API) ──────────────────────────────────────────────────
OECD_CLI_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.STES,DSD_STES@DF_CLI,4.0/"
    "OECDALL+USA+DEU+GBR+JPN+CHN+IND+BRA+ZAF.M.LI.AA.ST"
    "?startPeriod=2018-01&format=jsondata"
)

# ── Placeholder registry ───────────────────────────────────────────────────────
# Indicators with NO free real-time REST API — explicitly labeled.
PLACEHOLDERS = {
    "shipping_rates":    {
        "name": "Container Shipping Rates",
        "reason": "Freightos Baltic Daily Index — paid subscription required",
        "alt": "https://fbx.freightos.com (commercial)",
    },
    "embi_spreads":      {
        "name": "EM Sovereign Spreads (EMBI)",
        "reason": "JPMorgan EMBI — Bloomberg Terminal required",
        "alt": "Subscribe to Bloomberg or use FRED BAMLHY as EM proxy",
    },
    "move_index":        {
        "name": "MOVE Bond Volatility Index",
        "reason": "ICE BofA MOVE — ICE Data Services subscription",
        "alt": "Compute from FRED yield data (rolling std)",
    },
    "semiconductor":     {
        "name": "Semiconductor Sales",
        "reason": "WSTS data — member-only access",
        "alt": "https://www.wsts.org (membership required)",
    },
    "electricity_eu":    {
        "name": "EU Electricity Spot Prices",
        "reason": "ENTSO-E requires registration + complex API auth",
        "alt": "https://transparency.entsoe.eu",
    },
    "fertilizer_prices": {
        "name": "Fertilizer Prices (Urea, Ammonia, Potash)",
        "reason": "World Bank Pink Sheet — Excel download only, no REST API",
        "alt": "https://www.worldbank.org/en/research/commodity-markets",
    },
    "lme_inventories":   {
        "name": "LME Metal Inventories",
        "reason": "London Metal Exchange — LME Data subscription",
        "alt": "https://www.lme.com/en/metals (subscription)",
    },
    "gpr_index":         {
        "name": "Geopolitical Risk Index (Caldara & Iacoviello)",
        "reason": "Academic dataset — no live REST API; requires manual download",
        "alt": "https://www.matteoiacoviello.com/gpr.htm (CSV download)",
    },
    "lithium_prices":    {
        "name": "Lithium Carbonate Prices",
        "reason": "Fastmarkets / S&P Global — commercial data",
        "alt": "Nickel (NI=F on yfinance) available as proxy",
    },
    "cobalt_prices":     {
        "name": "Cobalt Prices",
        "reason": "Fastmarkets / LME — subscription required",
        "alt": "No free real-time source",
    },
}

# ── Regime Thresholds ─────────────────────────────────────────────────────────
THRESH = {
    "vix_normal": 18,       "vix_elevated": 25,     "vix_high": 35,
    "curve_inverted": -10,  "curve_flat": 50,        "curve_steep": 150,
    "dxy_strong": 103,      "dxy_very_strong": 106,
    "oil_surge": 0.08,      "oil_crash": -0.08,
    "copper_boom": 0.07,    "copper_bust": -0.07,
    "hy_normal": 350,       "hy_stress": 500,        "hy_crisis": 700,
    "breakeven_normal": 2.0,"breakeven_high": 2.8,   "breakeven_very_high": 3.5,
    "nfci_normal": 0.0,     "nfci_tight": 0.5,       "nfci_crisis": 1.5,
    # Sector
    "eu_gas_storage_low":   40.0,   # % full → stress
    "eu_gas_storage_crisis":20.0,   # % full → crisis
    "us_oil_inv_dev_stress":-0.05,  # -5% vs 5Y avg
    "fao_fpi_stress":       130,
    "fao_fpi_crisis":       160,
    "indpro_recession":     -2.0,   # YoY % → recession signal
}

# ── Visual ────────────────────────────────────────────────────────────────────
LEVEL_COLOR  = {"alert": "#c0392b", "warning": "#d35400", "info": "#2980b9"}
LEVEL_BG     = {"alert": "#fdf3f2", "warning": "#fef9f0", "info": "#f0f7fd"}
LEVEL_ICON   = {"alert": "🔴",      "warning": "🟠",      "info": "🔵"}
LEVEL_ORDER  = {"alert": 0,         "warning": 1,          "info": 2}
CONF_COLOR   = {"high": "#c0392b",  "medium": "#d35400",   "low": "#7f8c8d"}
HORIZON_ICON = {
    "near-term (1-4 weeks)":    "⚡",
    "medium-term (1-6 months)": "📅",
    "structural (6-18 months)": "🏗",
}
SECTOR_ICON = {
    "Energy":            "⚡",
    "Agriculture":       "🌾",
    "Chemicals":         "🧪",
    "Industrials":       "🏭",
    "Technology":        "💾",
    "Critical Minerals": "⛏️",
}
