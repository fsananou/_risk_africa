"""
config.py — API keys, series IDs, thresholds, visual constants.

FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html):
  Option A: .streamlit/secrets.toml  →  [api_keys]  fred = "your_key"
  Option B: environment variable     →  export FRED_API_KEY=your_key

Without a FRED key the app falls back to US Treasury Direct + yfinance + synthetic data.
"""

from __future__ import annotations
import os

try:
    import streamlit as st
    def get_fred_key() -> str | None:
        try:
            return (st.secrets.get("api_keys", {}).get("fred")
                    or os.getenv("FRED_API_KEY"))
        except Exception:
            return os.getenv("FRED_API_KEY")
except ImportError:
    def get_fred_key() -> str | None:
        return os.getenv("FRED_API_KEY")


# ── FRED Series ────────────────────────────────────────────────────────────────
FRED_SERIES = {
    "us2y":           "DGS2",           # 2-Year Treasury Constant Maturity
    "us10y":          "DGS10",          # 10-Year Treasury Constant Maturity
    "us30y":          "DGS30",          # 30-Year Treasury Constant Maturity
    "us3m":           "DGS3MO",         # 3-Month Treasury Bill
    "breakeven5y":    "T5YIE",          # 5-Year Breakeven Inflation Rate
    "breakeven10y":   "T10YIE",         # 10-Year Breakeven Inflation Rate
    "breakeven5y5y":  "T5YIFR",         # 5-Year, 5-Year Forward Inflation Expectation
    "hy_spread":      "BAMLH0A0HYM2",   # ICE BofA US HY OAS (bps)
    "ig_spread":      "BAMLC0A0CM",     # ICE BofA US IG OAS (bps)
    "brent_fred":     "DCOILBRENTEU",   # Brent Crude Oil (USD/barrel, daily)
    "copper_fred":    "PCOPPUSDM",      # Global Copper Price (USD/metric ton, monthly)
    "nfci":           "NFCI",           # Chicago Fed National Financial Conditions Index
    "vix_fred":       "VIXCLS",         # CBOE VIX (daily close)
    "move_proxy":     "BAMLMOVE",       # MOVE bond vol index (if available)
    "term_premium":   "THREEFYTP10",    # 10-Year Term Premium (Kim-Wright model, Fed)
}

# ── World Bank Indicators ─────────────────────────────────────────────────────
WB_BASE = "https://api.worldbank.org/v2"
WB_INDICATORS = {
    "fx_reserves": "FI.RES.TOTL.CD",    # Total reserves incl. gold (current USD)
    "fdi_inflows": "BX.KLT.DINV.CD.WD", # FDI net inflows (current USD)
    "ext_debt":    "DT.DOD.DECT.CD",    # External debt stocks, total (current USD)
    "gdp_usd":     "NY.GDP.MKTP.CD",    # GDP (current USD)
    "imports":     "NE.IMP.GNFS.CD",    # Imports of goods & services (current USD)
    "population":  "SP.POP.TOTL",       # Population
}
WB_AFRICA = ["NGA", "ZAF", "KEN", "EGY", "ETH", "GHA", "TZA", "ZMB", "MOZ", "CIV",
             "DZA", "AGO", "SDN", "UGA", "CMR"]
WB_EM_BROAD = ["NGA", "ZAF", "KEN", "EGY", "GHA", "BRA", "MEX", "IND", "IDN", "CHN",
               "TUR", "ZMB", "ARG", "CHL", "VNM"]

# ── IMF DataMapper ─────────────────────────────────────────────────────────────
IMF_BASE = "https://www.imf.org/external/datamapper/api/v1"
IMF_INDICATORS = {
    "gdp_usd":   "NGDPD",       # GDP (USD bn)
    "inflation": "PCPIPCH",     # CPI inflation (% change)
    "ca_gdp":    "BCA_NGDPD",   # Current account (% of GDP)
    "gov_debt":  "GGXWDG_NGDP", # General government gross debt (% GDP)
    "reserves":  "RESA",        # Reserves (USD bn) — if available
}
IMF_COUNTRIES = ["US", "CN", "DE", "JP", "GB", "IN", "BR", "ZA", "NG", "EG",
                 "MX", "KE", "ET", "GH", "ID", "VN", "TR", "AR"]

# ── yfinance Tickers ──────────────────────────────────────────────────────────
TICKERS = {
    # Rates & vol
    "vix":    "^VIX",   "us10y": "^TNX",  "us30y": "^TYX",
    "us5y":   "^FVX",   "us3m":  "^IRX",
    # FX
    "dxy":    "DX-Y.NYB", "eurusd": "EURUSD=X", "usdjpy": "USDJPY=X",
    "usdbrl": "USDBRL=X", "usdzar": "USDZAR=X", "usdtry": "USDTRY=X",
    "usdcnh": "USDCNH=X", "usdinr": "USDINR=X", "usdmxn": "USDMXN=X",
    # Commodities
    "brent":  "BZ=F",  "wti":    "CL=F",   "copper": "HG=F",
    "gold":   "GC=F",  "silver": "SI=F",   "natgas": "NG=F",
    "wheat":  "ZW=F",  "corn":   "ZC=F",
    # Equities / ETFs
    "spx":    "^GSPC", "eem":   "EEM",    "hyg":   "HYG",
    "emb":    "EMB",   "tip":   "TIP",    "lqd":   "LQD",
    "iyr":    "IYR",   "xlf":   "XLF",
}

# ── Regime Thresholds ─────────────────────────────────────────────────────────
THRESH = {
    # VIX
    "vix_normal": 18, "vix_elevated": 25, "vix_high": 35,
    # Yield curve slope (bps, 10Y − 2Y)
    "curve_inverted": -10, "curve_flat": 50, "curve_steep": 150,
    # DXY
    "dxy_strong": 103, "dxy_very_strong": 106,
    # Oil / Copper 1M % change
    "oil_surge": 0.08, "oil_crash": -0.08,
    "copper_boom": 0.07, "copper_bust": -0.07,
    # GPR (Caldara & Iacoviello, baseline ≈ 100)
    "gpr_normal": 100, "gpr_elevated": 130, "gpr_high": 200,
    # Shipping (BDI-like index)
    "shipping_normal": 1_500, "shipping_stress": 2_500, "shipping_crisis": 4_000,
    # EM sovereign spreads (EMBI, bps)
    "em_spread_normal": 300, "em_spread_stress": 450, "em_spread_crisis": 600,
    # HY spread (bps)
    "hy_normal": 350, "hy_stress": 500, "hy_crisis": 700,
    # Breakeven inflation (%)
    "breakeven_normal": 2.0, "breakeven_high": 2.8, "breakeven_very_high": 3.5,
    # Chicago Fed NFCI (higher = tighter)
    "nfci_normal": 0.0, "nfci_tight": 0.5, "nfci_crisis": 1.5,
    # Financial conditions proxy (HYG 1M return)
    "fin_cond_tightening": -0.04, "fin_cond_easing": 0.02,
    # FX reserves adequacy (months of imports)
    "fx_res_adequate": 3.0, "fx_res_low": 1.5,
}

# ── Visual ────────────────────────────────────────────────────────────────────
LINE_COLORS = [
    "#2c3e50", "#c0392b", "#16a085", "#d35400",
    "#8e44ad", "#2980b9", "#7f8c8d", "#27ae60",
    "#e67e22", "#1abc9c",
]
LEVEL_COLOR = {"alert": "#c0392b", "warning": "#d35400", "info": "#2980b9"}
LEVEL_BG    = {"alert": "#fdf3f2", "warning": "#fef9f0", "info": "#f0f7fd"}
LEVEL_ICON  = {"alert": "🔴", "warning": "🟠", "info": "🔵"}
LEVEL_ORDER = {"alert": 0, "warning": 1, "info": 2}
CONF_COLOR  = {"high": "#c0392b", "medium": "#d35400", "low": "#7f8c8d"}
HORIZON_ICON = {
    "near-term (1-4 weeks)":   "⚡",
    "medium-term (1-6 months)": "📅",
    "structural (6-18 months)": "🏗",
}
