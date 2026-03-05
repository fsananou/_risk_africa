"""
config.py — thresholds, ticker mappings, visual constants.
All tunable parameters live here so the rules engine stays clean.
"""

# ── Regime thresholds ─────────────────────────────────────────────────────────
THRESH = {
    # Volatility (VIX)
    "vix_normal":    18,
    "vix_elevated":  25,
    "vix_high":      35,
    # Yield curve slope (bps, 10Y − 2Y)
    "curve_inverted": -10,
    "curve_flat":      50,
    "curve_steep":    150,
    # DXY
    "dxy_strong":      103,
    "dxy_very_strong": 106,
    # Oil / Copper (1-month % change)
    "oil_surge":    0.08,
    "oil_crash":   -0.08,
    "copper_boom":  0.07,
    "copper_bust": -0.07,
    # Geopolitical Risk Index (Caldara & Iacoviello baseline ≈ 100)
    "gpr_normal":   100,
    "gpr_elevated": 130,
    "gpr_high":     200,
    # Shipping / BDI-like index
    "shipping_normal": 1_500,
    "shipping_stress": 2_500,
    "shipping_crisis": 4_000,
    # EM sovereign spreads (EMBI, bps)
    "em_spread_normal": 300,
    "em_spread_stress": 450,
    "em_spread_crisis": 600,
    # Financial conditions proxy (HYG ETF 1M total return)
    "fin_cond_tightening": -0.04,
    "fin_cond_easing":      0.02,
}

# ── yfinance tickers ──────────────────────────────────────────────────────────
TICKERS = {
    # Rates
    "vix":    "^VIX",
    "us10y":  "^TNX",
    "us30y":  "^TYX",
    "us5y":   "^FVX",
    "us3m":   "^IRX",
    # FX
    "dxy":    "DX-Y.NYB",
    "eurusd": "EURUSD=X",
    "usdjpy": "USDJPY=X",
    "usdbrl": "USDBRL=X",
    "usdzar": "USDZAR=X",
    "usdtry": "USDTRY=X",
    "usdcnh": "USDCNH=X",
    # Commodities
    "brent":  "BZ=F",
    "wti":    "CL=F",
    "copper": "HG=F",
    "gold":   "GC=F",
    "natgas": "NG=F",
    # Equities / ETFs
    "spx":    "^GSPC",
    "eem":    "EEM",   # MSCI EM
    "hyg":    "HYG",   # HY bond → financial conditions proxy
    "emb":    "EMB",   # EM bonds
    "tip":    "TIP",   # TIPS → inflation proxy
}

# ── Chart colors ──────────────────────────────────────────────────────────────
LINE_COLORS = [
    "#2c3e50", "#c0392b", "#16a085", "#d35400",
    "#8e44ad", "#2980b9", "#7f8c8d", "#27ae60",
]

# Signal styling
LEVEL_COLOR = {"alert": "#c0392b", "warning": "#d35400", "info": "#2980b9"}
LEVEL_BG    = {"alert": "#fdf3f2", "warning": "#fef9f0", "info": "#f0f7fd"}
LEVEL_ICON  = {"alert": "🔴", "warning": "🟠", "info": "🔵"}
LEVEL_ORDER = {"alert": 0, "warning": 1, "info": 2}
