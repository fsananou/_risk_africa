"""
Global Macro Dashboard — for economists.
Clean, information-dense, forward-looking.

Run:  streamlit run macro_dashboard.py
Data: yfinance · US Treasury Direct · RSS feeds (no API key needed)
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding: 0.6rem 1.8rem 2rem; }
  .section-label {
    font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: #aaa; font-weight: 700; border-bottom: 1px solid #e8e8e8;
    padding-bottom: 3px; margin: 10px 0 8px;
  }
  div[data-testid="metric-container"] {
    background: #f7f7f7; border-radius: 3px; padding: 5px 10px;
  }
  .dtab { width:100%; font-size:0.81rem; border-collapse:collapse; font-family:monospace; }
  .dtab th { color:#bbb; font-size:0.6rem; text-transform:uppercase;
             letter-spacing:0.07em; font-weight:600; padding:2px 6px; }
  .dtab td { padding: 4px 6px; border-bottom: 1px solid #f4f4f4; }
  .up   { color: #1e7e34 !important; font-weight:600; }
  .down { color: #c0392b !important; font-weight:600; }
  .news-item { padding: 7px 0; border-bottom: 1px solid #f0f0f0; }
  .news-title { font-size:0.84rem; color:#1a1a1a; font-weight:500;
                text-decoration:none; line-height:1.35; }
  .news-title:hover { text-decoration:underline; }
  .news-meta { font-size:0.67rem; color:#bbb; margin-top:2px; }
  .tag { background:#f0f0f0; border-radius:2px; padding:1px 5px;
         font-size:0.62rem; color:#888; margin-right:3px; }
  .signal-ok   { color:#1e7e34; font-weight:700; }
  .signal-warn { color:#d35400; font-weight:700; }
  .signal-bad  { color:#c0392b; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Tickers ───────────────────────────────────────────────────────────────────
EQUITY = {
    "S&P 500":    "^GSPC",
    "Nasdaq 100": "^NDX",
    "DJIA":       "^DJI",
    "Euro Stoxx": "^STOXX50E",
    "DAX":        "^GDAXI",
    "FTSE 100":   "^FTSE",
    "Nikkei 225": "^N225",
    "Hang Seng":  "^HSI",
    "CSI 300":    "000300.SS",
    "MSCI EM":    "EEM",
    "Nifty 50":   "^NSEI",
    "Bovespa":    "^BVSP",
}

FX = {
    "DXY":     "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CNH": "USDCNH=X",
    "USD/BRL": "USDBRL=X",
    "USD/ZAR": "USDZAR=X",
    "USD/TRY": "USDTRY=X",
    "USD/MXN": "USDMXN=X",
    "USD/INR": "USDINR=X",
}

COMMODITIES = {
    "Brent":   "BZ=F",
    "WTI":     "CL=F",
    "Gold":    "GC=F",
    "Silver":  "SI=F",
    "Copper":  "HG=F",
    "Nat Gas": "NG=F",
    "Wheat":   "ZW=F",
    "Corn":    "ZC=F",
}

RATES = {
    "US 3M":  "^IRX",
    "US 5Y":  "^FVX",
    "US 10Y": "^TNX",
    "US 30Y": "^TYX",
    "VIX":    "^VIX",
}

PULSE = {
    "S&P 500": "^GSPC",
    "10Y UST": "^TNX",
    "DXY":     "DX-Y.NYB",
    "WTI":     "CL=F",
    "Gold":    "GC=F",
    "VIX":     "^VIX",
    "MSCI EM": "EEM",
}

NEWS_FEEDS = [
    ("Reuters",          "https://feeds.reuters.com/reuters/businessNews"),
    ("WSJ Markets",      "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("The Economist",    "https://www.economist.com/finance-and-economics/rss.xml"),
    ("IMF",              "https://www.imf.org/en/News/RSS"),
    ("Project Syndicate","https://www.project-syndicate.org/rss/"),
    ("World Bank",       "https://feeds.worldbank.org/RSS/blogs"),
    ("FT",               "https://www.ft.com/economics?format=rss"),
]

LINE_COLORS = [
    "#2c3e50", "#c0392b", "#16a085", "#d35400",
    "#8e44ad", "#2980b9", "#7f8c8d", "#27ae60",
    "#e67e22", "#1abc9c", "#95a5a6", "#e74c3c",
]

YIELD_MAP = {
    "1 Mo": 1/12, "2 Mo": 2/12, "3 Mo": 3/12, "4 Mo": 4/12,
    "6 Mo": 6/12, "1 Yr": 1, "2 Yr": 2, "3 Yr": 3, "5 Yr": 5,
    "7 Yr": 7, "10 Yr": 10, "20 Yr": 20, "30 Yr": 30,
}

# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def fetch_prices(tickers_tuple: tuple, period: str = "3mo") -> pd.DataFrame:
    names = [t[0] for t in tickers_tuple]
    syms  = [t[1] for t in tickers_tuple]
    sym_to_name = dict(zip(syms, names))
    try:
        raw = yf.download(syms, period=period, auto_adjust=True,
                          progress=False, threads=True)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    try:
        close = raw["Close"].copy()
        close = close.rename(columns=sym_to_name)
    except Exception:
        return pd.DataFrame()
    return close


@st.cache_data(ttl=3600)
def fetch_yield_curve() -> tuple:
    today = datetime.now()
    dfs = []
    for year in [today.year - 1, today.year]:
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
        return pd.Series(dtype=float), pd.DataFrame()
    full = (pd.concat(dfs)
              .drop_duplicates("Date")
              .sort_values("Date")
              .reset_index(drop=True))
    num_cols = [c for c in full.columns if c != "Date"]
    latest = full.iloc[-1][num_cols].astype(float)
    hist = full.set_index("Date")[num_cols].astype(float)
    return latest, hist


@st.cache_data(ttl=1800)
def fetch_news(max_per_feed: int = 8) -> list:
    items = []
    for source, url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
            for e in feed.entries[:max_per_feed]:
                t = e.get("published_parsed") or e.get("updated_parsed")
                pub = datetime(*t[:6]) if t else datetime.now()
                title = e.get("title", "").strip()
                if title:
                    items.append({
                        "title":  title,
                        "link":   e.get("link", ""),
                        "source": source,
                        "pub":    pub,
                    })
        except Exception:
            continue
    items.sort(key=lambda x: x["pub"], reverse=True)
    # Deduplicate by title prefix
    seen, out = set(), []
    for it in items:
        key = it["title"][:55].lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

# ── Helpers ───────────────────────────────────────────────────────────────────
def last_chg(df: pd.DataFrame, col: str):
    s = df[col].dropna() if col in df.columns else pd.Series(dtype=float)
    if len(s) < 2:
        return (s.iloc[-1] if len(s) else np.nan), np.nan
    return s.iloc[-1], (s.iloc[-1] / s.iloc[-2] - 1) * 100


def fmt(val, dec=2):
    if pd.isna(val): return "—"
    if val > 10_000: return f"{val:,.0f}"
    if val > 1_000:  return f"{val:,.1f}"
    return f"{val:.{dec}f}"


def age_str(pub: datetime) -> str:
    delta = datetime.utcnow() - pub
    if delta.days > 0:    return f"{delta.days}d ago"
    if delta.seconds > 3600: return f"{delta.seconds // 3600}h ago"
    return f"{max(1, delta.seconds // 60)}m ago"


# ── Chart helpers ─────────────────────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="monospace", size=10, color="#333"),
    margin=dict(l=42, r=8, t=22, b=28),
    xaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd"),
    yaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd"),
    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)",
                orientation="h", y=-0.18),
    height=240,
    hovermode="x unified",
)


def line_fig(df: pd.DataFrame, title: str = "", norm: bool = False,
             height: int = 240) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        s = df[col].dropna()
        y = (s / s.iloc[0] * 100) if norm and len(s) > 0 else s
        fig.add_trace(go.Scatter(
            x=s.index, y=y, name=col,
            line=dict(width=1.5, color=LINE_COLORS[i % len(LINE_COLORS)]),
            mode="lines",
        ))
    layout = {**_LAYOUT, "height": height}
    if title:
        layout["title"] = dict(text=title, font=dict(size=10, color="#888"))
    fig.update_layout(**layout)
    return fig


def curve_fig(curve: pd.Series) -> go.Figure:
    pts = [(YIELD_MAP[k], curve[k]) for k in YIELD_MAP if k in curve.index]
    x, y = zip(*pts) if pts else ([], [])
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color="#2c3e50", width=2),
        marker=dict(size=5, color="#2c3e50"),
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**{**_LAYOUT,
        "xaxis": dict(title="Maturity (years)",
                      tickvals=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
                      ticktext=["3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"],
                      gridcolor="#f0f0f0", linecolor="#ddd"),
        "yaxis": dict(title="Yield (%)", gridcolor="#f0f0f0", linecolor="#ddd"),
        "hovermode": "x",
    })
    return fig


def curve_hist_fig(hist: pd.DataFrame) -> go.Figure:
    cols = [k for k in YIELD_MAP if k in hist.columns]
    x_vals = [YIELD_MAP[k] for k in cols]
    n = len(hist)
    snaps = [(lbl, idx) for lbl, idx in
             [("Today", -1), ("1M ago", -22), ("3M ago", -63), ("1Y ago", -252)]
             if abs(idx) <= n]
    colors = ["#2c3e50", "#7f8c8d", "#aab7c4", "#d5d8dc"]
    widths = [2.5, 1.5, 1.2, 1.0]
    fig = go.Figure()
    for i, (lbl, idx) in enumerate(snaps):
        row = hist.iloc[idx]
        y = [row[c] for c in cols if not pd.isna(row.get(c, np.nan))]
        fig.add_trace(go.Scatter(
            x=x_vals[:len(y)], y=y, name=lbl,
            line=dict(color=colors[i], width=widths[i]),
            mode="lines+markers", marker=dict(size=4),
        ))
    fig.update_layout(**{**_LAYOUT,
        "xaxis": dict(title="Maturity (years)",
                      tickvals=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
                      ticktext=["3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"],
                      gridcolor="#f0f0f0", linecolor="#ddd"),
        "yaxis": dict(title="Yield (%)", gridcolor="#f0f0f0", linecolor="#ddd"),
        "legend": dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    })
    return fig


# ── Table builder ─────────────────────────────────────────────────────────────
def make_table(df: pd.DataFrame, tickers: dict, dec: int = 2) -> str:
    h = ('<table class="dtab"><tr>'
         '<th style="text-align:left;">Name</th>'
         '<th style="text-align:right;">Last</th>'
         '<th style="text-align:right;">1D&nbsp;%</th></tr>')
    for name in tickers:
        if name not in df.columns:
            continue
        price, chg = last_chg(df, name)
        sign = "+" if not pd.isna(chg) and chg > 0 else ""
        chg_str = f"{sign}{chg:.2f}%" if not pd.isna(chg) else "—"
        cls = ("up" if chg > 0 else "down") if not pd.isna(chg) else ""
        h += (f'<tr><td>{name}</td>'
              f'<td style="text-align:right;">{fmt(price, dec)}</td>'
              f'<td style="text-align:right;"><span class="{cls}">{chg_str}</span></td></tr>')
    h += "</table>"
    return h

# ── Macro signals ─────────────────────────────────────────────────────────────
def render_signals(curve: pd.Series, rates_df: pd.DataFrame):
    """Small forward-looking signal bar."""
    signals = []

    # Yield curve slope
    s10 = curve.get("10 Yr", np.nan)
    s2  = curve.get("2 Yr",  np.nan)
    s3m = curve.get("3 Mo",  np.nan)
    if not (pd.isna(s10) or pd.isna(s2)):
        sp = (s10 - s2) * 100
        if sp < -25:
            signals.append(("Yield curve", f"Inverted {sp:+.0f} bps", "bad",
                            "Deep inversion — historical recession precursor"))
        elif sp < 0:
            signals.append(("Yield curve", f"Inverted {sp:+.0f} bps", "warn",
                            "Mild inversion — watch economic data"))
        else:
            signals.append(("Yield curve", f"{sp:+.0f} bps", "ok",
                            "Positive slope — normal"))

    # VIX
    if "VIX" in rates_df.columns:
        vix, _ = last_chg(rates_df, "VIX")
        if not pd.isna(vix):
            if vix >= 35:
                signals.append(("VIX", f"{vix:.1f}", "bad", "High stress / crisis mode"))
            elif vix >= 22:
                signals.append(("VIX", f"{vix:.1f}", "warn", "Elevated volatility"))
            else:
                signals.append(("VIX", f"{vix:.1f}", "ok", "Calm market"))

    # 10Y-3M spread (alternative recession signal)
    if not (pd.isna(s10) or pd.isna(s3m)):
        sp2 = (s10 - s3m) * 100
        label = "warn" if sp2 < 0 else "ok"
        signals.append(("10Y–3M", f"{sp2:+.0f} bps", label,
                        "Inverted" if sp2 < 0 else "Normal"))

    if not signals:
        return

    st.markdown('<div class="section-label">Forward Indicators</div>', unsafe_allow_html=True)
    cols = st.columns(len(signals))
    for i, (name, val, level, note) in enumerate(signals):
        cls = f"signal-{level}"
        cols[i].markdown(
            f"**{name}**  \n"
            f'<span class="{cls}">{val}</span>  \n'
            f'<span style="font-size:0.72rem;color:#aaa;">{note}</span>',
            unsafe_allow_html=True,
        )

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.utcnow()

    # Header
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'margin-bottom:6px;">'
        f'<span style="font-size:1.1rem;font-weight:700;font-family:monospace;'
        f'letter-spacing:0.05em;">GLOBAL MACRO DASHBOARD</span>'
        f'<span style="font-size:0.68rem;color:#bbb;font-family:monospace;">'
        f'{now.strftime("%Y-%m-%d  %H:%M UTC")}'
        f'&nbsp;·&nbsp;prices cached 15 min</span></div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Settings")
        period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y"], index=1)
        st.markdown("---")
        st.caption(
            "**Data sources**\n"
            "- yfinance (prices, ETFs)\n"
            "- US Treasury Direct (yields)\n"
            "- RSS feeds (news)\n\n"
            "No API key required."
        )
        if st.button("Clear cache & refresh"):
            st.cache_data.clear()
            st.rerun()

    # ── Pulse bar ─────────────────────────────────────────────────────────────
    pulse_df = fetch_prices(tuple(PULSE.items()), period="5d")
    pcols = st.columns(len(PULSE))
    for i, name in enumerate(PULSE):
        price, chg = last_chg(pulse_df, name)
        sign = "+" if not pd.isna(chg) and chg > 0 else ""
        delta = f"{sign}{chg:.2f}%" if not pd.isna(chg) else None
        pcols[i].metric(name, fmt(price), delta)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["Equities", "Rates & Curves", "FX & EM", "Commodities", "News"])

    # ════════ Equities ════════════════════════════════════════════════════════
    with tabs[0]:
        eq_df = fetch_prices(tuple(EQUITY.items()), period=period)
        c1, c2 = st.columns([1, 2.8])
        with c1:
            st.markdown('<div class="section-label">Indices — 1D</div>', unsafe_allow_html=True)
            st.markdown(make_table(eq_df, EQUITY), unsafe_allow_html=True)
        with c2:
            dm = [c for c in ["S&P 500","Nasdaq 100","Euro Stoxx","DAX","FTSE 100","Nikkei 225"]
                  if c in eq_df.columns]
            em = [c for c in ["MSCI EM","CSI 300","Nifty 50","Bovespa","Hang Seng"]
                  if c in eq_df.columns]
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="section-label">Developed — rebased 100</div>', unsafe_allow_html=True)
                if dm: st.plotly_chart(line_fig(eq_df[dm], norm=True), use_container_width=True)
            with cb:
                st.markdown('<div class="section-label">Emerging — rebased 100</div>', unsafe_allow_html=True)
                if em: st.plotly_chart(line_fig(eq_df[em], norm=True), use_container_width=True)

    # ════════ Rates & Curves ══════════════════════════════════════════════════
    with tabs[1]:
        rates_df = fetch_prices(tuple(RATES.items()), period="1y")
        curve, curve_hist = fetch_yield_curve()

        # Forward-looking signals
        render_signals(curve, rates_df)
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">US Treasury Yield Curve — Latest</div>', unsafe_allow_html=True)
            if not curve.empty:
                st.plotly_chart(curve_fig(curve), use_container_width=True)
                # Key spreads
                s10 = curve.get("10 Yr", np.nan)
                s2  = curve.get("2 Yr",  np.nan)
                s3m = curve.get("3 Mo",  np.nan)
                s30 = curve.get("30 Yr", np.nan)
                m1, m2, m3 = st.columns(3)
                def sp(a, b):
                    return f"{(a - b) * 100:+.0f} bps" if not (pd.isna(a) or pd.isna(b)) else "—"
                m1.metric("10Y – 2Y",  sp(s10, s2))
                m2.metric("10Y – 3M",  sp(s10, s3m))
                m3.metric("30Y – 2Y",  sp(s30, s2))
            else:
                st.warning("Yield curve data unavailable.")

        with c2:
            st.markdown('<div class="section-label">Curve Shifts — Today vs 1M / 3M / 1Y</div>', unsafe_allow_html=True)
            if not curve_hist.empty:
                st.plotly_chart(curve_hist_fig(curve_hist), use_container_width=True)

        # Rates history
        st.markdown('<div class="section-label">Treasury Yields — 1 Year</div>', unsafe_allow_html=True)
        rate_cols = [c for c in ["US 3M","US 5Y","US 10Y","US 30Y"] if c in rates_df.columns]
        if rate_cols:
            st.plotly_chart(line_fig(rates_df[rate_cols], height=220), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="section-label">VIX — Equity Fear Gauge</div>', unsafe_allow_html=True)
            if "VIX" in rates_df.columns:
                vfig = line_fig(rates_df[["VIX"]], height=200)
                vfig.add_hline(y=20, line_dash="dot", line_color="#ccc",
                               annotation_text="20", annotation_font_size=9)
                vfig.add_hline(y=30, line_dash="dot", line_color="#c0392b",
                               annotation_text="30 — stress", annotation_font_size=9)
                st.plotly_chart(vfig, use_container_width=True)

        with c4:
            st.markdown('<div class="section-label">Rate Table</div>', unsafe_allow_html=True)
            if not curve.empty:
                tenors = ["1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"]
                h = '<table class="dtab"><tr><th>Tenor</th><th style="text-align:right;">Yield</th></tr>'
                for t in tenors:
                    if t in curve.index and not pd.isna(curve[t]):
                        h += f'<tr><td>{t}</td><td style="text-align:right;">{curve[t]:.2f}%</td></tr>'
                h += "</table>"
                st.markdown(h, unsafe_allow_html=True)

    # ════════ FX & EM ═════════════════════════════════════════════════════════
    with tabs[2]:
        fx_df = fetch_prices(tuple(FX.items()), period=period)
        c1, c2 = st.columns([1, 2.8])
        with c1:
            st.markdown('<div class="section-label">FX Rates — 1D</div>', unsafe_allow_html=True)
            st.markdown(make_table(fx_df, FX, dec=4), unsafe_allow_html=True)
        with c2:
            dm_fx = [c for c in ["DXY","EUR/USD","GBP/USD","USD/JPY","USD/CHF"] if c in fx_df.columns]
            em_fx = [c for c in ["USD/BRL","USD/ZAR","USD/TRY","USD/MXN","USD/INR","USD/CNH"] if c in fx_df.columns]
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="section-label">DM FX — rebased 100</div>', unsafe_allow_html=True)
                if dm_fx: st.plotly_chart(line_fig(fx_df[dm_fx], norm=True), use_container_width=True)
            with cb:
                st.markdown('<div class="section-label">EM FX vs USD — rebased 100 (↑ = weaker EM)</div>', unsafe_allow_html=True)
                if em_fx: st.plotly_chart(line_fig(fx_df[em_fx], norm=True), use_container_width=True)

    # ════════ Commodities ═════════════════════════════════════════════════════
    with tabs[3]:
        com_df = fetch_prices(tuple(COMMODITIES.items()), period=period)
        c1, c2 = st.columns([1, 2.8])
        with c1:
            st.markdown('<div class="section-label">Commodities — 1D</div>', unsafe_allow_html=True)
            st.markdown(make_table(com_df, COMMODITIES), unsafe_allow_html=True)
        with c2:
            energy = [c for c in ["Brent","WTI","Nat Gas"] if c in com_df.columns]
            metals = [c for c in ["Gold","Silver","Copper"] if c in com_df.columns]
            agri   = [c for c in ["Wheat","Corn"] if c in com_df.columns]
            ca, cb = st.columns(2)
            with ca:
                st.markdown('<div class="section-label">Energy — rebased 100</div>', unsafe_allow_html=True)
                if energy: st.plotly_chart(line_fig(com_df[energy], norm=True), use_container_width=True)
                st.markdown('<div class="section-label">Agriculture — rebased 100</div>', unsafe_allow_html=True)
                if agri: st.plotly_chart(line_fig(com_df[agri], norm=True, height=200), use_container_width=True)
            with cb:
                st.markdown('<div class="section-label">Metals — rebased 100</div>', unsafe_allow_html=True)
                if metals: st.plotly_chart(line_fig(com_df[metals], norm=True), use_container_width=True)

    # ════════ News ════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-label">Latest Economic & Market News</div>', unsafe_allow_html=True)
        with st.spinner("Loading news feeds…"):
            news = fetch_news()
        if not news:
            st.info("News feeds temporarily unavailable.")
        else:
            for item in news[:50]:
                st.markdown(
                    f'<div class="news-item">'
                    f'<a class="news-title" href="{item["link"]}" target="_blank">{item["title"]}</a>'
                    f'<div class="news-meta"><span class="tag">{item["source"]}</span>'
                    f' {age_str(item["pub"])}</div></div>',
                    unsafe_allow_html=True,
                )

    # Footer
    st.markdown("---")
    fc1, fc2 = st.columns([1, 5])
    with fc1:
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()
    with fc2:
        st.markdown(
            '<span style="font-size:0.68rem;color:#ccc;">'
            'yfinance · US Treasury Direct · RSS feeds — '
            'prices 15 min · yields 1 hr · news 30 min</span>',
            unsafe_allow_html=True,
        )


main()
