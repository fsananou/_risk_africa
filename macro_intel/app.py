"""
app.py — Global Macro Intelligence Dashboard.

Architecture:
  data_fetchers  →  indicators  →  rules_engine  →  Streamlit UI

Run:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_fetchers as df_
import indicators as ind_
import rules_engine as re_
from config import LEVEL_BG, LEVEL_COLOR, LEVEL_ICON, LINE_COLORS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Intel",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding: 0.6rem 1.8rem 2rem; }
  .section-label {
    font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em;
    color: #aaa; font-weight: 700; border-bottom: 1px solid #e8e8e8;
    padding-bottom: 3px; margin: 10px 0 8px;
  }
  div[data-testid="metric-container"] {
    background: #f7f7f7; border-radius: 3px; padding: 6px 10px;
  }
  .insight-card {
    border-left: 4px solid #ddd; padding: 10px 14px;
    border-radius: 3px; margin-bottom: 8px;
  }
  .insight-badge {
    display: inline-block; font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.06em; padding: 2px 7px; border-radius: 2px;
    color: white; margin-right: 6px; text-transform: uppercase;
  }
  .insight-cat  { font-size: 0.68rem; color: #999; }
  .insight-head { font-size: 0.92rem; font-weight: 600; margin: 4px 0; }
  .insight-detail { font-size: 0.81rem; color: #444; line-height: 1.55; }
  .insight-watch { font-size: 0.71rem; color: #999; margin-top: 5px; }
  .stress-pill {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
  }
  .dtab { width:100%; font-size:0.81rem; border-collapse:collapse; font-family:monospace; }
  .dtab th { color:#bbb; font-size:0.6rem; text-transform:uppercase;
             letter-spacing:0.07em; font-weight:600; padding:2px 6px; }
  .dtab td { padding: 4px 6px; border-bottom: 1px solid #f4f4f4; }
  .up   { color: #1e7e34 !important; font-weight: 600; }
  .down { color: #c0392b !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Chart helpers ──────────────────────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="monospace", size=10, color="#333"),
    margin=dict(l=42, r=8, t=24, b=28),
    xaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd"),
    yaxis=dict(gridcolor="#f0f0f0", linecolor="#ddd"),
    legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)",
                orientation="h", y=-0.20),
    height=240, hovermode="x unified",
)

YIELD_MAP = {
    "1 Mo": 1/12, "2 Mo": 2/12, "3 Mo": 3/12, "4 Mo": 4/12,
    "6 Mo": 6/12, "1 Yr": 1, "2 Yr": 2, "3 Yr": 3, "5 Yr": 5,
    "7 Yr": 7, "10 Yr": 10, "20 Yr": 20, "30 Yr": 30,
}


def lfig(df: pd.DataFrame, title: str = "", norm: bool = False,
         height: int = 240, yaxis_title: str = "") -> go.Figure:
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
        layout["title"] = dict(text=title, font=dict(size=10, color="#777"))
    if yaxis_title:
        layout["yaxis"] = {**layout.get("yaxis", {}), "title": yaxis_title}
    fig.update_layout(**layout)
    return fig


def curve_snapshot(curve: pd.Series) -> go.Figure:
    pts = [(YIELD_MAP[k], float(curve[k])) for k in YIELD_MAP
           if k in curve.index and not pd.isna(curve[k])]
    if not pts:
        return go.Figure()
    x, y = zip(*pts)
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
        "hovermode": "x", "height": 230,
    })
    return fig


def curve_history(hist: pd.DataFrame) -> go.Figure:
    cols = [k for k in YIELD_MAP if k in hist.columns]
    x_vals = [YIELD_MAP[k] for k in cols]
    n = len(hist)
    snaps = [(lbl, idx) for lbl, idx in
             [("Today", -1), ("1M ago", -22), ("3M ago", -63), ("1Y ago", -252)]
             if abs(idx) <= n]
    clrs = ["#2c3e50", "#7f8c8d", "#aab7c4", "#d5d8dc"]
    widths = [2.5, 1.5, 1.2, 1.0]
    fig = go.Figure()
    for i, (lbl, idx) in enumerate(snaps):
        row = hist.iloc[idx]
        y = [float(row[c]) for c in cols if not pd.isna(row.get(c, np.nan))]
        fig.add_trace(go.Scatter(
            x=x_vals[:len(y)], y=y, name=lbl,
            line=dict(color=clrs[i], width=widths[i]),
            mode="lines+markers", marker=dict(size=4),
        ))
    fig.update_layout(**{**_LAYOUT,
        "xaxis": dict(title="Maturity (years)",
                      tickvals=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
                      ticktext=["3M","6M","1Y","2Y","3Y","5Y","7Y","10Y","20Y","30Y"],
                      gridcolor="#f0f0f0", linecolor="#ddd"),
        "yaxis": dict(title="Yield (%)", gridcolor="#f0f0f0", linecolor="#ddd"),
        "height": 230,
    })
    return fig


def _metric_row(items: list[tuple[str, str, str | None]]):
    """Render a compact metric row. items = [(label, value, delta_str), ...]"""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


def _fmt(v, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) > 10_000: return f"{v:,.0f}"
    if abs(v) > 1_000:  return f"{v:,.1f}"
    return f"{v:.{dec}f}"


def _pct(v, dec=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{dec}f}%"


# ── Insight rendering ──────────────────────────────────────────────────────────
def render_insight(ins: dict):
    level  = ins["level"]
    color  = LEVEL_COLOR[level]
    bg     = LEVEL_BG[level]
    icon   = LEVEL_ICON[level]
    watch  = " &nbsp;·&nbsp; ".join(ins.get("watch", []))
    st.markdown(f"""
    <div class="insight-card" style="background:{bg};border-left-color:{color};">
      <div>
        <span class="insight-badge" style="background:{color};">{level}</span>
        <span class="insight-cat">{ins['category']}</span>
      </div>
      <div class="insight-head">{icon}&nbsp;{ins['headline']}</div>
      <div class="insight-detail">{ins['detail']}</div>
      <div class="insight-watch"><strong>Watch:</strong> {watch}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Stress gauge ───────────────────────────────────────────────────────────────
def render_stress_gauge(total: int):
    if total <= 1:
        color, label = "#1e7e34", "LOW"
    elif total <= 3:
        color, label = "#d35400", "MODERATE"
    elif total <= 5:
        color, label = "#c0392b", "ELEVATED"
    else:
        color, label = "#7b241c", "HIGH"
    st.markdown(
        f'<span class="stress-pill" style="background:{color};color:white;">'
        f'GLOBAL STRESS: {label} ({total}/8)</span>',
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    now = datetime.utcnow()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Settings")
        period = st.selectbox("Market data period", ["3mo", "6mo", "1y", "2y"], index=2)
        st.markdown("---")
        st.caption(
            "**Data sources**\n"
            "- yfinance — market prices\n"
            "- US Treasury Direct — yield curve\n"
            "- Synthetic — GPR, shipping, EM spreads,\n"
            "  FX reserves, minerals, FDI, defense\n\n"
            "Prices: 15 min cache · Yields: 1 hr · Structural: 24 hr"
        )
        if st.button("Clear cache & refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Load all data ─────────────────────────────────────────────────────────
    with st.spinner("Loading data…"):
        mkt       = df_.get_market_data(period=period)
        curve_lat, curve_hist = df_.get_yield_curve()
        gpr       = df_.get_gpr()
        shipping  = df_.get_shipping_stress()
        em_sp     = df_.get_em_spreads()
        fx_res    = df_.get_fx_reserves()
        minerals  = df_.get_critical_minerals()
        defense   = df_.get_defense_spending()
        fdi       = df_.get_fdi_flows()

    ind = ind_.compute_all(mkt, curve_hist, gpr, shipping, em_sp, fx_res)
    insights = re_.run(ind)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:4px;">'
        f'<span style="font-size:1.1rem;font-weight:700;font-family:monospace;'
        f'letter-spacing:0.05em;">MACRO INTELLIGENCE SYSTEM</span>'
        f'<span style="font-size:0.68rem;color:#bbb;font-family:monospace;">'
        f'{now.strftime("%Y-%m-%d  %H:%M UTC")}</span></div>',
        unsafe_allow_html=True,
    )

    # Quick pulse row: 7 key numbers
    def last_chg(col):
        s = mkt[col].dropna() if col in mkt.columns else pd.Series(dtype=float)
        if len(s) < 2: return (_fmt(s.iloc[-1]) if len(s) else "—", None)
        price = s.iloc[-1]
        chg = (price / s.iloc[-2] - 1) * 100
        sign = "+" if chg >= 0 else ""
        return _fmt(price), f"{sign}{chg:.2f}%"

    pulse = [
        ("S&P 500",  *last_chg("spx")),
        ("10Y UST",  *last_chg("us10y")),
        ("DXY",      *last_chg("dxy")),
        ("WTI",      *last_chg("wti")),
        ("Gold",     *last_chg("gold")),
        ("VIX",      *last_chg("vix")),
        ("MSCI EM",  *last_chg("eem")),
    ]
    _metric_row(pulse)
    st.markdown("---")

    # ── Regime Assessment ─────────────────────────────────────────────────────
    render_stress_gauge(ind.get("total_stress_score", 0))
    st.markdown(
        '<div class="section-label" style="margin-top:8px;">Regime Assessment — Rules Engine Output</div>',
        unsafe_allow_html=True,
    )

    alerts   = [i for i in insights if i["level"] == "alert"]
    warnings = [i for i in insights if i["level"] == "warning"]
    infos    = [i for i in insights if i["level"] == "info"]

    # Display alerts full-width, then warn+info in 2 cols
    for ins in alerts:
        render_insight(ins)

    if warnings or infos:
        c1, c2 = st.columns(2)
        with c1:
            for ins in warnings:
                render_insight(ins)
        with c2:
            for ins in infos:
                render_insight(ins)

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📈 Macro Regime",
        "🌍 Geo-Risk",
        "💰 Capital Flows",
        "🌐 EM & Africa",
        "🔮 Market-Implied",
    ])

    # ════════ Tab 1: Macro Regime ══════════════════════════════════════════════
    with tabs[0]:
        # Key metrics
        _metric_row([
            ("DXY",         _fmt(ind.get("dxy")),     _pct(ind.get("dxy_1m_chg"))),
            ("10Y Yield",   f"{ind.get('us10y', 0):.2f}%", None),
            ("2Y Yield",    f"{ind.get('us2y', 0):.2f}%", None),
            ("Curve Slope", f"{ind.get('curve_slope', 0):+.0f} bps",
             ind.get("curve_direction", "")),
            ("VIX",         _fmt(ind.get("vix")),     _pct(ind.get("vix_1m_chg"))),
            ("Oil (Brent)", _fmt(ind.get("oil")),     _pct(ind.get("oil_1m_chg"))),
            ("Copper",      _fmt(ind.get("copper"), 3), _pct(ind.get("copper_1m_chg"))),
        ])

        ca, cb = st.columns(2)
        with ca:
            st.markdown('<div class="section-label">Yield Curve — Current</div>',
                        unsafe_allow_html=True)
            if not curve_lat.empty:
                st.plotly_chart(curve_snapshot(curve_lat), use_container_width=True)
                s10 = ind.get("us10y", np.nan)
                s2  = ind.get("us2y",  np.nan)
                s3m = ind.get("us3m",  np.nan)
                mc1, mc2, mc3 = st.columns(3)
                def sp(a, b): return f"{(a-b)*100:+.0f} bps" if not (np.isnan(a) or np.isnan(b)) else "—"
                mc1.metric("10Y–2Y",  sp(s10, s2))
                mc2.metric("10Y–3M",  sp(s10, s3m))
                mc3.metric("Regime",  ind.get("curve_regime", "—").upper())
        with cb:
            st.markdown('<div class="section-label">Curve Shifts (Today vs 1M/3M/1Y ago)</div>',
                        unsafe_allow_html=True)
            if not curve_hist.empty:
                st.plotly_chart(curve_history(curve_hist), use_container_width=True)

        st.markdown('<div class="section-label">10Y−2Y Slope History</div>',
                    unsafe_allow_html=True)
        slope_s = ind.get("curve_slope_series")
        if slope_s is not None and not slope_s.empty:
            sfig = lfig(slope_s.to_frame("10Y−2Y slope"), height=200, yaxis_title="bps")
            sfig.add_hline(y=0, line_dash="dash", line_color="#c0392b",
                           annotation_text="0 = inversion", annotation_font_size=9)
            st.plotly_chart(sfig, use_container_width=True)

        cc, cd = st.columns(2)
        with cc:
            st.markdown('<div class="section-label">VIX — 1 Year</div>',
                        unsafe_allow_html=True)
            if "vix" in mkt.columns:
                vfig = lfig(mkt[["vix"]], height=200)
                vfig.add_hline(y=25, line_dash="dot", line_color="#d35400",
                               annotation_text="25 elevated", annotation_font_size=9)
                vfig.add_hline(y=35, line_dash="dot", line_color="#c0392b",
                               annotation_text="35 high", annotation_font_size=9)
                st.plotly_chart(vfig, use_container_width=True)
        with cd:
            st.markdown('<div class="section-label">Commodities — rebased to 100</div>',
                        unsafe_allow_html=True)
            com_cols = [c for c in ["brent", "copper", "gold"] if c in mkt.columns]
            if com_cols:
                st.plotly_chart(lfig(mkt[com_cols], norm=True, height=200),
                                use_container_width=True)

    # ════════ Tab 2: Geo-Risk ══════════════════════════════════════════════════
    with tabs[1]:
        _metric_row([
            ("GPR Index",    f"{ind.get('gpr', 0):.0f}",
             _pct(ind.get("gpr_mom_chg"))),
            ("GPR Regime",   ind.get("gpr_regime", "—").upper(), None),
            ("Shipping",     f"{ind.get('shipping', 0):,.0f}",
             _pct(ind.get("shipping_1m_chg"))),
            ("Ship Regime",  ind.get("shipping_regime", "—").upper(), None),
            ("Oil 1M",       _pct(ind.get("oil_1m_chg")), None),
            ("Geo Stress",   f"{ind.get('geo_stress_score', 0)}/3", None),
        ])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Geopolitical Risk Index (GPR)</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                lfig(gpr.to_frame("GPR"), yaxis_title="Index (baseline=100)"),
                use_container_width=True)
            st.caption(
                "GPR (Caldara & Iacoviello, 2022): measures geopolitical risk from newspaper coverage. "
                "Baseline = 100. Values >130 historically correlate with lower investment, "
                "higher energy prices, and defense spending acceleration."
            )
        with c2:
            st.markdown('<div class="section-label">Shipping Stress Index</div>',
                        unsafe_allow_html=True)
            sfig = lfig(shipping.to_frame("Shipping"), yaxis_title="Index")
            sfig.add_hline(y=2500, line_dash="dot", line_color="#d35400",
                           annotation_text="Stress", annotation_font_size=9)
            sfig.add_hline(y=4000, line_dash="dot", line_color="#c0392b",
                           annotation_text="Crisis", annotation_font_size=9)
            st.plotly_chart(sfig, use_container_width=True)
            st.caption(
                "Proxy for Baltic Dry Index / container rate composite. "
                "Shipping stress signals supply-chain disruption, "
                "typically triggered by chokepoint crises (Red Sea, Hormuz, Panama)."
            )

        st.markdown('<div class="section-label">Critical Minerals — Price Index (Jan 2020 = 100)</div>',
                    unsafe_allow_html=True)
        if not minerals.empty:
            st.plotly_chart(lfig(minerals, norm=False, height=230), use_container_width=True)
        st.caption(
            "Critical minerals are the structural backbone of the energy transition and defense supply chains. "
            "Lithium & cobalt → EV batteries. Copper → electrification. "
            "Rare earths → defense, wind turbines, EVs. "
            "Price surges signal upstream supply bottlenecks or demand acceleration."
        )

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="section-label">Defense Spending (% of GDP)</div>',
                        unsafe_allow_html=True)
            if not defense.empty:
                st.plotly_chart(lfig(defense, yaxis_title="% GDP", height=220),
                                use_container_width=True)
        with c4:
            st.markdown('<div class="section-label">Oil vs Shipping — rebased to 100</div>',
                        unsafe_allow_html=True)
            combined = pd.DataFrame({
                "Oil (Brent)": mkt["brent"] if "brent" in mkt.columns else pd.Series(dtype=float),
                "Shipping":    shipping.resample("B").last().reindex(mkt.index, method="ffill"),
            }).dropna()
            if not combined.empty:
                st.plotly_chart(lfig(combined, norm=True, height=220),
                                use_container_width=True)

    # ════════ Tab 3: Capital Flows & Structural ════════════════════════════════
    with tabs[2]:
        st.markdown('<div class="section-label">FDI Inflows by Region (USD bn)</div>',
                    unsafe_allow_html=True)
        if not fdi.empty:
            st.plotly_chart(lfig(fdi, height=240, yaxis_title="USD bn"),
                            use_container_width=True)
        st.caption(
            "FDI is the most durable form of capital flow — it signals where production "
            "is being built for the next decade. South/SE Asia leads on manufacturing relocation. "
            "Sub-Saharan Africa benefits from commodity FDI but needs to deepen manufacturing linkages."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Defense Budgets — Long View</div>',
                        unsafe_allow_html=True)
            if not defense.empty:
                st.plotly_chart(lfig(defense, yaxis_title="% GDP"), use_container_width=True)
            st.caption(
                "NATO Europe's defense spending is at a 30-year high and rising. "
                "Russia's budget is approaching wartime levels. "
                "Defense spending shifts reshape industrial policy, R&D allocation, and critical mineral demand."
            )
        with c2:
            st.markdown('<div class="section-label">Critical Minerals — Strategic Exposure</div>',
                        unsafe_allow_html=True)
            if not minerals.empty:
                st.plotly_chart(lfig(minerals, norm=False), use_container_width=True)

        # Structural themes table
        st.markdown('<div class="section-label">Structural Foresight — Power Map</div>',
                    unsafe_allow_html=True)
        themes = [
            ("Energy Transition", "Solar, wind, SMR, hydrogen — driving copper, lithium, cobalt demand through 2040",
             "DRC, Chile, Indonesia, Morocco"),
            ("Industrial Relocation", "Supply chain diversification from China — Vietnam, India, Mexico, Morocco leading",
             "Vietnam, India, Mexico, Morocco"),
            ("Digital Sovereignty", "US/EU/China splitting cloud, AI, semiconductor supply chains",
             "US, Taiwan, Netherlands (ASML), India"),
            ("Defense Architecture", "NATO 2% GDP target, AUKUS, Gulf rearmament — structural spending shift",
             "US, EU, Gulf, Australia"),
            ("Debt Diplomacy", "China BRI 2.0, Gulf SWFs, IMF restructuring — reshaping EM financing",
             "Africa, South Asia, LatAm"),
            ("Food Security", "Wheat corridor disruption, fertilizer supply, climate crop risk",
             "Ukraine, Russia, Morocco (OCP), Saudi Arabia"),
            ("Water Stress", "Aquifer depletion, drought risk intensifying in MENA, SSA, South Asia",
             "Egypt, India, Morocco, LatAm"),
        ]
        h = ('<table class="dtab"><tr>'
             '<th style="text-align:left;">Theme</th>'
             '<th style="text-align:left;">Dynamics</th>'
             '<th style="text-align:left;">Key geographies</th></tr>')
        for t, d, g in themes:
            h += f'<tr><td><strong>{t}</strong></td><td>{d}</td><td style="color:#888;">{g}</td></tr>'
        h += "</table>"
        st.markdown(h, unsafe_allow_html=True)

    # ════════ Tab 4: EM & Africa ═══════════════════════════════════════════════
    with tabs[3]:
        embi_val    = ind.get("embi", np.nan)
        africa_val  = ind.get("africa_spreads", np.nan)
        fx_min      = ind.get("fx_res_min_chg", np.nan)
        worst       = ind.get("fx_res_worst_country", "")

        _metric_row([
            ("EMBI Global",     f"{embi_val:.0f} bps" if not np.isnan(embi_val) else "—",
             "widening ↑" if ind.get("em_spreads_widening") else "stable"),
            ("Africa Spread",   f"{africa_val:.0f} bps" if not np.isnan(africa_val) else "—", None),
            ("EM Regime",       ind.get("em_regime", "—").upper(), None),
            ("FX Reserves",     "Stress" if ind.get("fx_res_deteriorating") else "Stable",
             f"Worst: {worst}" if worst else None),
            ("Dollar Regime",   ind.get("dollar_regime", "—").upper(), None),
        ])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">EM Sovereign Spreads (bps)</div>',
                        unsafe_allow_html=True)
            if not em_sp.empty:
                sfig = lfig(em_sp, yaxis_title="bps")
                sfig.add_hline(y=450, line_dash="dot", line_color="#d35400",
                               annotation_text="Stress", annotation_font_size=9)
                sfig.add_hline(y=600, line_dash="dot", line_color="#c0392b",
                               annotation_text="Crisis", annotation_font_size=9)
                st.plotly_chart(sfig, use_container_width=True)
        with c2:
            st.markdown('<div class="section-label">African FX Reserves (USD bn)</div>',
                        unsafe_allow_html=True)
            if not fx_res.empty:
                st.plotly_chart(
                    lfig(fx_res, yaxis_title="USD bn"),
                    use_container_width=True)

        st.markdown('<div class="section-label">EM FX vs USD — rebased to 100 (↑ = weaker EM)</div>',
                    unsafe_allow_html=True)
        em_fx_cols = [c for c in ["usdbrl","usdzar","usdtry","usdcnh","usdjpy"] if c in mkt.columns]
        labels = {"usdbrl": "USD/BRL", "usdzar": "USD/ZAR", "usdtry": "USD/TRY",
                  "usdcnh": "USD/CNH", "usdjpy": "USD/JPY"}
        if em_fx_cols:
            em_fx_df = mkt[em_fx_cols].rename(columns=labels)
            st.plotly_chart(lfig(em_fx_df, norm=True, height=220), use_container_width=True)

        # Debt vulnerability snapshot
        st.markdown('<div class="section-label">Africa Sovereign Debt Stress — Snapshot</div>',
                    unsafe_allow_html=True)
        debt_data = [
            ("Ghana",     "Post-restructuring", 850, "IMF program (2023)", "High"),
            ("Ethiopia",  "G20 restructuring",  1100,"Pending resolution",  "Very High"),
            ("Kenya",     "Eurobond maturity",   650, "$2bn due 2024 (rolled)", "High"),
            ("Egypt",     "IMF program",         700, "Extended 2024", "High"),
            ("Nigeria",   "Manageable",          420, "Tight FX reserves",   "Moderate"),
            ("Zambia",    "Post-restructuring",  600, "G20 restructured 2023", "High"),
            ("Angola",    "Oil-backed debt",     480, "China bilateral", "Moderate"),
        ]
        h = ('<table class="dtab"><tr>'
             '<th>Country</th><th style="text-align:right;">Spread (bps)</th>'
             '<th>Status</th><th>Note</th><th>Risk</th></tr>')
        risk_color = {"Very High": "#c0392b", "High": "#d35400",
                      "Moderate": "#7f8c8d", "Low": "#1e7e34"}
        for country, status, spread, note, risk in debt_data:
            color = risk_color.get(risk, "#333")
            h += (f'<tr><td><strong>{country}</strong></td>'
                  f'<td style="text-align:right;font-family:monospace;">{spread}</td>'
                  f'<td>{status}</td><td style="color:#888;">{note}</td>'
                  f'<td><span style="color:{color};font-weight:700;">{risk}</span></td></tr>')
        h += "</table>"
        st.markdown(h, unsafe_allow_html=True)

    # ════════ Tab 5: Market-Implied ════════════════════════════════════════════
    with tabs[4]:
        st.markdown(
            "**What does the future cost today?** Market prices are the best real-time consensus "
            "forecast available — imperfect but continuous.",
            unsafe_allow_html=False,
        )

        _metric_row([
            ("10Y Yield",       f"{ind.get('us10y', 0):.2f}%", None),
            ("Curve Slope",     f"{ind.get('curve_slope', 0):+.0f} bps",
             ind.get("curve_direction", "")),
            ("HY Credit (1M)",  _pct(ind.get("hyg_1m_chg")), None),
            ("EM Bonds (1M)",   _pct(ind.get("emb_1m_chg")), None),
            ("Gold 1M",         _pct(ind.get("gold_1m_chg")), None),
            ("VIX",             _fmt(ind.get("vix")),         None),
        ])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-label">Treasury Yields — 1Y History</div>',
                        unsafe_allow_html=True)
            rate_cols = [c for c in ["us3m", "us5y", "us10y", "us30y"] if c in mkt.columns]
            rate_labels = {"us3m": "3M", "us5y": "5Y", "us10y": "10Y", "us30y": "30Y"}
            if rate_cols:
                st.plotly_chart(
                    lfig(mkt[rate_cols].rename(columns=rate_labels), yaxis_title="%"),
                    use_container_width=True)
        with c2:
            st.markdown('<div class="section-label">Credit & EM Bond ETFs — rebased to 100</div>',
                        unsafe_allow_html=True)
            etf_cols = [c for c in ["hyg", "emb", "tip"] if c in mkt.columns]
            etf_labels = {"hyg": "HYG (HY credit)", "emb": "EMB (EM bonds)", "tip": "TIP (TIPS/inflation)"}
            if etf_cols:
                st.plotly_chart(
                    lfig(mkt[etf_cols].rename(columns=etf_labels), norm=True),
                    use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="section-label">Gold vs DXY — divergence = stress signal</div>',
                        unsafe_allow_html=True)
            gd_cols = {k: v for k, v in {"gold": "Gold", "dxy": "DXY"}.items() if k in mkt.columns}
            if gd_cols:
                st.plotly_chart(
                    lfig(mkt[list(gd_cols.keys())].rename(columns=gd_cols), norm=True),
                    use_container_width=True)
            st.caption(
                "Gold and DXY normally move inversely. When both rise together, "
                "it signals a flight to *all* safe havens — a stress regime. "
                "When gold rises and DXY falls, it often indicates structural de-dollarization demand."
            )
        with c4:
            st.markdown('<div class="section-label">MSCI EM vs SPX — relative performance</div>',
                        unsafe_allow_html=True)
            eq_cols = {k: v for k, v in {"spx": "S&P 500", "eem": "MSCI EM"}.items() if k in mkt.columns}
            if eq_cols:
                st.plotly_chart(
                    lfig(mkt[list(eq_cols.keys())].rename(columns=eq_cols), norm=True),
                    use_container_width=True)
            st.caption(
                "EM underperformance vs SPX is often driven by: USD strength, rate differentials, "
                "commodity price weakness, or geopolitical risk. "
                "Persistent EM lag signals capital repatriation to DM safe havens."
            )

        # Forward-looking framework table
        st.markdown('<div class="section-label">Market-Implied Signals — Interpretation Guide</div>',
                    unsafe_allow_html=True)
        signals_table = [
            ("Curve inverting",       "Rate-cut expectations rising OR growth fears → recession watch"),
            ("VIX > 25",              "Tail-risk hedging → risk-off positioning, USD bid"),
            ("HYG falling",           "Credit conditions tightening → business cycle late stage"),
            ("EMB falling + DXY ↑",   "EM stress → watch IMF program requests, Eurobond spreads"),
            ("Gold + DXY rising",      "Flight to ALL safe havens → acute systemic stress"),
            ("Copper rising",         "Growth recovery priced → industrial cycle turning up"),
            ("Oil > $100 (backwarded)","Supply shock risk → stagflation scenario gaining probability"),
            ("TIP falling",           "Real rates rising → tighter financial conditions globally"),
        ]
        h = ('<table class="dtab"><tr>'
             '<th style="text-align:left;">Signal</th>'
             '<th style="text-align:left;">What markets are saying</th></tr>')
        for sig, meaning in signals_table:
            h += f'<tr><td style="font-family:monospace;">{sig}</td><td>{meaning}</td></tr>'
        h += "</table>"
        st.markdown(h, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    fc1, fc2 = st.columns([1, 5])
    with fc1:
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()
    with fc2:
        st.markdown(
            '<span style="font-size:0.68rem;color:#ccc;">'
            'Market data: yfinance (15 min) · Yields: US Treasury Direct (1 hr) · '
            'GPR / Shipping / EM / Minerals: synthetic (illustrative)</span>',
            unsafe_allow_html=True,
        )


main()
