"""
app.py — Macro Intelligence Dashboard
======================================
Streamlit application wiring together all modules:
  data_fetchers → indicators → rules_engine → inference_engine → narrative_generator

Run:
    streamlit run macro_intel/app.py
    # or from repo root:
    streamlit run macro_intel/app.py --server.port 8501
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
import data_fetchers as df_mod
import indicators as ind_mod
import inference_engine as inf_eng
import narrative_generator as narr
import rules_engine as rules_eng

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Intelligence — Africa & EM",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .insight-alert   { background:#fdf3f2; border-left:4px solid #c0392b; padding:10px 14px; border-radius:6px; margin:6px 0; }
  .insight-warning { background:#fef9f0; border-left:4px solid #d35400; padding:10px 14px; border-radius:6px; margin:6px 0; }
  .insight-info    { background:#f0f7fd; border-left:4px solid #2980b9; padding:10px 14px; border-radius:6px; margin:6px 0; }
  .inference-high   { background:#fdf3f2; border-left:4px solid #c0392b; padding:8px 12px; border-radius:4px; margin:4px 0; }
  .inference-medium { background:#fef9f0; border-left:4px solid #d35400; padding:8px 12px; border-radius:4px; margin:4px 0; }
  .inference-low    { background:#f4f4f4; border-left:4px solid #7f8c8d; padding:8px 12px; border-radius:4px; margin:4px 0; }
  .source-badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.72rem; font-weight:600; margin:2px; }
  .source-live  { background:#d5f5e3; color:#1a5e2a; }
  .source-mock  { background:#fdebd0; color:#7d4b0a; }
  .regime-crisis   { background:#fdf3f2; color:#c0392b; padding:8px 14px; border-radius:6px; font-weight:700; }
  .regime-stressed { background:#fef9f0; color:#d35400; padding:8px 14px; border-radius:6px; font-weight:700; }
  .regime-cautious { background:#fefdf0; color:#b7950b; padding:8px 14px; border-radius:6px; font-weight:700; }
  .regime-benign   { background:#eafaf1; color:#1a5e2a; padding:8px 14px; border-radius:6px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _source_badge(label: str) -> str:
    is_mock = "synthetic" in label.lower() or "mock" in label.lower()
    css = "source-mock" if is_mock else "source-live"
    return f'<span class="source-badge {css}">{label}</span>'


def _regime_html(regime: str) -> str:
    emoji = {"crisis": "🔴", "stressed": "🟠", "cautious": "🟡", "benign": "🟢"}.get(regime, "⚪")
    return f'<div class="regime-{regime}">{emoji} Macro Regime: <b>{regime.upper()}</b></div>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    refresh_btn = st.button("🔄 Refresh Data", use_container_width=True)
    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("**FRED API Key** (optional)")
    fred_key_input = st.text_input(
        "FRED Key", value="", type="password",
        placeholder="Leave blank to use env/secrets",
        label_visibility="collapsed",
    )
    if fred_key_input:
        import os
        os.environ["FRED_API_KEY"] = fred_key_input

    st.divider()
    show_inference = st.toggle("Forward-Looking Inference", value=True)
    show_narrative = st.toggle("Weekly Narrative Note",     value=True)
    show_raw       = st.toggle("Raw Indicator Values",      value=False)

    st.divider()
    st.caption("**Africa Focus Countries**")
    africa_focus = st.multiselect(
        "Countries (ISO3)", options=cfg.WB_AFRICA,
        default=["NGA", "KEN", "EGY", "GHA", "ZAF"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"As of: {datetime.date.today().strftime('%d %b %Y')}")
    st.caption("Data: FRED · World Bank · IMF · Yahoo Finance")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def load_market():
    return df_mod.get_market_data()

@st.cache_data(ttl=3600)
def load_yields():
    return df_mod.get_yields()

@st.cache_data(ttl=3600)
def load_fred():
    return df_mod.get_fred_macro()

@st.cache_data(ttl=86400)
def load_gpr():
    return df_mod.get_gpr()

@st.cache_data(ttl=3600)
def load_shipping():
    return df_mod.get_shipping()

@st.cache_data(ttl=3600)
def load_em_spreads():
    return df_mod.get_em_spreads()

@st.cache_data(ttl=86400)
def load_wb_reserves():
    return df_mod.get_worldbank_reserves()

@st.cache_data(ttl=86400)
def load_wb_fdi():
    return df_mod.get_worldbank_fdi()

@st.cache_data(ttl=86400)
def load_imf():
    return df_mod.get_imf_macro()

@st.cache_data(ttl=3600)
def load_sanctions():
    return df_mod.get_sanctions_intensity()

@st.cache_data(ttl=3600)
def load_cyber():
    return df_mod.get_cyber_risk()

@st.cache_data(ttl=86400)
def load_minerals():
    return df_mod.get_critical_minerals()


# ── Load all data ─────────────────────────────────────────────────────────────
with st.spinner("Loading macro data…"):
    mkt,        mkt_src  = load_market()
    yields,     yld_src  = load_yields()
    fred,       fred_src = load_fred()
    gpr_df,     gpr_src  = load_gpr()
    ship_df,    ship_src = load_shipping()
    em_spreads, em_src   = load_em_spreads()
    fx_res,     fx_src   = load_wb_reserves()
    fdi_df,     fdi_src  = load_wb_fdi()
    imf_df,     imf_src  = load_imf()
    sanctions,  san_src  = load_sanctions()
    cyber,      cyb_src  = load_cyber()
    minerals,   min_src  = load_minerals()

ind = ind_mod.compute_all(
    mkt=mkt, yields=yields, fred=fred,
    gpr=gpr_df, ship=ship_df, sanctions=sanctions, cyber=cyber,
    em_spreads=em_spreads, fx_reserves=fx_res, minerals=minerals,
)
ind.update({
    "curve_source": yld_src, "vix_source": mkt_src, "dxy_source": mkt_src,
    "oil_source": mkt_src, "copper_source": mkt_src,
    "gpr_source": gpr_src, "em_spread_source": em_src, "fx_res_source": fx_src,
})

rules      = rules_eng.run(ind)
inferences = inf_eng.run(ind, rules)
note       = narr.generate(ind, rules, inferences)


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_regime = st.columns([3, 1])
with col_title:
    st.title("🌍 Macro Intelligence — Africa & EM")
    st.caption("Forward-looking macro risk · Global indicators · Africa sovereign analysis")
with col_regime:
    st.markdown(_regime_html(ind.get("macro_regime", "benign")), unsafe_allow_html=True)
    st.caption(
        f"Financial stress: **{ind.get('financial_stress_score', 0)}/7** · "
        f"Geo-risk: **{ind.get('geo_stress_score', 0)}/4**"
    )

all_sources = {mkt_src, yld_src, fred_src, gpr_src, em_src, fx_src}
badges_html = " ".join(_source_badge(s) for s in sorted(all_sources) if s)
st.markdown(f"**Data:** {badges_html}", unsafe_allow_html=True)
st.divider()


# ── Pulse bar (KPIs) ──────────────────────────────────────────────────────────
def _metric_col(col, label, val, fmt=".1f", suffix=""):
    v_str = f"{val:{fmt}}{suffix}" if not _nan(val) else "N/A"
    col.metric(label, v_str)


kpi = st.columns(8)
_metric_col(kpi[0], "VIX",       ind.get("vix",          np.nan))
_metric_col(kpi[1], "DXY",       ind.get("dxy",          np.nan))
_metric_col(kpi[2], "Curve",     ind.get("curve_slope",  np.nan), fmt="+.0f", suffix=" bps")
_metric_col(kpi[3], "EMBI",      ind.get("embi",         np.nan), fmt=".0f",  suffix=" bps")
_metric_col(kpi[4], "HY Spread", ind.get("hy_spread",    np.nan), fmt=".0f",  suffix=" bps")
oil_chg = ind.get("oil_1m_chg", np.nan)
cu_chg  = ind.get("copper_1m_chg", np.nan)
gd_chg  = ind.get("gold_1m_chg", np.nan)
_metric_col(kpi[5], "Oil 1M",    oil_chg * 100 if not _nan(oil_chg) else np.nan, fmt="+.1f", suffix="%")
_metric_col(kpi[6], "Copper 1M", cu_chg  * 100 if not _nan(cu_chg)  else np.nan, fmt="+.1f", suffix="%")
_metric_col(kpi[7], "Gold 1M",   gd_chg  * 100 if not _nan(gd_chg)  else np.nan, fmt="+.1f", suffix="%")
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🚨 Signals",
    "📈 Rates & Curves",
    "💵 FX & Dollar",
    "🛢️ Commodities",
    "🌍 EM & Africa",
    "🔮 Inference",
    "📰 Weekly Note",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Cross-Asset Signals")

    for r in rules:
        lvl = r["level"]
        icon = cfg.LEVEL_ICON.get(lvl, "")
        with st.container():
            st.markdown(
                f'<div class="insight-{lvl}">'
                f'<b>{icon} [{lvl.upper()}] {r["category"]}</b>: {r["headline"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Detail & watch items"):
                st.write(r["detail"])
                st.markdown("**Watch:**")
                for w in r.get("watch", []):
                    st.markdown(f"- {w}")

    if not rules:
        st.success("No stress signals detected.")

    st.divider()
    col_fs, col_gs = st.columns(2)
    with col_fs:
        fin_score = ind.get("financial_stress_score", 0)
        st.markdown("**Financial Stress Score**")
        st.progress(fin_score / 7, text=f"{fin_score}/7 indicators elevated")

    with col_gs:
        geo_score = ind.get("geo_stress_score", 0)
        st.markdown("**Geopolitical Stress Score**")
        st.progress(geo_score / 4, text=f"{geo_score}/4 indicators elevated")

    if show_raw:
        st.divider()
        st.subheader("Raw Indicator Values")
        st.dataframe(
            pd.DataFrame([{"Indicator": k, "Value": str(v)} for k, v in sorted(ind.items())]),
            use_container_width=True, height=400,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RATES & CURVES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Yield Curve & Rates")

    if yields is not None and not yields.empty:
        tenor_map = {"3M": "us3m", "2Y": "us2y", "5Y": "us5y", "10Y": "us10y", "30Y": "us30y"}
        curve_pts = {
            label: yields[col].dropna().iloc[-1]
            for label, col in tenor_map.items()
            if col in yields.columns and not yields[col].dropna().empty
        }
        if curve_pts:
            slope = ind.get("curve_slope", np.nan)
            curve_rg = ind.get("curve_regime", "")
            title_sfx = f" — {curve_rg} ({slope:+.0f} bps)" if not _nan(slope) else ""
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=list(curve_pts.keys()), y=list(curve_pts.values()),
                mode="lines+markers", line=dict(color="#2c3e50", width=3), marker=dict(size=10),
            ))
            fig_curve.update_layout(title=f"US Treasury Yield Curve{title_sfx}",
                                    xaxis_title="Tenor", yaxis_title="Yield (%)", height=360)
            st.plotly_chart(fig_curve, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if "us10y" in yields.columns and "us2y" in yields.columns:
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=yields.index, y=yields["us10y"], name="10Y", line=dict(color="#2980b9")))
                fig_ts.add_trace(go.Scatter(x=yields.index, y=yields["us2y"],  name="2Y",  line=dict(color="#c0392b")))
                fig_ts.update_layout(title="2Y vs 10Y Yield", yaxis_title="%", height=280)
                st.plotly_chart(fig_ts, use_container_width=True)

        with col_b:
            if "us10y" in yields.columns and "us2y" in yields.columns:
                spread = (yields["us10y"] - yields["us2y"]) * 100
                fig_sp = go.Figure()
                fig_sp.add_trace(go.Bar(
                    x=spread.index, y=spread,
                    marker_color=["#c0392b" if v < 0 else "#27ae60" for v in spread],
                ))
                fig_sp.add_hline(y=0, line_dash="dash", line_color="black")
                fig_sp.update_layout(title="2Y–10Y Spread (bps)", yaxis_title="bps", height=280)
                st.plotly_chart(fig_sp, use_container_width=True)

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Key Rate Indicators**")
        tp   = ind.get("term_premium", np.nan)
        be5  = ind.get("breakeven5y", np.nan)
        be55 = ind.get("breakeven5y5y", np.nan)
        st.metric("Term Premium (10Y proxy)",  f"{tp:.2f}%"  if not _nan(tp)   else "N/A")
        st.metric("5Y Breakeven Inflation",    f"{be5:.2f}%" if not _nan(be5)  else "N/A")
        st.metric("5Y5Y Forward Inflation",    f"{be55:.2f}%" if not _nan(be55) else "N/A")

    with col_d:
        st.markdown("**Vol & Inflation Regime**")
        move   = ind.get("move_proxy", np.nan)
        inf_rg = ind.get("inflation_regime", "")
        vix_ts = ind.get("vix_term_structure", "N/A")
        st.metric("MOVE Proxy (bond vol)",  f"{move:.0f}" if not _nan(move) else "N/A")
        st.metric("Inflation Regime",       inf_rg.upper() if inf_rg else "N/A")
        st.metric("VIX Term Structure",     vix_ts.upper())

    if fred is not None and "nfci" in fred.columns:
        st.divider()
        nfci = fred["nfci"].dropna()
        fig_nfci = go.Figure()
        fig_nfci.add_trace(go.Bar(
            x=nfci.index, y=nfci,
            marker_color=["#c0392b" if v > 0.5 else "#27ae60" for v in nfci],
        ))
        fig_nfci.add_hline(y=0, line_dash="dash", line_color="black")
        fig_nfci.update_layout(title="Chicago Fed NFCI (>0 = tighter conditions)", height=250)
        st.plotly_chart(fig_nfci, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FX & DOLLAR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("FX & Dollar Regime")

    dxy_series = mkt.get("dxy") if mkt else None
    if dxy_series is not None and not dxy_series.empty:
        fig_dxy = go.Figure()
        fig_dxy.add_trace(go.Scatter(x=dxy_series.index, y=dxy_series, name="DXY",
                                     line=dict(color="#2c3e50", width=2)))
        fig_dxy.add_hline(y=cfg.THRESH["dxy_strong"],      line_dash="dot",
                          line_color="#d35400", annotation_text="Strong")
        fig_dxy.add_hline(y=cfg.THRESH["dxy_very_strong"], line_dash="dot",
                          line_color="#c0392b", annotation_text="Very Strong")
        fig_dxy.update_layout(title="US Dollar Index (DXY)", yaxis_title="Index", height=300)
        st.plotly_chart(fig_dxy, use_container_width=True)

    # EM FX table
    st.divider()
    fx_pairs = {
        "USD/BRL": "usdbrl", "USD/ZAR": "usdzar", "USD/TRY": "usdtry",
        "USD/CNH": "usdcnh", "USD/INR": "usdinr", "USD/MXN": "usdmxn",
    }
    fx_rows = []
    for label, key in fx_pairs.items():
        series = mkt.get(key) if mkt else None
        if series is not None and len(series) >= 22:
            last   = series.iloc[-1]
            chg_1m = last / series.iloc[-22] - 1
            chg_ytd = last / series.iloc[0] - 1
            fx_rows.append({
                "Pair": label, "Latest": f"{last:.2f}",
                "1M Chg": f"{chg_1m*100:+.1f}%", "YTD Chg": f"{chg_ytd*100:+.1f}%",
            })
    if fx_rows:
        st.markdown("**EM FX Performance**")
        st.dataframe(pd.DataFrame(fx_rows).set_index("Pair"), use_container_width=True)

    # FX Reserves
    st.divider()
    st.subheader("FX Reserve Coverage (World Bank)")
    if fx_res is not None and not fx_res.empty:
        fig_res = go.Figure()
        for iso in (africa_focus or cfg.WB_AFRICA[:5]):
            if iso in fx_res.columns:
                fig_res.add_trace(go.Scatter(
                    x=fx_res.index, y=fx_res[iso] / 1e9,
                    mode="lines+markers", name=iso,
                ))
        fig_res.update_layout(title="FX Reserves (USD bn)", yaxis_title="USD bn", height=300)
        st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("World Bank FX reserve data unavailable.")

    usd_vuln = ind.get("usd_debt_vulnerability", np.nan)
    if not _nan(usd_vuln):
        st.metric("USD Debt Vulnerability Score (EM broad)", f"{usd_vuln:.1f}/3")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMMODITIES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Commodities & Supply Chains")

    comm_map = {
        "Brent Crude (USD/bbl)":  "brent",
        "WTI Crude (USD/bbl)":    "wti",
        "Copper (HG contract)":   "copper",
        "Gold (USD/oz)":          "gold",
        "Natural Gas":            "natgas",
        "Wheat":                  "wheat",
    }
    col_charts = st.columns(2)
    for idx, (label, key) in enumerate(comm_map.items()):
        series = mkt.get(key) if mkt else None
        if series is None or series.empty:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series, name=label, line=dict(width=2)))
        fig.update_layout(title=label, height=250, margin=dict(l=10, r=10, t=40, b=10))
        col_charts[idx % 2].plotly_chart(fig, use_container_width=True)

    st.divider()
    col_cg, col_ship = st.columns(2)
    with col_cg:
        cg_regime = ind.get("copper_gold_regime", "")
        cg_ratio  = ind.get("copper_gold_ratio", np.nan)
        st.markdown("**Copper / Gold Ratio** *(growth vs safety)*")
        st.metric("Cu/Gold Regime", cg_regime.upper() if cg_regime else "N/A")
        st.metric("Cu/Gold Ratio",  f"{cg_ratio:.2f}" if not _nan(cg_ratio) else "N/A")
        st.caption("Rising ratio → risk-on. Falling → risk-off / fear.")

    with col_ship:
        ship_rg = ind.get("shipping_regime", "")
        st.markdown("**Shipping / Supply Chain**")
        st.metric("Shipping Regime", ship_rg.upper() if ship_rg else "N/A")
        if ship_df is not None and not ship_df.empty and "bdi" in ship_df.columns:
            fig_ship = go.Figure()
            fig_ship.add_trace(go.Scatter(x=ship_df.index, y=ship_df["bdi"], name="BDI"))
            fig_ship.add_hline(y=cfg.THRESH["shipping_stress"], line_dash="dot", line_color="#d35400")
            fig_ship.add_hline(y=cfg.THRESH["shipping_crisis"], line_dash="dot", line_color="#c0392b")
            fig_ship.update_layout(title="Baltic Dry Index", height=220)
            st.plotly_chart(fig_ship, use_container_width=True)

    st.divider()
    st.subheader("Critical Minerals (Africa)")
    if minerals is not None and not minerals.empty:
        st.dataframe(minerals.tail(5), use_container_width=True)
    else:
        st.info("Critical minerals data unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EM & AFRICA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Emerging Markets & Africa Sovereign")

    if em_spreads is not None and not em_spreads.empty:
        fig_em = go.Figure()
        for col in em_spreads.columns[:6]:
            fig_em.add_trace(go.Scatter(x=em_spreads.index, y=em_spreads[col], name=col))
        fig_em.add_hline(y=cfg.THRESH["em_spread_stress"], line_dash="dot",
                         line_color="#d35400", annotation_text="Stress")
        fig_em.add_hline(y=cfg.THRESH["em_spread_crisis"], line_dash="dot",
                         line_color="#c0392b", annotation_text="Crisis")
        fig_em.update_layout(title="EM Sovereign Spreads (EMBI proxy, bps)",
                             yaxis_title="bps", height=300)
        st.plotly_chart(fig_em, use_container_width=True)

    col_embi, col_africa = st.columns(2)
    with col_embi:
        embi   = ind.get("embi", np.nan)
        em_rg  = ind.get("em_regime", "")
        em_wide = ind.get("em_spreads_widening", False)
        st.metric("EMBI Spread", f"{embi:.0f} bps" if not _nan(embi) else "N/A",
                  delta="↑ widening" if em_wide else None)
        st.metric("EM Regime",   em_rg.upper() if em_rg else "N/A")

    with col_africa:
        af_sp  = ind.get("africa_spreads", np.nan)
        fx_det = ind.get("fx_res_deteriorating", False)
        worst  = ind.get("fx_res_worst_country", "")
        st.metric("Africa Composite Spread", f"{af_sp:.0f} bps" if not _nan(af_sp) else "N/A")
        if fx_det:
            st.warning(f"FX reserve drawdown: {worst or 'Multiple countries'}")

    st.divider()
    st.subheader("IMF Macro Indicators")
    if imf_df is not None and not imf_df.empty:
        st.caption(f"Source: {imf_src}")
        st.dataframe(imf_df.tail(5), use_container_width=True)
    else:
        st.info("IMF DataMapper data unavailable.")

    st.divider()
    st.subheader("FDI Inflows (World Bank)")
    if fdi_df is not None and not fdi_df.empty:
        fig_fdi = go.Figure()
        for iso in (africa_focus or cfg.WB_AFRICA[:5]):
            if iso in fdi_df.columns:
                fig_fdi.add_trace(go.Bar(name=iso, x=fdi_df.index, y=fdi_df[iso] / 1e9))
        fig_fdi.update_layout(barmode="group", title="FDI Net Inflows (USD bn)",
                              yaxis_title="USD bn", height=280)
        st.plotly_chart(fig_fdi, use_container_width=True)

    st.info(
        "📌 **Eurobond maturity wall 2025–2027**: Ghana, Kenya, Ethiopia, Egypt each face "
        "significant Eurobond redemptions. Monitor IMF program status, FX reserve cover, "
        "and Eurobond yield spreads."
    )

    col_san, col_cyb = st.columns(2)
    with col_san:
        san_elev = ind.get("sanctions_elevated", False)
        st.markdown("**Sanctions Regime**")
        if san_elev:
            st.warning("Elevated sanctions intensity detected")
        else:
            st.success("Sanctions intensity: normal")

    with col_cyb:
        st.markdown("**Cyber Risk**")
        if cyber is not None and not cyber.empty:
            st.dataframe(cyber.tail(3), use_container_width=True)
        else:
            st.info("Cyber risk data unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Forward-Looking Inference Engine")
    st.caption(
        "Conditional IF–THEN rules: trigger × context → forward statement. "
        "Sorted by confidence: high → medium → low."
    )

    if not show_inference:
        st.info("Enable 'Forward-Looking Inference' in the sidebar.")
    elif not inferences:
        st.success("No forward-looking risk signals triggered. Regime is broadly stable.")
    else:
        for conf_label, conf_key in [
            ("HIGH CONFIDENCE", "high"),
            ("MEDIUM CONFIDENCE", "medium"),
            ("LOW / SPECULATIVE", "low"),
        ]:
            group = [i for i in inferences if i.get("confidence") == conf_key]
            if not group:
                continue
            st.markdown(f"#### {conf_label}")
            for inf in group:
                with st.container():
                    st.markdown(
                        f'<div class="inference-{conf_key}">'
                        f'{inf.get("icon","📌")} <b>{inf.get("category","")}</b> · '
                        f'<i>{inf.get("horizon","")}</i><br>'
                        f'<b>{inf.get("statement","")}</b><br>'
                        f'<small>Trigger: {inf.get("trigger","")} | '
                        f'Context: {inf.get("context","")}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Implication"):
                        st.write(inf.get("implication", ""))
            st.divider()

    st.subheader("Cross-Asset Correlation Regime")
    corr_rg = ind.get("correlation_regime", "")
    st.metric("Equity–Bond Correlation", corr_rg.upper() if corr_rg else "N/A")
    st.caption(
        "Positive correlation (stress regime) = bonds and equities selling off together. "
        "Historically marks stagflationary or acute stress periods."
    )

    eem = mkt.get("eem") if mkt else None
    emb = mkt.get("emb") if mkt else None
    if eem is not None and emb is not None and not eem.empty and not emb.empty:
        fig_cross = go.Figure()
        fig_cross.add_trace(go.Scatter(
            x=eem.index, y=eem / eem.iloc[0] * 100,
            name="EEM (EM Equities)", line=dict(color="#c0392b"),
        ))
        fig_cross.add_trace(go.Scatter(
            x=emb.index, y=emb / emb.iloc[0] * 100,
            name="EMB (EM Bonds)", line=dict(color="#2980b9"),
        ))
        fig_cross.update_layout(title="EM Equities vs EM Bonds (indexed to 100)",
                                yaxis_title="Index", height=280)
        st.plotly_chart(fig_cross, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — WEEKLY NOTE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader(f"Weekly Macro Briefing — {note['as_of']}")

    if not show_narrative:
        st.info("Enable 'Weekly Narrative Note' in the sidebar.")
    else:
        regime_key = note["regime_label"].lower()
        st.markdown(_regime_html(regime_key), unsafe_allow_html=True)
        st.markdown(f"### {note['headline']}")
        st.caption(f"*{note['regime_desc']}*")
        st.divider()

        for sec in note["sections"]:
            with st.expander(f"**{sec['title']}**", expanded=True):
                st.markdown(sec["body"])
                for bullet in sec["bullets"]:
                    st.markdown(f"- {bullet}")

        st.divider()
        if note.get("data_sources"):
            st.caption(f"**Data sources:** {' · '.join(note['data_sources'])}")
        st.caption(note["footer"])

        md_text = narr.to_markdown(note)
        st.download_button(
            label="📥 Download Briefing (Markdown)",
            data=md_text.encode("utf-8"),
            file_name=f"macro_briefing_{datetime.date.today().isoformat()}.md",
            mime="text/markdown",
            use_container_width=True,
        )
