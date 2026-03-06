"""
app.py — Macro Intelligence Dashboard (real data only)
=======================================================
9-panel Streamlit application.

Run: streamlit run macro_intel/app.py
     # or from macro_intel/: streamlit run app.py

Panels:
  1. This Week's Signals
  2. Global Macro Regime
  3. Geo-Risk (real data only; placeholders labelled)
  4. Capital Flows & Structural Themes
  5. EM & Africa Stress Map
  6. Market-Implied Forward Layer
  7. Sector Intelligence (Energy, Agriculture, Chemicals, Industrials, Tech, Minerals)
  8. Sector Dependency Map
  9. Weekly Narrative
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
import note_generator as note_gen
import rules_engine as rules_eng
import sector_dependencies as sec_dep

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Macro Intelligence — Real Data",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .insight-alert   {background:#fdf3f2;border-left:4px solid #c0392b;padding:10px 14px;border-radius:6px;margin:6px 0;}
  .insight-warning {background:#fef9f0;border-left:4px solid #d35400;padding:10px 14px;border-radius:6px;margin:6px 0;}
  .insight-info    {background:#f0f7fd;border-left:4px solid #2980b9;padding:10px 14px;border-radius:6px;margin:6px 0;}
  .inference-high  {background:#fdf3f2;border-left:4px solid #c0392b;padding:8px 12px;border-radius:4px;margin:4px 0;}
  .inference-medium{background:#fef9f0;border-left:4px solid #d35400;padding:8px 12px;border-radius:4px;margin:4px 0;}
  .inference-low   {background:#f4f4f4;border-left:4px solid #7f8c8d;padding:8px 12px;border-radius:4px;margin:4px 0;}
  .source-badge    {display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.72rem;font-weight:600;margin:2px;}
  .source-live     {background:#d5f5e3;color:#1a5e2a;}
  .source-failed   {background:#fde8e8;color:#922b21;}
  .source-ph       {background:#f3e5f5;color:#5b2c6f;}
  .placeholder-box {background:#f8f0ff;border:1px dashed #8e44ad;border-radius:6px;padding:10px 14px;color:#5b2c6f;font-size:0.88rem;}
  .regime-crisis  {background:#fdf3f2;color:#c0392b;padding:8px 14px;border-radius:6px;font-weight:700;}
  .regime-stressed{background:#fef9f0;color:#d35400;padding:8px 14px;border-radius:6px;font-weight:700;}
  .regime-cautious{background:#fefdf0;color:#b7950b;padding:8px 14px;border-radius:6px;font-weight:700;}
  .regime-benign  {background:#eafaf1;color:#1a5e2a;padding:8px 14px;border-radius:6px;font-weight:700;}
  .sector-crisis  {background:#fdf3f2;border-left:4px solid #c0392b;padding:8px;border-radius:4px;}
  .sector-stress  {background:#fef9f0;border-left:4px solid #d35400;padding:8px;border-radius:4px;}
  .sector-normal  {background:#eafaf1;border-left:4px solid #27ae60;padding:8px;border-radius:4px;}
  .sector-nodata  {background:#f4f4f4;border-left:4px solid #7f8c8d;padding:8px;border-radius:4px;}
  .prop-chain     {background:#fef9f0;border:1px solid #d35400;border-radius:6px;padding:10px;margin:4px 0;}
</style>
""", unsafe_allow_html=True)


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _source_badge(label: str) -> str:
    if "PLACEHOLDER" in label:
        return f'<span class="source-badge source-ph">{label[:40]}</span>'
    if "FAILED" in label or "NO API KEY" in label:
        return f'<span class="source-badge source-failed">{label[:50]}</span>'
    return f'<span class="source-badge source-live">{label}</span>'


def _regime_html(regime: str) -> str:
    emoji = {"crisis":"🔴","stressed":"🟠","cautious":"🟡","benign":"🟢"}.get(regime,"⚪")
    return f'<div class="regime-{regime}">{emoji} Macro Regime: <b>{regime.upper()}</b></div>'


def _placeholder_box(name: str, reason: str) -> None:
    st.markdown(
        f'<div class="placeholder-box">⚫ <b>PLACEHOLDER — {name}</b><br>'
        f'<small>{reason}</small></div>',
        unsafe_allow_html=True,
    )


def _sector_card(name: str, data: dict) -> None:
    status = data.get("status","no_data")
    css = {"crisis":"sector-crisis","stress":"sector-stress",
           "contraction":"sector-crisis","slowing":"sector-stress",
           "normal":"sector-normal","comfortable":"sector-normal",
           "expanding":"sector-normal"}.get(status,"sector-nodata")
    icon = cfg.SECTOR_ICON.get(name,"📦")
    headline = data.get("headline","No data")
    avail = data.get("data_available", False)
    srcs  = ", ".join(data.get("data_sources",[]))
    st.markdown(
        f'<div class="{css}">'
        f'<b>{icon} {name}</b> — <i>{status.upper()}</i><br>'
        f'<small>{headline}</small><br>'
        f'<small style="color:#7f8c8d">Sources: {srcs}</small>'
        f'</div>',
        unsafe_allow_html=True,
    )
    for ph in data.get("placeholders", []):
        st.markdown(f'<div class="placeholder-box" style="margin:2px 0;font-size:0.78rem;">⚫ PLACEHOLDER: {ph}</div>',
                    unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    refresh_btn = st.button("🔄 Refresh Data", use_container_width=True)
    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("**API Keys** (free — improves data coverage)")
    fred_key = st.text_input("FRED API Key", value="", type="password",
                              placeholder="fred.stlouisfed.org (free)",
                              label_visibility="collapsed")
    eia_key  = st.text_input("EIA API Key",  value="", type="password",
                              placeholder="eia.gov/opendata (free)",
                              label_visibility="collapsed")
    if fred_key:
        import os; os.environ["FRED_API_KEY"] = fred_key
    if eia_key:
        import os; os.environ["EIA_API_KEY"]  = eia_key

    st.divider()
    show_inference   = st.toggle("Forward-Looking Inference",   value=True)
    show_narrative   = st.toggle("Weekly Narrative Note",       value=True)
    show_sector_deps = st.toggle("Sector Dependency Map",       value=True)
    show_raw         = st.toggle("Raw Indicator Values",        value=False)

    st.divider()
    africa_focus = st.multiselect(
        "Africa Focus Countries (ISO3)", options=cfg.WB_AFRICA,
        default=["NGA","KEN","EGY","GHA","ZAF"], label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"As of: {datetime.date.today().strftime('%d %b %Y')}")
    st.caption("Real data: FRED · EIA · AGSI+ · FAO · World Bank · IMF · OECD · Yahoo Finance")


# ── Data loading with source tracking ─────────────────────────────────────────
@st.cache_data(ttl=900,   show_spinner=False)
def load_market():    return df_mod.get_market_data()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_yields():    return df_mod.get_yields()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_fred():      return df_mod.get_fred_macro()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_eia_oil():   return df_mod.get_eia_oil_inventories()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_eia_gas():   return df_mod.get_eia_gas_storage_us()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_eu_gas():    return df_mod.get_eu_gas_storage()

@st.cache_data(ttl=86400, show_spinner=False)
def load_fao():       return df_mod.get_fao_food_price_index()

@st.cache_data(ttl=86400, show_spinner=False)
def load_wb_res():    return df_mod.get_worldbank_reserves()

@st.cache_data(ttl=86400, show_spinner=False)
def load_wb_fdi():    return df_mod.get_worldbank_fdi()

@st.cache_data(ttl=86400, show_spinner=False)
def load_wb_debt():   return df_mod.get_worldbank_ext_debt()

@st.cache_data(ttl=86400, show_spinner=False)
def load_imf():       return df_mod.get_imf_macro()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_oecd():      return df_mod.get_oecd_cli()

# Placeholders (always return None)
@st.cache_data(ttl=86400, show_spinner=False)
def load_shipping():  return df_mod.get_shipping_rates()

@st.cache_data(ttl=86400, show_spinner=False)
def load_embi():      return df_mod.get_embi_spreads()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_gpr():       return df_mod.get_gpr_index()


with st.spinner("Loading real-world data…"):
    mkt,      mkt_src  = load_market()
    yields,   yld_src  = load_yields()
    fred,     fred_src = load_fred()
    oil_inv,  eia_o_src= load_eia_oil()
    us_gas,   eia_g_src= load_eia_gas()
    eu_gas,   agsi_src = load_eu_gas()
    fao_fpi,  fao_src  = load_fao()
    fx_res,   wb_r_src = load_wb_res()
    fdi,      wb_f_src = load_wb_fdi()
    ext_debt, wb_d_src = load_wb_debt()
    imf_data, imf_src  = load_imf()
    oecd_cli, oecd_src = load_oecd()
    _,        ship_src = load_shipping()
    _,        embi_src = load_embi()
    gpr_data, gpr_src  = load_gpr()

ALL_SOURCES = {
    "Market": mkt_src, "Yields": yld_src, "FRED": fred_src,
    "EIA Oil": eia_o_src, "EIA Gas": eia_g_src, "AGSI+": agsi_src,
    "FAO": fao_src, "WB Reserves": wb_r_src, "WB FDI": wb_f_src,
    "WB Debt": wb_d_src, "IMF": imf_src, "OECD": oecd_src,
    "Shipping": ship_src, "EMBI": embi_src, "GPR": gpr_src,
}

# Compute
ind = ind_mod.compute_all(
    mkt=mkt, yields=yields, fred=fred,
    oil_inv=oil_inv, eu_gas=eu_gas, us_gas=us_gas, fao_fpi=fao_fpi,
    fx_reserves=fx_res, fdi=fdi, ext_debt=ext_debt, oecd_cli=oecd_cli,
)

rules        = rules_eng.run(ind)
inferences   = inf_eng.run(ind, rules)
sectors      = sec_dep.assess_sectors(ind)
propagations = sec_dep.run(ind)
note         = narr.generate(ind, rules, inferences, propagations, sectors)
daily        = note_gen.daily_note(ind, rules, inferences, propagations, sectors)
hourly       = note_gen.hourly_note(ind, rules, inferences)

last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")


# ── Header ─────────────────────────────────────────────────────────────────────
col_t, col_r = st.columns([3, 1])
with col_t:
    st.title("🌍 Macro Intelligence — Real Data Only")
    st.caption(f"Real APIs · No synthetic data · Placeholders labelled · Updated: {last_updated}")
with col_r:
    st.markdown(_regime_html(ind.get("macro_regime","benign")), unsafe_allow_html=True)
    st.caption(
        f"Financial stress: **{ind.get('financial_stress_score',0)}/7** · "
        f"Geo-risk: **{ind.get('geo_stress_score',0)}/4**"
    )

# Source badges
badges = " ".join(_source_badge(f"{k}: {v}") for k, v in ALL_SOURCES.items())
st.markdown(f"**Data sources:** {badges}", unsafe_allow_html=True)
st.divider()


# ── Pulse KPIs ─────────────────────────────────────────────────────────────────
def _kpi(col, label, val, fmt=".1f", suffix=""):
    v = f"{val:{fmt}}{suffix}" if not _nan(val) else "N/A"
    col.metric(label, v)

kpi = st.columns(9)
_kpi(kpi[0], "VIX",       ind.get("vix",          np.nan))
_kpi(kpi[1], "DXY",       ind.get("dxy",          np.nan))
_kpi(kpi[2], "Curve",     ind.get("curve_slope",  np.nan), fmt="+.0f", suffix=" bps")
_kpi(kpi[3], "Brent",     ind.get("brent",        np.nan), suffix=" $/bbl")
_kpi(kpi[4], "Copper 1M", ind.get("copper_1m_chg", np.nan) * 100 if not _nan(ind.get("copper_1m_chg")) else np.nan, fmt="+.1f", suffix="%")
_kpi(kpi[5], "HY Spread", ind.get("hy_spread",    np.nan), fmt=".0f",  suffix=" bps")
eu_g = ind.get("eu_gas_storage_pct", np.nan)
_kpi(kpi[6], "EU Gas",    eu_g,                           fmt=".1f",  suffix="% full")
fao  = ind.get("fao_fpi",  np.nan)
_kpi(kpi[7], "FAO FPI",   fao,                            fmt=".0f")
_kpi(kpi[8], "OECD CLI",  ind.get("oecd_cli_oecdall", np.nan), fmt=".1f")
st.divider()


# ── 9 Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🚨 1. Signals",
    "📈 2. Macro Regime",
    "🌐 3. Geo-Risk",
    "💰 4. Capital Flows",
    "🌍 5. EM & Africa",
    "🔮 6. Market-Implied",
    "⚡ 7. Sectors",
    "🔗 8. Sector Dependencies",
    "📰 9. Weekly Note",
    "📋 10. Daily Note",
    "⏱️ 11. Hourly Note",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("This Week's Signals")
    st.caption("Rules fire only on real data — silent if data unavailable.")

    if not rules:
        st.info("No rules fired — check data availability.")
    else:
        for r in rules:
            lvl = r["level"]
            icon = cfg.LEVEL_ICON.get(lvl,"")
            st.markdown(
                f'<div class="insight-{lvl}">'
                f'<b>{icon} [{lvl.upper()}] {r["category"]}</b>: {r["headline"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Detail & watch items"):
                st.write(r["detail"])
                st.markdown("**Watch:** " + " · ".join(r.get("watch",[])))

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Financial Stress Score**")
        st.progress(ind.get("financial_stress_score",0) / 7,
                    text=f"{ind.get('financial_stress_score',0)}/7 indicators elevated")
    with c2:
        st.markdown("**Geo / Sector Stress Score**")
        st.progress(ind.get("geo_stress_score",0) / 4,
                    text=f"{ind.get('geo_stress_score',0)}/4 indicators elevated")

    if show_raw:
        st.divider()
        st.subheader("Raw Indicator Values")
        st.dataframe(
            pd.DataFrame([{"Indicator": k, "Value": str(v)} for k, v in sorted(ind.items())]),
            use_container_width=True, height=400,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MACRO REGIME
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Global Macro Regime")

    # Yield curve shape
    if yields is not None and not yields.empty:
        tenor_map = {"3M":"3M","2Y":"2Y","5Y":"5Y","10Y":"10Y","30Y":"30Y"}
        curve_pts = {
            label: float(yields[col].dropna().iloc[-1])
            for label, col in tenor_map.items()
            if col in yields.columns and not yields[col].dropna().empty
        }
        if curve_pts:
            slope = ind.get("curve_slope", np.nan)
            curve_rg = ind.get("curve_regime","")
            title = f"US Treasury Yield Curve — {curve_rg.upper()}"
            if not _nan(slope):
                title += f" ({slope:+.0f} bps)"
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(curve_pts.keys()), y=list(curve_pts.values()),
                mode="lines+markers", line=dict(color="#2c3e50", width=3), marker=dict(size=10),
            ))
            fig.update_layout(title=title, xaxis_title="Tenor", yaxis_title="Yield (%)",
                              height=320, annotations=[dict(text=f"Source: {yld_src}",
                              showarrow=False, xref="paper", yref="paper", x=1, y=-0.15)])
            st.plotly_chart(fig, use_container_width=True)

        c_a, c_b = st.columns(2)
        with c_a:
            if "2Y" in yields.columns and "10Y" in yields.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=yields.index, y=yields["10Y"], name="10Y",
                                         line=dict(color="#2980b9")))
                fig.add_trace(go.Scatter(x=yields.index, y=yields["2Y"], name="2Y",
                                         line=dict(color="#c0392b")))
                fig.update_layout(title=f"2Y vs 10Y — {yld_src}", height=280)
                st.plotly_chart(fig, use_container_width=True)
        with c_b:
            if "2Y" in yields.columns and "10Y" in yields.columns:
                spread = (yields["10Y"] - yields["2Y"]) * 100
                fig = go.Figure()
                fig.add_trace(go.Bar(x=spread.index, y=spread,
                    marker_color=["#c0392b" if v < 0 else "#27ae60" for v in spread]))
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(title=f"2Y–10Y Spread (bps) — {yld_src}", height=280)
                st.plotly_chart(fig, use_container_width=True)
    else:
        _placeholder_box("Yield Curve", yld_src)

    # FRED macro
    st.divider()
    c_c, c_d = st.columns(2)
    with c_c:
        st.markdown("**Rate Indicators**")
        tp   = ind.get("term_premium", np.nan)
        be5  = ind.get("breakeven5y", np.nan)
        be55 = ind.get("breakeven5y5y", np.nan)
        move = ind.get("move_proxy", np.nan)
        st.metric("Term Premium (ACM/FRED)", f"{tp:.2f}%" if not _nan(tp) else "N/A — no FRED key")
        st.metric("5Y Breakeven (FRED)",     f"{be5:.2f}%" if not _nan(be5) else "N/A — no FRED key")
        st.metric("5Y5Y Forward (FRED)",     f"{be55:.2f}%" if not _nan(be55) else "N/A — no FRED key")
        st.metric("MOVE proxy (FRED yields)",f"{move:.0f}" if not _nan(move) else "N/A")
    with c_d:
        st.markdown("**Regimes**")
        for label, key in [("Inflation Regime","inflation_regime"),
                            ("Financial Conditions","fin_cond_regime"),
                            ("HY Credit Regime","hy_regime"),
                            ("IP Momentum","indpro_regime"),
                            ("OECD CLI","oecd_cli_regime"),
                            ("Corr Regime","correlation_regime")]:
            v = ind.get(key, "unknown")
            st.metric(label, v.upper() if v else "N/A")

    if fred is not None and "nfci" in fred:
        nfci = fred["nfci"].dropna()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=nfci.index, y=nfci,
            marker_color=["#c0392b" if v > 0.5 else "#27ae60" for v in nfci]))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title=f"Chicago Fed NFCI (>0 = tighter) — FRED", height=240)
        st.plotly_chart(fig, use_container_width=True)
    elif fred is None:
        _placeholder_box("NFCI", fred_src)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEO-RISK
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Geo-Risk (Real Data Only)")
    st.info("Only indicators with real free API sources are shown. "
            "Placeholders are labelled where no real API exists.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**US Oil Inventories — EIA**")
        if oil_inv is not None and not oil_inv.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=oil_inv.index, y=oil_inv, name="US Crude Stocks (Mbbls)"))
            if len(oil_inv) >= 260:
                avg5y = oil_inv.iloc[-260:].mean()
                fig.add_hline(y=avg5y, line_dash="dot", line_color="#d35400",
                              annotation_text="5Y avg")
            fig.update_layout(title="US Crude Oil Stocks (EIA)", yaxis_title="Thousand bbls",
                              height=260)
            st.plotly_chart(fig, use_container_width=True)
            inv_dev = ind.get("us_oil_inventory_dev", np.nan)
            st.caption(f"Deviation vs 5Y avg: {inv_dev*100:+.1f}%" if not _nan(inv_dev) else "")
        else:
            _placeholder_box("US Oil Inventories", eia_o_src)

    with c2:
        st.markdown("**EU Gas Storage — AGSI+**")
        if eu_gas is not None and not eu_gas.empty and "full_pct" in eu_gas.columns:
            pct = eu_gas["full_pct"].dropna()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pct.index, y=pct, name="EU Storage % Full",
                                     line=dict(color="#2980b9")))
            fig.add_hline(y=cfg.THRESH["eu_gas_storage_low"],
                          line_dash="dot", line_color="#d35400", annotation_text="Low")
            fig.add_hline(y=cfg.THRESH["eu_gas_storage_crisis"],
                          line_dash="dot", line_color="#c0392b", annotation_text="Crisis")
            fig.update_layout(title="EU Gas Storage % Full (AGSI+ / GIE)", yaxis_title="%",
                              height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("EU Gas Storage", agsi_src)

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**US Natural Gas Storage — EIA**")
        if us_gas is not None and not us_gas.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=us_gas.index, y=us_gas, name="US Gas Storage (Bcf)"))
            fig.update_layout(title="US Nat Gas Storage (EIA)", yaxis_title="Bcf", height=240)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("US Nat Gas Storage", eia_g_src)

    with c4:
        st.markdown("**OECD CLI — OECD**")
        if oecd_cli is not None and not oecd_cli.empty:
            fig = go.Figure()
            for col in oecd_cli.columns[:5]:
                fig.add_trace(go.Scatter(x=oecd_cli.index, y=oecd_cli[col], name=col))
            fig.add_hline(y=100, line_dash="dash", line_color="black", annotation_text="Trend")
            fig.update_layout(title="OECD Composite Leading Indicators (OECD SDMX)",
                              yaxis_title="Index", height=240)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("OECD CLI", oecd_src)

    # GPR — real if CSV available, else placeholder
    st.divider()
    st.markdown("**Geopolitical Risk Index — Caldara & Iacoviello (GPR)**")
    if gpr_data is not None and not gpr_data.empty:
        gpr_last  = float(gpr_data.dropna().iloc[-1])
        gpr_avg5y = float(gpr_data.iloc[-min(len(gpr_data), 60):].mean())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gpr_data.index, y=gpr_data,
                                 name="GPR", line=dict(color="#8e44ad")))
        fig.add_hline(y=gpr_avg5y, line_dash="dot", line_color="#7f8c8d",
                      annotation_text="Recent avg")
        fig.update_layout(
            title=f"Geopolitical Risk Index — {gpr_src} | Latest: {gpr_last:.0f}",
            yaxis_title="GPR Index", height=260,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Source: {gpr_src}")
    else:
        _placeholder_box("GPR — Caldara & Iacoviello", gpr_src)
        st.caption(
            "To activate: download `gpr_daily.csv` from "
            "[matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm) "
            "and place it in `macro_intel/data/`"
        )

    st.divider()
    st.markdown("**Placeholders — Real-time API unavailable**")
    for key, meta in cfg.PLACEHOLDERS.items():
        if key == "gpr_index":
            continue   # Already handled above
        _placeholder_box(meta["name"], meta["reason"] + " | " + meta.get("alt",""))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CAPITAL FLOWS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Capital Flows & Structural Themes")

    # DXY
    dxy_s = mkt.get("dxy") if mkt is not None else None
    if dxy_s is not None and not dxy_s.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dxy_s.index, y=dxy_s, name="DXY", line=dict(color="#2c3e50")))
        fig.add_hline(y=cfg.THRESH["dxy_strong"],      line_dash="dot",
                      line_color="#d35400", annotation_text="Strong")
        fig.add_hline(y=cfg.THRESH["dxy_very_strong"], line_dash="dot",
                      line_color="#c0392b", annotation_text="Very Strong")
        fig.update_layout(title=f"USD Index (DXY) — {mkt_src}", height=280)
        st.plotly_chart(fig, use_container_width=True)
    else:
        _placeholder_box("DXY", mkt_src)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**FDI Inflows by Region — World Bank**")
        if fdi is not None and not fdi.empty:
            fig = go.Figure()
            for col in fdi.columns:
                fig.add_trace(go.Bar(name=col, x=fdi.index, y=fdi[col]))
            fig.update_layout(barmode="group", title=f"FDI Net Inflows (USD bn) — {wb_f_src}",
                              yaxis_title="USD bn", height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("FDI Flows", wb_f_src)

    with c2:
        st.markdown("**External Debt — World Bank**")
        if ext_debt is not None and not ext_debt.empty:
            fig = go.Figure()
            for col in ext_debt.columns[:6]:
                fig.add_trace(go.Scatter(x=ext_debt.index, y=ext_debt[col] / 1e9,
                                         name=col, mode="lines+markers"))
            fig.update_layout(title=f"External Debt Stocks (USD bn) — {wb_d_src}",
                              height=280)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("External Debt", wb_d_src)

    _placeholder_box("Cross-border Banking Flows (BIS)",
                     "BIS SDMX API requires complex authentication for full LBS data")
    _placeholder_box("EM Portfolio Flows (IIF)",
                     "IIF data requires subscription — weekly flows unavailable free")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EM & AFRICA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("EM & Africa Stress Map")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("EM Regime",      ind.get("em_regime","unknown").upper())
        st.metric("EM Equity 1M",   f"{ind.get('eem_1m_chg',np.nan)*100:+.1f}%"
                  if not _nan(ind.get("eem_1m_chg")) else "N/A")
        st.metric("EM FX Stress Avg", f"{ind.get('em_fx_stress_avg',np.nan)*100:+.1f}%"
                  if not _nan(ind.get("em_fx_stress_avg")) else "N/A")
        st.metric("USD Vulnerability", ind.get("usd_debt_vulnerability","unknown").upper()
                  if ind.get("usd_debt_vulnerability") else "N/A")

    with c2:
        if ind.get("fx_res_deteriorating"):
            st.warning(f"⚠️ FX reserve drawdown: **{ind.get('fx_res_worst_country','unknown')}** most exposed (World Bank)")
        else:
            st.success("FX reserves: no acute drawdown detected")
        _placeholder_box("EMBI Sovereign Spreads", embi_src)

    # EM FX table
    st.divider()
    st.markdown("**EM FX Performance — Yahoo Finance**")
    if mkt is not None:
        fx_pairs = {
            "USD/BRL":"usdbrl","USD/ZAR":"usdzar","USD/TRY":"usdtry",
            "USD/CNH":"usdcnh","USD/INR":"usdinr","USD/MXN":"usdmxn",
        }
        rows = []
        for label, key in fx_pairs.items():
            s = mkt.get(key)
            if s is not None and len(s) >= 22:
                last = float(s.iloc[-1])
                chg_1m = last / float(s.iloc[-22]) - 1
                rows.append({"Pair":label, "Latest":f"{last:.2f}",
                             "1M Chg":f"{chg_1m*100:+.1f}%"})
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Pair"), use_container_width=True)
    else:
        _placeholder_box("EM FX", mkt_src)

    # FX Reserves chart
    st.divider()
    st.markdown("**FX Reserves — World Bank**")
    if fx_res is not None and not fx_res.empty:
        fig = go.Figure()
        for iso in (africa_focus or cfg.WB_AFRICA[:5]):
            if iso in fx_res.columns:
                fig.add_trace(go.Scatter(x=fx_res.index, y=fx_res[iso],
                                         mode="lines+markers", name=iso))
        fig.update_layout(title=f"FX Reserves (World Bank) — {wb_r_src}",
                          yaxis_title="USD", height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        _placeholder_box("FX Reserves", wb_r_src)

    # IMF data
    st.divider()
    st.markdown("**IMF Macro Indicators — IMF DataMapper**")
    if imf_data is not None and len(imf_data) > 0:
        for label, df_imf in imf_data.items():
            if df_imf is not None and not df_imf.empty:
                with st.expander(f"**{label.replace('_',' ').title()}** (IMF)"):
                    st.dataframe(df_imf.tail(5), use_container_width=True)
    else:
        _placeholder_box("IMF Macro Data", imf_src)

    st.info("📌 **Eurobond maturity wall 2025–2027**: Ghana, Kenya, Ethiopia, Egypt each face "
            "significant Eurobond redemptions. Monitor FX reserve cover and IMF program status.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MARKET-IMPLIED
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Market-Implied Forward Layer")
    st.caption(f"Source: {mkt_src} | Breakevens: {fred_src}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**VIX & Volatility**")
        st.metric("VIX",              f"{ind.get('vix',np.nan):.1f}" if not _nan(ind.get("vix")) else "N/A")
        st.metric("VIX Regime",       ind.get("vix_regime","unknown").upper())
        st.metric("VIX Term Structure",ind.get("vix_term_structure","unknown").upper())
        st.metric("MOVE Proxy",       f"{ind.get('move_proxy',np.nan):.0f}" if not _nan(ind.get("move_proxy")) else "N/A")
        _placeholder_box("ICE MOVE Index", cfg.PLACEHOLDERS["move_index"]["reason"])

    with c2:
        st.markdown("**Inflation Expectations (FRED)**")
        be5  = ind.get("breakeven5y",   np.nan)
        be55 = ind.get("breakeven5y5y", np.nan)
        tp   = ind.get("term_premium",  np.nan)
        st.metric("5Y Breakeven",   f"{be5:.2f}%"  if not _nan(be5)  else "N/A — no FRED key")
        st.metric("5Y5Y Forward",   f"{be55:.2f}%" if not _nan(be55) else "N/A — no FRED key")
        st.metric("Term Premium",   f"{tp:.2f}%"   if not _nan(tp)   else "N/A — no FRED key")
        st.metric("Inflation Regime",ind.get("inflation_regime","unknown").upper())

    with c3:
        st.markdown("**Cross-Asset Signals**")
        st.metric("Eq-Bond Correlation", f"{ind.get('eq_bond_corr',np.nan):.2f}"
                  if not _nan(ind.get("eq_bond_corr")) else "N/A")
        st.metric("Correlation Regime",  ind.get("correlation_regime","unknown").upper())
        st.metric("Cu/Gold Regime",      ind.get("copper_gold_regime","unknown").upper())
        st.metric("Systemic Stress",     "YES ⚠️" if ind.get("systemic_stress_signal") else "No")

    # EEM vs EMB chart
    if mkt is not None:
        eem = mkt.get("eem")
        emb = mkt.get("emb")
        if eem is not None and emb is not None and not eem.empty and not emb.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eem.index, y=eem/eem.iloc[0]*100,
                                     name="EEM (EM Equities)", line=dict(color="#c0392b")))
            fig.add_trace(go.Scatter(x=emb.index, y=emb/emb.iloc[0]*100,
                                     name="EMB (EM Bonds)", line=dict(color="#2980b9")))
            fig.update_layout(title=f"EM Equities vs EM Bonds (indexed 100) — {mkt_src}",
                              height=280)
            st.plotly_chart(fig, use_container_width=True)

    # Inference panel
    if show_inference:
        st.divider()
        st.subheader("Forward-Looking Inference Engine")
        st.caption("Only rules backed by real data fire. Silent if data unavailable.")
        if not inferences:
            st.success("No forward-looking risk signals triggered.")
        else:
            for conf_label, conf_key in [
                ("HIGH CONFIDENCE","high"),("MEDIUM CONFIDENCE","medium"),("LOW","low")
            ]:
                group = [i for i in inferences if i.get("confidence") == conf_key]
                if not group: continue
                st.markdown(f"#### {conf_label}")
                for inf in group:
                    srcs = ", ".join(inf.get("data_sources",[]))
                    st.markdown(
                        f'<div class="inference-{conf_key}">'
                        f'{inf.get("icon","📌")} <b>{inf.get("category","")}</b> · '
                        f'<i>{inf.get("horizon","")}</i><br>'
                        f'<b>{inf.get("statement","")}</b><br>'
                        f'<small>Trigger: {inf.get("trigger","")} | '
                        f'Context: {inf.get("context","")}</small><br>'
                        f'<small style="color:#2980b9">Sources: {srcs}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("Implication"):
                        st.write(inf.get("implication",""))
                st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SECTORS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Sector Intelligence Panel")
    st.caption("Real data where available · Placeholders labelled for unavailable sources")

    # Sector cards
    cols = st.columns(3)
    for idx, (name, data) in enumerate(sectors.items()):
        with cols[idx % 3]:
            _sector_card(name, data)

    st.divider()

    # Energy charts (real)
    st.subheader("⚡ Energy — Detailed")
    c_e1, c_e2 = st.columns(2)
    with c_e1:
        brent_s = mkt.get("brent") if mkt is not None else None
        if brent_s is not None and not brent_s.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=brent_s.index, y=brent_s, name="Brent Crude"))
            fig.update_layout(title=f"Brent Crude (USD/bbl) — {mkt_src}", height=240)
            st.plotly_chart(fig, use_container_width=True)
    with c_e2:
        natgas_s = mkt.get("natgas") if mkt is not None else None
        if natgas_s is not None and not natgas_s.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=natgas_s.index, y=natgas_s, name="Henry Hub (NG=F)"))
            fig.update_layout(title=f"US Nat Gas (Henry Hub) — {mkt_src}", height=240)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("EU Electricity Prices", cfg.PLACEHOLDERS["electricity_eu"]["reason"])

    # Agriculture charts (real)
    st.divider()
    st.subheader("🌾 Agriculture — Detailed")
    c_a1, c_a2 = st.columns(2)
    with c_a1:
        if fao_fpi is not None and not fao_fpi.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fao_fpi.index, y=fao_fpi, name="FAO FPI",
                                     line=dict(color="#27ae60")))
            fig.add_hline(y=cfg.THRESH["fao_fpi_stress"], line_dash="dot",
                          line_color="#d35400", annotation_text="Stress")
            fig.add_hline(y=cfg.THRESH["fao_fpi_crisis"], line_dash="dot",
                          line_color="#c0392b", annotation_text="Crisis")
            fig.update_layout(title=f"FAO Food Price Index — {fao_src}",
                              yaxis_title="Index (2014-16=100)", height=260)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("FAO Food Price Index", fao_src)

    with c_a2:
        wheat_s = mkt.get("wheat") if mkt is not None else None
        corn_s  = mkt.get("corn")  if mkt is not None else None
        if wheat_s is not None or corn_s is not None:
            fig = go.Figure()
            if wheat_s is not None and not wheat_s.empty:
                fig.add_trace(go.Scatter(x=wheat_s.index,
                                         y=wheat_s/float(wheat_s.iloc[0])*100,
                                         name="Wheat (ZW=F, indexed)"))
            if corn_s is not None and not corn_s.empty:
                fig.add_trace(go.Scatter(x=corn_s.index,
                                         y=corn_s/float(corn_s.iloc[0])*100,
                                         name="Corn (ZC=F, indexed)"))
            fig.update_layout(title=f"Wheat & Corn Futures (indexed 100) — {mkt_src}", height=260)
            st.plotly_chart(fig, use_container_width=True)

    _placeholder_box("Fertilizer Prices (Urea, Ammonia, Potash)",
                     cfg.PLACEHOLDERS["fertilizer_prices"]["reason"])

    # Chemicals
    st.divider()
    st.subheader("🧪 Chemicals — Placeholders")
    _placeholder_box("Fertilizer Prices", cfg.PLACEHOLDERS["fertilizer_prices"]["reason"])
    _placeholder_box("Petrochemical feedstock (Naphtha, Ethane)",
                     "No free real-time REST API. LME and ICIS require subscriptions.")

    # Critical Minerals
    st.divider()
    st.subheader("⛏️ Critical Minerals")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        copper_s = mkt.get("copper") if mkt is not None else None
        if copper_s is not None and not copper_s.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=copper_s.index, y=copper_s,
                                     name="Copper (HG=F)", line=dict(color="#d35400")))
            fig.update_layout(title=f"Copper Futures (HG=F) — {mkt_src}", height=240)
            st.plotly_chart(fig, use_container_width=True)
    with c_m2:
        nickel_s = mkt.get("nickel") if mkt is not None else None
        if nickel_s is not None and not nickel_s.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nickel_s.index, y=nickel_s,
                                     name="Nickel (NI=F)", line=dict(color="#8e44ad")))
            fig.update_layout(title=f"Nickel Futures (NI=F) — {mkt_src}", height=240)
            st.plotly_chart(fig, use_container_width=True)
        else:
            _placeholder_box("Nickel", mkt_src)
    _placeholder_box("Lithium Prices",  cfg.PLACEHOLDERS["lithium_prices"]["reason"])
    _placeholder_box("Cobalt Prices",   cfg.PLACEHOLDERS["cobalt_prices"]["reason"])
    _placeholder_box("LME Inventories", cfg.PLACEHOLDERS["lme_inventories"]["reason"])

    # Technology
    st.divider()
    st.subheader("💾 Technology — Placeholders")
    _placeholder_box("Semiconductor Sales (WSTS)", cfg.PLACEHOLDERS["semiconductor"]["reason"])
    st.caption("No real-time free API exists for semiconductor sales data. "
               "Sector ETF (XLK) available on yfinance as rough proxy only.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — SECTOR DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader("Sector Dependency Map")
    st.caption("Cross-sector propagation chains — only fire on real indicator data")

    if not show_sector_deps:
        st.info("Enable 'Sector Dependency Map' in the sidebar.")
    elif not propagations:
        st.success("No cross-sector propagation chains triggered. Sector conditions broadly stable.")
    else:
        for prop in propagations:
            strength = prop.get("strength","")
            icon_str = prop.get("icon","🔗")
            st.markdown(
                f'<div class="prop-chain">'
                f'{icon_str} <b>{prop.get("from_sector","")} → {prop.get("to_sector","")}</b>'
                f' <span style="color:#7f8c8d;font-size:0.85rem">({strength} linkage)</span><br>'
                f'<b>{prop.get("headline","")}</b><br>'
                f'<small>{prop.get("mechanism","")}</small><br>'
                f'<small><i>Signal: {prop.get("signal","")}</i></small><br>'
                f'<small style="color:#2980b9">Sources: {", ".join(prop.get("data_sources",[]))}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Implication"):
                st.write(prop.get("implication",""))

    # Sector connectivity diagram (text-based)
    st.divider()
    st.subheader("Known Sector Linkages")
    st.markdown("""
| From Sector | Transmission Channel | To Sector | Data Available? |
|---|---|---|---|
| Energy (Gas) | Haber-Bosch feedstock | Chemicals / Fertilizers | ✅ AGSI+ (gas) |
| Chemicals | Fertilizer supply crunch | Agriculture (yields) | ✅ AGSI+ + FAO |
| Agriculture | Food CPI in EM (40-60% basket) | EM/Africa Stress | ✅ FAO + yfinance |
| Energy (Oil) | Transport/diesel cost | Industrials, Agriculture | ✅ yfinance |
| Critical Minerals (Cu) | EV/grid capex | Electrification timeline | ✅ yfinance |
| Macro (USD) | Debt service cost | EM/Africa | ✅ yfinance + WB |
| Industrials (IP) | Base metal demand | Critical Minerals | ✅ FRED |
| Shipping rates | Supply chain costs | Industrials, Chemicals | ⚫ PLACEHOLDER |
| Fertilizer prices | Crop input costs | Agriculture | ⚫ PLACEHOLDER |
| LME inventories | Metal supply signal | Critical Minerals | ⚫ PLACEHOLDER |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — WEEKLY NOTE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:  # noqa: E741 — index 8 = tab 9
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
                for b in sec["bullets"]:
                    st.markdown(f"- {b}")

        st.divider()
        if note.get("data_sources"):
            st.caption(f"**Real data sources used:** {' · '.join(sorted(set(note['data_sources'])))}")
        st.caption(note["footer"])

        md_text = narr.to_markdown(note)
        st.download_button(
            label="📥 Download Briefing (Markdown)",
            data=md_text.encode("utf-8"),
            file_name=f"macro_briefing_{datetime.date.today().isoformat()}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — DAILY NOTE
# ══════════════════════════════════════════════════════════════════════════════

def _render_note_section(sec: dict) -> None:
    """Render a note_generator section inside Streamlit."""
    level_css = {
        "alert":   ("insight-alert",   "🔴"),
        "warning": ("insight-warning",  "🟠"),
        "good":    ("insight-info",     "🟢"),
        "info":    ("insight-info",     "🔵"),
    }
    css, icon = level_css.get(sec.get("level", "info"), ("insight-info", "⚪"))
    with st.expander(f"{icon} **{sec['title']}**", expanded=True):
        for row in sec.get("rows", []):
            if not row.strip():
                st.write("")
            elif row.startswith("⚫") or "PLACEHOLDER" in row.upper():
                st.markdown(
                    f'<div class="placeholder-box" style="margin:2px 0;font-size:0.82rem;">{row}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<div class="{css}" style="padding:4px 10px;margin:2px 0;">{row}</div>',
                            unsafe_allow_html=True)


with tabs[9]:
    st.subheader(f"Daily Intelligence Note — {daily['as_of']}")
    st.caption("9-section intelligence briefing | Real data only | PLACEHOLDERs labelled")

    regime_key = daily.get("regime", "benign")
    st.markdown(_regime_html(regime_key), unsafe_allow_html=True)
    st.markdown(f"**{daily['headline']}**")
    st.divider()

    for sec in daily.get("sections", []):
        _render_note_section(sec)

    st.divider()
    st.caption(daily.get("footer", ""))

    txt = note_gen.to_text(daily)
    st.download_button(
        label="📥 Download Daily Note (text)",
        data=txt.encode("utf-8"),
        file_name=f"daily_note_{datetime.date.today().isoformat()}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — HOURLY NOTE
# ══════════════════════════════════════════════════════════════════════════════

with tabs[10]:
    st.subheader(f"Hourly Macro Snapshot — {hourly['as_of']}")
    st.caption("Condensed 4-section real-time snapshot | Refreshes on page reload")

    regime_key_h = hourly.get("regime", "benign")
    st.markdown(_regime_html(regime_key_h), unsafe_allow_html=True)
    st.markdown(f"**{hourly['headline']}**")
    st.divider()

    for sec in hourly.get("sections", []):
        _render_note_section(sec)

    st.divider()
    st.caption(hourly.get("footer", ""))

    txt_h = note_gen.to_text(hourly)
    st.download_button(
        label="📥 Download Hourly Snapshot (text)",
        data=txt_h.encode("utf-8"),
        file_name=f"hourly_note_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
