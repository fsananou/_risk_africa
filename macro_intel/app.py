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
  /* ── Signal boxes: sharp rectangle, title + body visible directly ── */
  .sig-box        {border:1px solid;padding:12px 16px;margin:6px 0;border-radius:0;}
  .sig-alert      {background:#fdf3f2;border-color:#c0392b;border-left:5px solid #c0392b;}
  .sig-warning    {background:#fef9f0;border-color:#d35400;border-left:5px solid #d35400;}
  .sig-info       {background:#f0f7fd;border-color:#2980b9;border-left:5px solid #2980b9;}
  .sig-good       {background:#eafaf1;border-color:#27ae60;border-left:5px solid #27ae60;}
  .sig-title      {font-weight:700;font-size:0.92rem;margin-bottom:4px;}
  .sig-body       {font-size:0.86rem;color:#2c3e50;margin-bottom:4px;line-height:1.5;}
  .sig-watch      {font-size:0.78rem;color:#7f8c8d;margin-top:4px;}
  /* ── Inference boxes ── */
  .inf-high   {background:#fdf3f2;border:1px solid #c0392b;border-left:5px solid #c0392b;padding:10px 14px;margin:5px 0;border-radius:0;}
  .inf-medium {background:#fef9f0;border:1px solid #d35400;border-left:5px solid #d35400;padding:10px 14px;margin:5px 0;border-radius:0;}
  .inf-low    {background:#f4f4f4;border:1px solid #95a5a6;border-left:5px solid #95a5a6;padding:10px 14px;margin:5px 0;border-radius:0;}
  /* ── Badges ── */
  .source-badge {display:inline-block;padding:2px 8px;border-radius:0;font-size:0.72rem;font-weight:600;margin:2px;}
  .source-live  {background:#d5f5e3;color:#1a5e2a;}
  .source-failed{background:#fde8e8;color:#922b21;}
  .source-ph    {background:#f3e5f5;color:#5b2c6f;}
  /* ── Placeholder box ── */
  .placeholder-box{background:#f8f0ff;border:1px dashed #8e44ad;border-radius:0;padding:10px 14px;color:#5b2c6f;font-size:0.88rem;}
  /* ── Regime banner ── */
  .regime-crisis  {background:#fdf3f2;color:#c0392b;padding:8px 14px;border-radius:0;font-weight:700;border-left:5px solid #c0392b;}
  .regime-stressed{background:#fef9f0;color:#d35400;padding:8px 14px;border-radius:0;font-weight:700;border-left:5px solid #d35400;}
  .regime-cautious{background:#fefdf0;color:#b7950b;padding:8px 14px;border-radius:0;font-weight:700;border-left:5px solid #b7950b;}
  .regime-benign  {background:#eafaf1;color:#1a5e2a;padding:8px 14px;border-radius:0;font-weight:700;border-left:5px solid #27ae60;}
  /* ── Sector cards ── */
  .sector-crisis {background:#fdf3f2;border-left:5px solid #c0392b;padding:10px 14px;border-radius:0;margin:4px 0;}
  .sector-stress {background:#fef9f0;border-left:5px solid #d35400;padding:10px 14px;border-radius:0;margin:4px 0;}
  .sector-normal {background:#eafaf1;border-left:5px solid #27ae60;padding:10px 14px;border-radius:0;margin:4px 0;}
  .sector-nodata {background:#f4f4f4;border-left:5px solid #95a5a6;padding:10px 14px;border-radius:0;margin:4px 0;}
  /* ── Propagation chain ── */
  .prop-chain    {background:#fef9f0;border:1px solid #d35400;border-radius:0;padding:10px;margin:4px 0;}
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


def _card(label: str, value: str, note: str = "", color: str = "#2c3e50") -> str:
    """Horizontal metric card in sig-box style. Returns HTML string."""
    note_html = (f'&nbsp;<span style="color:#7f8c8d;font-size:0.76rem;">{note}</span>'
                 if note else "")
    return (
        f'<div class="sig-box" style="border-left:5px solid {color};'
        f'padding:8px 14px;margin:3px 0;">'
        f'<span style="font-weight:600;font-size:0.86rem;color:#2c3e50;">{label}</span>'
        f'<span style="float:right;font-size:0.88rem;">'
        f'<b style="color:{color};">{value}</b>{note_html}</span></div>'
    )


def _regime_color(regime: str) -> str:
    return {"crisis":"#c0392b","stressed":"#d35400",
            "cautious":"#b7950b","benign":"#27ae60",
            "elevated":"#d35400","tight":"#c0392b",
            "loose":"#27ae60","normal":"#2980b9",
            "inverted":"#c0392b","flat":"#d35400",
            "steep":"#27ae60","high":"#c0392b",
            "moderate":"#d35400","low":"#27ae60",
            "expanding":"#27ae60","contracting":"#c0392b",
            "slowing":"#d35400","unknown":"#7f8c8d"}.get(regime.lower(),"#2c3e50")


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
    _fk_check = cfg.get_fred_key()
    _ek_check = cfg.get_eia_key()
    st.caption(f"FRED: {'✅' if _fk_check else '❌ missing'} · EIA: {'✅' if _ek_check else '❌ missing'}")
    st.caption(f"As of: {datetime.date.today().strftime('%d %b %Y')}")


# ── Resolve keys once (from secrets.toml → env var) ──────────────────────────
_FRED_KEY = cfg.get_fred_key() or ""
_EIA_KEY  = cfg.get_eia_key()  or ""

# ── Data loading with source tracking ─────────────────────────────────────────
@st.cache_data(ttl=900,   show_spinner=False)
def load_market():              return df_mod.get_market_data()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_yields(fk: str):       return df_mod.get_yields(fk)

@st.cache_data(ttl=3600,  show_spinner=False)
def load_fred(fk: str):         return df_mod.get_fred_macro(fk)

@st.cache_data(ttl=3600,  show_spinner=False)
def load_eia_oil(ek: str):      return df_mod.get_eia_oil_inventories(ek)

@st.cache_data(ttl=3600,  show_spinner=False)
def load_eia_gas(ek: str):      return df_mod.get_eia_gas_storage_us(ek)

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

@st.cache_data(ttl=86400, show_spinner=False)
def load_wb_cmo():    return df_mod.get_wb_cmo()

@st.cache_data(ttl=3600,  show_spinner=False)
def load_gpr():       return df_mod.get_gpr_index()

@st.cache_data(ttl=86400, show_spinner=False)
def load_wb_gni():    return df_mod.get_worldbank_gni_per_capita()

@st.cache_data(ttl=86400, show_spinner=False)
def load_wb_dservice(): return df_mod.get_worldbank_debt_service()


with st.spinner("Loading real-world data…"):
    mkt,      mkt_src  = load_market()
    yields,   yld_src  = load_yields(_FRED_KEY)
    fred,     fred_src = load_fred(_FRED_KEY)
    oil_inv,  eia_o_src= load_eia_oil(_EIA_KEY)
    us_gas,   eia_g_src= load_eia_gas(_EIA_KEY)
    eu_gas,   agsi_src = load_eu_gas()
    fao_fpi,  fao_src  = load_fao()
    fx_res,   wb_r_src = load_wb_res()
    fdi,      wb_f_src = load_wb_fdi()
    ext_debt, wb_d_src = load_wb_debt()
    imf_data, imf_src  = load_imf()
    oecd_cli, oecd_src = load_oecd()
    cmo_data, cmo_src  = load_wb_cmo()
    gpr_data, gpr_src  = load_gpr()
    gni_data, gni_src  = load_wb_gni()
    dservice, dsv_src  = load_wb_dservice()

ALL_SOURCES = {
    "Market": mkt_src, "Yields": yld_src, "FRED": fred_src,
    "EIA Oil": eia_o_src, "EIA Gas": eia_g_src, "AGSI+": agsi_src,
    "FAO": fao_src, "WB Reserves": wb_r_src, "WB FDI": wb_f_src,
    "WB Debt": wb_d_src, "WB CMO": cmo_src, "IMF": imf_src,
    "OECD": oecd_src, "GPR": gpr_src,
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
            lvl  = r["level"]
            icon = cfg.LEVEL_ICON.get(lvl, "")
            watch_str = "  ·  ".join(r.get("watch", []))
            st.markdown(
                f'<div class="sig-box sig-{lvl}">'
                f'<div class="sig-title">{icon} [{lvl.upper()}] {r["category"]} — {r["headline"]}</div>'
                f'<div class="sig-body">{r["detail"]}</div>'
                + (f'<div class="sig-watch">Watch: {watch_str}</div>' if watch_str else "")
                + "</div>",
                unsafe_allow_html=True,
            )

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
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:4px 12px;'
            f'border-bottom:1px solid #eee;font-size:0.82rem;">'
            f'<span style="color:#555;">{k}</span>'
            f'<span style="font-weight:600;color:#2c3e50;">{str(v)[:40]}</span></div>'
            for k, v in sorted(ind.items())
        )
        st.markdown(
            f'<div style="border:1px solid #ddd;max-height:400px;overflow-y:auto;">{rows_html}</div>',
            unsafe_allow_html=True,
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
    # FRED macro
    st.divider()
    c_c, c_d = st.columns(2)
    with c_c:
        st.markdown("**Rate Indicators**")
        tp   = ind.get("term_premium", np.nan)
        be5  = ind.get("breakeven5y", np.nan)
        be55 = ind.get("breakeven5y5y", np.nan)
        move = ind.get("move_proxy", np.nan)
        be5_c  = "#c0392b" if not _nan(be5)  and be5  > 2.8 else "#27ae60" if not _nan(be5)  and be5  < 2.0 else "#2980b9"
        be55_c = "#c0392b" if not _nan(be55) and be55 > 2.8 else "#27ae60" if not _nan(be55) and be55 < 2.0 else "#2980b9"
        tp_c   = "#c0392b" if not _nan(tp)   and tp   > 1.0 else "#27ae60" if not _nan(tp)   and tp   < 0   else "#2980b9"
        cards = "".join([
            _card("5Y Breakeven",     f"{be5:.2f}%"  if not _nan(be5)  else "N/A",  "FRED", be5_c),
            _card("5Y5Y Fwd Inflation",f"{be55:.2f}%" if not _nan(be55) else "N/A", "FRED", be55_c),
            _card("Term Premium",     f"{tp:.2f}%"   if not _nan(tp)   else "N/A",  "FRED", tp_c),
            _card("MOVE Proxy",       f"{move:.0f}"  if not _nan(move) else "N/A",  "FRED yields", "#2c3e50"),
        ])
        st.markdown(cards, unsafe_allow_html=True)
    with c_d:
        st.markdown("**Regime Dashboard**")
        regime_items = [
            ("Inflation",         "inflation_regime"),
            ("Financial Cond.",   "fin_cond_regime"),
            ("HY Credit",         "hy_regime"),
            ("IP Momentum",       "indpro_regime"),
            ("OECD CLI",          "oecd_cli_regime"),
            ("Corr Regime",       "correlation_regime"),
        ]
        cards = "".join(
            _card(label, v.upper() if v else "N/A", color=_regime_color(v or "unknown"))
            for label, key in regime_items
            for v in [ind.get(key, "unknown")]
        )
        st.markdown(cards, unsafe_allow_html=True)

    if fred is not None and "nfci" in fred:
        nfci = fred["nfci"].dropna()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=nfci.index, y=nfci,
            marker_color=["#c0392b" if v > 0.5 else "#27ae60" for v in nfci]))
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        fig.update_layout(title="Chicago Fed NFCI (>0 = tighter) — FRED", height=240)
        st.plotly_chart(fig, use_container_width=True)


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

    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**US Natural Gas Storage — EIA**")
        if us_gas is not None and not us_gas.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=us_gas.index, y=us_gas, name="US Gas Storage (Bcf)"))
            fig.update_layout(title="US Nat Gas Storage (EIA)", yaxis_title="Bcf", height=240)
            st.plotly_chart(fig, use_container_width=True)

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

    # Baltic Dry Index
    if fred is not None and "baltic_dry" in fred:
        bdi = fred["baltic_dry"].dropna()
        if not bdi.empty:
            st.divider()
            st.markdown("**Baltic Dry Index — FRED (BDIY)**")
            bdi_avg2y = float(bdi.iloc[-min(len(bdi), 520):].mean())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bdi.index, y=bdi, name="BDI",
                                     line=dict(color="#2c3e50")))
            fig.add_hline(y=bdi_avg2y, line_dash="dot", line_color="#7f8c8d",
                          annotation_text="2Y avg")
            fig.update_layout(
                title="Baltic Dry Index (global shipping demand) — FRED",
                yaxis_title="Index", height=260,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EM & AFRICA
# ══════════════════════════════════════════════════════════════════════════════

def _make_africa_choropleth(
    iso_values: dict,
    title: str,
    colorscale: str = "RdYlGn",
    reversescale: bool = False,
    suffix: str = "",
) -> go.Figure:
    """Build a compact Africa choropleth. iso_values: {ISO3: float}."""
    locs = list(iso_values.keys())
    zvals = list(iso_values.values())
    fig = go.Figure(go.Choropleth(
        locations=locs,
        z=zvals,
        locationmode="ISO-3",
        colorscale=colorscale,
        reversescale=reversescale,
        colorbar=dict(thickness=12, len=0.7, title=dict(text=suffix, side="right")),
        hovertemplate="<b>%{location}</b><br>" + suffix + " %{z:.1f}<extra></extra>",
    ))
    fig.update_geos(
        scope="africa",
        showframe=False,
        showcoastlines=True,
        coastlinecolor="#cccccc",
        showland=True,
        landcolor="#f5f5f5",
        showocean=True,
        oceancolor="#eaf3fb",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=340,
        margin=dict(l=0, r=0, t=35, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _imf_latest_by_country(imf_data: dict, key: str, current_year: int) -> dict:
    """Extract most recent value per country from IMF DataMapper result."""
    df = imf_data.get(key)
    if df is None or df.empty:
        return {}
    # prefer current_year or next year (forecast), fall back to most recent
    out = {}
    for iso2 in df.columns:
        col = df[iso2].dropna()
        if col.empty:
            continue
        # look for current year or forecast
        for yr in [current_year + 1, current_year, current_year - 1]:
            matches = col[col.index.year == yr]
            if not matches.empty:
                out[iso2] = float(matches.iloc[-1])
                break
        else:
            out[iso2] = float(col.iloc[-1])
    # IMF uses ISO2 — remap to ISO3 for choropleth
    iso2_to_iso3 = {
        "DZ":"DZA","AO":"AGO","BJ":"BEN","BW":"BWA","BF":"BFA","BI":"BDI",
        "CM":"CMR","CV":"CPV","CF":"CAF","TD":"TCD","KM":"COM","CG":"COG",
        "CD":"COD","CI":"CIV","DJ":"DJI","GQ":"GNQ","ER":"ERI","SZ":"SWZ",
        "ET":"ETH","GA":"GAB","GM":"GMB","GH":"GHA","GN":"GIN","GW":"GNB",
        "KE":"KEN","LS":"LSO","LR":"LBR","LY":"LBY","MG":"MDG","MW":"MWI",
        "ML":"MLI","MR":"MRT","MU":"MUS","MA":"MAR","MZ":"MOZ","NA":"NAM",
        "NE":"NER","NG":"NGA","RW":"RWA","ST":"STP","SN":"SEN","SL":"SLE",
        "SO":"SOM","ZA":"ZAF","SS":"SSD","SD":"SDN","TZ":"TZA","TG":"TGO",
        "TN":"TUN","UG":"UGA","ZM":"ZMB","ZW":"ZWE","EG":"EGY",
    }
    return {iso2_to_iso3.get(k, k): v for k, v in out.items()}


def _wb_latest_by_country(df: pd.DataFrame) -> dict:
    """Extract latest available value per country from WB annual DataFrame."""
    if df is None or df.empty:
        return {}
    out = {}
    for iso in df.columns:
        col = df[iso].dropna()
        if not col.empty:
            out[iso] = float(col.iloc[-1])
    return out


def _wb_trend_by_country(df: pd.DataFrame) -> dict:
    """YoY % change of latest vs prior year. Positive = rising."""
    if df is None or df.empty:
        return {}
    out = {}
    for iso in df.columns:
        col = df[iso].dropna()
        if len(col) >= 2:
            prev, last = float(col.iloc[-2]), float(col.iloc[-1])
            if prev != 0:
                out[iso] = (last - prev) / abs(prev) * 100
    return out


with tabs[4]:
    st.subheader("EM & Africa — Macro Atlas")
    _cy = datetime.date.today().year

    # ── Summary cards ──────────────────────────────────────────────────────────
    em_regime = ind.get("em_regime", "unknown")
    eem_chg   = ind.get("eem_1m_chg", np.nan)
    fx_stress = ind.get("em_fx_stress_avg", np.nan)
    usd_vuln  = ind.get("usd_debt_vulnerability", "unknown")

    summary_cards = "".join([
        _card("EM Regime",         em_regime.upper(),   color=_regime_color(em_regime)),
        _card("EM Equity 1M",      f"{eem_chg*100:+.1f}%" if not _nan(eem_chg) else "N/A",
              "EEM ETF", "#c0392b" if not _nan(eem_chg) and eem_chg < -0.03 else "#27ae60"),
        _card("EM FX Stress Avg",  f"{fx_stress*100:+.1f}%" if not _nan(fx_stress) else "N/A",
              "1M avg depreciation",
              "#c0392b" if not _nan(fx_stress) and fx_stress > 0.03 else "#27ae60"),
        _card("USD Debt Vulnerability", usd_vuln.upper() if usd_vuln else "N/A",
              color=_regime_color(usd_vuln or "unknown")),
    ])
    c_sum1, c_sum2 = st.columns(2)
    with c_sum1:
        st.markdown(summary_cards, unsafe_allow_html=True)
    with c_sum2:
        # EM FX cards
        st.markdown("**EM FX Performance — Yahoo Finance**")
        if mkt is not None:
            fx_pairs = {
                "USD/BRL":"usdbrl","USD/ZAR":"usdzar","USD/TRY":"usdtry",
                "USD/CNH":"usdcnh","USD/INR":"usdinr","USD/MXN":"usdmxn",
            }
            cells = []
            for label, key in fx_pairs.items():
                s = mkt.get(key)
                if s is not None and len(s) >= 22:
                    last = float(s.iloc[-1])
                    chg_1m = last / float(s.iloc[-22]) - 1
                    color = "#c0392b" if chg_1m > 0.01 else "#27ae60" if chg_1m < -0.01 else "#2c3e50"
                    cells.append(
                        f'<div class="sig-box" style="border-left:5px solid {color};'
                        f'padding:6px 14px;margin:3px 0;">'
                        f'<span style="font-weight:700;font-size:0.86rem;">{label}</span>'
                        f'<span style="float:right;">{last:.4f}'
                        f'&nbsp;&nbsp;<b style="color:{color};">{chg_1m*100:+.1f}%</b></span></div>'
                    )
            if cells:
                st.markdown("".join(cells), unsafe_allow_html=True)

    # ── Eurobond alert ─────────────────────────────────────────────────────────
    st.markdown(
        '<div class="sig-box sig-warning" style="margin:10px 0 4px 0;">'
        '<div class="sig-title">📌 Eurobond Maturity Wall 2025–2027</div>'
        '<div class="sig-body">Ghana, Kenya, Ethiopia, Egypt each face significant Eurobond '
        'redemptions. Monitor FX reserve cover and IMF program status.</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 4 Choropleth maps ──────────────────────────────────────────────────────
    st.markdown("#### Africa Macro Atlas — IMF & World Bank")
    st.caption(f"IMF forecasts · World Bank annual data · Data as of latest available · Current year reference: {_cy}")

    # Map 1 & 2: GDP Growth + CPI
    mc1, mc2 = st.columns(2)
    with mc1:
        gdp_map = _imf_latest_by_country(imf_data, "gdp_growth", _cy) if imf_data else {}
        if gdp_map:
            fig = _make_africa_choropleth(
                gdp_map, f"GDP Real Growth % — IMF ({_cy} forecast)",
                colorscale="RdYlGn", reversescale=False, suffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("GDP growth data unavailable")
    with mc2:
        cpi_map = _imf_latest_by_country(imf_data, "inflation", _cy) if imf_data else {}
        if cpi_map:
            fig = _make_africa_choropleth(
                cpi_map, f"CPI Inflation % — IMF ({_cy} forecast)",
                colorscale="RdYlGn", reversescale=True, suffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("CPI data unavailable")

    # Map 3 & 4: FX Reserves trend + Debt Service
    mc3, mc4 = st.columns(2)
    with mc3:
        fx_trend = _wb_trend_by_country(fx_res) if fx_res is not None else {}
        if fx_trend:
            fig = _make_africa_choropleth(
                fx_trend, "FX Reserves — YoY Change % (World Bank)",
                colorscale="RdYlGn", reversescale=False, suffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("FX reserve trend data unavailable")
    with mc4:
        dsv_map = _wb_latest_by_country(dservice) if dservice is not None else {}
        if dsv_map:
            fig = _make_africa_choropleth(
                dsv_map, "Debt Service as % of GNI (World Bank)",
                colorscale="RdYlGn", reversescale=True, suffix="%GNI",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Debt service data unavailable")

    # Map 5: GNI per capita (income / poverty proxy)
    gni_map = _wb_latest_by_country(gni_data) if gni_data is not None else {}
    if gni_map:
        st.markdown("#### Income Level — GNI per Capita (World Bank, current USD)")
        fig = _make_africa_choropleth(
            gni_map, "GNI per Capita (USD) — World Bank",
            colorscale="Viridis", reversescale=False, suffix="USD",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Govt debt map
    debt_map = _imf_latest_by_country(imf_data, "gov_debt", _cy) if imf_data else {}
    if debt_map:
        st.markdown("#### Govt Debt % GDP — IMF")
        fig = _make_africa_choropleth(
            debt_map, f"Government Debt % GDP — IMF ({_cy})",
            colorscale="RdYlGn", reversescale=True, suffix="%GDP",
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MARKET-IMPLIED
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Market-Implied Forward Layer")
    st.caption(f"Source: {mkt_src} | Breakevens: {fred_src}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**VIX & Volatility**")
        vix = ind.get("vix", np.nan)
        vix_regime = ind.get("vix_regime", "unknown")
        vix_c = "#c0392b" if not _nan(vix) and vix > 25 else "#d35400" if not _nan(vix) and vix > 18 else "#27ae60"
        st.markdown("".join([
            _card("VIX",               f"{vix:.1f}" if not _nan(vix) else "N/A", color=vix_c),
            _card("VIX Regime",        vix_regime.upper(), color=_regime_color(vix_regime)),
            _card("VIX Term Structure",ind.get("vix_term_structure","unknown").upper(),
                  color=_regime_color(ind.get("vix_term_structure","unknown"))),
            _card("MOVE Proxy",        f"{ind.get('move_proxy',np.nan):.0f}" if not _nan(ind.get("move_proxy")) else "N/A",
                  "FRED yields", "#2c3e50"),
        ]), unsafe_allow_html=True)

    with c2:
        st.markdown("**Inflation Expectations (FRED)**")
        be5  = ind.get("breakeven5y",   np.nan)
        be55 = ind.get("breakeven5y5y", np.nan)
        tp   = ind.get("term_premium",  np.nan)
        inf_regime = ind.get("inflation_regime","unknown")
        be5_c  = "#c0392b" if not _nan(be5)  and be5  > 2.8 else "#27ae60" if not _nan(be5)  and be5  < 2.0 else "#2980b9"
        be55_c = "#c0392b" if not _nan(be55) and be55 > 2.8 else "#27ae60" if not _nan(be55) and be55 < 2.0 else "#2980b9"
        st.markdown("".join([
            _card("5Y Breakeven",    f"{be5:.2f}%"  if not _nan(be5)  else "N/A", "FRED", be5_c),
            _card("5Y5Y Forward",    f"{be55:.2f}%" if not _nan(be55) else "N/A", "FRED", be55_c),
            _card("Term Premium",    f"{tp:.2f}%"   if not _nan(tp)   else "N/A", "FRED", "#2980b9"),
            _card("Inflation Regime",inf_regime.upper(), color=_regime_color(inf_regime)),
        ]), unsafe_allow_html=True)

    with c3:
        st.markdown("**Cross-Asset Signals**")
        corr = ind.get("eq_bond_corr", np.nan)
        corr_regime  = ind.get("correlation_regime","unknown")
        cu_regime    = ind.get("copper_gold_regime","unknown")
        systemic     = ind.get("systemic_stress_signal", False)
        systemic_c   = "#c0392b" if systemic else "#27ae60"
        st.markdown("".join([
            _card("Eq–Bond Corr",     f"{corr:.2f}" if not _nan(corr) else "N/A",
                  color="#c0392b" if not _nan(corr) and corr > 0.3 else "#27ae60"),
            _card("Correlation Regime",corr_regime.upper(), color=_regime_color(corr_regime)),
            _card("Cu/Gold Regime",    cu_regime.upper(), color=_regime_color(cu_regime)),
            _card("Systemic Stress",   "YES ⚠️" if systemic else "No", color=systemic_c),
        ]), unsafe_allow_html=True)

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
                    srcs = ", ".join(inf.get("data_sources", []))
                    st.markdown(
                        f'<div class="inf-{conf_key}">'
                        f'<div class="sig-title">{inf.get("icon","📌")} {inf.get("category","")} · <i>{inf.get("horizon","")}</i></div>'
                        f'<div class="sig-body"><b>{inf.get("statement","")}</b></div>'
                        f'<div class="sig-body">→ {inf.get("implication","")}</div>'
                        f'<div class="sig-watch">Trigger: {inf.get("trigger","")} | Context: {inf.get("context","")} | Sources: {srcs}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
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

    # WB CMO — fertilizer / commodity prices
    if cmo_data is not None:
        fert_keys = [k for k in cmo_data
                     if any(x in k.lower() for x in ["urea","dap","potash","phosphate","fertiliz"])]
        other_keys = [k for k in cmo_data if k not in fert_keys and
                      not any(x in k.lower() for x in ["crude","oil","gas","wheat","corn","rice","maize"])]
        if fert_keys:
            st.divider()
            st.subheader("🧪 Fertilizer Prices — World Bank CMO")
            fig = go.Figure()
            for k in fert_keys[:5]:
                s = cmo_data[k].dropna()
                if not s.empty:
                    fig.add_trace(go.Scatter(x=s.index, y=s, name=k, mode="lines+markers"))
            fig.update_layout(title=f"Fertilizer Prices (USD/mt) — {cmo_src}",
                              yaxis_title="USD/mt", height=280)
            st.plotly_chart(fig, use_container_width=True)
        if other_keys:
            st.divider()
            st.subheader("📊 Other CMO Commodities — World Bank")
            fig = go.Figure()
            for k in other_keys[:6]:
                s = cmo_data[k].dropna()
                if not s.empty:
                    fig.add_trace(go.Scatter(x=s.index, y=s, name=k, mode="lines+markers"))
            fig.update_layout(title=f"World Bank CMO Commodity Prices — {cmo_src}", height=280)
            st.plotly_chart(fig, use_container_width=True)

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
            strength = prop.get("strength", "")
            icon_str = prop.get("icon", "🔗")
            srcs     = ", ".join(prop.get("data_sources", []))
            st.markdown(
                f'<div class="prop-chain">'
                f'<div class="sig-title">{icon_str} {prop.get("from_sector","")} → {prop.get("to_sector","")} <span style="font-weight:400;color:#7f8c8d;font-size:0.84rem">({strength} linkage)</span></div>'
                f'<div class="sig-body"><b>{prop.get("headline","")}</b></div>'
                f'<div class="sig-body">{prop.get("mechanism","")}</div>'
                f'<div class="sig-body">→ {prop.get("implication","")}</div>'
                f'<div class="sig-watch">Signal: {prop.get("signal","")} | Sources: {srcs}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Sector connectivity diagram
    st.divider()
    st.subheader("Known Sector Linkages")
    _linkages = [
        ("⚡ Energy (Gas)",        "Haber-Bosch feedstock",       "🧪 Chemicals / Fertilizers", True,  "AGSI+"),
        ("🧪 Chemicals",           "Fertilizer supply crunch",    "🌾 Agriculture",              True,  "AGSI+ + FAO"),
        ("🌾 Agriculture",         "Food CPI (40–60% EM basket)", "🌍 EM/Africa Stress",         True,  "FAO + Yahoo"),
        ("⚡ Energy (Oil)",        "Transport & diesel cost",     "🏭 Industrials, 🌾 Agri",     True,  "Yahoo Finance"),
        ("⛏️ Critical Minerals", "EV/grid capex signal",        "🏭 Industrials",              True,  "Yahoo Finance"),
        ("💱 Macro (USD)",         "Debt service cost",           "🌍 EM/Africa",                True,  "Yahoo + WB"),
        ("🏭 Industrials (IP)",    "Base metal demand",           "⛏️ Critical Minerals",       True,  "FRED"),
        ("🚢 Shipping rates",      "Supply chain cost push",      "🏭 Industrials, 🧪 Chemicals",False, "PLACEHOLDER"),
        ("🌿 Fertilizer prices",   "Crop input cost",             "🌾 Agriculture",              False, "PLACEHOLDER"),
        ("🔩 LME inventories",     "Metal supply signal",         "⛏️ Critical Minerals",       False, "PLACEHOLDER"),
    ]
    rows_html = "".join(
        f'<div class="sig-box" style="border-left:5px solid {"#27ae60" if live else "#95a5a6"};'
        f'padding:7px 14px;margin:3px 0;display:flex;align-items:center;gap:0;">'
        f'<span style="min-width:180px;font-weight:600;font-size:0.84rem;">{frm}</span>'
        f'<span style="min-width:220px;font-size:0.82rem;color:#555;">→ {channel}</span>'
        f'<span style="min-width:200px;font-size:0.84rem;font-weight:600;">{to}</span>'
        f'<span style="margin-left:auto;font-size:0.78rem;color:{"#27ae60" if live else "#95a5a6"};">'
        f'{"✅" if live else "⚫"} {src}</span>'
        f'</div>'
        for frm, channel, to, live, src in _linkages
    )
    st.markdown(rows_html, unsafe_allow_html=True)


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
            bullets_html = "".join(f'<div class="sig-body">• {b}</div>' for b in sec["bullets"])
            st.markdown(
                f'<div class="sig-box sig-info">'
                f'<div class="sig-title">{sec["title"]}</div>'
                f'<div class="sig-body" style="color:#555;margin-bottom:6px;">{sec["body"]}</div>'
                f'{bullets_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

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
    """Render a note_generator section as a flat rectangle — no click needed."""
    level_key = sec.get("level", "info")
    icons = {"alert": "🔴", "warning": "🟠", "good": "🟢", "info": "🔵"}
    icon  = icons.get(level_key, "⚪")
    css   = f"sig-{level_key}" if level_key in ("alert","warning","good") else "sig-info"

    body_rows = []
    ph_rows   = []
    for row in sec.get("rows", []):
        if row.startswith("⚫") or "PLACEHOLDER" in row.upper():
            ph_rows.append(row)
        else:
            body_rows.append(row)

    rows_html = "".join(
        f'<div class="sig-body">{r}</div>' if r.strip() else "<br>"
        for r in body_rows
    )
    ph_html = "".join(
        f'<div class="placeholder-box" style="margin:3px 0;font-size:0.80rem;">{r}</div>'
        for r in ph_rows
    )
    st.markdown(
        f'<div class="sig-box {css}">'
        f'<div class="sig-title">{icon} {sec["title"]}</div>'
        f'{rows_html}{ph_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


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
