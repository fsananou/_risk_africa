"""
sector_dependencies.py — Cross-sector propagation logic.

Maps real indicator data to sector stress signals and downstream propagation chains.
Only fires when triggered by real data (no placeholders trigger rules).

Each propagation insight:
    from_sector  : str
    to_sector    : str
    mechanism    : str   (the transmission channel)
    signal       : str   (what indicator triggered it)
    strength     : "strong" | "moderate" | "weak"
    horizon      : str
    headline     : str
    implication  : str
    data_sources : list[str]

Also returns per-sector stress snapshots.
"""

from __future__ import annotations

import numpy as np

try:
    from config import SECTOR_ICON
except ImportError:
    SECTOR_ICON = {
        "Energy":           "⚡",
        "Agriculture":      "🌾",
        "Chemicals":        "🧪",
        "Industrials":      "🏭",
        "Technology":       "💻",
        "Critical Minerals":"⛏️",
    }


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(float(v)))


def _known(v, *invalid) -> bool:
    return not _nan(v) and v not in invalid


# ══════════════════════════════════════════════════════════════════════════════
# Sector stress assessment
# ══════════════════════════════════════════════════════════════════════════════

def assess_sectors(ind: dict) -> dict[str, dict]:
    """
    Return a stress assessment for each sector based on available real indicators.
    Each sector: {status, headline, indicators_used, data_available}
    """
    sectors = {}

    # ── ENERGY ────────────────────────────────────────────────────────────────
    energy_signals = []
    if _known(ind.get("oil_regime"), "unknown"):
        energy_signals.append(f"Oil: {ind['oil_regime']}")
    if _known(ind.get("eu_gas_storage_regime"), "unknown"):
        energy_signals.append(f"EU gas: {ind['eu_gas_storage_regime']} ({ind.get('eu_gas_storage_pct', '?'):.0f}% full)" if not _nan(ind.get("eu_gas_storage_pct")) else f"EU gas: {ind['eu_gas_storage_regime']}")
    if _known(ind.get("oil_inventory_regime"), "unknown"):
        energy_signals.append(f"US oil inventory: {ind['oil_inventory_regime']}")
    if _known(ind.get("natgas_1m_chg"), "unknown") and not _nan(ind.get("natgas_1m_chg")):
        chg = ind["natgas_1m_chg"]
        energy_signals.append(f"US nat gas: {chg*100:+.1f}% 1M")

    energy_crisis = (ind.get("eu_gas_storage_regime") == "crisis" or
                     ind.get("oil_regime") == "surging" or
                     ind.get("oil_inventory_regime") == "stress")
    energy_stress = (ind.get("eu_gas_storage_regime") == "stress")

    sectors["Energy"] = {
        "status": "crisis" if energy_crisis else "stress" if energy_stress else
                  "normal" if energy_signals else "no_data",
        "headline": "; ".join(energy_signals[:3]) if energy_signals else "Insufficient real data",
        "indicators_used": energy_signals,
        "data_available": bool(energy_signals),
        "data_sources": ["yfinance (Brent, WTI, Nat Gas)", "EIA (US inventories)", "AGSI+ (EU gas)"],
    }

    # ── AGRICULTURE ───────────────────────────────────────────────────────────
    agri_signals = []
    if _known(ind.get("fao_fpi_regime"), "unknown"):
        v = ind.get("fao_fpi", np.nan)
        agri_signals.append(f"FAO FPI: {ind['fao_fpi_regime']}" + (f" ({v:.0f})" if not _nan(v) else ""))
    if _known(ind.get("agri_stress_regime"), "unknown"):
        agri_signals.append(f"Crop prices: {ind['agri_stress_regime']}")
    if not _nan(ind.get("wheat_1m_chg")):
        agri_signals.append(f"Wheat: {ind['wheat_1m_chg']*100:+.1f}% 1M")
    if not _nan(ind.get("corn_1m_chg")):
        agri_signals.append(f"Corn: {ind['corn_1m_chg']*100:+.1f}% 1M")

    agri_status = ind.get("fao_fpi_regime", ind.get("agri_stress_regime", "unknown"))
    sectors["Agriculture"] = {
        "status": agri_status if agri_status not in ("unknown",) else
                  ("no_data" if not agri_signals else "normal"),
        "headline": "; ".join(agri_signals[:3]) if agri_signals else "Insufficient real data",
        "indicators_used": agri_signals,
        "data_available": bool(agri_signals),
        "data_sources": ["FAO FAOSTAT (Food Price Index)", "yfinance (wheat ZW=F, corn ZC=F)"],
    }

    # ── CHEMICALS ─────────────────────────────────────────────────────────────
    # Chemicals stress proxied by: EU gas storage (feedstock) + oil (naphtha feedstock)
    chem_signals = []
    if _known(ind.get("eu_gas_storage_regime"), "unknown"):
        chem_signals.append(f"Gas feedstock (EU): {ind['eu_gas_storage_regime']}")
    if _known(ind.get("oil_regime"), "unknown"):
        chem_signals.append(f"Oil/naphtha feedstock: {ind['oil_regime']}")
    chem_note = "Fertilizer prices: PLACEHOLDER (no free API)"

    chem_crisis = (ind.get("eu_gas_storage_regime") in ("crisis",) or
                   (ind.get("eu_gas_storage_regime") == "stress" and ind.get("oil_regime") == "surging"))
    sectors["Chemicals"] = {
        "status": "crisis" if chem_crisis else
                  "stress" if ind.get("eu_gas_storage_regime") == "stress" else
                  "normal" if chem_signals else "no_data",
        "headline": ("; ".join(chem_signals) + " | " + chem_note) if chem_signals else chem_note,
        "indicators_used": chem_signals,
        "data_available": bool(chem_signals),
        "placeholders": ["Fertilizer prices (World Bank Pink Sheet — no free REST API)"],
        "data_sources": ["AGSI+ (gas feedstock proxy)", "yfinance (oil/naphtha proxy)"],
    }

    # ── INDUSTRIALS ───────────────────────────────────────────────────────────
    ind_signals = []
    if _known(ind.get("indpro_regime"), "unknown"):
        v = ind.get("indpro_yoy", np.nan)
        ind_signals.append(f"US indpro: {ind['indpro_regime']}" + (f" ({v:+.1f}% YoY)" if not _nan(v) else ""))
    if _known(ind.get("oecd_cli_regime"), "unknown"):
        v = ind.get("oecd_cli_oecdall", np.nan)
        ind_signals.append(f"OECD CLI: {ind['oecd_cli_regime']}" + (f" ({v:.1f})" if not _nan(v) else ""))
    if _known(ind.get("copper_regime"), "unknown"):
        ind_signals.append(f"Copper (industrial proxy): {ind['copper_regime']}")
    ind_signals.append("Shipping: PLACEHOLDER (no free API)")

    ind_status = (
        "contraction" if ind.get("indpro_regime") == "contraction" or ind.get("oecd_cli_regime") == "contracting" else
        "slowing"     if ind.get("indpro_regime") == "slowing" else
        "normal"      if ind_signals else "no_data"
    )
    sectors["Industrials"] = {
        "status": ind_status,
        "headline": "; ".join(ind_signals[:3]),
        "indicators_used": ind_signals,
        "data_available": bool([s for s in ind_signals if "PLACEHOLDER" not in s]),
        "placeholders": ["Container shipping rates (Freightos — paid subscription)"],
        "data_sources": ["FRED (US Industrial Production)", "OECD SDMX (CLI)", "yfinance (copper)"],
    }

    # ── TECHNOLOGY ────────────────────────────────────────────────────────────
    tech_signals = []
    if not _nan(ind.get("xlf_1m_chg")):  # Using sector ETF as proxy
        tech_signals.append(f"XLF (financials — tech proxy): {ind.get('xlf_1m_chg',0)*100:+.1f}% 1M")
    tech_note = "Semiconductor sales: PLACEHOLDER (WSTS member-only)"
    sectors["Technology"] = {
        "status": "no_data",   # No direct real tech indicator available without subscription
        "headline": tech_note,
        "indicators_used": tech_signals,
        "data_available": False,
        "placeholders": [
            "Semiconductor sales (WSTS — membership required)",
            "Export control data (no free API)",
        ],
        "data_sources": ["yfinance (sector ETF proxy only — limited signal)"],
    }

    # ── CRITICAL MINERALS ─────────────────────────────────────────────────────
    min_signals = []
    if not _nan(ind.get("copper_1m_chg")):
        min_signals.append(f"Copper: {ind.get('copper_1m_chg',0)*100:+.1f}% 1M")
    if not _nan(ind.get("nickel_1m_chg")):
        min_signals.append(f"Nickel: {ind.get('nickel_1m_chg',0)*100:+.1f}% 1M")
    min_note = "Lithium/Cobalt: PLACEHOLDER (Fastmarkets/LME — no free API)"

    cu_stress = _known(ind.get("copper_regime"), "unknown") and ind.get("copper_regime") == "falling"
    sectors["Critical Minerals"] = {
        "status": "stress" if cu_stress else "normal" if min_signals else "no_data",
        "headline": ("; ".join(min_signals) + " | " + min_note) if min_signals else min_note,
        "indicators_used": min_signals,
        "data_available": bool(min_signals),
        "placeholders": [
            "Lithium prices (Fastmarkets — commercial)",
            "Cobalt prices (LME — subscription)",
            "LME inventory data (LME Data — subscription)",
        ],
        "data_sources": ["yfinance (copper HG=F, nickel NI=F)"],
    }

    return sectors


# ══════════════════════════════════════════════════════════════════════════════
# Cross-sector propagation
# ══════════════════════════════════════════════════════════════════════════════

def run(ind: dict) -> list[dict]:
    """
    Fire cross-sector propagation rules based on real indicator data.
    Returns list of propagation insights.
    """
    out: list[dict] = []

    def add(from_sector, to_sector, mechanism, signal, strength,
            horizon, headline, implication, sources):
        out.append({
            "from_sector":   from_sector,
            "to_sector":     to_sector,
            "mechanism":     mechanism,
            "signal":        signal,
            "strength":      strength,
            "horizon":       horizon,
            "headline":      headline,
            "implication":   implication,
            "data_sources":  sources,
            "icon":          f"{SECTOR_ICON.get(from_sector,'⚡')} → {SECTOR_ICON.get(to_sector,'📦')}",
        })

    eu_gas_rg  = ind.get("eu_gas_storage_regime", "unknown")
    oil_rg     = ind.get("oil_regime", "unknown")
    oil_inv_rg = ind.get("oil_inventory_regime", "unknown")
    fao_rg     = ind.get("fao_fpi_regime", "unknown")
    agri_rg    = ind.get("agri_stress_regime", "unknown")
    cu_rg      = ind.get("copper_regime", "unknown")
    indpro_rg  = ind.get("indpro_regime", "unknown")
    dollar_rg  = ind.get("dollar_regime", "unknown")
    em_rg      = ind.get("em_regime", "unknown")
    vix_rg     = ind.get("vix_regime", "unknown")

    # ── Chain 1: Gas → Chemicals → Agriculture ────────────────────────────────
    if _known(eu_gas_rg, "unknown") and eu_gas_rg in ("stress","crisis"):
        add("Energy", "Chemicals",
            "Natural gas = primary feedstock for nitrogen fertilizer synthesis (Haber-Bosch process)",
            f"EU gas storage {eu_gas_rg} (AGSI+)",
            "strong" if eu_gas_rg == "crisis" else "moderate",
            "near-term (1-4 weeks)",
            "Low EU gas → ammonia/urea plant curtailments → nitrogen fertilizer supply crunch",
            "European fertilizer production margins compressed. Global nitrogen supply tightens.",
            ["AGSI+ (GIE)"])

        add("Chemicals", "Agriculture",
            "Fertilizer shortage → higher input costs → reduced crop applications → lower yields next cycle",
            f"Triggered by EU gas storage {eu_gas_rg} → fertilizer supply crunch",
            "strong" if eu_gas_rg == "crisis" else "moderate",
            "medium-term (1-6 months)",
            "Fertilizer crunch → lower crop yields in next planting season → food price persistence",
            "Global food price inflation likely to be sustained. EM net food importers most at risk.",
            ["AGSI+ (GIE)", "FAO"])

    # ── Chain 2: Gas/Oil → Chemicals → EM ────────────────────────────────────
    if _known(oil_rg, "unknown") and oil_rg == "surging" and \
       _known(eu_gas_rg, "unknown") and eu_gas_rg in ("stress","crisis"):
        add("Energy", "Chemicals",
            "Oil spike raises naphtha prices (petrochemical feedstock); gas shortage adds to input cost pressure",
            "Oil surging (yfinance) + EU gas stress (AGSI+)",
            "strong",
            "near-term (1-4 weeks)",
            "Oil surge + gas shortage = double feedstock shock for petrochemicals",
            "Polymer, resin, and specialty chemical prices likely to spike. Manufacturing margins compress.",
            ["yfinance", "AGSI+ (GIE)"])

    # ── Chain 3: Agriculture → EM stress ─────────────────────────────────────
    if _known(fao_rg, "unknown") and fao_rg in ("stress","crisis"):
        add("Agriculture", "EM/Africa",
            "High food prices = largest share of EM CPI basket (40-60%) → inflation → FX pressure",
            f"FAO Food Price Index {fao_rg} (FAO FAOSTAT)",
            "strong" if fao_rg == "crisis" else "moderate",
            "medium-term (1-6 months)",
            "Food price crisis → EM inflation spike → central bank tightening → growth slowdown",
            "Food-importing EM (Egypt, Ethiopia, Nigeria, Kenya) face acute CPI and balance-of-payments stress.",
            ["FAO", "yfinance"])

    # ── Chain 4: Oil spike → Transport → Retail inflation ─────────────────────
    if _known(oil_rg, "unknown") and oil_rg == "surging":
        add("Energy", "Industrials",
            "Oil price → transport and logistics costs → input cost inflation across supply chains",
            "Oil surging (yfinance)",
            "strong",
            "near-term (1-4 weeks)",
            "Oil surge → transport cost surge → cost-push inflation across manufacturing and retail",
            "Trucking, aviation, shipping margins compress. Retail prices lag by 4-8 weeks. EM most affected.",
            ["yfinance"])

        add("Energy", "Agriculture",
            "Oil price → diesel cost for farming equipment and irrigation → input cost pressure",
            "Oil surging (yfinance)",
            "moderate",
            "medium-term (1-6 months)",
            "Oil surge → higher farm diesel costs → raises cost of crop production globally",
            "Agriculture operating costs rise. Food price floor higher. Smallholder farmers in EM most squeezed.",
            ["yfinance"])

    # ── Chain 5: Copper tightness → Electrification slowdown → Oil demand ─────
    if _known(cu_rg, "unknown") and cu_rg == "falling":
        add("Critical Minerals", "Industrials",
            "Copper = essential conductor for grid infrastructure, EV, and renewables",
            "Copper falling (yfinance) — may signal demand destruction",
            "moderate",
            "structural (6-18 months)",
            "Copper price decline may reflect slower EV/grid investment → electrification capex slowdown",
            "Fossil fuel demand structural decline slows. Oil demand stays higher for longer.",
            ["yfinance"])

    # ── Chain 6: Copper rising → Critical mineral demand → Supply bottleneck ──
    if _known(cu_rg, "unknown") and cu_rg == "rising":
        add("Industrials", "Critical Minerals",
            "Industrial/EV demand drives copper → creates co-demand for lithium, cobalt, nickel",
            "Copper rising (yfinance)",
            "moderate",
            "structural (6-18 months)",
            "Copper rally driven by electrification = rising demand for all EV battery minerals",
            "Lithium, cobalt, nickel supply chains under pressure. DRC, Congo, Australia, Chile exporters benefit.",
            ["yfinance"])

    # ── Chain 7: EM stress → Agri import collapse ─────────────────────────────
    if _known(em_rg, "unknown") and em_rg in ("stress","crisis") and \
       _known(fao_rg, "unknown") and fao_rg in ("stress","crisis"):
        add("EM/Africa", "Agriculture",
            "EM sovereign stress → import financing constraints → food import capacity collapses",
            f"EM stress (yfinance) + FAO FPI {fao_rg} (FAO)",
            "strong",
            "near-term (1-4 weeks)",
            "EM sovereign stress + food prices → food import financing crunch → acute food insecurity",
            "WFP/FAO emergency appeal likely for Sub-Saharan Africa and MENA. Watch Egypt, Ethiopia, Sudan.",
            ["yfinance", "FAO"])

    # ── Chain 8: IP contraction → Commodity demand destruction ───────────────
    if _known(indpro_rg, "unknown") and indpro_rg == "contraction":
        add("Industrials", "Critical Minerals",
            "Industrial production contraction → lower base metals demand → price pressure",
            "US IP contracting (FRED)",
            "moderate",
            "medium-term (1-6 months)",
            "IP contraction → softer copper, nickel demand → commodity exporter revenue declines",
            "Africa mining-dependent economies (DRC, Zambia, South Africa) face fiscal deterioration.",
            ["FRED"])

    # ── Chain 9: Strong dollar → EM debt service → Sovereign stress ──────────
    if _known(dollar_rg, "unknown") and dollar_rg in ("strong","very_strong") and \
       _known(em_rg, "unknown") and em_rg in ("stress","crisis"):
        add("Macro", "EM/Africa",
            "USD strength → higher cost of servicing USD-denominated sovereign debt",
            f"Strong USD (DXY — yfinance) + EM stress",
            "strong",
            "medium-term (1-6 months)",
            "USD strength + EM stress = debt service spiral for USD-denominated sovereign borrowers",
            "Countries with large USD debt maturity walls (2025-2027) most exposed. IMF programs likely.",
            ["yfinance", "World Bank"])

    return out
