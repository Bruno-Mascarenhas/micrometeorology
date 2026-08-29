"""Column names of the operational WRF point record — the shared contract.

``series_operacional.dat`` is written by
:data:`micrometeorology.wrf.operational_record.OPERATIONAL_CATALOG` and read by
the climatology export, the two station-graph CLIs and the monitoring export.
Those readers used to name the columns with string literals, so a rename in the
producer reached none of them: the overlay simply stopped appearing, with no
error and no failing test.

Nothing is imported here on purpose — the producer builds its catalog from these
names and every consumer refers to them, so the link is checked when the module
loads rather than when a plot comes out empty.
"""

from __future__ import annotations

__all__ = [
    "ALBEDO",
    "EMISSIVITY",
    "ES_PA",
    "E_HPA",
    "GLW_W_M2",
    "GRDFLX_W_M2",
    "HFX_W_M2",
    "LH_W_M2",
    "LWDNB_W_M2",
    "LWUP_AIR_W_M2",
    "LWUP_W_M2",
    "OPERATIONAL_COLUMN_NAMES",
    "PBLH_M",
    "PRECIP_MM",
    "PSFC_HPA",
    "Q2_G_KG",
    "RH_PCT",
    "SST_C",
    "SWDDIF_FARMS_W_M2",
    "SWDDIF_W_M2",
    "SWDDIR_FARMS_W_M2",
    "SWDDIR_W_M2",
    "SWDNB_W_M2",
    "SWDOWN_FARMS_W_M2",
    "SWDOWN_W_M2",
    "SWUPB_W_M2",
    "SWUP_W_M2",
    "T2_C",
    "U10_M_S",
    "USTAR_M_S",
    "V10_M_S",
    "WIND_DIR_DEG",
    "WIND_SPEED_M_S",
]

T2_C = "t2_c"
RH_PCT = "rh_pct"
PSFC_HPA = "psfc_hpa"
E_HPA = "e_hpa"
ES_PA = "es_pa"
Q2_G_KG = "q2_g_kg"
WIND_SPEED_M_S = "wind_speed_m_s"
WIND_DIR_DEG = "wind_dir_deg"
U10_M_S = "u10_m_s"
V10_M_S = "v10_m_s"
SWDOWN_W_M2 = "swdown_w_m2"
SWDNB_W_M2 = "swdnb_w_m2"
SWDOWN_FARMS_W_M2 = "swdown_farms_w_m2"
SWUPB_W_M2 = "swupb_w_m2"
SWUP_W_M2 = "swup_w_m2"
SWDDIF_W_M2 = "swddif_w_m2"
SWDDIF_FARMS_W_M2 = "swddif_farms_w_m2"
SWDDIR_W_M2 = "swddir_w_m2"
SWDDIR_FARMS_W_M2 = "swddir_farms_w_m2"
GLW_W_M2 = "glw_w_m2"
LWDNB_W_M2 = "lwdnb_w_m2"
LWUP_W_M2 = "lwup_w_m2"
LWUP_AIR_W_M2 = "lwup_air_w_m2"
ALBEDO = "albedo"
EMISSIVITY = "emissivity"
HFX_W_M2 = "hfx_w_m2"
LH_W_M2 = "lh_w_m2"
GRDFLX_W_M2 = "grdflx_w_m2"
USTAR_M_S = "ustar_m_s"
PBLH_M = "pblh_m"
SST_C = "sst_c"
PRECIP_MM = "precip_mm"

#: Every column the operational record can carry, in write order.
OPERATIONAL_COLUMN_NAMES: tuple[str, ...] = (
    T2_C,
    RH_PCT,
    PSFC_HPA,
    E_HPA,
    ES_PA,
    Q2_G_KG,
    WIND_SPEED_M_S,
    WIND_DIR_DEG,
    U10_M_S,
    V10_M_S,
    SWDOWN_W_M2,
    SWDNB_W_M2,
    SWDOWN_FARMS_W_M2,
    SWUPB_W_M2,
    SWUP_W_M2,
    SWDDIF_W_M2,
    SWDDIF_FARMS_W_M2,
    SWDDIR_W_M2,
    SWDDIR_FARMS_W_M2,
    GLW_W_M2,
    LWDNB_W_M2,
    LWUP_W_M2,
    LWUP_AIR_W_M2,
    ALBEDO,
    EMISSIVITY,
    HFX_W_M2,
    LH_W_M2,
    GRDFLX_W_M2,
    USTAR_M_S,
    PBLH_M,
    SST_C,
    PRECIP_MM,
)
