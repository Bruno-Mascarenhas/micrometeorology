"""Physics tests for the derived surface radiation budget.

Four independent kinds of evidence that the published numbers are right:

1. **Closed form** — analytic limits and hand-computed values, so the formulas
   are pinned against arithmetic rather than against themselves.
2. **Published bounds** — QC envelopes and observed global means that a surface
   flux cannot fall outside (see ``_REFERENCES`` below). Every entry there
   carries a marker for how far it was actually verified, and the bounds that
   are only sanity rails say so rather than borrowing false authority.
3. **WRF's own fluxes** — where a wrfout carries the RRTMG bottom-of-atmosphere
   diagnostics (``LWUPB``/``SWUPB``), the reconstruction is compared against
   them cell by cell.
4. **The surface energy budget on the operational run** — the 2026 nests carry
   no ``LWUPB``, so there LWUP is validated by requiring ``Rn + G = H + LE``
   to close over whole days. It closes to 0.6-1.4 W/m2 on all four nests, and
   to +22 W/m2 if the reflected term is dropped: the budget alone discriminates
   the right formula by a factor of ~20 on files with no ground truth in them.

Everything reading ``data/`` skips when that gitignored bulk data is not on
this machine — the closed-form and identity tests above carry the suite on
their own.

Plus the wiring: dispatcher registration, the daylight gate, and the
missing-input skip path.
"""

from datetime import UTC, datetime
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from micrometeorology.common.types import VARIABLE_NETCDF_MAP, WRFVariable
from micrometeorology.wrf import variables
from micrometeorology.wrf.reader import WRFDataset
from micrometeorology.wrf.value_source import (
    DAYLIGHT_ONLY_VARIABLES,
    DERIVED_RADIATION_SOURCES,
    build_value_frame_source,
    publishes_step,
)
from micrometeorology.wrf.variables import (
    MIN_COSZEN_FOR_CLEARNESS,
    SOLAR_CONSTANT,
    STEFAN_BOLTZMANN,
    _eccentricity_correction,
    compute_clearness_index,
    compute_net_longwave,
    compute_net_radiation,
    compute_net_shortwave,
    compute_sky_emissivity,
    compute_upwelling_longwave,
    compute_upwelling_shortwave,
)

# Sources the numeric assertions below are taken from, in one place so a bound
# that looks arbitrary can be traced to the text that fixes it.
#
# Each entry is marked with how far it was actually checked, because several
# of the classic radiation papers sit behind publisher blocks and are known
# here only through other authors quoting them:
#   [text]      full text retrieved and read
#   [quoted]    formula/values confirmed verbatim in a retrieved source that
#               quotes the original; the original itself was not retrieved
#   [record]    bibliographic record confirmed (Crossref/OpenAlex); contents
#               not retrieved
#   [standard]  standard textbook material, cited without page numbers because
#               a specific edition/page was not checked
_REFERENCES = """
The graybody form itself -- the primary evidence
- [text] WRF source. RRTMG's rtrnmc forms the upward radiance per band as
  radlu = rad0 + reflect*radld, with rad0 = semiss*B(Ts) and reflect =
  1 - semiss (phys/module_ra_rrtmg_lw.F, "Add in specular reflection of
  surface downward radiance"). The urban branch of Noah writes it longhand:
  rl_up_rural = -emiss*sigma*tsk**4 - (1.-emiss)*glw
  (phys/module_sf_noahdrv.F). Noah's net-longwave term emiss*GLW -
  emiss*sigma*T1**4 is the same identity with the reflected part already
  cancelled, not a scheme that omits it. So compute_upwelling_longwave is not
  an approximation of WRF -- it is WRF's own expression.
- [text] WRF carries STBOLT = 5.67051e-8 (share/module_model_constants.F) and
  Noah rounds to 5.67e-8, against the exact CODATA value used here. Worth
  ~0.01 W/m2 -- 1% of the canopy-temperature bias, hence negligible.

Surface radiation budget
- [standard] Oke, T. R. Boundary Layer Climates, 2nd ed., Routledge, 1987;
  Stull, R. B. An Introduction to Boundary Layer Meteorology, Kluwer, 1988;
  Arya, S. P. Introduction to Micrometeorology, 2nd ed., Academic Press, 2001.
  The four-stream budget Rn = (K-down - K-up) + (L-down - L-up) and graybody
  emission with the reflected (1 - eps) L-down term. No page numbers claimed.
- [text] Trenberth, K. E., Fasullo, J. T., & Kiehl, J. (2009). Bull. Amer.
  Meteor. Soc. 90(3), 311-323. doi:10.1175/2008BAMS2634.1. Global mean surface
  LWup 396 W/m2, LWdn 333, net thermal -63.
- [text] Wild, M., et al. (2015). Climate Dynamics 44, 3393-3429.
  doi:10.1007/s00382-014-2430-z. Land-only annual means: LWdn 306, net thermal
  -66, hence LWup 372 W/m2; net all-wave Rn 70 W/m2.
- [text] Oke (1987) Ch. 1 Sect. 3(c) p. 22 (full-text scan): net longwave "is
  usually negative, and relatively small (75 to 125 W/m2)".
- [text] Long, C. N., & Dutton, E. G. BSRN Global Network recommended QC
  tests, V2.0. Downwelling LW physically-possible 40-700 W/m2, extremely-rare
  60-500; UPwelling LW physically-possible 40-900, extremely-rare 60-700; the
  air-temperature envelope sigma*(Ta-15)^4 < LWup < sigma*(Ta+25)^4; and
  0.4*sigma*Ta^4 < LWdn < sigma*Ta^4 + 25, the globally valid bound on
  effective sky emissivity used below.

Sky emissivity
- [text] Yang, J., Quan, W., et al. (2023). Atmos. Chem. Phys. 23(8),
  4419-4430. doi:10.5194/acp-23-4419-2023. Sect. 3.1 Eq. (1) defines
  effective atmospheric emissivity as eps = DLR / (sigma*Ta^4) with Ta the
  screen-level air temperature -- exactly the definition implemented here.
- [text] Morales-Salinas, L., et al. (2023). Sci. Rep. 13, 14465.
  doi:10.1038/s41598-023-40499-6. Global clear-sky emissivity range
  0.34-0.97, mean 0.73.
- [text] Herrero, J., & Polo, M. J. (2012). Hydrol. Earth Syst. Sci. 16,
  3139-3147. doi:10.5194/hess-16-3139-2012. Six years of pyrgeometer data at
  2500 m: clear-sky mean 0.68, overcast above 0.95, reaching 1.
- [quoted] Brutsaert, W. (1975). Water Resour. Res. 11(5), 742-744.
  doi:10.1029/WR011i005p00742. eps_clear = 1.24*(e/T)^(1/7), e in hPa, T in K.
  Coefficient and units confirmed in Crawford & Duchon (1999) Eq. (16) p. 476;
  the 1975 original is publisher-blocked.

Clearness index
- [text] Duffie, J. A., & Beckman, W. A. Solar Engineering of Thermal
  Processes, 4th ed., Wiley, 2013. Eq. (1.10.1) p. 37 gives the
  extraterrestrial denominator Go = Gsc*[1 + 0.033*cos(360n/365)]*cos(theta_z)
  -- the S0 * E0 * cos(z) form used here. Sect. 2.9 pp. 71-72 fixes the
  notation: lowercase kt is the HOURLY index (Eq. 2.9.3), uppercase K_T the
  daily one (Eq. 2.9.2). Ours is instantaneous, so lowercase.
- [quoted] Erbs, D. G., Klein, S. A., & Duffie, J. A. (1982). Solar Energy
  28(4), 293-302. doi:10.1016/0038-092X(82)90302-4. The kd(kt) correlation
  with breakpoints 0.22 and 0.80, quoted verbatim as Duffie & Beckman Eq.
  (2.10.1) p. 76. Duffie & Beckman p. 77: above kt = 0.80 "there are very few
  data" -- which is why the upper bound asserted here sits just above 0.8.

Constants
- [record] Kopp, G., & Lean, J. L. (2011). Geophys. Res. Lett. 38, L01706.
  doi:10.1029/2010GL045777. Total solar irradiance 1360.8 +- 0.5 W/m2, the
  basis for S0 = 1361. Duffie & Beckman [text] still carry the older WRC value
  of 1367 W/m2; the 6 W/m2 difference is 0.4% on kt.
- [standard] CODATA 2018 Stefan-Boltzmann constant, exact under the 2019 SI
  redefinition: 5.670374419e-8 W m-2 K-4.

Solar-elevation cutoff
- [text] Shi, Y., & Long, C. N. Best Estimate Radiation Flux VAP,
  DOE/SC-ARM/TR-008. Uses SZA < 80 deg as its daytime threshold; this is the
  basis for MIN_COSZEN_FOR_CLEARNESS = 0.1736. BSRN's component-comparison
  tests tighten to SZA < 75 deg, relax between 75 and 93, and are "not
  possible" below 50 W/m2.

Land-surface parameter ranges
- [text] WRF run/VEGPARM.TBL, MODIFIED_IGBP_MODIS_NOAH. EMISSMIN/EMISSMAX span
  0.880 (urban) to 0.985 (croplands); ALBEDOMIN/ALBEDOMAX span 0.08 (water) to
  0.38 (barren), with snow/ice 0.55-0.70. Noah interpolates linearly on green
  vegetation fraction, so no value outside those intervals is reachable. NOTE:
  these are table lookups, not physical constraints -- USGS 27-category land
  use drops the emissivity floor to 0.830 and raises the albedo ceiling to
  0.60, so no test here asserts them as if they were physics.

Deliberately NOT cited, having been checked and found unsupportable:
- No 5-degree or 7-degree solar-elevation cutoff for kt appears anywhere in
  the BSRN QC document, contrary to how it is often quoted.
- No primary measurement source for a tropical clear-sky kt or sky-emissivity
  value was obtained, so no test below asserts a "tropical" literature range.
- No canonical published value for a tropical midday Rn peak was found; the
  bracket used here is a sanity rail, not a literature figure.
"""


# ---------------------------------------------------------------------------
# 1. Physical constants
# ---------------------------------------------------------------------------


def test_stefan_boltzmann_matches_codata():
    """sigma = 5.670374419e-8 W m-2 K-4 exactly (CODATA 2018, exact by SI defn)."""
    assert pytest.approx(5.670374419e-8, rel=1e-12) == STEFAN_BOLTZMANN


def test_solar_constant_matches_kopp_lean():
    """Kopp & Lean (2011) put total solar irradiance at 1360.8 +- 0.5 W/m2."""
    assert pytest.approx(1361.0, abs=1.0) == SOLAR_CONSTANT


def test_blackbody_emission_at_300k_is_the_textbook_value():
    """sigma * 300^4 = 459.3 W/m2 -- the number every radiation chapter opens with."""
    assert pytest.approx(459.3003, abs=1e-3) == STEFAN_BOLTZMANN * 300.0**4


# ---------------------------------------------------------------------------
# 2. Closed-form limits of the graybody law
# ---------------------------------------------------------------------------


def test_blackbody_surface_emits_sigma_t4_and_reflects_nothing():
    """eps = 1 collapses LWup to the Stefan-Boltzmann law; GLW must not appear."""
    tsk = np.array([[280.0, 300.0, 320.0]])
    emiss = np.ones_like(tsk)

    for glw in (0.0, 400.0, 1e6):  # the answer cannot depend on GLW at all
        lwup = compute_upwelling_longwave(emiss, tsk, np.full_like(tsk, glw))
        np.testing.assert_allclose(lwup, STEFAN_BOLTZMANN * tsk**4, rtol=1e-12)


def test_perfect_reflector_returns_the_incoming_sky_flux_unchanged():
    """eps = 0 emits nothing and mirrors GLW back -- the other analytic limit."""
    tsk = np.array([[280.0, 300.0, 320.0]])
    glw = np.array([[350.0, 400.0, 450.0]])

    lwup = compute_upwelling_longwave(np.zeros_like(tsk), tsk, glw)

    np.testing.assert_allclose(lwup, glw, rtol=1e-12)


def test_upwelling_longwave_hand_computed_value():
    """A worked case, computed by hand, pinning both terms at once.

    eps=0.95, TSK=300 K, GLW=400 W/m2:
        emission  = 0.95 * 5.670374419e-8 * 300^4 = 436.33531 W/m2
        reflected = 0.05 * 400                    =  20.00000 W/m2
        LWup                                      = 456.33531 W/m2
    """
    lwup = compute_upwelling_longwave(np.array([0.95]), np.array([300.0]), np.array([400.0]))

    assert lwup[0] == pytest.approx(456.33531, abs=1e-4)


def test_dropping_the_reflected_term_is_a_large_error_not_a_rounding_one():
    """Guards the (1 - eps) * GLW term against a 'simplification'.

    Over the real d02 archive this term is worth ~15 W/m2 in the mean; the
    point of the test is that omitting it is a physics change, not noise.
    """
    emiss, tsk, glw = np.array([0.95]), np.array([300.0]), np.array([400.0])

    with_reflection = compute_upwelling_longwave(emiss, tsk, glw)
    emission_only = emiss * STEFAN_BOLTZMANN * tsk**4

    assert (with_reflection - emission_only).item() == pytest.approx(20.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Budget identities -- the terms must add up
# ---------------------------------------------------------------------------


@pytest.fixture
def surface_state():
    """A deterministic spread of realistic tropical surface states."""
    rng = np.random.default_rng(1312)
    shape = (4, 8, 8)
    return {
        "emiss": rng.uniform(0.88, 0.99, shape),
        "tsk": rng.uniform(285.0, 315.0, shape),
        "glw": rng.uniform(320.0, 450.0, shape),
        "swdown": rng.uniform(0.0, 1000.0, shape),
        "albedo": rng.uniform(0.08, 0.40, shape),
        "t2": rng.uniform(288.0, 305.0, shape),
    }


def test_net_longwave_equals_downwelling_minus_upwelling(surface_state):
    """LWnet is evaluated in reduced form; it must still be GLW - LWup."""
    emiss, tsk, glw = surface_state["emiss"], surface_state["tsk"], surface_state["glw"]

    np.testing.assert_allclose(
        compute_net_longwave(emiss, tsk, glw),
        glw - compute_upwelling_longwave(emiss, tsk, glw),
        rtol=1e-9,
        atol=1e-9,
    )


def test_shortwave_up_and_net_partition_the_incoming_beam(surface_state):
    """SWup + SWnet == SWDOWN: albedo splits the beam, it does not create energy."""
    albedo, swdown = surface_state["albedo"], surface_state["swdown"]

    total = compute_upwelling_shortwave(albedo, swdown) + compute_net_shortwave(albedo, swdown)

    np.testing.assert_allclose(total, swdown, rtol=1e-12)


def test_net_radiation_is_exactly_its_two_components(surface_state):
    """Rn = SWnet + LWnet, bit-for-bit -- it is built from them."""
    state = surface_state
    np.testing.assert_array_equal(
        compute_net_radiation(
            state["swdown"], state["albedo"], state["emiss"], state["tsk"], state["glw"]
        ),
        compute_net_shortwave(state["albedo"], state["swdown"])
        + compute_net_longwave(state["emiss"], state["tsk"], state["glw"]),
    )


def test_net_radiation_expands_to_the_four_stream_budget(surface_state):
    """Rn == (SWdown - SWup) + (LWdown - LWup), Oke (1987) eq. 1.

    The four-stream form is the definition; the implementation uses the
    algebraically reduced one, so they are checked against each other.
    """
    state = surface_state
    four_stream = (
        state["swdown"]
        - compute_upwelling_shortwave(state["albedo"], state["swdown"])
        + state["glw"]
        - compute_upwelling_longwave(state["emiss"], state["tsk"], state["glw"])
    )

    np.testing.assert_allclose(
        compute_net_radiation(
            state["swdown"], state["albedo"], state["emiss"], state["tsk"], state["glw"]
        ),
        four_stream,
        rtol=1e-9,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# 4. Literature value ranges
# ---------------------------------------------------------------------------


def test_upwelling_longwave_falls_in_the_published_terrestrial_band(surface_state):
    """Oke (1987) Ch. 1: surface L-up over land runs roughly 300-500 W/m2.

    BSRN's QC limits for upwelling longwave are 40-900 W/m2 (physically
    possible) and 60-700 W/m2 (extremely rare). For scale, the observed global
    mean surface LWup is 396 W/m2 (Trenberth et al. 2009) and the land-only
    annual mean 372 W/m2 (Wild et al. 2015).

    The bracket is deliberately loose — the fixture reaches 315 K skin
    temperature, and hot dry land at midday genuinely reaches 550-590 W/m2, so
    a tight ceiling would be a false constraint. What this actually guards is
    the order of magnitude: a Celsius-for-Kelvin slip lands at ~1e-2 W/m2.
    """
    lwup = compute_upwelling_longwave(
        surface_state["emiss"], surface_state["tsk"], surface_state["glw"]
    )

    assert lwup.min() > 60.0
    assert lwup.max() < 700.0


def test_upwelling_longwave_is_monotonic_in_skin_temperature():
    """A hotter surface always emits more. Catches a sign or exponent slip."""
    tsk = np.linspace(270.0, 330.0, 40)
    emiss = np.full_like(tsk, 0.95)
    glw = np.full_like(tsk, 400.0)

    lwup = compute_upwelling_longwave(emiss, tsk, glw)

    assert np.all(np.diff(lwup) > 0)


def test_surface_warmer_than_the_effective_sky_loses_longwave():
    """LWnet < 0 whenever sigma*Tsk^4 exceeds GLW -- the usual daytime state."""
    tsk = np.array([300.0])  # sigma*T^4 = 459.3 W/m2
    glw = np.array([400.0])  # sky supplies less than the surface emits

    assert compute_net_longwave(np.array([0.95]), tsk, glw).item() < 0.0


def test_net_longwave_can_turn_positive_under_a_warm_overcast():
    """Sign is not hardcoded: a sky radiating more than the surface warms it."""
    tsk = np.array([280.0])  # sigma*T^4 = 348.5 W/m2
    glw = np.array([400.0])  # warm low cloud radiating down

    assert compute_net_longwave(np.array([0.95]), tsk, glw).item() > 0.0


def test_net_radiation_is_negative_at_night_and_strongly_positive_at_midday():
    """Rn flips sign across the day: gain under the sun, loss to the sky at night."""
    emiss, albedo, tsk, glw = (
        np.array([0.95]),
        np.array([0.18]),
        np.array([298.0]),
        np.array([380.0]),
    )

    night = compute_net_radiation(np.array([0.0]), albedo, emiss, tsk, glw)
    midday = compute_net_radiation(np.array([950.0]), albedo, emiss, tsk, glw)

    assert -150.0 < night.item() < 0.0
    assert 400.0 < midday.item() < 800.0


def test_sky_emissivity_spans_the_clear_to_overcast_range():
    """Clear-to-overcast sweep sits inside the observed global envelope.

    Morales-Salinas et al. (2023) report a global clear-sky effective
    emissivity range of 0.34-0.97 (mean 0.73); Herrero & Polo (2012) measure
    overcast values above 0.95, reaching 1. The BSRN QC envelope
    (0.4*sigma*Ta^4 < LWdn < sigma*Ta^4 + 25) brackets both.
    """
    t2 = np.array([298.0, 298.0])
    blackbody = STEFAN_BOLTZMANN * 298.0**4  # 447.2 W/m2
    glw = np.array([0.75 * blackbody, 0.97 * blackbody])

    eps_sky = compute_sky_emissivity(glw, t2)

    np.testing.assert_allclose(eps_sky, [0.75, 0.97], rtol=1e-9)
    assert np.all(eps_sky > 0.0)
    assert np.all(eps_sky <= 1.0)


def test_at_night_net_radiation_collapses_onto_net_longwave(surface_state):
    """With SWDOWN = 0 the shortwave terms vanish and Rn must BE LWnet.

    The cleanest internal consistency check available: it ties the whole
    budget together with no tolerance and no reference value, and it is the
    state the surface spends half its time in.
    """
    state = surface_state
    night = np.zeros_like(state["swdown"])

    np.testing.assert_array_equal(
        compute_net_radiation(night, state["albedo"], state["emiss"], state["tsk"], state["glw"]),
        compute_net_longwave(state["emiss"], state["tsk"], state["glw"]),
    )


def test_sky_emissivity_of_a_blackbody_sky_is_one():
    """The definition's fixed point: GLW = sigma*T2^4 -> eps_sky = 1."""
    t2 = np.array([290.0, 300.0, 310.0])

    np.testing.assert_allclose(
        compute_sky_emissivity(STEFAN_BOLTZMANN * t2**4, t2), 1.0, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# 5. Clearness index
# ---------------------------------------------------------------------------


def test_clearness_index_matches_its_definition_at_high_sun():
    """kt = GHI / (S0 * E0 * cos z), Erbs et al. (1982)."""
    swdown = np.array([900.0])
    coszen = np.array([0.9])
    eccentricity = np.array([1.0])

    kt = compute_clearness_index(swdown, coszen, eccentricity)

    assert kt.item() == pytest.approx(900.0 / (SOLAR_CONSTANT * 0.9), rel=1e-12)
    # Duffie & Beckman p. 77: above kt = 0.80 "there are very few data", so a
    # high-sun clear-sky value must land below that edge, not above it.
    assert 0.7 < kt.item() < 0.8


def test_clearness_index_is_undefined_below_the_elevation_floor():
    """Low sun and night publish no value rather than a horizon-singularity number."""
    swdown = np.array([100.0, 100.0, 100.0, 900.0])
    coszen = np.array([-0.9, 0.0, MIN_COSZEN_FOR_CLEARNESS / 2.0, 0.9])
    eccentricity = np.ones(4)

    kt = compute_clearness_index(swdown, coszen, eccentricity)

    assert np.isnan(kt[:3]).all(), "night and low sun must be no-value"
    assert np.isfinite(kt[3])


def test_clearness_index_never_divides_by_a_negative_denominator():
    """A negative COSZEN must never yield a negative kt (it would plot as 'clear')."""
    coszen = np.linspace(-1.0, 1.0, 201)
    kt = compute_clearness_index(np.full_like(coszen, 500.0), coszen, np.ones_like(coszen))

    finite = kt[np.isfinite(kt)]
    assert np.all(finite > 0.0)


def test_eccentricity_correction_stays_within_the_annual_swing():
    """E0 = (r0/r)^2 varies ~+-3.4% over the year (Spencer 1971)."""
    from datetime import UTC, datetime

    from micrometeorology.wrf.variables import _eccentricity_correction

    days = [datetime(2026, 1, 1, 12, tzinfo=UTC).replace(month=m) for m in range(1, 13)]
    e0 = _eccentricity_correction(days)

    assert 0.96 < e0.min() < 1.0
    assert 1.0 < e0.max() < 1.04
    # Perihelion is in early January, aphelion in early July.
    assert int(np.argmax(e0)) == 0
    assert int(np.argmin(e0)) == 6


# ---------------------------------------------------------------------------
# 6. Validation against WRF's own RRTMG fluxes (skips without the archive)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_LWUPB_FILE = _DATA_DIR / "wrfout_d02_2013-07-01_01_00_00-004_"
_SWUPB_FILE = _DATA_DIR / "wrfout_d04_2026-05-03_00_00_00"

#: The four nests of the 2026 operational run. None of them carries LWUPB, so
#: these are the files where the derivation has to stand on published bounds
#: and on the surface energy budget rather than on a ground-truth field.
_OPERATIONAL_DOMAINS = ["d01", "d02", "d03", "d04"]


def _operational_file(domain: str) -> Path:
    return _DATA_DIR / f"wrfout_{domain}_2026-05-03_00_00_00"


def _requires_operational(domain: str) -> None:
    if not _operational_file(domain).exists():
        pytest.skip(f"operational wrfout {domain} absent")


def _read_operational(domain: str) -> dict:
    """Load one operational nest's radiation inputs as float64 arrays.

    Step 0 is dropped everywhere: GLW is identically zero at the first output
    time because radiation has not been called yet, which is spin-up rather
    than a physical state.
    """
    names = (
        "EMISS",
        "TSK",
        "GLW",
        "SWDOWN",
        "ALBEDO",
        "T2",
        "COSZEN",
        "HFX",
        "LH",
        "GRDFLX",
        "SWUPB",
        "SWDNB",
        "Q2",
        "PSFC",
    )
    with netCDF4.Dataset(_operational_file(domain)) as ds:
        # Sliced at read time rather than after: step 0 never has to be
        # materialized, and every field then shares one time axis.
        fields: dict[str, np.ndarray] = {
            name: np.asarray(ds.variables[name][1:]).astype(np.float64) for name in names
        }
        # Land mask, not a field: 2-D and time-invariant, so it is neither
        # sliced nor float. Kept in the same dict because every caller wants it
        # alongside the fields it masks.
        fields["land"] = np.asarray(ds.variables["XLAND"][0]).astype(np.float64) < 1.5
    return fields


@pytest.mark.skipif(not _LWUPB_FILE.exists(), reason="archive wrfout with LWUPB absent")
def test_derived_lwup_reproduces_wrfs_own_lwupb():
    """The reconstruction is checked against the flux RRTMG actually wrote.

    LWUPB is the instantaneous upwelling longwave at the bottom of the
    atmosphere. It exists in the 2013 archive runs and NOT in the 2026
    operational runs, which is the whole reason this variable is derived.

    Tolerance: the radiation scheme uses the LSM's canopy-adjusted radiative
    temperature while we use grid-mean TSK, so a ~1-2 W/m2 land bias is
    structural, not a bug. 5 W/m2 on a ~420 W/m2 signal is ~1%.
    """
    with netCDF4.Dataset(_LWUPB_FILE) as ds:
        for step in (1, 12, 30, 48):
            emiss = ds.variables["EMISS"][step].astype(np.float64)
            tsk = ds.variables["TSK"][step].astype(np.float64)
            glw = ds.variables["GLW"][step].astype(np.float64)
            truth = ds.variables["LWUPB"][step].astype(np.float64)

            error = np.abs(compute_upwelling_longwave(emiss, tsk, glw) - truth)

            assert error.mean() < 5.0, f"step {step}: MAE {error.mean():.3f} W/m2"
            assert np.percentile(error, 99) < 25.0


@pytest.mark.skipif(not _LWUPB_FILE.exists(), reason="archive wrfout with LWUPB absent")
def test_derived_lwup_beats_the_emission_only_form_on_real_data():
    """The reflected term is not decoration: it is ~an order of magnitude of error.

    The empirical counterpart of the analytic guard above.
    """
    with netCDF4.Dataset(_LWUPB_FILE) as ds:
        step = 30
        emiss = ds.variables["EMISS"][step].astype(np.float64)
        tsk = ds.variables["TSK"][step].astype(np.float64)
        glw = ds.variables["GLW"][step].astype(np.float64)
        truth = ds.variables["LWUPB"][step].astype(np.float64)

        full = np.abs(compute_upwelling_longwave(emiss, tsk, glw) - truth).mean()
        emission_only = np.abs(emiss * STEFAN_BOLTZMANN * tsk**4 - truth).mean()

    assert full < emission_only / 5.0, f"full={full:.3f} emission_only={emission_only:.3f}"


@pytest.mark.skipif(not _SWUPB_FILE.exists(), reason="operational wrfout absent")
def test_upwelling_longwave_respects_the_bsrn_air_temperature_envelope():
    """BSRN QC: ``sigma*(Ta-15)^4 < LWup < sigma*(Ta+25)^4``.

    A surface cannot sit arbitrarily far from the air above it. This is the
    one published bound that scales with the state instead of being a fixed
    window, so it stays meaningful across seasons and climates.

    It is checked on real output rather than on the synthetic fixture because
    it constrains TSK and T2 *jointly*: the fixture draws the two from
    independent ranges and so contains combinations (skin 20 K below screen
    air) that no real surface produces and that this bound rightly rejects.
    """
    with netCDF4.Dataset(_SWUPB_FILE) as ds:
        for step in (12, 24, 36, 48):
            emiss = ds.variables["EMISS"][step].astype(np.float64)
            tsk = ds.variables["TSK"][step].astype(np.float64)
            glw = ds.variables["GLW"][step].astype(np.float64)
            t2 = ds.variables["T2"][step].astype(np.float64)

            lwup = compute_upwelling_longwave(emiss, tsk, glw)

            assert np.all(lwup > STEFAN_BOLTZMANN * (t2 - 15.0) ** 4), f"step {step}"
            assert np.all(lwup < STEFAN_BOLTZMANN * (t2 + 25.0) ** 4), f"step {step}"


@pytest.mark.skipif(not _SWUPB_FILE.exists(), reason="operational wrfout absent")
def test_derived_swup_reproduces_wrfs_own_swupb_exactly():
    """SWup = ALBEDO * SWDOWN is how WRF forms SWUPB, so this is exact."""
    with netCDF4.Dataset(_SWUPB_FILE) as ds:
        for step in (12, 15, 18):
            albedo = ds.variables["ALBEDO"][step].astype(np.float64)
            swdown = ds.variables["SWDOWN"][step].astype(np.float64)
            truth = ds.variables["SWUPB"][step].astype(np.float64)

            np.testing.assert_allclose(
                compute_upwelling_shortwave(albedo, swdown), truth, rtol=1e-6, atol=1e-4
            )


@pytest.mark.skipif(not _SWUPB_FILE.exists(), reason="operational wrfout absent")
def test_published_radiation_fields_stay_in_physical_range_on_a_real_run():
    """Every published frame of a real operational run must be physically sane.

    Bounds, where a source fixes one:

    - ``LWUP`` -- BSRN's physically-possible longwave window is 40-700 W/m2
      (Long & Dutton, QC V2.0). The upwelling side of a warm tropical surface
      cannot approach the lower limit, so the floor here is the loose 250.
    - ``SKY_EMISSIVITY`` -- the BSRN air-temperature consistency envelope
      ``0.4*sigma*Ta^4 < LWdn < sigma*Ta^4 + 25`` is exactly a bound on this
      quantity: 0.4 below, and ``1 + 25/(sigma*Ta^4)`` ~= 1.06 above at 300 K.
    - ``CLEARNESS_INDEX`` -- Duffie & Beckman p. 77 note there are very few
      data above kt = 0.80; cloud-edge enhancement genuinely exceeds it, so
      the ceiling allows headroom rather than clipping at 0.8.
    - The shortwave and net terms have no single citable envelope; those
      brackets are generous sanity rails, not literature values.
    """
    with WRFDataset(_SWUPB_FILE) as ds:
        metadata = ds.build_date_metadata()
        bounds = {
            WRFVariable.LWUP: (250.0, 700.0),
            WRFVariable.SWUP: (0.0, 400.0),
            WRFVariable.SWNET: (0.0, 1200.0),
            WRFVariable.LWNET: (-250.0, 50.0),
            WRFVariable.RNET: (-250.0, 1000.0),
            WRFVariable.SKY_EMISSIVITY: (0.4, 1.06),
            WRFVariable.CLEARNESS_INDEX: (0.0, 1.3),
        }
        for variable, (low, high) in bounds.items():
            source = build_value_frame_source(ds, variable)
            assert source is not None, variable
            # Step 0 carries GLW = 0: radiation has not been called yet, so it
            # is spin-up rather than a physical state.
            checked = 0
            for step in [m["index"] for m in metadata if publishes_step(variable, m)][1:]:
                frame = source.frame_for_step(step)
                values = frame[np.isfinite(frame)]
                if values.size == 0:
                    # Legitimately empty: the clearness index publishes no value
                    # anywhere on a step that is inside the 06-18 local gate but
                    # still below the solar-elevation floor. The frame ships as
                    # all-null, which is the honest answer, not a failure.
                    assert variable == WRFVariable.CLEARNESS_INDEX, variable
                    continue
                checked += 1
                assert values.min() >= low, f"{variable} step {step} min {values.min()}"
                assert values.max() <= high, f"{variable} step {step} max {values.max()}"
            assert checked > 0, f"{variable} published no usable frame"


# ---------------------------------------------------------------------------
# 6b. The 2026 operational run: no LWUPB anywhere, so the derivation is held
#     to published bounds and to the surface energy budget instead.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", _OPERATIONAL_DOMAINS)
def test_operational_energy_budget_closes(domain):
    """Rn + G = H + LE over whole days, on every nest.

    This is the load-bearing validation of LWUP for the operational run, which
    carries no LWUPB to compare against. Averaged over three whole days the
    land surface must shed exactly the radiation it absorbs; LWUP is the only
    term in Rn that is derived rather than read, so the closure is a direct
    test of it. Measured residual is -0.6 to -1.4 W/m2 across d01-d04.

    Steps 4..75 are exactly 3 x 24 h, which avoids aliasing the diurnal cycle
    into the mean. GRDFLX enters with a plus sign: WRF writes it positive
    upward here, which is what closes the budget (the opposite sign leaves a
    consistent +4 to +10 W/m2 residual).
    """
    _requires_operational(domain)
    f = _read_operational(domain)
    whole = slice(3, 75)  # step 0 already dropped, so this is the same 3 days
    land = np.broadcast_to(f["land"], f["HFX"][whole].shape)

    net_radiation = compute_net_radiation(f["SWDOWN"], f["ALBEDO"], f["EMISS"], f["TSK"], f["GLW"])[
        whole
    ][land].mean()
    residual = (
        net_radiation
        + f["GRDFLX"][whole][land].mean()
        - f["HFX"][whole][land].mean()
        - f["LH"][whole][land].mean()
    )

    assert abs(residual) < 5.0, f"{domain}: budget residual {residual:+.2f} W/m2"


@pytest.mark.parametrize("domain", _OPERATIONAL_DOMAINS)
def test_dropping_the_reflected_term_breaks_the_operational_budget(domain):
    """The (1 - eps)*GLW term is what makes the budget close, by ~20x.

    Same closure as above, recomputed with the emission-only LWUP. The
    residual jumps from ~1 W/m2 to ~+22 W/m2 on every nest, so the energy
    budget alone is sharp enough to reject the wrong formula on files that
    have no LWUPB to check against.
    """
    _requires_operational(domain)
    f = _read_operational(domain)
    whole = slice(3, 75)
    land = np.broadcast_to(f["land"], f["HFX"][whole].shape)
    sinks = (
        f["HFX"][whole][land].mean() - f["GRDFLX"][whole][land].mean() + f["LH"][whole][land].mean()
    )
    shortwave = compute_net_shortwave(f["ALBEDO"], f["SWDOWN"])

    correct = (shortwave + compute_net_longwave(f["EMISS"], f["TSK"], f["GLW"]))[whole]
    emission_only = (shortwave + (f["GLW"] - f["EMISS"] * STEFAN_BOLTZMANN * f["TSK"] ** 4))[whole]

    residual_correct = abs(correct[land].mean() - sinks)
    residual_wrong = abs(emission_only[land].mean() - sinks)

    assert residual_wrong > 15.0, f"{domain}: wrong formula only off by {residual_wrong:.2f}"
    assert residual_correct < residual_wrong / 10.0, (
        f"{domain}: correct={residual_correct:.2f} wrong={residual_wrong:.2f}"
    )


@pytest.mark.parametrize("domain", _OPERATIONAL_DOMAINS)
def test_operational_lwup_respects_published_bounds(domain):
    """BSRN absolute limits and the air-temperature envelope, on real output."""
    _requires_operational(domain)
    f = _read_operational(domain)
    lwup = compute_upwelling_longwave(f["EMISS"], f["TSK"], f["GLW"])

    # BSRN "extremely rare" window for upwelling longwave.
    assert lwup.min() > 60.0, f"{domain}: min {lwup.min():.1f}"
    assert lwup.max() < 700.0, f"{domain}: max {lwup.max():.1f}"
    # A surface cannot sit arbitrarily far from the air above it.
    assert np.all(lwup > STEFAN_BOLTZMANN * (f["T2"] - 15.0) ** 4), domain
    assert np.all(lwup < STEFAN_BOLTZMANN * (f["T2"] + 25.0) ** 4), domain


@pytest.mark.parametrize("domain", _OPERATIONAL_DOMAINS)
def test_operational_net_longwave_is_a_loss_over_land(domain):
    """Oke (1987) p. 22: net longwave is "usually negative", 75-125 W/m2.

    Over 99%+ of land cells and every nest. The few positive cells are warm
    overcast, which is a real state, so this is not asserted at 100%.
    """
    _requires_operational(domain)
    f = _read_operational(domain)
    lwnet = compute_net_longwave(f["EMISS"], f["TSK"], f["GLW"])
    land = np.broadcast_to(f["land"], lwnet.shape)

    negative_fraction = float((lwnet[land] < 0).mean())
    assert negative_fraction > 0.99, f"{domain}: only {negative_fraction:.1%} negative"
    assert 20.0 < float(np.median(np.abs(lwnet[land]))) < 150.0, domain


@pytest.mark.parametrize("domain", _OPERATIONAL_DOMAINS)
def test_operational_sky_emissivity_straddles_the_brutsaert_clear_sky_curve(domain):
    """eps_sky must sit near Brutsaert's clear-sky value, above it under cloud.

    Brutsaert (1975) ``eps_clear = 1.24*(e/T)^(1/7)`` evaluated from WRF's own
    Q2/PSFC/T2. A derived eps_sky that sat entirely on one side would mean the
    field is not tracking sky condition at all. Measured mean ratio is
    0.99-1.02 across the four nests, with both sides populated.
    """
    _requires_operational(domain)
    f = _read_operational(domain)
    vapour_pressure_hpa = f["Q2"] * f["PSFC"] / (0.622 + f["Q2"]) / 100.0
    brutsaert = 1.24 * (vapour_pressure_hpa / f["T2"]) ** (1.0 / 7.0)

    ratio = compute_sky_emissivity(f["GLW"], f["T2"]) / brutsaert

    assert 0.9 < float(ratio.mean()) < 1.15, f"{domain}: mean ratio {ratio.mean():.3f}"
    below = float((ratio < 1.0).mean())
    assert 0.05 < below < 0.95, f"{domain}: {below:.1%} below the clear-sky curve"


@pytest.mark.parametrize("domain", _OPERATIONAL_DOMAINS)
def test_operational_clearness_index_stays_below_the_erbs_ceiling(domain):
    """Duffie & Beckman p. 77: above kt = 0.80 "there are very few data"."""
    _requires_operational(domain)
    f = _read_operational(domain)
    with netCDF4.Dataset(_operational_file(domain)) as ds:
        times = [
            datetime.strptime(
                b"".join(ds.variables["Times"][i].tolist()).decode(), "%Y-%m-%d_%H:%M:%S"
            ).replace(tzinfo=UTC)  # WRF Times are UTC, as in WRFDataset.parse_times
            for i in range(ds.variables["Times"].shape[0])
        ]
    eccentricity = _eccentricity_correction(times)[1:, np.newaxis, np.newaxis]

    kt = compute_clearness_index(f["SWDOWN"], f["COSZEN"], eccentricity)
    valid = kt[np.isfinite(kt)]

    assert valid.min() > 0.0, f"{domain}: negative kt"
    assert float((valid > 0.80).mean()) < 0.05, f"{domain}: {(valid > 0.80).mean():.1%} above 0.80"
    assert valid.max() < 1.1, f"{domain}: max {valid.max():.3f}"


@pytest.mark.parametrize("domain", _OPERATIONAL_DOMAINS)
def test_operational_swup_matches_swupb_where_the_radiation_call_aligns(domain):
    """SWup = ALBEDO*SWDOWN reproduces SWUPB wherever the two are comparable.

    SWUPB and SWDNB are RRTMG diagnostics frozen at the last radiation call
    (RADT = 30 min on all four nests); SWDOWN is rescaled by the zenith angle
    every model step. On d04 the output times land on the radiation calls, so
    SWDOWN == SWDNB and our SWup matches SWUPB to 2e-4 W/m2. On d01-d03 the
    output clock drifts off the radiation clock, and the two disagree by up to
    ~180 W/m2 near sunrise purely from that lag.

    So the comparison is restricted to cells where SWDOWN == SWDNB, i.e. where
    the diagnostics really are contemporaneous. Publishing ALBEDO*SWDOWN
    rather than SWUPB is the deliberate choice: it keeps SWUP, SWNET and RNET
    consistent with the SWDOWN we publish beside them. The 3-day energy
    budget is indifferent between the two (they agree to <0.5 W/m2 in the
    mean), so nothing physical is lost.
    """
    _requires_operational(domain)
    f = _read_operational(domain)
    contemporaneous = f["SWDOWN"] == f["SWDNB"]
    assert contemporaneous.any(), f"{domain}: no comparable cells"

    swup = compute_upwelling_shortwave(f["ALBEDO"], f["SWDOWN"])

    np.testing.assert_allclose(
        swup[contemporaneous], f["SWUPB"][contemporaneous], rtol=1e-5, atol=1e-3
    )


# ---------------------------------------------------------------------------
# 7. Wiring: registration, gating, and the missing-input path
# ---------------------------------------------------------------------------


def _write_radiation_wrf_file(path: Path, *, nt: int = 24, start_hour_utc: int = 3) -> None:
    """A minimal wrfout carrying exactly the derived-radiation inputs.

    Deliberately independent of ``test_wrf_jobs._write_full_wrf_file``: that
    fixture's RNG draw order is pinned by the frozen byte oracles, so it must
    not grow new variables.
    """
    ny = nx = 6
    rng = np.random.default_rng(7)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", nt)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.setncattr("DX", 1000.0)
        ds.setncattr("DY", 1000.0)

        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array(
            [
                list(
                    f"2026-05-{3 + (start_hour_utc + i) // 24:02d}_{(start_hour_utc + i) % 24:02d}:00:00"
                )
                for i in range(nt)
            ],
            dtype="S1",
        )
        for name, low, high in [
            ("XLONG", -38.5, -38.0),
            ("XLAT", -13.5, -13.0),
            ("T2", 292.0, 303.0),
            ("TSK", 290.0, 312.0),
            ("EMISS", 0.88, 0.99),
            ("GLW", 330.0, 440.0),
            ("SWDOWN", 0.0, 950.0),
            ("ALBEDO", 0.08, 0.35),
            ("COSZEN", -0.95, 0.95),
        ]:
            variable = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            variable[:] = rng.uniform(low, high, size=(nt, ny, nx)).astype(np.float32)


def test_every_derived_variable_is_registered_end_to_end():
    """Enum member, output id, colormap and dispatcher entry must all agree."""
    from micrometeorology.common.types import VARIABLE_COLORMAPS

    derived = {
        WRFVariable.LWUP: "LWUP",
        WRFVariable.SWUP: "SWUP",
        WRFVariable.LWNET: "LWNET",
        WRFVariable.SWNET: "SWNET",
        WRFVariable.RNET: "RNET",
        WRFVariable.SKY_EMISSIVITY: "EPS_SKY",
        WRFVariable.CLEARNESS_INDEX: "KT",
    }
    for variable, output_id in derived.items():
        assert VARIABLE_NETCDF_MAP[variable] == output_id
        assert variable in VARIABLE_COLORMAPS
        assert variable in DERIVED_RADIATION_SOURCES


def test_only_the_solar_variables_are_daylight_gated():
    """Longwave and net radiation stay meaningful all night and must publish 24h."""
    assert {
        WRFVariable.SWDOWN,
        WRFVariable.SWUP,
        WRFVariable.SWNET,
        WRFVariable.CLEARNESS_INDEX,
    } == DAYLIGHT_ONLY_VARIABLES
    night = {"datetime_local": __import__("datetime").datetime(2026, 5, 3, 2)}
    assert not publishes_step(WRFVariable.SWUP, night)
    assert not publishes_step(WRFVariable.CLEARNESS_INDEX, night)
    assert publishes_step(WRFVariable.LWUP, night)
    assert publishes_step(WRFVariable.RNET, night)
    assert publishes_step(WRFVariable.LWNET, night)


def test_dispatcher_builds_every_derived_field_from_a_synthetic_file(tmp_path):
    wrf = tmp_path / "wrfout_d02_radiation.nc"
    _write_radiation_wrf_file(wrf)

    with WRFDataset(wrf) as ds:
        for variable in DERIVED_RADIATION_SOURCES:
            source = build_value_frame_source(ds, variable)
            assert source is not None, variable
            assert np.isfinite(source.scale_min), variable
            assert np.isfinite(source.scale_max), variable
            assert source.scale_min <= source.scale_max, variable

            frame = source.frame_for_step(5)
            assert frame.shape == (6, 6), variable


def test_derived_field_frames_agree_with_the_direct_computation(tmp_path):
    """The dispatcher must publish the extractor's values, not a re-derivation."""
    wrf = tmp_path / "wrfout_d02_radiation_agree.nc"
    _write_radiation_wrf_file(wrf)

    with WRFDataset(wrf) as ds:
        emiss = ds.get_variable("EMISS")
        tsk = ds.get_variable("TSK")
        glw = ds.get_variable("GLW")
        source = build_value_frame_source(ds, WRFVariable.LWUP)
        assert source is not None

        np.testing.assert_array_equal(
            source.frame_for_step(4),
            compute_upwelling_longwave(emiss, tsk, glw)[4],
        )


def test_a_wrfout_missing_an_input_skips_instead_of_raising(tmp_path):
    """No EMISS -> LWUP/LWNET/RNET report 'not found'; SWUP/SWNET still work."""
    wrf = tmp_path / "wrfout_d02_no_emiss.nc"
    _write_radiation_wrf_file(wrf)
    with netCDF4.Dataset(wrf, "a") as ds:
        ds.renameVariable("EMISS", "EMISS_RENAMED")

    with WRFDataset(wrf) as ds:
        for variable in (WRFVariable.LWUP, WRFVariable.LWNET, WRFVariable.RNET):
            assert build_value_frame_source(ds, variable) is None, variable
        for variable in (WRFVariable.SWUP, WRFVariable.SWNET, WRFVariable.CLEARNESS_INDEX):
            assert build_value_frame_source(ds, variable) is not None, variable


def test_clearness_index_bounds_survive_an_all_night_file(tmp_path):
    """An all-night file leaves kt entirely no-value; bounds must stay finite.

    The values writer rejects non-finite scale bounds outright, and bounds are
    computed before the daylight gate drops the steps.
    """
    wrf = tmp_path / "wrfout_d02_all_night.nc"
    _write_radiation_wrf_file(wrf)
    with netCDF4.Dataset(wrf, "a") as ds:
        ds.variables["COSZEN"][:] = -0.5  # sun below the horizon everywhere

    with WRFDataset(wrf) as ds:
        source = build_value_frame_source(ds, WRFVariable.CLEARNESS_INDEX)
        assert source is not None
        assert np.isfinite(source.scale_min)
        assert np.isfinite(source.scale_max)
        assert np.isnan(source.frame_for_step(3)).all()


def test_the_cold_start_step_is_not_published_at_all(tmp_path):
    """WRF writes step 0 before it calls radiation, so GLW is identically zero.

    Every GLW-derived field is then physically impossible there: net radiation
    reads -416 W/m2 domain-mean against a -60..+604 range for every other step,
    sky emissivity exactly 0.00 against a 0.72-0.99 legend, and upwelling
    longwave sits ~18 W/m2 low with the reflected (1 - eps) * GLW term missing
    (measured on the 2026-05-03 operational run). The publish gate and the
    colour-scale population must exclude the same steps, or the legend and the
    cells of one artifact disagree.
    """
    wrf = tmp_path / "wrfout_d02_cold_start.nc"
    _write_radiation_wrf_file(wrf)
    with netCDF4.Dataset(wrf, "a") as ds:
        ds.variables["GLW"][0, :, :] = 0.0  # the uninitialised radiation state

    with WRFDataset(wrf) as ds:
        for variable in (
            WRFVariable.LWUP,
            WRFVariable.LWNET,
            WRFVariable.RNET,
            WRFVariable.SKY_EMISSIVITY,
        ):
            source = build_value_frame_source(ds, variable)
            assert source is not None, variable
            assert np.isnan(source.frame_for_step(0)).all(), (
                f"{variable}: the cold-start frame must be entirely no-value so the "
                "publish gate drops it and the domain summary never records it"
            )
            assert np.isfinite(source.frame_for_step(1)).any(), (
                f"{variable}: only the uninitialised step goes"
            )


def test_a_continuation_run_keeps_its_first_step(tmp_path):
    """A run restarted from a previous forecast already has its radiation state."""
    wrf = tmp_path / "wrfout_d02_continuation.nc"
    _write_radiation_wrf_file(wrf)

    with WRFDataset(wrf) as ds:
        assert ds.get_variable("GLW")[0].max() > 0.0, "the fixture must not be a cold start"
        for variable in (WRFVariable.LWUP, WRFVariable.RNET, WRFVariable.SKY_EMISSIVITY):
            source = build_value_frame_source(ds, variable)
            assert source is not None, variable
            assert np.isfinite(source.frame_for_step(0)).any(), variable


def _add_surface_field(dataset, name: str, value: float) -> None:
    """Create a constant ``(Time, south_north, west_east)`` field on an open file."""
    shape = dataset.variables["T2"].shape
    variable = dataset.createVariable(name, "f4", ("Time", "south_north", "west_east"))
    variable[:] = np.full(shape, value, dtype="f4")


def test_wind_components_are_rotated_onto_true_north(tmp_path):
    """``U10``/``V10`` are GRID-relative; a bearing from them is off by alpha.

    The operational domains are Mercator, where SINALPHA is 0 everywhere and
    the rotation is the identity, so a missing rotation stays invisible until a
    Lambert or polar domain is added and every arrow is drawn wrong.
    """
    wrf = tmp_path / "wrfout_d02_rotated.nc"
    _write_radiation_wrf_file(wrf)
    with netCDF4.Dataset(wrf, "a") as ds:
        _add_surface_field(ds, "COSALPHA", 0.0)
        _add_surface_field(ds, "SINALPHA", 1.0)
        _add_surface_field(ds, "U10", 3.0)
        _add_surface_field(ds, "V10", 4.0)

    with WRFDataset(wrf) as ds:
        u, v, _low, _high = variables.extract_wind(ds)

    # A 90 deg rotation: (u, v) = (3, 4) becomes (4, -3), same speed.
    np.testing.assert_allclose(u, 4.0, atol=1e-5)
    np.testing.assert_allclose(v, -3.0, atol=1e-5)
    np.testing.assert_allclose(np.hypot(u, v), 5.0, atol=1e-5)


def test_a_file_without_the_rotation_fields_keeps_its_components(tmp_path):
    wrf = tmp_path / "wrfout_d02_unrotated.nc"
    _write_radiation_wrf_file(wrf)
    with netCDF4.Dataset(wrf, "a") as ds:
        _add_surface_field(ds, "U10", 3.0)
        _add_surface_field(ds, "V10", 4.0)

    with WRFDataset(wrf) as ds:
        u, v, _low, _high = variables.extract_wind(ds)

    np.testing.assert_allclose(u, 3.0, atol=1e-5)
    np.testing.assert_allclose(v, 4.0, atol=1e-5)
