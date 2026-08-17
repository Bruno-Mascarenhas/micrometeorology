# Data quality control — the LabMiM station archive

What runs, in what order, on what evidence. This is the authoritative description
of the quality control the published record passes through; the research that
produced the thresholds is in `docs/arqueologia/qc/`, and which file and which
sensor column feeds each variable over which period is in
`docs/acervo-da-estacao.md`.

The governing idea is one sentence: **a gate that only rejects the impossible does
not detect the improbable.** For years this archive had range gates alone. A
barometer that dropped 17 hPa for a single five-minute sample and returned passed
every bound — 997 hPa is a perfectly possible pressure — and 551 such excursions
reached the published hourly means, shifting individual hours by up to 3.70 hPa
against a site whose entire synoptic range is about 17 hPa.

---

## Where it runs

Quality control belongs to `labmim-archive` (`cli/build_archive.py`) and to
nothing else. **No consumer applies its own filtering**, and none reads the
unfiltered frame: `station_5min_raw.parquet` has zero consumers in the codebase,
and every downstream CLI — climatology, sky, monitoring, comparison — reads either
`station_5min_qc.parquet` or `station_hourly.parquet`.

That matters for reading defect reports. The monitoring page labels its
five-minute layer "raw"; that is **resolution, not provenance**. It is the QC'd
frame at five-minute spacing. A defect visible there survived QC rather than
bypassing it.

```
build_five_minute_frame          merge the .dat archive against an explicit manifest
  └─ write station_5min_raw      immutable: values as the logger wrote them
mask_sentinels                   1,093,223 samples   logger fill values (-99999, NAN, ...)
apply_physical_limits            236,815             range gates, raw logger units
apply_calibrations               1,227               a `factor: null` record voids its window
mask_step_excursions             839       ← statistical
mask_persistent_runs             60,824    ← statistical
unify_sensor_columns                                 era-to-era channel unification (COPIES)
close_net_radiation              36,241 recomposed
mask_night_corrupted_days        187,826             53 days of timestamp corruption
mask_impossible_shortwave        559                 BSRN per-component ceiling
apply_physical_limits (2nd)      579                 re-check in calibrated units
  └─ write station_5min_qc
aggregate_to_hourly                                  means, sums, vector means
  └─ write station_hourly        86,579 hours
```

Every removing stage reports its tally into `archive_report.json`, and the
invariant is that those tallies sum to the raw-to-QC delta. A stage that removes
nothing still reports zero rather than omitting its key.

### Why the statistical stage sits exactly there

**After calibration**, because its thresholds are in physical units (hPa, °C,
%RH). This is the same reason a second range-gate pass exists after calibration.

**Before unification**, because `unify_sensor_columns` *copies*: each unified
channel keeps a raw twin under the logger's own name. Masking the raw alias
propagates to the unified channel for free. Placed after unification, a mask would
have to reach both names or the artifact publishes the rejected value under the
other one.

Masking per raw alias also means **per instrument**, so no era boundary can
manufacture a false step across a sensor swap.

---

## The range gates

`sensor_limits` in `configs/micromet/default.yaml`. Each bound is the wider of the
physically impossible limit from the standards (WMO-No. 8; Zahumenský,
WMO/TD-No. 1236) and the empirically absurd limit measured over the whole archive.
They exist to remove instrument failures, not meteorological extremes.

A rejected sample becomes `NaN`. Nothing is ever clipped to the bound: a value
that fails the gate becomes missing rather than becoming the threshold.

---

## The statistical gates

Two tests, and what each can see is different.

### Excursion (spike/dip)

A value that leaves the series and comes back within one or two samples, returning
to within `return_tol` of where it left. Only the **interior** is masked; the
sample that returns is good data.

**The return is the discriminator, not the size of the jump.** This is the whole
design, and it is measured: a plain step test at 1.5 °C flags 305 temperature
samples, half of them coincident with rain — those are convective cold pools, real
weather the station exists to record. Requiring recovery within two samples drops
that to 32 samples, because a cold pool does not recover in five minutes and a
sensor dropout does.

| family | threshold | removed | evidence |
|---|---|---|---|
| pressure | 1.0 hPa | 609 | rain-coincidence *below* the archive baseline; 97.4% of it sits inside the range gate |
| temperature | 3.0 °C | 212 | WMO one-minute step limit, conservative on a 5-minute mean |
| humidity | 15 %RH | 18 | fifteen quantiser counts on the integer-recorded channels |

The pressure threshold is **read off a gap, not chosen**: genuine five-minute
variability at this site has p99 = 0.20 hPa, and the smallest artifact step
measured is 11.21 hPa. Anything from 0.3 to 10.0 hPa removes the same ~605
samples. 1.0 hPa is picked because it clears the coarsest barometer's 1 hPa
quantisation.

Residual risk, stated: the temperature excursions on `Temp_WXT_Avg` and
`AirT1_C_Avg` still show about twice the baseline rain-coincidence. The physical
argument for keeping them is that a genuine 3 °C drop which fully recovers within
ten minutes is not a cold pool, and rain is exactly when a sensor gets wet and
misbehaves.

### Persistence (exact-repeat run)

More than `min_run` bitwise-identical consecutive samples. A data gap **breaks** a
run: identical values either side of missing hours are not evidence that anything
held still in between.

| family | window | removed | what it caught |
|---|---|---|---|
| wind speed | 36 (3 h) | 41,517 | an anemometer at two distinct values for 18 consecutive days |
| wind direction | 36 (3 h) | 17,042 | a vane frozen at 71.6° while its anemometer averaged 1.3 m/s |
| humidity | 144 (12 h) | 2,265 | 146 h frozen at 72.47 %RH; 42.7 h at 48.3 %RH |
| temperature | 36 (3 h) | 0 | regression guard; longest genuine run in ten years is 33 |
| pressure | 36 (3 h) | 0 | regression guard; longest genuine run is 14 |

The windows are **locally derived** from the longest genuine run per family, not
transplanted. WMO/TD-No. 1236 specifies a minimum-variability-over-60-minutes
form; measured here it flags 3.6% of one temperature channel and 10.1% of one
humidity channel, because a damped maritime tropical climate genuinely holds still
longer than the continental stations that criterion was written for.

Humidity needs four times temperature's window for a mechanical reason: `RH`,
`RH1` and `RH2` are integer-recorded, so digitisation alone produces genuine
61-sample runs. At 144 those channels fall to exactly zero flagged while both real
rails are still caught in full.

#### Why there is no exemption for calm

An earlier revision carried one, keyed to the wind speed: a repeated reading at or
below 0.1 m/s was treated as the instrument reporting calm rather than failing, and
for a direction channel the level was read off the paired speed, because the logger
writes direction zero whenever speed is zero.

It was removed, because the measurement says it protects the wrong thing. Genuine
calm runs at this site reach 2 samples at p99 and 6 at their longest; the rails run
1,318 to 3,060 samples — 110 to 255 hours of a coastal site reporting no wind at
all. A level-keyed exemption cannot tell those apart, and a sensor that dies
reporting zero reads calm for as long as it stays dead, so the exemption protected
exactly the failures the test exists to find. `min_run` separates the two
populations on its own, with three orders of magnitude to spare.

The same measurement kills the more principled-looking variant: keying the level to
the propeller's own 0.281 m/s stall floor collapses the catch from 33,037 to 816
and makes an 18-day dead anemometer disappear entirely.

---|---|---|---|
| `BP1_mbar_Avg` | 1,566 | 0 | sentinel — 2.62 hPa is impossible |
| `RH1` | 1,567 | 0 | sentinel |
| `WS1_ms_GMX` | 1,592 | 26 | persistence — 2.62 m/s is a perfectly ordinary wind |

A value-based gate cannot see a fault code that lands inside its channel's
physical range. Only duration separates it from a real reading there, and only the
persistence test measures duration.

The 26 survivors on the anemometer are not residue: they are twenty-six isolated
single samples spread from 2025-05 to 2026-07, a year before the failure, and they
are genuine wind measurements that happen to read 2.62 m/s. The 1,566-sample rail
was removed in full from all three channels. Across the rest of the archive 2.62
appears in ten other columns, always isolated or in pairs, and every one is kept.

---

## What is deliberately not tested

Negative results, recorded so they are not rediscovered.

**No step test on wind direction.** With correct circular differencing the genuine
change distribution has no gap: p99 is 126–166° across channels, and a 135° gate
still flags up to 3.4% of every channel. Light wind genuinely wanders. The
Oklahoma Mesonet, which operates on native five-minute data, sets its
direction-step threshold to 360° — it declines the test too. Direction is
controlled by persistence and by its paired speed instead.

**No step test on radiation or net flux.** Run as a control, an excursion test at
300 W/m² masks 3,653 samples of one pyranometer, and the flagged events cluster
between local hours 9.9 and 14.7 — on solar noon, which an instrument fault has no
reason to prefer. Those are cloud-edge and cumulus-enhancement transitions, the
tropical physics the station exists to record. Radiation faults are already handled
by geometry-aware machinery: `mask_impossible_shortwave`, the BSRN μ₀ ceiling and
the whole-day timestamp-corruption mask.

**No test at all on precipitation.** Only 2.15% of samples are wet, so zero is the
normal state: 96.1% of samples sit in constant runs longer than an hour. Any
persistence test would flag the record. A step test is equally wrong — rainfall is
discontinuous by nature, and that is the measurement. The 0–25.4 mm per-interval
range gate is sufficient.

---

## Limits

**There is no flag layer, and it is blocked.** Everything above either masks to
`NaN` or does nothing; there is no way to publish "suspect but retained".
`aggregate_to_hourly` selects numeric columns, so any numeric flag column added to
the five-minute frame would be silently mean-averaged into `station_hourly` as a
meaningless float. Until aggregation excludes or bitwise-aggregates flag columns,
every test that would flag a population containing real weather has to stay out —
which is why the plain step tests and the WMO range-persistence forms are not here.

**One inter-sensor check is waiting on that layer.** A direction standard deviation
of exactly zero while the paired speed exceeds 1 m/s is the WMO-No. 8 signature of a
jammed vane; 1,862 such samples sit inside the configured bounds today, invisible to
every shipped test. It is not implemented because WMO's own framing makes it
non-definitive — it says the *pair* disagrees, not which instrument is at fault —
and under this repository's rule a mask is an irreversible `NaN`.

**One drift fault is invisible to all of it.** `RH_WXT_Avg` never exceeds
90.70 %RH across 416,889 samples and has a monthly maximum below 95% in every month
of its deployment, which at a saturating tropical coastal site is a low bias of
roughly 10%. No step or persistence test can see a sensor that moves correctly
around the wrong centre.

**Three pressure excursions survive**, out of 551. They are longer or less
symmetric than `max_len: 2` admits.

---

## Reproducing the numbers

```bash
uv run labmim-archive -d data -o output/archive --strict
cat output/archive/archive_report.json
```

Counts in this document are net-new: what each stage removes *beyond* what the
stages before it already took. They were measured on the frame at the exact
pipeline slot the stage occupies, not on `station_5min_qc.parquet` — that frame is
censored by the very gates under revision, and measuring a proposed gate against it
understates it.

---

## References

Every control below was checked against a source that was actually opened. Where
a published threshold did not transfer to this site, that is stated with the
measurement that showed it — a citation attached to a number we did not use would
be worse than none.

Three categories, and the third is the one to read first.

### Controls with a verified published source

| control | source |
|---|---|
| range gates, all variables | WMO (2008), *Guide to Meteorological Instruments and Methods of Observation*, WMO-No. 8, 7th ed. — Zahumenský, I. (2004), *Guidelines on Quality Control Procedures for Data from Automatic Weather Stations*, WMO CBS/OPAG-IOS/ET AWS-3/Doc. 4(1) |
| excursion (spike/dip) form | Fiebrich, C. A. et al. (2010), *Quality Assurance Procedures for Mesoscale Meteorological Data*, J. Atmos. Oceanic Technol. 27, 1565–1582, sec. 3.b.1, reporting Graybeal et al. (2002): spike/dip tests outperform plain step tests |
| temperature excursion, 3.0 °C | Zahumenský (2004), ch. II sec. II.b — the one-minute air-temperature step limit, conservative applied to a five-minute mean |
| humidity excursion, 15 %RH | between Shafer, M. A. et al. (2000), *Quality Assurance Procedures in the Oklahoma Mesonetwork*, J. Atmos. Oceanic Technol. 17, 474–494, tab. 3 (20 %RH per 5 min) and Zahumenský (2004) (10 %RH per 1 min) |
| streak/persistence parametrised by reporting resolution | Dunn, R. J. H. et al. (2012), *HadISD: a quality-controlled global synoptic report database…*, Climate of the Past 8, 1649–1679 |
| BSRN physically-possible ceilings, per component | Long, C. N. & Shi, Y. (2008), *An Automated Quality Assessment and Control Algorithm for Surface Radiation Measurements*, Open Atmos. Sci. J. 2, 23–37 — Long, C. N. & Dutton, E. G. (2002, rev. 2009), *BSRN Global Network recommended QC tests V2.0* |
| diffuse must not exceed global | Long & Shi (2008), the two-component comparison test |
| tipping-bucket quantisation | Lewis, E., Pritchard, D. et al., *GSDR-QC* reference implementation, Newcastle University Water Group — Vaisala Oyj, *WXT530 Series User Guide* M211840EN-G, precipitation measurement principle |
| blocked-gauge detection by dry run | Meira, M. A. (2021), *Quality control procedures for sub-hourly rainfall data*, MSc thesis, UFRGS — Castro, M. L., Vichete, W. D. & Filho, L. L. (2026), *METBRA25Y: Brazil Surface Meteorology Archive with Harmonized Variables and Quality Control*, arXiv:2605.08701 |
| saturation check on hygrometers | Estévez, J., Gavilán, P. & García-Marín, A. P. (2011), *Data validation procedures in agricultural meteorology*, Adv. Sci. Res. 6, 141–146 — AASC (2019), *Recommendations and Best Practices for Mesonets*, v1 |
| calm-fill convention on wind direction | Lucio-Eceiza, E. E. et al. (2018), *Quality Control of Surface Wind Observations in Northeastern North America*, Parts I and II, J. Atmos. Oceanic Technol. 35, 159–182 and 183–205 |
| the decision NOT to step-test wind direction | Shafer et al. (2000), tab. 3 — the Oklahoma Mesonet operates on native five-minute data and sets its direction-step threshold to 360°, declining the test |

### Controls derived locally, because the published form does not transfer

Each of these was measured against the published criterion first, and the
measurement is why it was not used.

| control | published form, and what it did here |
|---|---|
| persistence windows (36 samples for T and P, 144 for RH) | WMO/TD-No. 1236 specifies minimum VARIABILITY over 60 minutes. Measured here it flags 3.6 % of one temperature channel and 10.1 % of one humidity channel — a damped maritime tropical climate genuinely holds still longer than the continental stations that criterion was written for. The windows here come from the longest genuine run per family over ten years. |
| pressure excursion, 1.0 hPa | No published five-minute limit fits a sea-level tropical site where genuine variability has p99 = 0.20 hPa. Read off a two-order-of-magnitude gap: the smallest artifact step measured is 11.21 hPa. |
| calm exemption, 0.1 m/s with a duration ceiling | Keyed to the instrument's own stall floor (0.281 m/s) — the principled-looking choice — the catch collapses from 33,037 to 816 and an 18-day dead anemometer disappears. |
| humidity saturation floor, 93.0 %RH | The literature prescribes the test, not the number. Read off a gap: healthy channels never put a month below 95.0, faulty ones never above 90.7. |
| diffuse-exceeds-global floor, 200 W/m² | The rule is absolute in the literature. Applied absolutely here it fires on 126,404 samples, of which all but 30 are night thermal offset around zero. |

### Controls with neither a source nor a local measurement

The most important list in this document, because these are in production on
habit alone.

- **The longwave range gates** `CG3Up [250, 550]` and `CG3Dn [300, 650]` carry no
  citation, and they contradict the BSRN limits this same file cites for
  shortwave. They also remove only the tip of a known CNR1 Pt-100 fault rather
  than the fault.
- **`panel_temp`** passes through the whole pipeline and into `station_hourly`
  with no entry in any of the three control lists.
- **The three dewpoint channels** `DP_C_Avg`, `DP1_C_Avg`, `DP2_C_Avg` have a
  range gate and no statistical test, while every sibling T and RH channel of the
  same instrument has both.
- **`WindDir_SD1_WVT`**, 267,252 samples of σ-θ, has only a range gate.
- **Nine wind channels** have no persistence rule, including `WindDir` with
  212,681 samples.

### Sources consulted that produced a deliberate negative

Recording these so the same ground is not covered twice.

- **Segovia-Cardozo, D. A. et al. (2021)**, *Understanding the Mechanical Biases
  of Tipping-Bucket Rain Gauges*, Water 13, 2285 — the CUSUM inter-tip test needs
  tip timestamps. Our data is interval totals, and 47.3 % of wet samples contain
  two or more tips, carrying 81.66 % of the archive's depth.
- **Muchan, K. & Dixon, H. (2019)**, Hydrology Research 50(6), 1564–1576, and
  **Rombeek, N. et al. (2025)**, HESS — undercatch and wind correction are
  corrections to the measurement, not quality control of it, and would rewrite
  values rather than reject them.
- **Jiménez, P. A. et al. (2010)**, J. Appl. Meteor. Climatol. 49, 308–325 — the
  wind-direction step test, declined for the reason Shafer et al. give.
- **Long & Shi (2008)** minimum of −4 W/m² — measured here as 100 % false
  positive: 21,256 samples of uncorrected thermal offset from IR loss, exactly
  the effect the same paper describes. Masking would bias the night mean upward
  and destroy the diagnostic.
