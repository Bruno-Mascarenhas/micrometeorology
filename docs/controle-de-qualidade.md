# Data quality control — the LabMiM station archive

What runs, in what order, on what evidence. This is the authoritative description
of the quality control the published record passes through; the research that
produced the thresholds is in `docs/arqueologia/qc/`.

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
mask_step_excursions             849       ← statistical
mask_persistent_runs             48,585    ← statistical
unify_sensor_columns                                 era-to-era channel unification (COPIES)
close_net_radiation              36,241 recomposed
mask_night_corrupted_days        135,828             53 days of timestamp corruption
mask_impossible_shortwave        339                 BSRN geometric ceiling
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
| pressure | 1.0 hPa | 626 | rain-coincidence *below* the archive baseline; 97.4% of it sits inside the range gate |
| temperature | 3.0 °C | 207 | WMO one-minute step limit, conservative on a 5-minute mean |
| humidity | 15 %RH | 16 | fifteen quantiser counts on the integer-recorded channels |

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
| wind speed | 36 (3 h) | 33,971 | an anemometer at two distinct values for 18 consecutive days |
| wind direction | 36 (3 h) | 9,995 | a vane frozen at 71.6° while its anemometer averaged 1.3 m/s |
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

#### The calm exemption, and the trap in it

`exempt_at_or_below` skips a level at which a repeated reading is the instrument
reporting calm rather than failing. For a **direction** channel it is evaluated on
the paired speed from `sensor_wind_speed_column_map`, because the logger writes
direction zero whenever speed is zero; without the pairing, 4,765 calm-fill zeros
would be masked as a jammed vane.

The exemption level is the most dangerous single number in the configuration.
Setting it to 0.1 m/s costs nothing. Setting it to 0.281 m/s — the propeller's own
stall floor, which looks far more principled — collapses the catch from 33,037 to
816 and makes the 18-day dead anemometer disappear entirely. **Never key the
exemption to the instrument's own calm constant.**

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
