# The station archive — which file, which sensor, which period

This document exists so nobody has to redo the archaeology of the archive. It
answers one question: **for variable X in period Y, which LBM `.dat` file is the
source, and which column of that file carries the measurement?**

Everything here was generated from three sources inside the repository and
verified against the data, not transcribed by hand:

- the explicit manifest in `sensors/archive.py` (`LENTA_MANIFEST`, `RAIN_MANIFEST`);
- the `sensor_switches` block of `configs/micromet/calibrations.yaml`, which is
  the authority on which raw column carries each variable and when;
- the measurement of `output/archive/station_5min_qc.parquet`, for real coverage.

Intervals are **naive local** stamps, as the logger wrote them.

---

## Why there is a manifest, and not a glob

Reading `data/dados-labmim/*.dat` produces a wrong record in four ways, all of
them measured in an audit of every table in the archive:

1. **`*.dat` discards the rotation files.** Three `.backup` tables are the ONLY
   source of one austral winter each — JJA 2020, JJA 2022, and June to mid-July
   2024.
2. **The directory holds more than one station.** `BTS_*` is another site
   (CR1000X serial 9429), the `celsolar` and `calibracao` tables are parallel
   instrument campaigns, and the `solar` and `radiacao` families sample every
   minute.
3. **The names lie.** `dados-labmim/LBM_lenta.dat` is the RAIN table — field 8 of
   the TOA5 header says `LBM_rain` — and it is the sole source of February 2019.
4. **Three clock defects do not fit in configuration.** They need the bytes
   repaired before the merge, which `stage_archive` does in a scratch directory:
   nothing here ever writes into `data/`.

---

## The chronology per variable

Each cell is the raw column that carries that variable in that period. An em dash
means the variable **does not exist** in the archive there — not that it is
missing through failure, but that no instrument measures it.

#### surface

| from | T | rh | Td | pressure | WS | WD | precip |
|---|---|---|---|---|---|---|---|
| 2016-09-29 a 2019-03-15 | `Temp1_Avg` | `RH1_Avg` | — | — | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-03-15 a 2019-03-18 | `Temp1_Avg` | `RH1_Avg` | — | `AirPressure` | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-03-18 a 2019-03-18 | `Temp1_Avg` | `RH1_Avg` | — | — | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-03-18 a 2019-05-31 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT_Avg` | `WS_ms_S_WVT` | `WindDir_D1_WVT` | `PL01_mm_Tot` |
| 2019-05-31 a 2019-05-31 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT_Avg` | — | — | `PL01_mm_Tot` |
| 2019-05-31 a 2019-06-10 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT_Avg` | `WS_WXT_Avg` | `WD_WXT_Avg` | `PL01_mm_Tot` |
| 2019-06-10 a 2023-02-20 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT` | `WS_WXT_Avg` | `WD_WXT_Avg` | `PL01_mm_Tot` |
| 2023-02-20 a 2023-03-10 | `Temp1_Avg` | `RH1_Avg` | — | `Pmb_WXT` | — | — | `PL01_mm_Tot` |
| 2023-03-10 a 2024-07-19 | `Temp1_Avg` | `RH1_Avg` | — | — | — | — | `PL01_mm_Tot` |
| 2024-07-19 a 2025-03-12 | `Temp1_Avg` | `RH1_Avg` | — | — | `WS_ms` | `WindDir` | `PL01_mm_Tot` |
| 2025-03-12 a 2025-03-19 | — | — | — | — | `WS_ms` | `WindDir` | `PL01_mm_Tot` |
| 2025-03-19 a 2025-03-28 | `AirT_C_Avg` | `RH` | `DP_C_Avg` | `BP_mbar_Avg` | `WS_ms` | `WindDir` | `PL01_mm_Tot` |
| 2025-03-28 onward | `AirT1_C_Avg` | `RH1` | `DP1_C_Avg` | `BP1_mbar_Avg` | `WS_ms` | `WindDir` | `PL01_mm_Tot` |

#### radiation

| from | Sw_dw | Sw_dif | Sw_up | Lw_dw | Lw_up | Sw_par | Sw_uv |
|---|---|---|---|---|---|---|---|
| 2016-09-29 a 2018-08-20 | `PSP1_Wm2_Avg` | — | — | `PIR1_Wm2_Avg` | — | `PAR_Wm2_Avg` | — |
| 2018-08-20 a 2018-08-21 | — | — | — | `PIR1_Wm2_Avg` | — | `PAR_Wm2_Avg` | — |
| 2018-08-21 a 2018-10-16 | — | — | — | — | — | `PAR_Wm2_Avg` | — |
| 2018-10-16 a 2018-11-13 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_cor_Avg` | `CG3Dn_Wm2_cor_Avg` | `PAR_Wm2_Avg` | — |
| 2018-11-13 a 2019-02-26 | `CM3Up_Wm2_Avg` | `PSP1_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_cor_Avg` | `CG3Dn_Wm2_cor_Avg` | `PAR_Wm2_Avg` | — |
| 2019-02-26 a 2019-03-15 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | — | — | `PAR_Wm2_Avg` | — |
| 2019-03-15 a 2019-03-18 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_Corr_Avg` | `CG3Dn_Wm2_Corr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-03-18 a 2019-03-19 | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2_Corr_Avg` | `CG3Dn_Wm2_Corr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-03-19 a 2019-03-19 | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | — | — | `PAR_Wm2_Avg` | — |
| 2019-03-19 a 2019-08-31 | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-08-31 a 2019-10-01 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2019-10-01 a 2020-03-06 | `CM3Up_Wm2_Avg` | `CMP21_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2020-03-06 a 2020-06-01 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2020-06-01 a 2025-03-12 | `CM3Up_Wm2_Avg` | `CMP21_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2025-03-12 a 2025-03-19 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | — |
| 2025-03-19 a 2025-05-14 | `CM3Up_Wm2_Avg` | — | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | `CUV5_Wm2_Avg` |
| 2025-05-14 onward | `CM3Up_Wm2_Avg` | `PSP_Wm2_Avg` | `CM3Dn_Wm2_Avg` | `CG3Up_Wm2Cr_Avg` | `CG3Dn_Wm2Cr_Avg` | `PAR_Wm2_Avg` | `CUV5_Wm2_Avg` |

#### net radiation

| from | Net_CNR1 | Net_NRLite | Tbody | qc_flag |
|---|---|---|---|---|
| 2016-09-29 a 2018-08-20 | — | `NRLite_Wm2_Corr_Avg` | `T_C1_Avg` | — |
| 2018-08-20 a 2018-10-16 | — | — | `T_C1_Avg` | — |
| 2018-10-16 a 2019-03-15 | — | — | `CNR1TK_Avg` | — |
| 2019-03-15 a 2019-03-19 | — | `NRLite_Wm2_Avg` | `CNR1TK_Avg` | — |
| 2019-03-19 a 2022-04-11 | `Net_Wm2_Avg` | `NRLite_Wm2_Avg` | `CNR1TK_Avg` | — |
| 2022-04-11 a 2025-03-19 | `Net_Wm2_Avg` | — | `CNR1TK_Avg` | — |
| 2025-03-19 a 2025-03-28 | `Net_Wm2_Avg` | — | `CNR1TK_Avg` | `MetSENS_Status` |
| 2025-03-28 onward | `Net_Wm2_Avg` | — | `CNR1TK_Avg` | `MetSENS1_Status` |

### The three gaps that decide a subset

Reading the matrix above top to bottom shows the station did not measure
everything all the time. Three holes are large enough to invalidate a whole
subset if they go unnoticed:

- **Pressure, 740 days with no data** — the Vaisala `Pmb_WXT` ends on 2023-03-10
  and no barometer arrives until the Gill on 2025-03-19. The variable's coverage
  across the archive is **71.7%**, the worst of all.
- **Wind, 515 days** — the WXT sonic dies on 2023-02-20 and `WS_ms` only appears
  on 2024-07-19. The same holds for speed and direction, which share the
  instrument.
- **Dew point, 66.5% coverage** — there is no dew-point instrument before the
  Gill MetSENS, in 2025. Every `Td` series earlier than that is an absent sensor,
  not an acquisition failure.

| variable | samples | coverage | largest gap | the gap starts at |
|---|---|---|---|---|
| `Lw_dw` | 975,129 | 93.9% | 57 days | 2018-08-20 23:55 |
| `Lw_up` | 785,257 | 95.3% | 17 days | 2019-02-26 09:30 |
| `Net_CNR1` | 780,993 | 94.8% | 17 days | 2019-02-26 09:35 |
| `Net_NRLite` | 481,458 | 82.8% | 206 days | 2018-08-20 23:55 |
| `Sw_dif` ¹ | 359,022 | 88.1% | 87 days | 2020-03-06 |
| `Sw_dw` ¹ | 489,924 | 94.4% | 56 days | 2018-08-21 |
| `Sw_par` ¹ | 457,794 | 88.2% | 112 days | 2017-05-19 |
| `Sw_up` ¹ | 398,179 | 96.7% | 16 days | 2019-02-27 |
| `Sw_uv` ¹ | 70,467 | 96.4% | 9 days | 2026-04-15 |
| `T` | 906,939 | 89.8% | 112 days | 2024-11-27 05:45 |
| `Tbody` | 988,815 | 95.2% | 56 days | 2018-08-21 08:25 |
| `Td` | 79,292 | 66.5% | 67 days | 2025-12-21 18:50 |
| `WD` | 861,533 | 82.9% | 515 days | 2023-02-20 13:30 |
| `WS` | 837,110 | 80.6% | 515 days | 2023-02-20 13:30 |
| `precip` | 996,977 | 96.0% | 78 days | 2024-02-12 23:55 |
| `pressure` | 558,248 | 71.7% | 740 days | 2023-03-10 06:10 |
| `qc_flag` | 142,759 | 96.3% | 10 days | 2026-04-14 05:45 |
| `ur` | 907,061 | 93.5% | 112 days | 2024-11-27 05:50 |

Coverage is measured against the 5-minute grid between each variable's first and
last sample, on the quality-controlled frame — that is, already net of what QC
removed. `docs/quality-control.md` describes what each stage removes and why.

¹ **Shortwave: counted in daylight only.** From `mask_nocturnal_shortwave` onward
the nocturnal hour does not exist in these channels — a pyranometer with the sun
below the horizon reports its own thermal offset, not a flux — so both the samples
and the denominator cover only the instants with the sun above the horizon. That
is why the sample count of these five rows is about half the others' while
coverage stays comparable: what left was the night, on both sides of the fraction.
The gap of these rows is measured in calendar days with no valid daylight sample,
not in consecutive missing minutes, which for a channel without night would count
every small hours as an interruption. `Net_CNR1` falls from 783,548 to 780,993
because the BSRN floor and the sign rule, in rejecting a component, reject the net
flux derived from it. `Sw_par` falls 1.5 pp more than the other channels because
it carried 7,812 exact zeros in broad daylight — a quantum sensor reading nothing
with the sun above the horizon.

Shortwave means computed over this archive are **daylight** means and are not
comparable to 24 h means published before this change; a daily insolation total
requires restoring the nocturnal hours as zero before integrating.

---

## Traps that have already cost time

These are the reasons the matrix above cannot be reconstructed by inspecting
column names.

**A new name does not mean a new sensor.** `PSP1_Wm2_Avg` becomes `PSP_Wm2_Avg`
at the v11 program change, on 2019-03-15, and it is the same pyranometer: the
programmed mV→W/m² multiplier is 119.474 = 1000/8.37 under both spellings, in
every year. Likewise `CG3Up_Wm2_cor_Avg` → `CG3Up_Wm2_Corr_Avg` →
`CG3Up_Wm2Cr_Avg` are three spellings of the same corrected channel.

**An almost identical name can be another sensor.** `RH1_Avg` is the HMP; `RH1`
is unit 1 of the Gill MetSENS. They are different instruments, six years apart,
and the difference in name is one suffix.

**The aggregation token changes mid-record.** On 2025-03-19 humidity goes from
`Avg` to `Smp` — from interval mean to instantaneous sample. The unified
variable's name does not change, but the quantity does.

**Kelvin columns that look like something else.** `T_C1_Avg` and `T_D1_Avg` in the
v4/v9 era are the CASE and DOME thermistors of the Eppley PIR, in kelvin (294.7 to
309.0 K, correlation 0.9955 between them) — they are neither air temperature nor
dew point.

**Two units in parallel for a month and a half.** Between 2025-03-28 and
2025-05-14 both Gill MetSENS units record simultaneously (`AirT1_C_Avg` and
`AirT2_C_Avg`). Unit 1 was chosen for continuity; unit 2 exists in the archive and
enters no unified variable.

**Sentinels that are not NaN.** Humidity writes a genuine `0.0` between 2018-08-27
and 2018-10-16, and `-100` between 2024-11-27 and 2025-03-12; temperature writes
`-100` in that same second window and `1000` from 2025-12-13 onward. None of them
is a physical value and none is missing — which is why `mask_sentinels` exists and
runs before anything else.

---

## The files, measured

Two logger tables, on the same 5-minute grid, joined by JOIN and not by
concatenation. The intervals below were read from each file, not from notes.

#### lenta table

| file | from | to | rows | repair | why it is in the manifest |
|---|---|---|---|---|---|
| `dados-labmim/LBM_lenta_2016.dat` | 2016-09-29 13:40 | 2016-12-31 23:55 | 25,102 | — | start of record, 2016-09-29 |
| `dados-labmim/LBM_lenta_2017.dat` | 2017-01-01 00:00 | 2017-12-31 23:55 | 103,106 | — | all of 2017, complete JJA |
| `dados-labmim/LBM_lenta_2018_1.dat` | 2018-01-01 00:05 | 2018-10-16 13:40 | 78,865 | — | 2018-01..2018-10-16, JJA 2018 |
| `dados-labmim/LBM_lenta_2018-2019.dat` | 2018-10-16 13:50 | 2019-02-26 09:30 | 38,152 | — | CNR1 commissioning era |
| `dados-labmim/LBM_lenta_2019.dat.backup` | 2019-03-15 11:20 | 2019-03-15 15:55 | 56 | — | sole source of 2019-03-15 afternoon |
| `dados-labmim/LBM_lenta_2019.dat.1.backup` | 2019-03-15 17:05 | 2019-03-18 09:05 | 769 | — | sole source of 2019-03-15..18 |
| `dados-labmim/LBM_lenta_2019.dat.2.backup` | 2019-03-18 12:55 | 2019-03-19 08:25 | 232 | — | sole source of 2019-03-18..19, WXT arrives |
| `dados-labmim/LBM_lenta_2019.dat.3.backup` | 2019-03-19 10:05 | 2019-03-20 15:00 | 348 | — | sole source of 2019-03-19..05-31 |
| `dados-labmim/LBM_lenta_2019_0531.dat` | 2019-03-20 15:55 | 2019-05-31 08:30 | 20,622 | — | 2019-05-31 onward |
| `dados-labmim/LBM_lenta_2019_0631.dat` | 2019-05-31 09:10 | 2019-06-10 15:30 | 2,951 | — | 2019-06 onward |
| `dados-labmim/LBM_lenta_2019_1011.dat` | 2019-06-10 15:35 | 2019-10-11 09:45 | 35,292 | — | 2019-10 onward, CMP21 diffuse begins |
| `dados-labmim/LBM_lenta_2019.dat` | 2019-10-01 00:00 | 2020-01-07 00:00 | 28,211 | `drop-late-tail` | 110-row tail is mis-stamped; the clock-fixed 2020_03 table carries it correctly |
| `dados-labmim/LBM_lenta_2020_03.dat` | 2020-01-01 00:00 | 2020-03-06 10:55 | 18,848 | `clock+1h` | headerless CSV, and 16855 rows are one hour early |
| `dados-labmim/LBM_lenta_2020.dat.backup` | 2020-03-06 11:00 | 2020-09-23 10:40 | 57,814 | — | SOLE SOURCE OF JJA 2020 |
| `dados-labmim/LBM_lenta_2020.dat` | 2020-09-23 11:05 | 2021-07-26 14:25 | 88,109 | — | rest of 2020 |
| `dados-labmim/LBM_lenta_2021.dat` | 2021-07-26 13:40 | 2022-04-11 10:30 | 74,472 | — | all of 2021 |
| `dados-labmim/LBM_lenta_2022.dat.backup` | 2022-04-11 10:40 | 2022-09-16 14:00 | 44,881 | — | SOLE SOURCE OF JJA 2022 |
| `dados-labmim/LBM_lenta_2022.dat` | 2022-09-16 14:05 | 2023-08-18 14:25 | 96,476 | — | rest of 2022 (superset of data/LBM_lenta_2022.dat) |
| `dados-labmim/CR5000_LBM_lenta_18-21082023.dat` | 2023-08-18 14:35 | 2023-08-21 09:25 | 803 | — | 2023-08 spare-logger block |
| `dados-labmim/LBM_lenta_2023.dat` | 2023-08-21 09:30 | 2024-03-14 17:00 | 59,389 | — | 2023 |
| `dados-labmim/LBM_lenta_2023_14032024.dat` | 2024-03-14 17:05 | 2024-03-18 09:00 | 1,056 | — | 2024-03 handover |
| `dados-labmim/LBM_lenta_2024.dat.backup` | 2024-03-18 09:05 | 2024-07-19 15:00 | 35,495 | — | SOLE SOURCE OF JUNE AND 1-19 JULY 2024 |
| `dados-labmim/LBM_lenta_2024.dat` | 2024-07-19 15:15 | 2025-03-12 13:20 | 67,931 | — | rest of 2024 |
| `dados-labmim/LBM_lenta_2025.dat.backup` | 2025-03-12 13:25 | 2025-03-19 11:15 | 1,991 | — | 2025-03 Gill MetSENS commissioning |
| `dados-labmim/LBM_lenta_2025.dat.1.backup` | 2025-03-19 11:20 | 2025-03-19 12:55 | 20 | — | 2025-03 commissioning |
| `dados-labmim/LBM_lenta_2025.dat.2.backup` | 2025-03-19 13:10 | 2025-03-19 13:25 | 4 | — | 2025-03 commissioning |
| `dados-labmim/LBM_lenta_2025.dat.3.backup` | 2025-03-19 14:00 | 2025-03-28 10:30 | 2,550 | — | 2025-03 commissioning |
| `dados-labmim/LBM_lenta_2025.dat.4.backup` | 2025-03-28 10:35 | 2025-05-14 15:20 | 13,579 | — | 2025-03-28..05-14, dual GMX units |
| `LBM_lenta_2025.dat` | 2025-05-14 15:25 | 2026-08-15 23:30 | 126,606 | — | v22 era to 2026-08-12; PSP takes over diffuse |

#### rain table

| file | from | to | rows | repair | why it is in the manifest |
|---|---|---|---|---|---|
| `dados-labmim/LBM_rain_2016.dat` | 2016-09-29 13:40 | 2016-12-31 23:55 | 25,305 | — | start of rain record |
| `dados-labmim/LBM_rain_2017.dat` | 2017-01-01 00:05 | 2017-12-31 23:55 | 103,178 | — | 2017 |
| `dados-labmim/LBM_rain_2018_2019.dat` | 2018-01-01 00:05 | 2019-01-31 11:00 | 109,565 | — | 2018 into 2019 |
| `dados-labmim/LBM_lenta.dat` | 2019-01-31 11:30 | 2019-02-26 09:30 | 7,463 | — | MISNAMED: TOA5 field 8 is LBM_rain. Unique source of 2019-01-31..02-26 |
| `dados-labmim/LBM_rain_2019.dat` | 2019-03-15 11:20 | 2020-01-07 00:00 | 85,497 | `drop-late-tail` | same 110-row mis-stamped tail |
| `dados-labmim/LBM_rain_2020.dat` | 2020-01-01 00:00 | 2021-07-26 14:25 | 164,759 | — | 2020 (clock slip is in the lenta table, not here) |
| `dados-labmim/LBM_rain_2021.dat` | 2021-07-26 13:40 | 2022-04-11 10:30 | 74,470 | — | 2021 |
| `dados-labmim/LBM_rain_2022.dat` | 2022-04-11 10:40 | 2023-08-18 14:30 | 141,355 | — | 2022 (superset of data/LBM_rain_2022.dat) |
| `dados-labmim/CR5000_LBM_rain_18-21082023.dat` | 2023-08-18 14:35 | 2023-08-21 09:30 | 804 | `keep-2023-block` | only the 804-row 2023-08 block; 892 scattered pre-2016 rows are a spare logger |
| `dados-labmim/LBM_rain_2023.dat` | 2023-08-21 09:35 | 2024-03-14 17:00 | 59,386 | — | 2023 |
| `dados-labmim/LBM_rain2023_14032024.dat` | 2024-03-14 17:05 | 2024-03-18 09:00 | 1,056 | — | 2024-03 handover |
| `dados-labmim/LBM_rain_2024.dat` | 2024-03-18 09:05 | 2025-03-12 13:20 | 103,424 | — | 2024 |
| `LBM_rain_2025.dat` | 2025-03-12 13:25 | 2026-08-15 23:30 | 144,904 | — | 2025 to 2026-08-12 |

---

## How to use this

To reproduce any published number, do not read the `.dat` files: use the database.

```bash
uv run labmim-archive -d data -o output/archive --strict
```

That writes three artifacts plus an `archive_report.json` tabulating what each
stage removed. `station_5min_raw.parquet` is immutable and carries the values as
the logger wrote them, sentinels included; `station_5min_qc.parquet` is the frame
the hourly means come from; `station_hourly.parquet` is the aggregation.

To know which raw column answers for a variable at an instant, the authority is
the code, not this document:

```python
from micrometeorology.sensors.calibration import load_sensor_switches

switches = load_sensor_switches("configs/micromet/calibrations.yaml")
```

This file is a reading of that block plus the measurement of the archive. If the
two disagree, the block is right and this document is stale.
