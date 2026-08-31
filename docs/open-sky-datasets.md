# Open sky-image datasets

Catalogue of the pre-training candidates for this station's all-sky model,
ordered by proximity to the problem here — **Salvador, 13.0°S, humid tropical**,
DHI estimation from a fisheye image.

The survey behind this list is Nie et al. (2024), *Open-source sky image
datasets for solar forecasting with deep learning: a comprehensive survey*,
Renewable and Sustainable Energy Reviews 189, arXiv:2211.14709, which catalogues
**72 open datasets**, 45 of them with measured irradiance. The sizes and
coverages below come from its Table 6.

None of the entries requires payment.

| # | dataset | site | latitude | coverage | size | irradiance | camera | access |
|---|---|---|---|---|---|---|---|---|
| 1 | ARM-TWP Darwin | Australia | **12.4°S** | 12.5 years | 222 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) — free account |
| 2 | ARM-GoAmazon | Manacapuru, Brazil | 3.2°S | 1.9 year | 54 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) |
| 3 | ARM-TWP Manus / Nauru | PNG / Nauru | 2°S / 0.5°S | 10.5 / 10.8 years | 144 / 177 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) |
| 4 | ARM-LASIC | Ascension Island | 8°S | 1.5 year | 37 GB | GHI/DNI/DHI 1 min | TSI | [arm.gov](https://www.arm.gov/) |
| 5 | SKIPP'D | Stanford, USA | 37.4°N | 2.7 years | 1,700 GB | PV power | fisheye | [purl.stanford.edu/dj417rh1007](https://purl.stanford.edu/dj417rh1007) |
| **6** | **UCSD-Folsom** | California, USA | 38.6°N | 3.0 years | 50 GB | GHI/DNI/DHI 1 min | **fisheye** | [zenodo.org/records/2826939](https://zenodo.org/records/2826939) |
| 7 | SRRL-BMS | Golden, USA | 39.7°N | decades | — | GHI/DNI/DHI 1 min | TSI / ASI | [midcdmz.nrel.gov](https://midcdmz.nrel.gov/apps/sitehome.pl?site=BMS) |
| 8 | SIRTA | Palaiseau, France | 48.7°N | years | — | GHI/DNI/DHI | fisheye | [sirta.ipsl.fr](https://sirta.ipsl.fr/) — registration |
| — | SIPM | Rio de Janeiro | 22.9°S | 26 days, 0.3 GB | — | — | — | too small for pre-training |

## The trade-off that decides the choice

The ARM sites use a **TSI** — a hemispheric mirror with a shadow band that
**occludes the sun** so the sensor does not saturate. Folsom, SKIPP'D and SIRTA
use a fisheye pointed straight up, like this station's camera.

That weighs more than climate for this task. The circumsolar region is what
governs the diffuse split (Perez et al. 1990), and the `sunangle` arm measured
that giving the network the sun's position drops the elevation-band bias
amplitude from 17.47 to 2.65 W/m². Pre-training on skies with the sun covered
teaches the backbone to read a sky without the feature that matters most here.

Hence two different bets:

- **right climate, wrong camera** — ARM-TWP Darwin, 12.4°S and 12.5 years, a
  solar regime almost identical to this one;
- **right camera, wrong climate** — UCSD-Folsom, fisheye with the sun visible,
  measured DHI, and the dataset the Varaschin & Silva (2025) benchmark uses,
  which gives comparability with the literature.

**Chosen to start: Folsom.** Transfer moves the visual backbone, and what it has
to learn is how a sky looks with the sun inside the frame. A climate difference
the fine-tune on the 55 local days corrects; a difference in optical geometry it
does not.

This is reasoning from what was measured here, not a measured result. Deciding
for real means pre-training on both and comparing, which gets cheap once the
ingestion of the first one exists.

## UCSD-Folsom, what was downloaded

Zenodo DOI [10.5281/zenodo.2826939](https://doi.org/10.5281/zenodo.2826939),
licence CC BY-NC 4.0 — free, non-commercial use. Cite Pedro, Larson & Coimbra
(2019), *A comprehensive dataset for the accelerated development and
benchmarking of solar forecasting methods*, Journal of Renewable and Sustainable
Energy 11(3), 036102.

Three files out of the full record matter here:

| file | size | why |
|---|---|---|
| `Folsom_irradiance.csv` | 76.5 MB | GHI, DNI and **DHI** at 1 min, the target |
| `Folsom_weather.csv` | 138.8 MB | wind, for the `bare` tier features |
| `Folsom_sky_images_2014.tar.bz2` | 13.8 GB | the images; 2015 and 2016 add 35.5 GB |

The rest of the record — pre-extracted features, satellite, NAM, forecast targets
and the forecasting scripts — does not serve this use: feature extraction here is
our own, and their targets are forecasts with a horizon, not estimation at t=0.

Measured on `Folsom_irradiance.csv`: **1,552,320 rows**, from 2014-01-02 08:00 to
2016-12-31 07:59 UTC, 1-min cadence, 618 NaN in `dhi` and **732,122 rows above
20 W/m² of GHI**. Against this station's 46,014 rows over 81 days, one year of
Folsom is already an order of magnitude more.

## The Folsom clock: which timestamp is the capture instant

Every Folsom frame carries two times that disagree: the one in the **file name**
and the one in the **modification date**. Choosing between them is the most
consequential decision in the ingestion, and it is measured rather than preferred.

Varaschin & Silva (2025, arXiv:2503.21966, sec. 5.2.1) trained and tested the
same model under all four combinations and measured that file-name alignment
costs **62.52 W/m² of RMSE against 37.21** for date-modified — a 25 W/m² gap,
larger than the entire spread between the ten architectures they compared.

Two independent facts say which one is the capture instant:

1. The daily mean disagreement between the two **drifts with time**, from about
   zero in early 2014 to roughly **700 s** by the end of 2016 (their Fig. 7).
   That is a clock never resynchronised, not noise.
2. The file-name seconds **pile up on `:00` and `:59`** while the modification
   seconds spread evenly over all sixty (their Fig. 8). The name is an assigned
   label; the mtime is when the file was written.

Measured on the 2014 archive extracted here, over 250,609 frames: a median
disagreement of 29.0 s, with a monthly step (−79.2 s in June). **45.7 % of the
frames would pair with the wrong minute** if the name were used.

### Why there is no 30 s filter

`FOLSOM_LOST_TIMESTAMPS_S` is 6 h, not the 30 s per frame of Varaschin & Silva
(2025, sec. 3.6). Measured on the extracted 2014 archive, over its first 6,683
frames, the disagreement is already **median 14 s, p95 34 s** — a 30 s gate would
discard 12 % of the year's opening days, and with the drift reaching ~700 s by
the end of 2016 the same gate would erase almost everything, in silence. The
2.5 % discard rate they report describes the subset they worked on, not a
whole-archive ingest.

Since the modification date **is** the capture instant, there is no per-frame
arbitration to do. What a check is still worth is catching an extraction that
threw the times away (`tar -m`), which shows up as a disagreement of hours or
days, not seconds.

### Irradiance clock offset

`FOLSOM_TIMESTAMP_OFFSET_S = -20.0 s`. The same work optimised the offset per
dataset by cross-validation and measured −20 s as Folsom's best, worth
40.24 → 37.21 W/m² of test RMSE. It is small next to the file-name defect, and
included because it is measured rather than guessed.

### What was ingested

230,791 rows over 351 days, maximum solar elevation 74.8° — consistent with the
theory for 38.642°N.
