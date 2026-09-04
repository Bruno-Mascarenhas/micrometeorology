# The all-sky label join, and the factorial that measured it

This document exists so nobody has to rediscover why the image-sensor pairing was
offset, nor redo the measurement that quantified its cost.

Everything here was measured on LabMiM's own archive, not transcribed from the
literature. Intervals are **naive local** stamps, as the logger wrote them.

---

## 1. The CR5000 stamps the average at the end of the window

A row written at `t` is the average over `(t − 5 min, t]`, not over
`[t, t + 5 min)` and not over a window centred on `t`.

The check uses the one pair of tables in the archive that samples the same signal
at two cadences at the same time:

| table | cadence | period |
|---|---|---|
| `dados-labmim/LBM_solar_2024.dat` | 1 min | 2024-03-18 09:01 .. 2024-07-19 15:00 |
| `dados-labmim/LBM_lenta_2024.dat.backup` | 5 min | 2024-03-18 09:05 .. 2024-07-19 15:00 |

The `.backup` is the only source of that 5-minute stretch — one more case of the
rule `docs/station-archive.md` already documents: a bare `*.dat` glob would have
discarded the evidence.

Rebuilding the 5-minute average from the 1-minute series under both conventions
and comparing against what the logger actually wrote in `CM3Up_Wm2_Avg`:

| convention | RMS vs. the 5-min table | r | n |
|---|---|---|---|
| **end-stamped** `(t−5, t]` | **0.083 W/m²** | **1.000000** | 35,492 |
| begin-stamped `[t, t+5)` | 64.79 W/m² | 0.974 | 35,492 |

That is not high correlation: it is identity. The convention is settled.

## 2. The cost of pairing on the raw stamp

`allsky.data.alignment.CenterFrame.pair` runs `np.searchsorted` over the raw
stamps, so the frame at instant τ gets the label whose temporal centroid sits
2.5 min before where the pipeline treats it.

Measured against the instantaneous 1-minute truth, over 79,860 daylight samples
(the gate is `CM3Up_Wm2_Avg > 20 W/m²` on the 1-minute series; pairing is
`merge_asof(direction="nearest", tolerance=5min)`):

```
offset applied to the stamp    RMS of the label error (GHI)
        0.00 min                     94.00 W/m²   <- what the code did
       -2.50 min                     74.99 W/m²   <- window centroid
                                     -20.2 %
```

The full sweep from −6 to +2 min has its minimum **exactly at −2.50**, the value
the end-stamp convention predicts. Theory and measurement agree:

```
-3.50 -> 81.15    -2.50 -> 74.99    -1.50 -> 81.69    0.00 -> 94.00
```

This is **label noise**: error no architecture removes.

## 3. The correction

`PrepareSensorConfig.timestamp_offset_minutes` (default `0.0`) adds minutes to
every stamp before pairing; `-2.5` moves the row to the centroid of the window it
actually averages. It shifts **only** which row is paired and the recorded
`distance_minutes` — never the values.

The field enters the manifest's resume hash (`"sensor"` belongs to
`_MANIFEST_CONFIG_SECTIONS`), so changing it invalidates the manifest instead of
silently reusing it, and it is recorded in the sidecar as
`thresholds.sensor_timestamp_offset_minutes`.

Over one and the same set of frames, the offset changes **49.9 % of the
`target_dhi`** (RMS of the difference 21.6 W/m²) and **14.4 % of the
`sky_class`**.

## 4. The factorial that measured the downstream effect

Design: alignment × pooling × task, 3 seeds per cell, 24 cells. The two alignment
arms share one frame tree, one embedding store and the same `split_id` — only the
manifest labels differ, so the comparison is genuinely paired.

Dataset: 46,015 samples, 81 days (2026-06-03 .. 08-26), `bare` tier, chronological
split with a 1-day gap (train 30,704 / val 7,003 / test 7,138).

Test, DHI RMSE in W/m²:

| align | pooling | task | s42 | s43 | s44 | mean | sd |
|---|---|---|---|---|---|---|---|
| raw | cls | multitask | 35.59 | 35.32 | 35.05 | **35.32** | 0.27 |
| raw | cls | DHI-only | 34.43 | 34.52 | 35.06 | 34.67 | 0.34 |
| raw | cls+mean | multitask | 34.58 | 34.58 | 34.92 | 34.69 | 0.20 |
| raw | cls+mean | DHI-only | 33.38 | 33.71 | 32.50 | 33.20 | 0.62 |
| cen | cls | multitask | 32.46 | 31.88 | 32.47 | 32.27 | 0.34 |
| cen | cls | DHI-only | 31.49 | 31.10 | 32.41 | 31.67 | 0.67 |
| cen | cls+mean | multitask | 32.81 | 30.79 | 31.01 | 31.54 | 1.11 |
| **cen** | **cls+mean** | **DHI-only** | 30.21 | 31.19 | 31.18 | **30.86** | 0.56 |

Main effects, paired (the other factors and the seed held fixed):

```
alignment  raw -> centroid    -2.89 +/- 0.22 W/m²
pooling    cls -> cls+mean    -0.91 +/- 0.22
task       multitask -> DHI-only -0.86 +/- 0.27
```

All three are real (|mean| > 2·standard error). Among the factors tested OVER
FROZEN EMBEDDINGS, alignment is the largest — larger than the two readout effects
combined. Section 8 shows that unfreezing the backbone is larger still.

The seed-to-seed σ on the baseline is 0.77 % of it (0.27 of 35.32), so effects on
the order of 1 % are measurable in this test block. Without that number none of
the deltas above could be asserted.

## 5. The bias is structured by solar geometry

The aggregate MBE of the best frozen cell is **−4.94 +/- 3.09** W/m² over 3 seeds
(−8.49 / −3.46 / −2.87). Over the factorial's 24 cells it is −3.42 +/- 3.42, with
a range from −9.62 to +1.86 and 20 of the 24 negative. **The aggregate value is
noisy and must not be quoted from a single seed.**

What IS robust is the structure by elevation, monotonic across every seed:

| solar elevation | frozen MBE (3 seeds) | n |
|---|---|---|
| 10–20° | +12.62 +/- 4.65 | 1,002 |
| 20–35° | +0.25 +/- 5.25 | 1,558 |
| 35–50° | −8.16 +/- 4.01 | 1,688 |
| 50–90° | −11.95 +/- 1.29 | 2,890 |

The model **underestimates the diffuse when the sun is high and overestimates
when it is low**. The reading by sky condition (`clear` −9.59, `cloudy` +5.36,
`partly_cloudy_diffuse` −18.12) comes from a single seed and serves as an
indication, not as a number.

## 6. Limits of the design, stated

- The test block falls **entirely in August** and is 57 % clear sky. A number
  measured here does not transport to another season without verification.
- None of this is comparable with the 38.4 W/m² published on 2026-08-13: a
  different test window, 81 days instead of 65, a different feature tier and, in
  the centroid arm, half the labels different.
- `skill_persistence` is −1.50 and always will be negative in this task: the
  estimate is at t=0 and 1-minute persistence is almost the answer.

## 7. The diffuse noise floor, measured and not transported

The question that decides whether it is worth going on with the model: **how much
of the remaining error is imposed by the 5-minute label?**

It was answered by direct measurement, not by transporting from GHI. The 2024
1-minute table carries `CMP21_Wm2_Avg` alive (18,187 distinct values), and in that
period the CMP21 *was* the site's diffuse channel (`docs/station-archive.md`,
range 2020-06-01 .. 2025-03-12). Comparing the instantaneous 1-minute diffuse
against the 5-minute label the pipeline would pick, over 79,743 daylight samples
(same daylight gate `CM3Up_Wm2_Avg > 20 W/m²`, plus the physical gate on the
diffuse itself given on each row):

| physical gate | raw stamp | centroid | reduction |
|---|---|---|---|
| [0, 800] W/m² | 21.66 | **15.35** | 29.1 % |
| [0, 600] | 21.56 | 14.92 | 30.8 % |
| [5, 500] | 21.25 | 14.73 | 30.7 % |
| [0, 1000] | 22.61 | 16.35 | 27.7 % |

The floor is ~15 W/m², stable across gates. With the best cell at 30.86 W/m² and
assuming model error and label noise independent:

```
model error = sqrt(30.86² − 15.35²) = 26.77 W/m²
fraction of the error VARIANCE that is label = 24.7 %
```

**Three quarters of the error is still the model's.** The architecture is not at
the ceiling the label imposes, and continuing to improve it has a return.

Caveats, because this is a measurement on another instrument in another era: the
2024 CMP21 runs under program v19, not the current v22, and the site's diffuse
sensor today is the PSP. The number transports as an order of magnitude, not as a
calibrated constant. It is still direct evidence, and it replaces an earlier
estimate of ~25 W/m² obtained by multiplying the GHI value by a factor — which
overestimated the floor by about 60 % and, if accepted, would have recommended
abandoning the modelling far too early.

## 8. The fine-tune: the frozen backbone was the bottleneck

Image mode, centroid join, single DHI head, 40 epochs with early stopping, 2 seeds
per depth. `image_size` 224, ~51 s/epoch, 482 MiB of VRAM.

| unfrozen depth | s42 | s43 | mean | vs. frozen |
|---|---|---|---|---|
| frozen (best cell) | — | — | 30.86 | — |
| last 2 of 12 blocks | 28.17 | 27.61 | 27.89 | −9.6 % |
| last 4 | 24.99 | 26.97 | 25.98 | −15.8 % |
| **all 12** | 21.11 | 20.50 | **20.80** | **−32.6 %** |

Monotonic in depth, and larger than every factor of the factorial combined.
R² 0.939, MAE 14.5, `skill_clearsky` +0.72. **No run hit the 40-epoch limit** —
all stopped early between 12 and 29, so "train longer" was not the lever; the
lever was how many parameters could learn.

## 9. The ceiling inverted

Taking the floor from section 7 **as an order of magnitude, not as a constant** —
it was measured on another instrument in another era, and the decomposition below
further assumes model error and label noise are independent, which was not
verified:

| assumed floor | implied model error | fraction of variance that is label |
|---|---|---|
| 14.7 W/m² (gate [5,500]) | 14.7 | 50 % |
| 15.35 (gate [0,800]) | 14.0 | 54 % |
| 16.35 (gate [0,1000]) | 12.8 | 62 % |

Whatever the gate, the qualitative reading is the same and it is what matters:
**before the fine-tune the label accounted for about a quarter of the error
variance; at 20.80 it accounts for the majority.** That reorders the priorities —
restoring a 1-minute table with `PSP_Wm2_Avg` went from a second-order lever to a
first-order one, not because the label got worse but because the model got better.

The exact number depends on two unverified assumptions (transport between
instruments, and independence), so it guides priority rather than serving as a
target.

## 10. The −8 W/m² bias survived everything tested

The fine-tune **did not remove** the bias — and, measured over 3 seeds on each
side, increased it: frozen **−4.94 +/- 3.09**, 12-block fine-tune
**−7.57 +/- 1.36**. At the same time it TIGHTENED the structure by elevation (the
seed-to-seed deviation falls from 1.3–5.3 to 0.7–2.2 W/m²):

| elevation | frozen | fine-tune 12 |
|---|---|---|
| 10–20° | +12.62 +/- 4.65 | +5.99 +/- 0.72 |
| 20–35° | +0.25 +/- 5.25 | −4.16 +/- 0.72 |
| 35–50° | −8.16 +/- 4.01 | −9.37 +/- 2.20 |
| 50–90° | −11.95 +/- 1.29 | −13.06 +/- 1.68 |

So: the fine-tune halved the RMSE and made the bias sharper and more
reproducible, not smaller. Two hypotheses for the cause were tested and **both
failed**.

**Hypothesis 1 — regression to the mean under MSE.** Loss sweep over the winning
configuration:

| loss | RMSE | MAE | MBE | R² |
|---|---|---|---|---|
| mse (3 seeds) | 20.74 +/- 0.32 | 14.44 | −7.57 | 0.939 |
| mae | 20.31 | 14.05 | −8.46 | 0.942 |
| huber | 21.80 | 15.45 | −11.21 | 0.933 |
| heteroscedastic | 32.41 | 23.07 | −17.09 | 0.852 |

The `mae` arm was the falsification: L1 seeks the median, so on a right-skewed
target it should **worsen** the bias. It did not. And `heteroscedastic`, which the
regression-head literature pointed to for tail underestimation, was by far the
worst. The bias survives L2, L1, Huber and Gaussian NLL: it is not an artifact of
the loss family. No loss beats MSE measurably — `mae` is within 1.3 σ on a single
seed.

**Hypothesis 2 — target drift in the chronological split.** Also fails, and in the
opposite direction:

```
mean DHI:  train 136.19   val 130.22   test 121.60 W/m²
```

The test set has LOWER diffuse than the training set, so a model anchored on the
training mean would overestimate (positive MBE). The observed one is negative.

**What is known:** the bias is more negative at high sun (−13.06 above 50°, +7.59
between 10° and 20°) and the test window has a mean elevation of 42.2° against
37.4° in training, which amplifies the effect. **What is not known:** the cause.
It is recorded as an open question. The next indicated test is to separate label
from model — checking whether the shade-ring correction applied to the PSP in
`sensors/calibration.py` has a residual elevation dependence, by comparing the
corrected diffuse against the modelled clear-sky diffuse per elevation band.

## 11. What this implies

In order of measured size, starting from the configuration as published:

| change | effect | cost |
|---|---|---|
| unfreeze the whole DINOv2 | **−10.1 W/m²** | config; ~30 min of GPU |
| fix the join to the centroid | **−2.9** | 5 lines of code |
| `cls+mean` pooling | −0.9 | one config line |
| single DHI head | −0.9 | one config line |
| change the loss | nothing measurable | — |

Published baseline 35.32 → **20.80 W/m²**, a 41 % reduction.

The approach was **not** doomed: it was handcuffed by a frozen backbone. The
finding the literature supports is exactly that — finetuning vs. frozen, and not
one specific architecture.

From here on the label accounts for the majority of the error variance (50–62 %
depending on the assumed gate), so the next lever is acquisition cadence, not the
encoder. And the bias remains unexplained: it is stable and monotonic in solar
elevation, it grew with the fine-tune (−4.94 → −7.57 W/m²), and the two
hypotheses tested were falsified.

See `docs/station-archive.md` for the chronology of the tables and
`docs/allsky-archive.md` for the camera clock.

---

## 12. How the reported run was produced, and what is left open

The configs for each cell of the study are NOT versioned — they are derived
mechanically from `configs/allsky/experiments/_base.yaml` plus the
`models/image_only.yaml` fragment, varying only four keys. To reconstruct the
study it is enough to generate the combinations:

| axis | values | where |
|---|---|---|
| alignment | `timestamp_offset_minutes` 0.0 and −2.5 | a copy of `local_prepare.yaml`, each with its own `output.dataset_dir` |
| pooling | `embeddings.pooling` `cls` and `cls+mean` | same; the cls+mean store needs `precompute-embeddings -o <dir>/embeddings_clsmean` |
| task | `targets.kindex.enabled` / `targets.sky.enabled` | in the experiment config |
| depth | `model.unfreeze_last_n` 2, 4, 12 with `data.input_mode: image` | in the experiment config |

The two alignment arms need distinct `output.dataset_dir`: the manifest name is
the constant `DATASET_MANIFEST_FILENAME`, so two offsets in the same directory
overwrite each other in silence.

**The run that produced the numbers above used a shortcut**: a single frame tree
and a single embedding store, with the centroid arm's manifest built in the same
directory and renamed. The labels, the splits and the `split_id` are the same ones
the recipe above produces; what the shortcut saved was re-extracting 46 thousand
frames.

### ImageNet standardization: a real defect whose fix made things worse

Measured over 3 seeds on each side, in the winning configuration:

| input | RMSE | MAE | MBE |
|---|---|---|---|
| raw `[0,1]` (as every run until 2026-08-28) | **20.74 +/- 0.33** | 14.44 | −7.57 |
| standardized by DINOv2's statistics | 21.35 +/- 0.22 | 14.56 | −7.34 |

The disagreement between the two paths was real — the embedding one standardized,
the image one did not. But **fixing it cost +0.61 W/m²**, about 2.7σ of the
standard error of the difference. A known and uncontrolled confounder:
`backbone_lr = 1e-5` and `lr = 3e-4` were chosen in the non-standardized regime,
and standardizing multiplies the input scale by 4–8×. The experiment that decides
is re-tuning `backbone_lr` under standardization; until then, "correct" and
"better" do not coincide here.

**A second gap, created by the 2026-08-28 standardization — half closed.**
Serving no longer diverges from training: `allsky.snapshot._image_as_chw` runs
the same chain the dataset does and ends in `imagenet_standardize`, and
`test_the_live_frame_is_prepared_the_way_the_training_frames_are` pins the two
together over the preprocessing pipeline and the resize.

What remains open is dispatch by **checkpoint vintage**. Every image-mode
checkpoint produced up to 2026-08-28 was trained on raw `[0, 1]` input, and the
snapshot now standardizes for all of them — so serving one of those older
checkpoints applies a recipe it was never fitted on, which is the same ~1.3σ
divergence in the red channel, now pointing the other way.
`training/checkpointing.py` records `backbone_info` with name, revision,
pooling, dim and frozen, and none of that distinguishes the two input recipes,
so there is still nothing to dispatch on. Recording the input recipe in
`backbone_info` is what closes it.

**A known gap, older than that change:** `allsky.snapshot._sensor_row_near` pairs
the sensor row on the raw stamp and does not know about
`timestamp_offset_minutes`, so the inference path does not reproduce the training
join. It was not fixed alongside because the right offset there depends on the
stamp convention of the processed CSV the snapshot consumes, and today that path
cannot even read the `labmim-sensor-process` export: the export's time column is
the unnamed index, and `SENSOR_TIME_COLUMNS` accepts only
`timestamp`/`TIMESTAMP`/`datetime`/`time`, so the read fails before pairing. The
two must be resolved together, with their own measurement.
