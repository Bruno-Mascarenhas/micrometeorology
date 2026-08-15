# All-sky archive mirror (Planetário da UFBA)

How the `allsky sync-archive` and `allsky snapshot` commands get frames from the
Salvador meteor camera, what the source actually publishes, and the one finding
that changes how those frames must be timestamped.

## The source

`https://planetario.ufba.br/camera-de-meteoros-de-salvador` contains nothing but
an iframe pointing at `https://allsky.planetario.ufba.br/`, an
[AllskyTeam/allsky](https://github.com/AllskyTeam/allsky) v2024.12.06_06 website
for a ZWO ASI678MC on an OrangePi. Its archive is a flat, date-keyed listing:

| URL | Content |
| --- | --- |
| `videos/` | HTML index, one `<a href='./allsky-YYYYMMDD.mp4'>` per day |
| `videos/allsky-YYYYMMDD.mp4` | one-day timelapse, 15–32 MB, 1080p |
| `videos/thumbnails/allsky-YYYYMMDD.jpg` | listing thumbnail (403 on directory listing) |
| `keograms/keogram-YYYYMMDD.jpg` | keogram (not mirrored) |
| `startrails/startrails-YYYYMMDD.jpg` | startrails (not mirrored) |
| `image.jpg` | current full-resolution frame, refreshed every few seconds |
| `data.json` | today's sunrise/sunset, day/night capture flags |

There is **no** per-image archive: `images/` is 404, so the timelapse videos are
the only route to historical frames.

The archive is a rolling window — roughly 95 days when this was written
(2026-05-08 … 2026-08-10). Days fall off the end, which is the whole reason the
ledger never forgets a date it has seen (below) and the reason to run the sync
daily rather than backfilling occasionally.

## TLS: the chain is broken, and we repair it rather than skip it

`allsky.planetario.ufba.br` presents a valid leaf signed by *RNP ICPEdu GR46 OV
TLS CA 2025*, but the chain it serves contains a **different** CA's intermediate
(*RNP ICPEdu OV SSL CA 2019*). Every default trust store therefore fails with
`unable to get local issuer certificate` — curl, Python and browsers alike.

The correct intermediate is published at the leaf's Authority Information Access
URL. `build_ssl_context()` fetches it once, caches it in
`<data-dir>/.state/rnp-icpedu-gr46-2025.pem`, and injects it into an otherwise
stock context; the chain then builds to the GlobalSign Root R46 already in the
system store and the connection is **fully verified**. Verification is never
silently disabled.

Escape hatches, in order of preference: `--ca-file` with a PEM bundle (offline
machines), then `--insecure` (which logs a warning and makes the downloaded
training data unauthenticated).

The client is HTTPS-only by construction: its `OpenerDirector` is assembled with
no `file:` or `ftp:` handler, so a redirect cannot turn a download into a local
file read. Plain HTTP is enabled only for a `http://` base URL, which is how the
tests point it at a local server.

## Frame timestamps: read the overlay, do not model the cadence

**This is the finding that matters for training data.** `VideoConfig` maps frame
*N* of a timelapse to `start_time + N × minutes_per_frame` (defaults `06:00` and
`1.0`). This camera does not work that way, in two independent respects.

Ground truth is available: AllSky burns `YYYYMMDD HH:MM:SS` into the top-left of
every frame. Reading it off `allsky-20260810.mp4` gives:

| frame | overlay stamp | implied min/frame |
| ---: | --- | ---: |
| 0 | 2026-08-10 05:29:23 | — |
| 100 | 2026-08-10 07:12:27 | 1.031 |
| 300 | 2026-08-10 10:37:40 | 1.026 |
| 500 | 2026-08-10 14:04:00 | 1.026 |
| 700 | 2026-08-10 17:28:45 | 1.025 |
| 740 | 2026-08-10 18:11:16 | 1.063 |
| 800 | 2026-08-10 19:30:14 | 1.316 |
| 1000 | 2026-08-10 23:56:47 | 1.354 |
| 1200 | 2026-08-11 04:22:06 | 1.366 |
| 1255 | 2026-08-11 05:28:00 | 1.198 |

1. **The capture interval changes between day and night** — about 1.03 min/frame
   in daylight and about 1.32 min/frame after sunset, because the camera's
   exposure (and therefore its cadence) is longer at night. No single
   `minutes_per_frame` fits a whole video.
2. **Videos do not all start at the same hour.** The timelapse day runs roughly
   sunrise to sunrise, and on days when the camera only ran at night it covers
   only that:

   | video | first stamp | last stamp |
   | --- | --- | --- |
   | `allsky-20260508.mp4` | 2026-05-08 17:40:38 | 2026-05-09 04:59:36 |
   | `allsky-20260620.mp4` | 2026-06-20 05:31:49 | 2026-06-21 05:25:12 |
   | `allsky-20260810.mp4` | 2026-08-10 05:29:23 | 2026-08-11 05:22:46 |

Applying the `06:00 + 1.0 min` model to `allsky-20260810.mp4` mislabels its last
frame by **2 h 33 min**; applied to `allsky-20260508.mp4` it labels a 17:40
night frame as 06:00 in the morning.

The worst of that lands on night frames, which the night-elevation filter drops
anyway, so the number that matters is the error on the **daylight rows that
reach training**. Preparing `allsky-20260701.mp4` both ways and matching rows by
frame index gives it directly:

| label error, modelled vs overlay | value |
| --- | --- |
| median | 13.5 min |
| range | 4.6 – 24.1 min |
| rows beyond the 5 min pairing tolerance | 96.7 % |

Those timestamps drive the solar-geometry features and the merge against the
station logger, so nearly every daylight sample would be paired to a measurement
taken ten to twenty minutes from the sky it shows.

**And nothing downstream would notice.** The pairing distance is measured
against the *same* wrong timestamp, so it looks small and `ALIGNMENT_FAR` never
fires; the k-index divides a paired GHI by a clear-sky reference computed at
that same wrong time, so it comes out at a healthy ≈ 1.0. A manifest built on
mislabelled frames is internally consistent. The only thing that contradicts it
is the sky in the image, which is why the timestamp has to be read off the frame
rather than modelled.

Both entry points therefore read the overlay by default: `sync-archive --extract`
(via `--timestamps overlay`) and `prepare-local` (via `video.timestamps:
overlay` in the `PrepareConfig`). Setting either to `modelled` restores the old
`start_time`/`minutes_per_frame` mapping and logs a warning saying why you
probably do not want it.

**Implication for data prepared before this change:** any manifest built by an
earlier `prepare-local` carries the modelled timestamps and the error above, and
looks perfectly healthy from the inside. Re-running `prepare-local` rebuilds it
from overlay timestamps on its own — the extract step re-extracts whenever the
config that shaped the frames changed, so no `--force` is needed.

### Accuracy, and how a misread is caught

Over the eight archived days measured (9 187 frames, every frame read), the
reader produced a timestamp for **100 %** of frames, all strictly increasing,
with 11 frames (0.12 %) needing the correction below and none falling back to
interpolation.

Three layers keep a misread out of the data:

1. The reading itself must parse, and its date must be the video's day or the
   next; otherwise it is unreadable.
2. Where a cell's runner-up scores close to the winner — the `8` against `6`
   pair this font produces, when the neighbouring glyph's ink reaches into the
   search window — the pixels do not settle the digit. The capture sequence
   does: a frame landing outside its neighbours' bracket is re-read from the
   alternatives the pixels support, keeping the one nearest the bracket
   midpoint, and is flagged `corrected`. A reading its neighbours agree with is
   never second-guessed, and an unambiguous glyph is never overruled.
3. An isolated frame that still cannot be read is interpolated from its
   neighbours and flagged `interpolated`; timestamps that run **backwards**
   after all that are a hard error, and a video with more than 20 % unreadable
   frames is refused outright.

Gaps outside the usual 20 s – 10 min band are reported, not refused: the camera
really does emit occasional extra captures seconds apart (2026-07-15 has two),
and really does stop for hours.

### How the overlay is read

The stamp is drawn in a fixed bitmap font at fixed pixel columns, in a pure blue
(`RGB ≈ 5,3,205`) that separates cleanly from any sky, day or night. The reader
keys on that colour, slices the 14 digit cells, and matches each against an
embedded bank of 125 glyph exemplars, allowing a ±3 px shift because narrow
glyphs are positioned slightly differently between frames. Accuracy on the 15
hand-labelled frames used to build the bank is 15/15 under leave-one-out.

Reads are then *validated*, which is what makes them trustworthy: the date must
be the video's day or the next, stamps must increase, and the implied interval
must be between 20 s and 10 min. A frame failing any check is interpolated from
its neighbours and flagged `interpolated`; a video where more than 20 % of frames
fail is refused outright rather than timestamped from a guess.

Frame filenames keep the repo's minute-resolution convention
(`allsky-YYYYMMDD-HHMM.jpg`) so they continue to match the manifest's
`sample_id`. Because real captures are 61–92 s apart, two frames occasionally
land in the same minute; the second is skipped and counted rather than
overwriting the first.

## Deduplication

`<data-dir>/.state/archive-ledger.json` records, per day: the downloaded video
(path, size, sha256, `Last-Modified`), the frame extraction (directory, count,
`step`, `resize`, timestamp source) and every upload destination reached.

Two rules matter:

- **Entries are never removed.** The server drops days off its rolling window;
  a ledger that forgot them would re-upload the whole Drive folder the first
  time that happened.
- **Work is planned per *outcome*, not per download.** A day is selected when the
  video is missing, *or* frames were never extracted with the current
  `step`/`resize`/timestamp source, *or* any requested upload destination is
  absent from its record. So an upload that failed yesterday retries today without
  re-downloading the video, and adding `--upload` after a plain backfill uploads
  the days you already hold.
- **A re-extraction always re-uploads.** The frames destination is keyed by the
  day alone, so the record of the set being replaced names the same remote folder
  as its replacement; reading it as "already there" would leave Drive holding
  frames that no longer exist locally, with nothing to reconcile them.
- **A day the overlay reader refuses is skipped, not fatal.** `sync-archive`
  follows the same rule `prepare-local` does below: the refusal is filed in the
  ledger against the extraction parameters and the video's digest, so the daily
  job stops re-decoding that video every night and still exits 0. A re-download
  or a different `--step` clears the record and earns a fresh attempt.

The server ignores `If-Modified-Since` (it answers 200 with the full body), so
conditional GET is not the dedup mechanism — the ledger is, and it decides before
any request goes out.

A `flock` on `<ledger>.lock` makes a second concurrent run fail fast instead of
racing the first.

## Google Drive uploads

Transport is `rclone`, not the Drive REST API: it ships its own OAuth client (no
Google Cloud project), refreshes tokens indefinitely, does chunked resumable
uploads with a post-transfer checksum, and needs no browser after the first
configuration. A Drive API client left in "Testing" invalidates refresh tokens
after about a week, which would break an unattended cron job weekly.

One-time setup:

```sh
sudo apt install rclone
rclone config          # n) new remote -> name it "gdrive", type: drive
rclone lsd gdrive:     # verify
```

Remote layout under `--drive-root` (default `LabMiM/allsky`):

```
videos/allsky-YYYYMMDD.mp4
frames/YYYYMMDD/allsky-YYYYMMDD-HHMM.jpg
snapshots/YYYYMMDD/allsky-YYYYMMDD-HHMMSS.jpg  (+ .json, + .prediction.json)
```

A day's frames upload in a **single** rclone invocation; one process per JPEG
would cost more in process startup and OAuth than in transfer.

## Commands

Backfill everything the server still has, extracting frames as you go:

```sh
.venv/bin/allsky sync-archive --extract --step 10 --resize 512
```

Daily job, videos and frames to Drive. Day *D*'s video appears around 08:40 on
*D+1*, so run after 10:00 local:

```cron
30 10 * * * cd /home/brunosm/labmim/micrometeorology && \
  .venv/bin/allsky sync-archive --extract --step 10 --resize 512 \
  --upload both --drive-remote gdrive >> logs/allsky-sync.log 2>&1
```

Useful flags: `--list` (plan only), `--dry-run`, `--limit N` (backfill in
slices), `--since` / `--until`, `--prune-uploaded` (drop the local mp4 once it is
on Drive — the ledger still remembers it, so it is not re-downloaded),
`--rclone-arg` (repeatable, e.g. `--rclone-arg --bwlimit --rclone-arg 4M`).

Live frame, with a prediction:

```sh
.venv/bin/allsky snapshot --out output/allsky/live \
  --checkpoint output/allsky/experiments/v4_film/run/best.ckpt \
  --sensor-csv data/processed/station-latest.csv
```

The `--sensor-csv` file is the **processed** station export, in the published
physical units and with a time column the reader recognises; what it must look
like, and what happens to a reading it cannot screen, is in *The snapshot
prediction caveat* below.

An **embedding-mode** checkpoint carries no backbone of its own: the live frame is
encoded through the recipe recorded in the embedding store it was trained
against, and the path to that store is the absolute one baked in by the machine
that trained. `--embeddings-dir` overrides it, for a checkpoint copied off that
machine with its store shipped beside it. The store's `embeddings.meta.json` has
to name the whole recipe — backbone, revision, pooling, dim and storage dtype —
and a backbone that cannot reproduce it (a different pinned revision, a different
width, a different transform) fails the command instead of encoding the frame
differently from the vectors the model was fitted on. The flag is rejected for an
image-mode checkpoint, which reads no store at all.

## Feeding the sensor side: what `prepare-local` needs before it runs

The multimodal models read a sky image **and** the engineered sensor vector, so
every training row is a frame married to a logger record. `prepare-local` builds
that marriage, and three things decide whether it is sound.

**1. The logger has to cover the video days.** `build_manifest` pairs each frame
to the nearest sensor record within `sensor.tolerance_minutes` (5 min by
default) and drops the rest, so a logger export that stops before the archive
begins yields nothing. `prepare-local` now checks this **before extracting
anything** — extraction and visual QC run for minutes per video, and finding out
afterwards costs an hour for an empty manifest. It prints both ranges, exits
non-zero when they do not overlap, and warns when only some video days are
covered:

```
sensor coverage: 2025-05-14 15:25 .. 2026-04-24 13:00 (95572 records)
video days:      2026-05-20 .. 2026-08-01 (5 videos)
ERROR: the sensor record and the videos do not overlap, ...
```

**2. Roughly half of each video is night, and night is dropped.** With
`night_filter.min_solar_elevation_deg` at its default, a full 24-hour video
contributes only its daylight span:

| video | frames | rows reaching the manifest | daylight span |
| --- | ---: | ---: | --- |
| a full day (e.g. `20260701`) | ~1350 | ~610 (45 %) | ~06:45 – 16:30 |
| a night-only day (`20260508`, `20260520`) | ~600 | **0** | — |

Night-only days contribute nothing to a diffuse-radiation dataset, and they are
not the exception this table makes them look like: **the camera ran dusk-to-dawn
only until 2026-06-02 and switched to 24-hour capture on 2026-06-04**
(`20260603` is the transition). It is a meteor camera, so night is its purpose
and daylight is the recent addition. The break is visible in the published file
sizes without downloading anything — 13–17 MB for a night, 29–35 MB for a full
day — and over the 96 days the server held on 2026-08-12 it split 27 night-only
against 69 full days. Budget disk and expectations on the daylight half of the
full days alone.

**3. The frame timestamps have to be real**, for the reason in the section
above: the pairing is what consumes them, and it cannot tell a wrong timestamp
from a right one.

Some days cannot supply them. `allsky-20260604` steps its clock 7 s backwards at
frame 851 (20:19:53 -> 20:19:46) — the overlay reads cleanly on both sides, so
this is the camera's own clock and not an OCR slip — and the overlay reader
refuses to timestamp a sequence that goes backwards. `prepare-local` **skips**
such a day, names it, and repeats the list at the end of the extract step: one
unusable day out of 96 must not end a two-hour extraction or wedge the daily job
above. A skipped day contributes no manifest rows, exactly like a night-only
one, and the run still exits 0 — the fault is a permanent property of that day's
bytes, so an exit code that can never go green would signal nothing.

Once `prepare-local` succeeds, `allsky validate-dataset` is the gate to run
before training — it checks the manifest schema, the feature finiteness, the
anti-leakage policy, split integrity and that every referenced JPEG exists.

## Snapshot timestamps: the host's clocks disagree with each other

`allsky snapshot` names the captured file from **the overlay burned into the
frame**, not from the HTTP `Last-Modified` header, because that header is not
trustworthy on this host. Two requests seconds apart, while the true UTC time
was 23:31 and local Salvador time 20:31:

| request | `Date` | `Last-Modified` | overlay |
| --- | --- | --- | --- |
| first | `20:31:16 GMT` | `20:31:16 GMT` | — |
| second | `23:31:26 GMT` | `23:30:27 GMT` | `2026-08-11 20:30:07` |

The site sets an `SRV=wsrv01|…` affinity cookie: it is a load-balanced cluster,
and at least one backend reports **local time labelled GMT**. A client that
converts that value from UTC lands three hours in the past — an error large
enough to pair a night frame with an afternoon sensor row.

The source order is therefore: an explicit `timestamp=` argument, then the frame
overlay, then `Last-Modified`, then the local clock. The overlay and the header
are each accepted only if they fall within 10 minutes of now — a live frame is
by definition current, which is exactly the invariant that rejects a bad-node
response (and a rare glyph misread). The sidecar records all of them
(`overlay_stamp`, `server_last_modified`, `server_last_modified_as_local`) plus
which one was used, in `captured_at_source`.

The ~20 s gap between the overlay stamp and the file's mtime is the exposure and
processing time, and is immaterial at the 15-minute sensor-pairing tolerance.

## The snapshot prediction caveat

The multimodal models take a sky image **and** the engineered sensor vector.
Four of its columns — solar elevation, zenith, azimuth sin/cos, day-of-year
sin/cos — come from the timestamp and site, so a live frame always has them
exactly. The rest (air temperature, dew point, humidity, pressure, wind) come
from the station logger, which the camera does not publish.

`--sensor-csv` supplies them from a time-indexed station export whose time column
is named `timestamp`, `TIMESTAMP`, `datetime` or `time`, and whose values carry
the **published physical units of the processed export** — °C, %, mbar, m s⁻¹,
degrees, W m⁻² — at whatever averaging interval it was written on, not the
logger's raw pre-calibration signal and not necessarily a 5-minute grid. The row
nearest the capture time within `--tolerance-minutes` (default 15) is used, and a
row outside that window is refused rather than passed off as current conditions.

That row is then screened against the physical gates `sensor_limits` declares in
`configs/micromet/default.yaml` — the same gates the archive build applies — and
no longer against the raw-unit sentinel table: an hour that railed for only part
of its samples averages to a finite number no sentinel literal matches, and would
otherwise pass the finiteness screen and be served as a measurement. A value
outside its gate is dropped and imputed, and every column that happened to is
named in a warning. A column the configuration declares no gate for is dropped
for the same reason — a reading nothing can screen is not one a prediction should
be built on — and is imputed like any other missing column. If the configuration
declares **no** `sensor_limits` at all, there is nothing to screen a live reading
against and the command fails (exit 1) rather than serving the row: restore
`configs/micromet/default.yaml`, or predict without `--sensor-csv`.

Without the option, each missing column is imputed at its **training mean** (so
the standardized value is exactly 0), the prediction is still produced, and the
imputed column names are written into `<stem>.prediction.json` under
`features.imputed` and printed as a warning. A prediction from geometry plus an
average day is a weaker prediction; the output says which one you got.
