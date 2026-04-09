# LabMiM — Site Architecture

## Folder Structure

```
site/
├── index.html                    ← Homepage
├── mapas_interativos.html        ← Interactive maps (Leaflet + Canvas)
├── mapas_meteorologicos.html     ← Meteorological map videos
├── monitoring.html               ← Environmental monitoring
├── climatologia.html             ← Climatology
├── team.html                     ← Team page
│
├── assets/                       ← All static resources
│   ├── css/
│   │   ├── template.css          ← Base template layout
│   │   ├── style.css             ← Global custom styles
│   │   ├── modern.css            ← Modern design system
│   │   ├── custom-themes.css     ← Theme variables & colors
│   │   └── maps.css              ← Interactive maps styles
│   ├── js/
│   │   ├── map-manager.js        ← MeteoMapManager class
│   │   ├── map-init.js           ← Map bootstrapping
│   │   ├── variables-config.js   ← Variable definitions & calculations
│   │   ├── charts-manager.js     ← Time-series chart rendering
│   │   ├── script-mapas.js       ← Meteorological maps page logic
│   │   ├── video.js              ← Video player controls
│   │   └── workers/
│   │       ├── color-calc.worker.js   ← Web Worker: color interpolation
│   │       └── json-parser.worker.js  ← Web Worker: JSON fetch+parse
│   ├── img/                      ← Logos, covers, partner images
│   ├── icon/                     ← Variable icons (mapas meteorológicos)
│   ├── graphs/                   ← Plotly HTMLs + PNGs (monitoring)
│   ├── json/                     ← Configuration JSONs
│   └── video/                    ← WebM video animations
│
├── geoJSON/                      ← Pipeline-generated grid geometry
│   ├── D01.geojson
│   ├── D02.geojson
│   ├── D03.geojson
│   └── D04.geojson
│
└── JSON/                         ← Pipeline-generated value data
    ├── D01_TEMP_001.json
    ├── D01_TEMP_002.json
    └── ...
```

## Data Flow

```
WRF Model (NetCDF)
    │
    ├─ labmim-wrf-geojson  (Python CLI)
    │   ├─ geoJSON/D0X.geojson     ← 1 per domain (grid geometry)
    │   └─ JSON/D0X_VAR_NNN.json   ← 1 per domain×variable×timestep
    │
    └─ labmim-wrf-figures  (Python CLI)
        └─ assets/video/*.webm     ← Animated map videos
```

## Data Contract

### GeoJSON (`geoJSON/{domain}.geojson`)

One file per WRF domain. Contains the grid geometry (identical across all variables).

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "resolucao_m": [27000, 27000]
  },
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1,lat1], [lon2,lat2], ...]]
      },
      "properties": {
        "linear_index": 0
      }
    }
  ]
}
```

**Key decisions:**
- Coordinates use **10 decimal places** (~0.01 mm precision)
- `colormap` was removed — it's now in `VARIABLES_CONFIG` (frontend)
- Compact JSON (no whitespace/indent)

### Values JSON (`JSON/{domain}_{variable}_{timestep:03d}.json`)

One file per domain × variable × timestep. Contains flat array of values indexed by `linear_index`.

```json
{
  "metadata": {
    "scale_values": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
    "date_time": "01/01/2024 12:00:00",
    "wind": { ... }  // optional, only for eolico variables
  },
  "values": [23.45, 24.12, null, ...]
}
```

**Key decisions:**
- Values rounded to **2 decimal places**
- Compact JSON (no whitespace/indent)
- `null` for missing data

## Module Map

| File | Purpose | Size |
|---|---|---|
| `mapas_interativos.html` | HTML structure only (no inline CSS/JS) | ~22 KB |
| `assets/css/maps.css` | All map-specific styles | ~40 KB |
| `assets/css/custom-themes.css` | Theme variables & colors | ~12 KB |
| `assets/js/map-manager.js` | `MeteoMapManager` class — core map logic | ~74 KB |
| `assets/js/map-init.js` | Bootstrap code — creates app + charts manager | ~2 KB |
| `assets/js/variables-config.js` | `VARIABLES_CONFIG` — variable definitions, scales, calculations | ~24 KB |
| `assets/js/charts-manager.js` | `ChartsManager` — time-series chart rendering | ~20 KB |
| `assets/js/workers/color-calc.worker.js` | Web Worker — offloads color interpolation | ~2 KB |
| `assets/js/workers/json-parser.worker.js` | Web Worker — offloads JSON fetch+parse | ~1 KB |

## Performance Optimizations

### Backend (Python pipeline)
1. **1 GeoJSON per domain** — eliminates 32 duplicate files (36 → 4)
2. **10 decimal precision** — reduced coordinate size
3. **Compact JSON** — no indent/whitespace (~40% smaller)

### Frontend
1. **In-memory JSON cache** (`_jsonCache`) — avoids re-downloading on variable switch
2. **Slider debounce** (100ms) — prevents avalanche of requests during drag
3. **Domain-only grid caching** — shared across all variables
4. **Batch-limited time series** — 10 concurrent fetches (not 73 × N)
5. **Web Workers** — offload color computation and JSON parsing to separate threads
6. **`<script defer>`** for CDNs — unblocks HTML parser
7. **`requestAnimationFrame` batching** — prevents DOM thrashing during grid updates
8. **Canvas renderer** — Leaflet uses `<canvas>` instead of SVG for grid

## Adding a New Variable

### 1. Python pipeline

In `src/labmim_micrometeorology/common/types.py`:
- Add entry to `WRFVariable` enum
- Add entry to `VARIABLE_COLORMAPS` 
- Add entry to `VARIABLE_NETCDF_MAP`

In `scripts/micromet/process_wrf_geojson.py`:
- Add handling in `_build_json_tasks_for_domain()` (or use the generic `else` branch)
- No GeoJSON changes needed — grid is shared

### 2. Frontend

In `assets/js/variables-config.js`:
- Add entry to `VARIABLES_CONFIG` with: `id`, `colors`, `unit`, `label`, `specificInfo()`

In `mapas_interativos.html`:
- Add `<option>` to the variable selector dropdown

### 3. Shell pipeline

No changes needed to `processa_wrf_04_python.sh` — just pass the new variable name with `-v`.
