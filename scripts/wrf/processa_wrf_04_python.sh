#!/bin/bash
# processa_wrf_04_python.sh — WRF Figure & GeoJSON Generation
#
# Updated to use the labmim-micrometeorology Python package (Cartopy-based).
# No longer requires a separate conda basemap environment.
#
# Called by processa_wrf_00_system.scpt with arguments:
#   $1 = yyyymmdd       (simulation date)
#   $2 = WRFoutput      (directory with wrfout_d0X files)
#   $3 = DOMini         (first domain number, e.g. 1)
#   $4 = DOMfim         (last domain number, e.g. 4)
#   $5+ = GRAFICOS_WRF  (variables: temperature wind rain SWDOWN HFX LH ...)

echo " "
echo '    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐'
echo "    + PYTHON - GENERATING WRF FIGURES AND JSON"
echo '    └──────────────────────────────────────────────────────────────────────────────────────────────────────────┘'
echo " "

# Read arguments
yyyymmdd=$1
WRFoutput=$2
DOMini=$3
DOMfim=$4
shift 4
GRAFICOS_WRF=("$@")

year=${yyyymmdd:0:4}
month=${yyyymmdd:4:2}
day=${yyyymmdd:6:2}

# Working directories
dir_local=$( grep dir_local /home/models/WRF/wrf-op/.FUNCTIONS_WRF_environment | awk '{print $2}' )

# Python environment (project venv — no basemap needed)
source ${dir_local}/venv/bin/activate 2>/dev/null || true

# Number of workers: use all but 4 cores
NPROC=$(nproc)
WORKERS=$((NPROC - 4))
if [ $WORKERS -lt 1 ]; then WORKERS=1; fi

echo "    + PYTHON - date            = ${yyyymmdd}"
echo "    + PYTHON - WRF output dir  = ${WRFoutput}"
echo "    + PYTHON - domains         = d0${DOMini} to d0${DOMfim}"
echo "    + PYTHON - workers         = ${WORKERS} / ${NPROC} cores"
echo "    + PYTHON - variables       = ${GRAFICOS_WRF[*]}"

# Build domain flags
DOMAIN_FLAGS=""
for d in $(seq $DOMini $DOMfim); do
    DOMAIN_FLAGS="${DOMAIN_FLAGS} -D ${d}"
done

# Build variable flags
VAR_FLAGS=""
for v in "${GRAFICOS_WRF[@]}"; do
    VAR_FLAGS="${VAR_FLAGS} -v ${v}"
done

# ─── Phase 1: Generate Figures + WebM ───────────────────────────────────────

echo "    + PYTHON - Phase 1: Generating figures and videos..."

labmim-wrf-figures \
    --wrf-dir "${WRFoutput}" \
    --date "${yyyymmdd}" \
    ${DOMAIN_FLAGS} \
    --output "${WRFoutput}/figures" \
    ${VAR_FLAGS} \
    --workers ${WORKERS} \
    --also-video \
    --log-level INFO

echo "    + PYTHON - Phase 1 complete"

# ─── Phase 2: Generate GeoJSON + JSON for site ──────────────────────────────

# Determine site output paths
dir_maps=${dir_local}/site
path_geojson=${dir_maps}/GeoJSON
path_json=${dir_maps}/JSON

echo "    + PYTHON - Phase 2: Generating GeoJSON and JSON..."
echo "    + PYTHON - GeoJSON dir = ${path_geojson}"
echo "    + PYTHON - JSON dir    = ${path_json}"

# Clean old files
rm -f ${path_geojson}/* 2>/dev/null
rm -f ${path_json}/* 2>/dev/null

# NOTE: We intentionally omit variable flags here so that labmim-wrf-geojson
# uses its DEFAULT_VARS list (which includes all variables needed for the
# interactive maps). The legacy system had separate variable sets for figures
# (GRAFICOS_WRF) and interactive maps (INTERATIVEMAP). Since this script now
# handles both, we let Phase 1 use GRAFICOS_WRF for figures, and Phase 2 use
# the Python-side defaults for the maps. If custom variables are passed via
# --var-flags, the Python pipeline normalizes legacy names (e.g. poteolico50,
# poteolico100, poteolico150 → poteolico) automatically.

labmim-wrf-geojson \
    --wrf-dir "${WRFoutput}" \
    --date "${yyyymmdd}" \
    ${DOMAIN_FLAGS} \
    --output-dir "${path_json}" \
    --geojson-dir "${path_geojson}" \
    --workers ${WORKERS} \
    --log-level INFO

echo "    + PYTHON - Phase 2 complete"

# ─── Cleanup ────────────────────────────────────────────────────────────────

# Remove intermediate PNGs (already encoded into WebM)
rm -f ${WRFoutput}/figures/*.png 2>/dev/null

echo " "
echo "    + PYTHON - All done"
echo " "
