#!/bin/bash

# ============================================================
# Ising GPU - Critical Temperature Sweep Continuation
# ============================================================
# Resumes an interrupted or partial critical sweep by checking
# which (L, T, run_tag) configurations are already recorded in
# the output CSV (default: master_results.csv) and skipping them.
#
# Usage:
#   ./continue_critical.sh [OPTIONS]
#
# Options:
#   -N <nsweeps>      Measurement sweeps per run (overrides per-size defaults)
#   -R <num_runs>     Total independent runs per config  [default: 1]
#   -b <bc>           Boundary conditions obc|pbc        [default: pbc]
#   -o <outfile>      Output CSV file                    [default: master_results.csv]
#   -d                Dry-run: inspect completed and pending jobs without running
#   -u                Only skip if marked thermalized (is_thermalized == 1)
#   -h                Show this help and exit
#
# Configuration (grid, sizes, sweeps) matches run_critical.sh.
# ============================================================

set -euo pipefail
export LC_ALL=C

# ---- defaults ------------------------------------------------
NSWEEPS=""   # empty = use per-size NSWEEPS_MAP; set via -N to override all sizes
NUM_RUNS=1
BC="pbc"
OUTFILE="master_results.csv"
DRY_RUN=false
REQUIRE_THERMALIZED=false

# Clean exit on Ctrl+C / SIGINT
trap 'echo -e "\nInterrupted by user. Exiting..."; exit 130' INT TERM

# ---- parse flags ---------------------------------------------
while getopts "N:R:b:o:duh" opt; do
    case $opt in
        N) NSWEEPS="$OPTARG" ;;
        R) NUM_RUNS="$OPTARG" ;;
        b) BC="$OPTARG" ;;
        o) OUTFILE="$OPTARG" ;;
        d) DRY_RUN=true ;;
        u) REQUIRE_THERMALIZED=true ;;
        h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option -$OPTARG. Use -h for help." >&2; exit 1 ;;
    esac
done

# ---- validate binary -----------------------------------------
if [[ "$DRY_RUN" = false && ! -x "./run_shared.x" ]]; then
    echo "ERROR: ./run_shared.x not found or not executable." >&2
    echo "       Run:  make run_shared.x" >&2
    exit 1
fi

if [[ "$BC" != "pbc" && "$BC" != "obc" ]]; then
    echo "ERROR: -b must be 'pbc' or 'obc'." >&2
    exit 1
fi

# ============================================================
# Configuration — matches run_critical.sh
# ============================================================

# Lattice sizes as log2(L): L = 2^p
# p=6  ->   64x64
# p=7  ->  128x128
# p=8  ->  256x256
# p=9  ->  512x512
POWERS=(6 7 8)

# Per-size sweep counts (accounts for critical slowing down: tau ~ L^2.17).
# The -N flag overrides all sizes with a single value when specified.
declare -A NSWEEPS_MAP
NSWEEPS_MAP[6]=5000000    #  64x64  — tau ~150
NSWEEPS_MAP[7]=10000000   # 128x128 — tau ~700
NSWEEPS_MAP[8]=30000000   # 256x256 — tau ~3000
NSWEEPS_MAP[9]=20000000   # 512x512 — tau ~8000

# Critical temperature (exact 2D Ising on square lattice, J=1, k_B=1)
TC=2.269

# Fine grid around Tc: from Tc-DELTA to Tc+DELTA in STEP increments
DELTA=0.03          # half-width of temperature window
STEP=0.005          # temperature spacing (adjust for finer/coarser grid)

# ============================================================

# ---- build temperature list with python for precision -------
PYTHON_BIN="python3"
if [[ -x ".venv/bin/python3" ]]; then
    PYTHON_BIN=".venv/bin/python3"
fi

TEMPS=$("$PYTHON_BIN" -c "
tc = $TC
dl = $DELTA
st = $STEP
t = tc - dl
while t <= tc + dl + st * 0.5:
    print(f'{t:.6f}')
    t += st
")

NTEMPS=$(echo "$TEMPS" | wc -l)
TOTAL_PLANNED=$((NUM_RUNS * ${#POWERS[@]} * NTEMPS))

# ---- load completed runs from CSV ----------------------------
declare -A COMPLETED
LOADED_COUNT=0

if [[ -f "$OUTFILE" ]]; then
    while IFS=, read -r c_L c_T c_tag c_therm rest || [[ -n "$c_L" ]]; do
        c_L=$(echo "$c_L" | tr -d '[:space:]')
        # Skip header or invalid lines (L must be integer)
        [[ "$c_L" =~ ^[0-9]+$ ]] || continue

        c_T=$(echo "$c_T" | tr -d '[:space:]')
        c_tag=$(echo "$c_tag" | tr -d '[:space:]')
        c_therm=$(echo "$c_therm" | tr -d '[:space:]')

        if [[ "$REQUIRE_THERMALIZED" = true && "$c_therm" != "1" ]]; then
            continue
        fi

        c_T_fmt=$(printf "%.4f" "$c_T" 2>/dev/null || echo "$c_T")
        key="${c_L}_${c_T_fmt}_${c_tag}"
        if [[ -z "${COMPLETED[$key]:-}" ]]; then
            COMPLETED["$key"]=1
            LOADED_COUNT=$((LOADED_COUNT + 1))
        fi
    done < "$OUTFILE"
fi

echo "============================================================"
echo " Ising GPU — Critical Sweep Continuation"
echo "============================================================"
echo " Mode          : $(if $DRY_RUN; then echo "DRY RUN (no simulations)"; else echo "CONTINUATION RUN"; fi)"
echo " Binary        : ./run_shared.x"
echo " Output CSV    : $OUTFILE"
echo " Saved entries : $LOADED_COUNT matching runs indexed"
echo " Boundary cond : $BC"
echo " Sweeps/run    : ${NSWEEPS:-per-size (64→5M, 128→10M, 256→30M, 512→20M)}"
echo " Ind. runs     : $NUM_RUNS"
echo " Lattice sizes : L = 2^{${POWERS[*]}}"
echo " Tc (exact)    : $TC"
echo " T window      : [$(echo "$TEMPS" | head -1), $(echo "$TEMPS" | tail -1)]"
echo " T step        : $STEP  ($NTEMPS temperatures)"
echo " Total planned : $TOTAL_PLANNED jobs"
echo "============================================================"
echo ""

TOTAL=0
SKIPPED=0
DONE=0
FAILED=0
FAILED_LIST=()

# ---- main loop -----------------------------------------------
for ((run=1; run<=NUM_RUNS; run++)); do
    TAG="crit_run${run}"
    echo "============================================================"
    echo " Run ${run}/${NUM_RUNS}  (tag: ${TAG})"
    echo "============================================================"

    for t in $TEMPS; do
        t_fmt=$(printf "%.4f" "$t")

        for p in "${POWERS[@]}"; do
            L=$((1 << p))
            TOTAL=$((TOTAL + 1))

            N="${NSWEEPS:-${NSWEEPS_MAP[$p]}}"
            key="${L}_${t_fmt}_${TAG}"

            printf "  L=%-5d  T=%-10s  N=%-10d  run=%d/%d  ... " "$L" "$t" "$N" "$run" "$NUM_RUNS"

            if [[ -n "${COMPLETED[$key]:-}" ]]; then
                SKIPPED=$((SKIPPED + 1))
                echo "SKIPPED (already in $OUTFILE)"
                continue
            fi

            if [[ "$DRY_RUN" = true ]]; then
                DONE=$((DONE + 1))
                echo "WOULD RUN (dry-run)"
                continue
            fi

            # Execute run_shared.x appending to $OUTFILE
            if ./run_shared.x \
                    -L "$p"   \
                    -T "$t"   \
                    -N "$N"   \
                    -b "$BC"  \
                    -t "$TAG" \
                    -o "$OUTFILE" \
                    2>/dev/null; then
                DONE=$((DONE + 1))
                COMPLETED["$key"]=1
                echo "OK"
            else
                FAILED=$((FAILED + 1))
                FAILED_LIST+=("L=${L} T=${t} run=${run}")
                echo "FAILED"
            fi
        done
    done
done

# ---- summary -------------------------------------------------
echo ""
echo "============================================================"
if [[ "$DRY_RUN" = true ]]; then
    echo " Dry-Run Summary"
else
    echo " Critical Sweep Summary"
fi
echo "============================================================"
echo " Total configs : $TOTAL"
echo " Already saved : $SKIPPED (skipped)"
if [[ "$DRY_RUN" = true ]]; then
    echo " Would execute : $DONE"
else
    echo " Succeeded     : $DONE"
    echo " Failed        : $FAILED"
fi
if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo " Failed configurations:"
    for item in "${FAILED_LIST[@]}"; do
        echo "   - $item"
    done
fi
echo " Output CSV    : $OUTFILE"
echo "============================================================"
