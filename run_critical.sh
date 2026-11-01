#!/bin/bash

# ============================================================
# Ising GPU - Critical Temperature Investigation
# ============================================================
# Mirrors run_all_shared.sh but targets a fine temperature grid
# around Tc ≈ 2.26919 with multiple lattice sizes to allow
# finite-size scaling analysis (chi, Cv, Binder cumulant).
#
# Usage:
#   ./run_critical.sh [OPTIONS]
#
# Options (override via environment or positional args):
#   -N <nsweeps>      Measurement sweeps per run     [default: 3000000]
#   -R <num_runs>     Independent runs per config    [default: 8]
#   -b <bc>           Boundary conditions obc|pbc    [default: pbc]
#   -o <outfile>      Output CSV file                [default: master_results.csv]
#   -h                Show this help and exit
#
# Lattice sizes and temperature grid are defined in the
# "Configuration" section below — edit directly for experiments.
# ============================================================

set -euo pipefail

# ---- defaults ------------------------------------------------
NSWEEPS=3000000
NUM_RUNS=8
BC="pbc"
OUTFILE="master_results.csv"

# ---- parse flags ---------------------------------------------
while getopts "N:R:b:o:h" opt; do
    case $opt in
        N) NSWEEPS="$OPTARG" ;;
        R) NUM_RUNS="$OPTARG" ;;
        b) BC="$OPTARG" ;;
        o) OUTFILE="$OPTARG" ;;
        h)
            sed -n '2,/^set -/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option -$OPTARG. Use -h for help." >&2; exit 1 ;;
    esac
done

# ---- validate binary -----------------------------------------
if [[ ! -x "./run_shared.x" ]]; then
    echo "ERROR: ./run_shared.x not found or not executable." >&2
    echo "       Run:  make run_shared.x" >&2
    exit 1
fi

if [[ "$BC" != "pbc" && "$BC" != "obc" ]]; then
    echo "ERROR: -b must be 'pbc' or 'obc'." >&2
    exit 1
fi

# ============================================================
# Configuration — edit these to change the sweep
# ============================================================

# Lattice sizes as log2(L): L = 2^p
# p=7  ->  128x128
# p=8  ->  256x256
# p=9  ->  512x512
# p=10 -> 1024x1024
POWERS=(7 8 9 10)

# Critical temperature (exact 2D Ising on square lattice, J=1, k_B=1)
TC=2.26918531421

# Fine grid around Tc: from Tc-DELTA to Tc+DELTA in STEP increments
# Using bc for precision; seq can lose digits near Tc
DELTA=0.05          # half-width of temperature window
STEP=0.005          # temperature spacing (adjust for finer/coarser grid)

# ============================================================

# ---- build temperature list with python for precision -------
TEMPS=$(python3 -c "
import numpy as np
tc  = $TC
dl  = $DELTA
st  = $STEP
temps = np.arange(tc - dl, tc + dl + st*0.5, st)
for t in temps:
    print(f'{t:.6f}')
")

NTEMPS=$(echo "$TEMPS" | wc -l)

echo "============================================================"
echo " Ising GPU — Critical Temperature Fine-Grid Sweep"
echo "============================================================"
echo " Binary        : ./run_shared.x"
echo " Output CSV    : $OUTFILE"
echo " Boundary cond : $BC"
echo " Sweeps/run    : $NSWEEPS"
echo " Ind. runs     : $NUM_RUNS"
echo " Lattice sizes : L = 2^{${POWERS[*]}}"
echo " Tc (exact)    : $TC"
echo " T window      : [$(echo "$TEMPS" | head -1), $(echo "$TEMPS" | tail -1)]"
echo " T step        : $STEP  ($NTEMPS temperatures)"
echo " Total jobs    : $((NUM_RUNS * ${#POWERS[@]} * NTEMPS))"
echo "============================================================"
echo ""

TOTAL=0
DONE=0
FAILED=0
FAILED_LIST=()

# ---- main loop -----------------------------------------------
# Outer: independent run index (for ensemble averaging)
for ((run=1; run<=NUM_RUNS; run++)); do
    TAG="crit_run${run}"
    echo "============================================================"
    echo " Run ${run}/${NUM_RUNS}  (tag: ${TAG})"
    echo "============================================================"

    # Middle: temperatures (coarse → fine grid, centred on Tc)
    for t in $TEMPS; do

        # Inner: lattice sizes (small first so GPU is never idle)
        for p in "${POWERS[@]}"; do
            L=$((1 << p))
            TOTAL=$((TOTAL + 1))

            printf "  L=%-5d  T=%-10s  run=%d/%d  ... " "$L" "$t" "$run" "$NUM_RUNS"

            # Each call appends one row to $OUTFILE
            if ./run_shared.x \
                    -L "$p"       \
                    -T "$t"       \
                    -N "$NSWEEPS" \
                    -b "$BC"      \
                    -t "$TAG"     \
                    -o "$OUTFILE" \
                    2>/dev/null; then
                DONE=$((DONE + 1))
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
echo " Critical sweep completed"
echo "============================================================"
echo " Total jobs    : $TOTAL"
echo " Successful    : $DONE"
echo " Failed        : $FAILED"
if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo " Failed configurations:"
    for item in "${FAILED_LIST[@]}"; do
        echo "   - $item"
    done
fi
echo " Output CSV    : $OUTFILE"
echo "============================================================"
