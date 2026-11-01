#!/bin/bash

# ==========================================
# Ising GPU - Ensemble Production Run
# ==========================================

# Exit immediately if a command fails
set -e 

# Global Simulation Parameters
NSWEEPS=3000000
BC="pbc"
NUM_RUNS=5  # Number of independent experiments per temp/size

# Lattice sizes represented as powers of 2
POWERS=(9 10)
TEMP_LOW=2.26
TEMP_HIGH=2.28
TEMP_STEP=0.001
TEMPS=$(seq $TEMP_LOW $TEMP_STEP $TEMP_HIGH)

echo "Starting Ensemble Grid Sweep..."
echo "Sweeps: $NSWEEPS | BC: $BC"
echo "Lattices: L = 2^${POWERS[*]}"
echo "Target window: ${TEMP_LOW} to ${TEMP_HIGH} with step ${TEMP_STEP}"
echo "Independent runs per configuration: $NUM_RUNS"
echo "=========================================="

# 1. Loop for multiple independent runs FIRST
for ((run=1; run<=NUM_RUNS; run++)); do
    TAG="therm_${run}"
    echo "=========================================="
    echo "Starting Full Temperature Sweep for Run ${run}/${NUM_RUNS}..."
    
    # 2. Loop over temperatures
    for t in $TEMPS; do
        formatted_temp=$(printf "%.3f" $t)
        
        # 3. Loop over each lattice size
        for p in "${POWERS[@]}"; do
            size=$((1 << p))
            # Execute the pre-compiled binary
            # Usage: ./run_shared.x -L <log2_size> -T <temp> -N <nsweeps> -b <obc|pbc> [-t <tag>] [-o <output.csv>] [-s <seed>] [-v]
            ./run_shared.x -L $p -T $formatted_temp -N $NSWEEPS -b $BC -t $TAG
            
        done
    done
done

echo ""
echo "=========================================="
echo "All ensemble simulations completed successfully!"
echo "=========================================="