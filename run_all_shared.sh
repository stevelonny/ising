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
TEMPS=$(seq 2.18 0.01 2.30)

echo "Starting Ensemble Grid Sweep..."
echo "Sweeps: $NSWEEPS | BC: $BC"
echo "Lattices: L = 2^${POWERS[*]}"
echo "Target window: ${TEMPS}"
echo "Independent runs per configuration: $NUM_RUNS"
echo "=========================================="

# 1. Loop for multiple independent runs FIRST
for ((run=1; run<=NUM_RUNS; run++)); do
    TAG="therm_${run}"
    echo "=========================================="
    echo "Starting Full Temperature Sweep for Run ${run}/${NUM_RUNS}..."
    
    # 2. Loop over temperatures
    for t in $TEMPS; do
        formatted_temp=$(printf "%.2f" $t)
        
        # 3. Loop over each lattice size
        for p in "${POWERS[@]}"; do
            size=$((1 << p))
            # Execute the pre-compiled binary
            # Usage: ./run_shared.x <power2> <temp> <nsweeps> <obc|pbc> [run_tag]
            ./run_shared.x $p $formatted_temp $NSWEEPS $BC $TAG
            
        done
    done
done

echo ""
echo "=========================================="
echo "All ensemble simulations completed successfully!"
echo "=========================================="