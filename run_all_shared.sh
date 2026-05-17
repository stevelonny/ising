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

# Lattice sizes represented as powers of 2: 
POWERS=(9 10 11)

echo "Starting Ensemble Grid Sweep..."
echo "Sweeps: $NSWEEPS | BC: $BC"
echo "Lattices: L = 2^${POWERS[*]}"
echo "Target window: T in [2.15, 2.30]"
echo "Independent runs per configuration: $NUM_RUNS"
echo "=========================================="

# Loop over each lattice size
for p in "${POWERS[@]}"; do
    size=$((1 << p))
    echo ""
    echo ">>> Starting lattice size L=$size <<<"
    
    # Loop over temperatures from 2.15 to 2.30 in steps of 0.01
    for t in $(seq 2.15 0.01 2.30); do
        formatted_temp=$(printf "%.2f" $t)
        echo "  -> Running T = $formatted_temp"
        
        # Loop for multiple independent runs
        for ((run=1; run<=NUM_RUNS; run++)); do
            # Create a unique tag for this specific experiment
            TAG="prod_run${run}"
            echo "     - Executing Experiment ${run}/${NUM_RUNS}..."
            
            # Execute the pre-compiled binary
            # Usage: ./ising_gpu <power2> <temp> <nsweeps> <obc|pbc> [run_tag]
            #./ising_gpu $p $formatted_temp $NSWEEPS $BC $TAG
            
        done
    done
    echo ">>> Completed lattice size L=$size <<<"
done

echo ""
echo "=========================================="
echo "All ensemble simulations completed successfully!"
echo "=========================================="