#!/bin/bash

# Script to run Ising model simulations across all temperatures with both boundary conditions
# Usage: ./run_all_temps.sh [power_of_2] [nblocksweeps] [blocksize] [num_runs]

POWER=${1:-10}          # Grid size: 2^POWER (default: 10 = 1024x1024)
NBLOCKSWEEPS=${2:-10}   # Number of block sweeps (default: 10)
BLOCKSIZE=${3:-100}     # Block size (default: 100)
NUM_RUNS=${4:-3}        # Number of runs per temperature/boundary (default: 3)

# Create logs directory
LOG_DIR="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Ising Model Temperature Sweep"
echo "========================================="
echo "Grid size: 2^$POWER = $((2**POWER))x$((2**POWER))"
echo "Block sweeps: $NBLOCKSWEEPS"
echo "Block size: $BLOCKSIZE"
echo "Number of runs per configuration: $NUM_RUNS"
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Counters
TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

# Loop through temperatures
for BOUNDARY in "pbc" "obc"; do
    echo "=============== $BOUNDARY ==============="
    
    for TEMP_INT in {1..35}; do
        TEMP=$(echo "scale=1; $TEMP_INT * 0.1" | bc)
        
        for RUN in $(seq 1 $NUM_RUNS); do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            
            echo -n "T=$TEMP run$RUN... "
            
            LOG_FILE="$LOG_DIR/T${TEMP}_${BOUNDARY}_run${RUN}.log"
            
            if timeout 3600 ./run.x $POWER $TEMP $NBLOCKSWEEPS $BLOCKSIZE $BOUNDARY > "$LOG_FILE" 2>&1; then
                # Check for successful thermalization
                if grep -q "Final thermalization: SUCCESS" "$LOG_FILE"; then
                    echo "✓"
                    SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
                else
                    echo "✗ (therm)"
                    FAILED_RUNS=$((FAILED_RUNS + 1))
                fi
            else
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 124 ]; then
                    echo "✗ (timeout)"
                else
                    echo "✗ (err)"
                fi
                FAILED_RUNS=$((FAILED_RUNS + 1))
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "Completed!"
echo "Total runs: $TOTAL_RUNS"
echo "Successful: $SUCCESS_RUNS"
echo "Failed: $FAILED_RUNS"
echo "Data saved to: size_*_b*_n*_bc_*.dat"
echo "Logs saved to: $LOG_DIR"
echo "=========================================="
