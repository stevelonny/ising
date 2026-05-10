#!/bin/bash

# Script to run Ising model simulations across all temperatures with both boundary conditions
# Usage: ./run_all_temps.sh [power_of_2] [nblocksweeps] [blocksize] [num_runs]

POWER=${1:-10}          # Grid size: 2^POWER (default: 10 = 1024x1024)
NBLOCKSWEEPS=${2:-10}   # Number of block sweeps (default: 10)
BLOCKSIZE=${3:-100}     # Block size (default: 100)
NUM_RUNS=${4:-3}        # Number of runs per temperature/boundary (default: 3)

# Create directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs_${TIMESTAMP}"
DATA_DIR="data_${TIMESTAMP}"
mkdir -p "$LOG_DIR" "$DATA_DIR"

# Summary file
SUMMARY_FILE="summary_${TIMESTAMP}.txt"

echo "========================================="
echo "Ising Model Temperature Sweep"
echo "========================================="
echo "Grid size: 2^$POWER = $((2**POWER))x$((2**POWER))"
echo "Block sweeps: $NBLOCKSWEEPS"
echo "Block size: $BLOCKSIZE"
echo "Number of runs per configuration: $NUM_RUNS"
echo "Log directory: $LOG_DIR"
echo "Data directory: $DATA_DIR"
echo "=========================================="
echo ""

# Counters and tracking
TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0
RETRY_QUEUE=()

{
    echo "========================================="
    echo "Ising Model Temperature Sweep"
    echo "========================================="
    echo "Started: $(date)"
    echo "Grid size: 2^$POWER = $((2**POWER))x$((2**POWER))"
    echo "Block sweeps: $NBLOCKSWEEPS"
    echo "Block size: $BLOCKSIZE"
    echo "Number of runs per configuration: $NUM_RUNS"
    echo "=========================================="
    echo ""
} > "$SUMMARY_FILE"

# Loop through temperatures
for BOUNDARY in "pbc" "obc"; do
    echo "=============== $BOUNDARY ==============="
    
    for TEMP_INT in {1..35}; do
        TEMP=$(echo "scale=1; $TEMP_INT * 0.1" | bc)
        
        for RUN in $(seq 1 $NUM_RUNS); do
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            
            echo -n "T=$TEMP run$RUN... "
            
            LOG_FILE="$LOG_DIR/T${TEMP}_${BOUNDARY}_run${RUN}.log"
            RUN_TAG="T${TEMP}_${RUN}"
            
            if timeout 3600 bash -c "cd '$DATA_DIR' && ../run.x $POWER $TEMP $NBLOCKSWEEPS $BLOCKSIZE $BOUNDARY '$RUN_TAG' > '../$LOG_FILE' 2>&1"; then
                # Check for successful thermalization
                if grep -q "Final thermalization: SUCCESS" "$LOG_FILE"; then
                    echo "✓"
                    SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
                else
                    echo "✗ (therm)"
                    RETRY_QUEUE+=("$TEMP:$BOUNDARY:$RUN")
                    FAILED_RUNS=$((FAILED_RUNS + 1))
                fi
            else
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 124 ]; then
                    echo "✗ (timeout)"
                else
                    echo "✗ (err)"
                fi
                RETRY_QUEUE+=("$TEMP:$BOUNDARY:$RUN")
                FAILED_RUNS=$((FAILED_RUNS + 1))
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "RETRY PHASE"
echo "=========================================="
echo "Failed runs to retry: ${#RETRY_QUEUE[@]}"
echo ""

RETRY_SUCCESS=0
RETRY_FAILED=0

# Retry failed runs
for RETRY_ITEM in "${RETRY_QUEUE[@]}"; do
    IFS=':' read -r TEMP BOUNDARY RUN <<< "$RETRY_ITEM"
    
    echo -n "RETRY T=$TEMP $BOUNDARY run$RUN... "
    
    LOG_FILE="$LOG_DIR/RETRY_T${TEMP}_${BOUNDARY}_run${RUN}.log"
    RUN_TAG="T${TEMP}_${RUN}"
    
    if timeout 3600 bash -c "cd '$DATA_DIR' && ../run.x $POWER $TEMP $NBLOCKSWEEPS $BLOCKSIZE $BOUNDARY '$RUN_TAG' > '../$LOG_FILE' 2>&1"; then
        if grep -q "Final thermalization: SUCCESS" "$LOG_FILE"; then
            echo "✓"
            RETRY_SUCCESS=$((RETRY_SUCCESS + 1))
            SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
            FAILED_RUNS=$((FAILED_RUNS - 1))
        else
            echo "✗"
            RETRY_FAILED=$((RETRY_FAILED + 1))
        fi
    else
        echo "✗"
        RETRY_FAILED=$((RETRY_FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "EXECUTION SUMMARY"
echo "=========================================="
echo "Total runs attempted: $TOTAL_RUNS"
echo "Successful on first try: $((SUCCESS_RUNS - RETRY_SUCCESS))"
echo "Successful after retry: $RETRY_SUCCESS"
echo "Total successful: $SUCCESS_RUNS"
echo "Permanently failed: $FAILED_RUNS"
echo "Data directory: $DATA_DIR"
echo "Log directory: $LOG_DIR"
echo "=========================================="

{
    echo ""
    echo "=========================================="
    echo "EXECUTION SUMMARY"
    echo "=========================================="
    echo "Total runs attempted: $TOTAL_RUNS"
    echo "Successful on first try: $((SUCCESS_RUNS - RETRY_SUCCESS))"
    echo "Successful after retry: $RETRY_SUCCESS"
    echo "Total successful: $SUCCESS_RUNS"
    echo "Permanently failed: $FAILED_RUNS"
    echo ""
    echo "Data directory: $DATA_DIR"
    echo "  Files created: $(ls -1 "$DATA_DIR" 2>/dev/null | wc -l)"
    echo "Log directory: $LOG_DIR"
    echo ""
    echo "Completed: $(date)"
    echo "=========================================="
} >> "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
