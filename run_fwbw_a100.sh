#!/bin/bash

# ==============================================================================
# CUDASW4 Forward-Backward Execution Script for A100
# ==============================================================================

# 1. Environment Setup
# If nvcc is not in your path, uncomment and edit the lines below:
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc (CUDA compiler) not found in PATH."
    echo "Please ensure CUDA is installed and added to your environment."
    exit 1
fi

echo "Using NVCC: $(which nvcc)"

# 2. Build the project
echo "----------------------------------------------------"
echo "Step 1: Compiling CUDASW4..."
echo "----------------------------------------------------"
make clean
make release -j$(nproc)

if [ ! -f ./align ] || [ ! -f ./makedb ]; then
    echo "Error: Compilation failed."
    exit 1
fi

# 3. Data and Database Paths
QUERY_FILE="allqueries.fasta"
DB_DIR="test_db_dir"
DB_NAME="test_db"

if [ ! -f "$QUERY_FILE" ]; then
    echo "Error: Query file '$QUERY_FILE' not found."
    exit 1
fi

# 4. Build Database
echo ""
echo "----------------------------------------------------"
echo "Step 2: Constructing database in $DB_DIR..."
echo "----------------------------------------------------"
mkdir -p "$DB_DIR"
./makedb "$QUERY_FILE" "$DB_DIR/$DB_NAME"

# 5. Run FwBw Alignment
echo ""
echo "----------------------------------------------------"
echo "Step 3: Running Forward-Backward on A100..."
echo "----------------------------------------------------"

# Optimization flags for A100:
# --fwbw        : Enables probabilistic alignment
# --uploadFull  : Keeps database in VRAM (A100 has plenty)
# --tsv         : Formats output for easy parsing
# --verbose     : Shows timing information
./align --query "$QUERY_FILE" \
        --db "$DB_DIR/$DB_NAME" \
        --fwbw \
        --uploadFull \
        --tsv \
        --verbose \
        --of results_fwbw.tsv

echo ""
echo "----------------------------------------------------"
echo "Execution Complete. Results: results_fwbw.tsv"
echo "----------------------------------------------------"
