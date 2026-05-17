#!/bin/bash
# Setup virtual environment for Ising GPU Benchmark Suite

set -e

PROJECT_DIR="/home/steve/projects/ising"
VENV_DIR="${PROJECT_DIR}/venv"

echo "=========================================="
echo "Setting up Python virtual environment"
echo "=========================================="

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        echo "Removed old venv"
    else
        echo "Using existing venv"
        source "$VENV_DIR/bin/activate"
        exit 0
    fi
fi

echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing required packages..."
python -m pip install \
    pandas \
    numpy \
    matplotlib \
    seaborn

echo "=========================================="
echo "Virtual environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run benchmarks:"
echo "  python $PROJECT_DIR/run_benchmarks.py"
echo ""
