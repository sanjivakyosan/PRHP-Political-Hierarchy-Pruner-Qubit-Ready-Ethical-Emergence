#!/bin/bash
# Setup script to ensure all dependencies including Qiskit are installed

set -e  # Exit on error

echo "=========================================="
echo "PRHP Framework - Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements from requirements.txt..."
python3 -m pip install -r requirements.txt

# Ensure Qiskit is working
echo ""
echo "Verifying Qiskit installation..."
python3 scripts/ensure_qiskit.py

echo ""
echo "=========================================="
echo "✓ Environment setup complete!"
echo "=========================================="

