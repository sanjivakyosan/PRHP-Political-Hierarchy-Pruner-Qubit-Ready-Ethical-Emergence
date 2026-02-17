# Qiskit Setup and Installation Guide

## Overview

The PRHP framework requires Qiskit for quantum simulation capabilities. This guide ensures Qiskit is always installed and running correctly.

## Quick Setup

### Option 1: Using Requirements (Recommended)

```bash
pip install -r requirements.txt
```

This will install:
- `qiskit>=0.40.0` - Core Qiskit library
- `qiskit-aer>=0.12.0` - Quantum simulator (required for Qiskit 2.x)

### Option 2: Using Setup Script

```bash
# Make script executable (first time only)
chmod +x scripts/setup_environment.sh

# Run setup
./scripts/setup_environment.sh
```

### Option 3: Manual Installation

```bash
pip install qiskit qiskit-aer
```

## Verification

### Check Installation

Run the verification script:

```bash
python3 scripts/ensure_qiskit.py
```

Or use the Python module:

```bash
python3 src/qiskit_check.py
```

### Expected Output

```
✓ Qiskit found: version 2.2.3
✓ qiskit-aer found: version 0.17.2
✓ Qiskit functionality test passed
✓ Qiskit is installed and working correctly
```

## Automatic Checks

### On Import

The framework automatically checks for Qiskit when importing quantum modules:

```python
from src.qubit_hooks import compute_phi
# If Qiskit is missing, a helpful warning is shown
```

### In Code

You can explicitly check Qiskit availability:

```python
from src.qiskit_check import ensure_qiskit_available, check_qiskit_installation

# Check and get info
is_installed, version, error = check_qiskit_installation()
if is_installed:
    print(f"Qiskit {version} is ready!")

# Or raise error if not available
try:
    ensure_qiskit_available()
except ImportError as e:
    print(e)  # Shows installation instructions
```

## CI/CD Integration

### GitHub Actions

If you have CI/CD workflows, add this step:

```yaml
- name: Install Qiskit
  run: |
    pip install qiskit qiskit-aer
    python3 scripts/ensure_qiskit.py
```

### Pre-commit Hook

You can add a pre-commit check:

```bash
#!/bin/bash
# .git/hooks/pre-commit
python3 scripts/ensure_qiskit.py || exit 1
```

## Troubleshooting

### Qiskit Not Found

**Error**: `Qiskit not found; using classical approximation.`

**Solution**:
```bash
pip install qiskit qiskit-aer
python3 scripts/ensure_qiskit.py
```

### qiskit-aer Missing (Qiskit 2.x)

**Error**: `qiskit-aer is missing (required for Qiskit 2.x)`

**Solution**:
```bash
pip install qiskit-aer
```

### Version Conflicts

If you encounter version conflicts:

```bash
pip install --upgrade qiskit qiskit-aer
```

### Virtual Environment

If using a virtual environment:

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Then install
pip install -r requirements.txt
```

## Requirements

### Minimum Versions

- Python: 3.8+
- Qiskit: 0.40.0+
- qiskit-aer: 0.12.0+ (for Qiskit 2.x)

### Recommended Versions

- Qiskit: 2.0.0+ (latest stable)
- qiskit-aer: 0.17.0+ (latest stable)

## Testing

After installation, test Qiskit functionality:

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
state = Statevector(qc)
print("✓ Qiskit is working!")
```

## Support

If you encounter issues:

1. Check Qiskit installation: `python3 scripts/ensure_qiskit.py`
2. Verify Python version: `python3 --version` (should be 3.8+)
3. Check pip: `pip --version`
4. Try reinstalling: `pip install --upgrade --force-reinstall qiskit qiskit-aer`

## Status

✅ **Qiskit is currently installed and working**

- Qiskit version: 2.2.3
- qiskit-aer version: 0.17.2
- Status: Ready for quantum simulations

