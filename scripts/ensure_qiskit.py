#!/usr/bin/env python3
"""
Script to ensure Qiskit is installed and working.
This script checks for Qiskit installation and installs it if missing.
"""

import sys
import subprocess
import importlib

def check_qiskit_installed():
    """Check if Qiskit is installed and working."""
    try:
        import qiskit
        version = qiskit.__version__
        print(f"✓ Qiskit found: version {version}")
        return True, version
    except ImportError:
        print("✗ Qiskit not found")
        return False, None

def check_qiskit_aer_installed():
    """Check if qiskit-aer is installed (required for Qiskit 2.x)."""
    try:
        import qiskit_aer
        version = qiskit_aer.__version__
        print(f"✓ qiskit-aer found: version {version}")
        return True, version
    except ImportError:
        print("✗ qiskit-aer not found (optional for Qiskit 1.x, required for 2.x)")
        return False, None

def check_qiskit_functionality():
    """Test if Qiskit is working correctly."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, DensityMatrix
        
        # Create a simple test circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        
        # Test statevector
        statevector = Statevector(qc)
        rho = DensityMatrix(statevector)
        
        print("✓ Qiskit functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Qiskit functionality test failed: {e}")
        return False

def install_qiskit():
    """Install Qiskit using pip."""
    print("\nInstalling Qiskit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qiskit", "qiskit-aer"])
        print("✓ Qiskit installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install Qiskit: {e}")
        return False

def main():
    """Main function to ensure Qiskit is installed and working."""
    print("=" * 60)
    print("Qiskit Installation Check")
    print("=" * 60)
    print()
    
    # Check Qiskit
    qiskit_installed, qiskit_version = check_qiskit_installed()
    
    if not qiskit_installed:
        print("\nQiskit is not installed. Installing...")
        if not install_qiskit():
            print("\n✗ Failed to install Qiskit. Please install manually:")
            print("  pip install qiskit qiskit-aer")
            return 1
        # Re-check after installation
        qiskit_installed, qiskit_version = check_qiskit_installed()
    
    # Check qiskit-aer (for Qiskit 2.x)
    aer_installed, aer_version = check_qiskit_aer_installed()
    
    if not aer_installed and qiskit_version and int(qiskit_version.split('.')[0]) >= 2:
        print("\nqiskit-aer is required for Qiskit 2.x. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "qiskit-aer"])
            print("✓ qiskit-aer installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install qiskit-aer: {e}")
            print("  Note: Some features may not work without qiskit-aer")
    
    # Test functionality
    print("\nTesting Qiskit functionality...")
    if not check_qiskit_functionality():
        print("\n✗ Qiskit installation may be incomplete")
        return 1
    
    print("\n" + "=" * 60)
    print("✓ Qiskit is installed and working correctly")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())

