"""
Qiskit availability check module.
This module ensures Qiskit is available and provides helpful error messages if not.

Copyright © sanjivakyosan 2025
"""

import sys
from typing import Tuple, Optional

def check_qiskit_installation() -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if Qiskit is installed and working.
    
    Returns:
        Tuple of (is_installed, qiskit_version, error_message)
    """
    try:
        import qiskit
        qiskit_version = qiskit.__version__
        
        # Try to import key components
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
        
        # Check for qiskit-aer (required for Qiskit 2.x)
        try:
            import qiskit_aer
            aer_version = qiskit_aer.__version__
        except ImportError:
            # Check if we're on Qiskit 2.x
            major_version = int(qiskit_version.split('.')[0])
            if major_version >= 2:
                return False, qiskit_version, (
                    f"Qiskit {qiskit_version} is installed, but qiskit-aer is missing. "
                    "Please install it with: pip install qiskit-aer"
                )
            aer_version = None
        
        return True, qiskit_version, None
        
    except ImportError as e:
        return False, None, (
            f"Qiskit is not installed. Please install it with: pip install qiskit qiskit-aer\n"
            f"Original error: {e}"
        )
    except Exception as e:
        return False, None, f"Error checking Qiskit installation: {e}"

def ensure_qiskit_available():
    """
    Ensure Qiskit is available, raise error with helpful message if not.
    
    Raises:
        ImportError: If Qiskit is not available with installation instructions
    """
    is_installed, version, error = check_qiskit_installation()
    
    if not is_installed:
        raise ImportError(
            f"\n{'='*60}\n"
            f"QISKIT NOT AVAILABLE\n"
            f"{'='*60}\n"
            f"{error}\n"
            f"\nTo install Qiskit, run:\n"
            f"  pip install -r requirements.txt\n"
            f"  or\n"
            f"  pip install qiskit qiskit-aer\n"
            f"\nYou can also run the setup script:\n"
            f"  python3 scripts/ensure_qiskit.py\n"
            f"{'='*60}\n"
        )
    
    return version

def get_qiskit_info() -> dict:
    """
    Get information about Qiskit installation.
    
    Returns:
        Dictionary with Qiskit information
    """
    is_installed, version, error = check_qiskit_installation()
    
    info = {
        "installed": is_installed,
        "version": version,
        "error": error
    }
    
    if is_installed:
        try:
            import qiskit_aer
            info["aer_version"] = qiskit_aer.__version__
        except ImportError:
            info["aer_version"] = None
            info["aer_required"] = int(version.split('.')[0]) >= 2 if version else False
    
    return info

if __name__ == "__main__":
    """Run as script to check Qiskit installation."""
    print("=" * 60)
    print("Qiskit Installation Check")
    print("=" * 60)
    print()
    
    is_installed, version, error = check_qiskit_installation()
    
    if is_installed:
        print(f"✓ Qiskit is installed: version {version}")
        try:
            import qiskit_aer
            print(f"✓ qiskit-aer is installed: version {qiskit_aer.__version__}")
        except ImportError:
            major_version = int(version.split('.')[0]) if version else 0
            if major_version >= 2:
                print("✗ qiskit-aer is missing (required for Qiskit 2.x)")
                print("  Install with: pip install qiskit-aer")
                sys.exit(1)
            else:
                print("  Note: qiskit-aer is optional for Qiskit 1.x")
        
        print("\n✓ Qiskit is ready to use")
        sys.exit(0)
    else:
        print(f"✗ {error}")
        print("\nTo install Qiskit, run:")
        print("  pip install -r requirements.txt")
        print("  or")
        print("  python3 scripts/ensure_qiskit.py")
        sys.exit(1)

