"""
PRHP: Political Hierarchy Pruner with Qubit-Ready Ethical Emergence

A framework for simulating political hierarchy pruning using quantum game theory
and Integrated Information Theory (IIT) phi measures.

Copyright © sanjivakyosan 2025
"""
# Check Qiskit availability on import (optional, won't fail if missing)
try:
    from .qiskit_check import check_qiskit_installation
    _qiskit_installed, _qiskit_version, _qiskit_error = check_qiskit_installation()
    if not _qiskit_installed:
        import warnings
        warnings.warn(
            f"Qiskit is not installed. Quantum features will use classical approximations.\n"
            f"To install: pip install qiskit qiskit-aer\n"
            f"Or run: python3 scripts/ensure_qiskit.py",
            ImportWarning
        )
except ImportError:
    # qiskit_check module not available, skip check
    pass

from .prhp_core import simulate_prhp, dopamine_hierarchy
from .qubit_hooks import (
    compute_phi, compute_phi_delta, entangle_nodes_variant,
    inject_phase_flip, recalibrate_novelty, compute_novelty_entropy
)
from .political_pruner import (
    apply_firewall, simulate_pruner_levels, threshold_qubit,
    phi_oracle, create_hierarchy_hamiltonian
)
from .virus_extinction import (
    simulate_viral_cascade, forecast_extinction_risk
)
from .meta_empirical import (
    bayesian_update_novelty, meta_empirical_validation,
    full_meta_empirical_loop, synthesize_astrobiology_ethics
)

__version__ = "1.0.0"
__all__ = [
    'simulate_prhp',
    'dopamine_hierarchy',
    'compute_phi',
    'compute_phi_delta',
    'entangle_nodes_variant',
    'inject_phase_flip',
    'recalibrate_novelty',
    'compute_novelty_entropy',
    'apply_firewall',
    'simulate_pruner_levels',
    'threshold_qubit',
    'phi_oracle',
    'create_hierarchy_hamiltonian',
    'simulate_viral_cascade',
    'forecast_extinction_risk',
    'bayesian_update_novelty',
    'meta_empirical_validation',
    'full_meta_empirical_loop',
    'synthesize_astrobiology_ethics'
]

