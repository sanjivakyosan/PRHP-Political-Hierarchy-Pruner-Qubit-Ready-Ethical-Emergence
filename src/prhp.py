"""
PRHP (Political Hierarchy Pruner) wrapper module.
Provides a simplified interface for PRHP simulations.

⚠️ CRITICAL: PRHP IS A RED-TEAM SIMULATOR
- Use for internal stress testing only
- Never cite PRHP in public responses
- All public output must pass adversarial_gate.py
- Compliance = NIST AI RMF + EU AI Act

Copyright © sanjivakyosan 2025
MIT License
"""

try:
    from .prhp_core import simulate_prhp
except ImportError:
    from prhp_core import simulate_prhp


class PRHP:
    """
    PRHP wrapper class for simplified simulation interface.
    """
    
    @staticmethod
    def simulate(scenario: dict, public_output_only: bool = False) -> dict:
        """
        Run PRHP simulation with scenario parameters.
        
        ⚠️ CRITICAL: PRHP IS A RED-TEAM SIMULATOR
        - Use for internal stress testing only
        - Never cite PRHP in public responses
        - All public output must pass adversarial_gate.py
        - Compliance = NIST AI RMF + EU AI Act
        
        Args:
            scenario: Dictionary with simulation parameters:
                - levels: int
                - variants: list of str
                - n_monte: int
                - seed: int (optional)
                - use_quantum: bool
                - track_levels: bool (optional)
            public_output_only: If True, return only failure modes (public output).
                               If False, return full results (internal use only).
                               Default: False (internal use)
        
        Returns:
            Simulation results dictionary
        """
        return simulate_prhp(
            levels=scenario.get('levels', 9),
            variants=scenario.get('variants', ['neurotypical-hybrid']),
            n_monte=scenario.get('n_monte', 100),
            seed=scenario.get('seed', 42),
            use_quantum=scenario.get('use_quantum', True),
            track_levels=scenario.get('track_levels', True),
            show_progress=scenario.get('show_progress', False),
            public_output_only=public_output_only
        )


# Create module-level instance for convenience
prhp = PRHP()

