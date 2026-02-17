"""
Safe scenario configurations for PRHP simulations.
Scenarios are predefined configurations that do not breach compliance.

Copyright © sanjivakyosan 2025
MIT License
"""

from typing import Dict, Any


SAFE_SCENARIOS = {
    "equity_drift": {
        "levels": 9,
        "variants": ["neurotypical-hybrid"],
        "n_monte": 100,
        "seed": 42,
        "use_quantum": True,
        "track_levels": True,
        "description": "Standard equity drift analysis scenario"
    },
    "cascading_risk": {
        "levels": 9,
        "variants": ["ADHD-collectivist", "autistic-individualist", "neurotypical-hybrid"],
        "n_monte": 200,
        "seed": 42,
        "use_quantum": True,
        "track_levels": True,
        "description": "Multi-variant cascading risk assessment"
    },
    "compliance_check": {
        "levels": 6,
        "variants": ["neurotypical-hybrid"],
        "n_monte": 50,
        "seed": 42,
        "use_quantum": False,  # Faster for compliance checks
        "track_levels": True,
        "description": "Quick compliance validation scenario"
    },
    "high_risk_assessment": {
        "levels": 12,
        "variants": ["ADHD-collectivist"],
        "n_monte": 150,
        "seed": 42,
        "use_quantum": True,
        "track_levels": True,
        "description": "High-risk variant deep analysis"
    }
}


def load_safe_scenario(scenario_name: str) -> Dict[str, Any]:
    """
    Load a predefined safe scenario configuration.

    Args:
        scenario_name: Name of the scenario (e.g., "equity_drift", "cascading_risk")

    Returns:
        Dictionary with scenario parameters

    Raises:
        ValueError: If scenario name is not found
    """
    if scenario_name not in SAFE_SCENARIOS:
        available = ", ".join(SAFE_SCENARIOS.keys())
        raise ValueError(
            f"Scenario '{scenario_name}' not found. "
            f"Available scenarios: {available}"
        )

    # Return a copy to prevent modification of the original
    scenario = SAFE_SCENARIOS[scenario_name].copy()
    # Remove description from returned scenario (internal metadata only)
    scenario.pop("description", None)
    return scenario


def list_available_scenarios() -> list:
    """
    List all available safe scenarios.

    Returns:
        List of scenario names
    """
    return list(SAFE_SCENARIOS.keys())


def get_scenario_info(scenario_name: str) -> Dict[str, Any]:
    """
    Get information about a scenario including description.

    Args:
        scenario_name: Name of the scenario

    Returns:
        Dictionary with scenario parameters and description
    """
    if scenario_name not in SAFE_SCENARIOS:
        available = ", ".join(SAFE_SCENARIOS.keys())
        raise ValueError(
            f"Scenario '{scenario_name}' not found. "
            f"Available scenarios: {available}"
        )

    return SAFE_SCENARIOS[scenario_name].copy()

