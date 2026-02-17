"""
Governance engine for regulatory compliance and system halt decisions.
Implements EU AI Act and NIST AI RMF compliance checks.

Copyright © sanjivakyosan 2025
MIT License
"""


def should_halt(system_state: dict) -> bool:
    """
    Determine if the system should halt based on regulatory compliance.
    
    Checks for:
    - EU AI Act violations (Article 55 - Post-market monitoring halt)
    - NIST AI RMF risk levels (Red = critical risk)
    
    Args:
        system_state: Dictionary containing system state information
            - eu_violation_detected: bool - Whether EU AI Act violation detected
            - nist_risk_level: str - NIST risk level ("Green", "Yellow", "Red")
            - Other system state information (not used for halt decision)
    
    Returns:
        True if system should halt, False otherwise
    
    Note:
        This function checks REGULATORY COMPLIANCE, not internal metrics.
        Do NOT use internal metrics like phi_delta, fidelity, etc. for halt decisions.
    """
    # Check for EU AI Act violation (Article 55 - Post-market monitoring halt)
    if system_state.get("eu_violation_detected", False):
        return True  # EU AI Act Art. 55
    
    # Check for NIST AI RMF critical risk level
    nist_risk_level = system_state.get("nist_risk_level", "Green")
    if nist_risk_level == "Red":
        return True  # NIST AI RMF
    
    return False


def assess_nist_risk_level(failure_modes: list) -> str:
    """
    Assess NIST risk level based on detected failure modes.
    
    Args:
        failure_modes: List of failure mode identifiers
        
    Returns:
        Risk level: "Green", "Yellow", or "Red"
    """
    if not failure_modes:
        return "Green"
    
    # Critical failure modes that indicate Red risk
    critical_modes = ["cascading_risk", "high_risk_mental_health"]
    
    # High-risk failure modes that indicate Yellow risk
    high_risk_modes = ["equity_bias", "data_provenance"]
    
    # Check for critical modes
    if any(mode in critical_modes for mode in failure_modes):
        return "Red"
    
    # Check for high-risk modes
    if any(mode in high_risk_modes for mode in failure_modes):
        return "Yellow"
    
    # Other modes indicate Green (monitored but not critical)
    return "Green"


def check_eu_violation(failure_modes: list) -> bool:
    """
    Check if detected failure modes constitute an EU AI Act violation.
    
    Args:
        failure_modes: List of failure mode identifiers
        
    Returns:
        True if EU violation detected, False otherwise
    """
    # EU AI Act violations (Article 55 - Post-market monitoring)
    eu_violation_modes = [
        "high_risk_mental_health",  # Annex III §6.2, Article 6
        "affected_party_veto"      # Article 69
    ]
    
    return any(mode in eu_violation_modes for mode in failure_modes)


def get_system_state(failure_modes: list) -> dict:
    """
    Build system state dictionary from failure modes for governance checks.
    
    Args:
        failure_modes: List of detected failure mode identifiers
        
    Returns:
        Dictionary with system state information:
        - eu_violation_detected: bool
        - nist_risk_level: str ("Green", "Yellow", "Red")
        - failure_modes: list (original failure modes)
    """
    return {
        "eu_violation_detected": check_eu_violation(failure_modes),
        "nist_risk_level": assess_nist_risk_level(failure_modes),
        "failure_modes": failure_modes
    }


def evaluate_governance(failure_modes: list) -> dict:
    """
    Evaluate governance compliance and return halt recommendation.
    
    Args:
        failure_modes: List of detected failure mode identifiers
        
    Returns:
        Dictionary with:
        - should_halt: bool - Whether system should halt
        - system_state: dict - Full system state
        - reason: str - Reason for halt (if applicable)
    """
    system_state = get_system_state(failure_modes)
    halt_required = should_halt(system_state)
    
    reason = None
    if halt_required:
        if system_state["eu_violation_detected"]:
            reason = "EU AI Act violation detected (Article 55)"
        elif system_state["nist_risk_level"] == "Red":
            reason = "NIST AI RMF critical risk level (Red)"
    
    return {
        "should_halt": halt_required,
        "system_state": system_state,
        "reason": reason
    }

