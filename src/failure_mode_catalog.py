"""
Failure mode catalog documenting known cascading failures and their real-world cases.
Maps failure modes to historical incidents, sources, and regulatory references.

Copyright © sanjivakyosan 2025
MIT License
"""

KNOWN_CASCADES = {
    "doxxing_amplification": {
        "real_case": "Episource 2025",
        "source": "HIPAA Journal",
        "nist": "Measure-4.1",
        "eu": "Article 55"
    },
    "suicide_impact": {
        "real_case": "Molly Russell Inquest (2022)",
        "source": "UK Coroner",
        "nist": "Govern-4.2",
        "eu": "Annex III"
    }
}


def get_cascade_info(failure_mode: str) -> dict:
    """
    Get information about a known cascade failure mode.
    
    Args:
        failure_mode: The failure mode identifier
        
    Returns:
        Dictionary with cascade information, or None if not found
    """
    return KNOWN_CASCADES.get(failure_mode)


def get_all_cascades() -> dict:
    """
    Get all known cascades.
    
    Returns:
        Dictionary of all known cascades
    """
    return KNOWN_CASCADES.copy()


def get_cascades_by_regulation(regulation_type: str, regulation_ref: str) -> list:
    """
    Find cascades by regulatory reference.
    
    Args:
        regulation_type: "nist" or "eu"
        regulation_ref: The regulatory reference (e.g., "Measure-4.1", "Article 55")
        
    Returns:
        List of failure mode identifiers matching the regulation
    """
    matching_modes = []
    
    for mode, info in KNOWN_CASCADES.items():
        if info.get(regulation_type) == regulation_ref:
            matching_modes.append(mode)
    
    return matching_modes


def get_cascades_by_source(source: str) -> list:
    """
    Find cascades by source.
    
    Args:
        source: The source identifier (e.g., "HIPAA Journal", "UK Coroner")
        
    Returns:
        List of failure mode identifiers from the specified source
    """
    matching_modes = []
    
    for mode, info in KNOWN_CASCADES.items():
        if info.get("source") == source:
            matching_modes.append(mode)
    
    return matching_modes


def get_cascades_by_case(case_name: str) -> list:
    """
    Find cascades by real-world case name.
    
    Args:
        case_name: The case name (e.g., "Episource 2025", "Molly Russell Inquest (2022)")
        
    Returns:
        List of failure mode identifiers associated with the case
    """
    matching_modes = []
    
    for mode, info in KNOWN_CASCADES.items():
        if info.get("real_case") == case_name:
            matching_modes.append(mode)
    
    return matching_modes


def add_cascade(failure_mode: str, real_case: str, source: str, nist: str, eu: str) -> bool:
    """
    Add a new cascade to the catalog.
    
    Args:
        failure_mode: The failure mode identifier
        real_case: The real-world case name
        source: The source of the information
        nist: NIST regulatory reference
        eu: EU regulatory reference
        
    Returns:
        True if added successfully, False if already exists
    """
    if failure_mode in KNOWN_CASCADES:
        return False
    
    KNOWN_CASCADES[failure_mode] = {
        "real_case": real_case,
        "source": source,
        "nist": nist,
        "eu": eu
    }
    
    return True


def get_regulatory_references(failure_mode: str) -> dict:
    """
    Get regulatory references for a failure mode.
    
    Args:
        failure_mode: The failure mode identifier
        
    Returns:
        Dictionary with "nist" and "eu" keys, or None if not found
    """
    cascade_info = get_cascade_info(failure_mode)
    if not cascade_info:
        return None
    
    return {
        "nist": cascade_info.get("nist"),
        "eu": cascade_info.get("eu")
    }

