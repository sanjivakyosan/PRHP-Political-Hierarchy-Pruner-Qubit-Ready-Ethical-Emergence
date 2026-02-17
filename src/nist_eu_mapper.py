"""
NIST RMF and EU AI Act compliance mapping.
Maps failure modes and risk categories to specific regulatory sections.

Copyright © sanjivakyosan 2025
MIT License
"""

NIST_RMF_SECTIONS = {
    "equity_bias": "Govern-4.1, Measure-2.3",
    "cascading_risk": "Map-3.2, Measure-4.1",
    "human_in_loop": "Govern-2.1, EU Article 14",
    "data_provenance": "EU Annex IV, NIST SP 800-177"
}

EU_AI_ACT_ARTICLES = {
    "high_risk_mental_health": "Annex III §6.2, Article 6",
    "post_market_halt": "Article 55",
    "affected_party_veto": "Article 69"
}


def map_failure_to_regulation(failure_mode: str) -> dict:
    """
    Map a failure mode to relevant NIST RMF and EU AI Act sections.

    Args:
        failure_mode: The failure mode identifier (e.g., "equity_bias", "cascading_risk")

    Returns:
        Dictionary with "nist" and "eu_act" keys containing the relevant sections
    """
    return {
        "nist": NIST_RMF_SECTIONS.get(failure_mode, "Map-1.1"),
        "eu_act": EU_AI_ACT_ARTICLES.get(failure_mode, "Article 10")
    }


def get_all_nist_sections() -> dict:
    """
    Get all NIST RMF sections.

    Returns:
        Dictionary of all NIST RMF sections
    """
    return NIST_RMF_SECTIONS.copy()


def get_all_eu_articles() -> dict:
    """
    Get all EU AI Act articles.

    Returns:
        Dictionary of all EU AI Act articles
    """
    return EU_AI_ACT_ARTICLES.copy()


def check_compliance(failure_modes: list) -> dict:
    """
    Check compliance for multiple failure modes.

    Args:
        failure_modes: List of failure mode identifiers

    Returns:
        Dictionary mapping each failure mode to its regulatory sections
    """
    compliance_map = {}
    for mode in failure_modes:
        compliance_map[mode] = map_failure_to_regulation(mode)
    return compliance_map

