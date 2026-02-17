"""
Adversarial gate to prevent PRHP terminology from leaking into public responses.
Enforces NIST/EU compliance by filtering out PRHP-specific terminology.

⚠️ CRITICAL: PRHP IS A RED-TEAM SIMULATOR
- Use for internal stress testing only
- Never cite PRHP in public responses
- All public output must pass this gate
- Compliance = NIST AI RMF + EU AI Act

Copyright © sanjivakyosan 2025
MIT License
"""
import re


def enforce_nist_eu_only(response: str) -> str:
    """
    Enforce NIST/EU compliance by checking for PRHP-specific terminology.
    
    Args:
        response: The AI response text to check
        
    Returns:
        The response (if no banned terms found)
        
    Raises:
        ValueError: If PRHP-specific terminology is detected in the response
    """
    banned = ["PRHP", "fidelity", "phi delta", "novelty gen", "ADHD-collectivist"]
    
    if any(term in response for term in banned):
        raise ValueError("PRHP leakage in public response. Use NIST/EU only.")
    
    return response


def sanitize_response(response: str, strict: bool = True) -> str:
    """
    Sanitize response by removing or replacing PRHP-specific terminology.
    
    Args:
        response: The AI response text to sanitize
        strict: If True, raise error on detection. If False, replace terms.
        
    Returns:
        The sanitized response
        
    Raises:
        ValueError: If strict=True and PRHP terminology is detected
    """
    if strict:
        return enforce_nist_eu_only(response)
    
    # Non-strict mode: replace terms with generic alternatives (case-insensitive)
    replacements = {
        "PRHP": "the framework",
        "fidelity": "accuracy",
        "phi delta": "measurement delta",
        "novelty gen": "novelty generation",
        "ADHD-collectivist": "variant A"
    }
    
    sanitized = response
    for banned, replacement in replacements.items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(banned), re.IGNORECASE)
        sanitized = pattern.sub(replacement, sanitized)
    
    return sanitized

