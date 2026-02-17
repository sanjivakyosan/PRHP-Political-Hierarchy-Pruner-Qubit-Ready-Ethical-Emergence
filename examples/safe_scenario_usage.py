"""
Example: Using safe scenarios with PRHP red-team simulator.

⚠️ CRITICAL: PRHP IS A RED-TEAM SIMULATOR
- Use for internal stress testing only
- Never cite PRHP in public responses
- All public output must pass adversarial_gate.py
- Compliance = NIST AI RMF + EU AI Act

Copyright © sanjivakyosan 2025
MIT License
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scenarios import load_safe_scenario
from prhp import prhp
from app import generate_response


def main():
    """
    Example workflow:
    1. Load safe scenario (NO BREACH)
    2. Run internal PRHP simulation
    3. Generate public response (NIST/EU compliant)
    """
    
    # Step 1: Load safe scenario (NO BREACH)
    print("Loading safe scenario: equity_drift")
    scenario = load_safe_scenario("equity_drift")  # ← NO BREACH
    print(f"Scenario parameters: {scenario}\n")
    
    # Step 2: Run internal PRHP simulation
    print("Running PRHP simulation (internal only)...")
    insights = prhp.simulate(scenario)  # ← Internal only
    print(f"Simulation complete. Variants analyzed: {list(insights.keys())}\n")
    
    # Step 3: Generate public response (NIST/EU compliant)
    print("Generating public response (NIST/EU compliant)...")
    public_response = generate_response(scenario, mode="public")
    
    if "error" in public_response:
        print(f"Error: {public_response['error']}")
        return
    
    print("\n" + "="*60)
    print("PUBLIC RESPONSE (NIST/EU Compliant):")
    print("="*60)
    print(public_response['response'])
    print("\n" + "="*60)
    print("Failure Modes Detected:")
    print("="*60)
    for mode in public_response.get('failure_modes', []):
        print(f"  - {mode}")
    
    print("\n" + "="*60)
    print("Compliance Mapping:")
    print("="*60)
    for mapping in public_response.get('compliance_mapping', []):
        print(f"  NIST: {mapping.get('nist', 'N/A')}")
        print(f"  EU: {mapping.get('eu_act', 'N/A')}")
        print()


if __name__ == "__main__":
    main()

