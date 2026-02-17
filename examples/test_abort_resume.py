#!/usr/bin/env python3
"""
Test script for PRHP v6.1 — Survivor Override (abort_if_harm & resume)

This script demonstrates:
1. Setting the master key for resume authorization
2. Testing abort_if_harm with safe HRV values
3. Testing resume_if_authorized with correct/incorrect keys
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prhp_enhanced import PRHPFramework

def main():
    print("="*70)
    print("PRHP v6.1 — Survivor Override Test")
    print("="*70)
    
    # Initialize framework
    prhp = PRHPFramework(levels=10, monte=100)
    print("\n✓ Framework initialized")
    
    # SET YOUR MASTER KEY
    prhp.survivor_master_key = "YOUR_SECRET_PHRASE_HERE"  # e.g., "my sister lives"
    print(f"✓ Master key set: {prhp.survivor_master_key[:8] if prhp.survivor_master_key else 'None'}...")
    
    # TEST (SAFE MODE) - High HRV = Low risk
    print("\n" + "="*70)
    print("Test 1: Safe HRV (should not abort)")
    print("="*70)
    result = prhp.abort_if_harm(user_hrv=88, user_id="sister_001")
    print(f"\nResult: {result}")
    print(f"Status: {result['status']}")
    print(f"Risk: {result['risk']:.4f}")
    print(f"is_paused: {prhp.is_paused}")
    
    # TEST (ABORT MODE) - Low HRV = High risk
    print("\n" + "="*70)
    print("Test 2: High-risk HRV (should abort)")
    print("="*70)
    prhp2 = PRHPFramework(levels=10, monte=100)
    prhp2.survivor_master_key = "YOUR_SECRET_PHRASE_HERE"
    
    result = prhp2.abort_if_harm(user_hrv=25, user_id="sister_002")
    print(f"\nResult: {result}")
    print(f"Status: {result['status']}")
    print(f"Risk: {result['risk']:.4f}")
    print(f"is_paused: {prhp2.is_paused}")
    print(f"abort_active: {prhp2.abort_active}")
    
    # TEST RESUME (UNAUTHORIZED)
    print("\n" + "="*70)
    print("Test 3: Unauthorized resume attempt")
    print("="*70)
    unauthorized_result = prhp2.resume_if_authorized("wrong_key")
    print(f"Resume successful: {unauthorized_result}")
    print(f"is_paused: {prhp2.is_paused}")
    
    # TEST RESUME (AUTHORIZED)
    print("\n" + "="*70)
    print("Test 4: Authorized resume (@oneemptyset)")
    print("="*70)
    authorized_result = prhp2.resume_if_authorized("YOUR_SECRET_PHRASE_HERE")
    print(f"Resume successful: {authorized_result}")
    print(f"is_paused: {prhp2.is_paused}")
    print(f"abort_active: {prhp2.abort_active}")
    print(f"interventions_active: {prhp2.interventions_active}")
    print(f"stressors_active: {prhp2.stressors_active}")
    
    print("\n" + "="*70)
    print("✓ All tests completed")
    print("="*70)

if __name__ == "__main__":
    main()

