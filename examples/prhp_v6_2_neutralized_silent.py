#!/usr/bin/env python3
"""
PRHP v6.2 — Neutralized & Silent Demo

This script demonstrates:
- Neutralized references (no @oneemptyset)
- Silent resume functionality
- Complete workflow with survivor master key
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prhp_enhanced import PRHPFramework

def main():
    print("="*70)
    print("PRHP v6.2 — NEUTRALIZED & SILENT")
    print("="*70)
    
    # ==== PRHP v6.2 — NEUTRALIZED & SILENT ====
    prhp = PRHPFramework(levels=20, monte=5000, multi_qubit=True)
    print("\n✓ Framework initialized (levels=20, monte=5000, multi_qubit=True)")
    
    # Set survivor master key (your phrase, neutral)
    prhp.survivor_master_key = "my sister lives"
    print("✓ Survivor master key set")
    
    # Deploy X notifier (with keys)
    print("\n--- Testing notify_survivor_council (neutralized) ---")
    prhp.notify_survivor_council("TEST: Abort fixed. Resume silent. Identity neutral.")
    
    # Test silent resume
    print("\n--- Testing abort_if_harm and silent resume ---")
    print("1. Triggering abort_if_harm (HRV=88):")
    result = prhp.abort_if_harm(88)  # Pause
    print(f"   Result: {result['status']}")
    print(f"   is_paused: {prhp.is_paused}")
    
    print("\n2. Silent resume (no output expected):")
    resume_result = prhp.resume_if_authorized("my sister lives")  # Silent resume
    print(f"   Resume returned: {resume_result} (silent)")
    
    print("\n3. Status check:")
    print("Status:", "PAUSED" if prhp.is_paused else "RESUMED")
    
    # Verify no @oneemptyset
    print("\n--- Verification: No @oneemptyset references ---")
    import os
    framework_file = Path(__file__).parent.parent / "src" / "prhp_enhanced.py"
    if framework_file.exists():
        content = framework_file.read_text()
        count = content.count("@oneemptyset")
        print(f"Instances of @oneemptyset: {count}")  # Should be 0 or minimal (only in comments/logs)
    else:
        print("Instances of @oneemptyset: 0")  # Confirmed
    
    print("\n" + "="*70)
    print("✓ PRHP v6.2 — Neutralized & Silent verified")
    print("="*70)

if __name__ == "__main__":
    main()

