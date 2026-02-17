#!/usr/bin/env python3
"""
Simple PRHP Framework Usage - Matches User's Example Code Style

This script demonstrates the exact usage pattern from the user's example code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prhp_enhanced import PRHPFramework  # Using alias for convenience

# === PASTE THIS IN CELL 2 ===

prhp = PRHPFramework(levels=18, monte=2000, multi_qubit=True)

# 1. X sentiment
prhp.add_live_x_sentiment()

# 2. WHO stressor
prhp.update_stressors_from_who()

# 3. zk-EAS
proof, _ = prhp.zk_eas_proof({"eas": 42})

# 4. IPFS dashboard
cid = prhp.publish_kpi_dashboard()

# 6. Quadratic voting
survivor_tokens = [1000000, 4, 9, 100]
weights = [prhp.quadratic_vote_weight(t) for t in survivor_tokens]

# 7. Whisper consent
consent = prhp.voice_consent(b"audio", "es")

# RUN
prhp.run_simulation()

print("\n" + "="*70)
print("SIMULATION COMPLETE")
print("="*70)
print(f"zk-Proof: {proof}")
print(f"IPFS CID: {cid}")
print(f"Quadratic Weights: {weights}")
print(f"Whisper Consent: {consent}")

