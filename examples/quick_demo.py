#!/usr/bin/env python3
"""
Quick PRHP Framework Demo - Streamlined Execution

A simplified demonstration of all PRHP features in a clean, linear workflow.
Matches the execution style of the user's comprehensive example.

Usage:
    python examples/quick_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prhp_enhanced import EnhancedPRHPFramework

# === INITIALIZATION ===
print("="*70)
print("PRHP v5.0 QUICK DEMO — ALL FEATURES INTEGRATED")
print("="*70)
print()

prhp = EnhancedPRHPFramework(levels=18, monte=2000)

# === ALL 7 UPGRADES ===

# 1. Real-time X sentiment (rage = -0.94)
print("1. Live X/Twitter Sentiment Analysis")
try:
    avg_sent = prhp.add_live_x_sentiment(hashtag="#AIEatsThePoor", sample_secs=30)
    if avg_sent is not None:
        print(f"   → X-Space sentiment {avg_sent:.3f} → co-auth weight adjusted")
    else:
        print("   → [MOCK] Sentiment: -0.94 → co_auth_weight = 0.488 (maxed out)")
        prhp.add_victim_co_authorship(feedback_intensity=0.06, co_auth_weight=0.488)
except Exception as e:
    print(f"   → [MOCK] Sentiment analysis skipped: {e}")
    prhp.add_victim_co_authorship(feedback_intensity=0.06, co_auth_weight=0.488)

# 2. WHO stressor injection (fake alert detected)
print("\n2. WHO RSS Feed Monitoring")
try:
    alerts = prhp.update_stressors_from_who(num_entries=3)
    if sum(alerts.values()) > 0:
        print(f"   → WHO alerts detected: {sum(alerts.values())} alert(s)")
    else:
        print("   → [MOCK] WHO alert → harm_cascade impact ↓0.08 (fake detected)")
        prhp.stressor_impacts['harm_cascade']['fidelity'] -= 0.08
except Exception as e:
    print(f"   → [MOCK] WHO feed update skipped: {e}")

# 3. zk-EAS proof (on-device)
print("\n3. Zero-Knowledge EAS Proof")
user_data = {"eas": 42, "user_id": "test_user_123", "kpi_met": True}
try:
    result = prhp.zk_eas_proof(user_data)
    if result:
        proof, public_signals = result
        print(f"   → zk-SNARK proof generated: {len(str(proof))} chars")
        print(f"   → Public signals: {public_signals}")
    else:
        print("   → [MOCK] Proof: 0xabc..., PublicSignals: [42]")
except Exception as e:
    print(f"   → [MOCK] zk-SNARK proof skipped: {e}")

# 4. IPFS KPI dashboard
print("\n4. IPFS KPI Dashboard Publishing")
try:
    cid = prhp.publish_kpi_dashboard()
    if cid:
        print(f"   → KPI dashboard pinned → ipfs://{cid}")
    else:
        print("   → [MOCK] CID: ipfs://QmXyZ123...")
except Exception as e:
    print(f"   → [MOCK] IPFS publishing skipped: {e}")

# 5. Chainlink kill-switch (automated upkeep)
print("\n5. Automated Upkeep Check (Chainlink-Style)")
failed, reason = prhp.check_upkeep()
if failed:
    print(f"   → ⚠ Upkeep needed: {reason}")
    prhp.perform_upkeep()
    print("   → Interventions paused automatically (kill-switch activated)")
else:
    print("   → ✓ All KPIs within thresholds — no upkeep needed")

# 6. Quadratic voting
print("\n6. Quadratic Voting")
survivor_tokens = [1000000, 4, 9, 100]
weights = [prhp.quadratic_vote_weight(t) for t in survivor_tokens]
print(f"   → Survivor tokens: {survivor_tokens}")
print(f"   → Quadratic weights: {weights}")
print(f"   → Example: 1,000,000 tokens → {weights[0]} vote weight (O(√n) influence)")

# 7. Whisper consent (40+ langs)
print("\n7. Voice Consent Processing")
audio_bytes = b"quiero salir"  # Mock Spanish audio
lang_code = "es"
try:
    opt_out = prhp.voice_consent(audio_bytes, lang_code)
    if opt_out is not None:
        if opt_out:
            print(f"   → Opt-out detected in {lang_code} audio")
            print(f"   → Consent: REVOKED")
        else:
            print(f"   → No opt-out detected in {lang_code} audio")
            print(f"   → Consent: MAINTAINED")
    else:
        print("   → [MOCK] Spanish audio: 'quiero salir' → opt-out detected")
except Exception as e:
    print(f"   → [MOCK] Voice consent skipped: {e}")

# === RUN SIMULATION ===
print("\n" + "="*70)
print("PRHP v5.0 SIMULATION RUNNING — 4-QUBIT W-STATE ACTIVE")
print("All improvements LIVE — pruning in progress...")
print("="*70)
print()

prhp.run_simulation(
    stressors_active=True,
    interventions_active=True,
    multi_qubit=True,
    use_quantum=True,
    show_progress=True
)

# === SUMMARY ===
print("\n" + "="*70)
print("QUICK DEMO SUMMARY")
print("="*70)
print()

# KPI Status
kpis = prhp.define_kpis()
print("KPI Status:")
for variant, status in kpis.items():
    overall = status.get('overall', False)
    status_icon = "✓" if overall else "✗"
    print(f"  {status_icon} {variant}: {'PASS' if overall else 'FAIL'}")

print("\nFramework State:")
print(f"  - Interventions active: {prhp.interventions_active}")
print(f"  - Stressors active: {prhp.stressors_active}")
print(f"  - Variants: {len(prhp.variants)}")
print(f"  - Monte Carlo: {prhp.monte}")

print("\n" + "="*70)
print("ALL FEATURES DEMONSTRATED")
print("="*70)

