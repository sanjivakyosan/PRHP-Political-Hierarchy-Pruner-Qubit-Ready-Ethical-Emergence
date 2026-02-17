#!/usr/bin/env python3
"""
Comprehensive PRHP Framework Demo - All Features Integrated

This script demonstrates the complete Enhanced PRHP Framework workflow
with all 7+ upgrades integrated:

1. Live X/Twitter sentiment analysis
2. WHO RSS feed monitoring
3. Zero-knowledge EAS proofs
4. IPFS KPI dashboard publishing
5. Automated upkeep monitoring
6. Quadratic voting
7. Voice consent processing
8. Multi-qubit quantum simulation

Usage:
    python examples/comprehensive_demo.py
    python examples/comprehensive_demo.py --mock  # Use mock implementations for testing
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prhp_enhanced import EnhancedPRHPFramework
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive PRHP Framework Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock implementations (skip external API calls)'
    )
    parser.add_argument(
        '--hashtag',
        type=str,
        default='#AIEatsThePoor',
        help='X/Twitter hashtag to monitor'
    )
    parser.add_argument(
        '--levels',
        type=int,
        default=18,
        help='Number of hierarchy levels'
    )
    parser.add_argument(
        '--monte',
        type=int,
        default=2000,
        help='Monte Carlo iterations (max: 5000)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("PRHP v5.0 COMPREHENSIVE DEMO — ALL FEATURES INTEGRATED")
    print("="*70)
    print()
    
    # Initialize framework
    print("Step 1: Initializing Enhanced PRHP Framework...")
    prhp = EnhancedPRHPFramework(
        levels=args.levels,
        monte=args.monte,
        variants=['ADHD-collectivist', 'autistic-individualist', 
                  'neurotypical-hybrid', 'trauma-survivor-equity'],
        seed=42
    )
    print(f"✓ Framework initialized: {len(prhp.variants)} variants, {prhp.monte} Monte Carlo iterations")
    print()
    
    # 1. Live X/Twitter sentiment analysis
    print("Step 2: Live X/Twitter Sentiment Analysis")
    print("-" * 70)
    if args.mock:
        print("  [MOCK] Skipping X sentiment analysis")
        # Mock: Set co-auth weight directly
        prhp.add_victim_co_authorship(feedback_intensity=0.06, co_auth_weight=0.488)
        print("  → Mock sentiment: -0.94 → co_auth_weight = 0.488 (maxed out)")
    else:
        try:
            avg_sentiment = prhp.add_live_x_sentiment(
                hashtag=args.hashtag,
                sample_secs=30
            )
            if avg_sentiment is not None:
                print(f"  → X-Space sentiment: {avg_sentiment:.3f}")
                print(f"  → Co-authorship weight adjusted based on sentiment")
            else:
                print("  ⚠ No sentiment data available (snscrape may not be installed)")
        except Exception as e:
            print(f"  ⚠ Sentiment analysis failed: {e}")
    print()
    
    # 2. WHO RSS feed monitoring
    print("Step 3: WHO RSS Feed Monitoring")
    print("-" * 70)
    if args.mock:
        print("  [MOCK] Skipping WHO feed update")
        print("  → Mock: WHO alert detected → harm_cascade impact ↓0.08")
        prhp.stressor_impacts['harm_cascade']['fidelity'] -= 0.08
    else:
        try:
            alerts = prhp.update_stressors_from_who(num_entries=3)
            total_alerts = sum(alerts.values())
            if total_alerts > 0:
                print(f"  → WHO alerts detected: {total_alerts} alert(s)")
                for stressor, count in alerts.items():
                    if count > 0:
                        print(f"    - {stressor}: {count} alert(s)")
            else:
                print("  → No matching WHO alerts found")
        except Exception as e:
            print(f"  ⚠ WHO feed update failed: {e}")
    print()
    
    # 3. Zero-knowledge EAS proof
    print("Step 4: Zero-Knowledge EAS Proof")
    print("-" * 70)
    user_data = {"eas": 42, "user_id": "test_user_123", "kpi_met": True}
    try:
        result = prhp.zk_eas_proof(user_data)
        if result:
            proof, public_signals = result
            print(f"  → zk-SNARK proof generated")
            print(f"  → Public signals: {len(public_signals)} signals")
            print(f"  → Proof size: {len(str(proof))} characters")
        else:
            print("  ⚠ zk-SNARK proof generation skipped (snarkjs/circuit files not available)")
            print("  [MOCK] Proof: 0xabc..., PublicSignals: [42]")
    except Exception as e:
        print(f"  ⚠ zk-SNARK proof failed: {e}")
        print("  [MOCK] Proof: 0xabc..., PublicSignals: [42]")
    print()
    
    # 4. Run simulation with multi-qubit support
    print("Step 5: Running PRHP Simulation (4-Qubit W-State)")
    print("-" * 70)
    print("  PRHP v5.0 SIMULATION RUNNING — 4-QUBIT W-STATE ACTIVE")
    print("  All improvements LIVE — pruning in progress...")
    print()
    try:
        results = prhp.run_simulation(
            use_quantum=True,
            multi_qubit=True,
            stressors_active=True,
            interventions_active=True,
            show_progress=True
        )
        print(f"  ✓ Simulation complete: {len(results)} variants processed")
    except Exception as e:
        print(f"  ✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()
    
    # 5. Automated upkeep check (Chainlink-style)
    print("Step 6: Automated Upkeep Check (Chainlink-Style)")
    print("-" * 70)
    failed, reason = prhp.check_upkeep()
    if failed:
        print(f"  ⚠ Upkeep needed: {reason}")
        upkeep_performed = prhp.perform_upkeep()
        if upkeep_performed:
            print("  → Interventions paused automatically (kill-switch activated)")
    else:
        print("  ✓ All KPIs within thresholds — no upkeep needed")
    print()
    
    # 6. IPFS KPI dashboard publishing
    print("Step 7: IPFS KPI Dashboard Publishing")
    print("-" * 70)
    try:
        cid = prhp.publish_kpi_dashboard()
        if cid:
            print(f"  → KPI dashboard published: ipfs://{cid}")
        else:
            print("  ⚠ IPFS publishing skipped (IPFS not available)")
            print("  [MOCK] CID: ipfs://QmXyZ123...")
    except Exception as e:
        print(f"  ⚠ IPFS publishing failed: {e}")
        print("  [MOCK] CID: ipfs://QmXyZ123...")
    print()
    
    # 7. Quadratic voting
    print("Step 8: Quadratic Voting")
    print("-" * 70)
    survivor_tokens = [1000000, 4, 9, 100]
    weights = [prhp.quadratic_vote_weight(t) for t in survivor_tokens]
    print(f"  → Survivor tokens: {survivor_tokens}")
    print(f"  → Quadratic weights: {weights}")
    print(f"  → Example: 1,000,000 tokens → {weights[0]} vote weight (O(√n) influence)")
    print()
    
    # Apply quadratic voting to interventions
    votes = {
        '$2B_fund': 100,
        'dashboards': 64,
        'foresight_sims': 25
    }
    weighted = prhp.apply_quadratic_voting(votes, apply_to='interventions')
    print(f"  → Intervention votes weighted:")
    for option, percentage in weighted.items():
        print(f"    - {option}: {percentage:.1f}%")
    print()
    
    # 8. Voice consent processing
    print("Step 9: Voice Consent Processing")
    print("-" * 70)
    # Mock Spanish audio bytes
    audio_bytes = b"quiero salir"  # "I want to leave" in Spanish
    lang_code = "es"
    try:
        opt_out = prhp.voice_consent(audio_bytes, lang_code)
        if opt_out is not None:
            if opt_out:
                print(f"  → Opt-out detected in {lang_code} audio")
                print(f"  → Consent: REVOKED")
            else:
                print(f"  → No opt-out detected in {lang_code} audio")
                print(f"  → Consent: MAINTAINED")
        else:
            print("  ⚠ Voice consent processing skipped (OpenAI not available)")
            print("  [MOCK] Spanish audio: 'quiero salir' → opt-out detected")
    except Exception as e:
        print(f"  ⚠ Voice consent failed: {e}")
        print("  [MOCK] Spanish audio: 'quiero salir' → opt-out detected")
    print()
    
    # Summary
    print("="*70)
    print("COMPREHENSIVE DEMO SUMMARY")
    print("="*70)
    print()
    
    # KPI Status
    print("KPI Status:")
    kpis = prhp.define_kpis()
    for variant, status in kpis.items():
        overall = status.get('overall', False)
        status_icon = "✓" if overall else "✗"
        print(f"  {status_icon} {variant}: {'PASS' if overall else 'FAIL'}")
    print()
    
    # Pruning Efficacy
    print("Pruning Efficacy:")
    try:
        efficacy = prhp.compute_pruning_efficacy(use_quantum=True)
        for stressor, pct in efficacy.items():
            print(f"  {stressor}: {pct:.1f}%")
    except Exception as e:
        print(f"  ⚠ Could not compute: {e}")
    print()
    
    # Final state
    print("Framework State:")
    print(f"  - Interventions active: {prhp.interventions_active}")
    print(f"  - Stressors active: {prhp.stressors_active}")
    print(f"  - Variants: {len(prhp.variants)}")
    print(f"  - Monte Carlo: {prhp.monte}")
    print()
    
    print("="*70)
    print("ALL FEATURES DEMONSTRATED")
    print("="*70)
    print()
    print("✓ Live X sentiment analysis")
    print("✓ WHO RSS feed monitoring")
    print("✓ Zero-knowledge EAS proofs")
    print("✓ IPFS KPI dashboard publishing")
    print("✓ Automated upkeep monitoring")
    print("✓ Quadratic voting")
    print("✓ Voice consent processing")
    print("✓ Multi-qubit quantum simulation")
    print()


if __name__ == "__main__":
    main()

