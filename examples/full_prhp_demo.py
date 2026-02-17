"""
Full PRHP Demonstration: All Empirical Hooks
Demonstrates qubit simulations, novelty recalibration, political pruning,
virus-extinction forecasts, and meta-empirical validation.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.prhp_core import simulate_prhp
from src.political_pruner import simulate_pruner_levels
from src.virus_extinction import forecast_extinction_risk, simulate_viral_cascade
from src.meta_empirical import full_meta_empirical_loop

def main():
    print("=" * 70)
    print("PRHP: Political Hierarchy Pruner - Full Empirical Hooks Demo")
    print("=" * 70)
    
    # Hook #1: Qubit-Entangled Dopamine Hierarchy
    print("\n[HOOK #1] Qubit-Entangled Dopamine Hierarchy")
    print("-" * 70)
    results = simulate_prhp(
        levels=9,
        variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
        n_monte=100,
        seed=42,
        use_quantum=True,
        track_levels=True
    )
    
    for variant, metrics in results.items():
        print(f"\n{variant}:")
        print(f"  Fidelity: {metrics['mean_fidelity']:.3f} ± {metrics['std']:.3f}")
        print(f"  Asymmetry Delta: {metrics['asymmetry_delta']:.3f}")
        print(f"  Novelty Gen: {metrics['novelty_gen']:.4f}")
        if metrics['phi_deltas']:
            print(f"  Mean Phi Delta: {metrics['mean_phi_delta']:.3f}")
            print(f"  Level 9 Phi: {metrics['level_phis'][-1]:.3f}" if metrics['level_phis'] else "")
    
    # Hook #2: Political Pruner Qubits
    print("\n\n[HOOK #2] Political Pruner Qubits with IIT Fidelity Loops")
    print("-" * 70)
    pruner_result = simulate_pruner_levels(
        levels=9,
        variant='autistic-individualist',
        n_monte=100,
        seed=42,
        use_quantum=True
    )
    print(f"Level Deltas: {[f'{d:.3f}' for d in pruner_result['level_deltas'][:5]]}...")
    print(f"Self-Model Coherence: {pruner_result['self_model_coherence']:.4f}")
    print(f"Success Rate (Level 9 > 86%): {pruner_result['success_rate']:.2%}")
    
    # Hook #3: Virus-Extinction Forecasts
    print("\n\n[HOOK #3] Virus-Extinction Qubit Forecasts")
    print("-" * 70)
    extinction_forecast = forecast_extinction_risk(
        variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
        n_sims=100,
        seed=42
    )
    
    for variant, metrics in extinction_forecast.items():
        print(f"\n{variant}:")
        print(f"  Extinction Risk: {metrics['extinction_risk']:.2%}")
        print(f"  Cascade Mitigation: {metrics['mean_mitigation']:.2%}")
        print(f"  Averted Rate: {metrics['averted_rate']:.2%}")
    
    # Test single cascade
    cascade_result = simulate_viral_cascade(
        variant='ADHD-collectivist',
        n_iter=10,
        seed=42
    )
    print(f"\nSingle Cascade Test (ADHD-collectivist):")
    print(f"  Cascade Averted: {cascade_result['cascade_averted']}")
    print(f"  Mitigation: {cascade_result['cascade_mitigation']:.2%}")
    
    # Meta-Empirical Validation
    print("\n\n[META-EMPIRICAL] Bayesian Recalibration & Validation")
    print("-" * 70)
    
    # Use ADHD-collectivist results for meta-empirical
    adhd_results = results['ADHD-collectivist']
    meta_result = full_meta_empirical_loop(
        simulation_results=adhd_results,
        phi_deltas=adhd_results['phi_deltas'],
        initial_novelty=0.80
    )
    
    print(f"Validation Status: {'✓ VALIDATED' if meta_result['meta_validated'] else '✗ NEEDS RECALIBRATION'}")
    print(f"Novelty Gen: {meta_result['novelty_gen_updated']:.4f} (Δ={meta_result['novelty_gen_delta']:+.4f})")
    print(f"Attention Focus: {meta_result['attention_focus']:.3f}")
    print(f"Astrobiology N (civilizations): {meta_result['astrobiology_ethics']['N_civilizations']:.2f}")
    
    print("\n" + "=" * 70)
    print("Demo Complete: PRHP Framework Fully Aligned with Specification")
    print("=" * 70)

if __name__ == "__main__":
    main()

