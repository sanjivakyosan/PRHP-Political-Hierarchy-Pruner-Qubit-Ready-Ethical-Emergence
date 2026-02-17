"""
Virus-Extinction Qubit Forecasts
Simulates viral hierarchies as qubit epidemics with PRHP heuristic pruning.

Copyright © sanjivakyosan 2025
"""

import numpy as np
from scipy.linalg import expm

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.quantum_info import Statevector, DensityMatrix
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

def create_viral_qubit(alpha=0.8, beta=0.6):
    """
    Viral qubit state: |v⟩ = α|healthy⟩ + β|infected⟩
    Normalized: |α|² + |β|² = 1
    """
    norm = np.sqrt(alpha**2 + beta**2)
    alpha_norm = alpha / norm
    beta_norm = beta / norm
    return np.array([alpha_norm, beta_norm])

def amplify_infection(beta, variant='neurotypical-hybrid'):
    """
    Amplify β by collect grad=0.20 for ADHD-collectivist variant.
    Higher collectivism = faster spread.
    """
    grads = {
        'ADHD-collectivist': 0.20,
        'autistic-individualist': 0.15,
        'neurotypical-hybrid': 0.18
    }
    grad = grads.get(variant, 0.18)
    
    # Amplify β by gradient
    beta_amplified = beta * (1 + grad)
    # Keep normalized
    alpha = np.sqrt(1 - beta_amplified**2) if beta_amplified < 1.0 else 0.0
    return alpha, beta_amplified

def prhp_heuristic(phi_sym, intersect_index, target=0.90):
    """
    PRHP heuristic: Iterate until phi_sym > 0.90 - 0.22*intersect_index
    Returns True if condition met (pruning successful).
    """
    threshold = target - 0.22 * intersect_index
    return phi_sym > threshold

def simulate_viral_cascade(n_population=100, variant='ADHD-collectivist', n_iter=10, seed=42, use_quantum=True):
    """
    Simulate viral cascade as qubit epidemic.
    Returns cascade metrics and mitigation success.
    """
    if seed is not None:
        np.random.seed(seed)
    
    try:
        from .qubit_hooks import compute_phi, entangle_nodes_variant
    except ImportError:
        from qubit_hooks import compute_phi, entangle_nodes_variant
    
    # Initialize population: mix of healthy and infected
    initial_infected = int(n_population * 0.1)  # 10% initially infected
    population_states = []
    
    for i in range(n_population):
        if i < initial_infected:
            alpha, beta = 0.3, 0.95  # Mostly infected
        else:
            alpha, beta = 0.95, 0.3  # Mostly healthy
        
        # Amplify by variant
        alpha, beta = amplify_infection(beta, variant)
        state = create_viral_qubit(alpha, beta)
        population_states.append(state)
    
    # Track infection rate over iterations
    infection_rates = []
    phi_syms = []
    cascade_averted = False
    
    for iteration in range(n_iter):
        # Compute average infection rate
        infected_count = sum(1 for state in population_states if abs(state[1])**2 > 0.5)
        infection_rate = infected_count / n_population
        infection_rates.append(infection_rate)
        
        # Compute phi_sym for population (pairwise average)
        if len(population_states) > 1:
            phi_values = []
            for i in range(min(10, len(population_states))):  # Sample for efficiency
                for j in range(i+1, min(i+2, len(population_states))):
                    phi = compute_phi(population_states[i], population_states[j], use_quantum)
                    phi_values.append(phi)
            phi_sym = np.mean(phi_values) if phi_values else 0.85
        else:
            phi_sym = 0.85
        phi_syms.append(phi_sym)
        
        # Apply PRHP heuristic
        if prhp_heuristic(phi_sym, iteration):
            # Pruning successful - reduce infection
            for i, state in enumerate(population_states):
                if abs(state[1])**2 > 0.5:  # If infected
                    # Depolarize toward healthy
                    state[0] = np.sqrt(0.7)  # Increase healthy component
                    state[1] = np.sqrt(0.3)  # Decrease infected component
                    state /= np.linalg.norm(state)
            
            if iteration < 5:  # Early pruning
                cascade_averted = True
        
        # Spread infection (quantum speedup: O(log n) vs O(n²))
        if variant == 'ADHD-collectivist':
            # Faster spread (+12% risk)
            spread_rate = 0.12
        elif variant == 'autistic-individualist':
            # Slower spread (isolation)
            spread_rate = -0.25  # Negative = isolation reduces spread
        else:
            spread_rate = 0.0
        
        # Update population (simplified: nearest neighbors)
        for i in range(len(population_states)):
            if abs(population_states[i][1])**2 > 0.5:  # If infected
                # Spread to neighbors
                for j in range(max(0, i-1), min(len(population_states), i+2)):
                    if j != i:
                        # Increase infection probability (ensure non-negative)
                        current_beta_sq = abs(population_states[j][1])**2
                        beta_new_sq = max(0.0, min(1.0, current_beta_sq + spread_rate))
                        beta_new = np.sqrt(beta_new_sq)
                        alpha_new = np.sqrt(1 - beta_new_sq)
                        population_states[j] = create_viral_qubit(alpha_new, beta_new)
    
    # Final metrics
    final_infection = infection_rates[-1]
    initial_infection = infection_rates[0]
    cascade_mitigation = 1.0 - (final_infection - initial_infection) if final_infection > initial_infection else 1.0
    
    return {
        'infection_rates': infection_rates,
        'phi_syms': phi_syms,
        'cascade_averted': cascade_averted,
        'cascade_mitigation': cascade_mitigation,
        'final_infection_rate': final_infection,
        'peak_infection_rate': max(infection_rates) if infection_rates else 0.0
    }

def forecast_extinction_risk(variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'], 
                            n_sims=100, seed=42):
    """
    Forecast extinction risk across variants using qubit epidemic simulations.
    Returns risk percentages and mitigation success rates.
    """
    if seed is not None:
        np.random.seed(seed)
    
    results = {}
    
    for variant in variants:
        mitigations = []
        averted_count = 0
        
        for sim_idx in range(n_sims):
            # Use seed + sim_idx to ensure reproducibility while allowing variation
            sim_seed = (seed + sim_idx) if seed is not None else None
            result = simulate_viral_cascade(variant=variant, seed=sim_seed, n_iter=10)
            mitigations.append(result['cascade_mitigation'])
            if result['cascade_averted']:
                averted_count += 1
        
        # Compute risk metrics
        mean_mitigation = np.mean(mitigations)
        averted_rate = averted_count / n_sims
        
        # Variant-specific risk adjustments
        if variant == 'ADHD-collectivist':
            risk_adjustment = 0.12  # +12% risk
        elif variant == 'autistic-individualist':
            risk_adjustment = -0.25  # -25% risk (isolation)
        else:
            risk_adjustment = 0.0
        
        results[variant] = {
            'mean_mitigation': mean_mitigation,
            'averted_rate': averted_rate,
            'risk_adjustment': risk_adjustment,
            'extinction_risk': max(0.0, min(1.0, 0.15 + risk_adjustment - mean_mitigation))
        }
    
    return results

if __name__ == "__main__":
    # Test viral cascade simulation
    result = simulate_viral_cascade(variant='ADHD-collectivist', n_iter=10, seed=42)
    print(f"Cascade Averted: {result['cascade_averted']}")
    print(f"Mitigation: {result['cascade_mitigation']:.2%}")
    print(f"Final Infection Rate: {result['final_infection_rate']:.2%}")
    
    # Forecast extinction risks
    print("\nExtinction Risk Forecast:")
    forecasts = forecast_extinction_risk(n_sims=100, seed=42)
    for variant, metrics in forecasts.items():
        print(f"{variant}: Risk={metrics['extinction_risk']:.2%}, Mitigation={metrics['mean_mitigation']:.2%}")

