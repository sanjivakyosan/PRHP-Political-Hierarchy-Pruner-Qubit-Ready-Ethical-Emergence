# PRHP Framework Benchmarking Guide

## Overview

This document provides guidelines for benchmarking the PRHP (Political Hierarchy Pruner) framework to validate its performance claims, particularly the **84% fidelity** target with standard deviation < 0.025.

## Benchmarking Goals

- **Primary Goal**: Validate the 84% fidelity claim
- **Secondary Goals**: 
  - Measure performance across different variants
  - Validate quantum vs classical simulation differences
  - Test scalability with different Monte Carlo iterations

## Benchmarking Method

### Basic Benchmark

Run the core simulation with standard parameters:

```python
from src.prhp_core import simulate_prhp

results = simulate_prhp(
    levels=9,
    variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
    n_monte=100,
    seed=42,
    use_quantum=True,
    track_levels=True
)

# Check fidelity
for variant, data in results.items():
    fidelity = data['mean_fidelity']
    std = data['std']
    print(f"{variant}: {fidelity:.3f} ± {std:.3f}")
```

### Expected Results

- **Mean Fidelity**: ~0.84 (84%)
- **Standard Deviation**: < 0.025
- **Novelty Generation**: ~0.80

### Quantum vs Classical Comparison

```python
# Quantum simulation
results_quantum = simulate_prhp(n_monte=100, use_quantum=True)

# Classical simulation
results_classical = simulate_prhp(n_monte=100, use_quantum=False)

# Compare results
```

## Running Benchmarks

### Quick Benchmark

```bash
python examples/simple_usage.py
```

### Comprehensive Benchmark

```bash
python examples/comprehensive_demo.py
```

### Custom Benchmark Script

See `examples/` directory for more benchmark examples.

## Contributing Benchmarks

We welcome benchmark results from the community! To contribute:

1. Run benchmarks using the methods above
2. Document your environment (Python version, Qiskit version, hardware)
3. Share results via GitHub Issues or Pull Requests

## Notes

- Benchmarks should be run with Qiskit 2.x for quantum simulations
- Results may vary slightly based on hardware and random seed
- For reproducible results, always set `seed=42`

## Invitation

**@xai engineers and community**: Test and PR your benchmark results!

---

Copyright © sanjivakyosan 2025
