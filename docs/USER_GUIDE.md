# PRHP Framework User Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Configuration](#configuration)
6. [Visualization](#visualization)
7. [Performance Optimization](#performance-optimization)
8. [Examples](#examples)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.prhp_core import simulate_prhp

# Basic simulation
results = simulate_prhp(
    levels=9,
    variants=['ADHD-collectivist', 'autistic-individualist'],
    n_monte=100,
    seed=42
)

for variant, metrics in results.items():
    print(f"{variant}: Fidelity = {metrics['mean_fidelity']:.3f} ± {metrics['std']:.3f}")
```

## Basic Usage

### Running Simulations

```python
from src.prhp_core import simulate_prhp

# Full simulation with all features
results = simulate_prhp(
    levels=9,                              # Number of hierarchy levels
    variants=['ADHD-collectivist',         # Variants to simulate
              'autistic-individualist',
              'neurotypical-hybrid'],
    n_monte=100,                           # Monte Carlo iterations
    seed=42,                               # Random seed
    use_quantum=True,                      # Use quantum simulation
    track_levels=True,                     # Track per-level metrics
    show_progress=True                     # Show progress bars
)

# Access results
for variant, data in results.items():
    print(f"\n{variant}:")
    print(f"  Fidelity: {data['mean_fidelity']:.3f} ± {data['std']:.3f}")
    print(f"  Novelty Gen: {data['novelty_gen']:.4f}")
    print(f"  Asymmetry Delta: {data['asymmetry_delta']:.3f}")
    if data['level_phis']:
        print(f"  Level 9 Phi: {data['level_phis'][-1]:.3f}")
```

### Qubit Operations

```python
from src.qubit_hooks import entangle_nodes_variant, compute_phi, inject_phase_flip

# Entangle nodes with variant-specific operations
state_a, state_b, fidelity, symmetry = entangle_nodes_variant(
    variant='ADHD-collectivist',
    use_quantum=True,
    seed=42
)

# Compute phi
phi = compute_phi(state_a, state_b, use_quantum=True)
print(f"Phi: {phi:.3f}")

# Inject phase flip
state_b_modified = inject_phase_flip(
    state_b,
    flip_prob=0.25,
    variant='ADHD-collectivist'
)
```

### Political Pruner

```python
from src.political_pruner import simulate_pruner_levels

result = simulate_pruner_levels(
    levels=9,
    variant='autistic-individualist',
    n_monte=100,
    seed=42,
    use_quantum=True
)

print(f"Level Deltas: {result['level_deltas']}")
print(f"Self-Model Coherence: {result['self_model_coherence']:.4f}")
print(f"Success Rate: {result['success_rate']:.2%}")
```

### Virus Extinction Forecasts

```python
from src.virus_extinction import forecast_extinction_risk, simulate_viral_cascade

# Single cascade simulation
cascade = simulate_viral_cascade(
    n_population=100,
    variant='ADHD-collectivist',
    n_iter=10,
    seed=42
)

print(f"Cascade Averted: {cascade['cascade_averted']}")
print(f"Mitigation: {cascade['cascade_mitigation']:.2%}")

# Forecast extinction risks
forecasts = forecast_extinction_risk(
    variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
    n_sims=100,
    seed=42
)

for variant, metrics in forecasts.items():
    print(f"{variant}: Risk={metrics['extinction_risk']:.2%}")
```

### Meta-Empirical Validation

```python
from src.meta_empirical import full_meta_empirical_loop
from src.prhp_core import simulate_prhp

# Run simulation
results = simulate_prhp(levels=9, n_monte=100, seed=42)

# Meta-empirical validation
meta_result = full_meta_empirical_loop(
    simulation_results=results['ADHD-collectivist'],
    phi_deltas=results['ADHD-collectivist']['phi_deltas'],
    initial_novelty=0.80
)

print(f"Validated: {meta_result['meta_validated']}")
print(f"Novelty Gen: {meta_result['novelty_gen_updated']:.4f}")
```

## Advanced Features

### Using Configuration Files

```python
from src.config import load_config
from src.prhp_core import simulate_prhp

# Load configuration
config = load_config('config/default.yaml')

# Use config values
results = simulate_prhp(
    levels=config['simulation']['levels'],
    variants=config['variants'],
    n_monte=config['simulation']['n_monte'],
    seed=config['simulation']['seed']
)
```

### Visualization

```python
from src.prhp_core import simulate_prhp
from src.visualization import create_all_plots

# Run simulation
results = simulate_prhp(levels=9, n_monte=100, seed=42)

# Create all plots
create_all_plots(results, output_dir="plots")

# Or create individual plots
from src.visualization import plot_fidelity_comparison, plot_level_phis

plot_fidelity_comparison(results, save_path="plots/fidelity.png")
plot_level_phis(results, save_path="plots/level_phis.png")
```

### Parallel Processing

```python
from src.parallel import parallel_variant_simulation
from src.prhp_core import simulate_prhp

# Run simulations in parallel
results = parallel_variant_simulation(
    simulate_func=simulate_prhp,
    variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
    n_monte=100,
    n_jobs=4,  # Use 4 parallel workers
    levels=9,
    seed=42
)
```

### Performance Profiling

```python
from src.profiling import benchmark_simulation, profile_context
from src.prhp_core import simulate_prhp

# Benchmark a simulation
benchmark = benchmark_simulation(
    simulate_prhp,
    n_runs=5,
    levels=9,
    n_monte=100,
    seed=42
)

print(f"Mean time: {benchmark['mean_time']:.3f}s")
print(f"Std time: {benchmark['std_time']:.3f}s")

# Profile code block
with profile_context('profile.stats'):
    results = simulate_prhp(levels=9, n_monte=100, seed=42)
```

### Logging

```python
from src.utils import setup_logging, get_logger

# Configure logging
logger = setup_logging(level="DEBUG", log_file="prhp.log")

# Use logger
logger.info("Starting simulation")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error message")
```

## Configuration

Create a `config.yaml` file:

```yaml
simulation:
  levels: 9
  n_monte: 100
  seed: 42
  use_quantum: true
  track_levels: true

variants:
  - ADHD-collectivist
  - autistic-individualist
  - neurotypical-hybrid
```

Load and use:

```python
from src.config import load_config, get_config_value

config = load_config('config.yaml')
levels = get_config_value(config, 'simulation.levels')
```

## Examples

### Example 1: Basic Simulation

```python
from src.prhp_core import simulate_prhp

results = simulate_prhp(
    levels=9,
    variants=['ADHD-collectivist'],
    n_monte=100,
    seed=42
)

print(f"Fidelity: {results['ADHD-collectivist']['mean_fidelity']:.3f}")
```

### Example 2: Full Analysis

```python
from src.prhp_core import simulate_prhp
from src.meta_empirical import full_meta_empirical_loop
from src.visualization import create_all_plots

# Run simulation
results = simulate_prhp(levels=9, n_monte=100, seed=42, show_progress=True)

# Meta-empirical validation
for variant, data in results.items():
    meta = full_meta_empirical_loop(data, data['phi_deltas'])
    print(f"{variant}: Validated={meta['meta_validated']}")

# Create visualizations
create_all_plots(results, output_dir="plots")
```

### Example 3: Performance Comparison

```python
from src.profiling import compare_performance
from src.prhp_core import simulate_prhp

# Compare quantum vs classical
funcs = {
    'Quantum': lambda: simulate_prhp(levels=9, n_monte=50, use_quantum=True, show_progress=False),
    'Classical': lambda: simulate_prhp(levels=9, n_monte=50, use_quantum=False, show_progress=False)
}

comparison = compare_performance(funcs, n_runs=3)
```

## Troubleshooting

### Common Issues

1. **Qiskit not found**: Framework falls back to classical simulation automatically
2. **Memory issues**: Reduce `n_monte` or use parallel processing with fewer workers
3. **Slow performance**: Enable parallel processing or reduce `levels`

### Getting Help

- Check logs: `logger.setLevel("DEBUG")`
- Run tests: `python -m pytest tests/`
- Profile code: Use `profile_context` or `benchmark_simulation`

