# PRHP Simulation - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [What is PRHP Simulation?](#what-is-prhp-simulation)
3. [How It Works](#how-it-works)
4. [Parameters Reference](#parameters-reference)
5. [Output Metrics Explained](#output-metrics-explained)
6. [Usage Examples](#usage-examples)
7. [Understanding Results](#understanding-results)
8. [Best Practices](#best-practices)
9. [Technical Details](#technical-details)

---

## Overview

**PRHP** stands for **Political Hierarchy Pruner** - a quantum-computing framework that simulates neuro-cultural hierarchies using Integrated Information Theory (IIT) and quantum game theory principles.

The PRHP Simulation is a Monte Carlo-based quantum simulation that models how different neuro-cultural variants (ADHD-collectivist, autistic-individualist, neurotypical-hybrid) process information and maintain hierarchical structures.

---

## What is PRHP Simulation?

The PRHP Simulation is the core function of the framework that:

- **Simulates hierarchical structures** across multiple levels
- **Models neuro-cultural variants** with quantum operations
- **Tracks consciousness metrics** (phi values) using IIT principles
- **Applies political pruning** to maintain system coherence
- **Generates statistical results** through Monte Carlo iterations

### Key Concepts

- **Hierarchy Levels**: Represents layers of organizational or cognitive structure (typically 9 levels)
- **Neuro-Cultural Variants**: Different cognitive processing styles combined with cultural orientations
- **Quantum Fidelity**: Measure of how well quantum states are preserved
- **Phi (Φ)**: IIT-inspired consciousness measure based on integrated information
- **Asymmetry**: Measure of imbalance between system components
- **Novelty Generation**: System's ability to generate new states

---

## How It Works

The simulation follows this process:

1. **Initialization**: Creates entangled quantum states for each variant
2. **Level-by-Level Processing**: For each hierarchy level:
   - Injects predation effects (hierarchy-tuned waning)
   - Injects political hierarchy (power structure effects)
   - Applies firewall/pruning logic
   - Computes phi values and asymmetry
3. **Monte Carlo Iteration**: Repeats the process multiple times for statistical accuracy
4. **Aggregation**: Computes mean values, standard deviations, and variant-specific metrics

### Simulation Flow

```
For each variant:
  For each Monte Carlo iteration:
    Initialize quantum states (entanglement)
    For each hierarchy level (1 to N):
      → Inject predation (divergence effects)
      → Inject political hierarchy (power effects)
      → Apply firewall (pruning logic)
      → Compute phi values
      → Track metrics
    Store iteration results
  Aggregate across all iterations
  Compute variant-specific metrics
```

---

## Parameters Reference

### Core Parameters

#### `levels` (int, default: 9)
- **Range**: 1 to 20
- **Description**: Number of hierarchy levels to simulate
- **Effect**: 
  - More levels = deeper hierarchy simulation
  - Each level adds complexity and processing time
  - Typical value: 9 (represents a standard organizational hierarchy)
- **Example**: 
  - `levels=5`: Shallow hierarchy (quick simulation)
  - `levels=9`: Standard hierarchy (recommended)
  - `levels=15`: Deep hierarchy (detailed analysis)

#### `variants` (List[str], default: ['neurotypical-hybrid'])
- **Options**: 
  - `'ADHD-collectivist'`: Collective-oriented with ADHD cognitive patterns
  - `'autistic-individualist'`: Individual-oriented with autistic cognitive patterns
  - `'neurotypical-hybrid'`: Balanced neurotypical with hybrid cultural orientation
- **Description**: Neuro-cultural variants to simulate
- **Effect**: Each variant has different:
  - Dopamine gradients (0.20, 0.15, 0.18 respectively)
  - Asymmetry multipliers (+28%, -47%, +20%)
  - Quantum operations (σ_z, σ_x, Hadamard)
- **Example**: 
  - `variants=['neurotypical-hybrid']`: Single variant
  - `variants=['ADHD-collectivist', 'autistic-individualist']`: Compare two
  - `variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']`: All variants

#### `n_monte` (int, default: 100)
- **Range**: 1 to 2000
- **Description**: Number of Monte Carlo iterations
- **Effect**: 
  - More iterations = better statistical accuracy
  - Lower standard deviation in results
  - Longer computation time
- **Guidelines**:
  - `n_monte=10`: Quick test (low accuracy)
  - `n_monte=100`: Standard simulation (recommended)
  - `n_monte=500`: High accuracy (slower)
  - `n_monte=2000`: Maximum accuracy (very slow)

#### `seed` (int, optional, default: 42)
- **Range**: Any non-negative integer, or `None`
- **Description**: Random seed for reproducibility
- **Effect**: 
  - Same seed = same results (reproducible)
  - Different seed = different results (exploration)
  - `None` = random seed each run
- **Example**: 
  - `seed=42`: Reproducible results
  - `seed=None`: Random variation each run

#### `use_quantum` (bool, default: True)
- **Options**: `True` or `False`
- **Description**: Whether to use quantum simulation (Qiskit) or classical approximation
- **Effect**: 
  - `True`: Uses Qiskit for quantum operations (more accurate, requires Qiskit)
  - `False`: Uses classical numpy approximation (faster, no Qiskit needed)
- **Recommendation**: Use `True` if Qiskit is installed, `False` for faster testing

#### `track_levels` (bool, default: True)
- **Options**: `True` or `False`
- **Description**: Whether to track per-level metrics
- **Effect**: 
  - `True`: Returns detailed per-level phi deltas and metrics
  - `False`: Only returns aggregate metrics (faster, less data)
- **Recommendation**: Use `True` for detailed analysis, `False` for quick overview

#### `show_progress` (bool, default: True)
- **Options**: `True` or `False`
- **Description**: Whether to show progress bars during simulation
- **Effect**: 
  - `True`: Shows tqdm progress bars (user-friendly)
  - `False`: No progress bars (cleaner output, faster)
- **Note**: Automatically disabled in web UI

---

## Output Metrics Explained

### Primary Metrics

#### `mean_fidelity` (float)
- **Description**: Average quantum state fidelity across all iterations
- **Range**: Typically 0.80 to 0.90
- **Target**: 0.84 ± 0.025 (84% fidelity)
- **Interpretation**: 
  - Higher = better state preservation
  - Lower = more decoherence/noise
  - Measures how well quantum information is maintained

#### `std` (float)
- **Description**: Standard deviation of fidelity values
- **Range**: Typically 0.01 to 0.05
- **Target**: < 0.025 (low variance)
- **Interpretation**: 
  - Lower = more consistent results
  - Higher = more variation between iterations
  - Good simulations have std < 0.025

#### `asymmetry_delta` (float)
- **Description**: Variant-specific asymmetry measure
- **Variant Multipliers**:
  - ADHD-collectivist: +28% (1.28x multiplier)
  - Autistic-individualist: -47% (0.53x multiplier)
  - Neurotypical-hybrid: +20% (1.20x multiplier)
- **Interpretation**: 
  - Measures imbalance between system components
  - Higher = more asymmetry (collectivist/hybrid)
  - Lower = more symmetry (individualist)

#### `novelty_gen` (float)
- **Description**: Novelty generation capacity
- **Range**: Typically 0.78 to 0.82
- **Baseline**: 0.80
- **Formula**: `0.80 + 0.02 * (mean_fidelity - 0.84)`
- **Interpretation**: 
  - Higher = more ability to generate novel states
  - Increases with fidelity
  - Measures system creativity/adaptability

#### `mean_phi_delta` (float, optional)
- **Description**: Average change in phi (consciousness measure) per level
- **Range**: Typically 0.10 to 0.15
- **Target**: 0.12
- **Interpretation**: 
  - Measures consciousness evolution through hierarchy
  - Higher = more integrated information
  - IIT-inspired metric

#### `std_phi_delta` (float, optional)
- **Description**: Standard deviation of phi deltas
- **Interpretation**: Consistency of consciousness changes

### Per-Level Metrics (when `track_levels=True`)

#### `phi_deltas` (List[float])
- **Description**: Phi delta for each hierarchy level
- **Length**: Equal to `levels`
- **Interpretation**: Shows how phi changes at each level
- **Pattern**: Typically decreases with level (0.11 → 0.09 → ... → 0.03)

#### `level_phis` (List[float])
- **Description**: Average phi value at each level
- **Length**: Equal to `levels`
- **Interpretation**: Consciousness measure at each hierarchy level
- **Pattern**: Typically starts around 0.87, evolves through levels

#### `level_N` (Dict, for each level)
- **Description**: Per-level detailed metrics
- **Contains**:
  - `phi_delta_mean`: Average phi change at this level
  - `fidelity_mean`: Average fidelity at this level
  - `success`: Whether level achieved target (> 0.86)

### Aggregate Metrics

#### `mean_success_rate` (float)
- **Description**: Percentage of levels that achieved success (phi > 0.86)
- **Range**: 0.0 to 1.0
- **Interpretation**: Overall system health/coherence

#### `fidelity_traces` (List[float])
- **Description**: Fidelity changes between levels
- **Interpretation**: Shows fidelity evolution through hierarchy

---

## Usage Examples

### Basic Usage

```python
from src.prhp_core import simulate_prhp

# Simple simulation
results = simulate_prhp(
    levels=9,
    variants=['neurotypical-hybrid'],
    n_monte=100,
    seed=42
)

# Access results
for variant, data in results.items():
    print(f"{variant}:")
    print(f"  Fidelity: {data['mean_fidelity']:.3f} ± {data['std']:.3f}")
    print(f"  Novelty: {data['novelty_gen']:.4f}")
```

### Compare All Variants

```python
results = simulate_prhp(
    levels=9,
    variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
    n_monte=200,  # Higher accuracy for comparison
    seed=42
)

# Compare fidelities
for variant, data in results.items():
    print(f"{variant:25s} Fidelity: {data['mean_fidelity']:.4f}")
```

### Quick Test Run

```python
# Fast test with minimal iterations
results = simulate_prhp(
    levels=5,
    variants=['neurotypical-hybrid'],
    n_monte=10,
    use_quantum=False,  # Faster classical mode
    track_levels=False  # Skip detailed tracking
)
```

### Detailed Analysis

```python
# Comprehensive analysis with all details
results = simulate_prhp(
    levels=9,
    variants=['ADHD-collectivist', 'autistic-individualist'],
    n_monte=500,
    seed=42,
    use_quantum=True,
    track_levels=True  # Get per-level data
)

# Access per-level data
for variant, data in results.items():
    print(f"\n{variant}:")
    print(f"  Per-level phi deltas: {data['phi_deltas']}")
    print(f"  Per-level phis: {data['level_phis']}")
    
    # Access individual level data
    for level in range(1, 10):
        level_key = f'level_{level}'
        if level_key in data:
            level_data = data[level_key]
            print(f"  Level {level}: phi_delta={level_data['phi_delta_mean']:.4f}, "
                  f"fidelity={level_data['fidelity_mean']:.4f}, "
                  f"success={level_data['success']}")
```

### Web UI Usage

In the web interface:
1. Navigate to "PRHP Simulation" tab
2. Adjust sliders:
   - **Hierarchy Levels**: 1-20 (default: 9)
   - **Monte Carlo Iterations**: 10-1000 (default: 100)
   - **Random Seed**: 0-1000 (default: 42)
3. Select variants (checkboxes)
4. Toggle options:
   - Use Quantum Simulation
   - Track Per-Level Metrics
5. Click "Run Simulation"
6. View results in the output panel

---

## Understanding Results

### Interpreting Fidelity

- **0.84 ± 0.025**: Target range (good performance)
- **> 0.85**: Excellent (very stable)
- **0.80-0.84**: Good (acceptable)
- **< 0.80**: Poor (high decoherence)

### Interpreting Asymmetry Delta

- **ADHD-collectivist**: Typically 0.15-0.25 (higher asymmetry)
- **Autistic-individualist**: Typically 0.05-0.10 (lower asymmetry)
- **Neurotypical-hybrid**: Typically 0.10-0.18 (moderate asymmetry)

### Interpreting Novelty Generation

- **0.80-0.81**: Baseline (standard)
- **0.81-0.82**: Enhanced (good creativity)
- **< 0.80**: Reduced (limited adaptability)

### Variant Comparison

When comparing variants, look for:

1. **Fidelity differences**: Which variant maintains better quantum coherence?
2. **Asymmetry patterns**: How do collectivist vs individualist differ?
3. **Novelty generation**: Which variant generates more novel states?
4. **Per-level evolution**: How do variants differ across hierarchy levels?

### Success Indicators

A successful simulation should show:
- ✅ Fidelity around 0.84 ± 0.025
- ✅ Low standard deviation (< 0.025)
- ✅ Mean phi delta around 0.12
- ✅ Novelty gen in range 0.78-0.82
- ✅ Consistent per-level patterns

---

## Best Practices

### Parameter Selection

1. **For Quick Testing**:
   - `levels=5`, `n_monte=10`, `use_quantum=False`
   - Fast feedback, lower accuracy

2. **For Standard Analysis**:
   - `levels=9`, `n_monte=100`, `use_quantum=True`
   - Balanced accuracy and speed

3. **For Publication/Research**:
   - `levels=9`, `n_monte=500-1000`, `use_quantum=True`, `track_levels=True`
   - Maximum accuracy and detail

### Reproducibility

- Always use a fixed `seed` for reproducible results
- Document your parameter choices
- Run multiple seeds to check robustness

### Performance Optimization

- Use `use_quantum=False` for faster testing
- Set `track_levels=False` if you don't need per-level data
- Reduce `n_monte` for quick iterations
- Use fewer variants if comparing specific cases

### Error Handling

Common issues and solutions:

- **Low fidelity**: Increase `n_monte`, check quantum setup
- **High variance**: Increase `n_monte` for better statistics
- **Memory issues**: Reduce `levels` or `n_monte`
- **Slow performance**: Use `use_quantum=False` or reduce iterations

---

## Technical Details

### Quantum Operations

The simulation uses variant-specific quantum gates:

- **ADHD-collectivist**: σ_z rotations with Gaussian noise (σ=0.12)
- **Autistic-individualist**: σ_x rotations for isolation (σ=-0.08)
- **Neurotypical-hybrid**: Hadamard gates (balanced)

### Mathematical Foundations

1. **Phi Computation**: Based on IIT (Integrated Information Theory)
   - Uses mutual information from joint density matrices
   - Formula: `phi = 0.85 + 0.1 * mutual_info`

2. **Fidelity**: Quantum state fidelity
   - Measures overlap between states
   - Target: 84% (0.84)

3. **Asymmetry**: Normalized difference
   - `asymmetry = |phi_a - phi_b| / mean(phi_a, phi_b)`

4. **Novelty Generation**: Linear scaling with fidelity
   - `novelty_gen = 0.80 + 0.02 * (fidelity - 0.84)`

### Internal Processes

1. **Predation Injection**: Adds noise based on hierarchy level
   - `divergence = 0.22 * (1 - 0.03 * level)`
   - Decreases with level (waning effect)

2. **Political Hierarchy**: Power structure effects
   - `divergence = 0.15 * level / levels`
   - Increases with level (escalation)

3. **Firewall/Pruning**: Maintains coherence
   - Variant-specific thresholds
   - Depolarizes states when asymmetry too high

### Performance Characteristics

- **Time Complexity**: O(levels × n_monte × variants)
- **Space Complexity**: O(levels × n_monte)
- **Quantum Mode**: Slower but more accurate
- **Classical Mode**: Faster but approximate

### Dependencies

- **Required**: numpy, scipy
- **Optional**: qiskit (for quantum mode)
- **Progress**: tqdm (for progress bars)

---

## Additional Resources

- **Configuration**: See `config/default.yaml` for default parameters
- **Examples**: Check `examples/full_prhp_demo.py`
- **Visualization**: Use `src/visualization.py` for plotting results
- **Documentation**: See `docs/USER_GUIDE.md` for general framework usage

---

## Quick Reference Card

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `levels` | 9 | 1-20 | Hierarchy depth |
| `n_monte` | 100 | 1-1000 | Statistical accuracy |
| `seed` | 42 | Any int | Reproducibility |
| `use_quantum` | True | bool | Quantum vs classical |
| `track_levels` | True | bool | Per-level details |
| `variants` | ['neurotypical-hybrid'] | List[str] | Variants to simulate |

| Metric | Target | Interpretation |
|--------|--------|---------------|
| `mean_fidelity` | 0.84 ± 0.025 | Quantum coherence |
| `std` | < 0.025 | Result consistency |
| `asymmetry_delta` | Variant-specific | System balance |
| `novelty_gen` | 0.80-0.82 | Creativity capacity |
| `mean_phi_delta` | ~0.12 | Consciousness evolution |

---

**Last Updated**: 2025
**Version**: 1.0
**Framework**: PRHP (Political Hierarchy Pruner)

