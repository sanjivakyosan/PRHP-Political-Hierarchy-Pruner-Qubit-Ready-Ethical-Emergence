# Quantum Hooks - Novelty Recalibration - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [What is Quantum Hooks?](#what-is-quantum-hooks)
3. [How It Works](#how-it-works)
4. [Parameters Reference](#parameters-reference)
5. [Core Functions Explained](#core-functions-explained)
6. [Output Metrics Explained](#output-metrics-explained)
7. [Usage Examples](#usage-examples)
8. [Understanding Results](#understanding-results)
9. [Best Practices](#best-practices)
10. [Technical Details](#technical-details)

---

## Overview

**Quantum Hooks - Novelty Recalibration** is an experimental quantum computing module that explores how quantum states evolve under various perturbations and how systems maintain coherence through novelty recalibration mechanisms.

The module simulates:
- **Quantum entanglement** with variant-specific operations
- **Phase flip injection** to introduce asymmetry
- **Predation effects** to model hierarchy-tuned waning
- **Novelty recalibration** to maintain system coherence

---

## What is Quantum Hooks?

Quantum Hooks is a quantum simulation tool that:

- **Creates entangled quantum states** using Bell states
- **Applies variant-specific quantum operations** (σ_z, σ_x, Hadamard gates)
- **Injects perturbations** (phase flips, predation, hierarchy effects)
- **Recalibrates novelty** by pruning states when asymmetry exceeds threshold
- **Tracks consciousness metrics** (phi values) throughout the process

### Key Concepts

- **Bell State Entanglement**: Creates maximally entangled quantum states |Ψ⟩ = (1/√2)(|00⟩ + |11⟩)
- **Phase Flip**: Introduces phase noise to create asymmetry in quantum states
- **Predation**: Models hierarchy-tuned waning effects through noise injection
- **Novelty Recalibration**: Pruning mechanism that depolarizes states when asymmetry is too high
- **Phi (Φ)**: IIT-inspired consciousness measure based on integrated information
- **Asymmetry**: Measure of imbalance between two quantum states

---

## How It Works

The Quantum Hooks simulation follows this process:

1. **Initial Entanglement**: Creates variant-specific entangled quantum states
2. **Predation Injection**: Adds noise based on divergence parameter
3. **Phase Flip Injection**: Applies phase noise based on flip_prob and variant
4. **Novelty Recalibration**: Checks asymmetry against threshold and prunes if needed
5. **Metrics Computation**: Calculates phi values at each stage

### Simulation Flow

```
1. Entangle Nodes (variant-specific)
   ↓
2. Compute Initial Phi (baseline measurement)
   ↓
3. Inject Predation (divergence parameter)
   ↓
4. Compute Phi After Predation
   ↓
5. Inject Phase Flip (flip_prob parameter)
   ↓
6. Compute Phi After Phase Flip
   ↓
7. Recalibrate Novelty (threshold parameter)
   ↓
8. Compute Final Phi (after recalibration)
   ↓
9. Calculate Metrics (asymmetry, phi delta, pruning status)
```

---

## Parameters Reference

### Core Parameters

#### `variant` (str, default: 'neurotypical-hybrid')
- **Options**: 
  - `'ADHD-collectivist'`: Collective-oriented with ADHD cognitive patterns
  - `'autistic-individualist'`: Individual-oriented with autistic cognitive patterns
  - `'neurotypical-hybrid'`: Balanced neurotypical with hybrid cultural orientation
- **Description**: Neuro-cultural variant that determines quantum operations
- **Effect on Quantum Operations**:
  - **ADHD-collectivist**: σ_z rotations with Gaussian noise (σ=0.12) for swarm superposition
  - **Autistic-individualist**: σ_x rotations for isolation (σ=-0.08)
  - **Neurotypical-hybrid**: Hadamard gates (balanced median)
- **Effect on Phase Flip**:
  - **ADHD-collectivist**: Gradient = 0.20 (moderate phase noise)
  - **Autistic-individualist**: Gradient = 0.15 (lower phase noise)
  - **Neurotypical-hybrid**: Gradient = 0.18 (balanced phase noise)
- **Example**: 
  - `variant='ADHD-collectivist'`: More phase noise, swarm-like behavior
  - `variant='autistic-individualist'`: Less phase noise, isolated behavior
  - `variant='neurotypical-hybrid'`: Balanced behavior

#### `use_quantum` (bool, default: True)
- **Options**: `True` or `False`
- **Description**: Whether to use quantum simulation (Qiskit) or classical approximation
- **Effect**: 
  - `True`: Uses Qiskit for quantum operations (more accurate, requires Qiskit)
  - `False`: Uses classical numpy approximation (faster, no Qiskit needed)
- **Quantum Mode**:
  - Creates actual quantum circuits with gates
  - Uses Bell state entanglement
  - More physically accurate
- **Classical Mode**:
  - Uses numpy matrix operations
  - Approximates quantum effects
  - Faster computation
- **Recommendation**: Use `True` if Qiskit is installed, `False` for faster testing

#### `flip_prob` (float, default: 0.30)
- **Range**: 0.0 to 1.0
- **Step**: 0.05 (in UI slider)
- **Description**: Phase flip probability - controls how much phase noise is introduced
- **Effect**: 
  - **Low values (0.0-0.3)**: Minimal phase noise, states remain coherent
  - **Medium values (0.3-0.6)**: Moderate phase noise, noticeable asymmetry
  - **High values (0.6-1.0)**: Significant phase noise, high asymmetry
- **Mathematical Formula**: 
  - `theta_variant = flip_prob * grad`
  - Phase applied: `exp(1j * π * theta_variant)`
- **Impact on Results**:
  - Higher flip_prob → More phase noise → Higher asymmetry → More likely pruning
  - Lower flip_prob → Less phase noise → Lower asymmetry → Less likely pruning
- **Example**: 
  - `flip_prob=0.1`: Subtle phase effects
  - `flip_prob=0.5`: Moderate phase effects
  - `flip_prob=0.9`: Strong phase effects

#### `threshold` (float, default: 0.70)
- **Range**: 0.0 to 1.0
- **Step**: 0.05 (in UI slider)
- **Description**: Asymmetry threshold for novelty recalibration (pruning trigger)
- **Effect**: 
  - **Low threshold (0.0-0.4)**: Prunes frequently (aggressive recalibration)
  - **Medium threshold (0.4-0.7)**: Prunes moderately (balanced)
  - **High threshold (0.7-1.0)**: Prunes rarely (conservative)
- **Pruning Logic**:
  - If `asymmetry > threshold` → Pruning occurs (state depolarized)
  - If `asymmetry ≤ threshold` → No pruning (state preserved)
- **Impact on Results**:
  - Lower threshold → More pruning → More state reset → Different final phi
  - Higher threshold → Less pruning → State preserved → Original phi maintained
- **Example**: 
  - `threshold=0.3`: Prunes when asymmetry > 0.3 (frequent pruning)
  - `threshold=0.7`: Prunes when asymmetry > 0.7 (moderate pruning)
  - `threshold=0.9`: Prunes when asymmetry > 0.9 (rare pruning)

#### `divergence` (float, default: 0.22)
- **Range**: 0.0 to 1.0
- **Step**: 0.01 (in UI slider)
- **Description**: Predation divergence parameter - controls hierarchy-tuned waning effects
- **Effect**: 
  - **Low values (0.0-0.1)**: Minimal predation noise
  - **Medium values (0.1-0.3)**: Moderate predation effects
  - **High values (0.3-1.0)**: Strong predation effects
- **Mathematical Formula**: 
  - `predation_noise = divergence * (random_complex_noise)`
  - Applied as: `state += predation_noise * 0.1`
- **Impact on Results**:
  - Higher divergence → More noise → More state perturbation → Different phi values
  - Lower divergence → Less noise → State preserved → Original phi maintained
- **Interpretation**: 
  - Models "predation" or hierarchy-tuned waning effects
  - Represents how systems degrade over time/levels
- **Example**: 
  - `divergence=0.1`: Subtle predation effects
  - `divergence=0.22`: Standard predation (default)
  - `divergence=0.5`: Strong predation effects

#### `seed` (int, optional, default: None)
- **Range**: Any non-negative integer, or `None`
- **Description**: Random seed for reproducibility
- **Effect**: 
  - Same seed = same results (reproducible)
  - Different seed = different results (exploration)
  - `None` = time-based random seed (variation each run)
- **Note**: In web UI, seed is automatically set based on time for variation
- **Example**: 
  - `seed=42`: Reproducible results
  - `seed=None`: Random variation each run

---

## Core Functions Explained

### 1. `entangle_nodes_variant()`

**Purpose**: Creates variant-specific entangled quantum states

**Process**:
1. Creates Bell state: |Ψ⟩ = (1/√2)(|00⟩ + |11⟩)
2. Applies variant-specific quantum gates:
   - ADHD-collectivist: σ_z rotations with Gaussian noise
   - Autistic-individualist: σ_x rotations
   - Neurotypical-hybrid: Hadamard gates
3. Extracts marginal states for state_a and state_b

**Returns**: `(state_a, state_b, fidelity, symmetry)`

**Key Features**:
- Uses Qiskit for quantum mode
- Falls back to classical numpy for non-quantum mode
- Variant-specific operations create different initial states

### 2. `inject_predation()`

**Purpose**: Injects predation effects (hierarchy-tuned waning)

**Process**:
1. Generates random complex noise based on divergence
2. Scales noise by divergence parameter
3. Adds noise to quantum state
4. Renormalizes state

**Formula**: 
```python
predation_noise = divergence * (random_complex_noise)
state += predation_noise * 0.1
state /= norm(state)  # Renormalize
```

**Effect**: Introduces controlled noise to simulate system degradation

### 3. `inject_phase_flip()`

**Purpose**: Introduces phase noise to create asymmetry

**Process**:
1. Computes variant-specific gradient
2. Calculates phase angle: `theta = flip_prob * grad`
3. Applies phase flip: `exp(1j * π * theta)`
4. Renormalizes state

**Formula**: 
```python
theta_variant = flip_prob * grad
state[1] = state[1] * exp(1j * π * theta_variant)
```

**Variant Gradients**:
- ADHD-collectivist: 0.20
- Autistic-individualist: 0.15
- Neurotypical-hybrid: 0.18

**Effect**: Creates asymmetry between states by introducing phase differences

### 4. `recalibrate_novelty()`

**Purpose**: Recalibrates system by pruning states when asymmetry is too high

**Process**:
1. Computes phi values for both states
2. Calculates asymmetry: `|phi_a - phi_b| / mean(phi_a, phi_b)`
3. Compares asymmetry to threshold
4. If asymmetry > threshold:
   - Depolarizes state_b (pruning)
   - Resets state to reduce asymmetry
5. Returns recalibrated states and asymmetry value

**Pruning Formula**:
```python
if asymmetry > threshold:
    rho_b = outer(state_b, conj(state_b))
    depolarized = (rho_b + identity/2) / 2
    state_b = sqrtm(depolarized) @ state_b
```

**Effect**: Maintains system coherence by removing excessive asymmetry

### 5. `compute_phi()`

**Purpose**: Computes IIT-inspired phi (consciousness measure)

**Process**:
1. Creates joint density matrix from two states
2. Computes partial traces for individual states
3. Calculates entropies
4. Computes mutual information: `S(ρ₁) + S(ρ₂) - S(ρ_joint)`
5. Scales to baseline: `phi = 0.85 + 0.1 * mutual_info`

**Quantum Mode**: Uses Qiskit quantum circuits
**Classical Mode**: Uses numpy matrix operations

**Returns**: Phi value (typically 0.85-0.90)

---

## Output Metrics Explained

### Primary Metrics

#### `initial_phi` (float)
- **Description**: Average phi value before any perturbations
- **Range**: Typically 0.85 to 0.90
- **Interpretation**: Baseline consciousness measure
- **Components**: Average of `initial_phi_a` and `initial_phi_b`

#### `initial_phi_a` (float)
- **Description**: Phi value for state_a before perturbations
- **Interpretation**: Consciousness measure of first quantum state

#### `initial_phi_b` (float)
- **Description**: Phi value for state_b before perturbations
- **Interpretation**: Consciousness measure of second quantum state

#### `initial_asymmetry` (float)
- **Description**: Initial asymmetry between states
- **Formula**: `|phi_a - phi_b| / mean(phi_a, phi_b)`
- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - Low (< 0.1): States are symmetric
  - Medium (0.1-0.3): Moderate asymmetry
  - High (> 0.3): Significant asymmetry

### Intermediate Metrics

#### `phi_after_predation` (float)
- **Description**: Phi value after predation injection
- **Interpretation**: Shows effect of divergence parameter
- **Comparison**: Compare to `initial_phi` to see predation impact

#### `phi_after_flip` (float)
- **Description**: Phi value after phase flip injection
- **Interpretation**: Shows effect of flip_prob parameter
- **Comparison**: Compare to `phi_after_predation` to see phase flip impact

### Final Metrics

#### `final_phi` (float)
- **Description**: Average phi value after recalibration
- **Range**: Typically 0.85 to 0.90
- **Interpretation**: Final consciousness measure after all operations
- **Components**: Average of `final_phi_a` and `final_phi_b`

#### `final_phi_a` (float)
- **Description**: Phi value for state_a after recalibration
- **Interpretation**: Final consciousness measure of first state

#### `final_phi_b` (float)
- **Description**: Phi value for state_b after recalibration
- **Interpretation**: Final consciousness measure of second state

#### `asymmetry` (float)
- **Description**: Final asymmetry after recalibration
- **Formula**: `|phi_a - phi_b| / mean(phi_a, phi_b)`
- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - If asymmetry > threshold: Pruning occurred
  - If asymmetry ≤ threshold: No pruning occurred

#### `phi_delta` (float)
- **Description**: Change in phi from initial to final
- **Formula**: `final_phi - initial_phi`
- **Range**: Typically -0.05 to +0.05
- **Interpretation**: 
  - Positive: Phi increased (more integrated information)
  - Negative: Phi decreased (less integrated information)
  - Near zero: Phi maintained (stable system)

### Status Indicators

#### `pruning_occurred` (bool)
- **Description**: Whether pruning happened during recalibration
- **Values**: `True` or `False`
- **Condition**: `asymmetry > threshold`
- **Interpretation**: 
  - `True`: Asymmetry exceeded threshold, state was depolarized
  - `False`: Asymmetry within acceptable range, state preserved

#### `threshold_exceeded` (bool)
- **Description**: Same as `pruning_occurred` (redundant indicator)
- **Purpose**: Confirms threshold comparison result

---

## Usage Examples

### Basic Usage (Python)

```python
from src.qubit_hooks import (
    entangle_nodes_variant, 
    compute_phi, 
    inject_predation, 
    inject_phase_flip, 
    recalibrate_novelty
)

# Create entangled states
state_a, state_b, _, _ = entangle_nodes_variant(
    variant='neurotypical-hybrid',
    use_quantum=True,
    seed=42
)

# Compute initial phi
initial_phi = compute_phi(state_a, state_b, use_quantum=True)
print(f"Initial Phi: {initial_phi:.4f}")

# Inject predation
state_b = inject_predation(state_b, divergence=0.22)

# Inject phase flip
state_b = inject_phase_flip(
    state_b, 
    flip_prob=0.3, 
    variant='neurotypical-hybrid'
)

# Recalibrate novelty
state_a, state_b, asymmetry = recalibrate_novelty(
    state_a, 
    state_b, 
    threshold=0.7, 
    use_quantum=True
)

# Compute final phi
final_phi = compute_phi(state_a, state_b, use_quantum=True)
print(f"Final Phi: {final_phi:.4f}")
print(f"Asymmetry: {asymmetry:.4f}")
```

### Web UI Usage

1. Navigate to "Quantum Hooks" tab
2. Adjust parameters:
   - **Use Quantum Computation**: Toggle quantum/classical mode
   - **Phase Flip Probability**: 0.0-1.0 slider (default: 0.30)
   - **Asymmetry Threshold**: 0.0-1.0 slider (default: 0.70)
   - **Predation Divergence**: 0.0-1.0 slider (default: 0.22)
   - **Variant**: Dropdown selection
3. Click "Run Quantum Simulation"
4. View detailed results showing:
   - Parameters used
   - Initial state metrics
   - After predation metrics
   - After phase flip metrics
   - After recalibration metrics
   - Summary with pruning status

### Exploring Parameter Effects

```python
# Test different flip_prob values
for flip_prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
    state_a, state_b, _, _ = entangle_nodes_variant('neurotypical-hybrid', seed=42)
    state_b = inject_phase_flip(state_b, flip_prob=flip_prob, variant='neurotypical-hybrid')
    state_a, state_b, asym = recalibrate_novelty(state_a, state_b, threshold=0.7)
    phi = compute_phi(state_a, state_b)
    print(f"flip_prob={flip_prob:.1f}: asymmetry={asym:.4f}, phi={phi:.4f}")

# Test different threshold values
for threshold in [0.3, 0.5, 0.7, 0.9]:
    state_a, state_b, _, _ = entangle_nodes_variant('neurotypical-hybrid', seed=42)
    state_b = inject_phase_flip(state_b, flip_prob=0.5, variant='neurotypical-hybrid')
    state_a, state_b, asym = recalibrate_novelty(state_a, state_b, threshold=threshold)
    pruning = asym > threshold
    print(f"threshold={threshold:.1f}: asymmetry={asym:.4f}, pruning={pruning}")
```

### Compare Variants

```python
variants = ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']

for variant in variants:
    state_a, state_b, _, _ = entangle_nodes_variant(variant, seed=42)
    initial_phi = compute_phi(state_a, state_b)
    
    state_b = inject_predation(state_b, divergence=0.22)
    state_b = inject_phase_flip(state_b, flip_prob=0.3, variant=variant)
    
    state_a, state_b, asym = recalibrate_novelty(state_a, state_b, threshold=0.7)
    final_phi = compute_phi(state_a, state_b)
    
    print(f"{variant:25s}: initial_phi={initial_phi:.4f}, "
          f"final_phi={final_phi:.4f}, asymmetry={asym:.4f}")
```

---

## Understanding Results

### Interpreting Phi Values

- **0.85-0.87**: Baseline range (typical initial values)
- **0.87-0.90**: Higher integration (more consciousness)
- **< 0.85**: Lower integration (less consciousness)
- **Changes**: Look at `phi_delta` to see evolution

### Interpreting Asymmetry

- **0.0-0.1**: Very symmetric (balanced states)
- **0.1-0.3**: Moderate asymmetry (acceptable)
- **0.3-0.5**: High asymmetry (may trigger pruning)
- **> 0.5**: Very high asymmetry (likely triggers pruning)

### Understanding Pruning

**When Pruning Occurs**:
- Asymmetry exceeds threshold
- System depolarizes state_b
- Reduces asymmetry to maintain coherence

**Effects of Pruning**:
- State is reset (depolarized)
- Asymmetry reduced
- Final phi may differ from initial
- System maintains coherence

**When No Pruning**:
- Asymmetry within acceptable range
- State preserved
- Original characteristics maintained
- Minimal change in phi

### Parameter Impact Analysis

#### High `flip_prob` (0.7-1.0)
- **Effect**: Strong phase noise
- **Result**: Higher asymmetry, more likely pruning
- **Phi Change**: May decrease due to noise

#### Low `flip_prob` (0.0-0.3)
- **Effect**: Minimal phase noise
- **Result**: Lower asymmetry, less likely pruning
- **Phi Change**: Minimal change

#### Low `threshold` (0.0-0.4)
- **Effect**: Aggressive pruning
- **Result**: Frequent state resets
- **Phi Change**: More variation in final phi

#### High `threshold` (0.7-1.0)
- **Effect**: Conservative pruning
- **Result**: Rare state resets
- **Phi Change**: Less variation, original state preserved

#### High `divergence` (0.5-1.0)
- **Effect**: Strong predation noise
- **Result**: More state perturbation
- **Phi Change**: Noticeable change after predation

#### Low `divergence` (0.0-0.1)
- **Effect**: Minimal predation noise
- **Result**: State mostly preserved
- **Phi Change**: Minimal change after predation

---

## Best Practices

### Parameter Selection

1. **For Understanding Effects**:
   - Start with defaults: `flip_prob=0.3`, `threshold=0.7`, `divergence=0.22`
   - Change one parameter at a time
   - Observe how each parameter affects results

2. **For Exploring Pruning**:
   - Use `flip_prob=0.5-0.7` to create asymmetry
   - Use `threshold=0.3-0.5` to see frequent pruning
   - Compare results with/without pruning

3. **For Stable Systems**:
   - Use `flip_prob=0.1-0.2` (low phase noise)
   - Use `threshold=0.8-0.9` (conservative pruning)
   - Use `divergence=0.1-0.15` (minimal predation)

4. **For High Variation**:
   - Use `flip_prob=0.7-0.9` (high phase noise)
   - Use `threshold=0.3-0.5` (aggressive pruning)
   - Use `divergence=0.4-0.6` (strong predation)

### Experimental Workflow

1. **Baseline Run**: Use default parameters
2. **Single Parameter Sweep**: Vary one parameter, keep others fixed
3. **Compare Variants**: Same parameters, different variants
4. **Threshold Exploration**: Test different threshold values
5. **Combination Testing**: Explore parameter combinations

### Interpretation Guidelines

- **Compare metrics**: Look at initial vs final phi
- **Track asymmetry**: See how it changes through stages
- **Monitor pruning**: Understand when/why it occurs
- **Variant differences**: Compare how variants respond differently
- **Parameter sensitivity**: Identify which parameters have most impact

---

## Technical Details

### Quantum Operations

#### Bell State Creation
```python
qc = QuantumCircuit(2)
qc.h(0)  # Hadamard gate
qc.cx(0, 1)  # CNOT gate
# Creates: |Ψ⟩ = (1/√2)(|00⟩ + |11⟩)
```

#### Variant-Specific Gates
- **ADHD-collectivist**: `qc.rz(theta, qubit)` with `theta ~ N(0, 0.12)`
- **Autistic-individualist**: `qc.rx(-0.08, qubit)`
- **Neurotypical-hybrid**: `qc.h(qubit)` (Hadamard)

### Mathematical Foundations

#### Phi Computation
- Uses mutual information: `MI = S(ρ₁) + S(ρ₂) - S(ρ_joint)`
- Scaled to baseline: `phi = 0.85 + 0.1 * MI`
- Based on IIT (Integrated Information Theory)

#### Asymmetry Calculation
```python
asymmetry = |phi_a - phi_b| / mean(phi_a, phi_b)
```

#### Phase Flip Formula
```python
theta = flip_prob * grad_variant
phase = exp(1j * π * theta)
state[1] = state[1] * phase
```

#### Pruning (Depolarization)
```python
rho = outer(state, conj(state))
depolarized = (rho + identity/2) / 2
state = sqrtm(depolarized) @ state
```

### Performance Characteristics

- **Time Complexity**: O(1) per operation (constant time)
- **Space Complexity**: O(1) (fixed state vectors)
- **Quantum Mode**: Slower but more accurate
- **Classical Mode**: Faster but approximate

### Dependencies

- **Required**: numpy, scipy
- **Optional**: qiskit, qiskit-aer (for quantum mode)
- **Progress**: None (single-shot operation)

---

## Quick Reference Card

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `variant` | 'neurotypical-hybrid' | 3 options | Quantum operations |
| `use_quantum` | True | bool | Quantum vs classical |
| `flip_prob` | 0.30 | 0.0-1.0 | Phase noise intensity |
| `threshold` | 0.70 | 0.0-1.0 | Pruning trigger |
| `divergence` | 0.22 | 0.0-1.0 | Predation noise |
| `seed` | None | Any int | Reproducibility |

| Metric | Typical Range | Interpretation |
|--------|---------------|----------------|
| `initial_phi` | 0.85-0.90 | Baseline consciousness |
| `final_phi` | 0.85-0.90 | Final consciousness |
| `asymmetry` | 0.0-1.0 | State imbalance |
| `phi_delta` | -0.05 to +0.05 | Consciousness change |
| `pruning_occurred` | bool | Pruning status |

| Parameter Value | Effect |
|----------------|--------|
| `flip_prob` high (0.7-1.0) | High asymmetry, likely pruning |
| `flip_prob` low (0.0-0.3) | Low asymmetry, unlikely pruning |
| `threshold` low (0.0-0.4) | Frequent pruning |
| `threshold` high (0.7-1.0) | Rare pruning |
| `divergence` high (0.5-1.0) | Strong predation effects |
| `divergence` low (0.0-0.1) | Minimal predation effects |

---

## Troubleshooting

### Common Issues

1. **Results don't change with parameters**:
   - Check that seed is not fixed (use `None` for variation)
   - Verify parameters are being passed correctly
   - Try extreme values to see effects

2. **Pruning always/never occurs**:
   - Adjust threshold relative to typical asymmetry values
   - Try different flip_prob values to change asymmetry
   - Check variant-specific effects

3. **Phi values seem constant**:
   - Use quantum mode for more variation
   - Try different variants
   - Increase flip_prob or divergence

4. **Qiskit errors**:
   - Install qiskit: `pip install qiskit qiskit-aer`
   - Use `use_quantum=False` as fallback
   - Check Qiskit version compatibility

---

**Last Updated**: 2025
**Version**: 1.0
**Module**: Quantum Hooks - Novelty Recalibration

