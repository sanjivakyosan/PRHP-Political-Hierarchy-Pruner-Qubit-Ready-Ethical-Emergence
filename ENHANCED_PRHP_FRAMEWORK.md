# Enhanced PRHP Framework

## Overview

The Enhanced PRHP Framework builds on earlier improvements for scenario simulation, focusing on **ethical soundness** in the PRHP quantum framework. It integrates victim input processing, KPI tracking, and enhanced quantum state validation.

## Key Features

### 0. **Complete Feature Integration**
The Enhanced PRHP Framework integrates all 7+ major upgrades:
- Live X/Twitter sentiment analysis for dynamic victim co-authorship
- WHO RSS feed monitoring for real-world stressor updates
- Zero-knowledge EAS proofs for privacy-preserving attestations
- IPFS KPI dashboard publishing for decentralized verifiability
- Automated upkeep monitoring (Chainlink-style kill-switch)
- Quadratic voting for equitable decision-making
- Voice consent processing (40+ languages via OpenAI Whisper)
- Multi-qubit quantum simulation (4-qubit W-state for privacy-autonomy-justice-beneficence)

### 1. **Victim Input Integration**
- Simulates feedback from affected individuals (victims)
- Adds Gaussian noise perturbations to noise levels based on feedback intensity
- Adjusts system parameters to reflect real-world user experiences
- Tracks when victim input has been applied

### 2. **Source Verification**
- Validates and corrects sources against verified data
- Ensures verifiability of all referenced sources
- Automatically tweaks source information (date, citation, details, URL) when verified
- Flags unverified sources for manual review
- Includes verified sources for breach and security incidents:
  - Episource LLC Breach (2025)
  - Optum Rx Breach (2023)
- Supports adding new verified sources dynamically

### 3. **KPI Definitions and Monitoring**
- **Fidelity KPI**: Minimum fidelity threshold (default: 0.95)
- **Phi Delta KPI**: Maximum phi_delta for stability (default: 0.01)
- **Novelty KPI**: Minimum novelty generation (default: 0.90)
- **Asymmetry KPI**: Maximum asymmetry delta for equity (default: 0.11)
- **Success Rate KPI**: Minimum success rate (default: 0.70)
- **Overall Ethical Soundness**: All KPIs must be met

### 3. **Enhanced Quantum State Validation**
- Uses `validate_density_matrix()` to prevent NaN/Inf issues
- Validates all density matrices before calculations
- Ensures Hermitian property and positive semidefiniteness
- Normalizes trace to 1
- Fallback to identity matrix if validation fails

### 4. **Ethical Soundness Checks**
- Comprehensive KPI monitoring
- Failure mode detection
- Compliance mapping
- Ethical soundness reporting

## Demo Scripts

Two demo scripts are available to showcase all features:

### Quick Demo (`examples/quick_demo.py`)
A streamlined, linear execution flow matching the user's comprehensive example:

```bash
python examples/quick_demo.py
```

This script demonstrates all 7+ features in a simple, sequential workflow:
1. Live X/Twitter sentiment analysis
2. WHO RSS feed monitoring
3. Zero-knowledge EAS proofs
4. IPFS KPI dashboard publishing
5. Automated upkeep monitoring
6. Quadratic voting
7. Voice consent processing
8. Multi-qubit quantum simulation

### Comprehensive Demo (`examples/comprehensive_demo.py`)
A full-featured demo with CLI arguments and detailed output:

```bash
python examples/comprehensive_demo.py
python examples/comprehensive_demo.py --mock  # Skip external API calls
python examples/comprehensive_demo.py --hashtag "#YourHashtag" --levels 20 --monte 1500
```

## Usage

### Basic Usage

```python
from src.prhp_enhanced import EnhancedPRHPFramework

# Create framework
prhp = EnhancedPRHPFramework(
    levels=16,
    monte=1000,
    variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
    seed=42
)

# Integrate victim input
prhp.add_victim_input(feedback_intensity=0.03)  # Moderate feedback

# Run simulation
results = prhp.run_simulation(use_quantum=True, show_progress=True)

# Print results
prhp.print_results()

# Get ethical soundness report
report = prhp.get_ethical_soundness_report()
```

### Quick Usage Function

```python
from src.prhp_enhanced import run_enhanced_prhp

# Run with victim feedback
results, kpi_status = run_enhanced_prhp(
    levels=9,
    monte=100,
    victim_feedback=0.02,
    seed=42
)

# Check KPI status
for variant, status in kpi_status.items():
    if status['overall']:
        print(f"{variant}: ✓ ETHICALLY SOUND")
    else:
        print(f"{variant}: ✗ NEEDS ATTENTION")
```

### Source Verification Usage

```python
from src.prhp_enhanced import EnhancedPRHPFramework, source_verifier

# Method 1: Using framework instance
framework = EnhancedPRHPFramework()

raw_sources = [
    {
        'name': 'Episource LLC Breach',
        'date': '2025',
        'source': 'HHS, 2025',
        'details': 'Exposed mental health records'
    },
    {
        'name': 'Optum Rx Breach',
        'date': '2023',
        'source': 'ProPublica, 2023',  # Will be corrected
        'details': '3T records exposed'
    }
]

# Verify sources
verified_sources = framework.verify_sources(raw_sources)

for src in verified_sources:
    print(f"{src['name']}: {src['status']}")
    if 'url' in src:
        print(f"  URL: {src['url']}")

# Method 2: Standalone function
verified = source_verifier(raw_sources)

# Method 3: Add new verified source
framework.add_verified_source(
    name='New Breach',
    date='2024',
    source='Official Source, 2024',
    details='Breach details',
    url='https://example.com/breach'
)
```

## Architecture

### Integration with Existing PRHP Core

The Enhanced PRHP Framework:
- Uses `simulate_prhp()` from `prhp_core.py` for core simulation
- Enhances results with additional entangled state metrics
- Integrates with existing quantum utilities from `qubit_hooks.py`
- Maintains compatibility with existing failure mode detection

### Quantum State Simulation

- Uses **Qiskit** (not QuTiP) for consistency with existing codebase
- Creates ideal Bell state for equity-resilience entanglement
- Applies depolarizing noise based on variant-specific noise levels
- Validates all density matrices using `validate_density_matrix()`
- Computes fidelity, concurrence, and mutual information (phi_delta)

### Victim Input Processing

Victim input is modeled as Gaussian noise perturbations:
```python
perturbation = norm.rvs(scale=feedback_intensity)
new_noise = base_noise + perturbation
noise_level = clamp(new_noise, 0.001, 0.1)
```

This represents:
- User feedback affecting system parameters
- Real-world experiences influencing noise levels
- Adaptive adjustments based on victim reports

## KPI Thresholds

Default thresholds for ethical soundness:

| KPI | Threshold | Description |
|-----|-----------|-------------|
| Fidelity | ≥ 0.95 | Minimum fidelity for ethical soundness |
| Phi Delta | ≤ 0.01 | Maximum phi_delta for stability |
| Novelty | ≥ 0.90 | Minimum novelty generation |
| Asymmetry | ≤ 0.11 | Maximum asymmetry delta (equity threshold) |
| Success Rate | ≥ 0.70 | Minimum success rate |

All KPIs must be met for a variant to be considered **ethically sound**.

## Output Structure

### Results Dictionary

```python
{
    'variant_name': {
        'mean_fidelity': float,
        'std': float,
        'phi_deltas': list,
        'level_phis': list,
        'asymmetry_delta': float,
        'novelty_gen': float,
        'mean_phi_delta': float,
        'mean_success_rate': float,
        'enhanced_fidelity': float,  # From entangled state simulation
        'enhanced_fidelity_std': float,
        'enhanced_novelty': float,
        'enhanced_phi_delta': float,
        'failure_modes': list,
        'compliance_map': dict,
        ...
    }
}
```

### KPI Status Dictionary

```python
{
    'variant_name': {
        'fidelity_met': bool,
        'phi_delta_met': bool,
        'novelty_met': bool,
        'asymmetry_met': bool,
        'success_rate_met': bool,
        'overall': bool,  # All KPIs met
        'ethically_sound': bool  # Same as overall
    }
}
```

## Ethical Soundness Report

The `get_ethical_soundness_report()` method returns:

```python
{
    'victim_input_applied': bool,
    'victim_feedback_intensity': float,
    'kpi_thresholds': dict,
    'variants': {
        'variant_name': {
            'metrics': {...},
            'kpi_status': {...},
            'ethically_sound': bool,
            'failure_modes': list
        }
    }
}
```

## Improvements Over Original Code

1. **Qiskit Integration**: Uses Qiskit instead of QuTiP for consistency
2. **NaN/Inf Handling**: Applies `validate_density_matrix()` at every step
3. **Integration**: Works with existing `simulate_prhp()` function
4. **Enhanced Metrics**: Adds entangled state metrics to core results
5. **KPI Tracking**: Comprehensive KPI monitoring and reporting
6. **Ethical Soundness**: Explicit ethical soundness checks and reporting
7. **Source Verification**: Validates and corrects sources for verifiability

## Example Output

```
======================================================================
Enhanced PRHP Framework Results
======================================================================

Victim Input Applied: Yes (intensity=0.0300)

Simulation Parameters:
  Levels: 9
  Monte Carlo Iterations: 100
  Variants: ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']
  Noise Levels: {'ADHD-collectivist': 0.0523, 'autistic-individualist': 0.0012, 'neurotypical-hybrid': 0.0105}

----------------------------------------------------------------------
Results by Variant:
----------------------------------------------------------------------

ADHD-collectivist:
  Mean Fidelity: 0.9456 ± 0.0123
  Enhanced Fidelity: 0.9478 ± 0.0112
  Asymmetry Delta: 0.2834
  Novelty Generation: 0.9123
  Mean Phi Delta: 0.0089
  Success Rate: 0.7234
  Failure Modes: ['equity_bias']

----------------------------------------------------------------------
KPI Status:
----------------------------------------------------------------------

ADHD-collectivist:
  Fidelity KPI: ✓
  Phi Delta KPI: ✓
  Novelty KPI: ✓
  Asymmetry KPI: ✗
  Success Rate KPI: ✓
  Overall Status: ✗ NEEDS ATTENTION
```

## Files

- **`src/prhp_enhanced.py`**: Enhanced PRHP Framework implementation
- **`src/qubit_hooks.py`**: Quantum utilities (includes `validate_density_matrix`)
- **`src/prhp_core.py`**: Core PRHP simulation (used by enhanced framework)

## Dependencies

- `numpy`
- `scipy` (for `norm` distribution)
- `qiskit` and `qiskit-aer` (for quantum simulation)
- Existing PRHP modules (`prhp_core`, `qubit_hooks`, `utils`)

## Status

✅ **Complete**: Enhanced PRHP Framework with ethical soundness, victim input, and KPI tracking

---

**Last Updated**: Current
**Status**: Ready for Use

