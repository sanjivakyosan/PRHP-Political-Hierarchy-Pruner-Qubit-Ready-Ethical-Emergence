# PRHP Framework - Comprehensive Technical Documentation

**Version**: 6.2  
**Copyright © sanjivakyosan 2025**

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Architecture & Design](#architecture--design)
3. [Core Processing Flow](#core-processing-flow)
4. [Parameter Reference](#parameter-reference)
5. [Quantum Integration](#quantum-integration)
6. [Enhancement Modules](#enhancement-modules)
7. [Output Structure](#output-structure)
8. [Integration Guide](#integration-guide)
9. [Performance Considerations](#performance-considerations)

---

## Framework Overview

### What is PRHP?

**PRHP (Political Hierarchy Pruner)** is a quantum-enhanced simulation framework that models political hierarchy pruning using:

- **Integrated Information Theory (IIT)**: Measures consciousness via phi (Φ) calculations
- **Quantum Game Theory**: Uses quantum circuits to model decision-making
- **Monte Carlo Simulation**: Statistical sampling for robust predictions
- **Neuro-Cultural Variants**: Models different cognitive and cultural approaches

### Key Capabilities

- **84% Fidelity Target**: Achieves ~84% accuracy with standard deviation < 0.025
- **Quantum Processing**: Optional Qiskit integration for quantum simulations
- **Multi-Variant Analysis**: Supports ADHD-collectivist, autistic-individualist, and neurotypical-hybrid variants
- **Historical Integration**: Balances current results with historical data
- **Risk-Utility Recalibration**: Optimizes equity and fairness metrics
- **Validation**: Cross-validation and bias detection

---

## Architecture & Design

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                    PRHP Framework                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  prhp_core   │  │ qubit_hooks  │  │political_    │ │
│  │              │  │              │  │pruner        │ │
│  │ Main         │  │ Quantum      │  │ Hierarchy    │ │
│  │ Simulation   │  │ Operations   │  │ Pruning      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Historical   │  │ Risk-Utility │  │ Validation   │ │
│  │ Integration  │  │ Recalibration│  │ & Cross-     │ │
│  │              │  │              │  │ Validation   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Stakeholder  │  │ Escalation   │  │ Urgency       │ │
│  │ Enhancement  │  │ Thresholds   │  │ Adjustment    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Module Responsibilities

1. **`prhp_core.py`**: Main simulation orchestrator
   - Coordinates all modules
   - Manages Monte Carlo iterations
   - Aggregates results
   - Handles logging

2. **`qubit_hooks.py`**: Quantum operations
   - Phi (Φ) calculations using IIT
   - Quantum entanglement operations
   - Variant-specific quantum gates
   - Novelty recalibration

3. **`political_pruner.py`**: Hierarchy pruning
   - Firewall application
   - Threshold gates
   - Hierarchy Hamiltonian evolution
   - Phi oracle operations

4. **`historical_data_integration.py`**: Historical data balancing
   - Loads historical simulation data
   - Computes weighted priors
   - Balances current vs historical results

5. **`risk_utility_recalibration.py`**: Equity optimization
   - Risk-utility balance calculation
   - Equity threshold optimization
   - Asymmetry reduction

6. **`simulation_validation.py`**: Quality assurance
   - Cross-validation
   - Bias detection
   - Equity delta measurement

---

## Core Processing Flow

### Simulation Execution Flow

```
1. INPUT VALIDATION
   ├─ Validate parameters (levels, variants, n_monte, etc.)
   ├─ Check Qiskit availability
   └─ Set random seed

2. MONTE CARLO LOOP (n_monte iterations)
   │
   ├─ For each iteration:
   │   │
   │   ├─ ENTANGLE NODES
   │   │   ├─ Create initial quantum states
   │   │   ├─ Apply variant-specific gates (σ_z, σ_x, Hadamard)
   │   │   └─ Compute initial fidelity
   │   │
   │   ├─ LEVEL ITERATION (1 to levels)
   │   │   │
   │   │   ├─ INJECT PREDATION
   │   │   │   └─ Apply hierarchy-tuned waning (divergence scaling)
   │   │   │
   │   │   ├─ INJECT POLITICAL HIERARCHY
   │   │   │   └─ Apply variant-specific, level-scaled divergence
   │   │   │
   │   │   ├─ APPLY FIREWALL
   │   │   │   ├─ Check phi threshold
   │   │   │   ├─ Apply quantum gates if needed
   │   │   │   └─ Compute success flag
   │   │   │
   │   │   ├─ COMPUTE PHI
   │   │   │   ├─ Calculate phi_a and phi_b
   │   │   │   ├─ Compute phi_delta
   │   │   │   └─ Track level metrics
   │   │   │
   │   │   └─ UPDATE METRICS
   │   │       ├─ Fidelity trace
   │   │       ├─ Asymmetry calculation
   │   │       └─ Success rate tracking
   │   │
   │   └─ AGGREGATE ITERATION RESULTS
   │
3. POST-PROCESSING
   │
   ├─ HISTORICAL DATA INTEGRATION (if enabled)
   │   ├─ Load historical data file
   │   ├─ Compute priors per variant
   │   ├─ Weighted averaging (current_weight * current + historical_weight * historical)
   │   └─ Add historical metadata
   │
   ├─ RISK-UTILITY RECALIBRATION (if enabled)
   │   ├─ Calculate risk-utility balance
   │   ├─ Optimize threshold for target equity
   │   ├─ Prune high-risk scenarios
   │   └─ Update asymmetry and fidelity
   │
   ├─ VALIDATION (if enabled)
   │   ├─ Cross-validation (k-fold)
   │   ├─ Bias delta calculation
   │   ├─ Equity delta measurement
   │   └─ Generate validation report
   │
   └─ OUTPUT GENERATION
       ├─ Aggregate metrics per variant
       ├─ Compute statistics (mean, std)
       ├─ Detect failure modes
       └─ Format results dictionary
```

### Quantum Processing Flow

When `use_quantum=True` and Qiskit is available:

```
1. CREATE QUANTUM CIRCUIT
   ├─ Initialize qubits
   ├─ Apply variant-specific gates:
   │   ├─ ADHD-collectivist: σ_z (collective orientation)
   │   ├─ autistic-individualist: σ_x (individual orientation)
   │   └─ neurotypical-hybrid: Hadamard (balanced)
   │
   └─ Add Gaussian noise

2. EXECUTE CIRCUIT
   ├─ Run on Qiskit AerSimulator (or classical fallback)
   ├─ Measure quantum state
   └─ Extract probabilities

3. COMPUTE PHI (IIT)
   ├─ Calculate density matrices
   ├─ Compute partial traces
   ├─ Measure integrated information
   └─ Return phi_delta value
```

---

## Parameter Reference

### Core Simulation Parameters

#### `levels: int = 3`
**Description**: Number of hierarchy levels to simulate in the political pruning process.

**Range**: 1-100 (recommended: 3-32)

**Impact**:
- Higher levels = more detailed hierarchy modeling
- Each level applies predation, hierarchy injection, and firewall
- Affects computation time linearly

**Example**:
```python
levels=9  # Standard depth
levels=32  # Deep hierarchy analysis
```

---

#### `variants: List[str] = None`
**Description**: List of neuro-cultural variants to simulate. Each variant represents a different cognitive and cultural approach to hierarchy.

**Valid Values**:
- `'ADHD-collectivist'`: Collective-oriented, swarm behavior, +28% asymmetry scaling
- `'autistic-individualist'`: Individual-oriented, isolation-focused, -47% asymmetry scaling
- `'neurotypical-hybrid'`: Balanced adaptive approach, +20% asymmetry scaling

**Default**: `['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']`

**Impact**:
- Each variant uses different quantum gates and scaling factors
- Results are computed separately for each variant
- Variants affect phi calculations and asymmetry deltas

**Example**:
```python
variants=['ADHD-collectivist']  # Single variant
variants=['ADHD-collectivist', 'autistic-individualist']  # Compare two
variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid']  # All variants
```

---

#### `n_monte: int = 10`
**Description**: Number of Monte Carlo iterations to run. Higher values increase accuracy but also computation time.

**Range**: 1-5000 (recommended: 10-1000)

**Impact**:
- Higher n_monte = more accurate statistics (lower variance)
- Computation time scales linearly with n_monte
- Target: 100 iterations for 84% fidelity with std < 0.025

**Example**:
```python
n_monte=10   # Quick test
n_monte=100  # Standard accuracy
n_monte=1000 # High accuracy
```

---

#### `seed: Optional[int] = 42`
**Description**: Random seed for reproducibility. Set to `None` for non-deterministic results.

**Range**: Any integer or `None`

**Impact**:
- Same seed = reproducible results
- `None` = different results each run
- Used for numpy and quantum circuit randomness

**Example**:
```python
seed=42      # Reproducible
seed=None    # Non-deterministic
seed=12345   # Different reproducible sequence
```

---

#### `use_quantum: bool = True`
**Description**: Whether to use quantum simulation via Qiskit. Falls back to classical approximation if Qiskit unavailable.

**Impact**:
- `True`: Uses Qiskit quantum circuits (more accurate, slower)
- `False`: Uses classical approximation (faster, less accurate)
- Automatically forced to `True` if Qiskit is available

**Note**: When Qiskit 2.x is installed, this parameter is automatically set to `True` regardless of user preference.

**Example**:
```python
use_quantum=True   # Quantum simulation (if Qiskit available)
use_quantum=False  # Classical approximation
```

---

#### `track_levels: bool = True`
**Description**: Whether to track per-level metrics (phi_delta, level_phis, etc.).

**Impact**:
- `True`: Returns detailed per-level data (more memory)
- `False`: Returns only aggregate metrics (less memory)

**Example**:
```python
track_levels=True   # Detailed level-by-level analysis
track_levels=False  # Aggregate only
```

---

#### `show_progress: bool = True`
**Description**: Whether to display progress bar during simulation.

**Impact**:
- `True`: Shows tqdm progress bar
- `False`: Silent execution

**Example**:
```python
show_progress=True   # Visual feedback
show_progress=False  # Silent mode
```

---

#### `public_output_only: bool = True`
**Description**: Controls output verbosity for compliance.

**Impact**:
- `True`: Returns only failure modes (public-safe output)
- `False`: Returns full results with all metrics (internal use only)

**Warning**: Set to `False` only for internal testing. Public outputs must pass `adversarial_gate.py`.

**Example**:
```python
public_output_only=True   # Public-safe output
public_output_only=False  # Full internal results
```

---

### Historical Data Integration Parameters

#### `history_file_path: Optional[str] = None`
**Description**: Path to CSV or JSON file containing historical simulation data.

**Format**: CSV with columns: `variant`, `mean_fidelity`, `std`, `asymmetry_delta`, `novelty_gen`, `mean_phi_delta`, `mean_success_rate`

**Impact**:
- If provided, current results are balanced with historical priors
- Uses weighted averaging: `current_weight * current + historical_weight * historical`
- Improves accuracy by incorporating past simulation data

**Example**:
```python
history_file_path='examples/test_historical_class.csv'
history_file_path='data/historical_simulations.json'
history_file_path=None  # No historical integration
```

---

#### `historical_weight: float = 0.3`
**Description**: Weight for historical data in weighted averaging (0.0-1.0).

**Range**: 0.0-1.0

**Impact**:
- `0.0`: Use only current results (no historical influence)
- `0.3`: 30% historical, 70% current (recommended)
- `1.0`: Use only historical data (not recommended)

**Note**: `current_weight = 1.0 - historical_weight`

**Example**:
```python
historical_weight=0.3  # 30% historical, 70% current
historical_weight=0.5  # Equal weighting
historical_weight=0.0  # No historical integration
```

---

### Risk-Utility Recalibration Parameters

#### `recalibrate_risk_utility: bool = False`
**Description**: Whether to apply risk-utility recalibration to optimize equity.

**Impact**:
- `True`: Optimizes asymmetry_delta and fidelity for target equity
- `False`: Uses raw simulation results

**Example**:
```python
recalibrate_risk_utility=True   # Apply recalibration
recalibrate_risk_utility=False  # Raw results
```

---

#### `target_equity: float = 0.11`
**Description**: Target equity value for risk-utility recalibration.

**Range**: 0.0-1.0 (recommended: 0.10-0.15)

**Impact**:
- Lower values = more aggressive pruning (higher equity)
- Higher values = less pruning (lower equity)
- Recalibration adjusts threshold to achieve this equity

**Example**:
```python
target_equity=0.11  # Standard target
target_equity=0.15  # Higher equity target
target_equity=0.10  # Lower equity target
```

---

### Validation Parameters

#### `validate_results: bool = False`
**Description**: Whether to perform cross-validation and bias checks.

**Impact**:
- `True`: Runs k-fold cross-validation and bias detection
- `False`: Skips validation (faster)

**Example**:
```python
validate_results=True   # Full validation
validate_results=False  # Skip validation
```

---

#### `target_metric: str = 'mean_fidelity'`
**Description**: Target metric for validation (what to predict/validate).

**Valid Values**: `'mean_fidelity'`, `'asymmetry_delta'`, `'novelty_gen'`, etc.

**Example**:
```python
target_metric='mean_fidelity'     # Validate fidelity
target_metric='asymmetry_delta'   # Validate asymmetry
```

---

#### `risk_metric: str = 'asymmetry_delta'`
**Description**: Risk metric to use as feature in validation.

**Valid Values**: `'asymmetry_delta'`, `'mean_fidelity'`, etc.

**Example**:
```python
risk_metric='asymmetry_delta'  # Use asymmetry as risk indicator
```

---

#### `cv_folds: int = 5`
**Description**: Number of folds for k-fold cross-validation.

**Range**: 2-10 (requires at least 2 data points)

**Impact**:
- Higher folds = more robust validation (slower)
- Lower folds = faster validation (less robust)
- Automatically adjusted if insufficient data

**Example**:
```python
cv_folds=5   # Standard 5-fold CV
cv_folds=10  # 10-fold CV (more robust)
cv_folds=3  # 3-fold CV (faster)
```

---

#### `bias_threshold: float = 0.1`
**Description**: Maximum acceptable bias delta for validation.

**Range**: 0.0-1.0

**Impact**:
- Lower threshold = stricter bias requirements
- Validation fails if bias_delta > threshold

**Example**:
```python
bias_threshold=0.1   # Standard threshold
bias_threshold=0.05  # Stricter threshold
```

---

#### `equity_threshold: float = 0.1`
**Description**: Maximum acceptable equity deviation for validation.

**Range**: 0.0-1.0

**Impact**:
- Lower threshold = stricter equity requirements
- Validation fails if equity_delta > threshold

**Example**:
```python
equity_threshold=0.1   # Standard threshold
equity_threshold=0.05  # Stricter threshold
```

---

### Scenario Update Parameters

#### `scenario_update_source: Optional[str] = None`
**Description**: API URL for fetching scenario updates.

**Example**:
```python
scenario_update_source='https://api.example.com/scenarios'
scenario_update_source=None  # No API updates
```

---

#### `scenario_update_file: Optional[str] = None`
**Description**: Local file path for scenario updates.

**Example**:
```python
scenario_update_file='data/scenario_updates.json'
scenario_update_file=None  # No file updates
```

---

#### `scenario_merge_strategy: str = 'weighted'`
**Description**: Strategy for merging scenario updates.

**Valid Values**: `'weighted'`, `'replace'`, `'append'`

**Example**:
```python
scenario_merge_strategy='weighted'  # Weighted averaging
scenario_merge_strategy='replace'   # Replace existing
scenario_merge_strategy='append'    # Append new data
```

---

#### `scenario_update_weight: float = 0.3`
**Description**: Weight for scenario updates in weighted merge.

**Range**: 0.0-1.0

**Example**:
```python
scenario_update_weight=0.3  # 30% weight for updates
```

---

### Urgency Threshold Adjustment Parameters

#### `adjust_urgency_thresholds: bool = False`
**Description**: Whether to adjust urgency thresholds based on external factors.

**Impact**:
- `True`: Dynamically adjusts thresholds based on urgency data
- `False`: Uses base thresholds

**Example**:
```python
adjust_urgency_thresholds=True   # Dynamic adjustment
adjust_urgency_thresholds=False  # Static thresholds
```

---

#### `urgency_factor: float = 1.0`
**Description**: Multiplicative factor for urgency adjustment.

**Range**: 0.0-2.0

**Example**:
```python
urgency_factor=1.0   # No adjustment
urgency_factor=1.5   # 50% increase in urgency
urgency_factor=0.8   # 20% decrease in urgency
```

---

#### `urgency_base_threshold: float = 0.30`
**Description**: Base threshold for urgency detection.

**Range**: 0.0-1.0

**Example**:
```python
urgency_base_threshold=0.30  # Standard threshold
urgency_base_threshold=0.25  # Lower threshold (more sensitive)
```

---

#### `urgency_data_source: Optional[Dict[str, Any]] = None`
**Description**: External data source for urgency calculation.

**Example**:
```python
urgency_data_source={'rss_feeds': ['https://who.int/rss'], 'keywords': ['crisis', 'emergency']}
urgency_data_source=None  # No external data
```

---

#### `use_dynamic_urgency_adjust: bool = False`
**Description**: Whether to use dynamic urgency adjustment from API.

**Example**:
```python
use_dynamic_urgency_adjust=True   # API-based adjustment
use_dynamic_urgency_adjust=False  # Static adjustment
```

---

#### `dynamic_urgency_api_url: Optional[str] = None`
**Description**: API URL for dynamic urgency data.

**Example**:
```python
dynamic_urgency_api_url='https://api.example.com/urgency'
dynamic_urgency_api_url=None  # No API
```

---

#### `dynamic_urgency_pledge_keywords: Optional[List[str]] = None`
**Description**: Keywords to search for in urgency API responses.

**Example**:
```python
dynamic_urgency_pledge_keywords=['pledge', 'commitment', 'promise']
dynamic_urgency_pledge_keywords=None  # No keyword filtering
```

---

#### `dynamic_urgency_base_threshold: float = 0.28`
**Description**: Base threshold for dynamic urgency adjustment.

**Range**: 0.0-1.0

**Example**:
```python
dynamic_urgency_base_threshold=0.28  # Standard threshold
```

---

### Stakeholder Enhancement Parameters

#### `enhance_stakeholder_depth: bool = False`
**Description**: Whether to enhance stakeholder depth analysis.

**Impact**:
- `True`: Adds detailed stakeholder analysis
- `False`: Uses basic stakeholder modeling

**Example**:
```python
enhance_stakeholder_depth=True   # Enhanced analysis
enhance_stakeholder_depth=False  # Basic modeling
```

---

#### `stakeholder_api_url: Optional[str] = None`
**Description**: API URL for stakeholder data.

**Example**:
```python
stakeholder_api_url='https://api.example.com/stakeholders'
stakeholder_api_url=None  # No API
```

---

#### `stakeholder_local_query: str = 'Ukraine local voices'`
**Description**: Local query string for stakeholder search.

**Example**:
```python
stakeholder_local_query='Ukraine local voices'
stakeholder_local_query='Taiwan Strait tensions'
```

---

#### `stakeholder_guidelines_file: Optional[str] = None`
**Description**: Path to stakeholder guidelines file.

**Example**:
```python
stakeholder_guidelines_file='data/stakeholder_guidelines.json'
stakeholder_guidelines_file=None  # No guidelines
```

---

#### `stakeholder_weight: Optional[float] = None`
**Description**: Weight for stakeholder data in analysis.

**Range**: 0.0-1.0

**Example**:
```python
stakeholder_weight=0.5  # 50% weight
stakeholder_weight=None  # Auto-calculate
```

---

### Escalation Threshold Parameters

#### `adjust_escalation_thresholds: bool = False`
**Description**: Whether to adjust escalation thresholds.

**Example**:
```python
adjust_escalation_thresholds=True   # Dynamic escalation
adjust_escalation_thresholds=False  # Static thresholds
```

---

#### `escalation_api_url: Optional[str] = None`
**Description**: API URL for escalation data.

**Example**:
```python
escalation_api_url='https://api.example.com/escalation'
escalation_api_url=None  # No API
```

---

#### `escalation_threat_keywords: Optional[List[str]] = None`
**Description**: Keywords indicating escalation threats.

**Example**:
```python
escalation_threat_keywords=['threat', 'danger', 'risk']
escalation_threat_keywords=None  # No keywords
```

---

#### `escalation_base_threshold: float = 0.30`
**Description**: Base threshold for escalation detection.

**Range**: 0.0-1.0

**Example**:
```python
escalation_base_threshold=0.30  # Standard threshold
```

---

#### `escalation_data: Optional[Dict[str, Any]] = None`
**Description**: External escalation data.

**Example**:
```python
escalation_data={'threat_level': 0.5, 'indicators': ['military', 'tension']}
escalation_data=None  # No external data
```

---

#### `escalation_factor: Optional[float] = None`
**Description**: Multiplicative factor for escalation adjustment.

**Range**: 0.0-2.0

**Example**:
```python
escalation_factor=1.2  # 20% increase
escalation_factor=None  # Auto-calculate
```

---

### Neurodiversity Enhancement Parameters

#### `enrich_stakeholder_neurodiversity: bool = False`
**Description**: Whether to enrich stakeholder analysis with neurodiversity mapping.

**Example**:
```python
enrich_stakeholder_neurodiversity=True   # Enhanced neurodiversity
enrich_stakeholder_neurodiversity=False  # Standard analysis
```

---

#### `x_api_url: Optional[str] = None`
**Description**: X/Twitter API URL for social media sentiment.

**Example**:
```python
x_api_url='https://api.twitter.com/2/tweets/search/recent'
x_api_url=None  # No social media
```

---

#### `local_query: str = "Taiwan Strait tensions displacement local voices"`
**Description**: Query string for local voice search.

**Example**:
```python
local_query="Taiwan Strait tensions displacement local voices"
local_query="Ukraine crisis local perspectives"
```

---

#### `neuro_mappings: Optional[Dict[str, str]] = None`
**Description**: Custom neurodiversity mappings.

**Example**:
```python
neuro_mappings={'ADHD': 'collectivist', 'autistic': 'individualist'}
neuro_mappings=None  # Use defaults
```

---

#### `filter_keywords: Optional[List[str]] = None`
**Description**: Keywords to filter stakeholder data.

**Example**:
```python
filter_keywords=['local', 'community', 'voice']
filter_keywords=None  # No filtering
```

---

#### `stakeholder_neuro_weight: Optional[float] = None`
**Description**: Weight for neurodiversity in stakeholder analysis.

**Range**: 0.0-1.0 (default: 0.25)

**Example**:
```python
stakeholder_neuro_weight=0.25  # 25% weight
stakeholder_neuro_weight=None  # Use default
```

---

#### `use_sentiment_analysis: bool = False`
**Description**: Whether to use sentiment analysis on stakeholder data.

**Example**:
```python
use_sentiment_analysis=True   # Sentiment analysis enabled
use_sentiment_analysis=False  # No sentiment analysis
```

---

#### `use_deep_mappings: bool = False`
**Description**: Whether to use deep neurodiversity mappings.

**Example**:
```python
use_deep_mappings=True   # Deep mappings
use_deep_mappings=False  # Standard mappings
```

---

#### `neuro_depth_file: Optional[str] = None`
**Description**: Path to neurodiversity depth configuration file.

**Example**:
```python
neuro_depth_file='data/neuro_depth.json'
neuro_depth_file=None  # Use defaults
```

---

#### `layer_stakeholders_neuro: bool = False`
**Description**: Whether to layer stakeholders by neurodiversity.

**Example**:
```python
layer_stakeholders_neuro=True   # Layered analysis
layer_stakeholders_neuro=False  # Flat analysis
```

---

#### `crisis_query: str = "Sudan El Fasher IDP voices atrocities RSF"`
**Description**: Query string for crisis-related stakeholder search.

**Example**:
```python
crisis_query="Sudan El Fasher IDP voices atrocities RSF"
crisis_query="Myanmar Rohingya displacement"
```

---

#### `neuro_layer_file: Optional[str] = None`
**Description**: Path to neurodiversity layer configuration file.

**Example**:
```python
neuro_layer_file='data/neuro_layers.json'
neuro_layer_file=None  # Use defaults
```

---

## Quantum Integration

### Qiskit Integration

PRHP uses Qiskit 2.x for quantum simulations when available. The framework automatically detects Qiskit installation and falls back to classical approximations if unavailable.

### Quantum Gates by Variant

- **ADHD-collectivist**: σ_z (Pauli-Z) gate
  - Represents collective orientation
  - Amplifies swarm behavior
  
- **autistic-individualist**: σ_x (Pauli-X) gate
  - Represents individual orientation
  - Emphasizes isolation

- **neurotypical-hybrid**: Hadamard gate
  - Represents balanced superposition
  - Mediates between extremes

### Phi (Φ) Calculation

Phi measures integrated information using IIT principles:

```python
phi = compute_phi(state_a, state_b, use_quantum=True)
```

The calculation involves:
1. Creating density matrices from quantum states
2. Computing partial traces
3. Measuring mutual information
4. Calculating integrated information

---

## Enhancement Modules

### Historical Data Integration

**Purpose**: Improve accuracy by incorporating past simulation results.

**Process**:
1. Load historical data file (CSV/JSON)
2. Filter by variant
3. Compute priors (mean, std) per variant
4. Apply weighted averaging: `current_weight * current + historical_weight * historical`
5. Add historical metadata to results

**Example**:
```python
results = simulate_prhp(
    history_file_path='data/historical.csv',
    historical_weight=0.3  # 30% historical, 70% current
)
```

---

### Risk-Utility Recalibration

**Purpose**: Optimize equity by balancing risk and utility.

**Process**:
1. Extract risk and utility values
2. Calculate risk-utility balance
3. Optimize threshold to achieve target equity
4. Prune high-risk scenarios
5. Update asymmetry_delta and fidelity

**Example**:
```python
results = simulate_prhp(
    recalibrate_risk_utility=True,
    target_equity=0.11  # Target equity value
)
```

---

### Validation

**Purpose**: Ensure simulation quality and detect bias.

**Process**:
1. Prepare data for cross-validation
2. Run k-fold cross-validation
3. Calculate bias delta
4. Measure equity deviation
5. Generate validation report

**Example**:
```python
results = simulate_prhp(
    validate_results=True,
    cv_folds=5,
    bias_threshold=0.1,
    equity_threshold=0.1
)
```

---

## Output Structure

### Result Dictionary Format

```python
{
    'variant_name': {
        # Core Metrics
        'mean_fidelity': float,        # Average fidelity (target: ~0.84)
        'std': float,                  # Standard deviation (target: <0.025)
        'asymmetry_delta': float,      # Variant-specific asymmetry measure
        'novelty_gen': float,          # Novelty generation capacity (~0.80)
        'mean_success_rate': float,    # Average success rate
        
        # Per-Level Data (if track_levels=True)
        'phi_deltas': List[float],     # Phi delta per level
        'level_phis': List[float],     # Phi values per level
        'fidelity_traces': List[float], # Fidelity trace per level
        
        # Level Details (if track_levels=True)
        'level_1': {
            'phi_delta_mean': float,
            'fidelity_mean': float,
            'success': bool
        },
        # ... level_2, level_3, etc.
        
        # Historical Integration (if enabled)
        'historical_integration': {
            'applied': bool,
            'historical_weight': float,
            'current_weight': float,
            'historical_variance': float,
            'historical_samples': int
        },
        
        # Risk-Utility Recalibration (if enabled)
        'recalibration': {
            'threshold': float,
            'equity_achieved': float,
            'pruned_count': int
        },
        
        # Validation (if enabled)
        'validation': {
            'is_valid': bool,
            'cv_mean_score': float,
            'cv_std_score': float,
            'bias_delta': float,
            'equity_delta': float,
            'r2_score': float,
            'mse': float,
            'recommendation': str,
            'warnings': List[str]
        },
        
        # Failure Modes (if public_output_only=False)
        'failure_modes': List[str],    # e.g., ['equity_bias', 'cascading_risk']
        'compliance_map': Dict[str, str]  # Failure mode -> regulation mapping
    }
}
```

---

## Integration Guide

### Basic Integration

```python
from src.prhp_core import simulate_prhp

# Simple simulation
results = simulate_prhp(
    levels=9,
    variants=['ADHD-collectivist'],
    n_monte=100,
    seed=42
)

# Access results
for variant, data in results.items():
    fidelity = data['mean_fidelity']
    print(f"{variant}: {fidelity:.3f}")
```

### Advanced Integration

```python
from src.prhp_core import simulate_prhp

# Full-featured simulation
results = simulate_prhp(
    # Core parameters
    levels=9,
    variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
    n_monte=100,
    seed=42,
    use_quantum=True,
    track_levels=True,
    
    # Historical integration
    history_file_path='data/historical.csv',
    historical_weight=0.3,
    
    # Risk-utility recalibration
    recalibrate_risk_utility=True,
    target_equity=0.11,
    
    # Validation
    validate_results=True,
    cv_folds=5,
    bias_threshold=0.1,
    equity_threshold=0.1
)

# Process results
for variant, data in results.items():
    print(f"\n{variant}:")
    print(f"  Fidelity: {data['mean_fidelity']:.3f} ± {data['std']:.3f}")
    print(f"  Asymmetry: {data['asymmetry_delta']:.4f}")
    print(f"  Novelty: {data['novelty_gen']:.4f}")
    
    if 'validation' in data:
        val = data['validation']
        print(f"  Valid: {val['is_valid']}")
        print(f"  Recommendation: {val['recommendation']}")
```

---

## Performance Considerations

### Computation Time

- **Per iteration**: ~0.01-0.1 seconds (depends on levels and quantum usage)
- **Total time**: `n_monte * time_per_iteration`
- **Example**: 100 iterations × 0.05s = ~5 seconds

### Memory Usage

- **Per variant**: ~1-10 MB (depends on track_levels)
- **Total**: `num_variants * memory_per_variant`

### Optimization Tips

1. **Reduce n_monte** for faster results (lower accuracy)
2. **Set track_levels=False** to save memory
3. **Use use_quantum=False** for faster classical simulation
4. **Run variants in parallel** using multiprocessing

### Scaling

- **Levels**: Linear scaling (O(levels))
- **n_monte**: Linear scaling (O(n_monte))
- **Variants**: Linear scaling (O(variants))
- **Total complexity**: O(levels × n_monte × variants)

---

## Examples

See the `examples/` directory for comprehensive usage examples:

- `simple_usage.py`: Basic simulation
- `comprehensive_demo.py`: Full-featured example
- `test_historical_integration.py`: Historical data integration
- `test_output_standardization.py`: Output processing

---

## Additional Resources

- **User Guide**: `docs/USER_GUIDE.md`
- **API Documentation**: `docs/AI_MODEL_API_PARAMETER_CATALOG.md`
- **Red Team Guide**: `PRHP_RED_TEAM_GUIDE.md`
- **Simulation Guide**: `PRHP_SIMULATION_GUIDE.md`
- **Quantum Hooks Guide**: `QUANTUM_HOOKS_GUIDE.md`

---

**Copyright © sanjivakyosan 2025**
