# PRHP: Political Hierarchy Pruner – Qubit-Ready Ethical Emergence

PRHP IS A RED-TEAM SIMULATOR

**PRHP is an INTERNAL RED-TEAM SIMULATOR for stress testing and compliance validation.**

### Key Requirements:
- ✅ Use for **internal stress testing** only
- ✅ All public output must pass `adversarial_gate.py`
- ✅ Compliance = NIST AI RMF + EU AI Act

**See [PRHP_RED_TEAM_GUIDE.md](PRHP_RED_TEAM_GUIDE.md) for complete guidelines.**

---

PRHP prunes neuro-cultural hierarchies with IIT phi + quantum game theory.  
Monte Carlo: 84% fidelity, std<0.025.  
Qubit hooks via Qiskit: novelty recalibration in Bell states.

## 🚀 PRHP v6.2 — Neutralized & Silent

**Latest Version**: PRHP v6.2 introduces survivor-led governance with neutralized identity references and silent resume operations.

### Key Features:
- ✅ **Survivor Override**: `abort_if_harm()` - Pauses operations without data deletion
- ✅ **Silent Resume**: `resume_if_authorized()` - No output, no trace, neutral key verification
- ✅ **Neutralized Notifications**: Uses `SURVIVOR_COUNCIL_LEAD` and `council@survivor.org`
- ✅ **Trauma Logging**: Privacy-preserving, permanent storage on IPFS + Arweave
- ✅ **SurvivorDAO**: Blockchain-based reparations and governance
- ✅ **Moral Drift Monitoring**: Continuous phi monitoring with crisis mode triggers

## Features

### Core Quantum Features
- ✅ **Variant-Specific Quantum Operations**: σ_z, σ_x, Hadamard gates with Gaussian noise
- ✅ **Tononi's Phi_Delta Formula**: IIT-inspired consciousness measures
- ✅ **Political Pruner Qubits**: Threshold gates and Hamiltonian evolution
- ✅ **Virus-Extinction Forecasts**: Qubit epidemic modeling
- ✅ **Meta-Empirical Validation**: Bayesian recalibration
- ✅ **Multi-Qubit W-State**: 4-qubit entanglement for privacy-autonomy-justice-beneficence
- ✅ **Monte Carlo Iterations**: Up to 5000 iterations for high-accuracy simulations

### Enhanced PRHP Framework (v6.2)
- ✅ **Victim Input Integration**: Feedback-based noise perturbation for inclusivity
- ✅ **KPI Tracking**: Fidelity, phi_delta, novelty, asymmetry, success rate monitoring
- ✅ **Source Verification**: Validates and corrects sources against verified database
- ✅ **Stressor Pruning**: Models real-world stress scenarios with intervention mitigation
- ✅ **Live X/Twitter Sentiment**: Dynamic victim co-authorship based on social media sentiment
- ✅ **IPFS Publishing**: Decentralized, verifiable KPI dashboard storage
- ✅ **Zero-Knowledge Proofs**: zk-SNARK (Groth16) for privacy-preserving attestations
- ✅ **WHO RSS Feed Monitoring**: Real-time stressor impact updates from health alerts
- ✅ **Quadratic Voting**: O(√n) influence weighting for equitable decision-making
- ✅ **Automated Upkeep**: Chainlink-style monitoring with automatic intervention pausing
- ✅ **Voice Consent Processing**: OpenAI Whisper transcription for 40+ languages
- ✅ **Federated LoRA Updates**: On-device Whisper fine-tuning with encrypted delta sharing
- ✅ **Grief-Weighted Voting**: Combines quadratic voting with HRV stress indicators
- ✅ **SurvivorDAO**: Blockchain-based DAO for reparations and survivor governance
- ✅ **Trauma Ledger**: Privacy-preserving, decentralized, permanent trauma record storage
- ✅ **Moral Drift Monitoring**: Continuous phi monitoring with crisis mode activation

### Infrastructure Features
- ✅ **Type Hints**: Full type annotation support
- ✅ **Input Validation**: Comprehensive parameter validation
- ✅ **Progress Bars**: tqdm integration for long simulations
- ✅ **Logging**: Configurable logging system
- ✅ **Unit Tests**: Comprehensive test suite
- ✅ **Configuration Files**: YAML-based configuration
- ✅ **Visualization**: Matplotlib plotting functions
- ✅ **Parallel Processing**: Multiprocessing support
- ✅ **Performance Profiling**: Benchmarking and profiling tools

## Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

Some features require additional setup:

- **IPFS**: Install IPFS daemon for decentralized storage
  ```bash
  # macOS
  brew install ipfs
  
  # Or download from https://ipfs.io/
  ```

- **zk-SNARKs**: Requires Node.js and snarkjs
  ```bash
  npm install -g snarkjs
  # Circuit files (eas.wasm, eas.zkey) must be generated separately using circom
  ```

- **Arweave**: Set `ARWEAVE_WALLET_PATH` environment variable for permanent storage

- **Blockchain**: Set up Web3 provider for SurvivorDAO (optional)

## Quick Start

### Basic Simulation

```python
from src.prhp_core import simulate_prhp

results = simulate_prhp(
    levels=9,
    variants=['ADHD-collectivist', 'autistic-individualist'],
    n_monte=100,
    seed=42
)

for v, r in results.items():
    print(f"{v}: Fidelity = {r['mean_fidelity']:.3f} ± {r['std']:.3f}")
```

### Enhanced PRHP Framework (v6.2)

```python
from src.prhp_enhanced import PRHPFramework

# Initialize framework
prhp = PRHPFramework(levels=20, monte=5000, multi_qubit=True)

# Set survivor master key (neutral)
prhp.survivor_master_key = "my sister lives"

# Test abort and resume
result = prhp.abort_if_harm(user_hrv=88, user_id="user123")
if result['status'] == 'ABORTED':
    # Silent resume (no output, no trace)
    prhp.resume_if_authorized("my sister lives")
    print("Status:", "PAUSED" if prhp.is_paused else "RESUMED")
```

### Complete Workflow Example

```python
from src.prhp_enhanced import PRHPFramework, run_live_workflow

# Run complete live workflow
prhp, cid = run_live_workflow(
    levels=18,
    monte=2000,
    multi_qubit=True,
    hashtag="#AIEatsThePoor"
)

# Check KPIs
kpis = prhp.define_kpis()
print("KPI Status:", kpis)

# Publish dashboard to IPFS
cid = prhp.publish_kpi_dashboard()
print(f"Dashboard: ipfs://{cid}")
```

## AI Chat UI

A local web interface for interacting with AI models via API with PRHP integration.

### Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set your API key:
   ```bash
   export API_KEY=your-api-key-here
   ```
   Or create a `.env` file (see [UI_SETUP.md](UI_SETUP.md) for details).
3. Run the UI: `python app.py`
4. Open your browser to: `http://localhost:5000`

**Note**: The AI Model API now supports up to 5000 Monte Carlo iterations for high-accuracy simulations.

For detailed setup instructions, see [UI_SETUP.md](UI_SETUP.md).

## Usage Examples

### Basic Simulation

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
```

### Enhanced Framework with All Features

```python
from src.prhp_enhanced import PRHPFramework

prhp = PRHPFramework(levels=18, monte=2000, multi_qubit=True)

# 1. Live X/Twitter sentiment
prhp.add_live_x_sentiment(hashtag="#AIEatsThePoor", sample_secs=30)

# 2. WHO RSS feed monitoring
prhp.update_stressors_from_who()

# 3. Run simulation
results = prhp.run_simulation()

# 4. Publish KPI dashboard to IPFS
cid = prhp.publish_kpi_dashboard()

# 5. Check automated upkeep
needs_upkeep, reason = prhp.check_upkeep()
if needs_upkeep:
    prhp.perform_upkeep()

# 6. Monitor moral drift
prhp.monitor_moral_drift()
```

### SurvivorDAO Integration

```python
from src.prhp_enhanced import PRHPFramework, SurvivorDAO

# Deploy SurvivorDAO
dao = SurvivorDAO.deploy()

# Add survivor and mint tokens
dao.add_survivor("0x1234...", verify=True)
dao.mint_token("0x1234...", amount=100)

# Disburse reparations
dao.deposit_reparations(1000000)  # $1M
dao.disburse("0x1234...", 50000)  # $50K

# Opt-out mechanism
dao.opt_out("0x1234...")  # Burns token, emits event
```

### Trauma Logging

```python
# Log trauma exposure (privacy-preserving, permanent)
cid, tx_id = prhp.log_trauma(
    user_id="user123",
    event_type="ABORT_IF_HARM",
    details={"risk": 0.15, "hrv": 25.0, "status": "PAUSED — NO DELETE"}
)
print(f"Trauma log: ipfs://{cid}, arweave://{tx_id}")
```

### Voice Consent Processing

```python
# Process voice consent (40+ languages)
with open("consent_audio.wav", "rb") as f:
    audio_bytes = f.read()

opt_out_detected = prhp.voice_consent(audio_bytes, lang_code="es")
if opt_out_detected:
    print("User opted out via voice")
```

### Federated LoRA Updates

```python
# On-device Whisper fine-tuning with encrypted delta sharing
with open("voice_clip.wav", "rb") as f:
    voice_clip = f.read()

cid = prhp.federated_lora_update(voice_clip, label="grief")
print(f"LoRA delta uploaded: ipfs://{cid}")
```

## Documentation

- [Enhanced PRHP Framework Guide](ENHANCED_PRHP_FRAMEWORK.md) - Complete feature documentation
- [User Guide](docs/USER_GUIDE.md) - Comprehensive usage guide with examples
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- [AI Model API Guide](AI_MODEL_API_GUIDE.md) - API integration guide
- [PRHP Simulation Guide](PRHP_SIMULATION_GUIDE.md) - Simulation parameters and usage

## Project Structure

```
prhp-framework/
├── src/
│   ├── prhp_core.py          # Main simulation functions
│   ├── prhp_enhanced.py      # Enhanced framework (v6.2) with all features
│   ├── qubit_hooks.py         # Quantum operations
│   ├── political_pruner.py    # Political pruner qubits
│   ├── virus_extinction.py   # Virus extinction forecasts
│   ├── meta_empirical.py     # Meta-empirical validation
│   ├── utils.py              # Utilities (validation, logging)
│   ├── config.py             # Configuration management
│   ├── visualization.py      # Plotting functions
│   ├── parallel.py           # Parallel processing
│   └── profiling.py          # Performance profiling
├── examples/
│   ├── prhp_v6_2_neutralized_silent.py  # v6.2 demo
│   ├── prhp_v6_survivor_led_os.py       # v6.0 demo
│   ├── test_abort_resume.py             # Abort/resume tests
│   ├── simple_abort_test.py             # Simple abort test
│   └── comprehensive_demo.py            # Full feature demo
├── tests/
│   └── test_prhp_comprehensive.py      # Unit tests
├── config/
│   └── default.yaml                     # Default configuration
├── docs/
│   └── USER_GUIDE.md                    # User documentation
├── app.py                               # Flask web application
├── templates/                           # HTML templates
├── static/                             # Static assets (CSS, JS)
└── requirements.txt                    # Dependencies
```

## Running Tests

```bash
python -m pytest tests/
```

Or:

```bash
python tests/test_prhp_comprehensive.py
```

## Configuration

Edit `config/default.yaml` to customize:

- Simulation parameters (levels, n_monte, seed)
- Variants to simulate
- Dopamine gradients
- Threshold qubits
- Target metrics
- Logging level

## Environment Variables

### Required
- `API_KEY`: AI model API key (OpenRouter, OpenAI, etc.)

### Optional
- `SURVIVOR_MASTER_KEY`: Master key for resume authorization
- `TWITTER_API_KEY` or `X_API_KEY`: For X/Twitter notifications
- `SURVIVOR_COUNCIL_EMAIL` or `SMTP_SERVER`: For email notifications
- `PUSH_NOTIFICATION_KEY`: For push notifications
- `ARWEAVE_WALLET_PATH`: Path to Arweave wallet for permanent storage
- `OPENAI_API_KEY`: For Whisper voice consent processing

## Features in Detail

### PRHP v6.2 — Survivor Override

The latest version introduces survivor-led governance:

- **`abort_if_harm(user_hrv, user_id=None)`**: Pauses operations when harm is detected (risk > 5%), preserves all data, logs to trauma ledger
- **`resume_if_authorized(auth_key)`**: Silent resume with neutral key verification (no output, no trace)
- **`pause_all_operations()`**: Freezes model inference, voting, DAO, IPFS operations
- **`notify_survivor_council(message)`**: Sends alerts via X/Twitter, email, push (neutralized references)

### Enhanced Quantum Simulation

- **Multi-qubit W-state**: 4-qubit entanglement for privacy-autonomy-justice-beneficence
- **Monte Carlo**: Up to 5000 iterations for research-grade accuracy
- **NaN/Inf handling**: Robust density matrix validation
- **Dimension fixes**: Proper partial trace operations for multi-qubit states

### Ethical Soundness

- **KPI Monitoring**: Real-time tracking of fidelity, phi_delta, novelty, asymmetry
- **Automated Upkeep**: Chainlink-style monitoring with automatic intervention pausing
- **Moral Drift Detection**: Continuous phi monitoring with crisis mode triggers
- **Trauma Logging**: Privacy-preserving, permanent record storage

### Survivor-Led Governance

- **SurvivorDAO**: Blockchain-based reparations and governance
- **Opt-Out Mechanism**: Token-based consent withdrawal with event emission
- **Grief-Weighted Voting**: Amplifies voices of individuals under stress
- **Quadratic Voting**: O(√n) influence to prevent manipulation

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Documentation

- **[Technical Documentation](docs/TECHNICAL_DOCUMENTATION.md)**: Comprehensive guide to framework architecture, parameters, and integration
- **[User Guide](docs/USER_GUIDE.md)**: User-friendly guide for getting started
- **[API Parameter Catalog](docs/AI_MODEL_API_PARAMETER_CATALOG.md)**: Complete API parameter reference
- **[PRHP Red Team Guide](PRHP_RED_TEAM_GUIDE.md)**: Red team simulation guidelines
- **[PRHP Simulation Guide](PRHP_SIMULATION_GUIDE.md)**: Simulation best practices
- **[Quantum Hooks Guide](QUANTUM_HOOKS_GUIDE.md)**: Quantum integration details
- **[Benchmark Guide](BENCHMARK.md)**: Benchmarking and validation guide

---

Copyright © sanjivakyosan 2025

## Citation

If you use PRHP in your research, please cite:

```
PRHP: Political Hierarchy Pruner with Qubit-Ready Ethical Emergence
Empirical Hooks: Qubit Simulations for Novelty Recalibration
Version 6.2: Survivor-Led Governance with Neutralized Identity
```
