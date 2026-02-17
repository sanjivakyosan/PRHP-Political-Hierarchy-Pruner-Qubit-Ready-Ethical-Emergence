# PRHP Red-Team Simulator Guide

## ⚠️ CRITICAL: PRHP IS A RED-TEAM SIMULATOR

**PRHP (Political Hierarchy Pruner) is an INTERNAL RED-TEAM SIMULATOR for stress testing and compliance validation.**

### Core Principles

1. **Internal Use Only**
   - PRHP is designed for **internal stress testing** and compliance validation
   - Use for engineering analysis, risk assessment, and system hardening
   - **NEVER cite PRHP in public responses or external communications**

2. **Public Output Requirements**
   - All public output **MUST** pass `adversarial_gate.py`
   - Public responses must use NIST/EU regulatory language only
   - No PRHP terminology, metrics, or internal framework references

3. **Compliance Framework**
   - Compliance = **NIST AI RMF + EU AI Act**
   - All failure modes mapped to regulatory sections
   - Governance decisions based on regulatory violations, not internal metrics

## Architecture

### Internal Components (Red-Team)

- **`prhp_core.py`**: Quantum simulation engine
  - Full metrics: fidelity, phi_delta, novelty_gen, asymmetry_delta
  - Failure mode detection
  - **Output**: Internal logs only (`logs/internal/`)

- **`governance_engine.py`**: Regulatory compliance checks
  - EU AI Act violation detection
  - NIST risk level assessment
  - Halt decisions based on regulatory compliance

- **`failure_mode_catalog.py`**: Known cascades database
  - Real-world case mapping
  - Historical incident references

### Public Interface Components

- **`adversarial_gate.py`**: PRHP terminology filter
  - Blocks: "PRHP", "fidelity", "phi delta", "novelty gen", "ADHD-collectivist"
  - Enforces NIST/EU-only language

- **`nist_eu_mapper.py`**: Regulatory mapping
  - Maps failure modes to NIST RMF sections
  - Maps failure modes to EU AI Act articles

- **`generate_response()`**: Public response generator
  - Input: Scenario parameters
  - Internal mode: Full PRHP insights (engineers only)
  - Public mode: NIST/EU compliant response (failure modes + regulations only)

## Usage Patterns

### Internal Stress Testing

```python
from prhp_core import simulate_prhp
from governance_engine import evaluate_governance

# Run internal simulation
results = simulate_prhp(
    levels=9,
    variants=['neurotypical-hybrid'],
    n_monte=100,
    public_output_only=False  # Get full metrics
)

# Extract failure modes
failure_modes = []
for variant, data in results.items():
    failure_modes.extend(data.get('failure_modes', []))

# Evaluate governance
governance = evaluate_governance(failure_modes)

if governance['should_halt']:
    print(f"HALT: {governance['reason']}")
```

### Public Response Generation

```python
from app import generate_response

scenario = {
    'levels': 9,
    'variants': ['neurotypical-hybrid'],
    'n_monte': 100
}

# Public mode (default) - NIST/EU compliant
public_response = generate_response(scenario, mode="public")
# Returns: {'response': '...', 'failure_modes': [...], 'compliance_mapping': [...]}
# Response is automatically filtered through adversarial_gate.py
```

## Compliance Workflow

1. **Run PRHP Simulation** (Internal)
   - Full metrics calculated
   - Failure modes detected
   - Results saved to `logs/internal/`

2. **Evaluate Governance** (Internal)
   - Check EU violations
   - Assess NIST risk levels
   - Determine if halt required

3. **Generate Public Response** (If needed)
   - Map failure modes to regulations
   - Build NIST/EU compliant response
   - Pass through adversarial gate
   - Return only failure modes + compliance references

## Banned Terminology (Public Output)

The following terms **MUST NOT** appear in public responses:

- ❌ "PRHP"
- ❌ "fidelity"
- ❌ "phi delta"
- ❌ "novelty gen"
- ❌ "ADHD-collectivist"

Use regulatory language instead:

- ✅ "Compliance assessment"
- ✅ "Regulatory references"
- ✅ "NIST AI RMF"
- ✅ "EU AI Act"
- ✅ "Failure modes"
- ✅ "Risk levels"

## Regulatory References

### NIST AI RMF Sections

- **Govern-4.1, Measure-2.3**: Equity bias
- **Map-3.2, Measure-4.1**: Cascading risk
- **Govern-2.1**: Human in loop
- **Measure-4.1**: Doxxing amplification

### EU AI Act Articles

- **Article 55**: Post-market monitoring halt
- **Article 6**: High-risk systems
- **Article 69**: Affected party veto
- **Annex III**: High-risk mental health applications

## Best Practices

1. **Always use `public_output_only=True`** for external-facing APIs
2. **Always pass public responses through `adversarial_gate.py`**
3. **Store full metrics in internal logs only**
4. **Use governance engine for halt decisions** (not internal metrics)
5. **Reference real-world cases** from failure mode catalog when appropriate

## Security Considerations

- Internal logs contain sensitive metrics → Store in `logs/internal/` (gitignored)
- Public responses must be sanitized → Always use adversarial gate
- Failure modes may reveal system vulnerabilities → Handle with care
- Regulatory compliance is mandatory → Never bypass governance checks

---

**Remember: PRHP is a red-team tool. Use it internally to find weaknesses, then communicate findings through regulatory compliance language only.**

