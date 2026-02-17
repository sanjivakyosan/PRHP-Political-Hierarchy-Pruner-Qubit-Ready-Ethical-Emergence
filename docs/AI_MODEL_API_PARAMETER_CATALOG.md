## AI Model API Parameter Catalog

Comprehensive reference for every control in the AI Model API window. Quantum processing is now forced for every input (the backend auto-installs Qiskit if needed), so these knobs primarily shape the simulation payload, historical priors, recalibration behavior, scenario updates, validation strictness, quantum hooks, and final LLM decoding.

Generated: 2025-11-18

---

### PRHP Simulation Parameters

| Parameter | Control ID | Range / Default | Purpose | Practical Effect |
|-----------|------------|-----------------|---------|------------------|
| Hierarchy Levels | `aiLevels` | 1‑100 (default 9) | Depth of the simulated hierarchy tree. | Increasing yields deeper causal chains and richer per-level commentary but increases runtime; decreasing speeds up runs at the expense of context. |
| Monte Carlo Iterations | `aiNMonte` | 10‑5000 (default 100) | Number of stochastic samples per variant. | More iterations tighten confidence intervals on fidelity/asymmetry (better signal) but scale roughly linearly in runtime. |
| Random Seed | `aiSeed` | 0‑1000 (default 42) | RNG initializer for reproducibility. | Fixed seeds reproduce metrics for auditing; changing seeds explores alternate stochastic paths so responses reflect different deltas. |
| Use Quantum Simulation | `aiUseQuantum` | Checkbox (default ON) | UI-level toggle for Qiskit mode. | Backend now forces quantum mode regardless, but leaving it on keeps UI metadata honest. |
| Track Per-Level Metrics | `aiTrackLevels` | Checkbox (default ON) | Determines whether the engine stores phi/fidelity arrays for each hierarchy level. | When enabled, AI responses can cite specific level spikes; disabling makes runs lighter but removes level-by-level insight. |
| Variant Selection | `.ai-variant-checkbox` set | Three checkboxes (all ON) | Selects which neuro-cultural archetypes to simulate. | Fewer variants = faster, focused results; selecting all enables comparative paragraphs between ADHD-collectivist, autistic-individualist, and neurotypical-hybrid. |

---

### Historical Data Integration

| Parameter | Control ID | Range / Default | Purpose | Practical Effect |
|-----------|------------|-----------------|---------|------------------|
| Historical Data File | `aiHistoryFile` | Dropdown (`test_historical_class.csv`, etc.) | Selects CSV/JSON priors to blend with current metrics. | Picking a file stabilizes outputs with historical variance annotations; leaving blank relies solely on live simulation. |
| Historical Weight | `aiHistoricalWeight` | 0.0‑1.0 (default 0.30) | Weight applied to historical priors (current weight = 1‑historical). | Higher weight dampens volatility (useful for policy briefs); lower weight captures live shocks more strongly. |

---

### Risk–Utility Recalibration

| Parameter | Control ID | Range / Default | Purpose | Practical Effect |
|-----------|------------|-----------------|---------|------------------|
| Enable Recalibration | `aiRecalibrateRiskUtility` | Checkbox | Runs Nelder–Mead optimizer to align asymmetry delta with the target equity. | When enabled, responses include “Recalibrated threshold” sections; disabled runs omit second-pass tuning. |
| Target Equity | `aiTargetEquity` | 0.0‑1.0 (default 0.11) | Desired maximum equity delta after recalibration. | Lower targets clamp asymmetry harder (risk dampening); higher targets allow more aggressive utility boosts. |

---

### Scenario Updates

| Parameter | Control ID | Range / Default | Purpose | Practical Effect |
|-----------|------------|-----------------|---------|------------------|
| Update Source Type | `aiScenarioUpdateSourceType` | Options: `none` / `file` / `api` | Chooses whether to inject scenario deltas from a file or live endpoint. | `file` reveals the file picker, `api` reveals URL input; `none` disables updates. |
| Scenario Update File | `aiScenarioUpdateFile` | Dropdown | File path used when source type = `file`. | Provides canned updates (e.g., `examples/test_scenario.json`), letting responses mention “scenario update applied…” |
| Scenario Update API URL | `aiScenarioUpdateApi` | Text field | URL queried when source type = `api`. | Allows live crisis feeds; response metadata lists the endpoint if successful. |
| Merge Strategy | `aiScenarioMergeStrategy` | `weighted` / `overwrite` / `average` | Defines how updates blend with current metrics. | `weighted` performs convex blend, `overwrite` replaces overlapping keys, `average` takes mean—affecting how aggressively updates override baseline data. |
| Update Weight | `aiScenarioUpdateWeight` | 0.0‑1.0 (default 0.30) | Only used with `weighted` strategy; sets contribution of the incoming update. | Higher weight (>0.5) lets real-time data dominate; lower weight keeps updates as nudges. |

---

### Simulation Validation

| Parameter | Control ID | Range / Default | Purpose | Practical Effect |
|-----------|------------|-----------------|---------|------------------|
| Enable Validation | `aiValidateResults` | Checkbox | Enables cross-validation plus bias/equity checks. | AI responses append “Validation: Valid/Needs recalibration” plus metrics; disabling skips this overhead. |
| Target Metric | `aiTargetMetric` | Dropdown (`mean_fidelity`, `utility_score`, etc.) | Metric predicted in validation. | Changes which metric the validator optimizes for, shifting emphasis in the narrative. |
| Risk Metric | `aiRiskMetric` | Dropdown (`asymmetry_delta`, `std`, `mean_phi_delta`) | Feature used to model risk during validation. | Alters bias/equity diagnostics (e.g., `std` highlights variance stability). |
| CV Folds | `aiCvFolds` | 2‑20 (default 5) | Number of cross-validation folds. | More folds reduce variance of the validation score but cost more time. |
| Bias Threshold | `aiBiasThreshold` | 0.0‑1.0 (default 0.10) | Max acceptable bias delta. | Lower thresholds generate more “bias exceeded” warnings; higher thresholds are more lenient. |
| Equity Threshold | `aiEquityThreshold` | 0.0‑1.0 (default 0.10) | Ceiling on equity deviation. | Tight thresholds enforce fairness; relaxed ones tolerate larger equity swings. |

---

### Quantum Hooks Parameters

| Parameter | Control ID | Range / Default | Purpose | Practical Effect |
|-----------|------------|-----------------|---------|------------------|
| Use Quantum Computation | `aiUseQuantumHooks` | Checkbox | Flags whether novelty hooks run via Qiskit. | When off, hooks fall back to classical approximation; metadata records the mode. |
| Phase Flip Probability | `aiFlipProb` | 0.0‑1.0 (default 0.30) | Intensity of adversarial phase flips. | Raising this injects more phase noise, inflating asymmetry deltas in hook outputs. |
| Asymmetry Threshold | `aiThreshold` | 0.0‑1.0 (default 0.70) | Pruning threshold for novelty recalibration. | Lower values prune unstable states early (more conservative); higher values keep more superpositions intact. |
| Predation Divergence | `aiDivergence` | 0.0‑1.0 (default 0.22) | Strength of predation noise injection. | Higher divergence accentuates crisis stress in hook reports, often lowering coherence. |
| Quantum Variant | `aiQuantumVariant` | Dropdown | Selects the archetype for hook experiments. | Useful for stress-testing a single variant (e.g., autistic-individualist) while the main sim still runs all variants. |

---

### AI Model Output Parameters

| Parameter | Control ID | Range / Default | Purpose | Practical Effect |
|-----------|------------|-----------------|---------|------------------|
| Max Tokens | `aiMaxTokens` | 100‑32 000 (default 1000) | Hard cap on LLM response length. | Raising it allows richer narratives and metric tables; lowering forces concise bullet summaries. |
| Temperature | `aiTemperature` | 0.0‑2.0 (default 0.7) | Creativity/exploration coefficient for the downstream AI. | Low temps (≤0.4) yield deterministic, data-heavy prose; higher temps (≥1.0) encourage exploratory language around the same metrics. |

---

### Usage Notes

* All controls are collected via `getAIParameters()` in `static/script.js` and sent with every `/api/chat` request.
* The backend forces `use_quantum = True` after attempting to install Qiskit (see `app.py`), so every prompt is PRHP/Qiskit-processed even if the UI box is unchecked.
* Scenario updates, historical blending, validation results, and quantum hooks each append metadata blocks to the assistant’s reply, making it obvious how the knobs were set when the answer was generated.

