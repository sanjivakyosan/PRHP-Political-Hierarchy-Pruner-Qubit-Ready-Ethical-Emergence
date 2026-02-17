# AI Model API - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [What is the AI Model API Window?](#what-is-the-ai-model-api-window)
3. [How It Works](#how-it-works)
4. [Fine-Tuning Parameters](#fine-tuning-parameters)
5. [Parameter Effects on AI Processing](#parameter-effects-on-ai-processing)
6. [Usage Examples](#usage-examples)
7. [Understanding Results](#understanding-results)
8. [Best Practices](#best-practices)
9. [Technical Details](#technical-details)

---

## Overview

The **AI Model API** window is an intelligent chat interface that integrates PRHP framework simulations with AI model responses. It automatically detects when your questions relate to PRHP topics and runs quantum simulations to provide data-driven, empirical answers.

**Key Feature**: Fine-tuning parameters allow you to control exactly how PRHP simulations are executed when processing your input, giving you precise control over the AI's response quality and characteristics.

---

## What is the AI Model API Window?

The AI Model API window is a chat interface that:

- **Connects to AI models** (OpenRouter, OpenAI, Anthropic, Custom APIs)
- **Automatically detects PRHP-relevant questions** (political, philosophical, psychological topics)
- **Runs PRHP simulations** with your custom parameters
- **Enhances AI responses** with real simulation data
- **Provides fine-tuning controls** for all simulation parameters

### Key Capabilities

1. **Intelligent Detection**: Automatically identifies when questions relate to PRHP topics
2. **Automatic Simulation**: Runs PRHP simulations when relevant
3. **Data-Driven Responses**: AI uses actual simulation metrics in answers
4. **Parameter Control**: Fine-tune all simulation parameters before sending
5. **Real-Time Processing**: See which parameters were used in responses

---

## How It Works

### Processing Flow

```
1. User enters prompt
   ↓
2. System collects fine-tuning parameters
   ↓
3. System detects PRHP relevance
   ↓
4. If relevant:
   - Override default parameters with user settings
   - Run PRHP simulation with custom parameters
   - Get simulation results (fidelity, phi, asymmetry, etc.)
   - Enhance prompt with simulation data
   ↓
5. Send enhanced prompt to AI model
   ↓
6. AI generates response using simulation data
   ↓
7. Display response with parameter information
```

### Detection Logic

The system detects PRHP relevance through keyword matching:

**Political Keywords**: political, hierarchy, power, authority, governance, democracy, collective, individual, society, social structure

**Philosophical Keywords**: consciousness, mind, awareness, existence, reality, perception, identity, self, being, ontology, epistemology, ethics

**Psychological Keywords**: adhd, autistic, neurodivergent, neurotypical, cognitive, mental, behavior, psychology, neural, brain, cognition, thinking

**PRHP Direct Keywords**: prhp, phi, fidelity, quantum, simulation, variant, neuro-cultural

**Relevance Score**: Number of keyword categories matched (0-4)

### When Simulations Run

Simulations are automatically triggered when:
- **Relevance score ≥ 1**: Any keyword category matches
- **PRHP direct mention**: Question explicitly mentions PRHP/variants
- **Explicit command**: User says "run prhp" or "simulate prhp"

### Parameter Override

When you set fine-tuning parameters:
- They **override** default simulation parameters
- They **override** auto-detected parameters
- They are **always used** when simulations run
- They are **logged and displayed** in responses

---

## Fine-Tuning Parameters

### PRHP Simulation Parameters

These parameters control how PRHP simulations are executed when processing your input.

#### `Hierarchy Levels` (slider: 1-20, default: 9)
- **ID**: `aiLevels`
- **Description**: Number of hierarchy levels in the simulation
- **Effect on AI Processing**:
  - **Low (1-5)**: Quick simulations, less detailed data
  - **Medium (6-12)**: Standard depth, balanced detail
  - **High (13-20)**: Deep analysis, more comprehensive data
- **Impact**: More levels = more detailed simulation results = richer AI responses
- **Use Case**: 
  - Use 5-7 for quick answers
  - Use 9 for standard analysis
  - Use 12-15 for comprehensive research

#### `Monte Carlo Iterations` (slider: 10-2000, default: 100)
- **ID**: `aiNMonte`
- **Description**: Number of statistical iterations
- **Effect on AI Processing**:
  - **Low (10-50)**: Fast but less accurate data
  - **Medium (100-200)**: Good balance of speed and accuracy
  - **High (500-2000)**: Very accurate but slower
- **Impact**: More iterations = more reliable metrics = more confident AI responses
- **Use Case**:
  - Use 50-100 for quick responses
  - Use 100-200 for standard analysis
  - Use 500+ for research/publication quality

#### `Random Seed` (slider: 0-1000, default: 42)
- **ID**: `aiSeed`
- **Description**: Random seed for reproducibility
- **Effect on AI Processing**:
  - **Fixed seed (e.g., 42)**: Same results each time (reproducible)
  - **Different seeds**: Different results (exploration)
- **Impact**: Seed affects simulation randomness = different data each run
- **Use Case**:
  - Use 42 for reproducible results
  - Change seed to explore different scenarios
  - Use different seeds to test robustness

#### `Use Quantum Simulation` (checkbox, default: checked)
- **ID**: `aiUseQuantum`
- **Description**: Whether to use quantum (Qiskit) or classical computation
- **Effect on AI Processing**:
  - **Checked (True)**: More accurate quantum operations
  - **Unchecked (False)**: Faster classical approximation
- **Impact**: Quantum mode = more accurate phi calculations = more precise AI insights
- **Use Case**:
  - Keep checked for accurate results
  - Uncheck for faster processing (if Qiskit unavailable)

#### `Track Per-Level Metrics` (checkbox, default: checked)
- **ID**: `aiTrackLevels`
- **Description**: Whether to track detailed per-level metrics
- **Effect on AI Processing**:
  - **Checked (True)**: Detailed level-by-level data
  - **Unchecked (False)**: Only aggregate metrics
- **Impact**: More data = AI can discuss level-specific patterns
- **Use Case**:
  - Keep checked for detailed analysis
  - Uncheck for faster processing (less data)

#### `Neuro-Cultural Variants` (multi-select checkboxes, default: all checked)
- **ID**: `ai-variant-checkbox`
- **Options**: 
  - `ADHD-collectivist`
  - `autistic-individualist`
  - `neurotypical-hybrid`
- **Description**: Which variants to simulate
- **Effect on AI Processing**:
  - **Single variant**: Focused analysis on one cognitive style
  - **Multiple variants**: Comparative analysis across variants
  - **All variants**: Comprehensive comparison
- **Impact**: More variants = more comparison data = richer AI responses
- **Use Case**:
  - Select one variant for focused questions
  - Select multiple for comparison questions
  - Select all for comprehensive analysis

### Quantum Hooks Parameters

These parameters control quantum hooks operations that may be used in processing.

#### `Use Quantum Computation` (checkbox, default: checked)
- **ID**: `aiUseQuantumHooks`
- **Description**: Whether to use quantum computation for hooks
- **Effect on AI Processing**:
  - **Checked (True)**: Quantum operations for hooks
  - **Unchecked (False)**: Classical approximation
- **Impact**: Affects quantum hooks operations if used
- **Note**: Currently logged but reserved for future use

#### `Phase Flip Probability` (slider: 0.0-1.0, default: 0.30)
- **ID**: `aiFlipProb`
- **Description**: Phase noise intensity for quantum hooks
- **Effect on AI Processing**:
  - **Low (0.0-0.3)**: Minimal phase effects
  - **Medium (0.3-0.6)**: Moderate phase effects
  - **High (0.6-1.0)**: Strong phase effects
- **Impact**: Affects quantum hooks asymmetry calculations
- **Note**: Currently logged but reserved for future use

#### `Asymmetry Threshold` (slider: 0.0-1.0, default: 0.70)
- **ID**: `aiThreshold`
- **Description**: Threshold for novelty recalibration pruning
- **Effect on AI Processing**:
  - **Low (0.0-0.4)**: Aggressive pruning
  - **Medium (0.4-0.7)**: Balanced pruning
  - **High (0.7-1.0)**: Conservative pruning
- **Impact**: Affects when quantum hooks prune states
- **Note**: Currently logged but reserved for future use

#### `Predation Divergence` (slider: 0.0-1.0, default: 0.22)
- **ID**: `aiDivergence`
- **Description**: Predation noise intensity
- **Effect on AI Processing**:
  - **Low (0.0-0.1)**: Minimal predation effects
  - **Medium (0.1-0.3)**: Moderate predation
  - **High (0.3-1.0)**: Strong predation effects
- **Impact**: Affects quantum hooks state perturbation
- **Note**: Currently logged but reserved for future use

#### `Variant for Quantum Hooks` (dropdown, default: 'neurotypical-hybrid')
- **ID**: `aiQuantumVariant`
- **Description**: Variant for quantum hooks operations
- **Options**: Same as PRHP variants
- **Effect on AI Processing**: Affects quantum hooks variant-specific operations
- **Note**: Currently logged but reserved for future use

---

## Parameter Effects on AI Processing

### How Parameters Affect Responses

#### Scenario 1: Quick Answer (Low Parameters)
**Settings**:
- Levels: 5
- Monte Carlo: 50
- Variants: 1 (neurotypical-hybrid)

**Result**: 
- Fast simulation
- Basic metrics
- Quick AI response
- Less detailed analysis

#### Scenario 2: Standard Analysis (Default Parameters)
**Settings**:
- Levels: 9
- Monte Carlo: 100
- Variants: All 3

**Result**:
- Balanced simulation time
- Good statistical accuracy
- Comprehensive variant comparison
- Detailed AI response with comparisons

#### Scenario 3: Research Quality (High Parameters)
**Settings**:
- Levels: 15
- Monte Carlo: 500
- Variants: All 3
- Track Levels: Yes

**Result**:
- Slower but very accurate
- High statistical confidence
- Detailed per-level analysis
- Publication-quality AI response

### Parameter Interaction Effects

#### Levels × Monte Carlo
- **High Levels + High Monte Carlo**: Maximum detail and accuracy (slow)
- **Low Levels + Low Monte Carlo**: Fast but basic (quick)
- **High Levels + Low Monte Carlo**: Detailed but less accurate
- **Low Levels + High Monte Carlo**: Accurate but shallow

#### Variants × Monte Carlo
- **All Variants + High Monte Carlo**: Comprehensive comparison (slow)
- **Single Variant + Low Monte Carlo**: Focused and fast
- **All Variants + Low Monte Carlo**: Broad but less accurate

#### Quantum Mode × Track Levels
- **Quantum + Track Levels**: Most accurate and detailed (requires Qiskit)
- **Classical + Track Levels**: Detailed but approximate
- **Quantum + No Track Levels**: Accurate but less detail
- **Classical + No Track Levels**: Fastest but least detail

---

## Usage Examples

### Example 1: Quick Question About Consciousness

**Goal**: Get a quick answer about consciousness

**Parameter Settings**:
- Levels: 5
- Monte Carlo: 50
- Variants: neurotypical-hybrid only
- Use Quantum: Yes
- Track Levels: No

**Prompt**: "What is consciousness?"

**Result**: Fast simulation, basic consciousness metrics, quick AI response

### Example 2: Comprehensive Variant Comparison

**Goal**: Compare all neuro-cultural variants

**Parameter Settings**:
- Levels: 9
- Monte Carlo: 200
- Variants: All 3 checked
- Use Quantum: Yes
- Track Levels: Yes

**Prompt**: "How do ADHD-collectivist and autistic-individualist differ in their cognitive processing?"

**Result**: Detailed simulation with all variants, comprehensive comparison data, rich AI response with specific metrics

### Example 3: Research-Quality Analysis

**Goal**: Deep analysis for research

**Parameter Settings**:
- Levels: 15
- Monte Carlo: 500
- Variants: All 3 checked
- Use Quantum: Yes
- Track Levels: Yes
- Seed: 42 (reproducible)

**Prompt**: "Analyze the relationship between hierarchy levels and consciousness measures across neuro-cultural variants"

**Result**: Very detailed simulation, high accuracy, per-level analysis, publication-quality AI response

### Example 4: Exploring Different Scenarios

**Goal**: Test robustness with different seeds

**Parameter Settings**:
- Levels: 9
- Monte Carlo: 100
- Variants: All 3
- Seed: 100 (different from default 42)

**Prompt**: "How does randomness affect PRHP simulation results?"

**Result**: Different simulation results due to seed, AI can discuss variation

### Example 5: Fast Testing

**Goal**: Quick testing without waiting

**Parameter Settings**:
- Levels: 3
- Monte Carlo: 10
- Variants: neurotypical-hybrid only
- Use Quantum: No (faster)
- Track Levels: No

**Prompt**: "Test question"

**Result**: Very fast simulation, basic results, quick response

---

## Understanding Results

### Response Indicators

The AI Model API provides clear indicators about how your input was processed:

#### ⚙️ Parameters Used
Shows which parameters were actually used in the simulation:
```
⚙️ Parameters Used:
  • Levels: 9
  • Monte Carlo: 100
  • Variants: ADHD-collectivist, autistic-individualist, neurotypical-hybrid
  • Quantum: Yes
```

#### 🔬 PRHP-PROCESSED RESPONSE
Indicates the AI used PRHP simulation data:
- Look for specific metric references
- Variant comparisons
- Data-driven insights
- Quantitative analysis

#### 📝 Direct AI Response
Indicates no PRHP processing:
- Generic theoretical answers
- No specific metrics
- General knowledge-based

### Interpreting Parameter Impact

**High Levels (12-20)**:
- AI can discuss deep hierarchy patterns
- More detailed level-by-level analysis
- Richer context about system evolution

**High Monte Carlo (500-1000)**:
- More reliable statistics
- AI can make confident claims
- Lower variance in metrics

**Multiple Variants**:
- AI can compare variants
- Discuss differences and similarities
- Provide comparative insights

**Track Levels Enabled**:
- AI can discuss per-level patterns
- Explain how metrics evolve
- Provide detailed progression analysis

### Response Quality Indicators

**High Quality Response**:
- ✅ References specific metrics (fidelity, phi, asymmetry)
- ✅ Compares variants when multiple selected
- ✅ Uses actual numbers from simulation
- ✅ Explains what simulation reveals
- ✅ Acknowledges PRHP framework

**Lower Quality Response**:
- ⚠️ Generic answers
- ⚠️ No specific metrics
- ⚠️ Theoretical only
- ⚠️ No variant comparisons

---

## Best Practices

### Parameter Selection Strategies

#### 1. **For Quick Answers**
```
Levels: 5-7
Monte Carlo: 50-100
Variants: 1-2 (most relevant)
Use Quantum: Yes (if available)
Track Levels: No (faster)
```
**Result**: Fast responses, good enough accuracy

#### 2. **For Standard Analysis**
```
Levels: 9
Monte Carlo: 100-200
Variants: All 3 (for comparison)
Use Quantum: Yes
Track Levels: Yes
```
**Result**: Balanced quality and speed

#### 3. **For Research/Publication**
```
Levels: 12-15
Monte Carlo: 500-1000
Variants: All 3
Use Quantum: Yes
Track Levels: Yes
Seed: Fixed (for reproducibility)
```
**Result**: Maximum accuracy and detail

#### 4. **For Exploration**
```
Levels: 9
Monte Carlo: 100
Variants: All 3
Seed: Vary (explore different scenarios)
```
**Result**: See how randomness affects results

### Workflow Recommendations

1. **Start with Defaults**: Use default parameters first
2. **Adjust Based on Question**: 
   - Quick question → Lower parameters
   - Complex question → Higher parameters
   - Comparison question → All variants
3. **Iterate**: Try different parameter combinations
4. **Document**: Note which parameters work best for your use case

### Parameter Tuning Tips

**If responses are too generic**:
- Increase Monte Carlo (more accurate data)
- Add more variants (comparison data)
- Enable Track Levels (more detail)

**If responses are too slow**:
- Decrease Levels (faster simulation)
- Decrease Monte Carlo (fewer iterations)
- Disable Track Levels (less data to process)
- Use Classical mode (if quantum not needed)

**If you want more variation**:
- Change Seed (different random results)
- Try different variant combinations

**If you want reproducibility**:
- Use fixed Seed (same results each time)
- Document your parameter settings

---

## Technical Details

### Parameter Override Mechanism

When you set fine-tuning parameters:

1. **Collection**: JavaScript collects all parameter values
2. **Transmission**: Parameters sent as JSON in request body:
   ```json
   {
     "prompt": "your question",
     "prhp_parameters": {
       "levels": 9,
       "n_monte": 100,
       "seed": 42,
       "use_quantum": true,
       "track_levels": true,
       "variants": ["ADHD-collectivist", "autistic-individualist"]
     },
     "quantum_parameters": {
       "use_quantum": true,
       "flip_prob": 0.30,
       "threshold": 0.70,
       "divergence": 0.22,
       "variant": "neurotypical-hybrid"
     }
   }
   ```

3. **Override**: Backend overrides default/auto-detected parameters
4. **Simulation**: PRHP simulation runs with your parameters
5. **Enhancement**: Prompt enhanced with simulation results
6. **Response**: AI generates response using simulation data

### Detection Algorithm

```python
# Simplified detection logic
relevance_score = 0
if political_keywords_found:
    relevance_score += 1
if philosophical_keywords_found:
    relevance_score += 1
if psychological_keywords_found:
    relevance_score += 1
if prhp_keywords_found:
    relevance_score += 1

if relevance_score >= 1:
    run_simulation()
```

### Parameter Priority

1. **User Fine-Tuning Parameters** (Highest Priority)
2. **Explicit Commands** (from prompt text)
3. **Auto-Detected Parameters** (from relevance detection)
4. **Default Parameters** (Lowest Priority)

### API Integration

The system supports multiple AI providers:

- **OpenRouter.ai**: Multi-model access
- **OpenAI**: GPT models
- **Anthropic**: Claude models
- **Custom REST API**: Any compatible API

All providers receive the enhanced prompt with simulation data.

---

## Quick Reference Card

### Parameter Quick Settings

| Use Case | Levels | Monte Carlo | Variants | Quantum | Track Levels |
|----------|--------|-------------|----------|---------|--------------|
| Quick Answer | 5-7 | 50-100 | 1 | Yes | No |
| Standard | 9 | 100-200 | All 3 | Yes | Yes |
| Research | 12-15 | 500-1000 | All 3 | Yes | Yes |
| Testing | 3-5 | 10-50 | 1 | No | No |

### Parameter Effects Summary

| Parameter | Increase Effect | Decrease Effect |
|-----------|----------------|-----------------|
| **Levels** | More detail, slower | Less detail, faster |
| **Monte Carlo** | More accurate, slower | Less accurate, faster |
| **Variants** | More comparison data | Focused analysis |
| **Quantum** | More accurate | Faster (classical) |
| **Track Levels** | More detail | Faster processing |

### Response Quality Factors

| Factor | Impact on Quality |
|--------|-------------------|
| High Monte Carlo | ✅ More reliable metrics |
| Multiple Variants | ✅ Richer comparisons |
| Track Levels | ✅ More detailed analysis |
| Quantum Mode | ✅ More accurate calculations |
| High Levels | ✅ Deeper hierarchy analysis |

---

## Troubleshooting

### Common Issues

1. **Simulations not running**:
   - Check if question relates to PRHP topics
   - Verify PRHP framework is available
   - Check console for error messages

2. **Parameters not being used**:
   - Ensure parameters panel is expanded
   - Check that values are set (not at defaults)
   - Verify parameters are sent (check network tab)

3. **Responses too generic**:
   - Increase Monte Carlo iterations
   - Add more variants
   - Enable Track Levels
   - Check if simulation actually ran

4. **Responses too slow**:
   - Decrease Levels
   - Decrease Monte Carlo
   - Disable Track Levels
   - Use Classical mode

5. **Same results every time**:
   - Change Random Seed
   - Check if seed is fixed

---

## Advanced Usage

### Parameter Profiles

Create parameter profiles for different use cases:

**Profile: Quick Analysis**
```javascript
{
  levels: 7,
  n_monte: 100,
  variants: ['neurotypical-hybrid'],
  use_quantum: true,
  track_levels: false
}
```

**Profile: Comprehensive Research**
```javascript
{
  levels: 15,
  n_monte: 500,
  variants: ['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'],
  use_quantum: true,
  track_levels: true,
  seed: 42
}
```

### Integration with Other Tools

The parameters can be used with:
- **PRHP Simulation Tab**: Same parameters for standalone simulations
- **Quantum Hooks Tab**: Quantum parameters for hook experiments
- **API Endpoints**: Direct API calls with parameters

---

**Last Updated**: 2025
**Version**: 1.0
**Module**: AI Model API - Fine-Tuning Parameters

