// DOM elements (will be null until DOM is ready)
let inputBox, outputBox, sendBtn, clearBtn, clearInputBtn, loader, errorMessage, configInfo, statusIndicator;
let followUpInputBox, followUpSendBtn, clearFollowUpBtn, followUpLoader;
let paramInfoTooltip, paramInfoTitle, paramInfoRange, paramInfoPurpose, paramInfoEffect, paramInfoCloseBtn;

// Simulation elements (will be initialized in DOMContentLoaded)
let runSimulationBtn, clearSimBtn, simulationOutput, simLoader;

// Quantum hooks elements (will be initialized in DOMContentLoaded)
let runQuantumBtn, clearQuantumBtn, quantumOutput, quantumLoader;

// State
let isLoading = false;
let isSimulating = false;
let isQuantumSimulating = false;

const parameterCatalog = {
    aiLevels: {
        title: 'Hierarchy Levels',
        range: 'Range: 1-100 (default 9)',
        purpose: 'Controls depth of the simulated hierarchy tree.',
        effect: 'Higher values produce deeper causal layers and richer per-level commentary but increase runtime.'
    },
    aiNMonte: {
        title: 'Monte Carlo Iterations',
        range: 'Range: 10-5000 (default 100)',
        purpose: 'Determines how many stochastic samples are run per variant.',
        effect: 'Increasing tightens confidence intervals on fidelity/asymmetry while extending runtime roughly linearly.'
    },
    aiSeed: {
        title: 'Random Seed',
        range: 'Range: 0-1000 (default 42)',
        purpose: 'Sets RNG determinism for reproducible simulations.',
        effect: 'Keeping the seed fixed yields identical metrics; changing it explores alternate stochastic paths.'
    },
    aiUseQuantum: {
        title: 'Use Quantum Simulation',
        range: 'Checkbox (default on)',
        purpose: 'UI toggle for quantum mode.',
        effect: 'Backend forces Qiskit anyway, but leaving this on keeps metadata aligned with enforced quantum processing.'
    },
    aiTrackLevels: {
        title: 'Track Per-Level Metrics',
        range: 'Checkbox (default on)',
        purpose: 'Stores phi/fidelity arrays for each hierarchy depth.',
        effect: 'Enabling allows AI responses to cite specific level spikes; disabling reduces payload size and run time.'
    },
    aiVariants: {
        title: 'Neuro-Cultural Variants',
        range: 'Multi-select (default: all three)',
        purpose: 'Selects which archetypes (ADHD-collectivist, autistic-individualist, neurotypical-hybrid) to simulate.',
        effect: 'Fewer variants deliver faster, focused analysis; selecting all enables comparative narratives.'
    },
    aiHistoryFile: {
        title: 'Historical Data File',
        range: 'Dropdown (default test_historical_class.csv)',
        purpose: 'Adds historical priors that blend with live results.',
        effect: 'Using a file stabilizes metrics and surfaces historical variance notes; leaving blank relies solely on current data.'
    },
    aiHistoricalWeight: {
        title: 'Historical Weight',
        range: 'Range: 0.0-1.0 (default 0.30)',
        purpose: 'Controls how much historical priors influence the blended result.',
        effect: 'Higher weight dampens volatility (policy briefs); lower weight emphasizes fresh signals.'
    },
    aiRecalibrateRiskUtility: {
        title: 'Enable Risk-Utility Recalibration',
        range: 'Checkbox',
        purpose: 'Runs Nelder-Mead optimizer to balance asymmetry delta with fidelity.',
        effect: 'When enabled, outputs include recalibrated thresholds and can dampen risky variants to meet equity targets.'
    },
    aiTargetEquity: {
        title: 'Target Equity',
        range: 'Range: 0.0-1.0 (default 0.11)',
        purpose: 'Sets maximum allowable equity delta after recalibration.',
        effect: 'Lower targets enforce stricter fairness (reducing aggressive options); higher values allow more utility.'
    },
    aiScenarioUpdateSourceType: {
        title: 'Update Source Type',
        range: 'Options: none/file/api',
        purpose: 'Determines whether scenario deltas come from a file, an API, or are disabled.',
        effect: 'Switching to file/API reveals additional inputs so live or canned data can override baseline metrics.'
    },
    aiScenarioUpdateFile: {
        title: 'Scenario Update File',
        range: 'Dropdown',
        purpose: 'Pick a CSV/JSON file for scenario updates when source type = file.',
        effect: 'Applies canned crisis adjustments and surfaces “scenario update applied” metadata.'
    },
    aiScenarioUpdateApi: {
        title: 'Scenario Update API URL',
        range: 'Text input',
        purpose: 'Specifies the endpoint to fetch real-time updates when source type = api.',
        effect: 'Responses inherit live data traces and list the API in metadata.'
    },
    aiScenarioMergeStrategy: {
        title: 'Merge Strategy',
        range: 'Options: weighted/overwrite/average',
        purpose: 'Controls how incoming scenario data merges with current metrics.',
        effect: 'Weighted performs convex blend, overwrite replaces overlapping keys, average computes simple means.'
    },
    aiScenarioUpdateWeight: {
        title: 'Update Weight',
        range: 'Range: 0.0-1.0 (default 0.30)',
        purpose: 'Sets contribution of incoming data when using weighted merge.',
        effect: 'Higher weight (>0.5) lets real-time updates dominate; lower weight keeps them as gentle nudges.'
    },
    aiValidateResults: {
        title: 'Enable Simulation Validation',
        range: 'Checkbox',
        purpose: 'Runs cross-validation plus bias/equity checks on simulation outputs.',
        effect: 'Adds validation verdicts to responses; disabling skips this overhead.'
    },
    aiTargetMetric: {
        title: 'Target Metric',
        range: 'Options: mean_fidelity, utility_score, novelty_gen, mean_success_rate',
        purpose: 'Selects which metric the validator predicts.',
        effect: 'Shifts validation commentary toward the chosen metric.'
    },
    aiRiskMetric: {
        title: 'Risk Metric',
        range: 'Options: asymmetry_delta, std, mean_phi_delta',
        purpose: 'Sets the risk feature used in validation.',
        effect: 'Changing this alters which risk dimension bias/equity checks emphasize.'
    },
    aiCvFolds: {
        title: 'CV Folds',
        range: 'Range: 2-20 (default 5)',
        purpose: 'Determines folds for cross-validation.',
        effect: 'Higher folds provide more stable validation scores at increased runtime cost.'
    },
    aiBiasThreshold: {
        title: 'Bias Threshold',
        range: 'Range: 0.0-1.0 (default 0.10)',
        purpose: 'Maximum acceptable bias delta from validation.',
        effect: 'Lower thresholds flag more bias warnings; higher thresholds are lenient.'
    },
    aiEquityThreshold: {
        title: 'Equity Threshold',
        range: 'Range: 0.0-1.0 (default 0.10)',
        purpose: 'Maximum equity deviation tolerated.',
        effect: 'Tighter thresholds enforce fairness, looser thresholds allow larger equity swings.'
    },
    aiUseQuantumHooks: {
        title: 'Use Quantum Computation (Hooks)',
        range: 'Checkbox',
        purpose: 'Specifies whether novelty hooks operate quantum mechanically.',
        effect: 'When disabled, hooks use classical approximation; metadata records the mode.'
    },
    aiFlipProb: {
        title: 'Phase Flip Probability',
        range: 'Range: 0.0-1.0 (default 0.30)',
        purpose: 'Controls adversarial phase noise intensity in hooks.',
        effect: 'Higher values inflate asymmetry deltas and highlight stress scenarios.'
    },
    aiThreshold: {
        title: 'Asymmetry Threshold',
        range: 'Range: 0.0-1.0 (default 0.70)',
        purpose: 'Pruning cutoff for novelty recalibration.',
        effect: 'Lowering prunes unstable states sooner (conservative); raising preserves more superpositions.'
    },
    aiDivergence: {
        title: 'Predation Divergence',
        range: 'Range: 0.0-1.0 (default 0.22)',
        purpose: 'Determines predation noise level applied in hooks.',
        effect: 'Higher divergence emphasizes crisis stress and can reduce coherence.'
    },
    aiQuantumVariant: {
        title: 'Variant for Quantum Hooks',
        range: 'Dropdown',
        purpose: 'Selects which archetype the hook experiment targets.',
        effect: 'Allows stress-testing a single variant while the main simulation runs all variants.'
    },
    aiMaxTokens: {
        title: 'Max Tokens',
        range: 'Range: 100-32000 (default 1000)',
        purpose: 'Caps token budget for the final AI response.',
        effect: 'Increasing allows longer, detailed answers; decreasing forces concise summaries.'
    },
    aiTemperature: {
        title: 'Temperature',
        range: 'Range: 0.0-2.0 (default 0.7)',
        purpose: 'Controls creativity/exploration for the downstream AI.',
        effect: 'Low values yield deterministic, data-heavy prose; high values encourage exploratory language.'
    }
};

// Conversation state
let conversationHistory = [];
let currentConversationId = null;

// Global config (updated when config is loaded)
let globalConfig = {
    qiskit_available: false,
    qiskit_version: 0,
    prhp_available: false
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM Content Loaded - Initializing...');
    
    // IMMEDIATE: Check and clear loading state if stuck
    const immediateConfigCheck = () => {
        const configTextEl = document.getElementById('configText');
        if (configTextEl && configTextEl.textContent === 'Loading configuration...') {
            // Set a very short timeout to clear if still loading
            setTimeout(() => {
                if (configTextEl && configTextEl.textContent === 'Loading configuration...') {
                    console.warn('Immediate check: Config still loading, setting fallback');
                    configTextEl.textContent = 'Loading... (if stuck, refresh page)';
                }
            }, 2000);
        }
    };
    immediateConfigCheck();
    
    // Initialize DOM element references
    inputBox = document.getElementById('inputBox');
    outputBox = document.getElementById('outputBox');
    sendBtn = document.getElementById('sendBtn');
    clearBtn = document.getElementById('clearBtn');
    clearInputBtn = document.getElementById('clearInputBtn');
    loader = document.getElementById('loader');
    errorMessage = document.getElementById('errorMessage');
    configInfo = document.getElementById('configText');
    statusIndicator = document.getElementById('statusIndicator');
    
    // Simulation elements
    runSimulationBtn = document.getElementById('runSimulationBtn');
    clearSimBtn = document.getElementById('clearSimBtn');
    simulationOutput = document.getElementById('simulationOutput');
    simLoader = document.getElementById('simLoader');
    
    // Quantum hooks elements
    runQuantumBtn = document.getElementById('runQuantumBtn');
    clearQuantumBtn = document.getElementById('clearQuantumBtn');
    quantumOutput = document.getElementById('quantumOutput');
    quantumLoader = document.getElementById('quantumLoader');
    
    // Follow-up input elements
    followUpInputBox = document.getElementById('followUpInputBox');
    followUpSendBtn = document.getElementById('followUpSendBtn');
    clearFollowUpBtn = document.getElementById('clearFollowUpBtn');
    followUpLoader = document.getElementById('followUpLoader');
    paramInfoTooltip = document.getElementById('paramInfoTooltip');
    paramInfoTitle = document.getElementById('paramInfoTitle');
    paramInfoRange = document.getElementById('paramInfoRange');
    paramInfoPurpose = document.getElementById('paramInfoPurpose');
    paramInfoEffect = document.getElementById('paramInfoEffect');
    paramInfoCloseBtn = document.getElementById('paramInfoClose');
    
    console.log('DOM elements initialized:', {
        inputBox: !!inputBox,
        configInfo: !!configInfo,
        statusIndicator: !!statusIndicator,
        runSimulationBtn: !!runSimulationBtn,
        runQuantumBtn: !!runQuantumBtn,
        followUpInputBox: !!followUpInputBox
    });
    
    // Setup UI components first (don't wait for config)
    try {
    setupEventListeners();
    setupTabs();
        setupSliders();
        setupAIParameterSliders();
        setupParameterPanel();
        console.log('UI components initialized');
    } catch (error) {
        console.error('Error initializing UI components:', error);
    }
    
    // Allow Enter+Shift for new line, Enter alone to send
    if (inputBox) {
        inputBox.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!isLoading) {
                    sendMessage();
                }
            }
        });
    }
    
    // Follow-up input box event listeners
    if (followUpInputBox) {
        followUpInputBox.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!isLoading) {
                    sendFollowUpMessage();
                }
            }
        });
    }
    
    if (followUpSendBtn) {
        followUpSendBtn.addEventListener('click', () => {
            if (!isLoading) {
                sendFollowUpMessage();
            }
        });
    }
    
    if (clearFollowUpBtn) {
        clearFollowUpBtn.addEventListener('click', () => {
            if (followUpInputBox) {
                followUpInputBox.value = '';
                followUpInputBox.focus();
            }
        });
    }

    // Parameter info buttons
    document.querySelectorAll('.info-button').forEach((button) => {
        button.addEventListener('click', () => {
            const key = button.dataset.param;
            showParameterInfo(key);
        });
    });

    if (paramInfoCloseBtn) {
        paramInfoCloseBtn.addEventListener('click', hideParameterInfo);
    }
    if (paramInfoTooltip) {
        paramInfoTooltip.addEventListener('click', (event) => {
            if (event.target === paramInfoTooltip) {
                hideParameterInfo();
            }
        });
    }
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            hideParameterInfo();
        }
    });
    
    // Load config asynchronously (non-blocking)
    // Use setTimeout to ensure DOM is fully ready
    setTimeout(() => {
        console.log('Starting config load...');
        // Wrap in try-catch to catch any synchronous errors
        try {
            loadConfig().catch(err => {
                console.error('Config loading failed in catch:', err);
                // Ensure loading state is cleared even if loadConfig fails
                const configTextEl = document.getElementById('configText');
                if (configTextEl) {
                    if (configTextEl.textContent === 'Loading configuration...' || 
                        configTextEl.textContent.includes('Loading')) {
                        configTextEl.textContent = 'Configuration failed to load - Check console for details';
                    }
                }
            });
        } catch (syncError) {
            console.error('Synchronous error starting config load:', syncError);
            const configTextEl = document.getElementById('configText');
            if (configTextEl) {
                configTextEl.textContent = 'Configuration error - Please refresh the page';
            }
        }
    }, 200);
    
    // Fallback: Clear loading state after 8 seconds if still stuck
    setTimeout(() => {
        const configTextEl = document.getElementById('configText');
        if (configTextEl && (configTextEl.textContent === 'Loading configuration...' || configTextEl.textContent.includes('loading'))) {
            console.warn('Config loading timeout - clearing loading state');
            configTextEl.textContent = 'Configuration timeout - Server may be slow. Try refreshing.';
            const statusIndicatorEl = document.getElementById('statusIndicator');
            if (statusIndicatorEl) {
                statusIndicatorEl.classList.remove('warning', 'success');
                statusIndicatorEl.classList.add('error');
            }
        }
    }, 8000);
    
    // Note: Page scrolling is now handled automatically by the browser
});

// Setup slider value displays
function setupSliders() {
    // PRHP Simulation sliders
    const levelsSlider = document.getElementById('levels');
    const levelsValue = document.getElementById('levelsValue');
    if (levelsSlider && levelsValue) {
        levelsSlider.addEventListener('input', (e) => {
            levelsValue.textContent = e.target.value;
        });
    }
    
    const nMonteSlider = document.getElementById('nMonte');
    const nMonteValue = document.getElementById('nMonteValue');
    if (nMonteSlider && nMonteValue) {
        nMonteSlider.addEventListener('input', (e) => {
            nMonteValue.textContent = e.target.value;
        });
    }
    
    const seedSlider = document.getElementById('seed');
    const seedValue = document.getElementById('seedValue');
    if (seedSlider && seedValue) {
        seedSlider.addEventListener('input', (e) => {
            seedValue.textContent = e.target.value;
        });
    }
    
    // Quantum Hooks sliders
    const flipProbSlider = document.getElementById('flipProb');
    const flipProbValue = document.getElementById('flipProbValue');
    if (flipProbSlider && flipProbValue) {
        flipProbSlider.addEventListener('input', (e) => {
            flipProbValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    const thresholdSlider = document.getElementById('threshold');
    const thresholdValue = document.getElementById('thresholdValue');
    if (thresholdSlider && thresholdValue) {
        thresholdSlider.addEventListener('input', (e) => {
            thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    const divergenceSlider = document.getElementById('divergence');
    const divergenceValue = document.getElementById('divergenceValue');
    if (divergenceSlider && divergenceValue) {
        divergenceSlider.addEventListener('input', (e) => {
            divergenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
}

// Setup AI parameter sliders
function setupAIParameterSliders() {
    // PRHP Simulation sliders for AI
    const aiLevelsSlider = document.getElementById('aiLevels');
    const aiLevelsValue = document.getElementById('aiLevelsValue');
    if (aiLevelsSlider && aiLevelsValue) {
        aiLevelsSlider.addEventListener('input', (e) => {
            aiLevelsValue.textContent = e.target.value;
        });
    }
    
    const aiNMonteSlider = document.getElementById('aiNMonte');
    const aiNMonteValue = document.getElementById('aiNMonteValue');
    if (aiNMonteSlider && aiNMonteValue) {
        aiNMonteSlider.addEventListener('input', (e) => {
            aiNMonteValue.textContent = e.target.value;
        });
    }
    
    const aiSeedSlider = document.getElementById('aiSeed');
    const aiSeedValue = document.getElementById('aiSeedValue');
    if (aiSeedSlider && aiSeedValue) {
        aiSeedSlider.addEventListener('input', (e) => {
            aiSeedValue.textContent = e.target.value;
        });
    }
    
    // Quantum Hooks sliders for AI
    const aiFlipProbSlider = document.getElementById('aiFlipProb');
    const aiFlipProbValue = document.getElementById('aiFlipProbValue');
    if (aiFlipProbSlider && aiFlipProbValue) {
        aiFlipProbSlider.addEventListener('input', (e) => {
            aiFlipProbValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    const aiThresholdSlider = document.getElementById('aiThreshold');
    const aiThresholdValue = document.getElementById('aiThresholdValue');
    if (aiThresholdSlider && aiThresholdValue) {
        aiThresholdSlider.addEventListener('input', (e) => {
            aiThresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    const aiDivergenceSlider = document.getElementById('aiDivergence');
    const aiDivergenceValue = document.getElementById('aiDivergenceValue');
    if (aiDivergenceSlider && aiDivergenceValue) {
        aiDivergenceSlider.addEventListener('input', (e) => {
            aiDivergenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    // Historical data integration sliders
    const aiHistoricalWeightSlider = document.getElementById('aiHistoricalWeight');
    const aiHistoricalWeightValue = document.getElementById('aiHistoricalWeightValue');
    const aiCurrentWeightValue = document.getElementById('aiCurrentWeightValue');
    if (aiHistoricalWeightSlider && aiHistoricalWeightValue && aiCurrentWeightValue) {
        aiHistoricalWeightSlider.addEventListener('input', (e) => {
            const historicalWeight = parseFloat(e.target.value);
            const currentWeight = 1.0 - historicalWeight;
            aiHistoricalWeightValue.textContent = historicalWeight.toFixed(2);
            aiCurrentWeightValue.textContent = currentWeight.toFixed(2);
        });
    }
    
    // Risk-utility recalibration sliders
    const aiTargetEquitySlider = document.getElementById('aiTargetEquity');
    const aiTargetEquityValue = document.getElementById('aiTargetEquityValue');
    if (aiTargetEquitySlider && aiTargetEquityValue) {
        aiTargetEquitySlider.addEventListener('input', (e) => {
            aiTargetEquityValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    // Scenario updates sliders
    const aiScenarioUpdateWeightSlider = document.getElementById('aiScenarioUpdateWeight');
    const aiScenarioUpdateWeightValue = document.getElementById('aiScenarioUpdateWeightValue');
    if (aiScenarioUpdateWeightSlider && aiScenarioUpdateWeightValue) {
        aiScenarioUpdateWeightSlider.addEventListener('input', (e) => {
            aiScenarioUpdateWeightValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    // Simulation validation sliders
    const aiCvFoldsSlider = document.getElementById('aiCvFolds');
    const aiCvFoldsValue = document.getElementById('aiCvFoldsValue');
    if (aiCvFoldsSlider && aiCvFoldsValue) {
        aiCvFoldsSlider.addEventListener('input', (e) => {
            aiCvFoldsValue.textContent = e.target.value;
        });
    }
    
    const aiBiasThresholdSlider = document.getElementById('aiBiasThreshold');
    const aiBiasThresholdValue = document.getElementById('aiBiasThresholdValue');
    if (aiBiasThresholdSlider && aiBiasThresholdValue) {
        aiBiasThresholdSlider.addEventListener('input', (e) => {
            aiBiasThresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    const aiEquityThresholdSlider = document.getElementById('aiEquityThreshold');
    const aiEquityThresholdValue = document.getElementById('aiEquityThresholdValue');
    if (aiEquityThresholdSlider && aiEquityThresholdValue) {
        aiEquityThresholdSlider.addEventListener('input', (e) => {
            aiEquityThresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    // Scenario update source type selector
    const aiScenarioUpdateSourceType = document.getElementById('aiScenarioUpdateSourceType');
    const aiScenarioUpdateFileGroup = document.getElementById('aiScenarioUpdateFileGroup');
    const aiScenarioUpdateApiGroup = document.getElementById('aiScenarioUpdateApiGroup');
    if (aiScenarioUpdateSourceType && aiScenarioUpdateFileGroup && aiScenarioUpdateApiGroup) {
        aiScenarioUpdateSourceType.addEventListener('change', (e) => {
            const sourceType = e.target.value;
            if (sourceType === 'file') {
                aiScenarioUpdateFileGroup.style.display = 'block';
                aiScenarioUpdateApiGroup.style.display = 'none';
            } else if (sourceType === 'api') {
                aiScenarioUpdateFileGroup.style.display = 'none';
                aiScenarioUpdateApiGroup.style.display = 'block';
            } else {
                aiScenarioUpdateFileGroup.style.display = 'none';
                aiScenarioUpdateApiGroup.style.display = 'none';
            }
        });
    }
    
    // AI Model parameters sliders
    const aiMaxTokensSlider = document.getElementById('aiMaxTokens');
    const aiMaxTokensValue = document.getElementById('aiMaxTokensValue');
    if (aiMaxTokensSlider && aiMaxTokensValue) {
        aiMaxTokensSlider.addEventListener('input', (e) => {
            aiMaxTokensValue.textContent = e.target.value;
        });
    }
    
    const aiTemperatureSlider = document.getElementById('aiTemperature');
    const aiTemperatureValue = document.getElementById('aiTemperatureValue');
    if (aiTemperatureSlider && aiTemperatureValue) {
        aiTemperatureSlider.addEventListener('input', (e) => {
            aiTemperatureValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
    }
}

// Setup parameter panel toggle
function setupParameterPanel() {
    const toggleBtn = document.getElementById('toggleParamsBtn');
    const parameterPanel = document.getElementById('parameterPanel');
    const resetParamsBtn = document.getElementById('resetParamsBtn');
    
    if (toggleBtn && parameterPanel) {
        toggleBtn.addEventListener('click', () => {
            const isVisible = parameterPanel.style.display !== 'none';
            parameterPanel.style.display = isVisible ? 'none' : 'block';
            toggleBtn.textContent = isVisible ? '▼ Show Parameters' : '▲ Hide Parameters';
        });
    }
    
    if (resetParamsBtn) {
        resetParamsBtn.addEventListener('click', () => {
            resetAIParameters();
        });
    }
}

// Reset AI parameters to defaults
function resetAIParameters() {
    // PRHP Simulation defaults
    document.getElementById('aiLevels').value = 9;
    document.getElementById('aiLevelsValue').textContent = '9';
    document.getElementById('aiNMonte').value = 100;
    document.getElementById('aiNMonteValue').textContent = '100';
    document.getElementById('aiSeed').value = 42;
    document.getElementById('aiSeedValue').textContent = '42';
    document.getElementById('aiUseQuantum').checked = true;
    document.getElementById('aiTrackLevels').checked = true;
    
    // Reset variant checkboxes
    document.querySelectorAll('.ai-variant-checkbox').forEach(cb => {
        cb.checked = true;
    });
    
    // Quantum Hooks defaults
    document.getElementById('aiUseQuantumHooks').checked = true;
    document.getElementById('aiFlipProb').value = 0.30;
    document.getElementById('aiFlipProbValue').textContent = '0.30';
    document.getElementById('aiThreshold').value = 0.70;
    document.getElementById('aiThresholdValue').textContent = '0.70';
    document.getElementById('aiDivergence').value = 0.22;
    document.getElementById('aiDivergenceValue').textContent = '0.22';
    document.getElementById('aiQuantumVariant').value = 'neurotypical-hybrid';
    
    // Historical Data Integration defaults
    const historyFileSelect = document.getElementById('aiHistoryFile');
    if (historyFileSelect) {
        historyFileSelect.value = 'examples/test_historical_class.csv';
    }
    const aiHistoricalWeight = document.getElementById('aiHistoricalWeight');
    if (aiHistoricalWeight) {
        aiHistoricalWeight.value = 0.30;
        document.getElementById('aiHistoricalWeightValue').textContent = '0.30';
        document.getElementById('aiCurrentWeightValue').textContent = '0.70';
    }
    
    // Risk-Utility Recalibration defaults
    const aiRecalibrateRiskUtility = document.getElementById('aiRecalibrateRiskUtility');
    if (aiRecalibrateRiskUtility) {
        aiRecalibrateRiskUtility.checked = false;
    }
    const aiTargetEquity = document.getElementById('aiTargetEquity');
    if (aiTargetEquity) {
        aiTargetEquity.value = 0.11;
        document.getElementById('aiTargetEquityValue').textContent = '0.11';
    }
    
    // Scenario Updates defaults
    const aiScenarioUpdateSourceType = document.getElementById('aiScenarioUpdateSourceType');
    if (aiScenarioUpdateSourceType) {
        aiScenarioUpdateSourceType.value = 'none';
        document.getElementById('aiScenarioUpdateFileGroup').style.display = 'none';
        document.getElementById('aiScenarioUpdateApiGroup').style.display = 'none';
    }
    const aiScenarioUpdateFile = document.getElementById('aiScenarioUpdateFile');
    if (aiScenarioUpdateFile) {
        aiScenarioUpdateFile.value = '';
    }
    const aiScenarioUpdateApi = document.getElementById('aiScenarioUpdateApi');
    if (aiScenarioUpdateApi) {
        aiScenarioUpdateApi.value = '';
    }
    const aiScenarioMergeStrategy = document.getElementById('aiScenarioMergeStrategy');
    if (aiScenarioMergeStrategy) {
        aiScenarioMergeStrategy.value = 'weighted';
    }
    const aiScenarioUpdateWeight = document.getElementById('aiScenarioUpdateWeight');
    if (aiScenarioUpdateWeight) {
        aiScenarioUpdateWeight.value = 0.30;
        document.getElementById('aiScenarioUpdateWeightValue').textContent = '0.30';
    }
    
    // Simulation validation defaults
    document.getElementById('aiValidateResults').checked = false;
    document.getElementById('aiTargetMetric').value = 'mean_fidelity';
    document.getElementById('aiRiskMetric').value = 'asymmetry_delta';
    document.getElementById('aiCvFolds').value = 5;
    document.getElementById('aiCvFoldsValue').textContent = '5';
    document.getElementById('aiBiasThreshold').value = 0.1;
    document.getElementById('aiBiasThresholdValue').textContent = '0.1';
    document.getElementById('aiEquityThreshold').value = 0.1;
    document.getElementById('aiEquityThresholdValue').textContent = '0.1';
    
    // AI Model defaults
    document.getElementById('aiMaxTokens').value = 1000;
    document.getElementById('aiMaxTokensValue').textContent = '1000';
    document.getElementById('aiTemperature').value = 0.70;
    document.getElementById('aiTemperatureValue').textContent = '0.7';
}

function showParameterInfo(key) {
    if (!paramInfoTooltip || !paramInfoTitle || !paramInfoRange || !paramInfoPurpose || !paramInfoEffect) {
        return;
    }
    const data = parameterCatalog[key] || {
        title: 'Parameter Details',
        range: 'Range: see control',
        purpose: 'No additional information available.',
        effect: ''
    };
    paramInfoTitle.textContent = data.title;
    paramInfoRange.textContent = data.range;
    paramInfoPurpose.textContent = `Purpose: ${data.purpose}`;
    paramInfoEffect.textContent = data.effect ? `Effect: ${data.effect}` : '';
    paramInfoTooltip.classList.add('active');
    paramInfoTooltip.style.display = 'flex';
}

function hideParameterInfo() {
    if (paramInfoTooltip) {
        paramInfoTooltip.classList.remove('active');
        paramInfoTooltip.style.display = 'none';
    }
}

// Load configuration
async function loadConfig() {
    console.log('loadConfig() called');
    
    let configTextEl;
    let statusIndicatorEl;
    
    try {
        // Wait a bit to ensure DOM is fully ready
        await new Promise(resolve => setTimeout(resolve, 100));
        
        configTextEl = document.getElementById('configText');
        statusIndicatorEl = document.getElementById('statusIndicator');
        
        if (!configTextEl) {
            console.error('Config text element not found - element ID: configText');
            // Try to find it with a different approach
            const configInfo = document.getElementById('configInfo');
            if (configInfo) {
                const textSpan = configInfo.querySelector('#configText');
                if (textSpan) {
                    textSpan.textContent = 'Configuration element found but ID mismatch';
                }
            }
            return;
        }
        
        // Set a safety timeout to clear loading state if function hangs
        const safetyTimeout = setTimeout(() => {
            if (configTextEl && (configTextEl.textContent === 'Loading configuration...' || 
                                 configTextEl.textContent.includes('Loading'))) {
                console.warn('Safety timeout: Config loading appears stuck, setting fallback message');
                configTextEl.textContent = 'Configuration loading... (check console)';
            }
        }, 2000); // Reduced to 2 seconds
        
        console.log('Config element found, making API request...');
        
        // Set timeout for the fetch request
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.log('Config fetch timeout');
            controller.abort();
        }, 5000); // 5 second timeout
        
        console.log('Fetching /api/config...');
        let response;
        try {
            response = await fetch('/api/config', {
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json'
                },
                cache: 'no-cache'
            });
        } catch (fetchError) {
            clearTimeout(timeoutId);
            console.error('Fetch error:', fetchError);
            throw fetchError;
        }
        
        clearTimeout(timeoutId);
        console.log('Config response received:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const config = await response.json();
        console.log('Config data:', config);
        
        // Double-check that configTextEl still exists before updating
        if (!configTextEl) {
            console.error('configTextEl became null after fetch');
            return;
        }
        
        // Store config globally for use in message display
        globalConfig = {
            qiskit_available: config.qiskit_available || false,
            qiskit_version: config.qiskit_version || 0,
            prhp_available: config.prhp_available || false
        };
        
        const providerText = config.provider ? (config.provider.charAt(0).toUpperCase() + config.provider.slice(1)) : 'Unknown';
        const modelText = config.model || 'Unknown Model';
        const statusText = config.has_api_key ? 'Connected' : 'No API Key';
        const prhpStatus = config.prhp_available ? 'PRHP ✓' : 'PRHP ✗';
        const qiskitStatus = config.qiskit_available ? 
            `Qiskit v${config.qiskit_version} ✓` : 
            'Qiskit ✗ (using classical)';
        
        const configDisplayText = `${providerText} • ${modelText} • ${statusText} • ${prhpStatus} • ${qiskitStatus}`;
        console.log('Updating config text to:', configDisplayText);
        configTextEl.textContent = configDisplayText;
        
        // Update status indicator
        if (statusIndicatorEl) {
            statusIndicatorEl.classList.remove('error', 'warning', 'success');
            if (!config.has_api_key || !config.prhp_available) {
                statusIndicatorEl.classList.add('warning');
            } else {
                statusIndicatorEl.classList.add('success');
            }
        }
        
        console.log('Config UI updated successfully');
        
        // Clear safety timeout since we succeeded
        clearTimeout(safetyTimeout);
        
        if (!config.has_api_key) {
            showError('API key not configured. Please set API_KEY environment variable or create a .env file.');
        }
        
        if (!config.prhp_available) {
            showError('PRHP framework not available. Some features may not work.');
        }
        
        if (!config.qiskit_available) {
            console.warn('Qiskit not installed. Quantum features will use classical approximation. Install with: pip install qiskit qiskit-aer');
        }
    } catch (error) {
        // Clear safety timeout on error if it exists
        if (typeof safetyTimeout !== 'undefined') {
            clearTimeout(safetyTimeout);
        }
        console.error('Failed to load config:', error);
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        
        // Ensure we have the element before trying to update it
        if (!configTextEl) {
            console.error('configTextEl is null in catch block');
            // Try to get it again
            configTextEl = document.getElementById('configText');
            if (!configTextEl) {
                console.error('Still cannot find configTextEl');
                return;
            }
        }
        
        if (statusIndicatorEl) {
            statusIndicatorEl.classList.remove('warning', 'success');
            statusIndicatorEl.classList.add('error');
        }
        
        let errorMessage = 'Configuration error - Check console for details';
        if (error.name === 'AbortError') {
            errorMessage = 'Configuration timeout - Server may be slow. Try refreshing.';
        } else if (error.message && error.message.includes('Failed to fetch')) {
            errorMessage = 'Cannot connect to server - Check if Flask is running on port 5000';
        } else if (error.message) {
            errorMessage = `Configuration error: ${error.message}`;
        }
        
        // Update config text with error message
        try {
            configTextEl.textContent = errorMessage;
        } catch (e) {
            console.error('Could not update config text element:', e);
            // Final fallback if even updating the text fails
            const fallbackEl = document.getElementById('configText');
            if (fallbackEl) {
                fallbackEl.textContent = 'Configuration error - Please refresh the page';
            }
        }
    } finally {
        // Always clear loading state, even if there was an error
        console.log('Config loading completed (success or error)');
    }
}

// Setup event listeners
function setupEventListeners() {
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);
    if (clearBtn) clearBtn.addEventListener('click', clearConversation);
    if (clearInputBtn) clearInputBtn.addEventListener('click', clearInput);
    if (runSimulationBtn) runSimulationBtn.addEventListener('click', runSimulation);
    if (clearSimBtn) clearSimBtn.addEventListener('click', clearSimulationResults);
    if (runQuantumBtn) runQuantumBtn.addEventListener('click', runQuantumSimulation);
    if (clearQuantumBtn) clearQuantumBtn.addEventListener('click', clearQuantumResults);
    
    // Conversation management
    const newConvBtn = document.getElementById('newConversationBtn');
    const saveConvBtn = document.getElementById('saveConversationBtn');
    const loadConvBtn = document.getElementById('loadConversationBtn');
    const savedConvSelect = document.getElementById('savedConversationsSelect');
    
    if (newConvBtn) newConvBtn.addEventListener('click', startNewConversation);
    if (saveConvBtn) saveConvBtn.addEventListener('click', saveConversation);
    if (loadConvBtn) loadConvBtn.addEventListener('click', showLoadConversation);
    if (savedConvSelect) savedConvSelect.addEventListener('change', loadSelectedConversation);
    
    // Load saved conversations list
    loadSavedConversationsList();
    
    // Add Enter key support for input box
    if (inputBox) {
        inputBox.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
}

// Setup tabs
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Update buttons
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetTab}Tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// Get AI parameters
function getAIParameters() {
    // Get PRHP Simulation parameters
    const aiVariants = Array.from(document.querySelectorAll('.ai-variant-checkbox:checked')).map(cb => cb.value);
    
    const prhpParams = {
        levels: parseInt(document.getElementById('aiLevels').value) || 9,
        n_monte: parseInt(document.getElementById('aiNMonte').value) || 100,
        seed: parseInt(document.getElementById('aiSeed').value) || 42,
        use_quantum: document.getElementById('aiUseQuantum').checked,
        track_levels: document.getElementById('aiTrackLevels').checked,
        variants: aiVariants.length > 0 ? aiVariants : ['neurotypical-hybrid']
    };
    
    // Get Historical Data Integration parameters
    const historyFileSelect = document.getElementById('aiHistoryFile');
    const historyFile = historyFileSelect ? historyFileSelect.value : '';
    if (historyFile) {
        prhpParams.history_file_path = historyFile;
        prhpParams.historical_weight = parseFloat(document.getElementById('aiHistoricalWeight').value) || 0.30;
    }
    
    // Get Risk-Utility Recalibration parameters
    const recalibrateRiskUtility = document.getElementById('aiRecalibrateRiskUtility');
    if (recalibrateRiskUtility && recalibrateRiskUtility.checked) {
        prhpParams.recalibrate_risk_utility = true;
        prhpParams.target_equity = parseFloat(document.getElementById('aiTargetEquity').value) || 0.11;
    }
    
    // Get Scenario Updates parameters
    const scenarioUpdateSourceType = document.getElementById('aiScenarioUpdateSourceType');
    if (scenarioUpdateSourceType) {
        const sourceType = scenarioUpdateSourceType.value;
        if (sourceType === 'file') {
            const scenarioUpdateFile = document.getElementById('aiScenarioUpdateFile');
            const fileValue = scenarioUpdateFile ? scenarioUpdateFile.value : '';
            if (fileValue) {
                prhpParams.scenario_update_file = fileValue;
                prhpParams.scenario_merge_strategy = document.getElementById('aiScenarioMergeStrategy').value || 'weighted';
                prhpParams.scenario_update_weight = parseFloat(document.getElementById('aiScenarioUpdateWeight').value) || 0.30;
            }
        } else if (sourceType === 'api') {
            const scenarioUpdateApi = document.getElementById('aiScenarioUpdateApi');
            const apiValue = scenarioUpdateApi ? scenarioUpdateApi.value.trim() : '';
            if (apiValue) {
                prhpParams.scenario_update_source = apiValue;
                prhpParams.scenario_merge_strategy = document.getElementById('aiScenarioMergeStrategy').value || 'weighted';
                prhpParams.scenario_update_weight = parseFloat(document.getElementById('aiScenarioUpdateWeight').value) || 0.30;
            }
        }
    }
    
    // Get simulation validation parameters
    const aiValidateResults = document.getElementById('aiValidateResults');
    if (aiValidateResults && aiValidateResults.checked) {
        prhpParams.validate_results = true;
        prhpParams.target_metric = document.getElementById('aiTargetMetric').value || 'mean_fidelity';
        prhpParams.risk_metric = document.getElementById('aiRiskMetric').value || 'asymmetry_delta';
        prhpParams.cv_folds = parseInt(document.getElementById('aiCvFolds').value) || 5;
        prhpParams.bias_threshold = parseFloat(document.getElementById('aiBiasThreshold').value) || 0.1;
        prhpParams.equity_threshold = parseFloat(document.getElementById('aiEquityThreshold').value) || 0.1;
    }
    
    // Get Quantum Hooks parameters
    const quantumParams = {
        use_quantum: document.getElementById('aiUseQuantumHooks').checked,
        flip_prob: parseFloat(document.getElementById('aiFlipProb').value) || 0.30,
        threshold: parseFloat(document.getElementById('aiThreshold').value) || 0.70,
        divergence: parseFloat(document.getElementById('aiDivergence').value) || 0.22,
        variant: document.getElementById('aiQuantumVariant').value || 'neurotypical-hybrid'
    };
    
    // Get AI Model parameters
    const aiModelParams = {
        max_tokens: parseInt(document.getElementById('aiMaxTokens').value) || 1000,
        temperature: parseFloat(document.getElementById('aiTemperature').value) || 0.7
    };
    
    return {
        prhp: prhpParams,
        quantum: quantumParams,
        ai_model: aiModelParams
    };
}

// Add message to chat
function addMessageToChat(role, content, metadata = null) {
    const chatMessages = document.getElementById('chatMessages');
    
    // Remove placeholder if exists
    const placeholder = chatMessages.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    headerDiv.textContent = role === 'user' ? 'You' : 'AI Assistant';
    
    // Add PRHP status indicator for assistant messages
    if (role === 'assistant' && metadata && metadata.prhp_status) {
        const prhpStatus = metadata.prhp_status;
        const statusIndicator = document.createElement('div');
        statusIndicator.className = 'prhp-status-indicator';
        
        if (prhpStatus.simulation_success) {
            statusIndicator.textContent = '🔬 PRHP-PROCESSED RESPONSE 🔬';
            statusIndicator.style.color = '#60a5fa';
            statusIndicator.style.fontWeight = 'bold';
            statusIndicator.style.marginBottom = '8px';
        } else if (prhpStatus.simulation_attempted && !prhpStatus.simulation_success) {
            statusIndicator.textContent = '⚠️ PRHP Simulation Failed (Direct Response)';
            statusIndicator.style.color = '#fbbf24';
            statusIndicator.style.fontWeight = 'bold';
            statusIndicator.style.marginBottom = '8px';
        } else {
            statusIndicator.textContent = '📝 Direct AI Response (no PRHP processing)';
            statusIndicator.style.color = '#9ca3af';
            statusIndicator.style.fontWeight = 'normal';
            statusIndicator.style.marginBottom = '8px';
        }
        
        bubbleDiv.appendChild(statusIndicator);
        
        // Display simulation results if available
        if (prhpStatus.simulation_success && metadata.simulation_results && metadata.simulation_results.results) {
            const simResultsDiv = document.createElement('div');
            simResultsDiv.className = 'simulation-results-display';
            simResultsDiv.style.marginBottom = '12px';
            simResultsDiv.style.padding = '8px';
            simResultsDiv.style.backgroundColor = 'rgba(96, 165, 250, 0.1)';
            simResultsDiv.style.borderRadius = '4px';
            simResultsDiv.style.fontSize = '12px';
            
            let simText = '<strong>PRHP Simulation Data:</strong><br>';
            const results = metadata.simulation_results.results;
            for (const [variant, data] of Object.entries(results)) {
                simText += `<br><strong>${variant}:</strong><br>`;
                simText += `  Mean Fidelity: ${(data.mean_fidelity ?? 0).toFixed(4)} ± ${(data.std ?? 0).toFixed(4)}<br>`;
                simText += `  Asymmetry Delta: ${(data.asymmetry_delta ?? 0).toFixed(4)}<br>`;
                simText += `  Novelty Generation: ${(data.novelty_gen ?? 0).toFixed(4)}<br>`;
                if (data.mean_phi_delta !== null && data.mean_phi_delta !== undefined) {
                    simText += `  Mean Phi Delta: ${data.mean_phi_delta.toFixed(4)}<br>`;
                }
            }
            
            simResultsDiv.innerHTML = simText;
            bubbleDiv.appendChild(simResultsDiv);
        }
    }
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    bubbleDiv.appendChild(headerDiv);
    bubbleDiv.appendChild(contentDiv);
    
    // Add metadata if available (for both user and assistant messages)
    if (metadata) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        metadataDiv.style.marginTop = '12px';
        metadataDiv.style.padding = '8px';
        metadataDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
        metadataDiv.style.borderRadius = '6px';
        metadataDiv.style.fontSize = '12px';
        metadataDiv.style.color = '#999999';
        metadataDiv.style.lineHeight = '1.6';
        
        let metadataParts = [];
        
        // Qiskit Status
        if (globalConfig.qiskit_available) {
            metadataParts.push(`⚛️ Qiskit: v${globalConfig.qiskit_version} (Quantum enabled)`);
        } else {
            metadataParts.push(`⚛️ Qiskit: Not available (Classical mode)`);
        }
        
        // PRHP Status
        if (metadata.prhp_status) {
            const prhpStatus = metadata.prhp_status;
            if (prhpStatus.detected) {
                metadataParts.push(`🔬 PRHP: Detected (Score: ${prhpStatus.relevance_score || 0})`);
            }
            if (prhpStatus.simulation_attempted) {
                metadataParts.push(`🔬 Simulation: ${prhpStatus.simulation_success ? '✓ Success' : '✗ Failed'}`);
            }
            if (prhpStatus.parameters_used) {
                const params = prhpStatus.parameters_used;
                metadataParts.push(`🔬 PRHP Params: Levels=${params.levels || 'N/A'}, Monte=${params.n_monte || 'N/A'}, Quantum=${params.use_quantum ? 'Yes' : 'No'}`);
            }
        }
        
        // PRHP Parameters used
        if (metadata.prhp_parameters) {
            metadataParts.push(`📊 PRHP Config: Levels=${metadata.prhp_parameters.levels}, Monte=${metadata.prhp_parameters.n_monte}, Variants=${metadata.prhp_parameters.variants?.join(', ') || 'N/A'}`);
        }
        
        // AI Model Parameters
        if (metadata.ai_model_parameters) {
            metadataParts.push(`🤖 AI Model: MaxTokens=${metadata.ai_model_parameters.max_tokens}, Temperature=${metadata.ai_model_parameters.temperature}`);
        }
        
        // Historical Data Integration
        if (metadata.historical_integration) {
            const hist = metadata.historical_integration;
            metadataParts.push(`📊 Historical Data: Applied (${((hist.historical_weight ?? 0) * 100).toFixed(0)}% historical, ${((hist.current_weight ?? 0) * 100).toFixed(0)}% current)`);
            metadataParts.push(`📊 Historical Variance: ${(hist.historical_variance ?? 0).toFixed(4)}, Samples: ${hist.historical_samples ?? 0}`);
        }
        
        // Risk-Utility Recalibration
        if (metadata.recalibration) {
            const recal = metadata.recalibration;
            metadataParts.push(`⚖️ Risk-Utility Recalibration: Applied`);
            metadataParts.push(`⚖️ Target Equity: ${(recal.target_equity ?? 0).toFixed(4)}, Achieved: ${recal.achieved_equity ? recal.achieved_equity.toFixed(4) : 'N/A'}`);
            if (recal.iterations) {
                metadataParts.push(`⚖️ Optimization Iterations: ${recal.iterations}`);
            }
        }
        
        // Scenario Updates
        if (metadata.scenario_update) {
            const scenario = metadata.scenario_update;
            metadataParts.push(`🔄 Scenario Updates: Applied`);
            metadataParts.push(`🔄 Source: ${scenario.update_source || 'unknown'}, Strategy: ${scenario.merge_strategy}`);
            metadataParts.push(`🔄 Update Time: ${scenario.update_time || 'N/A'}, Weight: ${((scenario.update_weight ?? 0) * 100).toFixed(0)}%`);
        }
        
        // Simulation Validation
        if (metadata.validation) {
            const validation = metadata.validation;
            metadataParts.push(`✅ Simulation Validation: ${validation.is_valid ? 'Valid ✓' : 'Needs Recalibration ✗'}`);
            metadataParts.push(`✅ CV Score: ${validation.cv_mean_score ? validation.cv_mean_score.toFixed(4) : 'N/A'} ± ${validation.cv_std_score ? validation.cv_std_score.toFixed(4) : 'N/A'}`);
            metadataParts.push(`✅ Bias Delta: ${validation.bias_delta ? validation.bias_delta.toFixed(4) : 'N/A'}, Equity Delta: ${validation.equity_delta ? validation.equity_delta.toFixed(4) : 'N/A'}`);
            metadataParts.push(`✅ R² Score: ${validation.r2_score ? validation.r2_score.toFixed(4) : 'N/A'}, MSE: ${validation.mse ? validation.mse.toFixed(4) : 'N/A'}`);
            if (validation.warnings && validation.warnings.length > 0) {
                metadataParts.push(`⚠️ Warnings: ${validation.warnings.join(', ')}`);
            }
            metadataParts.push(`✅ Recommendation: ${validation.recommendation || 'N/A'}`);
        }
        
        if (metadataParts.length > 0) {
            metadataDiv.innerHTML = metadataParts.join('<br>');
            bubbleDiv.appendChild(metadataDiv);
        }
    }
    
    messageDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll page to show new message
    scrollToNewMessage();
    
    // Add to conversation history
    conversationHistory.push({
        role: role,
        content: content,
        timestamp: new Date().toISOString(),
        metadata: metadata
    });
    
    // Save to localStorage
    saveConversationToStorage();
}

// Scroll page to show new messages
function scrollToNewMessage() {
    // Small delay to ensure DOM is updated
    setTimeout(() => {
        window.scrollTo({
            top: document.body.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

// Send message to AI
async function sendMessage() {
    const prompt = inputBox.value.trim();
    
    if (!prompt) {
        showError('Please enter a message before sending.');
        return;
    }
    
    if (isLoading) {
        return;
    }
    
    // Set loading state
    isLoading = true;
    sendBtn.disabled = true;
    loader.style.display = 'block';
    sendBtn.querySelector('.btn-text').textContent = 'Sending...';
    hideError();
    
    // Get fine-tuning parameters
    const parameters = getAIParameters();
    
    // Add user message to chat
    addMessageToChat('user', prompt, {
        prhp_parameters: parameters.prhp,
        ai_model_parameters: parameters.ai_model
    });
    
    // Build conversation context for API
    const conversationMessages = conversationHistory
        .filter(msg => msg.role === 'user' || msg.role === 'assistant')
        .map(msg => ({
            role: msg.role === 'user' ? 'user' : 'assistant',
            content: msg.content
        }));
    
    // Clear input immediately for better UX
    inputBox.value = '';
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                prompt: prompt,
                conversation_history: conversationMessages.slice(0, -1), // Exclude current message
                prhp_parameters: parameters.prhp,
                quantum_parameters: parameters.quantum,
                ai_model_parameters: parameters.ai_model
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get response from API');
        }
        
        // Extract metadata from simulation results if available
        let historicalIntegration = null;
        let recalibration = null;
        let scenarioUpdate = null;
        let validation = null;
        if (data.simulation_results && data.simulation_results.results) {
            // Check if any variant has metadata
            for (const variant in data.simulation_results.results) {
                const variantData = data.simulation_results.results[variant];
                if (variantData.historical_integration && !historicalIntegration) {
                    historicalIntegration = variantData.historical_integration;
                }
                if (variantData.recalibration && !recalibration) {
                    recalibration = variantData.recalibration;
                }
                if (variantData.scenario_update && !scenarioUpdate) {
                    scenarioUpdate = variantData.scenario_update;
                }
                if (variantData.validation && !validation) {
                    validation = variantData.validation;
                }
            }
        }
        
        // Add AI response to chat with PRHP status
        addMessageToChat('assistant', data.response, {
            prhp_status: data.prhp_status || {},
            ai_model_parameters: data.ai_model_parameters_used,
            simulation_results: data.simulation_results,
            historical_integration: historicalIntegration,
            recalibration: recalibration,
            scenario_update: scenarioUpdate,
            validation: validation
        });
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}`);
        addMessageToChat('assistant', `Error: ${error.message}`, { error: true });
    } finally {
        // Reset loading state
        isLoading = false;
        sendBtn.disabled = false;
        loader.style.display = 'none';
        sendBtn.querySelector('.btn-text').textContent = 'Send';
        inputBox.focus();
    }
}

// Send follow-up message to AI
async function sendFollowUpMessage() {
    if (!followUpInputBox) return;
    
    const prompt = followUpInputBox.value.trim();
    
    if (!prompt) {
        showError('Please enter a message before sending.');
        return;
    }
    
    if (isLoading) {
        return;
    }
    
    // Set loading state
    isLoading = true;
    if (followUpSendBtn) {
        followUpSendBtn.disabled = true;
    }
    if (followUpLoader) {
        followUpLoader.style.display = 'block';
    }
    if (followUpSendBtn) {
        followUpSendBtn.querySelector('.btn-text').textContent = 'Sending...';
    }
    hideError();
    
    // Get fine-tuning parameters
    const parameters = getAIParameters();
    
    // Add user message to chat
    addMessageToChat('user', prompt, {
        prhp_parameters: parameters.prhp,
        ai_model_parameters: parameters.ai_model
    });
    
    // Build conversation context for API
    const conversationMessages = conversationHistory
        .filter(msg => msg.role === 'user' || msg.role === 'assistant')
        .map(msg => ({
            role: msg.role === 'user' ? 'user' : 'assistant',
            content: msg.content
        }));
    
    // Clear input immediately for better UX
    followUpInputBox.value = '';
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                prompt: prompt,
                conversation_history: conversationMessages.slice(0, -1), // Exclude current message
                prhp_parameters: parameters.prhp,
                quantum_parameters: parameters.quantum,
                ai_model_parameters: parameters.ai_model
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get response from API');
        }
        
        // Extract metadata from simulation results if available
        let historicalIntegration = null;
        let recalibration = null;
        let scenarioUpdate = null;
        let validation = null;
        if (data.simulation_results && data.simulation_results.results) {
            // Check if any variant has metadata
            for (const variant in data.simulation_results.results) {
                const variantData = data.simulation_results.results[variant];
                if (variantData.historical_integration && !historicalIntegration) {
                    historicalIntegration = variantData.historical_integration;
                }
                if (variantData.recalibration && !recalibration) {
                    recalibration = variantData.recalibration;
                }
                if (variantData.scenario_update && !scenarioUpdate) {
                    scenarioUpdate = variantData.scenario_update;
                }
                if (variantData.validation && !validation) {
                    validation = variantData.validation;
                }
            }
        }
        
        // Add AI response to chat with PRHP status
        addMessageToChat('assistant', data.response, {
            prhp_status: data.prhp_status || {},
            ai_model_parameters: data.ai_model_parameters_used,
            simulation_results: data.simulation_results,
            historical_integration: historicalIntegration,
            recalibration: recalibration,
            scenario_update: scenarioUpdate,
            validation: validation
        });
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Error: ${error.message}`);
        addMessageToChat('assistant', `Error: ${error.message}`, { error: true });
    } finally {
        // Reset loading state
        isLoading = false;
        if (followUpSendBtn) {
            followUpSendBtn.disabled = false;
        }
        if (followUpLoader) {
            followUpLoader.style.display = 'none';
        }
        if (followUpSendBtn) {
            followUpSendBtn.querySelector('.btn-text').textContent = 'Send';
        }
        if (followUpInputBox) {
            followUpInputBox.focus();
        }
    }
}

// Conversation Management Functions

// Start new conversation
function startNewConversation() {
    if (confirm('Start a new conversation? Current conversation will be cleared.')) {
        conversationHistory = [];
        currentConversationId = null;
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '<p class="placeholder">Start a conversation by entering a message above...</p>';
        hideError();
    }
}

// Clear conversation
function clearConversation() {
    if (confirm('Clear the current conversation?')) {
        conversationHistory = [];
        currentConversationId = null;
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '<p class="placeholder">Start a conversation by entering a message above...</p>';
        hideError();
    }
}

// Save conversation to localStorage
function saveConversationToStorage() {
    try {
        const conversations = JSON.parse(localStorage.getItem('prhp_conversations') || '{}');
        const id = currentConversationId || `conv_${Date.now()}`;
        conversations[id] = {
            id: id,
            history: conversationHistory,
            created: currentConversationId ? conversations[id]?.created || new Date().toISOString() : new Date().toISOString(),
            updated: new Date().toISOString()
        };
        localStorage.setItem('prhp_conversations', JSON.stringify(conversations));
        currentConversationId = id;
        loadSavedConversationsList();
    } catch (error) {
        console.error('Error saving conversation:', error);
    }
}

// Save conversation with name
async function saveConversation() {
    if (conversationHistory.length === 0) {
        showError('No conversation to save.');
        return;
    }
    
    const name = prompt('Enter a name for this conversation:', 
        `Conversation ${new Date().toLocaleString()}`);
    
    if (!name) return;
    
    try {
        const response = await fetch('/api/conversations/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: name,
                history: conversationHistory,
                conversation_id: currentConversationId
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            currentConversationId = data.conversation_id;
            showSuccess(`Conversation saved as "${name}"`);
            loadSavedConversationsList();
        } else {
            showError(data.error || 'Failed to save conversation');
        }
    } catch (error) {
        console.error('Error saving conversation:', error);
        showError('Failed to save conversation');
    }
}

// Load saved conversations list
async function loadSavedConversationsList() {
    try {
        const response = await fetch('/api/conversations/list');
        const data = await response.json();
        
        if (response.ok && data.conversations) {
            const select = document.getElementById('savedConversationsSelect');
            if (select) {
                select.innerHTML = '<option value="">Select a saved conversation...</option>';
                data.conversations.forEach(conv => {
                    const option = document.createElement('option');
                    option.value = conv.id;
                    option.textContent = `${conv.name} (${conv.timestamp})`;
                    select.appendChild(option);
                });
            }
        }
    } catch (error) {
        console.error('Error loading conversations list:', error);
    }
}

// Show load conversation selector
function showLoadConversation() {
    const selector = document.getElementById('conversationSelector');
    if (selector) {
        selector.style.display = selector.style.display === 'none' ? 'block' : 'none';
        loadSavedConversationsList();
    }
}

// Load selected conversation
async function loadSelectedConversation() {
    const select = document.getElementById('savedConversationsSelect');
    const conversationId = select.value;
    
    if (!conversationId) return;
    
    try {
        const response = await fetch(`/api/conversations/load/${conversationId}`);
        const data = await response.json();
        
        if (response.ok && data.history) {
            conversationHistory = data.history;
            currentConversationId = conversationId;
            
            // Clear and rebuild chat display
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = '';
            
            // Rebuild messages (without adding to history again)
            data.history.forEach(msg => {
                // Remove placeholder if exists (only need to check once, but safe to check each time)
                const placeholder = chatMessages.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
                const messageDiv = document.createElement('div');
                messageDiv.className = `message message-${msg.role}`;
                
                const bubbleDiv = document.createElement('div');
                bubbleDiv.className = 'message-bubble';
                
                const headerDiv = document.createElement('div');
                headerDiv.className = 'message-header';
                headerDiv.textContent = msg.role === 'user' ? 'You' : 'AI Assistant';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = msg.content;
                
                bubbleDiv.appendChild(headerDiv);
                bubbleDiv.appendChild(contentDiv);
                
                if (msg.metadata) {
                    const metadataDiv = document.createElement('div');
                    metadataDiv.className = 'message-metadata';
                    
                    let metadataText = '';
                    if (msg.metadata.prhp_parameters) {
                        metadataText += `PRHP: Levels=${msg.metadata.prhp_parameters.levels}, Monte=${msg.metadata.prhp_parameters.n_monte}`;
                    }
                    if (msg.metadata.ai_model_parameters) {
                        metadataText += ` | AI: MaxTokens=${msg.metadata.ai_model_parameters.max_tokens}, Temp=${msg.metadata.ai_model_parameters.temperature}`;
                    }
                    
                    if (metadataText) {
                        metadataDiv.textContent = metadataText;
                        bubbleDiv.appendChild(metadataDiv);
                    }
                }
                
                messageDiv.appendChild(bubbleDiv);
                chatMessages.appendChild(messageDiv);
            });
            
            // Set conversation history directly (don't add duplicates)
            conversationHistory = data.history;
            
            // Scroll page to show loaded messages
            scrollToNewMessage();
            
            showSuccess('Conversation loaded');
            document.getElementById('conversationSelector').style.display = 'none';
        } else {
            showError(data.error || 'Failed to load conversation');
        }
    } catch (error) {
        console.error('Error loading conversation:', error);
        showError('Failed to load conversation');
    }
}

// Show success message
function showSuccess(message) {
    const errorMsg = document.getElementById('errorMessage');
    if (errorMsg) {
        errorMsg.textContent = message;
        errorMsg.style.display = 'block';
        errorMsg.style.background = '#d1fae5';
        errorMsg.style.color = '#065f46';
        errorMsg.style.borderLeftColor = '#10b981';
        setTimeout(() => {
            errorMsg.style.display = 'none';
        }, 3000);
    }
}

// Clear input
function clearInput() {
    inputBox.value = '';
    inputBox.focus();
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Hide error message
function hideError() {
    errorMessage.style.display = 'none';
}

// Run PRHP simulation
async function runSimulation() {
    if (isSimulating) {
        return;
    }
    
    // Get selected variants
    const variantCheckboxes = document.querySelectorAll('.variant-checkbox:checked');
    const variants = Array.from(variantCheckboxes).map(cb => cb.value);
    
    if (variants.length === 0) {
        showError('Please select at least one variant.');
        return;
    }
    
    // Get parameters
    const levels = parseInt(document.getElementById('levels').value) || 9;
    const nMonte = parseInt(document.getElementById('nMonte').value) || 100;
    const seed = parseInt(document.getElementById('seed').value) || 42;
    const useQuantum = document.getElementById('useQuantum').checked;
    const trackLevels = document.getElementById('trackLevels').checked;
    
    // Set loading state
    isSimulating = true;
    runSimulationBtn.disabled = true;
    simLoader.style.display = 'block';
    runSimulationBtn.querySelector('.btn-text').textContent = 'Running...';
    hideError();
    
    // Clear previous results
    const placeholder = simulationOutput.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Show starting message
    addToSimulationOutput('Starting PRHP simulation...\n');
    addToSimulationOutput(`Parameters: levels=${levels}, variants=${variants.join(', ')}, n_monte=${nMonte}, seed=${seed}, quantum=${useQuantum}, track_levels=${trackLevels}\n\n`);
    
    try {
        const response = await fetch('/api/prhp/simulate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                levels: levels,
                variants: variants,
                n_monte: nMonte,
                seed: seed,
                use_quantum: useQuantum,
                track_levels: trackLevels
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Simulation failed');
        }
        
        // Display results
        displaySimulationResults(data.results, data.parameters);
        
    } catch (error) {
        console.error('Simulation error:', error);
        showError(`Simulation error: ${error.message}`);
        addToSimulationOutput(`\nError: ${error.message}\n`, 'error-message');
    } finally {
        // Reset loading state
        isSimulating = false;
        runSimulationBtn.disabled = false;
        simLoader.style.display = 'none';
        runSimulationBtn.querySelector('.btn-text').textContent = 'Run Simulation';
    }
}

// Display simulation results
function displaySimulationResults(results, parameters) {
    addToSimulationOutput('\n=== Simulation Results ===\n\n');
    
    for (const [variant, data] of Object.entries(results)) {
        const variantDiv = document.createElement('div');
        variantDiv.className = 'result-variant';
        
        variantDiv.innerHTML = `
            <h3>${variant}</h3>
            <div class="result-metric">
                <span class="result-metric-label">Mean Fidelity:</span>
                <span class="result-metric-value">${(data.mean_fidelity ?? 0).toFixed(4)} ± ${(data.std ?? 0).toFixed(4)}</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Asymmetry Delta:</span>
                <span class="result-metric-value">${(data.asymmetry_delta ?? 0).toFixed(4)}</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Novelty Generation:</span>
                <span class="result-metric-value">${(data.novelty_gen ?? 0).toFixed(4)}</span>
            </div>
            ${data.mean_phi_delta !== null && data.mean_phi_delta !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Mean Phi Delta:</span>
                <span class="result-metric-value">${data.mean_phi_delta.toFixed(4)}</span>
            </div>
            ` : ''}
            ${data.phi_deltas && data.phi_deltas.length > 0 ? `
            <div class="result-metric">
                <span class="result-metric-label">Phi Deltas (per level):</span>
                <span class="result-metric-value">${data.phi_deltas.slice(0, 5).map(d => (d ?? 0).toFixed(3)).join(', ')}${data.phi_deltas.length > 5 ? '...' : ''}</span>
            </div>
            ` : ''}
            ${data.level_phis && data.level_phis.length > 0 ? `
            <div class="result-metric">
                <span class="result-metric-label">Level Phis (last 5):</span>
                <span class="result-metric-value">${data.level_phis.slice(-5).map(p => (p ?? 0).toFixed(3)).join(', ')}</span>
            </div>
            ` : ''}
        `;
        
        simulationOutput.appendChild(variantDiv);
    }
    
    addToSimulationOutput('\n=== End of Results ===\n\n');
    simulationOutput.scrollTop = simulationOutput.scrollHeight;
}

// Add text to simulation output
function addToSimulationOutput(text, className = '') {
    const p = document.createElement('p');
    p.className = className;
    p.textContent = text;
    simulationOutput.appendChild(p);
    simulationOutput.scrollTop = simulationOutput.scrollHeight;
}

// Clear simulation results
function clearSimulationResults() {
    simulationOutput.innerHTML = '<p class="placeholder">Simulation results will appear here...</p>';
    hideError();
}

// Run Quantum Hooks simulation
async function runQuantumSimulation() {
    if (isQuantumSimulating) {
        return;
    }
    
    // Get parameters
    const useQuantum = document.getElementById('useQuantumHooks').checked;
    const flipProb = parseFloat(document.getElementById('flipProb').value) || 0.30;
    const threshold = parseFloat(document.getElementById('threshold').value) || 0.70;
    const divergence = parseFloat(document.getElementById('divergence').value) || 0.22;
    const variant = document.getElementById('quantumVariant').value;
    
    // Set loading state
    isQuantumSimulating = true;
    runQuantumBtn.disabled = true;
    quantumLoader.style.display = 'block';
    runQuantumBtn.querySelector('.btn-text').textContent = 'Running...';
    hideError();
    
    // Clear previous results
    const placeholder = quantumOutput.querySelector('.placeholder');
    if (placeholder) {
        placeholder.remove();
    }
    
    // Show starting message
    addToQuantumOutput('Starting quantum hooks simulation...\n');
    addToQuantumOutput(`Parameters: variant=${variant}, use_quantum=${useQuantum}, flip_prob=${flipProb.toFixed(2)}, threshold=${threshold.toFixed(2)}, divergence=${divergence.toFixed(2)}\n\n`);
    
    try {
        const response = await fetch('/api/prhp/quantum', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                variant: variant,
                use_quantum: useQuantum,
                flip_prob: flipProb,
                threshold: threshold,
                divergence: divergence
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Quantum simulation failed');
        }
        
        // Display results
        displayQuantumResults(data);
        
    } catch (error) {
        console.error('Quantum simulation error:', error);
        showError(`Quantum simulation error: ${error.message}`);
        addToQuantumOutput(`\nError: ${error.message}\n`, 'error-message');
    } finally {
        // Reset loading state
        isQuantumSimulating = false;
        runQuantumBtn.disabled = false;
        quantumLoader.style.display = 'none';
        runQuantumBtn.querySelector('.btn-text').textContent = 'Run Quantum Simulation';
    }
}

// Display quantum hooks results
function displayQuantumResults(data) {
    addToQuantumOutput('\n=== Quantum Hooks Results ===\n\n');
    
    // Display parameters used
    if (data.parameters) {
        addToQuantumOutput('Parameters Used:', 'result-section-header');
        const paramsDiv = document.createElement('div');
        paramsDiv.className = 'result-variant';
        paramsDiv.style.marginBottom = '16px';
        paramsDiv.innerHTML = `
            <div class="result-metric">
                <span class="result-metric-label">Phase Flip Probability:</span>
                <span class="result-metric-value">${(data.parameters.flip_prob ?? 0).toFixed(2)}</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Asymmetry Threshold:</span>
                <span class="result-metric-value">${(data.parameters.threshold ?? 0).toFixed(2)}</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Predation Divergence:</span>
                <span class="result-metric-value">${(data.parameters.divergence ?? 0).toFixed(2)}</span>
            </div>
            <div class="result-metric">
                <span class="result-metric-label">Use Quantum:</span>
                <span class="result-metric-value">${data.parameters.use_quantum ? 'Yes' : 'No'}</span>
            </div>
        `;
        quantumOutput.appendChild(paramsDiv);
    }
    
    const resultDiv = document.createElement('div');
    resultDiv.className = 'result-variant';
    
    // Build detailed results
    let resultsHTML = `
        <h3>${data.variant} - Detailed Results</h3>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 2px solid #e5e7eb;">
            <h4 style="font-size: 14px; color: #667eea; margin-bottom: 8px;">Initial State</h4>
            <div class="result-metric">
                <span class="result-metric-label">Initial Phi (Average):</span>
                <span class="result-metric-value">${(data.initial_phi ?? 0).toFixed(4)}</span>
            </div>
            ${data.initial_phi_a !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Initial Phi A:</span>
                <span class="result-metric-value">${data.initial_phi_a.toFixed(4)}</span>
            </div>
            ` : ''}
            ${data.initial_phi_b !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Initial Phi B:</span>
                <span class="result-metric-value">${data.initial_phi_b.toFixed(4)}</span>
            </div>
            ` : ''}
            ${data.initial_asymmetry !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Initial Asymmetry:</span>
                <span class="result-metric-value">${data.initial_asymmetry.toFixed(4)}</span>
            </div>
            ` : ''}
        </div>
        
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
            <h4 style="font-size: 14px; color: #667eea; margin-bottom: 8px;">After Predation (divergence=${data.parameters?.divergence?.toFixed(2) || 'N/A'})</h4>
            ${data.phi_after_predation !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Phi After Predation:</span>
                <span class="result-metric-value">${data.phi_after_predation.toFixed(4)}</span>
            </div>
            ` : ''}
        </div>
        
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
            <h4 style="font-size: 14px; color: #667eea; margin-bottom: 8px;">After Phase Flip (flip_prob=${data.parameters?.flip_prob?.toFixed(2) || 'N/A'})</h4>
            ${data.phi_after_flip !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Phi After Phase Flip:</span>
                <span class="result-metric-value">${data.phi_after_flip.toFixed(4)}</span>
            </div>
            ` : ''}
        </div>
        
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
            <h4 style="font-size: 14px; color: #667eea; margin-bottom: 8px;">After Recalibration (threshold=${data.parameters?.threshold !== undefined ? data.parameters.threshold.toFixed(2) : 'N/A'})</h4>
            <div class="result-metric">
                <span class="result-metric-label">Final Phi (Average):</span>
                <span class="result-metric-value">${(data.final_phi ?? 0).toFixed(4)}</span>
            </div>
            ${data.final_phi_a !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Final Phi A:</span>
                <span class="result-metric-value">${data.final_phi_a.toFixed(4)}</span>
            </div>
            ` : ''}
            ${data.final_phi_b !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Final Phi B:</span>
                <span class="result-metric-value">${data.final_phi_b.toFixed(4)}</span>
            </div>
            ` : ''}
            <div class="result-metric">
                <span class="result-metric-label">Post-Recalibration Asymmetry:</span>
                <span class="result-metric-value">${(data.asymmetry ?? 0).toFixed(4)}</span>
            </div>
            ${data.pruning_occurred !== undefined ? `
            <div class="result-metric">
                <span class="result-metric-label">Pruning Occurred:</span>
                <span class="result-metric-value" style="color: ${data.pruning_occurred ? '#dc2626' : '#16a34a'}; font-weight: bold;">
                    ${data.pruning_occurred ? 'Yes (asymmetry > threshold)' : 'No (asymmetry ≤ threshold)'}
                </span>
            </div>
            ` : ''}
        </div>
        
        <div style="margin-top: 12px; padding-top: 12px; border-top: 2px solid #667eea;">
            <h4 style="font-size: 14px; color: #667eea; margin-bottom: 8px;">Summary</h4>
            <div class="result-metric">
                <span class="result-metric-label">Phi Delta (Change):</span>
                <span class="result-metric-value" style="color: ${(data.phi_delta ?? 0) >= 0 ? '#16a34a' : '#dc2626'}; font-weight: bold;">
                    ${(data.phi_delta ?? 0) >= 0 ? '+' : ''}${(data.phi_delta ?? 0).toFixed(4)}
                </span>
            </div>
        </div>
    `;
    
    resultDiv.innerHTML = resultsHTML;
    quantumOutput.appendChild(resultDiv);
    
    addToQuantumOutput('\n=== End of Results ===\n\n');
    quantumOutput.scrollTop = quantumOutput.scrollHeight;
}

// Add text to quantum output
function addToQuantumOutput(text, className = '') {
    const p = document.createElement('p');
    p.className = className;
    p.textContent = text;
    quantumOutput.appendChild(p);
    quantumOutput.scrollTop = quantumOutput.scrollHeight;
}

// Clear quantum results
function clearQuantumResults() {
    quantumOutput.innerHTML = '<p class="placeholder">Quantum simulation results will appear here...</p>';
    hideError();
}


