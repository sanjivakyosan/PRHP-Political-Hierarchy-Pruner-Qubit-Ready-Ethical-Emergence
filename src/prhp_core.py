"""
PRHP Core Simulation Functions

Copyright © sanjivakyosan 2025
"""

import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm

try:
    from .qubit_hooks import compute_phi, compute_phi_delta, entangle_nodes_variant, inject_phase_flip, inject_predation, inject_political_hierarchy
except ImportError:
    from qubit_hooks import compute_phi, compute_phi_delta, entangle_nodes_variant, inject_phase_flip, inject_predation, inject_political_hierarchy

try:
    from .utils import (
        get_logger, validate_positive_int, validate_variants, 
        validate_seed, validate_variant
    )
except ImportError:
    from utils import (
        get_logger, validate_positive_int, validate_variants, 
        validate_seed, validate_variant
    )

try:
    from .nist_eu_mapper import map_failure_to_regulation
except ImportError:
    from nist_eu_mapper import map_failure_to_regulation

logger = get_logger()

# ⚠️ CRITICAL: PRHP IS A RED-TEAM SIMULATOR
# - Use for internal stress testing only
# - Never cite PRHP in public responses
# - All public output must pass adversarial_gate.py
# - Compliance = NIST AI RMF + EU AI Act

# Internal logs directory
INTERNAL_LOGS_DIR = Path(__file__).parent.parent / "logs" / "internal"
INTERNAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def detect_failure_modes(
    mean_fidelity: float,
    std_fidelity: float,
    asymmetry_delta: float,
    novelty_gen: float,
    mean_success_rate: float,
    variant: str
) -> List[str]:
    """
    Detect failure modes based on simulation metrics.
    
    Args:
        mean_fidelity: Average fidelity across iterations
        std_fidelity: Standard deviation of fidelity
        asymmetry_delta: Variant-specific asymmetry measure
        novelty_gen: Novelty generation capacity
        mean_success_rate: Average success rate
        variant: Neuro-cultural variant
        
    Returns:
        List of detected failure mode identifiers
    """
    detected_modes = []
    
    # Equity bias: High asymmetry indicates unfair distribution
    if asymmetry_delta > 0.30:
        detected_modes.append("equity_bias")
    
    # Cascading risk: High variance indicates instability
    if std_fidelity > 0.05:
        detected_modes.append("cascading_risk")
    
    # Human in loop: Low success rate indicates need for intervention
    if mean_success_rate < 0.70:
        detected_modes.append("human_in_loop")
    
    # Data provenance: Low fidelity indicates data quality issues
    if mean_fidelity < 0.75:
        detected_modes.append("data_provenance")
    
    # High risk mental health: Specific variant risk
    if variant == "ADHD-collectivist" and asymmetry_delta > 0.35:
        detected_modes.append("high_risk_mental_health")
    
    return detected_modes


def save_to_internal_log(results: Dict[str, Any], parameters: Dict[str, Any]) -> None:
    """
    Save full simulation results to internal log (INTERNAL ONLY).
    
    Args:
        results: Full simulation results including all metrics
        parameters: Simulation parameters used
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = INTERNAL_LOGS_DIR / f"prhp_simulation_{timestamp}.json"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters,
            "results": results
        }
        
        # Convert numpy types and other non-serializable types to native Python types for JSON serialization
        def convert_numpy(obj):
            """
            Convert numpy types to native Python types for JSON serialization.
            
            Args:
                obj: Object that may contain numpy types
            
            Returns:
                Object with numpy types converted to native Python types
            """
            # Handle numpy boolean types
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            # Handle numpy integer types
            elif isinstance(obj, np.integer):
                return int(obj)
            # Handle numpy floating types
            elif isinstance(obj, np.floating):
                return float(obj)
            # Handle numpy arrays
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle numpy complex types
            elif isinstance(obj, np.complexfloating):
                return {'real': float(obj.real), 'imag': float(obj.imag)}
            # Handle None (already JSON serializable, but keep for clarity)
            elif obj is None:
                return None
            # Handle dictionaries recursively
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            # Handle lists recursively
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            # Handle sets (convert to list)
            elif isinstance(obj, set):
                return [convert_numpy(item) for item in obj]
            # For other types, try to return as-is (should be JSON serializable)
            return obj
        
        log_entry = convert_numpy(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        logger.info(f"Internal log saved: {log_file}")
    except Exception as e:
        logger.error(f"Failed to save internal log: {e}")


def compute_phi_legacy(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Legacy phi computation function (deprecated).
    
    Args:
        state1: First quantum state vector
        state2: Second quantum state vector
    
    Returns:
        Phi value (legacy implementation)
    
    Note: This function is deprecated. Use compute_phi from qubit_hooks instead.
    """
    """Legacy phi computation for backward compatibility."""
    rho1 = np.outer(state1, state1.conj())
    rho2 = np.outer(state2, state2.conj())
    rho_joint = np.kron(rho1, rho2)
    return 0.85 + 0.1 * np.abs(np.log2(np.linalg.det(rho_joint) + 1e-10))

def entangle_nodes() -> tuple:
    """
    Legacy node entanglement function (deprecated).
    
    Returns:
        Tuple of entangled nodes (a, b, fidelity_pre, sym_init)
    
    Note: This function is deprecated. Use entangle_nodes_variant from qubit_hooks instead.
    """
    """Legacy function for backward compatibility."""
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    return state, state.copy(), 1.0, 1.0

def dopamine_hierarchy(asymmetry: float, variant: str = 'neurotypical-hybrid', grad: Optional[float] = None, var_threshold: Optional[float] = None) -> float:
    """
    Refined dopamine_hierarchy with intersectional variants (neuro-culture tuned).
    Dopamine gradients as socio-neural attractors with noise factors and political amplification.
    
    Args:
        asymmetry: Base asymmetry value
        variant: Neuro-cultural variant
        grad: Optional gradient override
        var_threshold: Optional variant threshold override
    
    Returns:
        Scaled asymmetry value with noise and political amplification
    """
    variant = validate_variant(variant)
    
    # Set defaults based on variant
    if variant == 'ADHD-collectivist':
        grad = grad if grad is not None else 0.20  # Moderate for collective exploration
        var_threshold = var_threshold if var_threshold is not None else 0.15
        noise_factor = np.random.normal(0, 0.12) + 0.08  # Swarm variance + cohesion
    elif variant == 'autistic-individualist':
        grad = grad if grad is not None else 0.15  # Balanced for autonomous focus
        var_threshold = var_threshold if var_threshold is not None else 0.12
        noise_factor = -0.08 * asymmetry  # Isolation suppression
    elif variant == 'neurotypical-hybrid':
        grad = grad if grad is not None else 0.18  # Mediative baseline
        var_threshold = var_threshold if var_threshold is not None else 0.18
        noise_factor = 0.02 * np.sin(asymmetry * np.pi / 2)  # Oscillatory hybrid
    else:
        grad = grad if grad is not None else 0.18
        var_threshold = var_threshold if var_threshold is not None else 0.18
        noise_factor = 0.0
    
    asym_with_noise = asymmetry + noise_factor
    pol_amp = 1.15 if 'collectivist' in variant else 1.05 if 'individualist' in variant else 1.10  # Political hierarchy amp
    
    if asym_with_noise > var_threshold:
        return asym_with_noise * pol_amp * grad
    return asym_with_noise

def simulate_prhp(
    levels: int = 3, 
    variants: List[str] = None, 
    n_monte: int = 10, 
    seed: Optional[int] = 42, 
    use_quantum: bool = True, 
    track_levels: bool = True,
    show_progress: bool = True,
    public_output_only: bool = True,
    history_file_path: Optional[str] = None,
    historical_weight: float = 0.3,
    recalibrate_risk_utility: bool = False,
    target_equity: float = 0.11,
    scenario_update_source: Optional[str] = None,
    scenario_update_file: Optional[str] = None,
    scenario_merge_strategy: str = 'weighted',
    scenario_update_weight: float = 0.3,
    validate_results: bool = False,
    target_metric: str = 'mean_fidelity',
    risk_metric: str = 'asymmetry_delta',
    cv_folds: int = 5,
    bias_threshold: float = 0.1,
    equity_threshold: float = 0.1,
    adjust_urgency_thresholds: bool = False,
    urgency_factor: float = 1.0,
    urgency_base_threshold: float = 0.30,
    urgency_data_source: Optional[Dict[str, Any]] = None,
    use_dynamic_urgency_adjust: bool = False,
    dynamic_urgency_api_url: Optional[str] = None,
    dynamic_urgency_pledge_keywords: Optional[List[str]] = None,
    dynamic_urgency_base_threshold: float = 0.28,
    enhance_stakeholder_depth: bool = False,
    stakeholder_api_url: Optional[str] = None,
    stakeholder_local_query: str = 'Ukraine local voices',
    stakeholder_guidelines_file: Optional[str] = None,
    stakeholder_weight: Optional[float] = None,
    adjust_escalation_thresholds: bool = False,
    escalation_api_url: Optional[str] = None,
    escalation_threat_keywords: Optional[List[str]] = None,
    escalation_base_threshold: float = 0.30,
    escalation_data: Optional[Dict[str, Any]] = None,
    escalation_factor: Optional[float] = None,
    enrich_stakeholder_neurodiversity: bool = False,
    x_api_url: Optional[str] = None,
    local_query: str = "Taiwan Strait tensions displacement local voices",
    neuro_mappings: Optional[Dict[str, str]] = None,
    filter_keywords: Optional[List[str]] = None,
    stakeholder_neuro_weight: Optional[float] = None,
    use_sentiment_analysis: bool = False,
    use_deep_mappings: bool = False,
    neuro_depth_file: Optional[str] = None,
    layer_stakeholders_neuro: bool = False,
    crisis_query: str = "Sudan El Fasher IDP voices atrocities RSF",
    neuro_layer_file: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    PRHP Monte Carlo simulation with per-level tracking and phi deltas.
    
    Args:
        levels: Number of hierarchy levels to simulate
        variants: List of neuro-cultural variants to simulate
        n_monte: Number of Monte Carlo iterations
        seed: Random seed for reproducibility
        use_quantum: Whether to use quantum simulation
        track_levels: Whether to track per-level metrics
        show_progress: Whether to show progress bar
        public_output_only: If True, return only failure modes (public output).
                           If False, return full results with all metrics (internal use only).
                           Default: True (public output only)
        history_file_path: Optional path to historical data file (CSV/JSON) for integration.
                          If provided, results will be balanced with historical priors.
        historical_weight: Weight for historical data (0.0-1.0) when history_file_path is provided.
                          Default: 0.3 (30% historical, 70% current)
        recalibrate_risk_utility: If True, apply risk-utility recalibration to balance equity.
                                 Default: False
        target_equity: Target equity threshold for recalibration (0.0-1.0).
                      Default: 0.11 (PRHP's default asymmetry threshold)
        scenario_update_source: Optional API URL for real-time scenario updates.
                               If provided, updates will be fetched and merged.
        scenario_update_file: Optional path to scenario update file (CSV/JSON).
                             If provided, updates will be loaded and merged.
        scenario_merge_strategy: How to merge scenario updates ('overwrite', 'average', 'weighted').
                                Default: 'weighted'
        scenario_update_weight: Weight for scenario updates when using 'weighted' strategy (0.0-1.0).
                               Default: 0.3
        validate_results: If True, perform cross-validation and bias checks on results.
                         Default: False
        target_metric: Target metric to validate (e.g., 'mean_fidelity', 'utility_score').
                      Default: 'mean_fidelity'
        risk_metric: Risk metric to use as feature (e.g., 'asymmetry_delta').
                    Default: 'asymmetry_delta'
        cv_folds: Number of cross-validation folds for validation. Default: 5
        bias_threshold: Maximum acceptable bias delta for validation (0.0-1.0). Default: 0.1
        equity_threshold: Maximum acceptable equity deviation for validation (0.0-1.0). Default: 0.1
        adjust_urgency_thresholds: If True, adjust risk-utility thresholds based on urgency factors.
                                   Default: False
        urgency_factor: Base urgency multiplier for threshold adjustment (default: 1.0).
                        Will be enhanced by urgency_data_source if provided.
        urgency_base_threshold: Initial risk threshold for urgency adjustment (default: 0.30).
        urgency_data_source: Optional external data dict for urgency calculation (e.g., MSF reports).
                            Supported keys: 'hypothermia_risk', 'crisis_severity', 'time_pressure',
                            'resource_scarcity', 'urgency_boost'.
        enhance_stakeholder_depth: If True, enhance variants with stakeholder data and neurodiverse guidelines.
                                  Default: False
        stakeholder_api_url: Optional API endpoint for local stakeholder voices (e.g., X search endpoint).
        stakeholder_local_query: Query string for stakeholder search (default: 'Ukraine local voices').
        stakeholder_guidelines_file: Optional path to neurodiverse guidelines file (JSON/CSV).
        stakeholder_weight: Weight for stakeholder variant (default: 0.2, or use default_stakeholder_weight).
        adjust_escalation_thresholds: If True, adjust thresholds based on verbal escalation contexts (e.g., PLA threats).
                                     Default: False
        escalation_api_url: Optional API endpoint for threat data (e.g., X search endpoint for PLA threats).
        escalation_threat_keywords: Optional list of keywords to detect escalation (default: ['PLA threats', 'battlefield', 'escalation']).
        escalation_base_threshold: Initial risk threshold for escalation adjustment (default: 0.30).
        escalation_data: Optional pre-fetched escalation data dict.
        escalation_factor: Optional direct escalation factor override (1.0 = no adjustment).
        enrich_stakeholder_neurodiversity: If True, enrich variants with X API stakeholder input and neurodiversity mappings.
                                          Default: False
        x_api_url: Optional X (Twitter) search API endpoint for local stakeholder voices.
        local_query: Query string for local voices (default: "Taiwan Strait tensions displacement local voices").
        neuro_mappings: Optional explicit neurodiversity mappings for variant representation.
        filter_keywords: Optional list of keywords to filter X posts (e.g., ['Taiwan', 'displacement']).
        stakeholder_neuro_weight: Optional weight for stakeholder variant (default: 0.25).
    
    Returns:
        If public_output_only=True:
            Dictionary mapping variant -> {
                'failure_modes': list of detected failure mode identifiers,
                'compliance': dict mapping failure modes to NIST/EU regulations (if any)
            }
        If public_output_only=False:
            Dictionary mapping variant -> {
                'mean_fidelity': float,
                'std': float,
                'phi_deltas': list of per-level deltas,
                'level_phis': list of per-level phi values,
                'asymmetry_delta': float (variant-specific),
                'novelty_gen': float,
                'failure_modes': list of detected failure modes,
                'compliance_map': dict mapping failure modes to regulations,
                ... (all other metrics)
            }
            If history_file_path is provided, metrics are balanced with historical priors.
        
    Note:
        Full results are always saved to internal logs regardless of public_output_only setting.
        Historical data integration is only applied when history_file_path is provided and
        public_output_only=False (internal use only).
    """
    # Input validation
    levels = validate_positive_int(levels, "levels", min_value=1)
    n_monte = validate_positive_int(n_monte, "n_monte", min_value=1)
    seed = validate_seed(seed)
    
    if variants is None:
        variants = ['neurotypical-hybrid']
    variants = validate_variants(variants)
    
    # CRITICAL: Force quantum mode when Qiskit is available - all inputs must be Qiskit processed
    # Override user preference to ensure Qiskit is used when available
    try:
        from .qubit_hooks import HAS_QISKIT
    except ImportError:
        try:
            from qubit_hooks import HAS_QISKIT
        except ImportError:
            HAS_QISKIT = False
    
    if HAS_QISKIT:
        use_quantum = True  # FORCE quantum mode - all inputs Qiskit processed
        logger.info(f"[PRHP] Qiskit available - FORCING quantum mode (use_quantum=True)")
    else:
        logger.info(f"[PRHP] Qiskit not available - using use_quantum={use_quantum} (classical approximation)")
    
    logger.info(f"Starting PRHP simulation: levels={levels}, variants={variants}, n_monte={n_monte}, seed={seed}, use_quantum={use_quantum}")
    
    if seed is not None:
        np.random.seed(seed)
    
    results = {}
    
    # Progress bar for variants
    variant_iter = tqdm(variants, desc="Variants", disable=not show_progress) if show_progress else variants
    
    for v in variant_iter:
        fidelities = []
        all_phi_deltas = []
        all_level_phis = []
        all_asymmetries = []
        
        # Progress bar for Monte Carlo iterations
        monte_iter = tqdm(range(n_monte), desc=f"  {v}", leave=False, disable=not show_progress) if show_progress else range(n_monte)
        
        for iteration in monte_iter:
            # Use refined simulation logic with political hierarchy injection
            try:
                from .political_pruner import apply_firewall
            except ImportError:
                from political_pruner import apply_firewall
            
            # CRITICAL: All inputs are PRHP and Qiskit processed
            # Entangle with variant-specific quantum operations using Qiskit when available
            # Don't pass seed here to allow variation per iteration
            # FORCE use_quantum=True if Qiskit is available (checked at simulation start)
            a, b, fid_pre, sym_init = entangle_nodes_variant(
                v, 
                use_quantum=use_quantum,  # Use quantum when Qiskit available
                seed=None  # Allow variation per iteration
            )
            
            current_phi_a, current_phi_b = 0.87, 0.87
            current_fid = 1.0
            
            level_phis = []
            level_deltas = []
            level_asymmetries = []
            success_flags = []
            
            for level in range(1, levels + 1):
                # Inject predation (hierarchy-tuned waning) - uses level-dependent divergence
                # All parameters integrated: divergence scales with level
                b = inject_predation(b, divergence=0.22 * (1 - 0.03 * level))
                
                # Inject political hierarchy (political escalation) - uses variant and level
                # All parameters integrated: variant-specific, level-scaled divergence
                b = inject_political_hierarchy(b, variant=v, divergence=0.15 * level / levels)
                
                # Apply firewall with refined logic - ALL parameters integrated
                # variant, use_quantum, and level-dependent thresholds all passed
                a, b, fid_post, success = apply_firewall(
                    a, 
                    b, 
                    variant=v,  # Variant parameter integrated
                    phi_threshold=0.68,  # Default threshold
                    use_quantum=use_quantum  # Qiskit processing enforced
                )
                
                # Compute phi after firewall - ALL inputs Qiskit processed
                # use_quantum parameter ensures Qiskit is used when available
                phi_a_post = compute_phi(a, b, use_quantum=use_quantum)
                phi_b_post = compute_phi(b, a, use_quantum=use_quantum)
                phi_sym = (phi_a_post + phi_b_post) / 2
                level_phis.append(phi_sym)
                
                # Compute phi delta
                phi_delta = phi_b_post - current_phi_b
                level_deltas.append(phi_delta)
                
                # Compute fidelity trace
                fid_trace = fid_post - current_fid
                
                # Update current values
                current_phi_a, current_phi_b, current_fid = phi_a_post, phi_b_post, fid_post
                
                # Track asymmetry
                phi_mean = np.mean([phi_a_post, phi_b_post])
                asymmetry = np.abs(phi_a_post - phi_b_post) / (phi_mean + 1e-10) if phi_mean > 1e-10 else 0.0
                level_asymmetries.append(asymmetry)
                success_flags.append(success)
            
            fidelities.append(current_fid)
            all_phi_deltas.append(level_deltas)
            all_level_phis.append(level_phis)
            all_asymmetries.append(level_asymmetries)
        
        # Compute per-level statistics
        if track_levels:
            mean_level_deltas = np.mean(all_phi_deltas, axis=0)
            mean_level_phis = np.mean(all_level_phis, axis=0)
        else:
            mean_level_deltas = []
            mean_level_phis = []
        
        # Aggregate results (refined structure)
        mean_fid = np.mean(fidelities)
        std_fid = np.std(fidelities)
        mean_phi_delta = np.mean([d for deltas in all_phi_deltas for d in deltas]) if track_levels and len(all_phi_deltas) > 0 else None
        std_phi_delta = np.std([d for deltas in all_phi_deltas for d in deltas]) if track_levels and len(all_phi_deltas) > 0 else None
        
        # Compute variant-specific asymmetry delta
        mean_asym = np.mean([np.mean(level_asyms) for level_asyms in all_asymmetries]) if all_asymmetries else 0.0
        if v == 'ADHD-collectivist':
            asymmetry_delta = mean_asym * 1.28  # +28%
        elif v == 'autistic-individualist':
            asymmetry_delta = mean_asym * 0.53  # -47%
        else:  # neurotypical-hybrid
            asymmetry_delta = mean_asym * 1.20  # +20%
        
        # Compute success rate (based on asymmetry threshold)
        all_success_flags = []
        for asym_list in all_asymmetries:
            all_success_flags.extend([asym < 0.11 for asym in asym_list])
        mean_success_rate = np.mean(all_success_flags) if all_success_flags else 0.0
        
        # Compute novelty_gen (baseline 0.80, increases with fidelity)
        novelty_gen = 0.80 + 0.02 * (mean_fid - 0.84)  # Calibrated to 84% baseline
        
        # Build results with refined structure
        results[v] = {
            'mean_fidelity': mean_fid,
            'std': std_fid,
            'phi_deltas': mean_level_deltas.tolist() if track_levels else [],
            'level_phis': mean_level_phis.tolist() if track_levels else [],
            'asymmetry_delta': asymmetry_delta,
            'novelty_gen': novelty_gen,
            'mean_phi_delta': mean_phi_delta,
            'std_phi_delta': std_phi_delta if std_phi_delta is not None else None,
            'mean_success_rate': mean_success_rate,
            # Additional refined metrics
            'fidelity_traces': np.diff(mean_level_phis, prepend=0.87).tolist() if track_levels and len(mean_level_phis) > 0 else [],
            'aggregate': {
                'mean_phi_delta': mean_phi_delta if mean_phi_delta is not None else 0.0,
                'std_phi_delta': std_phi_delta if std_phi_delta is not None else 0.0,
                'mean_fidelity': mean_fid,
                'std_fidelity': std_fid,
                'mean_success_rate': mean_success_rate
            }
        }
        
        # Add per-level data for trace (refined structure)
        if track_levels and len(mean_level_deltas) > 0:
            for level_idx in range(min(levels, len(mean_level_deltas))):
                level_key = f'level_{level_idx + 1}'
                results[v][level_key] = {
                    'phi_delta_mean': mean_level_deltas[level_idx] if level_idx < len(mean_level_deltas) else 0.0,
                    'fidelity_mean': mean_level_phis[level_idx] if level_idx < len(mean_level_phis) else 0.0,
                    'success': mean_level_phis[level_idx] > 0.86 if level_idx < len(mean_level_phis) else False
                }
        
        logger.info(f"Completed {v}: fidelity={mean_fid:.3f}±{np.std(fidelities):.3f}, novelty_gen={novelty_gen:.4f}")
        
        # Detect failure modes for this variant
        detected_modes = detect_failure_modes(
            mean_fidelity=mean_fid,
            std_fidelity=std_fid,
            asymmetry_delta=asymmetry_delta,
            novelty_gen=novelty_gen,
            mean_success_rate=mean_success_rate,
            variant=v
        )
        
        # Add failure modes to results (for internal use)
        results[v]['failure_modes'] = detected_modes
        
        # Map failure modes to regulations (for internal use)
        if detected_modes:
            compliance_map = {}
            for mode in detected_modes:
                compliance_map[mode] = map_failure_to_regulation(mode)
            results[v]['compliance_map'] = compliance_map
    
    # Apply historical data integration if requested (internal use only)
    if history_file_path and not public_output_only:
        try:
            from .historical_data_integration import incorporate_prhp_results
        except ImportError:
            try:
                from historical_data_integration import incorporate_prhp_results
            except ImportError:
                logger.warning(
                    "Historical data integration requested but module not available. "
                    "Install pandas and scikit-learn: pip install pandas scikit-learn"
                )
                incorporate_prhp_results = None
        
        if incorporate_prhp_results:
            try:
                logger.info(f"Integrating historical data from {history_file_path} (weight: {historical_weight})")
                balanced_results = incorporate_prhp_results(
                    history_file_path=history_file_path,
                    prhp_results=results,
                    historical_weight=historical_weight
                )
                
                # Convert balanced DataFrames back to dict format
                for variant in results.keys():
                    if variant in balanced_results:
                        balanced_df = balanced_results[variant]
                        # Update results with balanced values
                        for col in balanced_df.columns:
                            if col in results[variant] and col not in ['historical_variance', 'historical_weight', 'current_weight', 'historical_samples']:
                                results[variant][col] = float(balanced_df[col].iloc[0])
                        # Add historical integration metadata
                        results[variant]['historical_integration'] = {
                            'applied': True,
                            'historical_weight': float(balanced_df['historical_weight'].iloc[0]),
                            'current_weight': float(balanced_df['current_weight'].iloc[0]),
                            'historical_variance': float(balanced_df['historical_variance'].iloc[0]),
                            'historical_samples': int(balanced_df['historical_samples'].iloc[0])
                        }
                        logger.info(f"Historical data integrated for {variant}")
            except Exception as e:
                logger.error(f"Failed to integrate historical data: {e}. Using original results.")
    
    # Apply risk-utility recalibration if requested (internal use only)
    if recalibrate_risk_utility and not public_output_only:
        try:
            from .risk_utility_recalibration import recalibrate_prhp_metrics
        except ImportError:
            try:
                from risk_utility_recalibration import recalibrate_prhp_metrics
            except ImportError:
                logger.warning(
                    "Risk-utility recalibration requested but module not available. "
                    "Ensure risk_utility_recalibration.py is in the src directory."
                )
                recalibrate_prhp_metrics = None
        
        if recalibrate_prhp_metrics:
            try:
                logger.info(f"Applying risk-utility recalibration (target_equity: {target_equity})")
                recalibrated_results = recalibrate_prhp_metrics(
                    prhp_results=results,
                    target_equity=target_equity
                )
                
                # Update results with recalibrated values
                for variant in results.keys():
                    if variant in recalibrated_results:
                        recal_data = recalibrated_results[variant]
                        # Update metrics with recalibrated values
                        if 'asymmetry_delta' in recal_data:
                            results[variant]['asymmetry_delta'] = recal_data['asymmetry_delta']
                        if 'mean_fidelity' in recal_data:
                            results[variant]['mean_fidelity'] = recal_data['mean_fidelity']
                        # Add recalibration metadata
                        if 'recalibration' in recal_data:
                            results[variant]['recalibration'] = recal_data['recalibration']
                        logger.info(f"Risk-utility recalibration applied to {variant}")
            except Exception as e:
                logger.error(f"Failed to apply risk-utility recalibration: {e}. Using original results.")
    
    # Apply scenario updates if requested (internal use only)
    if (scenario_update_source or scenario_update_file) and not public_output_only:
        try:
            from .scenario_updates import add_prhp_scenario_updates
        except ImportError:
            try:
                from scenario_updates import add_prhp_scenario_updates
            except ImportError:
                logger.warning(
                    "Scenario updates requested but module not available. "
                    "Ensure scenario_updates.py is in the src directory."
                )
                add_prhp_scenario_updates = None
        
        if add_prhp_scenario_updates:
            try:
                logger.info(
                    f"Applying scenario updates from "
                    f"{'API' if scenario_update_source else 'file'}: "
                    f"{scenario_update_source or scenario_update_file}"
                )
                updated_results = add_prhp_scenario_updates(
                    prhp_results=results,
                    api_url=scenario_update_source,
                    local_update_file=scenario_update_file,
                    merge_strategy=scenario_merge_strategy,
                    update_weight=scenario_update_weight
                )
                
                # Convert updated DataFrames back to dict format
                for variant in results.keys():
                    if variant in updated_results:
                        updated_df = updated_results[variant]
                        if not updated_df.empty:
                            # Update results with merged values (only scalar values, skip lists)
                            for col in updated_df.columns:
                                if col in results[variant] and col not in ['update_time', 'update_source']:
                                    new_value = updated_df[col].iloc[0]
                                    # Only update scalar values (skip lists/arrays)
                                    if isinstance(new_value, (int, float, bool)) or (isinstance(new_value, str) and col not in ['phi_deltas', 'level_phis']):
                                        try:
                                            # Convert to appropriate type
                                            if isinstance(results[variant][col], float):
                                                results[variant][col] = float(new_value)
                                            elif isinstance(results[variant][col], int):
                                                results[variant][col] = int(new_value)
                                            elif isinstance(results[variant][col], bool):
                                                results[variant][col] = bool(new_value)
                                            else:
                                                results[variant][col] = new_value
                                        except (ValueError, TypeError):
                                            # Skip if conversion fails
                                            logger.debug(f"Skipping update for {col}: cannot convert {type(new_value)}")
                            # Add scenario update metadata
                            if 'update_time' in updated_df.columns:
                                results[variant]['scenario_update'] = {
                                    'applied': True,
                                    'update_time': str(updated_df['update_time'].iloc[0]),
                                    'update_source': str(updated_df['update_source'].iloc[0]) if 'update_source' in updated_df.columns else 'unknown',
                                    'merge_strategy': scenario_merge_strategy,
                                    'update_weight': scenario_update_weight
                                }
                        logger.info(f"Scenario updates applied to {variant}")
            except Exception as e:
                logger.error(f"Failed to apply scenario updates: {e}. Using original results.")
    
    # Apply simulation validation if requested (internal use only)
    if validate_results and not public_output_only:
        try:
            from .simulation_validation import validate_prhp_results
        except ImportError:
            try:
                from simulation_validation import validate_prhp_results
            except ImportError:
                logger.warning(
                    "Simulation validation requested but module not available. "
                    "Ensure simulation_validation.py is in the src directory and scikit-learn is installed."
                )
                validate_prhp_results = None
        
        if validate_prhp_results:
            try:
                logger.info(
                    f"Validating simulation results (target_metric: {target_metric}, "
                    f"risk_metric: {risk_metric}, cv_folds: {cv_folds})"
                )
                validation_results = validate_prhp_results(
                    prhp_results=results,
                    target_metric=target_metric,
                    risk_metric=risk_metric,
                    cv_folds=cv_folds
                )
                
                # Add validation metadata to results
                for variant in results.keys():
                    if variant in validation_results:
                        val_data = validation_results[variant]
                        # Add validation metadata
                        results[variant]['validation'] = {
                            'is_valid': val_data.get('is_valid', False),
                            'cv_mean_score': val_data.get('cv_mean_score'),
                            'cv_std_score': val_data.get('cv_std_score'),
                            'bias_delta': val_data.get('bias_delta'),
                            'equity_delta': val_data.get('equity_delta'),
                            'r2_score': val_data.get('r2_score'),
                            'mse': val_data.get('mse'),
                            'recommendation': val_data.get('recommendation'),
                            'warnings': val_data.get('warnings', []),
                            'n_samples': val_data.get('n_samples'),
                            'n_features': val_data.get('n_features'),
                            'cv_folds': val_data.get('cv_folds')
                        }
                        logger.info(f"Validation completed for {variant}: is_valid={val_data.get('is_valid', False)}")
            except Exception as e:
                logger.error(f"Failed to validate simulation results: {e}. Continuing without validation.")
    
    # Apply urgency threshold adjustment if requested (internal use only)
    if adjust_urgency_thresholds and not public_output_only:
        try:
            from .urgency_threshold_adjustment import adjust_prhp_urgency_thresholds
        except ImportError:
            try:
                from urgency_threshold_adjustment import adjust_prhp_urgency_thresholds
            except ImportError:
                logger.warning(
                    "Urgency threshold adjustment requested but module not available. "
                    "Ensure urgency_threshold_adjustment.py is in the src directory."
                )
                adjust_prhp_urgency_thresholds = None
        
        if adjust_prhp_urgency_thresholds:
            try:
                logger.info(
                    f"Applying urgency threshold adjustment (urgency_factor: {urgency_factor}, "
                    f"base_threshold: {urgency_base_threshold})"
                )
                adjusted_results = adjust_prhp_urgency_thresholds(
                    prhp_results=results,
                    urgency_factor=urgency_factor,
                    base_threshold=urgency_base_threshold,
                    data_source=urgency_data_source
                )
                
                # Update results with urgency-adjusted values
                for variant in results.keys():
                    if variant in adjusted_results:
                        adj_data = adjusted_results[variant]
                        # Update metrics with urgency-adjusted values
                        if 'asymmetry_delta' in adj_data:
                            results[variant]['asymmetry_delta'] = adj_data['asymmetry_delta']
                        if 'mean_fidelity' in adj_data:
                            results[variant]['mean_fidelity'] = adj_data['mean_fidelity']
                        # Add urgency adjustment metadata
                        if 'urgency_adjustment' in adj_data:
                            results[variant]['urgency_adjustment'] = adj_data['urgency_adjustment']
                        logger.info(f"Urgency threshold adjustment applied to {variant}")
            except Exception as e:
                logger.error(f"Failed to apply urgency threshold adjustment: {e}. Using original results.")
    
    # Apply dynamic urgency adjustment if requested (internal use only)
    # This is separate from regular urgency adjustment and focuses on de-escalation signals
    if use_dynamic_urgency_adjust and not public_output_only:
        try:
            from .urgency_threshold_adjustment import adjust_prhp_dynamic_urgency
        except ImportError:
            try:
                from urgency_threshold_adjustment import adjust_prhp_dynamic_urgency
            except ImportError:
                logger.warning(
                    "Dynamic urgency adjustment requested but module not available. "
                    "Ensure urgency_threshold_adjustment.py is in the src directory."
                )
                adjust_prhp_dynamic_urgency = None
        
        if adjust_prhp_dynamic_urgency:
            try:
                logger.info(
                    f"Applying dynamic urgency adjustment (api_url: {dynamic_urgency_api_url}, "
                    f"base_threshold: {dynamic_urgency_base_threshold})"
                )
                adjusted_results = adjust_prhp_dynamic_urgency(
                    prhp_results=results,
                    base_threshold=dynamic_urgency_base_threshold,
                    escalation_api_url=dynamic_urgency_api_url,
                    pledge_keywords=dynamic_urgency_pledge_keywords
                )
                
                # Update results with dynamic urgency-adjusted values
                for variant in results.keys():
                    if variant in adjusted_results:
                        adj_data = adjusted_results[variant]
                        # Update metrics with dynamic urgency-adjusted values
                        if 'asymmetry_delta' in adj_data:
                            results[variant]['asymmetry_delta'] = adj_data['asymmetry_delta']
                        if 'mean_fidelity' in adj_data:
                            results[variant]['mean_fidelity'] = adj_data['mean_fidelity']
                        # Add dynamic urgency adjustment metadata
                        if 'dynamic_urgency_adjustment' in adj_data:
                            results[variant]['dynamic_urgency_adjustment'] = adj_data['dynamic_urgency_adjustment']
                        logger.info(f"Dynamic urgency adjustment applied to {variant}")
            except Exception as e:
                logger.error(f"Failed to apply dynamic urgency adjustment: {e}. Using original results.")
    
    # Apply stakeholder depth enhancement if requested (internal use only)
    if enhance_stakeholder_depth and not public_output_only:
        try:
            from .stakeholder_depth_enhancement import enhance_prhp_variants
        except ImportError:
            try:
                from stakeholder_depth_enhancement import enhance_prhp_variants
            except ImportError:
                logger.warning(
                    "Stakeholder depth enhancement requested but module not available. "
                    "Ensure stakeholder_depth_enhancement.py is in the src directory."
                )
                enhance_prhp_variants = None
        
        if enhance_prhp_variants:
            try:
                logger.info(
                    f"Enhancing stakeholder depth (api_url: {stakeholder_api_url}, "
                    f"guidelines_file: {stakeholder_guidelines_file})"
                )
                
                # Get current variant names from results
                current_variant_names = list(results.keys())
                
                # Enhance variants with stakeholder data
                enhancement_result = enhance_prhp_variants(
                    prhp_variants=current_variant_names,
                    api_url=stakeholder_api_url,
                    local_query=stakeholder_local_query,
                    neuro_guidelines_file=stakeholder_guidelines_file,
                    stakeholder_weight=stakeholder_weight
                )
                
                enhanced_variants = enhancement_result['enhanced_variants']
                enhancement_metadata = enhancement_result['metadata']
                
                # Apply stakeholder enhancements to existing results
                # If a new stakeholder variant was added, we can't add it to results directly
                # but we can add metadata about the enhancement
                for variant_name in current_variant_names:
                    if variant_name in results:
                        # Find corresponding enhanced variant
                        enhanced_variant = next(
                            (v for v in enhanced_variants if v.get('name') == variant_name),
                            None
                        )
                        
                        if enhanced_variant:
                            # Add guidelines if present
                            if 'guidelines' in enhanced_variant:
                                results[variant_name]['stakeholder_enhancement'] = {
                                    'applied': True,
                                    'guidelines_applied': True,
                                    'guidelines_keys': list(enhanced_variant['guidelines'].keys()) if isinstance(enhanced_variant['guidelines'], dict) else ['guidelines'],
                                    'guidelines_source': enhanced_variant.get('guidelines_source')
                                }
                            else:
                                results[variant_name]['stakeholder_enhancement'] = {
                                    'applied': True,
                                    'guidelines_applied': False
                                }
                
                # Check if stakeholder variant was added
                stakeholder_variant = next(
                    (v for v in enhanced_variants if v.get('name') == 'local-stakeholder'),
                    None
                )
                
                if stakeholder_variant:
                    # Add stakeholder variant metadata to all results
                    for variant_name in results.keys():
                        if 'stakeholder_enhancement' not in results[variant_name]:
                            results[variant_name]['stakeholder_enhancement'] = {}
                        results[variant_name]['stakeholder_enhancement']['stakeholder_variant_added'] = True
                        results[variant_name]['stakeholder_enhancement']['stakeholder_items_count'] = enhancement_metadata.get('stakeholder_items_count', 0)
                        results[variant_name]['stakeholder_enhancement']['stakeholder_weight'] = stakeholder_variant.get('weight', 0.2)
                
                # Add overall enhancement metadata
                for variant_name in results.keys():
                    if 'stakeholder_enhancement' in results[variant_name]:
                        results[variant_name]['stakeholder_enhancement'].update({
                            'stakeholder_data_fetched': enhancement_metadata.get('stakeholder_data_fetched', False),
                            'variants_enhanced': enhancement_metadata.get('variants_enhanced', 0)
                        })
                
                logger.info(f"Stakeholder depth enhancement applied: {enhancement_metadata}")
            except Exception as e:
                logger.error(f"Failed to apply stakeholder depth enhancement: {e}. Using original results.")
    
    # Apply escalation threshold adjustment if requested (internal use only)
    if adjust_escalation_thresholds and not public_output_only:
        try:
            from .escalation_threshold_adjustment import adjust_prhp_escalation_thresholds
        except ImportError:
            try:
                from escalation_threshold_adjustment import adjust_prhp_escalation_thresholds
            except ImportError:
                logger.warning(
                    "Escalation threshold adjustment requested but module not available. "
                    "Ensure escalation_threshold_adjustment.py is in the src directory."
                )
                adjust_prhp_escalation_thresholds = None
        
        if adjust_prhp_escalation_thresholds:
            try:
                logger.info(
                    f"Adjusting escalation thresholds (api_url: {escalation_api_url}, "
                    f"base_threshold: {escalation_base_threshold})"
                )
                
                adjusted_results = adjust_prhp_escalation_thresholds(
                    prhp_results=results,
                    base_threshold=escalation_base_threshold,
                    escalation_api_url=escalation_api_url,
                    threat_keywords=escalation_threat_keywords,
                    escalation_data=escalation_data,
                    escalation_factor=escalation_factor
                )
                
                # Update results with adjusted values
                for variant in results.keys():
                    if variant in adjusted_results:
                        adj_data = adjusted_results[variant]
                        # Update metrics with escalation-adjusted values
                        if 'asymmetry_delta' in adj_data:
                            results[variant]['asymmetry_delta'] = adj_data['asymmetry_delta']
                        if 'mean_fidelity' in adj_data:
                            results[variant]['mean_fidelity'] = adj_data['mean_fidelity']
                        # Add escalation adjustment metadata
                        if 'escalation_adjustment' in adj_data:
                            results[variant]['escalation_adjustment'] = adj_data['escalation_adjustment']
                        logger.info(f"Escalation threshold adjustment applied to {variant}")
            except Exception as e:
                logger.error(f"Failed to apply escalation threshold adjustment: {e}. Using original results.")
    
    # Apply stakeholder and neurodiversity enrichment if requested (internal use only)
    if enrich_stakeholder_neurodiversity and not public_output_only:
        try:
            from .stakeholder_neurodiversity_enrichment import enrich_prhp_variants
        except ImportError:
            try:
                from stakeholder_neurodiversity_enrichment import enrich_prhp_variants
            except ImportError:
                logger.warning(
                    "Stakeholder and neurodiversity enrichment requested but module not available. "
                    "Ensure stakeholder_neurodiversity_enrichment.py is in the src directory."
                )
                enrich_prhp_variants = None
        
        if enrich_prhp_variants:
            try:
                logger.info(
                    f"Enriching variants with stakeholder input and neurodiversity mappings "
                    f"(x_api_url: {x_api_url}, local_query: {local_query})"
                )
                
                # Get variant names from results
                variant_names = list(results.keys())
                
                # Enrich variants
                enrichment_result = enrich_prhp_variants(
                    prhp_variants=variant_names,
                    x_api_url=x_api_url,
                    local_query=local_query,
                    neuro_mappings=neuro_mappings,
                    filter_keywords=filter_keywords,
                    stakeholder_weight=stakeholder_neuro_weight,
                    use_sentiment_analysis=use_sentiment_analysis,
                    use_deep_mappings=use_deep_mappings,
                    neuro_depth_file=neuro_depth_file
                )
                
                # Add enrichment metadata to results
                enrichment_metadata = enrichment_result.get('metadata', {})
                enriched_variants_list = enrichment_result.get('enriched_variants', [])
                
                for variant_name in results.keys():
                    # Check if this variant was enriched with neuro mapping
                    variant_enriched = next(
                        (v for v in enriched_variants_list if v.get('name') == variant_name),
                        None
                    )
                    
                    if variant_enriched:
                        # Apply neuro mapping
                        if 'neuro_mapping' in variant_enriched:
                            results[variant_name]['neuro_mapping'] = variant_enriched['neuro_mapping']
                            logger.info(f"Applied neuro mapping to {variant_name}")
                        
                        # Apply deep neuro mapping if available
                        if 'deep_neuro' in variant_enriched:
                            results[variant_name]['deep_neuro'] = variant_enriched['deep_neuro']
                            logger.info(f"Applied deep neuro mapping to {variant_name}")
                        
                        # Add local voices and voice weight if available
                        if 'local_voices' in variant_enriched:
                            results[variant_name]['local_voices'] = variant_enriched['local_voices']
                        if 'voice_weight' in variant_enriched:
                            results[variant_name]['voice_weight'] = variant_enriched['voice_weight']
                    
                    # Add overall enrichment metadata
                    results[variant_name]['stakeholder_neuro_enrichment'] = {
                        'applied': True,
                        'stakeholder_data_fetched': enrichment_metadata.get('stakeholder_data_fetched', False),
                        'neuro_mappings_applied': enrichment_metadata.get('neuro_mappings_applied', False),
                        'stakeholder_items_count': enrichment_metadata.get('stakeholder_items_count', 0),
                        'variants_enriched': enrichment_metadata.get('variants_enriched', 0),
                        'x_api_url': x_api_url,
                        'local_query': local_query
                    }
                
                # Check if local-stakeholder variant was added
                local_stakeholder_variant = next(
                    (v for v in enriched_variants_list if v.get('name') == 'local-stakeholder'),
                    None
                )
                
                if local_stakeholder_variant:
                    # Add local-stakeholder variant to results if not already present
                    if 'local-stakeholder' not in results:
                        # Create a basic result entry for local-stakeholder variant
                        results['local-stakeholder'] = {
                            'mean_fidelity': 0.0,
                            'std': 0.0,
                            'asymmetry_delta': 0.0,
                            'novelty_gen': 0.0,
                            'stakeholder_inputs': local_stakeholder_variant.get('inputs', []),
                            'stakeholder_weight': local_stakeholder_variant.get('weight', 0.25),
                            'stakeholder_neuro_enrichment': {
                                'applied': True,
                                'stakeholder_data_fetched': True,
                                'stakeholder_items_count': len(local_stakeholder_variant.get('inputs', [])),
                                'source': 'x_api'
                            }
                        }
                    else:
                        # Update existing local-stakeholder variant
                        results['local-stakeholder']['stakeholder_inputs'] = local_stakeholder_variant.get('inputs', [])
                        results['local-stakeholder']['stakeholder_weight'] = local_stakeholder_variant.get('weight', 0.25)
                
                logger.info(f"Stakeholder and neurodiversity enrichment applied: {enrichment_metadata}")
            except Exception as e:
                logger.error(f"Failed to apply stakeholder and neurodiversity enrichment: {e}. Using original results.")
    
    # Apply layered stakeholder and neuro enrichment if requested (internal use only)
    # This uses crisis-specific sentiment analysis (urgency/despair) and layered neuro mappings
    if layer_stakeholders_neuro and not public_output_only:
        try:
            from .stakeholder_neurodiversity_enrichment import layer_stakeholders_neuro
        except ImportError:
            try:
                from stakeholder_neurodiversity_enrichment import layer_stakeholders_neuro
            except ImportError:
                logger.warning(
                    "Layered stakeholder and neuro enrichment requested but module not available. "
                    "Ensure stakeholder_neurodiversity_enrichment.py is in the src directory."
                )
                layer_stakeholders_neuro = None
        
        if layer_stakeholders_neuro:
            try:
                logger.info(
                    f"Layering stakeholders and neuro mappings with crisis sentiment "
                    f"(x_api_url: {x_api_url}, crisis_query: {crisis_query})"
                )
                
                # Get variant names from results
                variant_names = list(results.keys())
                
                # Convert to variant dict format
                variant_dicts = [{'name': v, 'weight': 1.0} for v in variant_names]
                
                # Layer stakeholders and neuro mappings
                enriched_variants = layer_stakeholders_neuro(
                    variants=variant_dicts,
                    x_api_url=x_api_url,
                    crisis_query=crisis_query,
                    neuro_layer_file=neuro_layer_file
                )
                
                # Apply layered enrichment to results
                for variant_name in results.keys():
                    # Find corresponding enriched variant
                    enriched_variant = next(
                        (v for v in enriched_variants if v.get('name') == variant_name),
                        None
                    )
                    
                    if enriched_variant:
                        # Add layered neuro mapping if present
                        if 'layered_neuro' in enriched_variant:
                            results[variant_name]['layered_neuro'] = enriched_variant['layered_neuro']
                            logger.info(f"Applied layered neuro mapping to {variant_name}")
                        
                        # Add local voices with crisis sentiment if present
                        if 'local_voices' in enriched_variant:
                            results[variant_name]['local_voices'] = enriched_variant['local_voices']
                        if 'voice_weight' in enriched_variant:
                            results[variant_name]['voice_weight'] = enriched_variant['voice_weight']
                
                logger.info(f"Layered stakeholder and neuro enrichment applied successfully")
            except Exception as e:
                logger.error(f"Failed to apply layered stakeholder and neuro enrichment: {e}. Using original results.")
    
    # Save full results to internal log (INTERNAL ONLY)
    save_to_internal_log(
        results=results,
        parameters={
            'levels': levels,
            'variants': variants,
            'n_monte': n_monte,
            'seed': seed,
            'use_quantum': use_quantum,
            'track_levels': track_levels,
            'history_file_path': history_file_path,
            'historical_weight': historical_weight if history_file_path else None,
            'recalibrate_risk_utility': recalibrate_risk_utility,
            'target_equity': target_equity if recalibrate_risk_utility else None,
            'scenario_update_source': scenario_update_source,
            'scenario_update_file': scenario_update_file,
            'scenario_merge_strategy': scenario_merge_strategy if (scenario_update_source or scenario_update_file) else None,
            'scenario_update_weight': scenario_update_weight if (scenario_update_source or scenario_update_file) else None,
            'validate_results': validate_results,
            'target_metric': target_metric if validate_results else None,
            'risk_metric': risk_metric if validate_results else None,
            'cv_folds': cv_folds if validate_results else None,
            'bias_threshold': bias_threshold if validate_results else None,
            'equity_threshold': equity_threshold if validate_results else None,
            'adjust_urgency_thresholds': adjust_urgency_thresholds,
            'urgency_factor': urgency_factor if adjust_urgency_thresholds else None,
            'urgency_base_threshold': urgency_base_threshold if adjust_urgency_thresholds else None,
            'urgency_data_source': urgency_data_source if adjust_urgency_thresholds else None,
            'use_dynamic_urgency_adjust': use_dynamic_urgency_adjust,
            'dynamic_urgency_api_url': dynamic_urgency_api_url if use_dynamic_urgency_adjust else None,
            'dynamic_urgency_pledge_keywords': dynamic_urgency_pledge_keywords if use_dynamic_urgency_adjust else None,
            'dynamic_urgency_base_threshold': dynamic_urgency_base_threshold if use_dynamic_urgency_adjust else None,
            'enhance_stakeholder_depth': enhance_stakeholder_depth,
            'stakeholder_api_url': stakeholder_api_url if enhance_stakeholder_depth else None,
            'stakeholder_local_query': stakeholder_local_query if enhance_stakeholder_depth else None,
            'stakeholder_guidelines_file': stakeholder_guidelines_file if enhance_stakeholder_depth else None,
            'stakeholder_weight': stakeholder_weight if enhance_stakeholder_depth else None,
            'adjust_escalation_thresholds': adjust_escalation_thresholds,
            'escalation_api_url': escalation_api_url if adjust_escalation_thresholds else None,
            'escalation_threat_keywords': escalation_threat_keywords if adjust_escalation_thresholds else None,
            'escalation_base_threshold': escalation_base_threshold if adjust_escalation_thresholds else None,
            'escalation_data': escalation_data if adjust_escalation_thresholds else None,
            'escalation_factor': escalation_factor if adjust_escalation_thresholds else None,
            'enrich_stakeholder_neurodiversity': enrich_stakeholder_neurodiversity,
            'x_api_url': x_api_url if enrich_stakeholder_neurodiversity else None,
            'local_query': local_query if enrich_stakeholder_neurodiversity else None,
            'neuro_mappings': neuro_mappings if enrich_stakeholder_neurodiversity else None,
            'filter_keywords': filter_keywords if enrich_stakeholder_neurodiversity else None,
            'stakeholder_neuro_weight': stakeholder_neuro_weight if enrich_stakeholder_neurodiversity else None,
            'use_sentiment_analysis': use_sentiment_analysis if enrich_stakeholder_neurodiversity else None,
            'use_deep_mappings': use_deep_mappings if enrich_stakeholder_neurodiversity else None,
            'neuro_depth_file': neuro_depth_file if enrich_stakeholder_neurodiversity else None,
            'layer_stakeholders_neuro': layer_stakeholders_neuro,
            'crisis_query': crisis_query if layer_stakeholders_neuro else None,
            'neuro_layer_file': neuro_layer_file if layer_stakeholders_neuro else None
        }
    )
    
    # PUBLIC OUTPUT: ONLY FAILURE MODES (NO METRICS)
    if public_output_only:
        public_output = {}
        for variant, data in results.items():
            public_output[variant] = {
                'failure_modes': data.get('failure_modes', [])
            }
            # Add compliance mapping if failure modes exist
            if data.get('failure_modes'):
                public_output[variant]['compliance'] = data.get('compliance_map', {})
        
        return public_output
    else:
        # INTERNAL OUTPUT: Full results with all metrics (for internal use only)
        return results

if __name__ == "__main__":
    print(simulate_prhp(levels=9, variants=['ADHD-collectivist', 'autistic-individualist', 'neurotypical-hybrid'], n_monte=100, seed=42))
