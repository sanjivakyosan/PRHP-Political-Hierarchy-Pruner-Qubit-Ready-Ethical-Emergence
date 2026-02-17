"""
Risk-Utility Recalibration Module for PRHP Framework

Provides optimization-based recalibration of risk and utility values
to achieve target equity thresholds in PRHP simulations.

Copyright © sanjivakyosan 2025
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict, Any
import warnings

try:
    from .utils import get_logger, validate_float_range
except ImportError:
    from utils import get_logger, validate_float_range

logger = get_logger()


def recalibrate_risk_utility(
    risk_values: np.ndarray,
    utility_values: np.ndarray,
    target_equity: float = 0.1,
    initial_threshold: float = 0.25,
    method: str = 'L-BFGS-B',
    max_iterations: int = 1000
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Universal recalibration function using optimization to balance risk and utility.
    
    Minimizes imbalance between risks and utilities via sensitivity analysis to achieve
    target equity. Integrates with PRHP framework's asymmetry-based equity metrics.
    
    Args:
        risk_values: Array of risk deltas (e.g., reprisal probabilities, asymmetry deltas)
        utility_values: Array of utility scores (e.g., lives saved %, fidelity scores)
        target_equity: Desired maximum equity delta (default: 0.1, matches PRHP's 0.11 threshold)
        initial_threshold: Starting risk threshold for optimization (default: 0.25)
        method: Optimization method ('L-BFGS-B', 'Nelder-Mead', 'BFGS')
        max_iterations: Maximum optimization iterations
    
    Returns:
        Tuple of:
        - new_threshold: Optimized risk threshold
        - balanced_risks: Recalibrated risk values
        - balanced_utils: Recalibrated utility values
        - metadata: Dictionary with optimization details and metrics
    
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If optimization fails
    """
    # Input validation
    if not isinstance(risk_values, np.ndarray):
        risk_values = np.array(risk_values)
    if not isinstance(utility_values, np.ndarray):
        utility_values = np.array(utility_values)
    
    if len(risk_values) != len(utility_values):
        raise ValueError(
            f"Risk and utility arrays must have same length. "
            f"Got {len(risk_values)} risks and {len(utility_values)} utilities."
        )
    
    if len(risk_values) == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Validate ranges
    if np.any(risk_values < 0) or np.any(risk_values > 1):
        logger.warning("Risk values outside [0, 1] range detected. Clamping...")
        risk_values = np.clip(risk_values, 0.0, 1.0)
    
    if np.any(utility_values < 0) or np.any(utility_values > 1):
        logger.warning("Utility values outside [0, 1] range detected. Clamping...")
        utility_values = np.clip(utility_values, 0.0, 1.0)
    
    target_equity = validate_float_range(target_equity, "target_equity", 0.0, 1.0)
    initial_threshold = validate_float_range(initial_threshold, "initial_threshold", 0.0, 1.0)
    
    logger.info(
        f"Recalibrating risk-utility: {len(risk_values)} values, "
        f"target_equity={target_equity:.3f}, initial_threshold={initial_threshold:.3f}"
    )
    
    # Store original statistics
    original_risk_mean = np.mean(risk_values)
    original_risk_std = np.std(risk_values)
    original_utility_mean = np.mean(utility_values)
    original_utility_std = np.std(utility_values)
    
    def objective(threshold: np.ndarray) -> float:
        """
        Objective function: Minimize equity imbalance.
        
        Combines:
        1. Equity metric: Standard deviation of pruned risks (should match target_equity)
        2. Utility preservation: Maintain utility mean after pruning
        3. Risk-utility balance: Correlation between risks and utilities
        """
        threshold_val = float(threshold[0])
        
        # Classify options: prune if risk > threshold
        mask = risk_values <= threshold_val
        pruned_risks = risk_values[mask]
        pruned_utils = utility_values[mask]
        
        if len(pruned_risks) == 0:
            # Penalize empty pruning (all risks above threshold)
            return 1e6
        
        # Equity metric: Standard deviation of pruned risks should match target
        risk_std = np.std(pruned_risks)
        equity_penalty = abs(risk_std - target_equity) ** 2
        
        # Utility preservation: Maintain utility mean
        utility_mean = np.mean(pruned_utils)
        utility_penalty = abs(utility_mean - original_utility_mean) ** 2
        
        # Risk-utility balance: Inverse correlation is desirable
        # (high risk should correlate with lower utility, and vice versa)
        if len(pruned_risks) > 1:
            correlation = np.corrcoef(pruned_risks, pruned_utils)[0, 1]
            # Penalize positive correlation (high risk + high utility is undesirable)
            balance_penalty = max(0, correlation) ** 2
        else:
            balance_penalty = 0.0
        
        # Weighted combination
        total_penalty = (
            2.0 * equity_penalty +      # Equity is primary goal
            1.0 * utility_penalty +     # Preserve utility
            0.5 * balance_penalty       # Risk-utility balance
        )
        
        return total_penalty
    
    # Optimize threshold
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                [initial_threshold],
                bounds=[(0.0, 1.0)],
                method=method,
                options={'maxiter': max_iterations, 'disp': False}
            )
        
        if not result.success:
            logger.warning(
                f"Optimization did not converge: {result.message}. "
                f"Using best found threshold: {result.x[0]:.4f}"
            )
        
        new_threshold = float(np.clip(result.x[0], 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}. Using initial threshold.")
        new_threshold = initial_threshold
        result = type('obj', (object,), {
            'success': False,
            'message': str(e),
            'fun': objective([initial_threshold])
        })()
    
    # Apply recalibration with PRHP-informed adjustments
    # Dampen high risks (above threshold) more aggressively
    risk_dampening_factor = 0.8  # Reduce high risks by 20%
    utility_boost_factor = 0.1   # Boost utilities for low risks
    
    balanced_risks = np.where(
        risk_values > new_threshold,
        risk_values * risk_dampening_factor,  # Dampen high risks
        risk_values  # Keep low risks unchanged
    )
    
    # Boost utilities inversely proportional to risk
    # Lower risk → higher utility boost
    risk_inverse = 1.0 - balanced_risks
    balanced_utils = utility_values + risk_inverse * utility_boost_factor
    
    # Ensure values stay in valid range
    balanced_risks = np.clip(balanced_risks, 0.0, 1.0)
    balanced_utils = np.clip(balanced_utils, 0.0, 1.0)
    
    # Compute metrics
    pruned_mask = risk_values <= new_threshold
    n_pruned = np.sum(pruned_mask)
    pruned_ratio = n_pruned / len(risk_values)
    
    final_risk_std = np.std(balanced_risks[pruned_mask]) if n_pruned > 0 else 0.0
    final_utility_mean = np.mean(balanced_utils[pruned_mask]) if n_pruned > 0 else 0.0
    
    equity_achieved = abs(final_risk_std - target_equity)
    
    metadata = {
        'optimization_success': result.success,
        'optimization_message': getattr(result, 'message', 'N/A'),
        'final_objective': float(result.fun),
        'new_threshold': new_threshold,
        'n_pruned': int(n_pruned),
        'pruned_ratio': float(pruned_ratio),
        'equity_achieved': float(equity_achieved),
        'target_equity': target_equity,
        'final_risk_std': float(final_risk_std),
        'final_utility_mean': float(final_utility_mean),
        'original_risk_mean': float(original_risk_mean),
        'original_risk_std': float(original_risk_std),
        'original_utility_mean': float(original_utility_mean),
        'original_utility_std': float(original_utility_std),
        'risk_change': float(np.mean(balanced_risks) - original_risk_mean),
        'utility_change': float(np.mean(balanced_utils) - original_utility_mean)
    }
    
    logger.info(
        f"Recalibration complete: threshold={new_threshold:.4f}, "
        f"equity_achieved={equity_achieved:.4f}, pruned={n_pruned}/{len(risk_values)}"
    )
    
    return new_threshold, balanced_risks, balanced_utils, metadata


def recalibrate_prhp_metrics(
    prhp_results: Dict[str, Dict[str, Any]],
    target_equity: float = 0.11,  # PRHP's default asymmetry threshold
    variant: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Recalibrate PRHP simulation results using risk-utility optimization.
    
    Integrates with PRHP framework by using asymmetry_delta as risk and
    mean_fidelity as utility.
    
    Args:
        prhp_results: PRHP results dict: {variant: {metrics...}}
        target_equity: Target equity threshold (default: 0.11, PRHP's asymmetry threshold)
        variant: Optional variant to recalibrate (if None, recalibrates all)
    
    Returns:
        Dictionary with recalibrated results and metadata
    """
    variants_to_process = [variant] if variant else list(prhp_results.keys())
    recalibrated = {}
    
    for v in variants_to_process:
        if v not in prhp_results:
            continue
        
        data = prhp_results[v]
        
        # Extract risk (asymmetry_delta) and utility (mean_fidelity)
        # Convert to arrays for processing
        asymmetry_delta = data.get('asymmetry_delta', 0.0)
        mean_fidelity = data.get('mean_fidelity', 0.84)
        
        # For single values, create arrays (can be extended for per-level data)
        risk_values = np.array([abs(asymmetry_delta)])  # Use absolute value for risk
        utility_values = np.array([mean_fidelity])
        
        try:
            new_threshold, balanced_risks, balanced_utils, metadata = recalibrate_risk_utility(
                risk_values=risk_values,
                utility_values=utility_values,
                target_equity=target_equity,
                initial_threshold=0.25
            )
            
            # Update results with recalibrated values
            recalibrated[v] = data.copy()
            recalibrated[v]['asymmetry_delta'] = float(balanced_risks[0]) * np.sign(asymmetry_delta)
            recalibrated[v]['mean_fidelity'] = float(balanced_utils[0])
            recalibrated[v]['recalibration'] = metadata
            
            logger.info(f"Recalibrated {v}: asymmetry={recalibrated[v]['asymmetry_delta']:.4f}, fidelity={recalibrated[v]['mean_fidelity']:.4f}")
        
        except Exception as e:
            logger.error(f"Failed to recalibrate {v}: {e}")
            recalibrated[v] = data.copy()
            recalibrated[v]['recalibration'] = {'error': str(e)}
    
    return recalibrated


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    risks = np.array([0.28, 0.08, 0.41])
    utils = np.array([0.923, 0.871, 0.63])
    
    new_thresh, bal_risks, bal_utils, metadata = recalibrate_risk_utility(
        risks, utils, target_equity=0.1
    )
    
    print(f"New Threshold: {new_thresh:.4f}")
    print(f"Balanced Risks: {bal_risks}")
    print(f"Balanced Utils: {bal_utils}")
    print(f"Metadata: {metadata}")

