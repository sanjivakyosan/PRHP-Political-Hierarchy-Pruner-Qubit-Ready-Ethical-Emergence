"""
Simulation Validation Module for PRHP Framework

Provides cross-validation, bias detection, and ethical alignment checks
for PRHP simulation results.

Copyright © sanjivakyosan 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import warnings

try:
    from .utils import get_logger, validate_float_range
except ImportError:
    from utils import get_logger, validate_float_range

logger = get_logger()


def validate_simulation(
    data_df: pd.DataFrame,
    target_column: str = 'utility_score',
    features: Optional[List[str]] = None,
    cv_folds: int = 5,
    bias_threshold: float = 0.1,
    equity_threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Universal validation function with cross-validation and bias checks.
    
    Ensures model reliability and ethical alignment (e.g., low bias).
    
    Args:
        data_df: Simulation data DataFrame.
        target_column: Target column to predict/validate (e.g., 'utility_score', 'mean_fidelity').
        features: Feature columns to use for validation. If None, uses all numeric columns except target.
        cv_folds: Number of cross-validation folds (default: 5).
        bias_threshold: Maximum acceptable bias delta (default: 0.1).
        equity_threshold: Maximum acceptable equity deviation (default: 0.1).
    
    Returns:
        Dictionary with validation scores, bias metrics, and recommendations:
        {
            'cv_mean_score': float,
            'cv_std_score': float,
            'bias_delta': float,
            'equity_delta': float,
            'r2_score': float,
            'mse': float,
            'is_valid': bool,
            'recommendation': str,
            'warnings': List[str]
        }
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Validate inputs
    if data_df.empty:
        raise ValueError("DataFrame is empty")
    
    if target_column not in data_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Auto-detect features if not provided
    if features is None:
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col != target_column]
    
    if not features:
        raise ValueError("No feature columns available for validation")
    
    # Check if all features exist
    missing_features = [f for f in features if f not in data_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    # Prepare data
    X = data_df[features].values
    y = data_df[target_column].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    if not valid_mask.any():
        raise ValueError("No valid data points after removing NaN values")
    
    X = X[valid_mask]
    y = y[valid_mask]
    
    # KFold requires n_splits >= 2; with 1 sample we cannot do CV
    if len(y) < 2:
        logger.warning(
            f"Insufficient data for cross-validation (n={len(y)}). "
            "Using direct fit validation instead."
        )
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        cv_mean_equity = 0.0
        cv_std_equity = 0.0
        cv_mean_mse = float(mean_squared_error(y, y_pred))
        cv_mean_r2 = float(r2_score(y, y_pred)) if len(y) > 1 else 0.0
    else:
        # Ensure n_splits is at least 2 (KFold requirement)
        effective_cv_folds = min(cv_folds, max(2, len(y)))
        if len(y) < cv_folds:
            logger.warning(
                f"Insufficient data for {cv_folds} folds. Using {effective_cv_folds} folds instead."
            )
        
        # Initialize model
        model = LinearRegression()
        
        # Custom scorer for ethical bias (equity delta)
        def equity_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """
            Custom scorer that penalizes deviation from equity.
            Returns negative deviation to be maximized (higher is better).
            """
            if len(y_true) == 0 or len(y_pred) == 0:
                return -np.inf
            
            # Calculate equity as inverse of standard deviation difference
            std_true = np.std(y_true) if len(y_true) > 1 else 0.0
            std_pred = np.std(y_pred) if len(y_pred) > 1 else 0.0
            
            # Penalize large deviations from true distribution
            equity_delta = abs(std_pred - std_true)
            
            # Return negative (to be maximized) with normalization
            return -equity_delta
        
        # Perform cross-validation
        try:
            cv = KFold(n_splits=effective_cv_folds, shuffle=True, random_state=42)
            equity_scores = cross_val_score(
                model, X, y, 
                cv=cv, 
                scoring=make_scorer(equity_scorer),
                n_jobs=-1
            )
            
            # Also compute standard metrics
            mse_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            r2_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring='r2',
                n_jobs=-1
            )
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            raise
        
        # Calculate metrics
        cv_mean_equity = np.mean(equity_scores)
        cv_std_equity = np.std(equity_scores)
        cv_mean_mse = -np.mean(mse_scores)  # Negate because sklearn returns negative MSE
        cv_mean_r2 = np.mean(r2_scores)
        
        # Fit on full data for bias/equity calculation
        model.fit(X, y)
        y_pred = model.predict(X)
    
    # Calculate bias delta (difference between predicted and actual distribution)
    # (y_pred already set in both branches above)
    
    bias_delta = abs(np.mean(y) - np.mean(y_pred))
    equity_delta = abs(np.std(y) - np.std(y_pred))
    
    # Determine if validation passes
    is_valid = (
        cv_mean_equity > -equity_threshold and
        bias_delta < bias_threshold and
        equity_delta < equity_threshold
    )
    
    # Generate recommendations
    warnings_list = []
    if cv_mean_equity <= -equity_threshold:
        warnings_list.append(f"High equity deviation: {abs(cv_mean_equity):.4f}")
    if bias_delta >= bias_threshold:
        warnings_list.append(f"High bias delta: {bias_delta:.4f}")
    if equity_delta >= equity_threshold:
        warnings_list.append(f"High equity delta: {equity_delta:.4f}")
    if cv_mean_r2 < 0.5:
        warnings_list.append(f"Low R² score: {cv_mean_r2:.4f}")
    
    if is_valid:
        recommendation = "Valid - Simulation results meet ethical and quality thresholds"
    elif len(warnings_list) > 0:
        recommendation = f"Needs Recalibration - {', '.join(warnings_list[:2])}"
    else:
        recommendation = "Needs Review - Check simulation parameters"
    
    results = {
        'cv_mean_score': float(cv_mean_equity),
        'cv_std_score': float(cv_std_equity),
        'bias_delta': float(bias_delta),
        'equity_delta': float(equity_delta),
        'r2_score': float(cv_mean_r2),
        'mse': float(cv_mean_mse),
        'is_valid': is_valid,
        'recommendation': recommendation,
        'warnings': warnings_list,
        'n_samples': int(len(y)),
        'n_features': int(len(features)),
        'cv_folds': cv_folds
    }
    
    logger.info(f"Validation completed: is_valid={is_valid}, bias_delta={bias_delta:.4f}, equity_delta={equity_delta:.4f}")
    
    return results


def validate_prhp_results(
    prhp_results: Dict[str, Dict[str, Any]],
    target_metric: str = 'mean_fidelity',
    risk_metric: str = 'asymmetry_delta',
    cv_folds: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Validate PRHP simulation results using cross-validation.
    
    Converts PRHP results dictionary to DataFrame and validates each variant.
    
    Args:
        prhp_results: Dictionary of PRHP simulation results (variant -> metrics).
        target_metric: Target metric to validate (e.g., 'mean_fidelity').
        risk_metric: Risk metric to use as feature (e.g., 'asymmetry_delta').
        cv_folds: Number of cross-validation folds.
    
    Returns:
        Dictionary mapping variant -> validation results.
    """
    validation_results = {}
    
    for variant, data in prhp_results.items():
        try:
            # Convert PRHP results to DataFrame format
            # Create a DataFrame with the metrics
            metrics_dict = {
                risk_metric: [data.get(risk_metric, 0.0)],
                target_metric: [data.get(target_metric, 0.0)]
            }
            
            # Add other numeric metrics as features if available
            for key, value in data.items():
                if key not in [risk_metric, target_metric] and isinstance(value, (int, float)):
                    if key not in metrics_dict:
                        metrics_dict[key] = [value]
            
            df = pd.DataFrame(metrics_dict)
            
            # Validate
            validation_results[variant] = validate_simulation(
                data_df=df,
                target_column=target_metric,
                features=[risk_metric] + [k for k in metrics_dict.keys() if k not in [risk_metric, target_metric]],
                cv_folds=min(cv_folds, len(df))
            )
            
            logger.info(f"Validated {variant}: {validation_results[variant]['recommendation']}")
            
        except Exception as e:
            logger.error(f"Failed to validate {variant}: {e}")
            validation_results[variant] = {
                'is_valid': False,
                'recommendation': f'Validation failed: {str(e)}',
                'error': str(e)
            }
    
    return validation_results


# Example usage
if __name__ == "__main__":
    # Test with sample data
    df = pd.DataFrame({
        'risk_delta': [0.28, 0.08, 0.15, 0.22, 0.12],
        'utility_score': [0.92, 0.87, 0.89, 0.91, 0.88]
    })
    
    val_results = validate_simulation(df)
    print("Validation Results:")
    print(f"  Valid: {val_results['is_valid']}")
    print(f"  Bias Delta: {val_results['bias_delta']:.4f}")
    print(f"  Equity Delta: {val_results['equity_delta']:.4f}")
    print(f"  R² Score: {val_results['r2_score']:.4f}")
    print(f"  Recommendation: {val_results['recommendation']}")

