"""
Historical Data Integration for PRHP Framework

Integrates historical simulation data into current PRHP simulations
to improve accuracy and robustness through weighted averaging and scaling.

Copyright © sanjivakyosan 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import logging

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()


class HistoricalDataIntegrator:
    """
    Integrates historical PRHP simulation data into current simulations.
    
    This class provides methods to:
    - Load historical data from CSV/JSON files
    - Compute weighted priors from historical data
    - Apply historical priors to current simulation results
    - Balance current and historical data with configurable weights
    """
    
    def __init__(
        self,
        historical_weight: float = 0.3,
        current_weight: float = 0.7,
        key_columns: Optional[List[str]] = None
    ):
        """
        Initialize the historical data integrator.
        
        Args:
            historical_weight: Weight for historical data (0.0-1.0). Default: 0.3
            current_weight: Weight for current data (0.0-1.0). Default: 0.7
            key_columns: Columns to balance. If None, uses PRHP default metrics.
        
        Raises:
            ValueError: If weights don't sum to approximately 1.0
        """
        if abs(historical_weight + current_weight - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0. Got historical={historical_weight}, "
                f"current={current_weight}, sum={historical_weight + current_weight}"
            )
        
        self.historical_weight = historical_weight
        self.current_weight = current_weight
        
        # Default PRHP key columns if not specified
        if key_columns is None:
            self.key_columns = [
                'mean_fidelity',
                'std',
                'asymmetry_delta',
                'novelty_gen',
                'mean_phi_delta',
                'mean_success_rate'
            ]
        else:
            self.key_columns = key_columns
        
        self.scaler = StandardScaler()
        self.historical_priors: Optional[Dict[str, float]] = None
        self.historical_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.is_fitted = False
    
    def load_historical_data(
        self,
        history_file_path: Union[str, Path],
        variant: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical data from CSV or JSON file.
        
        Args:
            history_file_path: Path to historical data file
            variant: Optional variant filter (e.g., 'ADHD-collectivist')
        
        Returns:
            DataFrame with historical data
        
        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If file doesn't exist
        """
        history_file_path = Path(history_file_path)
        
        if not history_file_path.exists():
            raise FileNotFoundError(f"Historical data file not found: {history_file_path}")
        
        try:
            if history_file_path.suffix == '.csv':
                hist_df = pd.read_csv(history_file_path)
            elif history_file_path.suffix == '.json':
                hist_df = pd.read_json(history_file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {history_file_path.suffix}. "
                    "Use .csv or .json"
                )
            
            logger.info(f"Loaded historical data: {len(hist_df)} rows from {history_file_path}")
            
            # Filter by variant if specified
            if variant and 'variant' in hist_df.columns:
                hist_df = hist_df[hist_df['variant'] == variant]
                logger.info(f"Filtered to variant '{variant}': {len(hist_df)} rows")
            
            return hist_df
        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    def fit_historical_priors(
        self,
        hist_df: pd.DataFrame,
        variant_column: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute priors (mean, std) from historical data.
        
        Args:
            hist_df: Historical data DataFrame
            variant_column: Optional column name for variant grouping
        
        Returns:
            Dictionary mapping variant -> {metric -> {mean, std}}
        """
        # Filter to only key columns that exist
        available_columns = [col for col in self.key_columns if col in hist_df.columns]
        
        if not available_columns:
            raise ValueError(
                f"None of the key columns {self.key_columns} found in historical data. "
                f"Available columns: {list(hist_df.columns)}"
            )
        
        # Remove rows with missing values in key columns
        hist_clean = hist_df[available_columns].dropna()
        
        if len(hist_clean) == 0:
            raise ValueError("No valid rows after removing missing values")
        
        logger.info(f"Computing priors from {len(hist_clean)} historical rows")
        
        # Group by variant if specified
        if variant_column and variant_column in hist_df.columns:
            stats = {}
            for variant, group in hist_df.groupby(variant_column):
                group_clean = group[available_columns].dropna()
                if len(group_clean) > 0:
                    stats[variant] = {
                        col: {
                            'mean': float(group_clean[col].mean()),
                            'std': float(group_clean[col].std()),
                            'count': len(group_clean)
                        }
                        for col in available_columns
                    }
            self.historical_stats = stats
            logger.info(f"Computed priors for {len(stats)} variants")
        else:
            # Single set of priors for all data
            stats = {
                'all': {
                    col: {
                        'mean': float(hist_clean[col].mean()),
                        'std': float(hist_clean[col].std()),
                        'count': len(hist_clean)
                    }
                    for col in available_columns
                }
            }
            self.historical_stats = stats
        
        # Fit scaler on historical data
        self.scaler.fit(hist_clean[available_columns])
        self.is_fitted = True
        
        return stats
    
    def incorporate_historical_data(
        self,
        history_file_path: Union[str, Path],
        current_data: Union[pd.DataFrame, Dict[str, Any]],
        variant: Optional[str] = None,
        variant_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Universal function to integrate historical data into current simulation models.
        
        Balances by computing weighted averages and scaling features.
        
        Args:
            history_file_path: Path to CSV/JSON with historical data
            current_data: Current scenario data (DataFrame or PRHP results dict)
            variant: Optional variant name for filtering/grouping
            variant_column: Optional column name for variant in historical data
        
        Returns:
            Balanced DataFrame with historical priors incorporated
        
        Raises:
            ValueError: If data format is invalid
            FileNotFoundError: If historical file doesn't exist
        """
        # Load historical data
        hist_df = self.load_historical_data(history_file_path, variant=variant)
        
        # Convert current_data to DataFrame if needed
        if isinstance(current_data, dict):
            # Check if this is a flat metrics dict (PRHP results per variant) or nested {variant: {metrics}}
            is_flat_metrics = any(
                k in self.key_columns and isinstance(v, (int, float))
                for k, v in list(current_data.items())[:3]
            )
            if is_flat_metrics or (variant and variant not in current_data):
                # Flat metrics dict: {mean_fidelity: 0.88, std: 0.02, ...}
                # Filter to numeric values only for DataFrame columns
                flat_metrics = {k: v for k, v in current_data.items()
                               if isinstance(v, (int, float)) and k in self.key_columns}
                if not flat_metrics:
                    flat_metrics = {k: v for k, v in current_data.items()
                                   if isinstance(v, (int, float))}
                if flat_metrics:
                    current_df = pd.DataFrame([flat_metrics])
                elif variant and variant in current_data and isinstance(current_data[variant], dict):
                    current_df = pd.DataFrame([current_data[variant]])
                else:
                    raise ValueError("Current data dict has no numeric metrics")
            elif variant and variant in current_data:
                current_df = pd.DataFrame([current_data[variant]])
            elif len(current_data) > 0:
                # Nested: use first variant
                first_variant = list(current_data.keys())[0]
                first_val = current_data[first_variant]
                if isinstance(first_val, dict):
                    current_df = pd.DataFrame([first_val])
                    if variant is None:
                        variant = first_variant
                else:
                    raise ValueError("Current data dict is empty")
            else:
                raise ValueError("Current data dict is empty")
        else:
            current_df = current_data.copy()
        
        # Compute priors from historical data
        if not self.is_fitted or self.historical_stats is None:
            self.fit_historical_priors(hist_df, variant_column=variant_column)
        
        # Get priors for this variant (or 'all' if no variant grouping)
        if variant and variant in self.historical_stats:
            priors = self.historical_stats[variant]
        elif 'all' in self.historical_stats:
            priors = self.historical_stats['all']
        else:
            # Fallback: use first available variant's priors
            priors = list(self.historical_stats.values())[0]
        
        # Apply weighted balancing to current data
        balanced_df = current_df.copy()
        
        # Get available columns that exist in both current and historical data
        available_columns = [
            col for col in self.key_columns
            if col in balanced_df.columns and col in priors
        ]
        
        if not available_columns:
            logger.warning(
                f"No matching columns found. Current: {list(balanced_df.columns)}, "
                f"Historical: {list(priors.keys())}"
            )
            # Add historical variance metric and metadata (required by prhp_core)
            balanced_df['historical_variance'] = np.mean([
                priors[col]['std'] for col in priors.keys()
            ]) if priors else 0.0
            balanced_df['historical_weight'] = self.historical_weight
            balanced_df['current_weight'] = self.current_weight
            balanced_df['historical_samples'] = (
                sum(priors[col]['count'] for col in priors.keys()) // max(1, len(priors))
                if priors else 0
            )
            return balanced_df
        
        # Scale current data using historical scaler
        try:
            current_scaled = self.scaler.transform(balanced_df[available_columns])
            current_scaled_df = pd.DataFrame(
                current_scaled,
                columns=available_columns,
                index=balanced_df.index
            )
        except Exception as e:
            logger.warning(f"Could not scale current data: {e}. Using unscaled values.")
            current_scaled_df = balanced_df[available_columns]
        
        # Apply weighted averaging: current_weight * current + historical_weight * historical_mean
        for col in available_columns:
            if col in balanced_df.columns:
                historical_mean = priors[col]['mean']
                current_value = balanced_df[col].iloc[0] if len(balanced_df) > 0 else 0.0
                
                # Weighted average
                balanced_value = (
                    self.current_weight * current_value +
                    self.historical_weight * historical_mean
                )
                balanced_df[col] = balanced_value
        
        # Add historical variance metric (mean of historical stds)
        balanced_df['historical_variance'] = np.mean([
            priors[col]['std'] for col in available_columns
        ])
        
        # Add metadata
        balanced_df['historical_weight'] = self.historical_weight
        balanced_df['current_weight'] = self.current_weight
        balanced_df['historical_samples'] = sum(
            priors[col]['count'] for col in available_columns
        ) // len(available_columns) if available_columns else 0
        
        logger.info(
            f"Balanced data: {len(balanced_df)} rows, "
            f"weights: {self.current_weight:.1%} current, {self.historical_weight:.1%} historical"
        )
        
        return balanced_df


# Convenience function for backward compatibility
def incorporate_historical_data(
    history_file_path: str,
    current_data_df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
    historical_weight: float = 0.3,
    variant: Optional[str] = None
) -> pd.DataFrame:
    """
    Universal function to integrate historical data into current simulation models.
    
    This is a convenience wrapper around HistoricalDataIntegrator.
    
    Args:
        history_file_path: Path to CSV/JSON with historical data
        current_data_df: Current scenario data DataFrame
        key_columns: Columns to balance (default: PRHP metrics)
        historical_weight: Weight for historical data (0.0-1.0). Default: 0.3
        variant: Optional variant name for filtering
    
    Returns:
        Balanced DataFrame with historical priors
    """
    integrator = HistoricalDataIntegrator(
        historical_weight=historical_weight,
        current_weight=1.0 - historical_weight,
        key_columns=key_columns
    )
    
    return integrator.incorporate_historical_data(
        history_file_path=history_file_path,
        current_data=current_data_df,
        variant=variant
    )


def incorporate_prhp_results(
    history_file_path: str,
    prhp_results: Dict[str, Dict[str, Any]],
    historical_weight: float = 0.3,
    variant: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Integrate historical data into PRHP simulation results.
    
    Args:
        history_file_path: Path to historical data file
        prhp_results: PRHP results dict: {variant: {metrics...}}
        historical_weight: Weight for historical data. Default: 0.3
        variant: Optional variant to process (if None, processes all)
    
    Returns:
        Dictionary mapping variant -> balanced DataFrame
    """
    integrator = HistoricalDataIntegrator(historical_weight=historical_weight)
    
    results = {}
    variants_to_process = [variant] if variant else list(prhp_results.keys())
    
    for v in variants_to_process:
        if v in prhp_results:
            balanced = integrator.incorporate_historical_data(
                history_file_path=history_file_path,
                current_data=prhp_results[v],
                variant=v
            )
            results[v] = balanced
    
    return results


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage with DataFrame
    current_df = pd.DataFrame({
        'mean_fidelity': [0.84],
        'std': [0.02],
        'asymmetry_delta': [0.28],
        'novelty_gen': [0.92],
        'mean_phi_delta': [0.12]
    })
    
    # Note: This requires a historical data file
    # balanced = incorporate_historical_data('historical_crises.csv', current_df)
    # print(balanced)
    
    # Example 2: With PRHP results
    prhp_results = {
        'ADHD-collectivist': {
            'mean_fidelity': 0.84,
            'std': 0.025,
            'asymmetry_delta': 0.28,
            'novelty_gen': 0.90,
            'mean_phi_delta': 0.12
        }
    }
    
    # balanced_results = incorporate_prhp_results('historical_data.json', prhp_results)
    # print(balanced_results)

