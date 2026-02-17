"""
Test script for historical data integration with PRHP framework.

This demonstrates how to integrate historical simulation data
into current PRHP simulations.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from historical_data_integration import (
    incorporate_historical_data,
    incorporate_prhp_results,
    HistoricalDataIntegrator
)

# Try to import PRHP (may not be available)
try:
    from prhp_core import simulate_prhp
    PRHP_AVAILABLE = True
except ImportError:
    PRHP_AVAILABLE = False
    print("PRHP framework not available. Using mock data for demonstration.")


def create_sample_historical_data(filename: str = "sample_historical_data.csv"):
    """Create a sample historical data file for testing."""
    data = {
        'variant': [
            'ADHD-collectivist',
            'ADHD-collectivist',
            'autistic-individualist',
            'autistic-individualist',
            'neurotypical-hybrid',
            'neurotypical-hybrid'
        ],
        'mean_fidelity': [0.84, 0.83, 0.82, 0.81, 0.85, 0.84],
        'std': [0.025, 0.024, 0.023, 0.022, 0.026, 0.025],
        'asymmetry_delta': [0.28, 0.27, -0.47, -0.46, 0.20, 0.19],
        'novelty_gen': [0.90, 0.89, 0.88, 0.87, 0.92, 0.91],
        'mean_phi_delta': [0.12, 0.11, 0.10, 0.09, 0.13, 0.12],
        'mean_success_rate': [0.75, 0.74, 0.73, 0.72, 0.76, 0.75]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Created sample historical data: {filename}")
    return filename


def test_basic_integration():
    """Test basic historical data integration with DataFrame."""
    print("\n" + "="*60)
    print("Test 1: Basic Integration with DataFrame")
    print("="*60)
    
    # Create sample historical data
    hist_file = create_sample_historical_data("test_historical.csv")
    
    # Current simulation results
    current_df = pd.DataFrame({
        'mean_fidelity': [0.86],  # Slightly higher than historical
        'std': [0.03],
        'asymmetry_delta': [0.30],
        'novelty_gen': [0.93],
        'mean_phi_delta': [0.14]
    })
    
    print("\nCurrent Data:")
    print(current_df)
    
    # Integrate historical data
    balanced = incorporate_historical_data(
        history_file_path=hist_file,
        current_data_df=current_df,
        historical_weight=0.3
    )
    
    print("\nBalanced Data (30% historical, 70% current):")
    print(balanced)
    
    # Cleanup
    os.remove(hist_file)
    print(f"\n✅ Test 1 passed!")


def test_prhp_integration():
    """Test integration with PRHP simulation results."""
    print("\n" + "="*60)
    print("Test 2: PRHP Integration")
    print("="*60)
    
    if not PRHP_AVAILABLE:
        print("⚠️  PRHP not available. Using mock PRHP results.")
        prhp_results = {
            'ADHD-collectivist': {
                'mean_fidelity': 0.84,
                'std': 0.025,
                'asymmetry_delta': 0.28,
                'novelty_gen': 0.90,
                'mean_phi_delta': 0.12
            },
            'neurotypical-hybrid': {
                'mean_fidelity': 0.85,
                'std': 0.026,
                'asymmetry_delta': 0.20,
                'novelty_gen': 0.92,
                'mean_phi_delta': 0.13
            }
        }
    else:
        print("Running PRHP simulation...")
        prhp_results = simulate_prhp(
            levels=9,
            variants=['ADHD-collectivist', 'neurotypical-hybrid'],
            n_monte=50,  # Small for quick test
            seed=42,
            show_progress=False,
            public_output_only=False  # Get full results with metrics
        )
    
    print("\nPRHP Results:")
    for variant, data in prhp_results.items():
        print(f"\n{variant}:")
        mean_fid = data.get('mean_fidelity', None)
        asym = data.get('asymmetry_delta', None)
        print(f"  Fidelity: {mean_fid:.4f}" if mean_fid is not None else "  Fidelity: N/A")
        print(f"  Asymmetry: {asym:.4f}" if asym is not None else "  Asymmetry: N/A")
    
    # Create historical data
    hist_file = create_sample_historical_data("test_historical_prhp.csv")
    
    # Integrate historical data
    balanced_results = incorporate_prhp_results(
        history_file_path=hist_file,
        prhp_results=prhp_results,
        historical_weight=0.3
    )
    
    print("\nBalanced Results:")
    for variant, balanced_df in balanced_results.items():
        print(f"\n{variant}:")
        if not balanced_df.empty and 'mean_fidelity' in balanced_df.columns:
            print(f"  Balanced Fidelity: {balanced_df['mean_fidelity'].iloc[0]:.4f}")
        else:
            print(f"  Balanced Fidelity: N/A (no data)")
        if not balanced_df.empty and 'historical_variance' in balanced_df.columns:
            print(f"  Historical Variance: {balanced_df['historical_variance'].iloc[0]:.4f}")
        if not balanced_df.empty and 'historical_samples' in balanced_df.columns:
            print(f"  Historical Samples: {int(balanced_df['historical_samples'].iloc[0])}")
    
    # Cleanup
    os.remove(hist_file)
    print(f"\n✅ Test 2 passed!")


def test_class_based_usage():
    """Test class-based usage with custom weights."""
    print("\n" + "="*60)
    print("Test 3: Class-Based Usage")
    print("="*60)
    
    # Create integrator with custom weights
    integrator = HistoricalDataIntegrator(
        historical_weight=0.4,  # 40% historical
        current_weight=0.6,     # 60% current
        key_columns=['mean_fidelity', 'asymmetry_delta', 'novelty_gen']
    )
    
    # Create historical data
    hist_file = create_sample_historical_data("test_historical_class.csv")
    
    # Load and fit historical data
    hist_df = integrator.load_historical_data(hist_file, variant='ADHD-collectivist')
    priors = integrator.fit_historical_priors(hist_df, variant_column='variant')
    
    print("\nHistorical Priors for ADHD-collectivist:")
    if 'ADHD-collectivist' in priors:
        for metric, stats in priors['ADHD-collectivist'].items():
            print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # Current data
    current_data = pd.DataFrame({
        'mean_fidelity': [0.86],
        'asymmetry_delta': [0.30],
        'novelty_gen': [0.93]
    })
    
    # Integrate
    balanced = integrator.incorporate_historical_data(
        history_file_path=hist_file,
        current_data=current_data,
        variant='ADHD-collectivist'
    )
    
    print("\nBalanced Data (40% historical, 60% current):")
    print(balanced[['mean_fidelity', 'asymmetry_delta', 'novelty_gen', 'historical_variance']])
    
    # Cleanup
    os.remove(hist_file)
    print(f"\n✅ Test 3 passed!")


if __name__ == "__main__":
    print("Historical Data Integration - Test Suite")
    print("="*60)
    
    try:
        test_basic_integration()
        test_prhp_integration()
        test_class_based_usage()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

