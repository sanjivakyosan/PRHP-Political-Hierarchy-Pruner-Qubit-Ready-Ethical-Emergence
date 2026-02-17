"""
Test script for scenario updates integration with PRHP framework.

This demonstrates how to integrate real-time scenario updates
from API or file sources into PRHP simulations.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from scenario_updates import (
    add_scenario_updates,
    add_prhp_scenario_updates,
    ScenarioUpdateManager
)

# Try to import PRHP (may not be available)
try:
    from prhp_core import simulate_prhp
    PRHP_AVAILABLE = True
except ImportError:
    PRHP_AVAILABLE = False
    print("PRHP framework not available. Using mock data for demonstration.")


def create_sample_update_file(filename: str = "sample_scenario_updates.json"):
    """Create a sample scenario update file for testing."""
    data = {
        "risk_delta": 0.32,
        "utility_score": 0.88,
        "mean_fidelity": 0.85,
        "asymmetry_delta": 0.25,
        "novelty_gen": 0.91,
        "source": "crisis_monitor",
        "timestamp": "2025-01-15T10:30:00Z"
    }
    
    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Created sample update file: {filename}")
    return filename


def test_basic_usage():
    """Test basic scenario updates with DataFrame."""
    print("\n" + "="*60)
    print("Test 1: Basic Usage with DataFrame")
    print("="*60)
    
    # Create sample update file
    update_file = create_sample_update_file("test_scenario_updates.json")
    
    # Current simulation data
    current_df = pd.DataFrame({
        'risk_delta': [0.28],
        'utility_score': [0.92],
        'mean_fidelity': [0.84]
    })
    
    print("\nCurrent Data:")
    print(current_df)
    
    # Test different merge strategies
    for strategy in ['overwrite', 'average', 'weighted']:
        print(f"\n--- Merge Strategy: {strategy} ---")
        updated = add_scenario_updates(
            local_update_file=update_file,
            current_df=current_df,
            update_keys=['risk_delta', 'utility_score'],
            merge_strategy=strategy,
            update_weight=0.3
        )
        print(updated[['risk_delta', 'utility_score', 'update_time']])
    
    # Cleanup
    os.remove(update_file)
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
                'asymmetry_delta': 0.28,
                'novelty_gen': 0.90,
                'mean_phi_delta': 0.12
            }
        }
    else:
        print("Running PRHP simulation...")
        prhp_results = simulate_prhp(
            levels=5,
            variants=['ADHD-collectivist'],
            n_monte=10,  # Small for quick test
            seed=42,
            show_progress=False,
            public_output_only=False
        )
    
    print("\nPRHP Results:")
    for variant, data in prhp_results.items():
        print(f"\n{variant}:")
        print(f"  Fidelity: {data.get('mean_fidelity', 'N/A')}")
        print(f"  Asymmetry: {data.get('asymmetry_delta', 'N/A')}")
    
    # Create update file
    update_file = create_sample_update_file("test_prhp_updates.json")
    
    # Apply scenario updates
    updated_results = add_prhp_scenario_updates(
        prhp_results=prhp_results,
        local_update_file=update_file,
        merge_strategy='weighted',
        update_weight=0.3
    )
    
    print("\nUpdated Results:")
    for variant, updated_df in updated_results.items():
        print(f"\n{variant}:")
        if not updated_df.empty:
            print(f"  Updated Fidelity: {updated_df['mean_fidelity'].iloc[0]:.4f}")
            print(f"  Updated Asymmetry: {updated_df['asymmetry_delta'].iloc[0]:.4f}")
            print(f"  Update Time: {updated_df['update_time'].iloc[0]}")
    
    # Cleanup
    os.remove(update_file)
    print(f"\n✅ Test 2 passed!")


def test_class_based_usage():
    """Test class-based usage with custom configuration."""
    print("\n" + "="*60)
    print("Test 3: Class-Based Usage")
    print("="*60)
    
    # Create manager with custom settings
    manager = ScenarioUpdateManager(
        merge_strategy='weighted',
        update_weight=0.4  # 40% weight for updates
    )
    
    # Create update file
    update_file = create_sample_update_file("test_class_updates.json")
    
    # Load updates
    updates = manager.load_from_file(update_file)
    print("\nLoaded Updates:")
    print(updates)
    
    # Current data
    current_data = pd.DataFrame({
        'risk_delta': [0.28],
        'utility_score': [0.92]
    })
    
    # Merge updates
    updated_df = manager.merge_updates(
        current_data=current_data,
        updates=updates,
        update_keys=['risk_delta', 'utility_score']
    )
    
    print("\nMerged Data (40% weight for updates):")
    print(updated_df[['risk_delta', 'utility_score', 'update_time']])
    
    print(f"\nUpdate History: {len(manager.update_history)} update(s)")
    
    # Cleanup
    os.remove(update_file)
    print(f"\n✅ Test 3 passed!")


if __name__ == "__main__":
    print("Scenario Updates Integration - Test Suite")
    print("="*60)
    
    try:
        test_basic_usage()
        test_prhp_integration()
        test_class_based_usage()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

