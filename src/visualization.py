"""
Visualization functions for PRHP framework results.

Copyright © sanjivakyosan 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()

def plot_fidelity_comparison(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot fidelity comparison across variants.
    
    Args:
        results: Results dictionary from simulate_prhp
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    variants = list(results.keys())
    fidelities = [results[v]['mean_fidelity'] for v in variants]
    stds = [results[v]['std'] for v in variants]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(variants))
    bars = ax.bar(x_pos, fidelities, yerr=stds, capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    ax.set_xlabel('Neuro-Cultural Variant', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('PRHP Fidelity Comparison Across Variants', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(variants, rotation=15, ha='right')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.84, color='r', linestyle='--', label='Target (0.84)')
    ax.legend()
    
    # Add value labels on bars
    for i, (bar, fid, std) in enumerate(zip(bars, fidelities, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{fid:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig

def plot_level_phis(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot phi values across levels for each variant.
    
    Args:
        results: Results dictionary from simulate_prhp
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (variant, data) in enumerate(results.items()):
        if 'level_phis' in data and len(data['level_phis']) > 0:
            levels = range(1, len(data['level_phis']) + 1)
            ax.plot(levels, data['level_phis'], marker='o', label=variant, 
                   color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('Level', fontsize=12)
    ax.set_ylabel('Phi (Symmetry)', fontsize=12)
    ax.set_title('Phi Evolution Across Hierarchy Levels', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig

def plot_phi_deltas(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot phi deltas across levels for each variant.
    
    Args:
        results: Results dictionary from simulate_prhp
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (variant, data) in enumerate(results.items()):
        if 'phi_deltas' in data and len(data['phi_deltas']) > 0:
            levels = range(1, len(data['phi_deltas']) + 1)
            ax.plot(levels, data['phi_deltas'], marker='s', label=variant,
                   color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel('Level', fontsize=12)
    ax.set_ylabel('Phi Delta', fontsize=12)
    ax.set_title('Phi Delta Evolution Across Levels', fontsize=14, fontweight='bold')
    ax.axhline(y=0.12, color='r', linestyle='--', label='Target (0.12)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig

def plot_metrics_summary(results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comprehensive metrics summary.
    
    Args:
        results: Results dictionary from simulate_prhp
        save_path: Optional path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    variants = list(results.keys())
    
    # Fidelity
    ax = axes[0, 0]
    fidelities = [results[v]['mean_fidelity'] for v in variants]
    stds = [results[v]['std'] for v in variants]
    ax.bar(variants, fidelities, yerr=stds, capsize=5, alpha=0.7)
    ax.set_ylabel('Fidelity')
    ax.set_title('Mean Fidelity')
    ax.axhline(y=0.84, color='r', linestyle='--')
    ax.tick_params(axis='x', rotation=15)
    
    # Novelty Gen
    ax = axes[0, 1]
    novelty_gens = [results[v]['novelty_gen'] for v in variants]
    ax.bar(variants, novelty_gens, alpha=0.7, color='green')
    ax.set_ylabel('Novelty Gen')
    ax.set_title('Novelty Generation')
    ax.tick_params(axis='x', rotation=15)
    
    # Asymmetry Delta
    ax = axes[1, 0]
    asymmetry_deltas = [results[v]['asymmetry_delta'] for v in variants]
    ax.bar(variants, asymmetry_deltas, alpha=0.7, color='orange')
    ax.set_ylabel('Asymmetry Delta')
    ax.set_title('Asymmetry Delta')
    ax.tick_params(axis='x', rotation=15)
    
    # Mean Phi Delta
    ax = axes[1, 1]
    phi_deltas = [results[v].get('mean_phi_delta', 0) or 0 for v in variants]
    ax.bar(variants, phi_deltas, alpha=0.7, color='purple')
    ax.set_ylabel('Mean Phi Delta')
    ax.set_title('Mean Phi Delta')
    ax.axhline(y=0.12, color='r', linestyle='--')
    ax.tick_params(axis='x', rotation=15)
    
    plt.suptitle('PRHP Metrics Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    return fig

def create_all_plots(results: Dict[str, Dict[str, Any]], output_dir: str = "plots") -> None:
    """
    Create all visualization plots.
    
    Args:
        results: Results dictionary from simulate_prhp
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    plot_fidelity_comparison(results, f"{output_dir}/fidelity_comparison.png")
    plot_level_phis(results, f"{output_dir}/level_phis.png")
    plot_phi_deltas(results, f"{output_dir}/phi_deltas.png")
    plot_metrics_summary(results, f"{output_dir}/metrics_summary.png")
    
    logger.info(f"Created all plots in {output_dir}")

