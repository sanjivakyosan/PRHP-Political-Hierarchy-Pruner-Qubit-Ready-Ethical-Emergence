"""
Parallel processing utilities for PRHP framework.

Copyright © sanjivakyosan 2025
"""

import numpy as np
from typing import Callable, List, Any, Optional, Dict
from functools import partial
import multiprocessing as mp

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()

def parallel_monte_carlo(
    func: Callable,
    n_iterations: int,
    n_jobs: int = -1,
    chunk_size: int = 10,
    **kwargs
) -> List[Any]:
    """
    Run Monte Carlo iterations in parallel.
    
    Args:
        func: Function to execute in parallel
        n_iterations: Number of iterations
        n_jobs: Number of parallel jobs (-1 for all CPUs)
        chunk_size: Number of iterations per chunk
        **kwargs: Additional arguments to pass to func
    
    Returns:
        List of results from each iteration
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    n_jobs = min(n_jobs, n_iterations)  # Don't use more jobs than iterations
    
    if n_jobs == 1:
        # Sequential execution
        logger.debug("Running Monte Carlo sequentially")
        return [func(i, **kwargs) for i in range(n_iterations)]
    
    logger.info(f"Running {n_iterations} Monte Carlo iterations in parallel with {n_jobs} workers")
    
    # Create partial function with kwargs
    func_partial = partial(func, **kwargs)
    
    # Create chunks
    chunks = [range(i, min(i + chunk_size, n_iterations)) 
              for i in range(0, n_iterations, chunk_size)]
    
    def process_chunk(chunk_range):
        """Process a chunk of iterations."""
        results = []
        for i in chunk_range:
            try:
                result = func_partial(i)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in iteration {i}: {e}")
                results.append(None)
        return results
    
    # Execute in parallel
    with mp.Pool(n_jobs) as pool:
        chunk_results = pool.map(process_chunk, chunks)
    
    # Flatten results
    results = [item for chunk in chunk_results for item in chunk]
    
    return results

def parallel_variant_simulation(
    simulate_func: Callable,
    variants: List[str],
    n_monte: int,
    n_jobs: int = -1,
    **kwargs
) -> Dict[str, Any]:
    """
    Run simulations for multiple variants in parallel.
    
    Args:
        simulate_func: Simulation function
        variants: List of variants to simulate
        n_monte: Number of Monte Carlo iterations per variant
        n_jobs: Number of parallel jobs
        **kwargs: Additional arguments to pass to simulate_func
    
    Returns:
        Dictionary mapping variant -> results
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    n_jobs = min(n_jobs, len(variants))
    
    if n_jobs == 1:
        # Sequential execution
        results = {}
        for variant in variants:
            results[variant] = simulate_func(variant=variant, n_monte=n_monte, **kwargs)
        return results
    
    logger.info(f"Running simulations for {len(variants)} variants in parallel with {n_jobs} workers")
    
    def simulate_variant(variant):
        """Simulate a single variant."""
        try:
            return variant, simulate_func(variant=variant, n_monte=n_monte, **kwargs)
        except Exception as e:
            logger.error(f"Error simulating {variant}: {e}")
            return variant, None
    
    with mp.Pool(n_jobs) as pool:
        variant_results = pool.map(simulate_variant, variants)
    
    return dict(variant_results)

