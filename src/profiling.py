"""
Performance profiling utilities for PRHP framework.

Copyright © sanjivakyosan 2025
"""

import time
import cProfile
import pstats
import io
import numpy as np
from typing import Callable, Any, Optional, Dict
from functools import wraps
from contextlib import contextmanager

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()

@contextmanager
def profile_context(output_file: Optional[str] = None):
    """
    Context manager for profiling code blocks.
    
    Usage:
        with profile_context('profile.stats'):
            # code to profile
    """
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        if output_file:
            profiler.dump_stats(output_file)
            logger.info(f"Profile saved to {output_file}")
        else:
            # Print to string
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            logger.debug(s.getvalue())

def time_function(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Usage:
        @time_function
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper

def benchmark_simulation(
    simulate_func: Callable,
    n_runs: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark a simulation function.
    
    Args:
        simulate_func: Simulation function to benchmark
        n_runs: Number of benchmark runs
        **kwargs: Arguments to pass to simulate_func
    
    Returns:
        Dictionary with benchmark statistics
    """
    logger.info(f"Benchmarking {simulate_func.__name__} with {n_runs} runs")
    
    times = []
    results = []
    
    for i in range(n_runs):
        start = time.time()
        result = simulate_func(**kwargs)
        elapsed = time.time() - start
        times.append(elapsed)
        results.append(result)
        logger.debug(f"Run {i+1}/{n_runs}: {elapsed:.3f}s")
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'total_time': np.sum(times),
        'times': times,
        'results': results
    }

def compare_performance(
    funcs: Dict[str, Callable],
    n_runs: int = 5,
    **kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of multiple functions.
    
    Args:
        funcs: Dictionary mapping name -> function
        n_runs: Number of benchmark runs per function
        **kwargs: Arguments to pass to functions
    
    Returns:
        Dictionary mapping function name -> benchmark stats
    """
    results = {}
    
    for name, func in funcs.items():
        logger.info(f"Benchmarking {name}...")
        benchmark = benchmark_simulation(func, n_runs=n_runs, **kwargs)
        results[name] = {
            'mean_time': benchmark['mean_time'],
            'std_time': benchmark['std_time'],
            'min_time': benchmark['min_time'],
            'max_time': benchmark['max_time']
        }
    
    # Print comparison
    logger.info("\nPerformance Comparison:")
    logger.info("-" * 60)
    for name, stats in results.items():
        logger.info(f"{name:30s} {stats['mean_time']:8.3f}s ± {stats['std_time']:.3f}s "
                   f"(min: {stats['min_time']:.3f}s, max: {stats['max_time']:.3f}s)")
    
    return results

