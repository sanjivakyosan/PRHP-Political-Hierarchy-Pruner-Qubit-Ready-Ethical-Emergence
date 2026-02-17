"""
Comprehensive unit tests for PRHP framework.
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prhp_core import simulate_prhp, dopamine_hierarchy
from src.qubit_hooks import (
    compute_phi, compute_phi_delta, entangle_nodes_variant,
    inject_phase_flip, recalibrate_novelty
)
from src.political_pruner import simulate_pruner_levels, threshold_qubit
from src.virus_extinction import simulate_viral_cascade, forecast_extinction_risk
from src.meta_empirical import bayesian_update_novelty, meta_empirical_validation
from src.utils import (
    validate_positive_int, validate_float_range, validate_variant,
    validate_variants, validate_seed
)

class TestValidation(unittest.TestCase):
    """Test input validation functions."""
    
    def test_validate_positive_int(self):
        self.assertEqual(validate_positive_int(5, "test"), 5)
        self.assertEqual(validate_positive_int(1, "test"), 1)
        with self.assertRaises(ValueError):
            validate_positive_int(0, "test")
        with self.assertRaises(TypeError):
            validate_positive_int("5", "test")
    
    def test_validate_float_range(self):
        self.assertEqual(validate_float_range(0.5, "test"), 0.5)
        self.assertEqual(validate_float_range(0.0, "test"), 0.0)
        self.assertEqual(validate_float_range(1.0, "test"), 1.0)
        with self.assertRaises(ValueError):
            validate_float_range(1.5, "test", 0.0, 1.0)
        with self.assertRaises(TypeError):
            validate_float_range("0.5", "test")
    
    def test_validate_variant(self):
        self.assertEqual(validate_variant("ADHD-collectivist"), "ADHD-collectivist")
        self.assertEqual(validate_variant("autistic-individualist"), "autistic-individualist")
        with self.assertRaises(ValueError):
            validate_variant("invalid-variant")
    
    def test_validate_variants(self):
        variants = validate_variants(["ADHD-collectivist", "neurotypical-hybrid"])
        self.assertEqual(len(variants), 2)
        with self.assertRaises(ValueError):
            validate_variants([])
        with self.assertRaises(ValueError):
            validate_variants("not-a-list")
    
    def test_validate_seed(self):
        self.assertEqual(validate_seed(42), 42)
        self.assertIsNone(validate_seed(None))
        with self.assertRaises(ValueError):
            validate_seed(-1)

class TestQubitHooks(unittest.TestCase):
    """Test qubit hooks functions."""
    
    def test_compute_phi(self):
        state1 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        state2 = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        phi = compute_phi(state1, state2, use_quantum=False)
        self.assertIsInstance(phi, (float, np.floating))
        self.assertGreater(phi, 0)
    
    def test_entangle_nodes_variant(self):
        a, b, fid, sym = entangle_nodes_variant("ADHD-collectivist", use_quantum=False, seed=42)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(b), 2)
        self.assertEqual(fid, 1.0)
        self.assertEqual(sym, 1.0)
    
    def test_inject_phase_flip(self):
        state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        flipped = inject_phase_flip(state, flip_prob=0.25, variant="ADHD-collectivist")
        self.assertEqual(len(flipped), 2)
        self.assertAlmostEqual(np.linalg.norm(flipped), 1.0, places=5)
    
    def test_recalibrate_novelty(self):
        state_a = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        state_b = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        a, b, asym = recalibrate_novelty(state_a, state_b, threshold=0.7, use_quantum=False)
        self.assertIsInstance(asym, (float, np.floating))
        self.assertGreaterEqual(asym, 0)

class TestPRHPCore(unittest.TestCase):
    """Test PRHP core functions."""
    
    def test_dopamine_hierarchy(self):
        asym = 0.2
        result = dopamine_hierarchy(asym, "ADHD-collectivist")
        self.assertAlmostEqual(result, 0.2 * 0.20, places=5)
    
    def test_simulate_prhp_basic(self):
        results = simulate_prhp(
            levels=3,
            variants=["neurotypical-hybrid"],
            n_monte=5,
            seed=42,
            show_progress=False
        )
        self.assertIn("neurotypical-hybrid", results)
        self.assertIn("mean_fidelity", results["neurotypical-hybrid"])
        self.assertGreater(results["neurotypical-hybrid"]["mean_fidelity"], 0)
    
    def test_simulate_prhp_multiple_variants(self):
        results = simulate_prhp(
            levels=3,
            variants=["ADHD-collectivist", "autistic-individualist"],
            n_monte=5,
            seed=42,
            show_progress=False
        )
        self.assertEqual(len(results), 2)
        self.assertIn("ADHD-collectivist", results)
        self.assertIn("autistic-individualist", results)

class TestPoliticalPruner(unittest.TestCase):
    """Test political pruner functions."""
    
    def test_threshold_qubit(self):
        self.assertTrue(threshold_qubit(0.7, "neurotypical-hybrid"))
        self.assertFalse(threshold_qubit(0.6, "neurotypical-hybrid"))
    
    def test_simulate_pruner_levels(self):
        result = simulate_pruner_levels(
            levels=3,
            variant="autistic-individualist",
            n_monte=5,
            seed=42,
            use_quantum=False
        )
        self.assertIn("level_deltas", result)
        self.assertIn("self_model_coherence", result)
        self.assertGreaterEqual(result["self_model_coherence"], 0)

class TestVirusExtinction(unittest.TestCase):
    """Test virus extinction functions."""
    
    def test_simulate_viral_cascade(self):
        result = simulate_viral_cascade(
            n_population=10,
            variant="ADHD-collectivist",
            n_iter=5,
            seed=42,
            use_quantum=False
        )
        self.assertIn("cascade_mitigation", result)
        self.assertIn("infection_rates", result)
        self.assertGreaterEqual(result["cascade_mitigation"], 0)
        self.assertLessEqual(result["cascade_mitigation"], 1)

class TestMetaEmpirical(unittest.TestCase):
    """Test meta-empirical functions."""
    
    def test_bayesian_update_novelty(self):
        phi_deltas = [0.11, 0.09, 0.08, 0.07]
        novelty = bayesian_update_novelty(0.80, phi_deltas)
        self.assertGreaterEqual(novelty, 0.80)
        self.assertLessEqual(novelty, 0.82)
    
    def test_meta_empirical_validation(self):
        results = {
            'mean_fidelity': 0.84,
            'mean_phi_delta': 0.12
        }
        validation = meta_empirical_validation(results)
        self.assertIn("fidelity_aligned", validation)
        self.assertIn("overall_validated", validation)

class TestReproducibility(unittest.TestCase):
    """Test reproducibility with seeds."""
    
    def test_reproducibility(self):
        results1 = simulate_prhp(
            levels=3,
            variants=["neurotypical-hybrid"],
            n_monte=10,
            seed=42,
            show_progress=False
        )
        results2 = simulate_prhp(
            levels=3,
            variants=["neurotypical-hybrid"],
            n_monte=10,
            seed=42,
            show_progress=False
        )
        # Results should be very similar (within numerical precision)
        fid1 = results1["neurotypical-hybrid"]["mean_fidelity"]
        fid2 = results2["neurotypical-hybrid"]["mean_fidelity"]
        self.assertAlmostEqual(fid1, fid2, places=3)

if __name__ == '__main__':
    unittest.main()

