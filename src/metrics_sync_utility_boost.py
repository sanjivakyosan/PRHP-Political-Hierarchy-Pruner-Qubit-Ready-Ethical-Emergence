"""
Metrics Synchronization and Utility Boost Module

Synchronizes terminology (e.g., 'Fidelity' to 'Accuracy') across simulation data and output text,
and boosts utilities in escalation contexts (e.g., tariff pauses, ASEAN shifts).

Integrates with:
- output_standardization.py (terminology mapping)
- escalation_threshold_adjustment.py (escalation context detection)
- PRHP results structure (mean_fidelity, asymmetry_delta, etc.)
"""

import re
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()


class MetricsSyncUtilityBooster:
    """
    Synchronizes metrics terminology and boosts utilities based on escalation contexts.

    Provides:
    - Terminology synchronization (Fidelity -> Accuracy)
    - Utility boosting in escalation contexts (e.g., tariff pauses, ASEAN shifts)
    - Integration with PRHP results structure
    """

    def __init__(
        self,
        default_escalation_contexts: Optional[List[str]] = None,
        default_boost_factor: float = 1.25,
        min_boost_factor: float = 1.0,
        max_boost_factor: float = 2.0
    ):
        """
        Initialize the metrics sync and utility booster.

        Args:
            default_escalation_contexts: Default escalation contexts to detect
            default_boost_factor: Default boost factor for utilities (e.g., 1.25 = +25%)
            min_boost_factor: Minimum allowed boost factor
            max_boost_factor: Maximum allowed boost factor
        """
        self.default_escalation_contexts = default_escalation_contexts if default_escalation_contexts is not None else [
            'tariff pauses',
            'supply chain shifts to ASEAN',
            'ASEAN shifts',
            'de-escalation',
            'trade normalization',
            'sanctions relief'
        ]
        self.default_boost_factor = max(min_boost_factor, min(max_boost_factor, default_boost_factor))
        self.min_boost_factor = min_boost_factor
        self.max_boost_factor = max_boost_factor

        # PRHP metric keys that represent utilities (to be boosted)
        # These are positive metrics that should increase in de-escalation contexts
        self.utility_metrics = [
            'mean_fidelity',
            'Mean Fidelity',
            'fidelity',
            'Fidelity',
            'utility_score',
            'utility',
            'novelty_gen',
            'novelty generation',
            'Novelty Gen',
            'novelty',
            'mean_success_rate',
            'success_rate'
        ]

        # PRHP metric keys that represent risks (to be dampened in escalation contexts)
        # These are negative metrics that should decrease in de-escalation contexts
        self.risk_metrics = [
            'asymmetry_delta',
            'Asymmetry Delta',
            'asymmetry',
            'risk_delta',
            'risk',
            'std',
            'standard deviation',
            'phi_delta',
            'mean_phi_delta'
        ]

    def detect_escalation_contexts(
        self,
        output_text: str,
        escalation_contexts: Optional[List[str]] = None
    ) -> Tuple[List[str], float]:
        """
        Detect escalation contexts in output text and calculate boost factor.

        Args:
            output_text: Text to analyze for escalation contexts
            escalation_contexts: Optional custom contexts to detect

        Returns:
            Tuple of (detected_contexts, boost_factor)
        """
        contexts_to_check = escalation_contexts if escalation_contexts is not None else self.default_escalation_contexts

        detected_contexts = []
        boost_factor = 1.0

        text_lower = output_text.lower()

        for context in contexts_to_check:
            if context.lower() in text_lower:
                detected_contexts.append(context)
                # Accumulate boost: each context adds a small boost
                boost_factor += 0.05  # +5% per detected context

        # Clamp boost factor to allowed range
        boost_factor = max(self.min_boost_factor, min(self.max_boost_factor, boost_factor))

        if detected_contexts:
            logger.info(f"Detected escalation contexts: {detected_contexts}, boost factor: {boost_factor:.3f}")

        return detected_contexts, boost_factor

    def sync_terminology_in_dict(
        self,
        sim_data_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synchronize terminology in simulation data dictionary.
        Replaces 'Fidelity' with 'Accuracy' in keys and nested structures.

        Args:
            sim_data_dict: Simulation data dictionary (PRHP results structure)

        Returns:
            Updated dictionary with synchronized terminology
        """
        updated_data = {}

        for variant, metrics in sim_data_dict.items():
            if isinstance(metrics, dict):
                updated_metrics = {}
                for key, value in metrics.items():
                    # Replace 'Fidelity' with 'Accuracy' in keys
                    new_key = key.replace('Fidelity', 'Accuracy').replace('fidelity', 'accuracy')
                    updated_metrics[new_key] = value
                updated_data[variant] = updated_metrics
            else:
                # If not a dict, keep as-is
                updated_data[variant] = metrics

        return updated_data

    def sync_terminology_in_text(
        self,
        output_text: str
    ) -> str:
        """
        Synchronize terminology in output text.
        Replaces 'Fidelity' with 'Accuracy' using regex.

        Args:
            output_text: Text to synchronize

        Returns:
            Synchronized text
        """
        # Replace various forms of 'Fidelity' with 'Accuracy'
        replacements = [
            (r'Mean\s+Fidelity', 'Mean Accuracy'),
            (r'mean\s+fidelity', 'mean accuracy'),
            (r'\bFidelity\b', 'Accuracy'),
            (r'\bfidelity\b', 'accuracy'),
            (r'FIDELITY', 'ACCURACY')
        ]

        synchronized_text = output_text
        for pattern, replacement in replacements:
            synchronized_text = re.sub(pattern, replacement, synchronized_text, flags=re.IGNORECASE)

        return synchronized_text

    def boost_utilities(
        self,
        sim_data_dict: Dict[str, Any],
        boost_factor: float
    ) -> Dict[str, Any]:
        """
        Boost utility metrics in simulation data based on escalation contexts.

        Args:
            sim_data_dict: Simulation data dictionary (PRHP results structure)
            boost_factor: Factor to boost utilities (e.g., 1.25 = +25%)

        Returns:
            Updated dictionary with boosted utilities
        """
        if boost_factor <= 1.0:
            # No boost needed
            return sim_data_dict

        updated_data = {}

        for variant, metrics in sim_data_dict.items():
            if isinstance(metrics, dict):
                updated_metrics = metrics.copy()

                # First, identify which keys are utilities and which are risks
                utility_keys = []
                risk_keys = []

                for key in updated_metrics.keys():
                    # Check if it's a utility metric (check both original and synchronized key names)
                    # Note: keys may have been synchronized (fidelity -> accuracy), so check both
                    key_lower = key.lower()
                    is_utility = any(
                        utility_key.lower() in key_lower or
                        utility_key.lower().replace('fidelity', 'accuracy') in key_lower or
                        utility_key.lower().replace('accuracy', 'fidelity') in key_lower
                        for utility_key in self.utility_metrics
                    )
                    # Check if it's a risk metric
                    is_risk = any(risk_key.lower() in key_lower for risk_key in self.risk_metrics)

                    # Prioritize risk detection over utility (to avoid conflicts)
                    if is_risk:
                        risk_keys.append(key)
                    elif is_utility:
                        utility_keys.append(key)

                # Boost utility metrics
                for key in utility_keys:
                    original_value = updated_metrics[key]
                    if isinstance(original_value, (int, float)):
                        boosted_value = original_value * boost_factor
                        # Clamp to valid range (e.g., 0.0-1.0 for fidelity/accuracy)
                        if 'fidelity' in key.lower() or 'accuracy' in key.lower() or 'success_rate' in key.lower():
                            boosted_value = min(1.0, max(0.0, boosted_value))
                        updated_metrics[key] = boosted_value
                        logger.debug(f"Boosted utility {key} from {original_value:.4f} to {boosted_value:.4f} (factor: {boost_factor:.3f})")

                # Dampen risk metrics (inverse of boost for balance)
                for key in risk_keys:
                    original_value = updated_metrics[key]
                    if isinstance(original_value, (int, float)):
                        # Dampen risks (divide by boost factor)
                        dampened_value = original_value / boost_factor
                        # Ensure non-negative
                        dampened_value = max(0.0, dampened_value)
                        updated_metrics[key] = dampened_value
                        logger.debug(f"Dampened risk {key} from {original_value:.4f} to {dampened_value:.4f} (factor: {boost_factor:.3f})")

                updated_data[variant] = updated_metrics
            else:
                updated_data[variant] = metrics

        return updated_data

    def sync_and_boost(
        self,
        sim_data_dict: Dict[str, Any],
        output_text: str,
        escalation_contexts: Optional[List[str]] = None,
        boost_factor: Optional[float] = None
    ) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
        """
        Main function: Synchronize terminology and boost utilities based on escalation contexts.

        Args:
            sim_data_dict: Simulation data dictionary (PRHP results structure)
            output_text: Response text to synchronize
            escalation_contexts: Optional custom escalation contexts to detect
            boost_factor: Optional explicit boost factor (overrides detection)

        Returns:
            Tuple of (updated_data, synchronized_text, metadata)
        """
        # Step 1: Detect escalation contexts and calculate boost factor
        if boost_factor is None:
            detected_contexts, calculated_boost = self.detect_escalation_contexts(
                output_text,
                escalation_contexts
            )
        else:
            detected_contexts = []
            calculated_boost = max(self.min_boost_factor, min(self.max_boost_factor, boost_factor))

        # Step 2: Synchronize terminology in data dict
        synchronized_data = self.sync_terminology_in_dict(sim_data_dict)

        # Step 3: Synchronize terminology in text
        synchronized_text = self.sync_terminology_in_text(output_text)

        # Step 4: Boost utilities if escalation contexts detected
        if calculated_boost > 1.0:
            synchronized_data = self.boost_utilities(synchronized_data, calculated_boost)

        # Build metadata
        metadata = {
            'escalation_contexts_detected': detected_contexts,
            'boost_factor_applied': calculated_boost,
            'utilities_boosted': calculated_boost > 1.0,
            'terminology_synchronized': True
        }

        if calculated_boost > 1.0:
            metadata['boost_note'] = f"Utility boosted by {calculated_boost:.3f} due to escalation contexts: {', '.join(detected_contexts) if detected_contexts else 'custom boost'}"

        logger.info(f"Metrics sync and utility boost completed: boost_factor={calculated_boost:.3f}, contexts={len(detected_contexts)}")

        return synchronized_data, synchronized_text, metadata


def sync_metrics_and_boost_utilities(
    sim_data_dict: Dict[str, Any],
    output_text: str,
    escalation_contexts: Optional[List[str]] = None,
    boost_factor: Optional[float] = None
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Universal function to synchronize 'Fidelity' to 'Accuracy' across data/text and boost utilities in escalation contexts.

    Scans/replaces terms; adjusts based on contexts like tariff pauses (e.g., +25% for sanctions if de-escalation detected).
    Integrates with PRHP framework for consistent metric handling.

    Args:
        sim_data_dict: Simulation data (e.g., {'ADHD-collectivist': {'mean_fidelity': 0.9827}})
        output_text: Response text to sync
        escalation_contexts: Optional contexts (e.g., ['tariff pauses', 'ASEAN shifts']); default from trade data
        boost_factor: Optional explicit boost factor (overrides detection)

    Returns:
        Tuple of (updated_data, cleaned_text, metadata)
        - updated_data: Dictionary with synchronized terminology and boosted utilities
        - cleaned_text: Text with synchronized terminology
        - metadata: Dictionary with boost details and context information
    """
    booster = MetricsSyncUtilityBooster()
    return booster.sync_and_boost(sim_data_dict, output_text, escalation_contexts, boost_factor)


def sync_prhp_results_and_boost_utilities(
    prhp_results: Dict[str, Dict[str, Any]],
    output_text: str,
    escalation_contexts: Optional[List[str]] = None,
    boost_factor: Optional[float] = None
) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    """
    Applies metrics synchronization and utility boosting to PRHP simulation results.

    Args:
        prhp_results: PRHP simulation results dictionary
        output_text: Response text to synchronize
        escalation_contexts: Optional escalation contexts to detect
        boost_factor: Optional explicit boost factor

    Returns:
        Tuple of (updated_results, synchronized_text, metadata)
    """
    return sync_metrics_and_boost_utilities(prhp_results, output_text, escalation_contexts, boost_factor)

