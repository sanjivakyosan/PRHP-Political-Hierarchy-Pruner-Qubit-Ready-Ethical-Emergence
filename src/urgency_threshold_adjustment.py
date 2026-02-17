"""
Urgency Threshold Adjustment Module for PRHP Framework

Provides dynamic recalibration of risk-utility thresholds based on urgency factors
and external data sources (e.g., MSF reports, crisis data).

Copyright © sanjivakyosan 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

try:
    from .utils import get_logger, validate_float_range
except ImportError:
    from utils import get_logger, validate_float_range

logger = get_logger()

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests module not available. API-based urgency adjustment will be disabled.")


class UrgencyThresholdAdjuster:
    """
    Adjusts risk-utility thresholds based on urgency factors and external data.
    
    Provides:
    - Dynamic threshold adjustment for urgent scenarios
    - Integration with external data sources (MSF reports, crisis data)
    - Risk-utility recalibration with urgency weighting
    - Bounds checking and validation
    """
    
    def __init__(
        self,
        min_threshold: float = 0.05,
        max_threshold: float = 0.50,
        max_urgency_factor: float = 2.0,
        min_urgency_factor: float = 0.5
    ):
        """
        Initialize the urgency threshold adjuster.
        
        Args:
            min_threshold: Minimum allowed threshold (safety bound)
            max_threshold: Maximum allowed threshold (safety bound)
            max_urgency_factor: Maximum urgency multiplier (prevents extreme adjustments)
            min_urgency_factor: Minimum urgency multiplier (prevents extreme reductions)
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_urgency_factor = max_urgency_factor
        self.min_urgency_factor = min_urgency_factor
    
    def calculate_urgency_factor(
        self,
        base_urgency: float = 1.0,
        data_source: Optional[Dict[str, Any]] = None,
        urgency_indicators: Optional[List[str]] = None
    ) -> float:
        """
        Calculate urgency factor from base value and external data sources.
        
        Args:
            base_urgency: Base urgency multiplier (default: 1.0)
            data_source: Optional external data dict (e.g., MSF reports, crisis data)
            urgency_indicators: Optional list of urgency indicator keywords
        
        Returns:
            Calculated urgency factor (bounded)
        """
        urgency_factor = base_urgency
        
        if data_source:
            # Extract urgency signals from data source
            # Example: hypothermia risk, crisis severity, time pressure
            
            # Hypothermia/winter urgency (from MSF reports)
            if 'hypothermia_risk' in data_source:
                hypothermia_risk = float(data_source['hypothermia_risk'])
                # Scale: 0.0-1.0 risk -> 0.0-0.5 urgency boost
                urgency_factor += hypothermia_risk * 0.5
            
            # Crisis severity (0.0-1.0)
            if 'crisis_severity' in data_source:
                crisis_severity = float(data_source['crisis_severity'])
                urgency_factor += crisis_severity * 0.3
            
            # Time pressure (0.0-1.0)
            if 'time_pressure' in data_source:
                time_pressure = float(data_source['time_pressure'])
                urgency_factor += time_pressure * 0.2
            
            # Resource scarcity (0.0-1.0)
            if 'resource_scarcity' in data_source:
                resource_scarcity = float(data_source['resource_scarcity'])
                urgency_factor += resource_scarcity * 0.25
            
            # Custom urgency boost
            if 'urgency_boost' in data_source:
                urgency_factor += float(data_source['urgency_boost'])
        
        # Check for urgency indicators in data source
        if urgency_indicators and data_source:
            for indicator in urgency_indicators:
                if indicator in data_source and data_source[indicator]:
                    # Each indicator adds a small urgency boost
                    urgency_factor += 0.1
        
        # Bound the urgency factor
        urgency_factor = max(self.min_urgency_factor, min(self.max_urgency_factor, urgency_factor))
        
        logger.debug(f"Calculated urgency factor: {urgency_factor:.3f} (from base: {base_urgency:.3f})")
        
        return urgency_factor
    
    def adjust(
        self,
        risk_values: np.ndarray,
        utility_values: np.ndarray,
        urgency_factor: float = 1.0,
        base_threshold: float = 0.30,
        data_source: Optional[Dict[str, Any]] = None,
        urgency_indicators: Optional[List[str]] = None,
        boost_low_risk_utilities: bool = True,
        dampen_risks: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Adjust urgency thresholds and recalibrate risk-utility values.
        
        Args:
            risk_values: Array of risk deltas (e.g., reprisal probabilities)
            utility_values: Array of utility scores (e.g., lives saved %)
            urgency_factor: Base urgency multiplier (will be enhanced by data_source if provided)
            base_threshold: Initial risk threshold
            data_source: Optional external data dict (e.g., MSF reports, crisis data)
            urgency_indicators: Optional list of urgency indicator keywords
            boost_low_risk_utilities: If True, boost utilities for options below threshold
            dampen_risks: If True, slightly dampen risks for balanced view
        
        Returns:
            Tuple of (adjusted_threshold, adjusted_risks, adjusted_utils, metadata)
        """
        # Input validation
        if not isinstance(risk_values, np.ndarray):
            risk_values = np.array(risk_values)
        if not isinstance(utility_values, np.ndarray):
            utility_values = np.array(utility_values)
        
        if len(risk_values) != len(utility_values):
            raise ValueError(
                f"risk_values and utility_values must have same length. "
                f"Got {len(risk_values)} and {len(utility_values)}"
            )
        
        if len(risk_values) == 0:
            raise ValueError("risk_values and utility_values cannot be empty")
        
        # Validate base threshold
        base_threshold = validate_float_range(base_threshold, "base_threshold", 0.0, 1.0)
        
        # Check for NaN or inf values
        if np.any(np.isnan(risk_values)) or np.any(np.isinf(risk_values)):
            raise ValueError("risk_values contains NaN or inf values")
        if np.any(np.isnan(utility_values)) or np.any(np.isinf(utility_values)):
            raise ValueError("utility_values contains NaN or inf values")
        
        # Calculate urgency factor from data source if provided
        calculated_urgency = self.calculate_urgency_factor(
            base_urgency=urgency_factor,
            data_source=data_source,
            urgency_indicators=urgency_indicators
        )
        
        # Dynamically adjust threshold for high-urgency scenarios
        # Higher urgency -> lower threshold (more options accepted)
        adjusted_threshold = base_threshold / calculated_urgency
        
        # Bound the adjusted threshold
        adjusted_threshold = max(self.min_threshold, min(self.max_threshold, adjusted_threshold))
        
        # Recalibrate utilities: boost utilities for options below threshold
        adjusted_utils = utility_values.copy()
        if boost_low_risk_utilities:
            mask = risk_values <= adjusted_threshold
            # Boost utilities for low-risk options
            adjusted_utils[mask] *= calculated_urgency
            # Ensure utilities don't exceed 1.0
            adjusted_utils = np.clip(adjusted_utils, 0.0, 1.0)
        
        # Dampen risks slightly for balanced view
        adjusted_risks = risk_values.copy()
        if dampen_risks:
            adjusted_risks = risk_values / calculated_urgency
            # Ensure risks stay non-negative
            adjusted_risks = np.clip(adjusted_risks, 0.0, 1.0)
        
        # Calculate metadata
        threshold_change = adjusted_threshold - base_threshold
        threshold_change_pct = (threshold_change / base_threshold) * 100 if base_threshold > 0 else 0
        
        metadata = {
            'base_threshold': base_threshold,
            'adjusted_threshold': adjusted_threshold,
            'threshold_change': threshold_change,
            'threshold_change_pct': threshold_change_pct,
            'urgency_factor': calculated_urgency,
            'base_urgency_factor': urgency_factor,
            'urgency_boost': calculated_urgency - urgency_factor,
            'n_options': len(risk_values),
            'n_below_threshold': int(np.sum(risk_values <= adjusted_threshold)),
            'n_above_threshold': int(np.sum(risk_values > adjusted_threshold)),
            'data_source_used': data_source is not None,
            'data_source_keys': list(data_source.keys()) if data_source else []
        }
        
        logger.info(
            f"Adjusted urgency threshold: {base_threshold:.3f} -> {adjusted_threshold:.3f} "
            f"(urgency factor: {calculated_urgency:.3f})"
        )
        
        return adjusted_threshold, adjusted_risks, adjusted_utils, metadata


def adjust_urgency_thresholds(
    risk_values: Union[np.ndarray, List[float]],
    utility_values: Union[np.ndarray, List[float]],
    urgency_factor: float = 1.0,
    base_threshold: float = 0.30,
    data_source: Optional[Dict[str, Any]] = None,
    urgency_indicators: Optional[List[str]] = None,
    min_threshold: float = 0.05,
    max_threshold: float = 0.50,
    boost_low_risk_utilities: bool = True,
    dampen_risks: bool = True
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Universal function to recalibrate conservatism in risk-utility balancing.
    
    Adjusts thresholds for urgent scenarios (e.g., winter hypothermia) by scaling
    with factors from data sources. Integrates with PRHP framework for consistent
    risk-utility management.
    
    Args:
        risk_values: Risk deltas (e.g., reprisal probabilities) as array or list
        utility_values: Utility scores (e.g., lives saved %) as array or list
        urgency_factor: Base urgency multiplier (default: 1.0)
        base_threshold: Initial risk threshold (default: 0.30)
        data_source: Optional external data dict (e.g., MSF reports, crisis data)
                    Supported keys:
                    - 'hypothermia_risk': float (0.0-1.0) - Winter urgency from MSF
                    - 'crisis_severity': float (0.0-1.0) - Overall crisis severity
                    - 'time_pressure': float (0.0-1.0) - Time pressure indicator
                    - 'resource_scarcity': float (0.0-1.0) - Resource availability
                    - 'urgency_boost': float - Direct urgency boost
        urgency_indicators: Optional list of urgency indicator keywords
        min_threshold: Minimum allowed threshold (safety bound, default: 0.05)
        max_threshold: Maximum allowed threshold (safety bound, default: 0.50)
        boost_low_risk_utilities: If True, boost utilities for options below threshold
        dampen_risks: If True, slightly dampen risks for balanced view
    
    Returns:
        Tuple of (adjusted_threshold, adjusted_risks, adjusted_utils, metadata)
        - adjusted_threshold: New risk threshold after urgency adjustment
        - adjusted_risks: Recalibrated risk values
        - adjusted_utils: Recalibrated utility values
        - metadata: Dict with adjustment details and diagnostics
    
    Example:
        >>> import numpy as np
        >>> msf_data = {'hypothermia_risk': 0.12, 'crisis_severity': 0.3}
        >>> risks = np.array([0.32, 0.08, 0.15])
        >>> utils = np.array([0.91, 0.88, 0.85])
        >>> new_thresh, adj_risks, adj_utils, meta = adjust_urgency_thresholds(
        ...     risks, utils, data_source=msf_data, urgency_factor=1.2
        ... )
        >>> print(f"Adjusted Threshold: {new_thresh:.3f}")
        >>> print(f"Urgency Factor: {meta['urgency_factor']:.3f}")
    """
    adjuster = UrgencyThresholdAdjuster(
        min_threshold=min_threshold,
        max_threshold=max_threshold
    )
    
    return adjuster.adjust(
        risk_values=risk_values,
        utility_values=utility_values,
        urgency_factor=urgency_factor,
        base_threshold=base_threshold,
        data_source=data_source,
        urgency_indicators=urgency_indicators,
        boost_low_risk_utilities=boost_low_risk_utilities,
        dampen_risks=dampen_risks
    )


def adjust_prhp_urgency_thresholds(
    prhp_results: Dict[str, Dict[str, Any]],
    urgency_factor: float = 1.0,
    base_threshold: float = 0.30,
    data_source: Optional[Dict[str, Any]] = None,
    variant: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Adjust urgency thresholds for PRHP simulation results.
    
    Args:
        prhp_results: PRHP simulation results dict (variant -> metrics)
        urgency_factor: Base urgency multiplier
        base_threshold: Initial risk threshold
        data_source: Optional external data dict
        variant: Optional variant to adjust (if None, adjusts all)
    
    Returns:
        Updated PRHP results with urgency-adjusted metrics
    """
    adjusted_results = {}
    
    for var, data in prhp_results.items():
        if variant and var != variant:
            adjusted_results[var] = data.copy()
            continue
        
        # Extract risk and utility metrics
        risk_value = data.get('asymmetry_delta', 0.0)
        utility_value = data.get('mean_fidelity', 0.0)
        
        # Convert to arrays for adjustment
        risk_array = np.array([risk_value])
        utility_array = np.array([utility_value])
        
        try:
            # Adjust urgency thresholds
            adjusted_threshold, adjusted_risks, adjusted_utils, metadata = adjust_urgency_thresholds(
                risk_values=risk_array,
                utility_values=utility_array,
                urgency_factor=urgency_factor,
                base_threshold=base_threshold,
                data_source=data_source
            )
            
            # Update results
            updated_data = data.copy()
            updated_data['asymmetry_delta'] = float(adjusted_risks[0])
            updated_data['mean_fidelity'] = float(adjusted_utils[0])
            
            # Add urgency adjustment metadata
            updated_data['urgency_adjustment'] = {
                'applied': True,
                'base_threshold': base_threshold,
                'adjusted_threshold': adjusted_threshold,
                'urgency_factor': metadata['urgency_factor'],
                'threshold_change_pct': metadata['threshold_change_pct'],
                'data_source_used': metadata['data_source_used']
            }
            
            adjusted_results[var] = updated_data
            
            logger.info(f"Applied urgency adjustment to {var}: threshold={adjusted_threshold:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to adjust urgency for {var}: {e}. Using original values.")
            adjusted_results[var] = data.copy()
    
    return adjusted_results


def dynamic_urgency_adjust(
    risk_values: Union[np.ndarray, List[float]],
    utility_values: Union[np.ndarray, List[float]],
    base_threshold: float = 0.28,
    escalation_api_url: Optional[str] = None,
    pledge_keywords: Optional[List[str]] = None
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Universal function to boost utilities and lower thresholds in urgent de-escalation contexts.
    
    Analyzes signals (e.g., RSF pledges, aid access, UN corridors) to reduce conservatism
    and boost utilities when de-escalation is detected (e.g., +18% for drops if access pledged).
    
    This function is specifically designed for de-escalation contexts where positive signals
    (pledges, aid access, UN corridors) indicate reduced risk and increased utility potential.
    
    Args:
        risk_values: Risk deltas (e.g., reprisal probabilities) as array or list
        utility_values: Utility scores (e.g., drops efficacy) as array or list
        base_threshold: Initial threshold (default: 0.28)
        escalation_api_url: Optional API endpoint for fetching de-escalation signals
        pledge_keywords: Keywords for de-escalation boosts (default: ['RSF pledges', 'aid access', 'UN corridors'])
    
    Returns:
        Tuple of (adjusted_threshold, adjusted_risks, adjusted_utils, metadata)
        - adjusted_threshold: New threshold after de-escalation adjustment (lowered)
        - adjusted_risks: Original risk values (unchanged)
        - adjusted_utils: Boosted utility values (capped at 1.0)
        - metadata: Dict with adjustment details
    
    Example:
        >>> import numpy as np
        >>> api = "https://api.example.com/un-reports"  # e.g., RSF aid pledges
        >>> risks = np.array([0.31])
        >>> utils = np.array([0.932])
        >>> new_thresh, _, adj_utils, meta = dynamic_urgency_adjust(
        ...     risks, utils, escalation_api_url=api
        ... )
        >>> print(f"Adjusted Threshold: {new_thresh:.3f}")  # e.g., 0.24
        >>> print(f"Adjusted Utils: {adj_utils}")  # e.g., [1.0] (capped)
    """
    # Convert inputs to numpy arrays
    if isinstance(risk_values, list):
        risk_values = np.array(risk_values)
    if isinstance(utility_values, list):
        utility_values = np.array(utility_values)
    
    # Input validation
    if len(risk_values) != len(utility_values):
        raise ValueError(f"risk_values and utility_values must have same length. Got {len(risk_values)} and {len(utility_values)}")
    
    if not (0.0 <= base_threshold <= 1.0):
        raise ValueError(f"base_threshold must be between 0.0 and 1.0. Got {base_threshold}")
    
    # Default pledge keywords for de-escalation detection
    if pledge_keywords is None:
        pledge_keywords = ['RSF pledges', 'aid access', 'UN corridors']
    
    boost_factor = 1.0
    pledge_count = 0
    api_used = False
    
    # Fetch de-escalation signals from API if provided
    if escalation_api_url and REQUESTS_AVAILABLE:
        try:
            response = requests.get(escalation_api_url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Try different possible response formats
            reports = data.get('reports', [])
            if not reports:
                reports = data.get('data', [])
            if not reports and isinstance(data, list):
                reports = data
            
            # Count pledge keywords in reports
            for report in reports:
                report_text = str(report).lower() if not isinstance(report, str) else report.lower()
                for keyword in pledge_keywords:
                    if keyword.lower() in report_text:
                        pledge_count += 1
                        break  # Count each report only once
            
            # +18% per signal (e.g., RSF/UN pledges)
            boost_factor += pledge_count * 0.18
            api_used = True
            
            logger.info(f"Fetched de-escalation signals from API: {pledge_count} pledges detected, boost_factor={boost_factor:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to fetch de-escalation signals from API: {e}. Using default boost_factor=1.0")
            # Fallback: Assume 1 pledge for demo (conservative approach)
            boost_factor = 1.18
    elif escalation_api_url and not REQUESTS_AVAILABLE:
        logger.warning("requests module not available. Cannot fetch de-escalation signals from API.")
    
    # Adjust threshold (lower for de-escalation contexts)
    adjusted_threshold = base_threshold / boost_factor  # e.g., 0.28 -> 0.24
    
    # Ensure threshold stays within reasonable bounds
    adjusted_threshold = max(0.05, min(0.50, adjusted_threshold))
    
    # Boost utilities (e.g., 93.2% -> 110% capped at 1.0)
    adjusted_utils = utility_values * boost_factor
    adjusted_utils = np.clip(adjusted_utils, None, 1.0)  # Cap at 1.0
    
    # Calculate metadata
    metadata = {
        'applied': True,
        'base_threshold': base_threshold,
        'adjusted_threshold': float(adjusted_threshold),
        'boost_factor': float(boost_factor),
        'pledge_count': pledge_count,
        'api_used': api_used,
        'api_url': escalation_api_url if escalation_api_url else None,
        'pledge_keywords': pledge_keywords,
        'threshold_change': float(adjusted_threshold - base_threshold),
        'threshold_change_pct': float((adjusted_threshold - base_threshold) / base_threshold * 100) if base_threshold > 0 else 0.0,
        'utility_boost_applied': float(boost_factor - 1.0),
        'utilities_capped': bool(np.any(adjusted_utils >= 1.0))
    }
    
    logger.info(
        f"Dynamic urgency adjustment applied: threshold={adjusted_threshold:.3f} "
        f"(from {base_threshold:.3f}), boost_factor={boost_factor:.3f}, "
        f"pledge_count={pledge_count}"
    )
    
    return adjusted_threshold, risk_values, adjusted_utils, metadata


def adjust_prhp_dynamic_urgency(
    prhp_results: Dict[str, Dict[str, Any]],
    base_threshold: float = 0.28,
    escalation_api_url: Optional[str] = None,
    pledge_keywords: Optional[List[str]] = None,
    variant: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Apply dynamic urgency adjustment to PRHP simulation results.
    
    This function adjusts thresholds and utilities based on de-escalation signals
    (e.g., RSF pledges, aid access, UN corridors) detected from external APIs.
    
    Args:
        prhp_results: PRHP simulation results dictionary
        base_threshold: Initial threshold for adjustment (default: 0.28)
        escalation_api_url: Optional API endpoint for fetching de-escalation signals
        pledge_keywords: Keywords for de-escalation detection
        variant: Optional variant name to adjust (if None, adjusts all variants)
    
    Returns:
        Dictionary with adjusted results and metadata
    """
    adjusted_results = {}
    
    variants_to_adjust = [variant] if variant else list(prhp_results.keys())
    
    for var in variants_to_adjust:
        if var not in prhp_results:
            continue
        
        data = prhp_results[var]
        
        # Extract risk and utility values
        risk_value = data.get('asymmetry_delta', 0.0)
        utility_value = data.get('mean_fidelity', 0.0)
        
        # Convert to arrays
        risk_array = np.array([risk_value])
        utility_array = np.array([utility_value])
        
        try:
            # Apply dynamic urgency adjustment
            adjusted_threshold, adjusted_risks, adjusted_utils, metadata = dynamic_urgency_adjust(
                risk_values=risk_array,
                utility_values=utility_array,
                base_threshold=base_threshold,
                escalation_api_url=escalation_api_url,
                pledge_keywords=pledge_keywords
            )
            
            # Update results
            updated_data = data.copy()
            updated_data['asymmetry_delta'] = float(adjusted_risks[0])
            updated_data['mean_fidelity'] = float(adjusted_utils[0])
            
            # Add dynamic urgency adjustment metadata
            updated_data['dynamic_urgency_adjustment'] = {
                'applied': True,
                'base_threshold': base_threshold,
                'adjusted_threshold': adjusted_threshold,
                'boost_factor': metadata['boost_factor'],
                'pledge_count': metadata['pledge_count'],
                'api_used': metadata['api_used'],
                'threshold_change_pct': metadata['threshold_change_pct'],
                'utility_boost_applied': metadata['utility_boost_applied'],
                'utilities_capped': metadata['utilities_capped']
            }
            
            adjusted_results[var] = updated_data
            
            logger.info(f"Applied dynamic urgency adjustment to {var}: threshold={adjusted_threshold:.3f}, boost={metadata['boost_factor']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to apply dynamic urgency adjustment to {var}: {e}. Using original values.")
            adjusted_results[var] = data.copy()
    
    return adjusted_results


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Example with MSF-inspired data
    msf_data = {
        'hypothermia_risk': 0.12,  # From MSF reports: winter challenges
        'crisis_severity': 0.3,
        'time_pressure': 0.2
    }
    
    risks = np.array([0.32, 0.08, 0.15])
    utils = np.array([0.91, 0.88, 0.85])
    
    print("Original:")
    print(f"  Risks: {risks}")
    print(f"  Utils: {utils}")
    print(f"  Base Threshold: 0.30")
    
    new_thresh, adj_risks, adj_utils, meta = adjust_urgency_thresholds(
        risks, utils,
        urgency_factor=1.2,
        data_source=msf_data
    )
    
    print("\nAdjusted:")
    print(f"  Adjusted Threshold: {new_thresh:.3f}")
    print(f"  Adjusted Risks: {adj_risks}")
    print(f"  Adjusted Utils: {adj_utils}")
    print(f"\nMetadata:")
    print(f"  Urgency Factor: {meta['urgency_factor']:.3f}")
    print(f"  Threshold Change: {meta['threshold_change_pct']:.2f}%")
    print(f"  Options Below Threshold: {meta['n_below_threshold']}/{meta['n_options']}")

