"""
Escalation Threshold Adjustment Module for PRHP Framework

Provides dynamic threshold adjustment based on verbal escalation contexts
(e.g., PLA threats, battlefield escalation, geopolitical tensions).

Copyright © sanjivakyosan 2025
"""

import numpy as np
import requests
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import warnings

try:
    from .utils import get_logger, validate_float_range
except ImportError:
    try:
        from utils import get_logger, validate_float_range
    except ImportError:
        import logging
        logger = logging.getLogger('prhp')
        def validate_float_range(value, name, min_val, max_val):
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if value < min_val or value > max_val:
                raise ValueError(f"{name} must be between {min_val} and {max_val}")
            return float(value)

logger = get_logger() if 'logger' not in locals() else logger


class EscalationThresholdAdjuster:
    """
    Adjusts risk-utility thresholds based on verbal escalation contexts.
    
    Provides:
    - Threat detection from external data sources
    - Dynamic threshold adjustment
    - Utility boosting in high-escalation scenarios
    - Metadata tracking
    """
    
    def __init__(
        self,
        min_threshold: float = 0.05,
        max_threshold: float = 0.50,
        min_escalation_factor: float = 0.5,
        max_escalation_factor: float = 2.0,
        threat_multiplier: float = 0.05,
        timeout: int = 10
    ):
        """
        Initialize the escalation threshold adjuster.
        
        Args:
            min_threshold: Minimum allowed threshold (default: 0.05)
            max_threshold: Maximum allowed threshold (default: 0.50)
            min_escalation_factor: Minimum escalation factor (default: 0.5)
            max_escalation_factor: Maximum escalation factor (default: 2.0)
            threat_multiplier: Multiplier for each detected threat (default: 0.05)
            timeout: Request timeout in seconds (default: 10)
        """
        self.min_threshold = validate_float_range(min_threshold, "min_threshold", 0.0, 1.0)
        self.max_threshold = validate_float_range(max_threshold, "max_threshold", 0.0, 1.0)
        self.min_escalation_factor = validate_float_range(min_escalation_factor, "min_escalation_factor", 0.1, 1.0)
        self.max_escalation_factor = validate_float_range(max_escalation_factor, "max_escalation_factor", 1.0, 5.0)
        self.threat_multiplier = validate_float_range(threat_multiplier, "threat_multiplier", 0.0, 0.5)
        self.timeout = max(1, int(timeout))
    
    def fetch_escalation_data(
        self,
        escalation_api_url: str,
        threat_keywords: List[str],
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fetch escalation data from API endpoint.
        
        Args:
            escalation_api_url: API endpoint URL
            threat_keywords: List of keywords to detect threats
            headers: Optional HTTP headers
            params: Optional query parameters
        
        Returns:
            Dictionary with escalation data:
            - 'threat_count': Number of detected threats
            - 'escalation_factor': Calculated escalation factor
            - 'posts_analyzed': Number of posts analyzed
            - 'keywords_found': List of found keywords
        """
        try:
            request_params = params.copy() if params else {}
            
            response = requests.get(
                escalation_api_url,
                params=request_params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract posts (handle various API response formats)
            if isinstance(data, dict):
                posts = data.get('posts', data.get('data', data.get('items', data.get('results', []))))
            elif isinstance(data, list):
                posts = data
            else:
                posts = []
            
            # Count threats
            threat_count = 0
            keywords_found = []
            
            for post in posts:
                if isinstance(post, dict):
                    content = post.get('content', post.get('text', post.get('message', '')))
                    if isinstance(content, str):
                        content_lower = content.lower()
                        for keyword in threat_keywords:
                            if keyword.lower() in content_lower:
                                threat_count += 1
                                if keyword not in keywords_found:
                                    keywords_found.append(keyword)
                                break  # Count each post only once
            
            # Calculate escalation factor
            escalation_factor = 1.0 + (threat_count * self.threat_multiplier)
            escalation_factor = max(self.min_escalation_factor, min(self.max_escalation_factor, escalation_factor))
            
            return {
                'threat_count': threat_count,
                'escalation_factor': escalation_factor,
                'posts_analyzed': len(posts),
                'keywords_found': keywords_found,
                'api_url': escalation_api_url
            }
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching escalation data from {escalation_api_url}")
            return {
                'threat_count': 0,
                'escalation_factor': 1.0,
                'posts_analyzed': 0,
                'keywords_found': [],
                'error': 'timeout'
            }
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching escalation data from {escalation_api_url}: {e}")
            return {
                'threat_count': 0,
                'escalation_factor': 1.0,
                'posts_analyzed': 0,
                'keywords_found': [],
                'error': str(e)
            }
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Error parsing escalation data response: {e}")
            return {
                'threat_count': 0,
                'escalation_factor': 1.0,
                'posts_analyzed': 0,
                'keywords_found': [],
                'error': f'parse_error: {e}'
            }
    
    def calculate_escalation_factor(
        self,
        base_escalation_factor: float = 1.0,
        escalation_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate escalation factor from base and external data.
        
        Args:
            base_escalation_factor: Base escalation factor (default: 1.0)
            escalation_data: Optional escalation data from API
        
        Returns:
            Calculated escalation factor (bounded)
        """
        escalation_factor = base_escalation_factor
        
        if escalation_data:
            # Use escalation factor from data if available
            if 'escalation_factor' in escalation_data:
                escalation_factor = escalation_data['escalation_factor']
            elif 'threat_count' in escalation_data:
                # Calculate from threat count
                escalation_factor = 1.0 + (escalation_data['threat_count'] * self.threat_multiplier)
        
        # Bound the escalation factor
        escalation_factor = max(self.min_escalation_factor, min(self.max_escalation_factor, escalation_factor))
        
        return escalation_factor
    
    def adjust(
        self,
        risk_values: np.ndarray,
        utility_values: np.ndarray,
        base_threshold: float = 0.30,
        escalation_api_url: Optional[str] = None,
        threat_keywords: Optional[List[str]] = None,
        escalation_data: Optional[Dict[str, Any]] = None,
        escalation_factor: Optional[float] = None
    ) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Adjust thresholds based on escalation contexts.
        
        Args:
            risk_values: Risk deltas (e.g., cyber pruning probabilities)
            utility_values: Utility scores (e.g., lives saved percentage)
            base_threshold: Initial risk threshold (default: 0.30)
            escalation_api_url: Optional API endpoint for threat data
            threat_keywords: Optional list of keywords to detect threats
            escalation_data: Optional pre-fetched escalation data
            escalation_factor: Optional direct escalation factor override
        
        Returns:
            Tuple of (adjusted_threshold, adjusted_risks, adjusted_utilities, metadata)
        """
        # Input validation
        if not isinstance(risk_values, np.ndarray):
            risk_values = np.array(risk_values)
        if not isinstance(utility_values, np.ndarray):
            utility_values = np.array(utility_values)
        
        if len(risk_values) != len(utility_values):
            raise ValueError(f"risk_values and utility_values must have same length (got {len(risk_values)} and {len(utility_values)})")
        
        if len(risk_values) == 0:
            raise ValueError("risk_values and utility_values must not be empty")
        
        # Validate base threshold
        base_threshold = validate_float_range(base_threshold, "base_threshold", 0.0, 1.0)
        
        # Fetch escalation data if URL provided and data not already provided
        if escalation_api_url and escalation_data is None:
            default_keywords = threat_keywords if threat_keywords else ['PLA threats', 'battlefield', 'escalation']
            escalation_data = self.fetch_escalation_data(
                escalation_api_url=escalation_api_url,
                threat_keywords=default_keywords
            )
        
        # Calculate escalation factor
        if escalation_factor is None:
            escalation_factor = self.calculate_escalation_factor(
                base_escalation_factor=1.0,
                escalation_data=escalation_data
            )
        
        # Adjust threshold (lower threshold in high-escalation scenarios)
        # Division is safe because escalation_factor is bounded to be >= min_escalation_factor > 0
        adjusted_threshold = base_threshold / escalation_factor
        adjusted_threshold = max(self.min_threshold, min(self.max_threshold, adjusted_threshold))
        
        # Boost utilities in high-escalation scenarios
        adjusted_utilities = utility_values * escalation_factor
        # Clamp utilities to valid range [0, 1] or reasonable maximum
        adjusted_utilities = np.clip(adjusted_utilities, 0.0, 1.0)
        
        # Risks remain unchanged (or could be adjusted if needed)
        adjusted_risks = risk_values.copy()
        
        # Build metadata
        metadata = {
            'base_threshold': float(base_threshold),
            'adjusted_threshold': float(adjusted_threshold),
            'escalation_factor': float(escalation_factor),
            'threshold_change': float(adjusted_threshold - base_threshold),
            'threshold_change_percent': float((adjusted_threshold - base_threshold) / base_threshold * 100) if base_threshold > 0 else 0.0,
            'utility_boost': float(escalation_factor - 1.0),
            'utility_boost_percent': float((escalation_factor - 1.0) * 100),
            'n_values': int(len(risk_values))
        }
        
        if escalation_data:
            metadata['escalation_data'] = {
                'threat_count': escalation_data.get('threat_count', 0),
                'posts_analyzed': escalation_data.get('posts_analyzed', 0),
                'keywords_found': escalation_data.get('keywords_found', []),
                'api_url': escalation_data.get('api_url')
            }
        
        logger.info(
            f"Escalation threshold adjustment: "
            f"base={base_threshold:.3f} -> adjusted={adjusted_threshold:.3f} "
            f"(factor={escalation_factor:.3f})"
        )
        
        return adjusted_threshold, adjusted_risks, adjusted_utilities, metadata


def adjust_escalation_thresholds(
    risk_values: np.ndarray,
    utility_values: np.ndarray,
    base_threshold: float = 0.30,
    escalation_api_url: Optional[str] = None,
    threat_keywords: Optional[List[str]] = None,
    escalation_data: Optional[Dict[str, Any]] = None,
    escalation_factor: Optional[float] = None
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Universal function to recalibrate thresholds based on verbal escalation contexts.
    
    Fetches/analyzes data for threats (e.g., PLA commentary) and adjusts to avoid conservatism.
    Integrates with PRHP framework for context-aware threshold adjustment.
    
    Args:
        risk_values: Risk deltas (e.g., cyber pruning probabilities)
        utility_values: Utility scores (e.g., lives saved percentage)
        base_threshold: Initial risk threshold (default: 0.30)
        escalation_api_url: Optional API endpoint for threat data (e.g., X search endpoint)
        threat_keywords: Optional list of keywords to detect escalation (default: ['PLA threats', 'battlefield', 'escalation'])
        escalation_data: Optional pre-fetched escalation data dict
        escalation_factor: Optional direct escalation factor override (1.0 = no adjustment)
    
    Returns:
        Tuple of (adjusted_threshold, adjusted_risks, adjusted_utilities, metadata):
        - adjusted_threshold: New threshold after escalation adjustment
        - adjusted_risks: Risk values (unchanged by default)
        - adjusted_utilities: Utility values boosted by escalation factor
        - metadata: Dictionary with adjustment details
    
    Example:
        >>> risks = np.array([0.35])  # Cyber risk
        >>> utils = np.array([0.95])
        >>> api = "https://api.example.com/x-search?query=PLA threats Taiwan"
        >>> new_thresh, adj_risks, adj_utils, meta = adjust_escalation_thresholds(
        ...     risks, utils, escalation_api_url=api
        ... )
        >>> print(f"Adjusted Threshold: {new_thresh:.3f}, Utils: {adj_utils}")
    """
    adjuster = EscalationThresholdAdjuster()
    
    return adjuster.adjust(
        risk_values=risk_values,
        utility_values=utility_values,
        base_threshold=base_threshold,
        escalation_api_url=escalation_api_url,
        threat_keywords=threat_keywords,
        escalation_data=escalation_data,
        escalation_factor=escalation_factor
    )


def adjust_prhp_escalation_thresholds(
    prhp_results: Dict[str, Dict[str, Any]],
    base_threshold: float = 0.30,
    escalation_api_url: Optional[str] = None,
    threat_keywords: Optional[List[str]] = None,
    escalation_data: Optional[Dict[str, Any]] = None,
    escalation_factor: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Adjust escalation thresholds for PRHP simulation results.
    
    Args:
        prhp_results: PRHP simulation results dictionary
        base_threshold: Initial risk threshold
        escalation_api_url: Optional API endpoint for threat data
        threat_keywords: Optional list of keywords to detect threats
        escalation_data: Optional pre-fetched escalation data
        escalation_factor: Optional direct escalation factor override
    
    Returns:
        Dictionary with adjusted results and metadata
    """
    adjuster = EscalationThresholdAdjuster()
    adjusted_results = {}
    
    for variant, data in prhp_results.items():
        try:
            # Extract risk and utility values
            risk_value = data.get('asymmetry_delta', 0.0)
            utility_value = data.get('mean_fidelity', 0.0)
            
            # Convert to arrays
            risk_values = np.array([risk_value])
            utility_values = np.array([utility_value])
            
            # Adjust thresholds
            adjusted_threshold, adjusted_risks, adjusted_utilities, metadata = adjuster.adjust(
                risk_values=risk_values,
                utility_values=utility_values,
                base_threshold=base_threshold,
                escalation_api_url=escalation_api_url,
                threat_keywords=threat_keywords,
                escalation_data=escalation_data,
                escalation_factor=escalation_factor
            )
            
            # Update results
            adjusted_data = data.copy()
            adjusted_data['asymmetry_delta'] = float(adjusted_risks[0])
            adjusted_data['mean_fidelity'] = float(adjusted_utilities[0])
            
            # Add escalation adjustment metadata
            adjusted_data['escalation_adjustment'] = metadata
            
            adjusted_results[variant] = adjusted_data
            
        except Exception as e:
            logger.error(f"Error adjusting escalation thresholds for {variant}: {e}")
            adjusted_results[variant] = data  # Use original data on error
    
    return adjusted_results


# Example usage
if __name__ == "__main__":
    # Example with sample data
    risks = np.array([0.35])  # Cyber risk
    utils = np.array([0.95])
    
    print("Original:")
    print(f"  Risk: {risks[0]:.3f}")
    print(f"  Utility: {utils[0]:.3f}")
    print(f"  Base Threshold: 0.30")
    
    # Adjust with mock escalation factor
    new_thresh, adj_risks, adj_utils, meta = adjust_escalation_thresholds(
        risks, utils, base_threshold=0.30, escalation_factor=1.2
    )
    
    print("\nAdjusted (escalation_factor=1.2):")
    print(f"  Risk: {adj_risks[0]:.3f}")
    print(f"  Utility: {adj_utils[0]:.3f}")
    print(f"  Adjusted Threshold: {new_thresh:.3f}")
    print(f"\nMetadata:")
    print(f"  Escalation Factor: {meta['escalation_factor']:.3f}")
    print(f"  Threshold Change: {meta['threshold_change']:.3f} ({meta['threshold_change_percent']:.1f}%)")
    print(f"  Utility Boost: {meta['utility_boost']:.3f} ({meta['utility_boost_percent']:.1f}%)")

