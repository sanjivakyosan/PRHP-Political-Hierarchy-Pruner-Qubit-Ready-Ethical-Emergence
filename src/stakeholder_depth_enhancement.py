"""
Stakeholder Depth Enhancement Module for PRHP Framework

Provides integration of external stakeholder data (local voices, neurodiverse guidelines)
into PRHP variant configurations for more inclusive and context-aware simulations.

Copyright © sanjivakyosan 2025
"""

import json
import requests
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import warnings

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    warnings.warn("pandas not available. CSV file support will be limited.")

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


class StakeholderDepthEnhancer:
    """
    Enhances PRHP variants with external stakeholder data and neurodiverse guidelines.
    
    Provides:
    - Local stakeholder input integration (X/web APIs, local files)
    - Neurodiverse guidelines integration
    - Variant weighting and configuration
    - Metadata tracking
    """
    
    def __init__(
        self,
        default_stakeholder_weight: float = 0.2,
        max_stakeholder_items: int = 5,
        timeout: int = 10,
        min_variant_weight: float = 0.0,
        max_variant_weight: float = 2.0
    ):
        """
        Initialize the stakeholder depth enhancer.
        
        Args:
            default_stakeholder_weight: Default weight for stakeholder variants (0.0-1.0)
            max_stakeholder_items: Maximum number of stakeholder items to include
            timeout: Request timeout in seconds
            min_variant_weight: Minimum allowed variant weight
            max_variant_weight: Maximum allowed variant weight
        """
        self.default_stakeholder_weight = validate_float_range(
            default_stakeholder_weight, "default_stakeholder_weight", 0.0, 1.0
        )
        self.max_stakeholder_items = max(1, int(max_stakeholder_items))
        self.timeout = max(1, int(timeout))
        self.min_variant_weight = min_variant_weight
        self.max_variant_weight = max_variant_weight
    
    def fetch_local_stakeholder_data(
        self,
        api_url: str,
        query: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch local stakeholder data from API endpoint.
        
        Args:
            api_url: API endpoint URL
            query: Search query for stakeholder voices
            headers: Optional HTTP headers
            params: Optional query parameters (query will be added if not present)
        
        Returns:
            List of stakeholder data items
        """
        try:
            request_params = params.copy() if params else {}
            if 'query' not in request_params:
                request_params['query'] = query
            
            response = requests.get(
                api_url,
                params=request_params,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract results (handle various API response formats)
            if isinstance(data, dict):
                results = data.get('results', data.get('data', data.get('items', [])))
            elif isinstance(data, list):
                results = data
            else:
                results = []
            
            # Limit results
            return results[:self.max_stakeholder_items]
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching stakeholder data from {api_url}")
            return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching stakeholder data from {api_url}: {e}")
            return []
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning(f"Error parsing stakeholder data response: {e}")
            return []
    
    def load_neurodiverse_guidelines(
        self,
        guidelines_file: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load neurodiverse guidelines from file.
        
        Args:
            guidelines_file: Path to guidelines file (JSON or CSV)
        
        Returns:
            Dictionary with guidelines data
        """
        guidelines_path = Path(guidelines_file)
        
        if not guidelines_path.exists():
            logger.warning(f"Guidelines file not found: {guidelines_file}")
            return {}
        
        try:
            if guidelines_path.suffix.lower() == '.json':
                with open(guidelines_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif guidelines_path.suffix.lower() == '.csv':
                if not HAS_PANDAS:
                    logger.warning("pandas not available. Cannot read CSV guidelines file.")
                    return {}
                df = pd.read_csv(guidelines_path)
                # Convert DataFrame to dict (use first row as default, or aggregate)
                return df.to_dict('records')[0] if len(df) > 0 else {}
            else:
                logger.warning(f"Unsupported guidelines file format: {guidelines_path.suffix}")
                return {}
        except Exception as e:
            logger.error(f"Error loading guidelines from {guidelines_file}: {e}")
            return {}
    
    def enhance_variants(
        self,
        current_variants: List[Dict[str, Any]],
        api_url: Optional[str] = None,
        local_query: str = 'Ukraine local voices',
        neuro_guidelines_file: Optional[Union[str, Path]] = None,
        stakeholder_weight: Optional[float] = None,
        add_stakeholder_variant: bool = True,
        apply_guidelines_to_hybrid: bool = True
    ) -> Dict[str, Any]:
        """
        Enhance variants with stakeholder data and neurodiverse guidelines.
        
        Args:
            current_variants: List of variant dicts (e.g., [{'name': 'ADHD-collectivist', 'weight': 1.0}])
            api_url: Optional API endpoint for local stakeholder voices
            local_query: Query string for stakeholder search
            neuro_guidelines_file: Optional path to neurodiverse guidelines file (JSON/CSV)
            stakeholder_weight: Weight for stakeholder variant (default: self.default_stakeholder_weight)
            add_stakeholder_variant: If True, add a new stakeholder variant
            apply_guidelines_to_hybrid: If True, apply guidelines to hybrid variants
        
        Returns:
            Dictionary with:
            - 'enhanced_variants': Enhanced variants list
            - 'metadata': Enhancement metadata
        """
        # Input validation
        if not isinstance(current_variants, list):
            raise ValueError("current_variants must be a list")
        
        if len(current_variants) == 0:
            logger.warning("No variants provided. Returning empty enhancement.")
            return {
                'enhanced_variants': [],
                'metadata': {
                    'stakeholder_data_fetched': False,
                    'guidelines_applied': False,
                    'variants_enhanced': 0
                }
            }
        
        # Validate variant structure
        for i, variant in enumerate(current_variants):
            if not isinstance(variant, dict):
                raise ValueError(f"Variant at index {i} must be a dictionary")
            if 'name' not in variant:
                raise ValueError(f"Variant at index {i} must have a 'name' key")
        
        enhanced = []
        metadata = {
            'stakeholder_data_fetched': False,
            'guidelines_applied': False,
            'variants_enhanced': 0,
            'stakeholder_items_count': 0,
            'guidelines_keys': []
        }
        
        # Copy current variants
        for variant in current_variants:
            enhanced_variant = variant.copy()
            enhanced.append(enhanced_variant)
        
        # Fetch local stakeholder data
        stakeholder_data = []
        if api_url:
            try:
                logger.info(f"Fetching stakeholder data from {api_url} with query: {local_query}")
                stakeholder_data = self.fetch_local_stakeholder_data(
                    api_url=api_url,
                    query=local_query
                )
                
                if stakeholder_data:
                    metadata['stakeholder_data_fetched'] = True
                    metadata['stakeholder_items_count'] = len(stakeholder_data)
                    logger.info(f"Fetched {len(stakeholder_data)} stakeholder items")
                    
                    # Add stakeholder variant if requested
                    if add_stakeholder_variant:
                        weight = stakeholder_weight if stakeholder_weight is not None else self.default_stakeholder_weight
                        weight = validate_float_range(weight, "stakeholder_weight", self.min_variant_weight, self.max_variant_weight)
                        
                        stakeholder_variant = {
                            'name': 'local-stakeholder',
                            'input': stakeholder_data,
                            'weight': weight,
                            'source': api_url,
                            'query': local_query
                        }
                        enhanced.append(stakeholder_variant)
                        logger.info(f"Added stakeholder variant with weight {weight}")
            except Exception as e:
                logger.error(f"Error processing stakeholder data: {e}")
        
        # Load and apply neurodiverse guidelines
        guidelines = {}
        if neuro_guidelines_file:
            try:
                logger.info(f"Loading neurodiverse guidelines from {neuro_guidelines_file}")
                guidelines = self.load_neurodiverse_guidelines(neuro_guidelines_file)
                
                if guidelines:
                    metadata['guidelines_applied'] = True
                    metadata['guidelines_keys'] = list(guidelines.keys())
                    logger.info(f"Loaded guidelines with keys: {metadata['guidelines_keys']}")
                    
                    # Apply guidelines to hybrid variants
                    if apply_guidelines_to_hybrid:
                        for variant in enhanced:
                            variant_name = variant.get('name', '').lower()
                            if 'hybrid' in variant_name:
                                # Add guidelines to variant
                                variant['guidelines'] = guidelines.get('safeguards', guidelines.get('guidelines', guidelines))
                                variant['guidelines_source'] = str(neuro_guidelines_file)
                                metadata['variants_enhanced'] += 1
                                logger.info(f"Applied guidelines to variant: {variant.get('name')}")
            except Exception as e:
                logger.error(f"Error processing neurodiverse guidelines: {e}")
        
        return {
            'enhanced_variants': enhanced,
            'metadata': metadata
        }


def enhance_stakeholder_depth(
    current_variants: List[Dict[str, Any]],
    api_url: Optional[str] = None,
    local_query: str = 'Ukraine local voices',
    neuro_guidelines_file: Optional[Union[str, Path]] = None,
    stakeholder_weight: Optional[float] = None,
    default_stakeholder_weight: float = 0.2,
    max_stakeholder_items: int = 5,
    add_stakeholder_variant: bool = True,
    apply_guidelines_to_hybrid: bool = True
) -> Dict[str, Any]:
    """
    Universal function to add depth via external stakeholder data.
    
    Integrates local input (e.g., from X/web) and neurodiverse guidelines into variants.
    Integrates with PRHP framework for context-aware variant enhancement.
    
    Args:
        current_variants: List of variant dicts (e.g., [{'name': 'ADHD-collectivist', 'weight': 1.0}])
        api_url: Optional API endpoint for local stakeholder voices (e.g., X search endpoint)
        local_query: Query string for stakeholder search (default: 'Ukraine local voices')
        neuro_guidelines_file: Optional path to neurodiverse guidelines file (JSON/CSV)
        stakeholder_weight: Weight for stakeholder variant (default: default_stakeholder_weight)
        default_stakeholder_weight: Default weight for stakeholder variants (0.0-1.0, default: 0.2)
        max_stakeholder_items: Maximum number of stakeholder items to include (default: 5)
        add_stakeholder_variant: If True, add a new stakeholder variant (default: True)
        apply_guidelines_to_hybrid: If True, apply guidelines to hybrid variants (default: True)
    
    Returns:
        Dictionary with:
        - 'enhanced_variants': Enhanced variants list with stakeholder data and guidelines
        - 'metadata': Enhancement metadata (stakeholder_data_fetched, guidelines_applied, etc.)
    
    Example:
        >>> variants = [
        ...     {'name': 'ADHD-collectivist', 'weight': 1.0},
        ...     {'name': 'neurotypical-hybrid', 'weight': 1.0}
        ... ]
        >>> result = enhance_stakeholder_depth(
        ...     variants,
        ...     api_url="https://api.example.com/x-search",
        ...     neuro_guidelines_file='neuro_ethics.json'
        ... )
        >>> print(result['enhanced_variants'])
        >>> print(result['metadata'])
    """
    enhancer = StakeholderDepthEnhancer(
        default_stakeholder_weight=default_stakeholder_weight,
        max_stakeholder_items=max_stakeholder_items
    )
    
    return enhancer.enhance_variants(
        current_variants=current_variants,
        api_url=api_url,
        local_query=local_query,
        neuro_guidelines_file=neuro_guidelines_file,
        stakeholder_weight=stakeholder_weight,
        add_stakeholder_variant=add_stakeholder_variant,
        apply_guidelines_to_hybrid=apply_guidelines_to_hybrid
    )


def enhance_prhp_variants(
    prhp_variants: List[str],
    api_url: Optional[str] = None,
    local_query: str = 'Ukraine local voices',
    neuro_guidelines_file: Optional[Union[str, Path]] = None,
    stakeholder_weight: Optional[float] = None
) -> Dict[str, Any]:
    """
    Enhance PRHP framework variants with stakeholder data.
    
    Converts PRHP variant names to variant dicts, enhances them, and returns
    enhanced configuration compatible with PRHP framework.
    
    Args:
        prhp_variants: List of PRHP variant names (e.g., ['ADHD-collectivist', 'neurotypical-hybrid'])
        api_url: Optional API endpoint for stakeholder data
        local_query: Query string for stakeholder search
        neuro_guidelines_file: Optional path to neurodiverse guidelines file
        stakeholder_weight: Weight for stakeholder variant
    
    Returns:
        Dictionary with enhanced variants and metadata
    """
    # Convert PRHP variant names to variant dicts
    variant_dicts = [{'name': v, 'weight': 1.0} for v in prhp_variants]
    
    # Enhance variants
    result = enhance_stakeholder_depth(
        current_variants=variant_dicts,
        api_url=api_url,
        local_query=local_query,
        neuro_guidelines_file=neuro_guidelines_file,
        stakeholder_weight=stakeholder_weight
    )
    
    return result


# Example usage
if __name__ == "__main__":
    # Example with PRHP variants
    variants = [
        {'name': 'ADHD-collectivist', 'weight': 1.0},
        {'name': 'neurotypical-hybrid', 'weight': 1.0}
    ]
    
    print("Original variants:")
    for v in variants:
        print(f"  - {v['name']}: weight={v['weight']}")
    
    # Enhance with stakeholder data (mock - no real API)
    result = enhance_stakeholder_depth(
        current_variants=variants,
        api_url=None,  # Would be "https://api.example.com/x-search" in real usage
        local_query='Ukraine local voices',
        neuro_guidelines_file=None  # Would be 'neuro_ethics.json' in real usage
    )
    
    print("\nEnhanced variants:")
    for v in result['enhanced_variants']:
        print(f"  - {v['name']}: weight={v.get('weight', 'N/A')}")
        if 'guidelines' in v:
            print(f"    Guidelines: {list(v['guidelines'].keys()) if isinstance(v['guidelines'], dict) else 'Applied'}")
        if 'input' in v:
            print(f"    Stakeholder items: {len(v['input'])}")
    
    print(f"\nMetadata:")
    print(f"  Stakeholder data fetched: {result['metadata']['stakeholder_data_fetched']}")
    print(f"  Guidelines applied: {result['metadata']['guidelines_applied']}")
    print(f"  Variants enhanced: {result['metadata']['variants_enhanced']}")

