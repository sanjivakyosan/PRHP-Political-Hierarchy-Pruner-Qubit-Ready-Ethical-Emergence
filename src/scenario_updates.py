"""
Scenario Updates Module for PRHP Framework

Provides real-time scenario updates from API or file sources for dynamic
integration into PRHP simulations.

Copyright © sanjivakyosan 2025
"""

import pandas as pd
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from pathlib import Path
import json

try:
    from .utils import get_logger, validate_float_range
except ImportError:
    from utils import get_logger, validate_float_range

logger = get_logger()


class ScenarioUpdateManager:
    """
    Manages real-time scenario updates from multiple sources.
    
    Supports:
    - API endpoints (JSON responses)
    - Local files (CSV/JSON)
    - Multiple merge strategies (overwrite, average, weighted)
    - PRHP framework integration
    """
    
    def __init__(
        self,
        merge_strategy: Literal['overwrite', 'average', 'weighted'] = 'weighted',
        update_weight: float = 0.3
    ):
        """
        Initialize scenario update manager.
        
        Args:
            merge_strategy: How to merge updates ('overwrite', 'average', 'weighted')
            update_weight: Weight for updates when using 'weighted' strategy (0.0-1.0)
        """
        self.merge_strategy = merge_strategy
        self.update_weight = validate_float_range(update_weight, "update_weight", 0.0, 1.0)
        self.update_history: List[Dict[str, Any]] = []
    
    def fetch_from_api(
        self,
        api_url: str,
        timeout: int = 10,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Fetch updates from API endpoint.
        
        Args:
            api_url: API endpoint URL
            timeout: Request timeout in seconds
            headers: Optional HTTP headers
        
        Returns:
            Dictionary with update data
        
        Raises:
            ValueError: If API request fails
            requests.RequestException: If network error occurs
        """
        try:
            logger.info(f"Fetching scenario updates from API: {api_url}")
            response = requests.get(api_url, timeout=timeout, headers=headers or {})
            response.raise_for_status()
            
            # Try to parse as JSON
            try:
                updates = response.json()
            except json.JSONDecodeError:
                # If not JSON, try to parse as text/CSV
                logger.warning("API response is not JSON, attempting CSV parse...")
                from io import StringIO
                updates_df = pd.read_csv(StringIO(response.text))
                updates = updates_df.iloc[-1].to_dict() if len(updates_df) > 0 else {}
            
            if not isinstance(updates, dict):
                # If response is a list, use last item
                if isinstance(updates, list) and len(updates) > 0:
                    updates = updates[-1]
                else:
                    raise ValueError(f"Unexpected API response format: {type(updates)}")
            
            logger.info(f"Successfully fetched {len(updates)} update fields from API")
            return updates
        
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise ValueError(f"Failed to fetch from API: {e}")
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        use_latest: bool = True
    ) -> Dict[str, Any]:
        """
        Load updates from local file.
        
        Args:
            file_path: Path to update file (CSV/JSON)
            use_latest: If True, use latest row (for CSV) or last item (for JSON list)
        
        Returns:
            Dictionary with update data
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Update file not found: {file_path}")
        
        try:
            if file_path.suffix == '.csv':
                update_df = pd.read_csv(file_path)
                if len(update_df) == 0:
                    raise ValueError("Update file is empty")
                
                if use_latest:
                    updates = update_df.iloc[-1].to_dict()
                else:
                    # Use first row
                    updates = update_df.iloc[0].to_dict()
                
                logger.info(f"Loaded {len(updates)} update fields from CSV: {file_path}")
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    if len(data) == 0:
                        raise ValueError("Update file is empty")
                    updates = data[-1] if use_latest else data[0]
                elif isinstance(data, dict):
                    updates = data
                else:
                    raise ValueError(f"Unexpected JSON format: {type(data)}")
                
                logger.info(f"Loaded {len(updates)} update fields from JSON: {file_path}")
            
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. Use .csv or .json"
                )
            
            return updates
        
        except Exception as e:
            logger.error(f"Error loading update file: {e}")
            raise
    
    def merge_updates(
        self,
        current_data: Union[pd.DataFrame, Dict[str, Any]],
        updates: Dict[str, Any],
        update_keys: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Merge updates into current data using configured strategy.
        
        Args:
            current_data: Current simulation data (DataFrame or dict)
            updates: Update data dictionary
            update_keys: Keys to update (if None, uses all keys in updates)
        
        Returns:
            Updated DataFrame
        """
        # Convert current_data to DataFrame if needed
        if isinstance(current_data, dict):
            # Handle PRHP results format: {variant: {metrics...}}
            if len(current_data) > 0 and isinstance(list(current_data.values())[0], dict):
                # PRHP results format - convert to DataFrame
                current_df = pd.DataFrame([list(current_data.values())[0]])
            else:
                current_df = pd.DataFrame([current_data])
        elif current_data is None:
            current_df = pd.DataFrame()
        else:
            current_df = current_data.copy()
        
        # Determine which keys to update
        if update_keys is None:
            update_keys = list(updates.keys())
        
        # Filter to only keys that exist in updates
        update_keys = [k for k in update_keys if k in updates]
        
        if not update_keys:
            logger.warning("No matching update keys found. Returning original data.")
            return current_df
        
        # Apply merge strategy
        for key in update_keys:
            if key not in current_df.columns:
                # Add new column
                current_df[key] = updates[key]
            else:
                # Merge existing column based on strategy
                if self.merge_strategy == 'overwrite':
                    current_df[key] = updates[key]
                
                elif self.merge_strategy == 'average':
                    # Average current and update values
                    current_df[key] = (current_df[key] + updates[key]) / 2.0
                
                elif self.merge_strategy == 'weighted':
                    # Weighted average: (1 - weight) * current + weight * update
                    current_weight = 1.0 - self.update_weight
                    current_df[key] = (
                        current_weight * current_df[key] +
                        self.update_weight * updates[key]
                    )
        
        # Add timestamp
        current_df['update_time'] = datetime.now().isoformat()
        current_df['update_source'] = 'api' if 'api' in str(updates.get('source', '')) else 'file'
        
        # Track update history
        self.update_history.append({
            'timestamp': datetime.now().isoformat(),
            'keys_updated': update_keys,
            'strategy': self.merge_strategy,
            'update_count': len(self.update_history) + 1
        })
        
        logger.info(
            f"Merged {len(update_keys)} fields using '{self.merge_strategy}' strategy"
        )
        
        return current_df
    
    def add_scenario_updates(
        self,
        current_data: Union[pd.DataFrame, Dict[str, Any]],
        api_url: Optional[str] = None,
        local_update_file: Optional[Union[str, Path]] = None,
        update_keys: Optional[List[str]] = None,
        timeout: int = 10
    ) -> pd.DataFrame:
        """
        Universal function to pull and merge real-time updates.
        
        Handles API or file sources for dynamic integration.
        
        Args:
            current_data: Current simulation data (DataFrame or PRHP results dict)
            api_url: Optional API endpoint (e.g., for crisis data)
            local_update_file: Optional path to update file (CSV/JSON)
            update_keys: Keys to update (if None, uses all keys in updates)
            timeout: API request timeout in seconds
        
        Returns:
            Updated DataFrame with merged updates
        
        Raises:
            ValueError: If neither api_url nor local_update_file provided
            FileNotFoundError: If local file doesn't exist
        """
        if not api_url and not local_update_file:
            raise ValueError(
                "Either api_url or local_update_file must be provided"
            )
        
        # Fetch updates
        updates = {}
        
        if api_url:
            try:
                updates = self.fetch_from_api(api_url, timeout=timeout)
            except Exception as e:
                logger.error(f"Failed to fetch from API: {e}")
                # Try file as fallback if provided
                if local_update_file:
                    logger.info(f"Falling back to file: {local_update_file}")
                    updates = self.load_from_file(local_update_file)
                else:
                    raise
        
        elif local_update_file:
            updates = self.load_from_file(local_update_file)
        
        if not updates:
            raise ValueError("No updates retrieved from source")
        
        # Merge updates
        updated_df = self.merge_updates(current_data, updates, update_keys)
        
        return updated_df


# Convenience function for backward compatibility
def add_scenario_updates(
    api_url: Optional[str] = None,
    local_update_file: Optional[Union[str, Path]] = None,
    current_df: Optional[pd.DataFrame] = None,
    update_keys: Optional[List[str]] = None,
    merge_strategy: Literal['overwrite', 'average', 'weighted'] = 'weighted',
    update_weight: float = 0.3
) -> pd.DataFrame:
    """
    Universal function to pull and merge real-time updates.
    
    This is a convenience wrapper around ScenarioUpdateManager.
    
    Args:
        api_url: Optional API endpoint (e.g., for crisis data)
        local_update_file: Optional path to update file (CSV/JSON)
        current_df: Current simulation data DataFrame
        update_keys: Keys to update (default: ['risk_delta', 'utility_score'])
        merge_strategy: How to merge updates ('overwrite', 'average', 'weighted')
        update_weight: Weight for updates when using 'weighted' strategy
    
    Returns:
        Updated DataFrame with merged updates
    """
    if update_keys is None:
        update_keys = ['risk_delta', 'utility_score']
    
    manager = ScenarioUpdateManager(
        merge_strategy=merge_strategy,
        update_weight=update_weight
    )
    
    return manager.add_scenario_updates(
        current_data=current_df,
        api_url=api_url,
        local_update_file=local_update_file,
        update_keys=update_keys
    )


def add_prhp_scenario_updates(
    prhp_results: Dict[str, Dict[str, Any]],
    api_url: Optional[str] = None,
    local_update_file: Optional[Union[str, Path]] = None,
    update_keys: Optional[List[str]] = None,
    merge_strategy: Literal['overwrite', 'average', 'weighted'] = 'weighted',
    update_weight: float = 0.3,
    variant: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Add scenario updates to PRHP simulation results.
    
    Args:
        prhp_results: PRHP results dict: {variant: {metrics...}}
        api_url: Optional API endpoint
        local_update_file: Optional path to update file
        update_keys: Keys to update (default: PRHP metrics)
        merge_strategy: How to merge updates
        update_weight: Weight for updates
        variant: Optional variant to update (if None, updates all)
    
    Returns:
        Dictionary mapping variant -> updated DataFrame
    """
    if update_keys is None:
        update_keys = [
            'mean_fidelity',
            'asymmetry_delta',
            'novelty_gen',
            'mean_phi_delta',
            'mean_success_rate'
        ]
    
    manager = ScenarioUpdateManager(
        merge_strategy=merge_strategy,
        update_weight=update_weight
    )
    
    results = {}
    variants_to_process = [variant] if variant else list(prhp_results.keys())
    
    for v in variants_to_process:
        if v not in prhp_results:
            continue
        
        try:
            updated_df = manager.add_scenario_updates(
                current_data=prhp_results[v],
                api_url=api_url,
                local_update_file=local_update_file,
                update_keys=update_keys
            )
            results[v] = updated_df
            logger.info(f"Scenario updates applied to {v}")
        except Exception as e:
            logger.error(f"Failed to apply updates to {v}: {e}")
            # Return original data as DataFrame
            results[v] = pd.DataFrame([prhp_results[v]])
    
    return results


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage with DataFrame
    current_df = pd.DataFrame({
        'risk_delta': [0.28],
        'utility_score': [0.92]
    })
    
    # Note: This requires an API endpoint or file
    # updated = add_scenario_updates(
    #     api_url="https://api.example.com/crisis-updates",
    #     current_df=current_df
    # )
    # print(updated)
    
    # Example 2: With PRHP results
    prhp_results = {
        'ADHD-collectivist': {
            'mean_fidelity': 0.84,
            'asymmetry_delta': 0.28,
            'novelty_gen': 0.90
        }
    }
    
    # updated_results = add_prhp_scenario_updates(
    #     prhp_results=prhp_results,
    #     local_update_file='scenario_updates.json'
    # )
    # print(updated_results)

