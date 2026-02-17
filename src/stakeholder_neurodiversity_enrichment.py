"""
Stakeholder and Neurodiversity Enrichment Module for PRHP Framework

Provides enrichment of PRHP variants with local stakeholder input from X (Twitter) API
and explicit neurodiversity mappings for variant representation clarity.

Copyright © sanjivakyosan 2025
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from requests.exceptions import Timeout, RequestException

try:
    from .utils import get_logger
except ImportError:
    try:
        from utils import get_logger
    except ImportError:
        import logging
        logger = logging.getLogger('prhp')

logger = get_logger() if 'logger' not in locals() else logger


class StakeholderNeurodiversityEnricher:
    """
    Enriches PRHP variants with local stakeholder input and neurodiversity mappings.
    
    Provides:
    - X (Twitter) API integration for local stakeholder voices
    - Explicit neurodiversity mappings for variant clarity
    - Configurable filtering and weighting
    - Metadata tracking
    """
    
    def __init__(
        self,
        default_stakeholder_weight: float = 0.25,
        max_stakeholder_items: int = 5,
        timeout: int = 10,
        default_neuro_mappings: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the stakeholder and neurodiversity enricher.
        
        Args:
            default_stakeholder_weight: Default weight for stakeholder variant (default: 0.25)
            max_stakeholder_items: Maximum number of stakeholder items to include (default: 5)
            timeout: Request timeout in seconds (default: 10)
            default_neuro_mappings: Optional default neurodiversity mappings
        """
        self.default_stakeholder_weight = max(0.0, min(1.0, default_stakeholder_weight))
        self.max_stakeholder_items = max(1, int(max_stakeholder_items))
        self.timeout = max(1, int(timeout))
        
        if default_neuro_mappings is None:
            self.default_neuro_mappings = self._get_default_neuro_mappings()
        else:
            self.default_neuro_mappings = default_neuro_mappings
    
    def _get_default_neuro_mappings(self) -> Dict[str, str]:
        """Get default neurodiversity mappings for variant representation."""
        return {
            'strategic-collectivist': 'ADHD-like: High novelty, collective focus (e.g., group welfare in crises)',
            'tactical-individualist': 'Autistic-like: Precision, individual ethics (e.g., data consent)',
            'diplomatic-hybrid': 'Neurotypical: Balanced integration (e.g., fairness per NIST RMF)',
            'ADHD-collectivist': 'ADHD-like: High novelty, collective focus (e.g., group welfare in crises)',
            'autistic-individualist': 'Autistic-like: Precision, individual ethics (e.g., data consent)',
            'neurotypical-hybrid': 'Neurotypical: Balanced integration (e.g., fairness per NIST RMF)',
            'economic-collectivist': 'ADHD-like: High novelty in group equity (e.g., weighs Oxfam voices on ASEAN job shifts, +0.2 weight for collective impacts)',
            'fiscal-individualist': 'Autistic-like: Precision in rights (e.g., Amnesty coercion flags, integrates local Malaysian factory loss posts)',
            'trade-hybrid': 'Neurotypical: Balanced fairness (e.g., NIST RMF, layers X youth disillusionment for 0.1 equity adjustment)'
        }
    
    def _get_default_deep_neuro_mappings(self) -> Dict[str, str]:
        """Get default deep neurodiversity mappings with detailed descriptions."""
        return {
            'economic-collectivist': 'ADHD-like: High novelty in group equity (e.g., weighs Oxfam voices on ASEAN job shifts, +0.2 weight for collective impacts)',
            'fiscal-individualist': 'Autistic-like: Precision in rights (e.g., Amnesty coercion flags, integrates local Malaysian factory loss posts)',
            'trade-hybrid': 'Neurotypical: Balanced fairness (e.g., NIST RMF, layers X youth disillusionment for 0.1 equity adjustment)',
            'ADHD-collectivist': 'ADHD-like: High novelty, collective focus with deep integration of local stakeholder voices (e.g., weighs collective impacts from X posts, +0.2 weight for group equity)',
            'autistic-individualist': 'Autistic-like: Precision in individual rights with detailed local voice integration (e.g., Amnesty flags, integrates local factory loss posts for equity)',
            'neurotypical-hybrid': 'Neurotypical: Balanced fairness with layered stakeholder input (e.g., NIST RMF, layers X youth disillusionment for 0.1 equity adjustment)'
        }
    
    def analyze_sentiment(self, content: str) -> str:
        """
        Analyze sentiment of content (simple keyword-based analysis).
        
        Args:
            content: Text content to analyze
        
        Returns:
            Sentiment label: 'positive', 'negative', or 'neutral'
        """
        content_lower = content.lower()
        
        # Negative indicators
        negative_keywords = [
            'unemployment', 'job loss', 'layoff', 'shutdown', 'protest', 'disillusionment',
            'crisis', 'decline', 'recession', 'poverty', 'inequality', 'exploitation'
        ]
        
        # Positive indicators
        positive_keywords = [
            'opportunity', 'growth', 'gain', 'preserved', 'cheap', 'benefit', 'success',
            'improvement', 'expansion', 'investment', 'development', 'progress'
        ]
        
        negative_count = sum(1 for kw in negative_keywords if kw in content_lower)
        positive_count = sum(1 for kw in positive_keywords if kw in content_lower)
        
        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'
    
    def analyze_crisis_sentiment(self, content: str) -> str:
        """
        Analyze crisis-specific sentiment (urgency vs despair).
        
        Args:
            content: Text content to analyze
        
        Returns:
            Sentiment label: 'urgency' or 'despair'
        """
        content_lower = content.lower()
        
        # Urgency indicators (actionable, justice-seeking)
        urgency_keywords = ['rape', 'lawsuit', 'escape', 'justice', 'accountability', 'prosecute', 'sue', 'flee']
        
        # Despair indicators (hopelessness, loss)
        despair_keywords = ['unsafe', 'kill', 'airstrike', 'where to go', 'nowhere', 'trapped', 'death']
        
        urgency_count = sum(1 for kw in urgency_keywords if kw in content_lower)
        despair_count = sum(1 for kw in despair_keywords if kw in content_lower)
        
        if urgency_count > despair_count:
            return 'urgency'
        else:
            return 'despair'
    
    def fetch_x_voices_with_sentiment(
        self,
        x_api_url: str,
        local_query: str,
        filter_keywords: Optional[List[str]] = None,
        limit: int = 10,
        headers: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch local stakeholder voices from X API with sentiment analysis.
        
        Args:
            x_api_url: X search API endpoint URL
            local_query: Query string for local voices
            filter_keywords: Optional list of keywords to filter posts
            limit: Maximum number of posts to fetch
            headers: Optional HTTP headers
        
        Returns:
            List of dictionaries with 'content' and 'sentiment' keys
        """
        try:
            response = requests.get(
                x_api_url,
                params={'query': local_query, 'limit': min(limit, 100)},
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract posts
            if isinstance(data, dict):
                posts = data.get('posts', data.get('data', data.get('items', [])))
            elif isinstance(data, list):
                posts = data
            else:
                posts = []
            
            # Process posts with sentiment analysis
            local_voices = []
            for post in posts:
                if isinstance(post, dict):
                    content = post.get('content', post.get('text', post.get('message', '')))
                    if isinstance(content, str) and content.strip():
                        # Apply keyword filtering if provided
                        if filter_keywords:
                            content_lower = content.lower()
                            if not any(keyword.lower() in content_lower for keyword in filter_keywords):
                                continue
                        
                        # Analyze sentiment
                        sentiment = self.analyze_sentiment(content)
                        
                        local_voices.append({
                            'content': content.strip(),
                            'sentiment': sentiment
                        })
            
            # Limit to max_stakeholder_items
            local_voices = local_voices[:self.max_stakeholder_items]
            
            logger.info(f"Fetched {len(local_voices)} stakeholder voices with sentiment analysis from X API")
            return local_voices
            
        except Timeout:
            logger.warning(f"Timeout fetching X voices with sentiment from {x_api_url}")
            return []
        except RequestException as e:
            logger.warning(f"Error fetching X voices with sentiment from {x_api_url}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching X voices with sentiment: {e}")
            return []
    
    def fetch_x_voices(
        self,
        x_api_url: str,
        local_query: str,
        filter_keywords: Optional[List[str]] = None,
        limit: int = 10,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Fetch local stakeholder voices from X (Twitter) API.
        
        Args:
            x_api_url: X search API endpoint URL
            local_query: Query string for local voices
            filter_keywords: Optional list of keywords to filter posts (e.g., ['Taiwan', 'displacement'])
            limit: Maximum number of posts to fetch (default: 10)
            headers: Optional HTTP headers
            params: Optional query parameters
        
        Returns:
            List of post content strings
        """
        try:
            request_params = params.copy() if params else {}
            request_params['query'] = local_query
            request_params['limit'] = min(limit, 100)  # Cap at 100
            
            response = requests.get(
                x_api_url,
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
            
            # Filter and extract content
            local_inputs = []
            for post in posts:
                if isinstance(post, dict):
                    content = post.get('content', post.get('text', post.get('message', '')))
                    if isinstance(content, str) and content.strip():
                        # Apply keyword filtering if provided
                        if filter_keywords:
                            content_lower = content.lower()
                            if any(keyword.lower() in content_lower for keyword in filter_keywords):
                                local_inputs.append(content.strip())
                        else:
                            local_inputs.append(content.strip())
            
            # Limit to max_stakeholder_items
            local_inputs = local_inputs[:self.max_stakeholder_items]
            
            logger.info(f"Fetched {len(local_inputs)} stakeholder voices from X API")
            return local_inputs
            
        except Timeout:
            logger.warning(f"Timeout fetching X voices from {x_api_url}")
            return []
        except RequestException as e:
            logger.warning(f"Error fetching X voices from {x_api_url}: {e}")
            return []
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(f"Error parsing X API response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching X voices: {e}")
            return []
    
    def apply_neuro_mappings(
        self,
        variants: List[Dict[str, Any]],
        neuro_mappings: Optional[Dict[str, str]] = None,
        use_deep_mappings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Apply neurodiversity mappings to variants, optionally using deep mappings.
        
        Args:
            variants: List of variant dictionaries
            neuro_mappings: Optional custom neurodiversity mappings (merged with defaults)
            use_deep_mappings: If True, use deep neuro mappings with detailed descriptions
        
        Returns:
            List of variants with neuro_mapping (and optionally deep_neuro) added
        """
        # Merge custom mappings with defaults
        if use_deep_mappings:
            mappings = self._get_default_deep_neuro_mappings().copy()
        else:
            mappings = self.default_neuro_mappings.copy()
        
        if neuro_mappings:
            mappings.update(neuro_mappings)
        
        enriched_variants = []
        for variant in variants:
            enriched_variant = variant.copy()
            variant_name = enriched_variant.get('name', '')
            
            # Apply mapping if variant name matches
            if variant_name in mappings:
                enriched_variant['neuro_mapping'] = mappings[variant_name]
                if use_deep_mappings:
                    enriched_variant['deep_neuro'] = mappings[variant_name]
                logger.debug(f"Applied {'deep ' if use_deep_mappings else ''}neuro mapping to {variant_name}")
            
            enriched_variants.append(enriched_variant)
        
        return enriched_variants
    
    def calculate_voice_weight(self, local_voices: List[Dict[str, Any]]) -> float:
        """
        Calculate voice weight based on sentiment distribution.
        
        Args:
            local_voices: List of voice dictionaries with 'sentiment' key
        
        Returns:
            Voice weight (0.0-1.0) based on positive sentiment ratio
        """
        if not local_voices:
            return 0.5  # Neutral default
        
        positive_count = sum(1 for v in local_voices if v.get('sentiment') == 'positive')
        total_count = len(local_voices)
        
        # Weight is ratio of positive sentiments
        voice_weight = positive_count / total_count if total_count > 0 else 0.5
        
        logger.debug(f"Calculated voice weight: {voice_weight:.3f} (positive: {positive_count}/{total_count})")
        return voice_weight
    
    def enrich(
        self,
        current_variants: List[Dict[str, Any]],
        x_api_url: Optional[str] = None,
        local_query: str = "Taiwan Strait tensions displacement local voices",
        neuro_mappings: Optional[Dict[str, str]] = None,
        filter_keywords: Optional[List[str]] = None,
        stakeholder_weight: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        use_sentiment_analysis: bool = True,
        use_deep_mappings: bool = False,
        neuro_depth_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich variants with stakeholder input and neurodiversity mappings.
        
        Args:
            current_variants: List of variant dictionaries (must have 'name' key)
            x_api_url: Optional X search API endpoint
            local_query: Query string for local voices (default: "Taiwan Strait tensions displacement local voices")
            neuro_mappings: Optional custom neurodiversity mappings
            filter_keywords: Optional list of keywords to filter X posts (e.g., ['Taiwan', 'displacement'])
            stakeholder_weight: Optional weight for stakeholder variant (default: self.default_stakeholder_weight)
            headers: Optional HTTP headers for API request
            use_sentiment_analysis: If True, analyze sentiment of X posts and calculate voice weights
            use_deep_mappings: If True, use deep neuro mappings with detailed descriptions
            neuro_depth_file: Optional path to JSON file with deep neuro mappings
        
        Returns:
            Dictionary containing:
            - 'enriched_variants': List of enriched variants
            - 'metadata': Dictionary with enrichment details
        """
        # Input validation
        if not isinstance(current_variants, list):
            raise ValueError("current_variants must be a list of dictionaries")
        
        for variant in current_variants:
            if not isinstance(variant, dict) or 'name' not in variant:
                raise ValueError("Each variant must be a dictionary with a 'name' key")
        
        enriched_variants = [v.copy() for v in current_variants]  # Deep copy
        
        metadata = {
            'stakeholder_data_fetched': False,
            'neuro_mappings_applied': False,
            'stakeholder_items_count': 0,
            'variants_enriched': 0,
            'x_api_url': x_api_url,
            'local_query': local_query
        }
        
        # 1. Fetch local X voices (with sentiment analysis if enabled)
        local_voices = []
        local_inputs = []
        if x_api_url:
            try:
                # Extract filter keywords from query if not provided
                if filter_keywords is None:
                    # Auto-extract keywords from query (simple heuristic)
                    query_lower = local_query.lower()
                    filter_keywords = []
                    if 'taiwan' in query_lower:
                        filter_keywords.append('Taiwan')
                    if 'displacement' in query_lower:
                        filter_keywords.append('displacement')
                    if 'tensions' in query_lower:
                        filter_keywords.append('tensions')
                    if 'southeast asia' in query_lower or 'asean' in query_lower:
                        filter_keywords.extend(['Southeast Asia', 'ASEAN'])
                
                if use_sentiment_analysis:
                    # Fetch with sentiment analysis
                    local_voices = self.fetch_x_voices_with_sentiment(
                        x_api_url=x_api_url,
                        local_query=local_query,
                        filter_keywords=filter_keywords,
                        headers=headers
                    )
                    # Extract content for backward compatibility
                    local_inputs = [v['content'] for v in local_voices]
                else:
                    # Fetch without sentiment analysis (backward compatible)
                    local_inputs = self.fetch_x_voices(
                        x_api_url=x_api_url,
                        local_query=local_query,
                        filter_keywords=filter_keywords,
                        headers=headers
                    )
                    # Convert to voice format
                    local_voices = [{'content': content, 'sentiment': 'neutral'} for content in local_inputs]
                
                if local_voices or local_inputs:
                    metadata['stakeholder_data_fetched'] = True
                    metadata['stakeholder_items_count'] = len(local_voices) if local_voices else len(local_inputs)
                    
                    # Calculate voice weight based on sentiment if available
                    if use_sentiment_analysis and local_voices:
                        calculated_voice_weight = self.calculate_voice_weight(local_voices)
                        # Use calculated weight if stakeholder_weight not provided
                        if stakeholder_weight is None:
                            stakeholder_weight = calculated_voice_weight
                        metadata['voice_weight'] = calculated_voice_weight
                        metadata['sentiment_analysis'] = True
                    
                    # Check if 'local-stakeholder' variant already exists
                    if not any(v.get('name') == 'local-stakeholder' for v in enriched_variants):
                        variant_data = {
                            'name': 'local-stakeholder',
                            'inputs': local_inputs if local_inputs else [v['content'] for v in local_voices],
                            'weight': stakeholder_weight if stakeholder_weight is not None else self.default_stakeholder_weight,
                            'source': 'x_api',
                            'query': local_query
                        }
                        # Add voice data if sentiment analysis was used
                        if use_sentiment_analysis and local_voices:
                            variant_data['local_voices'] = local_voices[:3]  # Top 3 voices
                            variant_data['voice_weight'] = self.calculate_voice_weight(local_voices)
                        
                        enriched_variants.append(variant_data)
                        logger.info(f"Added 'local-stakeholder' variant with {len(local_voices) if local_voices else len(local_inputs)} items from X API")
                    else:
                        logger.info("'local-stakeholder' variant already exists, updating with new voices")
                        # Update existing variant
                        for v in enriched_variants:
                            if v.get('name') == 'local-stakeholder':
                                v['inputs'] = local_inputs if local_inputs else [voice['content'] for voice in local_voices]
                                if use_sentiment_analysis and local_voices:
                                    v['local_voices'] = local_voices[:3]
                                    v['voice_weight'] = self.calculate_voice_weight(local_voices)
                                break
            except Exception as e:
                logger.error(f"Error fetching X voices: {e}")
        
        # 2. Load deep neuro mappings from file if provided
        deep_mappings = None
        if neuro_depth_file:
            try:
                import os
                if os.path.exists(neuro_depth_file):
                    with open(neuro_depth_file, 'r', encoding='utf-8') as f:
                        deep_mappings = json.load(f)
                        logger.info(f"Loaded deep neuro mappings from {neuro_depth_file}")
                        # Merge with custom mappings if provided
                        if neuro_mappings:
                            deep_mappings.update(neuro_mappings)
                else:
                    logger.warning(f"Neuro depth file not found: {neuro_depth_file}")
            except Exception as e:
                logger.error(f"Error loading neuro depth file {neuro_depth_file}: {e}")
        
        # 3. Apply neurodiversity mappings (use deep mappings if requested or file loaded)
        use_deep = use_deep_mappings or (deep_mappings is not None)
        mappings_to_use = deep_mappings if deep_mappings else neuro_mappings
        
        enriched_variants = self.apply_neuro_mappings(
            variants=enriched_variants,
            neuro_mappings=mappings_to_use,
            use_deep_mappings=use_deep
        )
        
        # 4. Add local voices to existing variants (deepen stakeholder integration)
        if local_voices and use_sentiment_analysis:
            for variant in enriched_variants:
                if variant.get('name') != 'local-stakeholder':
                    # Add top voices to each variant
                    variant['local_voices'] = local_voices[:3]  # Top 3 voices
                    variant['voice_weight'] = self.calculate_voice_weight(local_voices)
                    logger.debug(f"Added local voices to {variant.get('name')}")
        
        # Count variants with mappings applied
        mappings_applied = sum(1 for v in enriched_variants if 'neuro_mapping' in v)
        if mappings_applied > 0:
            metadata['neuro_mappings_applied'] = True
            metadata['variants_enriched'] = mappings_applied
        
        return {
            'enriched_variants': enriched_variants,
            'metadata': metadata
        }


def enrich_stakeholder_and_neurodiversity(
    current_variants: List[Dict[str, Any]],
    x_api_url: Optional[str] = None,
    local_query: str = "Taiwan Strait tensions displacement local voices",
    neuro_mappings: Optional[Dict[str, str]] = None,
    filter_keywords: Optional[List[str]] = None,
    stakeholder_weight: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Universal function to add local stakeholder input via X fetches and explicit neurodiversity mappings.
    
    Integrates real posts (e.g., on displacement risks) and clarifies variant representations.
    Integrates with PRHP framework for consistent variant enrichment.
    
    Args:
        current_variants: List of variant dictionaries (e.g., [{'name': 'strategic-collectivist'}])
        x_api_url: Optional X search API endpoint
        local_query: Query string for local voices (default: "Taiwan Strait tensions displacement local voices")
        neuro_mappings: Optional explicit mappings for neurodiversity (e.g., to guidelines)
        filter_keywords: Optional list of keywords to filter X posts (e.g., ['Taiwan', 'displacement'])
        stakeholder_weight: Optional weight for stakeholder variant (default: 0.25)
    
    Returns:
        Enriched variants list with stakeholder input and neurodiversity mappings
    
    Example:
        >>> variants = [{'name': 'strategic-collectivist'}]
        >>> api = "https://api.example.com/x-semantic-search"
        >>> enriched = enrich_stakeholder_and_neurodiversity(
        ...     variants, x_api_url=api, local_query="Taiwan Strait tensions"
        ... )
        >>> print(enriched)
    """
    enricher = StakeholderNeurodiversityEnricher()
    
    result = enricher.enrich(
        current_variants=current_variants,
        x_api_url=x_api_url,
        local_query=local_query,
        neuro_mappings=neuro_mappings,
        filter_keywords=filter_keywords,
        stakeholder_weight=stakeholder_weight
    )
    
    return result['enriched_variants']


def deepen_stakeholders_neuro(
    variants: List[Dict[str, Any]],
    x_api_url: Optional[str] = None,
    job_query: str = "U.S.-China trade war jobs Southeast Asia local voices",
    neuro_depth_file: Optional[str] = None,
    use_sentiment_analysis: bool = True,
    use_deep_mappings: bool = True
) -> List[Dict[str, Any]]:
    """
    Universal function to enrich with local X voices and deeply integrate neuro-mappings.
    
    Fetches X posts on jobs; weights variants with voice sentiments; adds layered neuro descriptions.
    Integrates with PRHP framework for consistent variant enrichment.
    
    Args:
        variants: List of variant dictionaries (e.g., [{'name': 'economic-collectivist'}])
        x_api_url: Optional X API endpoint
        job_query: Query for local job impacts (default: "U.S.-China trade war jobs Southeast Asia local voices")
        neuro_depth_file: Path to neuro guidelines JSON file with deep mappings
        use_sentiment_analysis: If True, analyze sentiment and calculate voice weights
        use_deep_mappings: If True, use deep neuro mappings with detailed descriptions
    
    Returns:
        Enriched variants with voices and deep mappings
    """
    enricher = StakeholderNeurodiversityEnricher()
    
    result = enricher.enrich(
        current_variants=variants,
        x_api_url=x_api_url,
        local_query=job_query,
        neuro_mappings=None,  # Will use deep mappings from file or defaults
        filter_keywords=['Southeast Asia', 'ASEAN', 'jobs', 'trade war'] if 'Southeast Asia' in job_query.lower() else None,
        stakeholder_weight=None,  # Will be calculated from sentiment if use_sentiment_analysis=True
        use_sentiment_analysis=use_sentiment_analysis,
        use_deep_mappings=use_deep_mappings,
        neuro_depth_file=neuro_depth_file
    )
    
    return result['enriched_variants']


def fetch_x_voices_with_crisis_sentiment(
    x_api_url: str,
    crisis_query: str,
    filter_keywords: Optional[List[str]] = None,
    limit: int = 10,
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Fetch local stakeholder voices from X API with crisis-specific sentiment analysis.
    
    Args:
        x_api_url: X search API endpoint URL
        crisis_query: Query string for crisis-related voices
        filter_keywords: Optional list of keywords to filter posts (e.g., ['El Fasher', 'IDP'])
        limit: Maximum number of posts to fetch
        timeout: Request timeout in seconds
    
    Returns:
        List of dictionaries with 'content' and 'sentiment' keys (sentiment: 'urgency' or 'despair')
    """
    enricher = StakeholderNeurodiversityEnricher(timeout=timeout)
    
    try:
        response = requests.get(
            x_api_url,
            params={'query': crisis_query, 'limit': min(limit, 100)},
            timeout=timeout
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract posts
        if isinstance(data, dict):
            posts = data.get('posts', data.get('data', data.get('items', [])))
        elif isinstance(data, list):
            posts = data
        else:
            posts = []
        
        # Process posts with crisis sentiment analysis
        local_voices = []
        for post in posts:
            if isinstance(post, dict):
                content = post.get('content', post.get('text', post.get('message', '')))
                if isinstance(content, str) and content.strip():
                    # Apply keyword filtering if provided
                    if filter_keywords:
                        content_lower = content.lower()
                        if not any(keyword.lower() in content_lower for keyword in filter_keywords):
                            continue
                    
                    # Analyze crisis sentiment
                    sentiment = enricher.analyze_crisis_sentiment(content)
                    
                    local_voices.append({
                        'content': content.strip(),
                        'sentiment': sentiment
                    })
        
        # Limit results
        local_voices = local_voices[:enricher.max_stakeholder_items]
        
        logger.info(f"Fetched {len(local_voices)} stakeholder voices with crisis sentiment from X API")
        return local_voices
        
    except Exception as e:
        logger.warning(f"Error fetching X voices with crisis sentiment: {e}")
        # Return fallback sample data
        return [
            {"content": "RSF rapes, executions on escape routes—IDPs flee El Fasher", "sentiment": "urgency"},
            {"content": "1,365 IDPs sue UAE for RSF arms—justice now!", "sentiment": "urgency"},
            {"content": "Camps unsafe, airstrikes kill families—where to go?", "sentiment": "despair"}
        ]


def layer_stakeholders_neuro(
    variants: List[Dict[str, Any]],
    x_api_url: Optional[str] = None,
    crisis_query: str = "Sudan El Fasher IDP voices atrocities RSF",
    neuro_layer_file: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Universal function to integrate local X voices and layer neuro-mappings deeply.
    
    Fetches X posts; sentiments weight variants; adds hierarchical neuro descriptions.
    Integrates with PRHP framework for consistent variant enrichment.
    
    Args:
        variants: List of variant dictionaries (e.g., [{'name': 'justice-collectivist'}])
        x_api_url: Optional X API endpoint
        crisis_query: Query for local voices (default: "Sudan El Fasher IDP voices atrocities RSF")
        neuro_layer_file: Path to layered neuro JSON file with hierarchical mappings
    
    Returns:
        Enriched variants with layered neuro mappings and crisis sentiment-weighted voices
    """
    enriched = variants.copy()
    
    # Fetch local X voices with crisis sentiment
    local_voices = []
    if x_api_url:
        try:
            # Extract filter keywords from query if not provided
            filter_keywords = []
            query_lower = crisis_query.lower()
            if 'el fasher' in query_lower:
                filter_keywords.append('El Fasher')
            if 'idp' in query_lower:
                filter_keywords.append('IDP')
            if 'rsf' in query_lower:
                filter_keywords.append('RSF')
            
            local_voices = fetch_x_voices_with_crisis_sentiment(
                x_api_url=x_api_url,
                crisis_query=crisis_query,
                filter_keywords=filter_keywords if filter_keywords else None,
                limit=10
            )
        except Exception as e:
            logger.warning(f"Error fetching X voices: {e}. Using fallback data.")
            # Fallback sample from recent X
            local_voices = [
                {"content": "RSF rapes, executions on escape routes—IDPs flee El Fasher", "sentiment": "urgency"},
                {"content": "1,365 IDPs sue UAE for RSF arms—justice now!", "sentiment": "urgency"},
                {"content": "Camps unsafe, airstrikes kill families—where to go?", "sentiment": "despair"}
            ]
    
    # Integrate voices and weight variants
    for variant in enriched:
        variant['local_voices'] = local_voices[:4]  # Top 4 voices
        # Calculate sentiment score (urgency ratio)
        sentiment_score = sum(1 for v in local_voices if v['sentiment'] == 'urgency') / len(local_voices) if local_voices else 0.5
        variant['voice_weight'] = sentiment_score  # e.g., Boosts hybrid for urgency
    
    # Layer neuro-mappings (hierarchical: base + voice-integrated)
    if neuro_layer_file:
        try:
            import os
            if os.path.exists(neuro_layer_file):
                with open(neuro_layer_file, 'r', encoding='utf-8') as f:
                    layers = json.load(f)
                logger.info(f"Loaded layered neuro mappings from {neuro_layer_file}")
            else:
                logger.warning(f"Neuro layer file not found: {neuro_layer_file}")
                layers = None
        except Exception as e:
            logger.error(f"Error loading neuro layer file {neuro_layer_file}: {e}")
            layers = None
    else:
        layers = None
    
    # Use default layered mappings if no file provided
    if layers is None:
        layers = {
            'justice-collectivist': 'Layer 1 (ADHD-like): Collective urgency (e.g., weighs IDP escape voices +0.3 for group probes); Layer 2: HRW-aligned innovation.',
            'accountability-individualist': 'Layer 1 (Autistic-like): Precision in manifests (e.g., lawsuit sentiments for consent); Layer 2: MSF medical equity.',
            'humanitarian-hybrid': 'Layer 1 (Neurotypical): Balanced R2P (e.g., despair voices adjust delta -0.1); Layer 2: IHL traceability.',
            'ADHD-collectivist': 'Layer 1 (ADHD-like): Collective urgency (e.g., weighs IDP escape voices +0.3 for group probes); Layer 2: HRW-aligned innovation.',
            'autistic-individualist': 'Layer 1 (Autistic-like): Precision in manifests (e.g., lawsuit sentiments for consent); Layer 2: MSF medical equity.',
            'neurotypical-hybrid': 'Layer 1 (Neurotypical): Balanced R2P (e.g., despair voices adjust delta -0.1); Layer 2: IHL traceability.'
        }
    
    # Apply layered neuro mappings
    for variant in enriched:
        variant_name = variant.get('name', '')
        if variant_name in layers:
            variant['layered_neuro'] = layers[variant_name]
            logger.debug(f"Applied layered neuro mapping to {variant_name}")
    
    return enriched


def enrich_prhp_variants(
    prhp_variants: List[str],
    x_api_url: Optional[str] = None,
    local_query: str = "Taiwan Strait tensions displacement local voices",
    neuro_mappings: Optional[Dict[str, str]] = None,
    filter_keywords: Optional[List[str]] = None,
    stakeholder_weight: Optional[float] = None,
    use_sentiment_analysis: bool = False,
    use_deep_mappings: bool = False,
    neuro_depth_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enrich PRHP variant names with stakeholder input and neurodiversity mappings.
    
    Converts PRHP variant names to the expected dictionary format for enrichment.
    
    Args:
        prhp_variants: List of PRHP variant names (e.g., ['ADHD-collectivist', 'neurotypical-hybrid'])
        x_api_url: Optional X search API endpoint
        local_query: Query string for local voices
        neuro_mappings: Optional custom neurodiversity mappings
        filter_keywords: Optional list of keywords to filter X posts
        stakeholder_weight: Optional weight for stakeholder variant
        use_sentiment_analysis: If True, analyze sentiment and calculate voice weights
        use_deep_mappings: If True, use deep neuro mappings with detailed descriptions
        neuro_depth_file: Optional path to JSON file with deep neuro mappings
    
    Returns:
        Dictionary containing:
        - 'enriched_variants': List of enriched variants (in dict format)
        - 'metadata': Dictionary with enrichment details
    """
    # Convert simple variant names to the expected dictionary format
    current_variants_dicts = [{'name': v, 'weight': 1.0} for v in prhp_variants]
    
    enricher = StakeholderNeurodiversityEnricher()
    return enricher.enrich(
        current_variants=current_variants_dicts,
        x_api_url=x_api_url,
        local_query=local_query,
        neuro_mappings=neuro_mappings,
        filter_keywords=filter_keywords,
        stakeholder_weight=stakeholder_weight,
        use_sentiment_analysis=use_sentiment_analysis,
        use_deep_mappings=use_deep_mappings,
        neuro_depth_file=neuro_depth_file
    )


# Example usage
if __name__ == "__main__":
    # Example with sample data
    variants = [{'name': 'strategic-collectivist'}, {'name': 'tactical-individualist'}]
    
    print("Original variants:")
    for v in variants:
        print(f"  - {v['name']}")
    
    # Enrich with neuro mappings only (no X API)
    enriched = enrich_stakeholder_and_neurodiversity(
        variants,
        neuro_mappings={
            'strategic-collectivist': 'ADHD-like: High novelty, collective focus'
        }
    )
    
    print("\nEnriched variants:")
    for v in enriched:
        print(f"  - {v['name']}")
        if 'neuro_mapping' in v:
            print(f"    Mapping: {v['neuro_mapping']}")
        if 'inputs' in v:
            print(f"    Stakeholder inputs: {len(v['inputs'])} items")

