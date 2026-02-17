"""
Output Standardization Module for PRHP Framework

Provides text standardization, term mapping, and typo correction
for PRHP simulation outputs and AI responses.

Copyright © sanjivakyosan 2025
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

try:
    from .utils import get_logger
except ImportError:
    from utils import get_logger

logger = get_logger()


class OutputStandardizer:
    """
    Standardizes and cleans PRHP framework output text.
    
    Provides:
    - Term mapping (e.g., 'fidelity' -> 'accuracy' for public output)
    - Typo correction
    - Format consistency
    - PRHP-specific term handling
    """
    
    def __init__(
        self,
        term_mappings: Optional[Dict[str, str]] = None,
        typo_corrections: Optional[List[Tuple[str, str]]] = None,
        preserve_prhp_terms: bool = False,
        case_sensitive: bool = False
    ):
        """
        Initialize the output standardizer.
        
        Args:
            term_mappings: Dictionary of term replacements (old -> new)
            typo_corrections: List of (pattern, replacement) tuples for regex fixes
            preserve_prhp_terms: If True, preserves PRHP-specific terms (for internal use)
            case_sensitive: If True, replacements are case-sensitive
        """
        self.preserve_prhp_terms = preserve_prhp_terms
        self.case_sensitive = case_sensitive
        
        # Default term mappings (can be overridden)
        if term_mappings is None:
            self.term_mappings = self._get_default_term_mappings()
        else:
            self.term_mappings = term_mappings
        
        # Default typo corrections (can be overridden)
        if typo_corrections is None:
            self.typo_corrections = self._get_default_typo_corrections()
        else:
            self.typo_corrections = typo_corrections
    
    def _get_default_term_mappings(self) -> Dict[str, str]:
        """Get default term mappings for PRHP framework."""
        mappings = {}
        
        # Only apply term mappings if not preserving PRHP terms (for public output)
        if not self.preserve_prhp_terms:
            # Standardize PRHP terms for public output
            mappings.update({
                'fidelity': 'accuracy',
                'Fidelity': 'Accuracy',
                'mean fidelity': 'mean accuracy',
                'Mean Fidelity': 'Mean Accuracy',
                'phi delta': 'asymmetry measure',
                'Phi Delta': 'Asymmetry Measure',
                'mean phi delta': 'mean asymmetry measure',
                'Mean Phi Delta': 'Mean Asymmetry Measure',
                'asymmetry delta': 'equity deviation',
                'Asymmetry Delta': 'Equity Deviation',
                'novelty generation': 'innovation capacity',
                'Novelty Generation': 'Innovation Capacity',
                'novelty_gen': 'innovation_capacity',
                'neuro-cultural': 'cognitive-cultural',
                'Neuro-cultural': 'Cognitive-cultural',
                'ADHD-collectivist': 'collective-oriented',
                'autistic-individualist': 'individual-oriented',
                'neurotypical-hybrid': 'balanced-adaptive'
            })
        
        return mappings
    
    def _get_default_typo_corrections(self) -> List[Tuple[str, str]]:
        """Get default typo correction patterns."""
        return [
            # Common typos - apply BEFORE term mappings to catch base terms
            (r'generationeration', 'generation'),  # Catch doubled "generation"
            (r'novelty generationeration', 'novelty generation'),
            (r'innovation capacityeration', 'innovation capacity'),  # After term mapping
            (r'capacityeration', 'capacity'),  # Generic pattern
            (r'measurement delta', 'phi delta'),
            (r'fideltiy', 'fidelity'),
            (r'fidleity', 'fidelity'),
            (r'asymetry', 'asymmetry'),
            (r'asymmtery', 'asymmetry'),
            (r'neurotypical', 'neurotypical'),  # Correct spelling
            (r'neurodivergent', 'neurodivergent'),  # Correct spelling
            (r'quantuum', 'quantum'),
            (r'quantm', 'quantum'),
            (r'simualtion', 'simulation'),
            (r'simulaton', 'simulation'),
            # PRHP-specific typos
            (r'prhp', 'PRHP'),
            (r'Prhp', 'PRHP'),
            (r'prHP', 'PRHP'),
            # Metric typos
            (r'mean\s+fideltiy', 'mean fidelity'),
            (r'std\s+dev', 'standard deviation'),
            (r'std\s+deviation', 'standard deviation'),
            # Formatting fixes (apply last)
            (r'\.\s*\.\s*\.', '...'),  # Normalize ellipsis
        ]
    
    def standardize(
        self,
        output_text: str,
        custom_term_mappings: Optional[Dict[str, str]] = None,
        custom_typo_corrections: Optional[List[Tuple[str, str]]] = None
    ) -> str:
        """
        Standardize output text with term mappings and typo corrections.
        
        Args:
            output_text: The text to standardize
            custom_term_mappings: Optional additional term mappings (merged with defaults)
            custom_typo_corrections: Optional additional typo corrections (appended to defaults)
        
        Returns:
            Standardized text
        """
        if not output_text or not isinstance(output_text, str):
            logger.warning("Invalid output_text provided to standardize()")
            return output_text if isinstance(output_text, str) else ""
        
        try:
            cleaned_text = output_text
            
            # Merge custom term mappings with defaults
            term_mappings = self.term_mappings.copy()
            if custom_term_mappings:
                term_mappings.update(custom_term_mappings)
            
            # Apply term mappings with proper case handling
            for old_term, new_term in term_mappings.items():
                if self.case_sensitive:
                    # Case-sensitive replacement
                    cleaned_text = re.sub(re.escape(old_term), new_term, cleaned_text)
                else:
                    # Case-insensitive replacement with case preservation
                    def replace_preserve_case(match):
                        matched = match.group(0)
                        # Preserve original case pattern
                        if matched.isupper():
                            return new_term.upper()
                        elif matched.islower():
                            return new_term.lower()
                        elif matched.istitle():
                            return new_term.title()
                        else:
                            return new_term
                    
                    cleaned_text = re.sub(
                        re.escape(old_term),
                        replace_preserve_case,
                        cleaned_text,
                        flags=re.IGNORECASE
                    )
            
            # Merge custom typo corrections with defaults
            typo_corrections = self.typo_corrections.copy()
            if custom_typo_corrections:
                typo_corrections.extend(custom_typo_corrections)
            
            # Apply typo corrections (regex patterns) - multiple passes for comprehensive fixes
            # First pass: fix base term typos
            for pattern, replacement in typo_corrections:
                try:
                    cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                    continue
            
            # Additional formatting fixes
            cleaned_text = self._fix_formatting(cleaned_text)
            
            # Second pass: fix any typos that may have been introduced by term mappings
            # (e.g., "innovation capacityeration" after mapping "novelty generationeration" -> "innovation capacity")
            for pattern, replacement in typo_corrections:
                try:
                    cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
                except re.error:
                    continue
            
            # Validate table headers if tables are present
            if 'table' in cleaned_text.lower() or '|' in cleaned_text:
                cleaned_text = self._fix_table_formatting(cleaned_text)
            
            logger.debug(f"Standardized text: {len(output_text)} -> {len(cleaned_text)} chars")
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error standardizing output text: {e}")
            return output_text  # Return original on error
    
    def _fix_formatting(self, text: str) -> str:
        """Fix common formatting issues."""
        # Remove excessive whitespace but preserve structure
        # Replace multiple spaces with single space (but preserve newlines)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix spacing around brackets
        text = re.sub(r'\s+([\(\[\{])', r'\1', text)
        text = re.sub(r'([\)\]\}])\s+', r'\1 ', text)
        
        # Normalize line breaks (preserve intentional breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _fix_table_formatting(self, text: str) -> str:
        """Fix table formatting issues."""
        lines = text.split('\n')
        fixed_lines = []
        in_table = False
        
        for line in lines:
            # Detect table lines (contain | or multiple spaces for alignment)
            if '|' in line or (line.count('  ') >= 2 and not line.strip().startswith('#')):
                in_table = True
                # Normalize table separators
                line = re.sub(r'\s*\|\s*', ' | ', line)
                # Ensure consistent spacing
                line = re.sub(r'\s+', ' ', line)
                fixed_lines.append(line)
            else:
                if in_table:
                    in_table = False
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def standardize_prhp_output(
        self,
        output_text: str,
        mode: str = 'public'
    ) -> str:
        """
        Standardize PRHP-specific output text.
        
        Args:
            output_text: Text to standardize
            mode: 'public' (sanitize PRHP terms) or 'internal' (preserve PRHP terms)
        
        Returns:
            Standardized text
        """
        if mode == 'public':
            # Use standard mappings (sanitize PRHP terms)
            return self.standardize(output_text)
        else:
            # Internal mode: preserve PRHP terms, only fix typos and formatting
            # Create a temporary standardizer with no term mappings
            temp_standardizer = OutputStandardizer(
                term_mappings={},  # No term mappings for internal mode
                typo_corrections=self.typo_corrections,  # Keep typo corrections
                preserve_prhp_terms=True,
                case_sensitive=self.case_sensitive
            )
            return temp_standardizer.standardize(output_text)


def standardize_output_text(
    output_text: str,
    term_mappings: Optional[Dict[str, str]] = None,
    typo_corrections: Optional[List[Tuple[str, str]]] = None,
    preserve_prhp_terms: bool = False,
    mode: str = 'public'
) -> str:
    """
    Universal function to fix inconsistencies in simulation outputs.
    
    Standardizes terms (e.g., replace 'fidelity' with 'accuracy') and corrects typos.
    Integrates with PRHP framework for consistent output formatting.
    
    Args:
        output_text: The generated response text
        term_mappings: Optional dict of terms to replace (key: old, value: new)
        typo_corrections: Optional list of (pattern, replacement) tuples for regex fixes
        preserve_prhp_terms: If True, preserves PRHP-specific terms (for internal use)
        mode: 'public' (sanitize) or 'internal' (preserve PRHP terms)
    
    Returns:
        Cleaned and standardized text
    """
    standardizer = OutputStandardizer(
        term_mappings=term_mappings,
        typo_corrections=typo_corrections,
        preserve_prhp_terms=preserve_prhp_terms
    )
    
    return standardizer.standardize_prhp_output(output_text, mode=mode)


# Example usage
if __name__ == "__main__":
    # Test with sample PRHP output
    raw_output = """
    Mean Fidelity: 0.9000 ± 0.025
    Asymmetry Delta: 0.12
    Novelty Generationeration: 0.801
    Mean Phi Delta: 0.05
    """
    
    print("Original output:")
    print(raw_output)
    print("\n" + "="*60 + "\n")
    
    # Public mode (sanitize PRHP terms)
    cleaned_public = standardize_output_text(raw_output, mode='public')
    print("Standardized (public mode):")
    print(cleaned_public)
    print("\n" + "="*60 + "\n")
    
    # Internal mode (preserve PRHP terms)
    cleaned_internal = standardize_output_text(raw_output, mode='internal')
    print("Standardized (internal mode):")
    print(cleaned_internal)

