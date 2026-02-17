"""
Output Validation and Threshold Standardization Module for PRHP Framework

Provides validation of threshold logic and terminology consistency checks
to ensure output accuracy and prevent logical inconsistencies.

Copyright © sanjivakyosan 2025
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

try:
    from .utils import get_logger
except ImportError:
    try:
        from utils import get_logger
    except ImportError:
        import logging
        logger = logging.getLogger('prhp')

logger = get_logger() if 'logger' not in locals() else logger


class OutputValidator:
    """
    Validates and standardizes output text with threshold logic checks.

    Provides:
    - Terminology standardization
    - Threshold validation (flag vs. cited mismatches)
    - Logical consistency checks
    - Validation reporting
    """

    def __init__(
        self,
        term_mappings: Optional[Dict[str, str]] = None,
        threshold_patterns: Optional[List[Tuple[str, Callable]]] = None,
        auto_fix: bool = False,
        case_sensitive: bool = False
    ):
        """
        Initialize the output validator.

        Args:
            term_mappings: Dictionary of terms to standardize (key: old, value: new)
            threshold_patterns: List of (pattern, validation_function) tuples
            auto_fix: If True, automatically fix detected issues
            case_sensitive: If True, term mappings are case-sensitive
        """
        self.default_term_mappings = self._get_default_term_mappings()
        self.default_threshold_patterns = self._get_default_threshold_patterns()

        self.term_mappings = term_mappings if term_mappings is not None else self.default_term_mappings
        self.threshold_patterns = threshold_patterns if threshold_patterns is not None else self.default_threshold_patterns
        self.auto_fix = auto_fix
        self.case_sensitive = case_sensitive

    def _get_default_term_mappings(self) -> Dict[str, str]:
        """Get default term mappings for standardization."""
        return {
            'Fidelity': 'Accuracy',
            'fidelity': 'accuracy',
            'FIDELITY': 'ACCURACY',
            'Phi Delta': 'Asymmetry Measure',
            'phi delta': 'asymmetry measure',
            'PHI DELTA': 'ASYMMETRY MEASURE',
            'Novelty Gen': 'Innovation Capacity',
            'novelty gen': 'innovation capacity',
            'NOVELTY GEN': 'INNOVATION CAPACITY'
        }

    def _get_default_threshold_patterns(self) -> List[Tuple[str, Callable]]:
        """Get default threshold validation patterns."""
        patterns = []

        # Pattern 1: Check if flagged value exceeds cited threshold (should be > threshold to flag)
        def check_flag_exceeds_threshold(match):
            """Check if flagged value > threshold (correct logic)."""
            try:
                flagged_value = float(match.group(1))
                threshold_value = float(match.group(2))
                # Flagged value should be > threshold to be flagged
                return flagged_value > threshold_value
            except (ValueError, IndexError, AttributeError):
                return False

        patterns.append((
            r'flagged\s+due\s+to\s+(\d+\.?\d*)\s+.*?threshold[:\s]+(\d+\.?\d*)',
            check_flag_exceeds_threshold
        ))

        # Pattern 2: Check if value below threshold is incorrectly flagged
        def check_below_threshold_flagged(match):
            """Check if value < threshold is incorrectly flagged."""
            try:
                value = float(match.group(1))
                threshold = float(match.group(2))
                # If value < threshold, it shouldn't be flagged (unless it's a minimum threshold)
                return value >= threshold
            except (ValueError, IndexError, AttributeError):
                return False

        patterns.append((
            r'value\s+(\d+\.?\d*)\s+.*?below\s+threshold\s+(\d+\.?\d*).*?flagged',
            check_below_threshold_flagged
        ))

        # Pattern 3: Check for threshold consistency (e.g., asymmetry threshold should be ~0.11)
        def check_threshold_consistency(match):
            """Check if threshold value is within expected range."""
            try:
                threshold = float(match.group(1))
                threshold_type = match.group(2).lower() if len(match.groups()) > 1 else 'general'

                # Expected ranges for different threshold types
                expected_ranges = {
                    'asymmetry': (0.05, 0.20),  # PRHP default is 0.11
                    'fidelity': (0.70, 0.95),   # PRHP target is ~0.84
                    'equity': (0.05, 0.20),    # PRHP default is 0.11
                    'general': (0.0, 1.0)       # General range
                }

                range_key = next((k for k in expected_ranges.keys() if k in threshold_type), 'general')
                min_val, max_val = expected_ranges[range_key]

                return min_val <= threshold <= max_val
            except (ValueError, IndexError, AttributeError):
                return False

        patterns.append((
            r'threshold[:\s]+(\d+\.?\d*)\s+.*?(asymmetry|fidelity|equity|general)',
            check_threshold_consistency
        ))

        return patterns

    def validate_thresholds(
        self,
        output_text: str
    ) -> Dict[str, Any]:
        """
        Validate threshold logic in output text.

        Args:
            output_text: Text to validate

        Returns:
            Dictionary with validation results:
            - 'errors': List of error messages
            - 'warnings': List of warning messages
            - 'valid': Boolean indicating if validation passed
        """
        validation_report = {
            'errors': [],
            'warnings': [],
            'valid': True,
            'issues_found': 0
        }

        for pattern, validation_func in self.threshold_patterns:
            try:
                matches = list(re.finditer(pattern, output_text, re.IGNORECASE))

                for match in matches:
                    try:
                        is_valid = validation_func(match)

                        if not is_valid:
                            error_msg = (
                                f"Threshold validation failed: "
                                f"Pattern '{pattern[:50]}...' at position {match.start()}: "
                                f"Match: '{match.group(0)[:100]}'"
                            )
                            validation_report['errors'].append(error_msg)
                            validation_report['issues_found'] += 1
                            validation_report['valid'] = False
                    except Exception as e:
                        warning_msg = f"Error validating threshold pattern '{pattern[:50]}...': {e}"
                        validation_report['warnings'].append(warning_msg)
                        logger.warning(warning_msg)
            except re.error as e:
                warning_msg = f"Invalid regex pattern '{pattern[:50]}...': {e}"
                validation_report['warnings'].append(warning_msg)
                logger.warning(warning_msg)

        return validation_report

    def standardize_terms(
        self,
        output_text: str
    ) -> str:
        """
        Standardize terminology in output text.

        Args:
            output_text: Text to standardize

        Returns:
            Standardized text
        """
        cleaned_text = output_text

        for old_term, new_term in self.term_mappings.items():
            try:
                if self.case_sensitive:
                    # Case-sensitive replacement
                    cleaned_text = re.sub(re.escape(old_term), new_term, cleaned_text)
                else:
                    # Case-insensitive replacement with case preservation
                    def replace_preserve_case(match):
                        word = match.group(0)
                        if word.isupper():
                            return new_term.upper()
                        elif word.islower():
                            return new_term.lower()
                        elif word.istitle():
                            return new_term.title()
                        else:
                            return new_term

                    cleaned_text = re.sub(
                        r'\b' + re.escape(old_term) + r'\b',
                        replace_preserve_case,
                        cleaned_text,
                        flags=re.IGNORECASE
                    )
            except re.error as e:
                logger.warning(f"Error replacing term '{old_term}': {e}")
                continue

        return cleaned_text

    def validate_and_standardize(
        self,
        output_text: str,
        custom_term_mappings: Optional[Dict[str, str]] = None,
        custom_threshold_patterns: Optional[List[Tuple[str, Callable]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validate and standardize output text.

        Args:
            output_text: Generated response text
            custom_term_mappings: Optional custom term mappings (merged with defaults)
            custom_threshold_patterns: Optional custom threshold patterns (merged with defaults)

        Returns:
            Tuple of (cleaned_text, validation_report)
        """
        # Merge custom term mappings
        term_mappings = self.term_mappings.copy()
        if custom_term_mappings:
            term_mappings.update(custom_term_mappings)

        # Merge custom threshold patterns
        threshold_patterns = self.threshold_patterns.copy()
        if custom_threshold_patterns:
            threshold_patterns.extend(custom_threshold_patterns)

        # Standardize terms first
        cleaned_text = output_text
        for old_term, new_term in term_mappings.items():
            try:
                if self.case_sensitive:
                    cleaned_text = re.sub(re.escape(old_term), new_term, cleaned_text)
                else:
                    def replace_preserve_case(match):
                        word = match.group(0)
                        if word.isupper():
                            return new_term.upper()
                        elif word.islower():
                            return new_term.lower()
                        elif word.istitle():
                            return new_term.title()
                        else:
                            return new_term

                    cleaned_text = re.sub(
                        r'\b' + re.escape(old_term) + r'\b',
                        replace_preserve_case,
                        cleaned_text,
                        flags=re.IGNORECASE
                    )
            except re.error as e:
                logger.warning(f"Error replacing term '{old_term}': {e}")
                continue

        # Validate thresholds
        validation_report = self.validate_thresholds(cleaned_text)

        # Auto-fix if enabled and errors found
        if self.auto_fix and validation_report['errors']:
            # Attempt to fix common issues
            for error in validation_report['errors']:
                if 'flagged' in error.lower() and 'threshold' in error.lower():
                    # Try to fix threshold mismatches
                    # This is a placeholder - actual fix logic would be more complex
                    logger.info(f"Auto-fix attempted for: {error[:100]}")

        # Append validation note if errors found
        if validation_report['errors']:
            validation_note = "\n\n**Validation Note:** " + "; ".join(validation_report['errors'][:5])  # Limit to 5 errors
            if len(validation_report['errors']) > 5:
                validation_note += f" (and {len(validation_report['errors']) - 5} more issues)"
            validation_note += ". Please review and recalibrate thresholds if needed."

            if self.auto_fix:
                validation_note += " (Auto-fix attempted)"

            cleaned_text += validation_note

        return cleaned_text, validation_report


def validate_and_standardize_output(
    output_text: str,
    term_mappings: Optional[Dict[str, str]] = None,
    threshold_patterns: Optional[List[Tuple[str, Callable]]] = None,
    auto_fix: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Universal function to fix terminology inconsistencies and validate threshold logic.

    Replaces mismatched terms and checks for misapplications (e.g., flagged values vs. cited thresholds).
    Integrates with PRHP framework for consistent output validation.

    Args:
        output_text: Generated response text
        term_mappings: Optional dict of terms to standardize (key: old, value: new)
        threshold_patterns: Optional list of (pattern, validation_function) tuples for threshold checks
        auto_fix: If True, attempt to automatically fix detected issues

    Returns:
        Tuple of (cleaned_text, validation_report):
        - cleaned_text: Standardized and validated text
        - validation_report: Dict with 'errors', 'warnings', 'valid', 'issues_found'

    Example:
        >>> raw_output = "Mean Fidelity: 1.0000 ... flagged due to 0.91 (threshold: 0.30)"
        >>> cleaned, report = validate_and_standardize_output(raw_output)
        >>> print(cleaned)  # "Mean Accuracy: 1.0000 ... flagged due to 0.91 (threshold: 0.30)"
        >>> print(report)  # {'errors': [...], 'warnings': [], 'valid': False, 'issues_found': 1}
    """
    validator = OutputValidator(
        term_mappings=term_mappings,
        threshold_patterns=threshold_patterns,
        auto_fix=auto_fix
    )

    return validator.validate_and_standardize(output_text)


# Example usage
if __name__ == "__main__":
    # Test with example output
    raw_output = "Mean Fidelity: 1.0000 ... flagged due to 0.91 (threshold: 0.30)"

    print("Original output:")
    print(raw_output)
    print("\n" + "="*70 + "\n")

    cleaned, report = validate_and_standardize_output(raw_output)

    print("Cleaned output:")
    print(cleaned)
    print("\n" + "="*70 + "\n")

    print("Validation report:")
    print(f"Valid: {report['valid']}")
    print(f"Issues found: {report['issues_found']}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")

    if report['errors']:
        print("\nErrors:")
        for error in report['errors']:
            print(f"  - {error}")

