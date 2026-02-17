#!/usr/bin/env python3
"""
Test script for output standardization module.

Tests the standardization of PRHP output text with term mappings and typo corrections.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from output_standardization import standardize_output_text, OutputStandardizer

def test_basic_standardization():
    """Test basic term mapping and typo correction."""
    print("="*60)
    print("Test 1: Basic Standardization")
    print("="*60)
    
    raw_output = """
    Mean Fidelity: 0.9000 ± 0.025
    Asymmetry Delta: 0.12
    Novelty Generationeration: 0.801
    Mean Phi Delta: 0.05
    """
    
    print("\nOriginal output:")
    print(raw_output)
    
    cleaned = standardize_output_text(raw_output, mode='public')
    
    print("\nStandardized (public mode):")
    print(cleaned)
    
    # Verify transformations
    assert 'Accuracy' in cleaned or 'accuracy' in cleaned, "Fidelity should be mapped to Accuracy"
    assert 'generationeration' not in cleaned, "Typo 'generationeration' should be fixed"
    assert 'innovation capacity' in cleaned.lower() or 'novelty generation' in cleaned.lower(), "Novelty generation should be handled"
    
    print("\n✓ Test 1 passed!")


def test_internal_mode():
    """Test internal mode (preserves PRHP terms)."""
    print("\n" + "="*60)
    print("Test 2: Internal Mode (Preserve PRHP Terms)")
    print("="*60)
    
    raw_output = """
    Mean Fidelity: 0.9000
    Phi Delta: 0.05
    ADHD-collectivist variant shows high fidelity
    """
    
    print("\nOriginal output:")
    print(raw_output)
    
    cleaned = standardize_output_text(raw_output, mode='internal')
    
    print("\nStandardized (internal mode):")
    print(cleaned)
    
    # Verify PRHP terms are preserved
    assert 'fidelity' in cleaned.lower(), "Fidelity should be preserved in internal mode"
    assert 'phi delta' in cleaned.lower(), "Phi delta should be preserved in internal mode"
    
    print("\n✓ Test 2 passed!")


def test_custom_mappings():
    """Test custom term mappings."""
    print("\n" + "="*60)
    print("Test 3: Custom Term Mappings")
    print("="*60)
    
    raw_output = "Mean Fidelity: 0.9000"
    
    print("\nOriginal output:")
    print(raw_output)
    
    custom_mappings = {
        'fidelity': 'precision',
        'Fidelity': 'Precision'
    }
    
    cleaned = standardize_output_text(
        raw_output,
        term_mappings=custom_mappings,
        mode='public'
    )
    
    print("\nStandardized with custom mappings:")
    print(cleaned)
    
    # Verify custom mapping
    assert 'Precision' in cleaned or 'precision' in cleaned, "Custom mapping should be applied"
    
    print("\n✓ Test 3 passed!")


def test_typo_corrections():
    """Test typo correction patterns."""
    print("\n" + "="*60)
    print("Test 4: Typo Corrections")
    print("="*60)
    
    test_cases = [
        ("fideltiy", "fidelity"),
        ("asymetry", "asymmetry"),
        ("quantuum", "quantum"),
        ("simualtion", "simulation"),
    ]
    
    for typo, correct in test_cases:
        raw = f"Test with {typo} in text"
        cleaned = standardize_output_text(raw, mode='internal')
        print(f"\nOriginal: {raw}")
        print(f"Cleaned:  {cleaned}")
        assert typo not in cleaned.lower(), f"Typo '{typo}' should be corrected"
        assert correct in cleaned.lower(), f"Correct spelling '{correct}' should be present"
    
    print("\n✓ Test 4 passed!")


def test_table_formatting():
    """Test table formatting fixes."""
    print("\n" + "="*60)
    print("Test 5: Table Formatting")
    print("="*60)
    
    raw_output = """
    | Metric | Value |
    |--------|-------|
    | Fidelity | 0.90 |
    | Phi Delta | 0.05 |
    """
    
    print("\nOriginal output:")
    print(raw_output)
    
    cleaned = standardize_output_text(raw_output, mode='public')
    
    print("\nStandardized:")
    print(cleaned)
    
    # Verify table structure is preserved
    assert '|' in cleaned, "Table separators should be preserved"
    
    print("\n✓ Test 5 passed!")


def test_formatting_fixes():
    """Test general formatting fixes."""
    print("\n" + "="*60)
    print("Test 6: Formatting Fixes")
    print("="*60)
    
    raw_output = "Test   with   multiple   spaces  .And  punctuation issues  ."
    
    print("\nOriginal output:")
    print(repr(raw_output))
    
    cleaned = standardize_output_text(raw_output, mode='public')
    
    print("\nStandardized:")
    print(repr(cleaned))
    
    # Verify formatting improvements
    assert '  ' not in cleaned, "Multiple spaces should be normalized"
    
    print("\n✓ Test 6 passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Output Standardization Test Suite")
    print("="*60)
    
    try:
        test_basic_standardization()
        test_internal_mode()
        test_custom_mappings()
        test_typo_corrections()
        test_table_formatting()
        test_formatting_fixes()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

