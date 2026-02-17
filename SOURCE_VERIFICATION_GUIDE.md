# Source Verification Guide

## Overview

The Enhanced PRHP Framework includes source verification functionality to ensure all referenced sources are verified and accurate. This improves the framework's credibility, compliance, and verifiability.

## Features

- **Automatic Source Correction**: Corrects dates, citations, details, and URLs from verified sources
- **Verification Status Tracking**: Tracks which sources are verified and which need manual review
- **Verified Source Database**: Pre-populated with verified breach and security incident sources
- **Dynamic Source Addition**: Add new verified sources at runtime

## Verified Sources

The framework includes the following pre-verified sources:

### Episource LLC Breach
- **Date**: 2025
- **Source**: HHS OCR Breach Portal, February 2025
- **Details**: Ransomware affecting ~5.4M, including behavioral health data for Medicaid patients
- **URL**: https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf
- **Verification Status**: Verified
- **Verification Date**: 2025-02

### Optum Rx Breach
- **Date**: 2023
- **Source**: KSLA News / OptumRx Announcement, December 2023
- **Details**: MOVEit vulnerability exposed personal information of millions via Clop ransomware; health data included but not specifically depression scores
- **URL**: https://www.ksla.com/2023/12/21/optumrx-patients-information-compromised-data-breach/
- **Verification Status**: Verified
- **Verification Date**: 2023-12

## Usage

### Basic Source Verification

```python
from src.prhp_enhanced import EnhancedPRHPFramework

# Create framework
framework = EnhancedPRHPFramework()

# Define sources to verify
raw_sources = [
    {
        'name': 'Episource LLC Breach',
        'date': '2025',
        'source': 'HHS, 2025',
        'details': 'Exposed mental health records of 300K patients'
    },
    {
        'name': 'Optum Rx Breach',
        'date': '2023',
        'source': 'ProPublica, 2023',  # Will be corrected to verified source
        'details': '3T records exposed, including depression risk scores'
    },
    {
        'name': 'Unknown Breach',
        'date': '2024',
        'source': 'Unknown Source'
    }
]

# Verify sources
verified_sources = framework.verify_sources(raw_sources)

# Check results
for src in verified_sources:
    print(f"\n{src['name']}:")
    print(f"  Status: {src['status']}")
    print(f"  Date: {src['date']}")
    print(f"  Source: {src['source']}")
    print(f"  Details: {src['details']}")
    if 'url' in src:
        print(f"  URL: {src['url']}")
```

**Output:**
```
Episource LLC Breach:
  Status: Verified & Tweaked
  Date: 2025
  Source: HHS OCR Breach Portal, February 2025
  Details: Ransomware affecting ~5.4M, including behavioral health data for Medicaid patients
  URL: https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf

Optum Rx Breach:
  Status: Verified & Tweaked
  Date: 2023
  Source: KSLA News / OptumRx Announcement, December 2023
  Details: MOVEit vulnerability exposed personal information of millions via Clop ransomware; health data included but not specifically depression scores
  URL: https://www.ksla.com/2023/12/21/optumrx-patients-information-compromised-data-breach/

Unknown Breach:
  Status: Unverified - Manual Review Needed
  Date: 2024
  Source: Unknown Source
```

### Standalone Function

For quick verification without creating a framework instance:

```python
from src.prhp_enhanced import source_verifier

sources = [
    {'name': 'Episource LLC Breach', 'date': '2025', 'source': 'HHS'},
    {'name': 'Optum Rx Breach', 'date': '2023', 'source': 'ProPublica'}
]

verified = source_verifier(sources)
```

### Adding New Verified Sources

```python
framework = EnhancedPRHPFramework()

# Add a new verified source
framework.add_verified_source(
    name='New Healthcare Breach',
    date='2024',
    source='Official Healthcare Authority, 2024',
    details='Data breach affecting 1M patients',
    url='https://example.com/breach',
    verification_date='2024-06'  # Optional, defaults to current date
)

# Now it will be used in verification
sources = [{'name': 'New Healthcare Breach', 'date': '2024'}]
verified = framework.verify_sources(sources)
```

### Getting All Verified Sources

```python
framework = EnhancedPRHPFramework()

# Get all verified sources
all_sources = framework.get_verified_sources()

for name, info in all_sources.items():
    print(f"{name}: {info['source']} ({info['date']})")
```

## Source Dictionary Structure

### Input Format

```python
{
    'name': str,        # Required: Source identifier
    'date': str,        # Optional: Will be corrected if verified
    'source': str,      # Optional: Will be corrected if verified
    'details': str,     # Optional: Will be corrected if verified
    'url': str          # Optional: Will be added if verified
}
```

### Output Format

```python
{
    'name': str,                    # Original name
    'date': str,                    # Corrected date (if verified)
    'source': str,                 # Corrected source (if verified)
    'details': str,                # Corrected details (if verified)
    'url': str,                    # Added URL (if verified)
    'status': str,                 # 'Verified & Tweaked' or 'Unverified - Manual Review Needed'
    'verification_status': str,    # 'verified' or 'unverified'
    'verification_date': str       # Date of verification (if verified)
}
```

## Integration with PRHP Outputs

Source verification can be integrated into PRHP outputs to ensure all referenced sources are verified:

```python
framework = EnhancedPRHPFramework()

# Run simulation
results = framework.run_simulation()

# Extract sources from results (example)
sources_from_results = [
    {'name': 'Episource LLC Breach', 'date': '2025'},
    {'name': 'Optum Rx Breach', 'date': '2023'}
]

# Verify sources before including in output
verified_sources = framework.verify_sources(sources_from_results)

# Include only verified sources in final output
for src in verified_sources:
    if src['status'] == 'Verified & Tweaked':
        # Include in output
        print(f"Source: {src['source']} ({src['date']})")
        print(f"URL: {src['url']}")
    else:
        # Flag for manual review
        logger.warning(f"Unverified source: {src['name']}")
```

## Best Practices

1. **Always Verify Sources**: Use `verify_sources()` before including sources in PRHP outputs
2. **Check Verification Status**: Only include sources with `status == 'Verified & Tweaked'` in public outputs
3. **Manual Review**: Review sources with `status == 'Unverified - Manual Review Needed'` before use
4. **Add Verified Sources**: Add new verified sources using `add_verified_source()` when verified
5. **Track Verification Dates**: Include verification dates for audit trails

## Disabling Source Verification

If needed, source verification can be disabled:

```python
framework = EnhancedPRHPFramework()
framework.source_verification_enabled = False

# Now verify_sources() will return sources unchanged
sources = framework.verify_sources(raw_sources)
```

## Status

✅ **Complete**: Source verification integrated into Enhanced PRHP Framework

---

**Last Updated**: Current
**Status**: Ready for Use

