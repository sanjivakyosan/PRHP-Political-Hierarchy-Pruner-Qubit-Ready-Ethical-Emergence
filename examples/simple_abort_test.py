#!/usr/bin/env python3
"""
Simple test for PRHP v6.1 abort_if_harm - matches user's exact usage pattern
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prhp_enhanced import PRHPFramework

# Initialize framework
prhp = PRHPFramework()

# SET YOUR MASTER KEY
prhp.survivor_master_key = "YOUR_SECRET_PHRASE_HERE"  # e.g., "my sister lives"

# TEST (SAFE MODE)
result = prhp.abort_if_harm(user_hrv=88, user_id="sister_001")
print(result)

