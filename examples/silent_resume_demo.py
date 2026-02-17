#!/usr/bin/env python3
"""
Demo of silent resume functionality - exact usage pattern
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prhp_enhanced import PRHPFramework

# Initialize framework
prhp = PRHPFramework()

# Set master key
prhp.survivor_master_key = "my sister lives"

# Pause system first
prhp.pause_all_operations()
print(f"System paused: is_paused={prhp.is_paused}")

# Silent resume - no output, no logs
prhp.resume_if_authorized("my sister lives")  # → True (no output)

# Verify system resumed
print(prhp.is_paused)  # → False (resumed)

