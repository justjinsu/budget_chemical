#!/usr/bin/env python3
"""
Main entry point for Monte Carlo analysis.

Usage: python run_monte_carlo.py [config_file]
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from budget_chemical.monte_carlo.runner import main

if __name__ == "__main__":
    main()