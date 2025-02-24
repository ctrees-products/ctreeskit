"""
Test suite for XR Analyzer package.

This package contains all tests for the xr_analyzer package.
Tests are organized by module and can be run using pytest.
"""

import os
import sys

# Add the src directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src')))
