"""
Shared pytest fixtures for PHYSAI-RL-ROBOT-X test suite.
"""
import sys
import os

# Ensure project root is on sys.path regardless of test invocation directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
