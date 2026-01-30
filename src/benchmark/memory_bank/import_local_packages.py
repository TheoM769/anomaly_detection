"""
Helper module to import futur.utils in Jupyter notebooks
"""
import sys
import os

# Get absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '/Users/theo.moreau/Documents/futur/'))

# Add to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT) 