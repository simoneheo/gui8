"""
Comparison Methods Module

This module contains streamlined statistical and visual comparison methods for analyzing
relationships between data channels. The methods have been optimized for the most
common use cases while maintaining comprehensive functionality.

Available comparison methods:
- Correlation analysis (Pearson, Spearman) with integrated RMSE
- Bland-Altman analysis for method comparison
- Residual analysis with multiple fitting methods
- Error distribution histogram analysis
- Time lag cross-correlation analysis
"""

import os
import importlib
from .base_comparison import BaseComparison
from .comparison_registry import ComparisonRegistry

# Import streamlined comparison methods
from .bland_altman_comparison import BlandAltmanComparison

# Initialize the registry on import to avoid double loading
try:
    if not ComparisonRegistry._initialized:
        ComparisonRegistry.initialize()
        print("[Comparison] Successfully initialized comparison registry on import")
except Exception as e:
    print(f"[Comparison] Error initializing comparison registry: {e}")

def load_all_comparisons(directory=None):
    """
    Load all comparison methods from the comparison directory.
    
    Args:
        directory (str, optional): Directory to load from. Defaults to current directory.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Registry should already be initialized from import
        if not ComparisonRegistry._initialized:
            ComparisonRegistry.initialize()
        
        # Auto-load all comparison methods from the comparison folder
        comparison_folder = os.path.dirname(__file__)
        for filename in os.listdir(comparison_folder):
            if filename.endswith(".py") and filename not in ("__init__.py", "base_comparison.py", "comparison_registry.py"):
                module_name = filename[:-3]
                try:
                    # Import the module to trigger the @register_comparison decorator
                    importlib.import_module(f".{module_name}", package="comparison")
                    print(f"[Comparison] Successfully loaded {module_name}")
                except Exception as e:
                    print(f"[Comparison] Error importing {module_name}: {e}")
        
        # Show loaded methods
        methods = ComparisonRegistry.all_comparisons()
        print(f"[Comparison] Loaded {len(methods)} comparison methods: {methods}")
        return True
    except Exception as e:
        print(f"[Comparison] Error loading comparison methods: {e}")
        return False

__all__ = [
    'BaseComparison',
    'ComparisonRegistry',
    'BlandAltmanComparison', 
    'load_all_comparisons'
]

# Version info
__version__ = '2.0.0'
__author__ = 'GUI8 Development Team' 