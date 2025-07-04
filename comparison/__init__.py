"""
Comparison Methods Module

This module contains various statistical and visual comparison methods for analyzing
relationships between data channels. It follows the same pattern as the steps and
mixers modules for consistency and extensibility.

Available comparison methods:
- Correlation analysis (Pearson, Spearman, Kendall)
- Bland-Altman analysis
- Residual analysis
- Time series comparison
- Statistical tests
- Custom comparison metrics
"""

from .base_comparison import BaseComparison
from .comparison_registry import ComparisonRegistry

# Import all comparison methods
from .correlation_comparison import CorrelationComparison
from .bland_altman_comparison import BlandAltmanComparison
from .residual_comparison import ResidualComparison
from .statistical_comparison import StatisticalComparison

def load_all_comparisons(directory=None):
    """
    Load all comparison methods from the comparison directory.
    
    Args:
        directory (str, optional): Directory to load from. Defaults to current directory.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize the comparison registry
        ComparisonRegistry.initialize()
        print("[Comparison] Successfully loaded all comparison methods")
        return True
    except Exception as e:
        print(f"[Comparison] Error loading comparison methods: {e}")
        return False

__all__ = [
    'BaseComparison',
    'ComparisonRegistry',
    'CorrelationComparison',
    'BlandAltmanComparison', 
    'ResidualComparison',
    'StatisticalComparison',
    'load_all_comparisons'
]

# Version info
__version__ = '1.0.0'
__author__ = 'GUI8 Development Team' 