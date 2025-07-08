"""
Comparison Methods Module

This module contains streamlined statistical and visual comparison methods for analyzing
relationships between data channels. The methods have been optimized for the most
common use cases while maintaining comprehensive functionality.

Available comparison methods:
- Correlation analysis (Pearson, Spearman) with integrated RMSE
- Bland-Altman analysis for method comparison
- Residual analysis with multiple fitting methods
- Statistical tests (t-test, Wilcoxon, KS test)
- Cross-correlation for time series analysis
"""

from .base_comparison import BaseComparison
from .comparison_registry import ComparisonRegistry

# Import streamlined comparison methods
from .correlation_comparison import CorrelationComparison
from .bland_altman_comparison import BlandAltmanComparison
from .residual_comparison import ResidualComparison
from .statistical_comparison import StatisticalComparison
from .cross_correlation_comparison import CrossCorrelationComparison

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
        print("[Comparison] Comparison methods are ready")
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
    'CrossCorrelationComparison',
    'load_all_comparisons'
]

# Version info
__version__ = '2.0.0'
__author__ = 'GUI8 Development Team' 