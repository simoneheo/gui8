"""
Correlation Comparison Method

This module implements correlation analysis between two data channels,
including Pearson, Spearman, and Kendall correlation coefficients.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class CorrelationComparison(BaseComparison):
    """
    Correlation analysis comparison method.
    
    Computes various correlation coefficients and related statistics
    to assess the linear and monotonic relationships between datasets.
    """
    
    name = "correlation"
    description = "Compute Pearson, Spearman, and Kendall correlation coefficients with significance tests"
    category = "Statistical"
    tags = ["scatter", "correlation","regression","Pearson","Spearman","Kendall"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "correlation_type", "type": "str", "default": "pearson", "options": ["pearson", "spearman", "kendall", "all"], "help": "Type of correlation to compute"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before calculating correlations"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "options": ["iqr", "zscore"], "help": "Method for detecting outliers"},
       
    ]
    
    # Plot configuration
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay = {
        'regression_line': {'default': True, 'label': 'Regression Line', 'help': 'Best-fit regression line', 'type': 'line'},
        'regression_equation': {'default': False, 'label': 'Regression Equation', 'help': 'Regression line formula (y = mx + b)', 'type': 'text'},
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'help': 'Correlation statistics on the plot', 'type': 'text'},
    }    
    

    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for correlation analysis
        
        This defines what gets plotted on X and Y axes for correlation visualization.
        
        Args:
            ref_data: Reference measurements (cleaned of NaN/infinite values)
            test_data: Test measurements (cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (x_data, y_data, metadata)
                x_data: Reference data for X-axis
                y_data: Test data for Y-axis
                metadata: Plot configuration dictionary
        """
        # Initialize outlier stats
        outlier_stats = {}
        
        # Handle outlier removal directly in main method
        if params.get("remove_outliers", False):
            outlier_method = params.get("outlier_method", "iqr")
            
            # Compute original correlation for impact analysis
            from scipy.stats import pearsonr
            orig_r, _ = pearsonr(ref_data, test_data)
            
            if outlier_method == "iqr":
                # IQR method on both datasets
                iqr_factor = 1.5
                
                # Reference data IQR
                ref_q25, ref_q75 = np.percentile(ref_data, [25, 75])
                ref_iqr = ref_q75 - ref_q25
                ref_lower = ref_q25 - iqr_factor * ref_iqr
                ref_upper = ref_q75 + iqr_factor * ref_iqr
                
                # Test data IQR
                test_q25, test_q75 = np.percentile(test_data, [25, 75])
                test_iqr = test_q75 - test_q25
                test_lower = test_q25 - iqr_factor * test_iqr
                test_upper = test_q75 + iqr_factor * test_iqr
                
                mask = ((ref_data >= ref_lower) & (ref_data <= ref_upper) & 
                    (test_data >= test_lower) & (test_data <= test_upper))
                
            elif outlier_method == "zscore":
                # Z-score method on both datasets
                z_threshold = 3
                
                ref_z = np.abs((ref_data - np.mean(ref_data)) / np.std(ref_data))
                test_z = np.abs((test_data - np.mean(test_data)) / np.std(test_data))
                
                mask = (ref_z <= z_threshold) & (test_z <= z_threshold)
                
            else:
                # No outlier removal
                mask = np.ones(len(ref_data), dtype=bool)
            
            # Apply outlier mask
            ref_clean = ref_data[mask]
            test_clean = test_data[mask]
            
            # Compute outlier impact statistics
            if len(ref_clean) > 2:
                clean_r, _ = pearsonr(ref_clean, test_clean)
                
                # Convert to numpy arrays for proper arithmetic
                orig_r_array = np.asarray(orig_r)
                clean_r_array = np.asarray(clean_r)
                
                outlier_stats = {
                    'original_correlation': float(orig_r_array.item()),
                    'clean_correlation': float(clean_r_array.item()),
                    'correlation_change': float((clean_r_array - orig_r_array).item()),
                    'outliers_removed': len(ref_data) - len(ref_clean),
                    'outlier_percentage': (len(ref_data) - len(ref_clean)) / len(ref_data) * 100,
                }
            else:
                outlier_stats = {'error': 'Too few points after outlier removal'}
        else:
            # No outlier removal - use original data
            ref_clean = ref_data
            test_clean = test_data
            outlier_stats = {}
        
        # Prepare metadata for plotting
        metadata = {
            'x_label': 'Reference Data',
            'y_label': 'Test Data',
            'title': 'Correlation',
            'notes': outlier_stats if outlier_stats else None
        }

        x_data = ref_clean
        y_data = test_clean

        return x_data, y_data, metadata
    
    def stats_script(self, x_data: List[float], y_data: List[float], 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for correlation analysis
        
        Args:
            x_data: Reference data (potentially transformed)
            y_data: Test data (potentially transformed)
            ref_data: Original reference data
            test_data: Original test data
            params: Method parameters dictionary
            
        Returns:
            Dictionary containing statistical results
        """
        from scipy.stats import pearsonr, spearmanr, kendalltau, linregress
        import numpy as np
        
        stats_results = {}
        
        # Correlation analysis
        correlation_type = params.get("correlation_type", "pearson")
        
        if correlation_type in ["pearson", "all"]:
            pearson_result = pearsonr(ref_data, test_data)
            pearson_r, pearson_p = pearson_result
            stats_results.update({
                'pearson_r': float(pearson_r),
                'pearson_p_value': float(pearson_p),
                'r_squared': float(pearson_r) ** 2
            })
        
        if correlation_type in ["spearman", "all"]:
            spearman_result = spearmanr(ref_data, test_data)
            spearman_r, spearman_p = spearman_result
            stats_results.update({
                'spearman_r': float(spearman_r),
                'spearman_p_value': float(spearman_p)
            })
        
        if correlation_type in ["kendall", "all"]:
            kendall_result = kendalltau(ref_data, test_data)
            kendall_tau, kendall_p = kendall_result
            stats_results.update({
                'kendall_tau': float(kendall_tau),
                'kendall_p_value': float(kendall_p)
            })
        
        # Regression analysis
        regression_result = linregress(ref_data, test_data)
        slope, intercept, r_value, p_value, std_err = regression_result
        
        # Predicted values and residuals
        predicted = slope * ref_data + intercept
        residuals = test_data - predicted
        
        # Regression statistics
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        # Convert numpy values to Python floats to avoid type issues
        stats_results.update({
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'min_ref': float(np.min(ref_data)),
            'max_ref': float(np.max(ref_data)),
        })
        
   
        # Agreement analysis
        mean_diff = np.mean(test_data - ref_data)
        std_diff = np.std(test_data - ref_data)
        
        # Limits of agreement
        confidence_level = 0.95
        z_score = 1.96  # Default for 95% confidence
                
        loa_lower = mean_diff - z_score * std_diff
        loa_upper = mean_diff + z_score * std_diff
        
        stats_results.update({
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'loa_lower': loa_lower,
            'loa_upper': loa_upper
        })
        
      
        
        return stats_results
    
    
    def _get_regression_line(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for regression line overlay."""
        try:
            slope = stats_results.get('slope')
            intercept = stats_results.get('intercept')
            
            if slope is None or intercept is None:
                print("[CorrelationOverlay] Missing regression parameters")
                return {}
            
            # Use hardcoded x range - will be auto-updated by pair_analyzer
            x_min, x_max = -1, 1
            
            # Calculate y values using regression equation: y = slope * x + intercept
            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept
            
            return {
                'x': [x_min, x_max],
                'y': [y_min, y_max],
            }
        except Exception as e:
            print(f"[CorrelationOverlay] Error getting regression line data: {e}")
            return {}
    
    def _get_regression_equation(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for regression equation overlay."""
        try:
            return {
                'text': f'y = {stats_results.get("slope"):.3f}x + {stats_results.get("intercept"):.3f}',  
            }
        except Exception as e:  
            print(f"[CorrelationOverlay] Error getting regression equation data: {e}")
            return {}
   
    def _get_statistical_results(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for statistical results text overlay."""
        
        # Only return the most informative statistics for text overlay
        essential_stats = {
            'r_value': stats_results.get('r_value'),
            'p_value': stats_results.get('p_value'),
            'r_squared': stats_results.get('r_squared'),
        }
        
        return essential_stats


    def _create_overlays(self, ref_data: np.ndarray, test_data: np.ndarray, 
                        stats_results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Create overlay definitions for correlation analysis.
        
        Returns a dictionary of overlay definitions that will be rendered by the base class.
        Each overlay definition contains type, main data, and style information.
        
        Args:
            ref_data: Reference data arrays
            test_data: Test data array
            params: Overlay configuration dictionary
            stats_results: Statistical results from stats_script method
            
        Returns:
            Dictionary of overlay definitions
        """
        
        regression_line = {
            'type': 'line',
            'show': params.get('regression_line', True),
            'label': 'Regression Line',
            'main': self._get_regression_line(stats_results)
        }

        regression_equation = {
            'type': 'text',
            'show': params.get('regression_equation', False),
            'label': 'Regression Equation',
            'main': self._get_regression_line(stats_results)
        }       

        statistical_results = {
            'type': 'text',
            'show': params.get('statistical_results', True),
            'label': 'Statistical Results',
            'main': self._get_statistical_results(stats_results)
        }

        overlays = {
            'regression_line': regression_line,
            'regression_equation': regression_equation,
            'statistical_results': statistical_results
        }
                
        return overlays


    @classmethod
    def get_description(cls) -> str:
        """
        Get a description of this comparison method for display in the wizard console.
        
        Returns:
            String description explaining what this comparison method does
        """
        return """Correlation Analysis: Measures the strength and direction of relationships between datasets.

• Pearson: Linear correlation coefficient (r) - best for normally distributed data
• Spearman: Rank-based correlation (ρ) - robust to outliers, non-parametric  
• Kendall: Rank-based correlation (τ) - most robust, good for small samples""" 
    
 