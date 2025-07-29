"""
Bland-Altman Comparison Method

This module implements Bland-Altman analysis for assessing agreement between two methods.
The Bland-Altman plot shows the difference between two measurements against their mean.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class BlandAltmanComparison(BaseComparison):
    """
    Bland-Altman analysis comparison method.
    
    Implements the Bland-Altman method for assessing agreement between two methods
    by plotting the difference between measurements against their mean.
    """
    
    name = "bland_altman"
    description = "Bland-Altman analysis for assessing agreement between two methods"
    category = "Statistical"
    tags = ["scatter","bland_altman","agreement"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "percentage_difference", "type": "bool", "default": False, "help": "Show differences as percentages instead of absolute values"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before calculating agreement statistics"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "options": ["iqr", "zscore"], "help": "Method for detecting outliers: IQR (robust) or Z-score (assumes normal distribution)"},
    ]
    
    # Plot configuration
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'bias_line': {'default': True, 'label': 'Bias Line', 'tooltip': 'Show horizontal line at mean difference (bias)', 'type': 'line'},
        'upper_loa_line': {'default': True, 'label': 'Upper Limit of Agreement', 'tooltip': 'Show upper limit of agreement line', 'type': 'line'},
        'lower_loa_line': {'default': True, 'label': 'Lower Limit of Agreement', 'tooltip': 'Show lower limit of agreement line', 'type': 'line'},
        'bias_confidence_interval': {'default': True, 'label': 'Bias Confidence Interval', 'tooltip': 'Show confidence interval around bias', 'type': 'fill'},
        'loa_lower_confidence_interval': {'default': True, 'label': 'Lower LoA Confidence Interval', 'tooltip': 'Show confidence interval around lower limit of agreement', 'type': 'fill'},
        'loa_upper_confidence_interval': {'default': True, 'label': 'Upper LoA Confidence Interval', 'tooltip': 'Show confidence interval around upper limit of agreement', 'type': 'fill'},
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'tooltip': 'Display statistical results on the plot', 'type': 'text'}
    }
   
    
    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation with outlier removal
        
        This defines what gets plotted on X and Y axes for Bland-Altman analysis.
        Includes outlier removal if requested in parameters.
        
        Modify this method to create custom plot types and transformations.
        
        Examples of modifications:
        - Percentage differences: differences = (differences / means) * 100
        - Log transforms: means = np.log(means), differences = np.log(test_data) - np.log(ref_data)
        - Ratio plots: ratios = test_data / ref_data instead of differences
        - Custom domain transformations for specific measurement types
        - Different outlier detection methods (IQR, Z-score, custom thresholds)
        
        Args:
            ref_data: Reference measurements (already cleaned of NaN/infinite values)
            test_data: Test measurements (already cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (x_data, y_data, metadata)
                x_data: Values for X-axis (typically means)
                y_data: Values for Y-axis (typically differences)
                metadata: Plot configuration dictionary
        """
        
        # === CORE BLAND-ALTMAN TRANSFORMATION ===
        means = (ref_data + test_data) / 2
        differences = test_data - ref_data
        
        # Apply percentage transformation if requested
        if params.get('percentage_difference', False):
            # Avoid division by zero and very small denominators
            # Use a more robust approach for percentage differences
            nonzero_mask = np.abs(means) > 1e-10  # Use absolute value and small threshold
            differences_pct = np.full_like(differences, np.nan)
            differences_pct[nonzero_mask] = (differences[nonzero_mask] / means[nonzero_mask]) * 100
            differences = differences_pct
        
        # === OUTLIER REMOVAL AND ANALYSIS ===
        # Remove NaN values first
        valid_mask = np.isfinite(differences) & np.isfinite(means)
        if np.sum(valid_mask) < len(differences):
            differences = differences[valid_mask]
            means = means[valid_mask]
            ref_data = ref_data[valid_mask]
            test_data = test_data[valid_mask]
        
        # Initialize outlier statistics
        outlier_stats = None
        
        # Check if outlier removal is requested
        if params.get('remove_outliers', False) and len(differences) > 0:
            method = params.get('outlier_method', 'iqr')
            
            if method == 'iqr':
                # IQR method on differences
                q1, q3 = np.percentile(differences, [25, 75])
                iqr = q3 - q1
                iqr_factor = 1.5
                lower_bound = q1 - iqr_factor * iqr
                upper_bound = q3 + iqr_factor * iqr
                mask = (differences >= lower_bound) & (differences <= upper_bound)
            elif method == 'zscore':
                # Z-score method on differences
                from scipy import stats
                z_threshold = 3.0
                z_scores = np.abs(stats.zscore(differences))
                mask = z_scores < z_threshold
            else:
                # No outlier removal or unsupported method
                mask = np.ones(len(differences), dtype=bool)
            
            # Apply outlier mask
            ref_clean = ref_data[mask]
            test_clean = test_data[mask]
            
            # Compute outlier statistics
            n_outliers = np.sum(~mask)
            if n_outliers > 0:
                outlier_stats = {
                    'n_outliers': n_outliers,
                    'n_original': len(ref_data),
                    'n_cleaned': len(ref_clean),
                    'outlier_percentage': (n_outliers / len(ref_data)) * 100,
                    'method': method,
                    'mask': mask,
                    'outlier_indices': np.where(~mask)[0],
                    'outlier_differences': differences[~mask],
                    'outlier_means': means[~mask]
                }
        else:
            # No outlier removal requested or no data
            ref_clean = ref_data
            test_clean = test_data
        
        # Calculate final plot data
        x_data = (ref_clean + test_clean)/2
        y_data = (test_clean - ref_clean)

        # Prepare metadata for plotting
        metadata = {
            'x_label': '(Test + Reference) / 2',
            'y_label': 'Test - Reference',
            'title': 'Bland-Altman',
            'notes': outlier_stats if outlier_stats else None,
        }
        
        return x_data, y_data, metadata
           
    
    def stats_script(self, x_data: np.ndarray, y_data: np.ndarray,
                     ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Bland-Altman statistical analysis

        Computes:
        - Bias and SD of differences
        - Limits of agreement (LoA)
        - Confidence intervals
        - Proportional bias test
        - Optional diagnostics (normality, LoA outliers)
        """
        import numpy as np
        from scipy import stats

        bias = np.mean(y_data)
        std_diff = np.std(y_data, ddof=1)
        n = len(y_data)

        agreement_multiplier = 1.96
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha / 2, n - 1) if n > 1 else 0
        
        if std_diff == 0.0 or np.isclose(std_diff, 0.0, atol=1e-10):
            upper_loa = lower_loa = bias
            bias_ci = (bias, bias)
            loa_upper_ci = (upper_loa, upper_loa)
            loa_lower_ci = (lower_loa, lower_loa)
        else:
            upper_loa = bias + agreement_multiplier * std_diff
            lower_loa = bias - agreement_multiplier * std_diff

            # Confidence intervals
            se_mean = std_diff / np.sqrt(n)
            se_loa = std_diff * np.sqrt(3 / n)
            bias_ci = (bias - t_value * se_mean, bias + t_value * se_mean)
            loa_upper_ci = (upper_loa - t_value * se_loa, upper_loa + t_value * se_loa)
            loa_lower_ci = (lower_loa - t_value * se_loa, lower_loa + t_value * se_loa)

        # Proportional bias: regression of diff vs mean
        slope, intercept, r_val, p_val, _ = stats.linregress(x_data, y_data)

        # Percentage of points outside LoA
        out_of_bounds = ((y_data < lower_loa) | (y_data > upper_loa)).sum()
        percent_outside = (out_of_bounds / n) * 100

        stats_results = {
            'bias': bias,
            'std_diff': std_diff,
            'upper_loa': upper_loa,
            'lower_loa': lower_loa,
            'bias_ci_lower': bias_ci[0],
            'bias_ci_upper': bias_ci[1],
            'loa_lower_ci_lower': loa_lower_ci[0],
            'loa_lower_ci_upper': loa_lower_ci[1],
            'loa_upper_ci_lower': loa_upper_ci[0],
            'loa_upper_ci_upper': loa_upper_ci[1],
            'confidence_level': confidence_level,
            'agreement_multiplier': agreement_multiplier,
            'proportional_bias_slope': slope,
            'proportional_bias_intercept': intercept,
            'proportional_bias_r': r_val,
            'proportional_bias_p_value': p_val,
            'percent_outside_loa': percent_outside,
            'sample_size': len(y_data)
        }

        return stats_results
    
    def _create_overlays(self, ref_data: np.ndarray, test_data: np.ndarray, 
                        stats_results: Dict[str, Any], params: dict) -> Dict[str, Dict[str, Any]]:
   
        # All overlays now use x_data for min/max
        bias_line = {
            'type': 'hline',
            'show': params.get('bias_line', True),
            'label': 'Bias Line',
            'main': self._get_bias_line(stats_results)
        }
        upper_loa_line = {
            'type': 'hline',
            'show': params.get('limits_of_agreement', True),
            'label': 'Upper Limit of Agreement',
            'main': self._get_upper_loa_line(stats_results)
        }
        lower_loa_line = {
            'type': 'hline',
            'show': params.get('limits_of_agreement', True),
            'label': 'Lower Limit of Agreement',
            'main': self._get_lower_loa_line(stats_results)
        }
        
        statistical_results = {
            'type': 'text',
            'show': params.get('statistical_results', True),
            'label': 'Statistical Results',
            'main': self._get_statistical_results(stats_results)
        }
        return {
            'bias_line': bias_line,
            'upper_loa_line': upper_loa_line,
            'lower_loa_line': lower_loa_line,            
            'statistical_results': statistical_results
        }
    def _get_bias_line(self, stats_results):
        bias = stats_results.get('bias', 0)     
        return {'y': bias}

    def _get_upper_loa_line(self, stats_results):
        upper_loa = stats_results.get('upper_loa', 0)
        return {'y': upper_loa}
    def _get_lower_loa_line(self, stats_results):
        lower_loa = stats_results.get('lower_loa', 0)
        return {'y': lower_loa}

    def _get_statistical_results(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for statistical results text overlay."""
        # Only return the most informative statistics for text overlay
        essential_stats = {
            'bias': stats_results.get('bias'),            
            'upper_loa': stats_results.get('upper_loa'),
            'lower_loa': stats_results.get('lower_loa'),            
        }
        return essential_stats
    

    @classmethod
    def get_description(cls) -> str:
        """
        Get a description of this comparison method for display in the wizard console.
        
        Returns:
            String description explaining what this comparison method does
        """
        return """Bland-Altman Analysis: Assesses agreement between two measurement methods.

• Plots differences (Test - Reference) against means ((Test + Reference)/2)
• Bias: Mean difference (systematic error between methods)
• Limits of Agreement: Range where 95% of differences are expected to lie
• Confidence Intervals: Uncertainty ranges for bias and limits
""" 