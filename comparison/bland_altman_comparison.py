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
    category = "Agreement"
    version = "1.0.0"
    tags = ["scatter", "agreement", "bias", "limits", "clinical", "validation"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "agreement_multiplier", "type": "float", "default": 1.96, "help": "Multiplier for limits of agreement (typically 1.96 for 95% limits)"},
        {"name": "test_proportional_bias", "type": "bool", "default": True, "help": "Test for proportional bias using regression"},
        {"name": "percentage_difference", "type": "bool", "default": False, "help": "Calculate percentage differences instead of absolute"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for confidence intervals"},
        {"name": "bootstrap_samples", "type": "int", "default": 1000, "help": "Number of bootstrap samples for confidence intervals"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before analysis"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "help": "Outlier detection method: iqr, zscore"}
    ]
    
    # Plot configuration
    plot_type = "bland_altman"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_bias_line': {'default': True, 'label': 'Show Bias Line', 'tooltip': 'Show horizontal line at mean difference (bias)'},
        'show_limits_of_agreement': {'default': True, 'label': 'Show Limits of Agreement', 'tooltip': 'Show upper and lower limits of agreement'},
        'show_confidence_intervals': {'default': True, 'label': 'Show Confidence Intervals', 'tooltip': 'Show confidence intervals around bias and limits'},
        'highlight_outliers': {'default': False, 'label': 'Highlight Outliers', 'tooltip': 'Highlight outlier points on the plot'},
        'show_statistical_results': {'default': False, 'label': 'Show Statistical Results', 'tooltip': 'Display statistical results on the plot'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the Bland-Altman analysis.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing Bland-Altman results with statistics and plot data
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Remove outliers if requested
        if self.kwargs.get('remove_outliers', False):
            ref_clean, test_clean = self._remove_outliers(ref_clean, test_clean)
        
        # Calculate statistics
        stats_results = self.calculate_stats(ref_clean, test_clean, ref_time, test_time)
        
        # Calculate means and differences for plotting
        means = (ref_clean + test_clean) / 2
        differences = test_clean - ref_clean
        
        # Convert to percentage differences if requested
        if self.kwargs.get('percentage_difference', False):
            differences = (differences / means) * 100
        
        # Prepare plot data
        plot_data = {
            'means': means,
            'differences': differences,
            'ref_data': ref_clean,
            'test_data': test_clean,
            'valid_ratio': valid_ratio
        }
        
        # Combine results
        results = {
            'method': self.name,
            'n_samples': len(ref_clean),
            'valid_ratio': valid_ratio,
            'statistics': stats_results,
            'plot_data': plot_data
        }
        
        # Store results
        self.results = results
        return results
    
    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate Bland-Altman statistics.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing Bland-Altman statistics
        """
        # Calculate means and differences
        means = (ref_data + test_data) / 2
        differences = test_data - ref_data
        
        # Convert to percentage differences if requested
        if self.kwargs.get('percentage_difference', False):
            differences = (differences / means) * 100
        
        # Basic statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Limits of agreement
        agreement_multiplier = self.kwargs.get('agreement_multiplier', 1.96)
        upper_loa = mean_diff + agreement_multiplier * std_diff
        lower_loa = mean_diff - agreement_multiplier * std_diff
        
        # Test for proportional bias
        proportional_bias = None
        if self.kwargs.get('test_proportional_bias', True):
            proportional_bias = self._test_proportional_bias(means, differences)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(differences, mean_diff, std_diff)
        print(f"[DEBUG] Calculated confidence intervals: {confidence_intervals}")
        
        # Prepare results
        stats_results = {
            'bias': mean_diff,
            'std_diff': std_diff,
            'upper_loa': upper_loa,
            'lower_loa': lower_loa,
            'agreement_limits': (lower_loa, upper_loa),
            'proportional_bias': proportional_bias,
            'confidence_intervals': confidence_intervals,
            'n_samples': len(differences)
        }
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate Bland-Altman plot with performance and overlay options.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Calculate means and differences
        means = (ref_data + test_data) / 2
        differences = test_data - ref_data
        
        # Convert to percentage differences if requested
        if self.kwargs.get('percentage_difference', False):
            differences = (differences / means) * 100
        
        # Apply performance optimizations
        means_plot, diff_plot = self._apply_performance_optimizations(means, differences, plot_config)
        
        # Create density plot based on configuration
        self._create_density_plot(ax, means_plot, diff_plot, plot_config)
        
        # Add Bland-Altman specific overlay elements
        self._add_bland_altman_overlays(ax, means, differences, plot_config, stats_results)
        
        # Add general overlay elements
        self._add_overlay_elements(ax, means_plot, diff_plot, plot_config, stats_results)
        
        # Set labels and title
        ax.set_xlabel('Mean of Two Methods')
        if self.kwargs.get('percentage_difference', False):
            ax.set_ylabel('Percentage Difference (%)')
            ax.set_title('Bland-Altman Plot (Percentage Differences)')
        else:
            ax.set_ylabel('Difference (Test - Reference)')
            ax.set_title('Bland-Altman Plot')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', False):
            ax.legend()
    
    def _add_bland_altman_overlays(self, ax, means: np.ndarray, differences: np.ndarray, 
                                  plot_config: Dict[str, Any] = None, 
                                  stats_results: Dict[str, Any] = None) -> None:
        """
        Add Bland-Altman specific overlay elements.
        
        Args:
            ax: Matplotlib axes object
            means: Mean values array
            differences: Difference values array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results
        """
        if plot_config is None:
            plot_config = {}
        
        if stats_results is None:
            print("[DEBUG] stats_results is None, returning early")
            return
        
        print(f"[DEBUG] stats_results keys: {list(stats_results.keys())}")
        
        # Add bias line
        if plot_config.get('show_bias_line', True):
            bias = stats_results.get('bias', 0)
            ax.axhline(y=bias, color='green', linestyle='-', linewidth=2, alpha=0.8, label=f'Bias: {bias:.3f}')
        
        # Add limits of agreement
        if plot_config.get('show_limits_of_agreement', True):
            upper_loa = stats_results.get('upper_loa', 0)
            lower_loa = stats_results.get('lower_loa', 0)
            ax.axhline(y=upper_loa, color='orange', linestyle='--', linewidth=2, alpha=0.8, 
                      label=f'Upper LoA: {upper_loa:.3f}')
            ax.axhline(y=lower_loa, color='orange', linestyle='--', linewidth=2, alpha=0.8, 
                      label=f'Lower LoA: {lower_loa:.3f}')
        
        # Add confidence intervals for bias and limits
        # Check both parameter names for backward compatibility
        show_ci = plot_config.get('show_confidence_intervals', plot_config.get('confidence_interval', True))
        print(f"[DEBUG] Confidence intervals - show_ci: {show_ci}, has_ci_data: {'confidence_intervals' in stats_results}")
        if show_ci and 'confidence_intervals' in stats_results:
            ci = stats_results['confidence_intervals']
            
            # Get the x-axis range for confidence intervals
            if len(means) > 0:
                x_min, x_max = np.min(means), np.max(means)
                # Add some padding to make the bands more visible
                x_range = x_max - x_min
                if x_range > 0:
                    x_min -= 0.05 * x_range
                    x_max += 0.05 * x_range
                else:
                    # If all means are the same, create a small range
                    x_min -= 0.1
                    x_max += 0.1
            else:
                # Fallback to axis limits if means is empty
                x_min, x_max = ax.get_xlim()
            
            # Confidence interval for bias
            if 'bias_ci' in ci:
                bias_lower, bias_upper = ci['bias_ci']
                print(f"[DEBUG] Drawing bias CI: {bias_lower:.3f} to {bias_upper:.3f}, x_range: [{x_min:.3f}, {x_max:.3f}]")
                if bias_lower != bias_upper:  # Only draw if there's a meaningful interval
                    ax.fill_between([x_min, x_max], bias_lower, bias_upper, 
                                   alpha=0.3, color='green', label='Bias 95% CI')
                    print(f"[DEBUG] Bias CI drawn successfully")
                else:
                    print(f"[DEBUG] Bias CI not drawn - identical bounds")
            
            # Confidence intervals for limits of agreement
            if 'loa_ci' in ci:
                loa_lower_ci, loa_upper_ci = ci['loa_ci']
                print(f"[DEBUG] Drawing LoA CIs - Lower: {loa_lower_ci[0]:.3f} to {loa_lower_ci[1]:.3f}, Upper: {loa_upper_ci[0]:.3f} to {loa_upper_ci[1]:.3f}")
                # Draw lower LoA confidence interval
                if loa_lower_ci[0] != loa_lower_ci[1]:
                    ax.fill_between([x_min, x_max], loa_lower_ci[0], loa_lower_ci[1], 
                                   alpha=0.3, color='orange', label='Lower LoA 95% CI')
                    print(f"[DEBUG] Lower LoA CI drawn successfully")
                else:
                    print(f"[DEBUG] Lower LoA CI not drawn - identical bounds")
                # Draw upper LoA confidence interval
                if loa_upper_ci[0] != loa_upper_ci[1]:
                    ax.fill_between([x_min, x_max], loa_upper_ci[0], loa_upper_ci[1], 
                                   alpha=0.3, color='red', label='Upper LoA 95% CI')
                    print(f"[DEBUG] Upper LoA CI drawn successfully")
                else:
                    print(f"[DEBUG] Upper LoA CI not drawn - identical bounds")
    
    def _test_proportional_bias(self, means: np.ndarray, differences: np.ndarray) -> Dict[str, Any]:
        """
        Test for proportional bias using linear regression.
        
        Args:
            means: Mean values array
            differences: Difference values array
            
        Returns:
            Dictionary containing proportional bias test results
        """
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(means, differences)
            
            # Test if slope is significantly different from zero
            t_stat = slope / std_err
            df = len(means) - 2
            p_value_slope = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value_slope,
                'std_err': std_err,
                'significant': p_value_slope < 0.05,
                'interpretation': "Significant proportional bias detected" if p_value_slope < 0.05 else "No significant proportional bias"
            }
        except Exception as e:
            return {
                'error': str(e),
                'significant': False,
                'interpretation': "Could not test for proportional bias"
            }
    
    def _calculate_confidence_intervals(self, differences: np.ndarray, mean_diff: float, 
                                      std_diff: float) -> Dict[str, Any]:
        """
        Calculate confidence intervals for bias and limits of agreement.
        
        Args:
            differences: Difference values array
            mean_diff: Mean difference (bias)
            std_diff: Standard deviation of differences
            
        Returns:
            Dictionary containing confidence intervals
        """
        n = len(differences)
        confidence_level = self.kwargs.get('confidence_level', 0.95)
        alpha = 1 - confidence_level
        
        # t-value for confidence interval
        t_value = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Standard error for mean difference
        se_mean = std_diff / np.sqrt(n)
        
        # Confidence interval for bias
        bias_ci = (mean_diff - t_value * se_mean, mean_diff + t_value * se_mean)
        
        # Standard error for limits of agreement
        agreement_multiplier = self.kwargs.get('agreement_multiplier', 1.96)
        se_loa = std_diff * np.sqrt(3/n)
        
        # Confidence intervals for limits of agreement
        upper_loa = mean_diff + agreement_multiplier * std_diff
        lower_loa = mean_diff - agreement_multiplier * std_diff
        
        loa_upper_ci = (upper_loa - t_value * se_loa, upper_loa + t_value * se_loa)
        loa_lower_ci = (lower_loa - t_value * se_loa, lower_loa + t_value * se_loa)
        
        return {
            'bias_ci': bias_ci,
            'loa_ci': (loa_lower_ci, loa_upper_ci),
            'confidence_level': confidence_level
        }
    
    def _remove_outliers(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from data using specified method."""
        method = self.kwargs.get('outlier_method', 'iqr')
        
        # Calculate differences for outlier detection
        differences = test_data - ref_data
        
        if method == 'iqr':
            # IQR method on differences
            q1, q3 = np.percentile(differences, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (differences >= lower_bound) & (differences <= upper_bound)
        elif method == 'zscore':
            # Z-score method on differences
            z_scores = np.abs(stats.zscore(differences))
            mask = z_scores < 3
        else:
            # No outlier removal
            mask = np.ones(len(ref_data), dtype=bool)
        
        return ref_data[mask], test_data[mask]
    
    def _format_statistical_text(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format statistical results for display on plot."""
        text_lines = []
        
        # Add bias
        bias = stats_results.get('bias', np.nan)
        if not np.isnan(bias):
            text_lines.append(f"Bias: {bias:.3f}")
        
        # Add limits of agreement
        upper_loa = stats_results.get('upper_loa', np.nan)
        lower_loa = stats_results.get('lower_loa', np.nan)
        if not np.isnan(upper_loa) and not np.isnan(lower_loa):
            text_lines.append(f"LoA: [{lower_loa:.3f}, {upper_loa:.3f}]")
        
        # Add proportional bias result
        if 'proportional_bias' in stats_results:
            prop_bias = stats_results['proportional_bias']
            if 'significant' in prop_bias:
                if prop_bias['significant']:
                    text_lines.append(f"Proportional bias: Yes (p={prop_bias.get('p_value', 'N/A'):.3f})")
                else:
                    text_lines.append("Proportional bias: No")
        
        # Add sample size
        n_samples = stats_results.get('n_samples', 0)
        if n_samples > 0:
            text_lines.append(f"N: {n_samples}")
        
        return text_lines
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Bland-Altman Analysis",
            "description": "Assesses agreement between two methods by plotting differences against means",
            "interpretation": {
                "bias": "Mean difference between methods (systematic error)",
                "limits_of_agreement": "Range within which 95% of differences are expected to lie",
                "proportional_bias": "Whether bias changes with magnitude of measurement",
                "confidence_intervals": "Uncertainty ranges for bias and limits of agreement"
            },
            "use_cases": [
                "Method comparison and validation",
                "Clinical measurement agreement",
                "Inter-rater reliability assessment",
                "Instrument calibration validation"
            ],
            "tips": [
                "Look for bias (mean difference) close to zero",
                "Check if limits of agreement are clinically acceptable",
                "Test for proportional bias (trend in differences)",
                "Consider percentage differences for ratio data",
                "Remove outliers cautiously - they may be real differences"
            ]
        } 