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
        {"name": "percentage_difference", "type": "bool", "default": False, "help": "Show differences as percentages instead of absolute values"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before calculating agreement statistics"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "options": ["iqr", "zscore"], "help": "Method for detecting outliers: IQR (robust) or Z-score (assumes normal distribution)"}
    ]
    
    # Plot configuration
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_bias_line': {'default': True, 'label': 'Show Bias Line', 'tooltip': 'Show horizontal line at mean difference (bias)', 'type': 'line'},
        'show_limits_of_agreement': {'default': True, 'label': 'Show Limits of Agreement', 'tooltip': 'Show upper and lower limits of agreement', 'type': 'line'},
        'show_statistical_results': {'default': False, 'label': 'Show Statistical Results', 'tooltip': 'Display statistical results on the plot', 'type': 'text'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the Bland-Altman analysis.
        
        Streamlined 3-step workflow:
        1. Validate input data (basic validation + remove NaN/infinite values)
        2. plot_script (core transformation + outlier removal)
        3. stats_script (statistical calculations)
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing Bland-Altman results with statistics and plot data
        """
        # === STEP 1: VALIDATE INPUT DATA ===
        # Basic validation (shape, type, length compatibility)
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        # Remove NaN and infinite values
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # === STEP 2: PLOT SCRIPT (core transformation + outlier removal) ===
        x_data, y_data, plot_metadata = self.plot_script(ref_clean, test_clean, self.kwargs)
        
        # === STEP 3: STATS SCRIPT (statistical calculations) ===
        stats_results = self.stats_script(x_data, y_data, ref_clean, test_clean, self.kwargs)
        
        # Prepare plot data
        plot_data = {
            'means': x_data,
            'differences': y_data,
            'ref_data': ref_clean,
            'test_data': test_clean,
            'valid_ratio': valid_ratio,
            'metadata': plot_metadata
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
        
        # === OUTLIER REMOVAL (moved after percentage calculation) ===
        ref_clean = ref_data
        test_clean = test_data
        
        if params.get('remove_outliers', False):
            method = params.get('outlier_method', 'iqr')
            
            # Remove NaN values before outlier detection
            valid_mask = np.isfinite(differences) & np.isfinite(means)
            if np.sum(valid_mask) < len(differences):
                print(f"[BlandAltman] Removing {len(differences) - np.sum(valid_mask)} invalid values before outlier detection")
                differences = differences[valid_mask]
                means = means[valid_mask]
                ref_clean = ref_clean[valid_mask]
                test_clean = test_clean[valid_mask]
            
            if len(differences) > 0:
                if method == 'iqr':
                    # IQR method on differences (now percentage differences if requested)
                    q1, q3 = np.percentile(differences, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    mask = (differences >= lower_bound) & (differences <= upper_bound)
                elif method == 'zscore':
                    # Z-score method on differences (now percentage differences if requested)
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(differences))
                    mask = z_scores < 3
                else:
                    # No outlier removal or unsupported method
                    mask = np.ones(len(differences), dtype=bool)
                
                # Apply outlier mask
                means = means[mask]
                differences = differences[mask]
                ref_clean = ref_clean[mask]
                test_clean = test_clean[mask]
                
                print(f"[BlandAltman] Outlier removal: {np.sum(~mask)} outliers removed using {method} method")
        else:
            # Still need to remove NaN values even without outlier removal
            valid_mask = np.isfinite(differences) & np.isfinite(means)
            if np.sum(valid_mask) < len(differences):
                print(f"[BlandAltman] Removing {len(differences) - np.sum(valid_mask)} invalid values")
                differences = differences[valid_mask]
                means = means[valid_mask]
                ref_clean = ref_clean[valid_mask]
                test_clean = test_clean[valid_mask]
        
        # Prepare metadata for plotting
        metadata = {
            'x_label': 'Mean of Two Methods',
            'y_label': 'Percentage Difference (%)' if params.get('percentage_difference', False) else 'Difference (Test - Reference)',
            'title': 'Bland-Altman Plot (Percentage Differences)' if params.get('percentage_difference', False) else 'Bland-Altman Plot',
            'plot_type': 'bland_altman',
            'is_percentage': params.get('percentage_difference', False),
            'outliers_removed': params.get('remove_outliers', False),
            'outlier_method': params.get('outlier_method', 'iqr') if params.get('remove_outliers', False) else None,
            'n_after_outlier_removal': len(ref_clean)
        }
        
        return means, differences, metadata
    
    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        BACKWARD COMPATIBILITY + SAFETY WRAPPER: Calculate Bland-Altman statistics.
        
        This method maintains compatibility with existing code and provides comprehensive
        validation and error handling around the core statistical calculations.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data (unused)
            test_time: Optional time array for test data (unused)
            
        Returns:
            Dictionary containing Bland-Altman statistics
        """
        # Get plot data using the script-based approach
        x_data, y_data, plot_metadata = self.plot_script(ref_data, test_data, self.kwargs)
        
        # === INPUT VALIDATION ===
        if len(x_data) != len(y_data):
            raise ValueError("X and Y data arrays must have the same length")
        
        if len(y_data) < 3:
            raise ValueError("Insufficient data for statistical analysis (minimum 3 samples required)")
        
        # Remove NaN values for statistical calculations
        valid_mask = np.isfinite(y_data) & np.isfinite(x_data)
        y_clean = y_data[valid_mask]
        x_clean = x_data[valid_mask]
        
        if len(y_clean) < 3:
            raise ValueError("Insufficient valid data points for statistical analysis")
        
        # === PURE CALCULATIONS (delegated to stats_script) ===
        stats_results = self.stats_script(x_clean, y_clean, ref_data, test_data, self.kwargs)
        
        # === OUTPUT VALIDATION ===
        if np.isnan(stats_results['bias']):
            # Handle same-channel comparison gracefully
            if np.allclose(ref_data, test_data, rtol=1e-10, atol=1e-10):
                print("[BlandAltman] Same-channel comparison detected - returning zero bias")
                stats_results['bias'] = 0.0
                stats_results['std_diff'] = 0.0
                stats_results['upper_loa'] = 0.0
                stats_results['lower_loa'] = 0.0

                return stats_results
            else:
                raise ValueError("Bias calculation produced NaN - check input data")
        if np.isnan(stats_results['std_diff']) or stats_results['std_diff'] <= 0:
            # Handle same-channel comparison gracefully
            if np.allclose(ref_data, test_data, rtol=1e-10, atol=1e-10):
                print("[BlandAltman] Same-channel comparison detected - returning zero standard deviation")
                stats_results['std_diff'] = 0.0
                stats_results['upper_loa'] = 0.0
                stats_results['lower_loa'] = 0.0
                return stats_results
            else:
                raise ValueError("Standard deviation calculation invalid - check input data variability")
        
        return stats_results
    
    def stats_script(self, x_data: np.ndarray, y_data: np.ndarray, 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for Bland-Altman analysis
        
        WARNING: Incorrect modifications may invalidate scientific conclusions and lead to 
        incorrect interpretation of measurement agreement.
        
        This calculates the core Bland-Altman statistics:
        - Bias (mean difference)
        - Limits of agreement
        - Confidence intervals
        - Proportional bias testing
        
        Only modify if you understand the statistical implications and theory
        behind Bland-Altman analysis.
        
        Args:
            x_data: X-axis data (typically means) - already cleaned
            y_data: Y-axis data (typically differences) - already cleaned
            ref_data: Original reference data (for additional calculations)
            test_data: Original test data (for additional calculations)
            params: Method parameters dictionary
            
        Returns:
            Dictionary containing statistical results
        """
        # Core Bland-Altman statistics
        bias = np.mean(y_data)
        std_diff = np.std(y_data, ddof=1)
        
        # Handle same-channel comparison (zero variance) gracefully
        if std_diff == 0.0 or np.isclose(std_diff, 0.0, atol=1e-10):
            print("[BlandAltman] Zero variance detected - likely same-channel comparison")
            std_diff = 0.0
            upper_loa = bias
            lower_loa = bias
        else:
            # Limits of agreement (using standard 1.96 for 95% limits)
            agreement_multiplier = 1.96
            upper_loa = bias + agreement_multiplier * std_diff
            lower_loa = bias - agreement_multiplier * std_diff
        

        
        # === CONFIDENCE INTERVALS (expanded inline) ===
        n = len(y_data)
        confidence_level = 0.95  # Standard 95% confidence level
        alpha = 1 - confidence_level
        
        # Handle same-channel comparison (zero variance) gracefully
        if std_diff == 0.0:
            # For zero variance, confidence intervals collapse to point estimates
            bias_ci = (bias, bias)
            loa_upper_ci = (upper_loa, upper_loa)
            loa_lower_ci = (lower_loa, lower_loa)
        else:
            # t-value for confidence interval
            t_value = stats.t.ppf(1 - alpha/2, n - 1) if n > 1 else 0
            
            # Standard error for mean difference (bias)
            se_mean = std_diff / np.sqrt(n)
            
            # Confidence interval for bias
            bias_ci = (bias - t_value * se_mean, bias + t_value * se_mean)
            
            # Standard error for limits of agreement
            se_loa = std_diff * np.sqrt(3/n)
            
            # Confidence intervals for limits of agreement
            loa_upper_ci = (upper_loa - t_value * se_loa, upper_loa + t_value * se_loa)
            loa_lower_ci = (lower_loa - t_value * se_loa, lower_loa + t_value * se_loa)
        
        confidence_intervals = {
            'bias_ci': bias_ci,
            'loa_ci': (loa_lower_ci, loa_upper_ci),
            'confidence_level': confidence_level
        }
        
        # Prepare results
        stats_results = {
            'bias': bias,
            'std_diff': std_diff,
            'upper_loa': upper_loa,
            'lower_loa': lower_loa,
            'agreement_limits': (lower_loa, upper_loa),
            'confidence_intervals': confidence_intervals,
            'n_samples': len(y_data),
            'n_valid': len(y_data),
            'agreement_multiplier': 1.96  # Standard value for reference
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
        
        # Use the plot_script for consistent transformation
        means, differences, plot_metadata = self.plot_script(ref_data, test_data, self.kwargs)
        
        # Apply performance optimizations
        means_plot, diff_plot = self._apply_performance_optimizations(means, differences, plot_config)
        
        # Create density plot based on configuration
        self._create_density_plot(ax, means_plot, diff_plot, plot_config)
        
        # Add Bland-Altman specific overlay elements
        self._add_bland_altman_overlays(ax, means, differences, plot_config, stats_results)
        
        # Add general overlay elements
        self._add_overlay_elements(ax, means_plot, diff_plot, plot_config, stats_results)
        
        # Set labels and title from metadata
        ax.set_xlabel(plot_metadata.get('x_label', 'Mean of Two Methods'))
        ax.set_ylabel(plot_metadata.get('y_label', 'Difference (Test - Reference)'))
        ax.set_title(plot_metadata.get('title', 'Bland-Altman Plot'))
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', False):
            ax.legend()
    
    def _create_density_plot(self, ax, means_plot: np.ndarray, diff_plot: np.ndarray, plot_config: Dict[str, Any]) -> None:
        """Create the main scatter plot for Bland-Altman analysis with individual pair styling."""
        # Check if we have individual pair styling information
        pair_styling = plot_config.get('pair_styling', [])
        
        if pair_styling:
            # Create scatter plot for each pair with individual styling
            for pair_info in pair_styling:
                ref_data = pair_info['ref_data']
                test_data = pair_info['test_data']
                marker = pair_info['marker']
                color = pair_info['color']
                pair_name = pair_info['pair_name']
                n_points = pair_info['n_points']
                
                # IMPORTANT: Use the plot_script method to ensure consistent outlier removal
                # This ensures the plot shows the same data that was used for statistics
                means, differences, metadata = self.plot_script(ref_data, test_data, self.kwargs)
                
                # Remove invalid values (additional safety check)
                valid_mask = np.isfinite(differences) & np.isfinite(means)
                means_clean = means[valid_mask]
                differences_clean = differences[valid_mask]
                
                # Create scatter plot for this pair
                # Only add label if legend is enabled
                if plot_config.get('show_legend', False):
                    ax.scatter(means_clean, differences_clean, 
                              marker=marker, 
                              s=50, 
                              c=color, 
                              alpha=0.6,
                              edgecolors='black',
                              linewidth=0.5,
                              label=f"{pair_name} (n={len(means_clean)})")
                else:
                    ax.scatter(means_clean, differences_clean, 
                              marker=marker, 
                              s=50, 
                              c=color, 
                              alpha=0.6,
                              edgecolors='black',
                              linewidth=0.5)
            
            # Add legend if multiple pairs and legend is enabled
            if len(pair_styling) > 1 and plot_config.get('show_legend', False):
                ax.legend(loc='best', fontsize=8)
        else:
            # Fallback to default styling if no pair information
            marker_style = plot_config.get('marker_style', 'o')
            marker_size = plot_config.get('marker_size', 50)
            marker_color = plot_config.get('marker_color', 'blue')
            alpha = plot_config.get('alpha', 0.6)
            
            # Create scatter plot
            ax.scatter(means_plot, diff_plot, 
                      marker=marker_style, 
                      s=marker_size, 
                      c=marker_color, 
                      alpha=alpha,
                      edgecolors='black',
                      linewidth=0.5)
    
    def _apply_performance_optimizations(self, means: np.ndarray, differences: np.ndarray, plot_config: Dict[str, Any]) -> tuple:
        """Apply performance optimizations like downsampling for large datasets."""
        # Get downsampling limit from config
        downsample_limit = plot_config.get('downsample', 2000)
        
        if len(means) > downsample_limit:
            # Downsample to improve performance
            indices = np.random.choice(len(means), downsample_limit, replace=False)
            means_plot = means[indices]
            diff_plot = differences[indices]
            print(f"[BlandAltman] Downsampled from {len(means)} to {len(means_plot)} points for plotting")
        else:
            means_plot = means
            diff_plot = differences
        
        return means_plot, diff_plot
    
    def _add_overlay_elements(self, ax, means_plot: np.ndarray, diff_plot: np.ndarray, 
                             plot_config: Dict[str, Any], stats_results: Dict[str, Any] = None) -> None:
        """Add general overlay elements like statistical text box."""
        # Add statistical results text box if requested
        if plot_config.get('show_statistical_results', False) and stats_results:
            text_lines = self._format_statistical_text(stats_results)
            if text_lines:
                text_str = '\n'.join(text_lines)
                ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
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
        print(f"[DEBUG] plot_config keys: {list(plot_config.keys())}")
        
        # Get style configurations from overlay configs
        bias_config = plot_config.get('bias_line_config', {})
        loa_config = plot_config.get('limits_of_agreement_config', {})
        ci_config = plot_config.get('confidence_intervals_config', {})
        
        # Add bias line
        show_bias = plot_config.get('show_bias_line', True)
        show_legend = plot_config.get('show_legend', False)
        print(f"[DEBUG] Bias line - show_bias: {show_bias}, show_legend: {show_legend}")
        if show_bias:
            bias = stats_results.get('bias', 0)
            bias_color = bias_config.get('color', '#2ecc71')  # Default green
            bias_style = bias_config.get('linestyle', '-')
            bias_width = bias_config.get('linewidth', 2)
            bias_alpha = bias_config.get('alpha', 0.8)
            
            # Only add label if legend is enabled
            if show_legend:
                ax.axhline(y=bias, color=bias_color, linestyle=bias_style, linewidth=bias_width, 
                          alpha=bias_alpha, label=f'Bias: {bias:.3f}')
                print(f"[DEBUG] Bias line drawn at y={bias:.3f} with color {bias_color}, label: Bias: {bias:.3f}")
            else:
                ax.axhline(y=bias, color=bias_color, linestyle=bias_style, linewidth=bias_width, 
                          alpha=bias_alpha)
                print(f"[DEBUG] Bias line drawn at y={bias:.3f} with color {bias_color}, no label")
        
        # Add limits of agreement
        show_loa = plot_config.get('show_limits_of_agreement', True)
        print(f"[DEBUG] Limits of agreement - show_loa: {show_loa}")
        if show_loa:
            upper_loa = stats_results.get('upper_loa', 0)
            lower_loa = stats_results.get('lower_loa', 0)
            loa_color = loa_config.get('color', '#f39c12')  # Default orange
            loa_style = loa_config.get('linestyle', '--')
            loa_width = loa_config.get('linewidth', 2)
            loa_alpha = loa_config.get('alpha', 0.8)
            
            # Only add labels if legend is enabled
            if show_legend:
                ax.axhline(y=upper_loa, color=loa_color, linestyle=loa_style, linewidth=loa_width, 
                          alpha=loa_alpha, label=f'Upper LoA: {upper_loa:.3f}')
                ax.axhline(y=lower_loa, color=loa_color, linestyle=loa_style, linewidth=loa_width, 
                          alpha=loa_alpha, label=f'Lower LoA: {lower_loa:.3f}')
                print(f"[DEBUG] LoA lines drawn at y={lower_loa:.3f} and y={upper_loa:.3f} with color {loa_color}, with labels")
            else:
                ax.axhline(y=upper_loa, color=loa_color, linestyle=loa_style, linewidth=loa_width, 
                          alpha=loa_alpha)
                ax.axhline(y=lower_loa, color=loa_color, linestyle=loa_style, linewidth=loa_width, 
                          alpha=loa_alpha)
                print(f"[DEBUG] LoA lines drawn at y={lower_loa:.3f} and y={upper_loa:.3f} with color {loa_color}, no labels")
        
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
            
            # Get CI colors from configurations - use same colors as the lines they represent
            ci_alpha = ci_config.get('alpha', 0.3)
            
            # Confidence interval for bias (use same color as bias line)
            if 'bias_ci' in ci:
                bias_lower, bias_upper = ci['bias_ci']
                print(f"[DEBUG] Drawing bias CI: {bias_lower:.3f} to {bias_upper:.3f}, x_range: [{x_min:.3f}, {x_max:.3f}]")
                if bias_lower != bias_upper:  # Only draw if there's a meaningful interval
                    bias_ci_color = bias_config.get('color', '#2ecc71')  # Same as bias line
                    if show_legend:
                        ax.fill_between([x_min, x_max], bias_lower, bias_upper, 
                                       alpha=ci_alpha, color=bias_ci_color, label='Bias 95% CI')
                        print(f"[DEBUG] Bias CI drawn successfully with color {bias_ci_color}, with label")
                    else:
                        ax.fill_between([x_min, x_max], bias_lower, bias_upper, 
                                       alpha=ci_alpha, color=bias_ci_color)
                        print(f"[DEBUG] Bias CI drawn successfully with color {bias_ci_color}, no label")
                else:
                    print(f"[DEBUG] Bias CI not drawn - identical bounds")
            
            # Confidence intervals for limits of agreement (use same color as LoA lines)
            if 'loa_ci' in ci:
                loa_lower_ci, loa_upper_ci = ci['loa_ci']
                print(f"[DEBUG] Drawing LoA CIs - Lower: {loa_lower_ci[0]:.3f} to {loa_lower_ci[1]:.3f}, Upper: {loa_upper_ci[0]:.3f} to {loa_upper_ci[1]:.3f}")
                loa_ci_color = loa_config.get('color', '#f39c12')  # Same as LoA lines
                
                # Draw lower LoA confidence interval
                if loa_lower_ci[0] != loa_lower_ci[1]:
                    if show_legend:
                        ax.fill_between([x_min, x_max], loa_lower_ci[0], loa_lower_ci[1], 
                                       alpha=ci_alpha, color=loa_ci_color, label='Lower LoA 95% CI')
                        print(f"[DEBUG] Lower LoA CI drawn successfully with color {loa_ci_color}, with label")
                    else:
                        ax.fill_between([x_min, x_max], loa_lower_ci[0], loa_lower_ci[1], 
                                       alpha=ci_alpha, color=loa_ci_color)
                        print(f"[DEBUG] Lower LoA CI drawn successfully with color {loa_ci_color}, no label")
                else:
                    print(f"[DEBUG] Lower LoA CI not drawn - identical bounds")
                    
                # Draw upper LoA confidence interval (same color as lower LoA CI)
                if loa_upper_ci[0] != loa_upper_ci[1]:
                    if show_legend:
                        ax.fill_between([x_min, x_max], loa_upper_ci[0], loa_upper_ci[1], 
                                       alpha=ci_alpha, color=loa_ci_color, label='Upper LoA 95% CI')
                        print(f"[DEBUG] Upper LoA CI drawn successfully with color {loa_ci_color}, with label")
                    else:
                        ax.fill_between([x_min, x_max], loa_upper_ci[0], loa_upper_ci[1], 
                                       alpha=ci_alpha, color=loa_ci_color)
                        print(f"[DEBUG] Upper LoA CI drawn successfully with color {loa_ci_color}, no label")
                else:
                    print(f"[DEBUG] Upper LoA CI not drawn - identical bounds")
    
    
    

    
    def _format_statistical_text(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format statistical results for text overlay."""
        lines = []
        
        bias = stats_results.get('bias', np.nan)
        if not np.isnan(bias):
            lines.append(f"Bias: {bias:.3f}")
        
        upper_loa = stats_results.get('upper_loa', np.nan)
        lower_loa = stats_results.get('lower_loa', np.nan)
        if not np.isnan(upper_loa) and not np.isnan(lower_loa):
            lines.append(f"LoA: [{lower_loa:.3f}, {upper_loa:.3f}]")
        
        n_samples = stats_results.get('n_samples', 0)
        if n_samples > 0:
            lines.append(f"N: {n_samples}")
        
        return lines
    
    def _get_overlay_functional_properties(self, overlay_id: str, overlay_type: str, 
                                         stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get functional properties for Bland-Altman overlays (no arbitrary styling)."""
        properties = {}
        
        if overlay_id == 'show_bias_line' and overlay_type == 'line':
            properties.update({
                'y_value': stats_results.get('bias', 0),
                'label': f'Bias: {stats_results.get("bias", 0):.3f}'
            })
        elif overlay_id == 'show_limits_of_agreement' and overlay_type == 'line':
            properties.update({
                'y_values': [
                    stats_results.get('lower_loa', 0),
                    stats_results.get('upper_loa', 0)
                ],
                'label': 'Limits of Agreement'
            })
        elif overlay_id == 'show_confidence_intervals' and overlay_type == 'fill':
            properties.update({
                'fill_between': True,
                'confidence_intervals': stats_results.get('confidence_intervals', {}),
                'label': '95% Confidence Intervals'
            })
        elif overlay_id == 'show_statistical_results' and overlay_type == 'text':
            properties.update({
                'position': (0.02, 0.98),
                'text_lines': self._format_statistical_text(stats_results)
            })
        elif overlay_id == 'show_legend' and overlay_type == 'legend':
            properties.update({
                'label': 'Legend'
            })
        
        return properties
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Bland-Altman Analysis",
            "description": "Assesses agreement between two methods by plotting differences against means",
            "interpretation": {
                "bias": "Mean difference between methods (systematic error)",
                "limits_of_agreement": "Range within which 95% of differences are expected to lie",
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
                "Consider percentage differences for ratio data",
                "Remove outliers cautiously - they may be real differences"
            ]
        } 