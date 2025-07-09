"""
Relative Error Time Series Comparison Method

This module implements relative error time series analysis between two data channels,
showing how the relative error (|test - ref| / ref) varies over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class RelativeErrorTimeSeriesComparison(BaseComparison):
    """
    Relative error time series comparison method.
    
    Analyzes how the relative error between test and reference datasets
    varies over time, providing insights into temporal patterns and stability.
    """
    
    name = "relative_error_time_series"
    description = "Time series analysis of relative error (|test - ref| / |ref|) with trend analysis"
    category = "Error Analysis"
    version = "1.0.0"
    tags = ["error", "relative", "time-series", "temporal", "trend"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "error_metric", "type": "str", "default": "relative", "help": "Error metric: relative, absolute, percentage"},
        {"name": "smoothing_window", "type": "int", "default": 10, "help": "Window size for smoothing filter"},
        {"name": "trend_analysis", "type": "bool", "default": True, "help": "Perform trend analysis on error series"},
        {"name": "outlier_threshold", "type": "float", "default": 3.0, "help": "Z-score threshold for outlier detection"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for statistical tests"}
    ]
    
    # Plot configuration
    plot_type = "line"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_threshold': {'default': True, 'label': 'Show Error Threshold', 'tooltip': 'Show horizontal line at error threshold'},
        'show_smoothed': {'default': True, 'label': 'Show Smoothed Error', 'tooltip': 'Show smoothed error trend line'},
        'show_trend_line': {'default': True, 'label': 'Show Trend Line', 'tooltip': 'Show linear trend line'},
        'show_confidence_bands': {'default': False, 'label': 'Show Confidence Bands', 'tooltip': 'Show confidence bands around trend'},
        'highlight_outliers': {'default': True, 'label': 'Highlight Outliers', 'tooltip': 'Highlight outlier points in the time series'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display error statistics on the plot'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the relative error time series analysis.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing relative error time series results with statistics and plot data
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Create time axis if not provided
        time_axis = ref_time if ref_time is not None else np.arange(len(ref_clean))
        
        # Calculate error values
        error_data = self._calculate_error_values(ref_clean, test_clean)
        
        # Calculate statistics
        stats_results = self.calculate_stats(ref_clean, test_clean, ref_time, test_time)
        
        # Prepare plot data
        plot_data = {
            'time': time_axis,
            'error_values': error_data,
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
        Calculate relative error time series statistics.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing time series statistics
        """
        # Calculate error values
        error_data = self._calculate_error_values(ref_data, test_data)
        time_axis = ref_time if ref_time is not None else np.arange(len(ref_data))
        
        # Initialize results
        stats_results = {
            'descriptive': {},
            'temporal': {},
            'outliers': {}
        }
        
        # Basic descriptive statistics
        stats_results['descriptive'] = self._compute_descriptive_stats(error_data)
        
        # Temporal analysis
        stats_results['temporal'] = self._compute_temporal_analysis(error_data, time_axis)
        
        # Outlier analysis
        stats_results['outliers'] = self._compute_outlier_analysis(error_data)
        
        # Trend analysis if requested
        if self.kwargs.get('trend_analysis', True):
            stats_results['trend'] = self._compute_trend_analysis(error_data, time_axis)
        
        # Smoothing analysis
        smoothing_window = self.kwargs.get('smoothing_window', 10)
        if smoothing_window > 1:
            stats_results['smoothed'] = self._compute_smoothed_analysis(error_data, smoothing_window)
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate relative error time series plot with overlays.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Calculate error values and time axis
        error_data = self._calculate_error_values(ref_data, test_data)
        time_axis = plot_config.get('time_axis', np.arange(len(ref_data)))
        
        # Create main time series plot
        self._create_error_time_series(ax, time_axis, error_data, plot_config)
        
        # Add overlay elements
        self._add_time_series_overlays(ax, time_axis, error_data, plot_config, stats_results)
        
        # Set labels and title
        error_type = self._get_error_type_label()
        ax.set_xlabel('Time')
        ax.set_ylabel(error_type)
        ax.set_title('Relative Error Time Series Analysis')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if there are overlays
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best')
    
    def _calculate_error_values(self, ref_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        """Calculate error values based on configuration."""
        error_metric = self.kwargs.get("error_metric", "relative")
        
        if error_metric == "absolute":
            error = np.abs(test_data - ref_data)
        elif error_metric == "percentage":
            with np.errstate(divide='ignore', invalid='ignore'):
                error = np.abs(test_data - ref_data) / np.abs(ref_data) * 100
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        else:  # relative
            with np.errstate(divide='ignore', invalid='ignore'):
                error = np.abs(test_data - ref_data) / np.abs(ref_data)
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        
        return error
    
    def _get_error_type_label(self) -> str:
        """Get appropriate y-axis label for error type."""
        error_metric = self.kwargs.get("error_metric", "relative")
        
        if error_metric == "absolute":
            return "Absolute Error"
        elif error_metric == "percentage":
            return "Percentage Error (%)"
        else:  # relative
            return "Relative Error"
    
    def _compute_descriptive_stats(self, error_data: np.ndarray) -> Dict[str, float]:
        """Compute basic descriptive statistics."""
        try:
            return {
                'mean': np.mean(error_data),
                'std': np.std(error_data),
                'var': np.var(error_data),
                'median': np.median(error_data),
                'mad': stats.median_abs_deviation(error_data),
                'min': np.min(error_data),
                'max': np.max(error_data),
                'range': np.ptp(error_data),
                'q25': np.percentile(error_data, 25),
                'q75': np.percentile(error_data, 75),
                'iqr': np.percentile(error_data, 75) - np.percentile(error_data, 25)
            }
        except Exception as e:
            return {
                'mean': np.nan, 'std': np.nan, 'var': np.nan,
                'median': np.nan, 'mad': np.nan, 'min': np.nan, 'max': np.nan,
                'range': np.nan, 'q25': np.nan, 'q75': np.nan, 'iqr': np.nan,
                'error': str(e)
            }
    
    def _compute_temporal_analysis(self, error_data: np.ndarray, time_axis: np.ndarray) -> Dict[str, Any]:
        """Compute temporal analysis statistics."""
        try:
            # Autocorrelation analysis
            autocorr = np.correlate(error_data - np.mean(error_data), 
                                   error_data - np.mean(error_data), mode='full')
            autocorr = autocorr / autocorr[len(autocorr)//2]
            
            # Stationarity test (simplified)
            window_size = max(10, len(error_data) // 10)
            windows = [error_data[i:i+window_size] for i in range(0, len(error_data)-window_size, window_size)]
            window_means = [np.mean(w) for w in windows]
            window_stds = [np.std(w) for w in windows]
            
            return {
                'autocorr_peak': np.max(autocorr[len(autocorr)//2+1:len(autocorr)//2+min(50, len(autocorr)//4)]),
                'mean_stability': np.std(window_means),
                'std_stability': np.std(window_stds),
                'n_windows': len(windows)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _compute_outlier_analysis(self, error_data: np.ndarray) -> Dict[str, Any]:
        """Analyze outliers in error time series."""
        try:
            threshold = self.kwargs.get('outlier_threshold', 3.0)
            z_scores = np.abs(stats.zscore(error_data))
            outlier_mask = z_scores > threshold
            
            return {
                'threshold': threshold,
                'n_outliers': np.sum(outlier_mask),
                'outlier_percentage': np.sum(outlier_mask) / len(error_data) * 100,
                'outlier_indices': np.where(outlier_mask)[0].tolist(),
                'outlier_values': error_data[outlier_mask].tolist(),
                'max_z_score': np.max(z_scores)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _compute_trend_analysis(self, error_data: np.ndarray, time_axis: np.ndarray) -> Dict[str, Any]:
        """Perform trend analysis on error time series."""
        try:
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_axis, error_data)
            
            # Trend classification
            if abs(slope) < std_err:
                trend_classification = "No trend"
            elif slope > 0:
                trend_classification = "Increasing"
            else:
                trend_classification = "Decreasing"
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err,
                'trend_classification': trend_classification
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _compute_smoothed_analysis(self, error_data: np.ndarray, window_size: int) -> Dict[str, Any]:
        """Compute smoothed error analysis."""
        try:
            # Simple moving average
            smoothed = np.convolve(error_data, np.ones(window_size)/window_size, mode='valid')
            
            # Smoothed statistics
            smoothed_mean = np.mean(smoothed)
            smoothed_std = np.std(smoothed)
            
            return {
                'smoothed_values': smoothed.tolist(),
                'smoothed_mean': smoothed_mean,
                'smoothed_std': smoothed_std,
                'window_size': window_size,
                'reduction_factor': smoothed_std / np.std(error_data)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _create_error_time_series(self, ax, time_axis: np.ndarray, error_data: np.ndarray, 
                                 plot_config: Dict[str, Any]) -> None:
        """Create the main error time series plot."""
        # Main time series plot
        ax.plot(time_axis, error_data, 'b-', linewidth=1, alpha=0.7, label='Error')
        
        # Highlight outliers if requested
        if plot_config.get('highlight_outliers', True):
            self._highlight_outlier_points(ax, time_axis, error_data)
    
    def _add_time_series_overlays(self, ax, time_axis: np.ndarray, error_data: np.ndarray, 
                                 plot_config: Dict[str, Any], 
                                 stats_results: Dict[str, Any] = None) -> None:
        """Add overlay elements to the time series plot."""
        if stats_results is None:
            return
        
        # Error threshold line
        if plot_config.get('show_threshold', True):
            self._add_threshold_line(ax, stats_results)
        
        # Smoothed error line
        if plot_config.get('show_smoothed', True) and 'smoothed' in stats_results:
            self._add_smoothed_line(ax, time_axis, stats_results['smoothed'])
        
        # Trend line
        if plot_config.get('show_trend_line', True) and 'trend' in stats_results:
            self._add_trend_line(ax, time_axis, stats_results['trend'])
        
        # Confidence bands
        if plot_config.get('show_confidence_bands', False):
            self._add_confidence_bands(ax, time_axis, error_data, stats_results)
        
        # Statistical results text
        if plot_config.get('show_statistical_results', True):
            self._add_statistical_text_box(ax, stats_results, plot_config)
    
    def _highlight_outlier_points(self, ax, time_axis: np.ndarray, error_data: np.ndarray) -> None:
        """Highlight outlier points in the time series."""
        try:
            threshold = self.kwargs.get('outlier_threshold', 3.0)
            z_scores = np.abs(stats.zscore(error_data))
            outlier_mask = z_scores > threshold
            
            if np.any(outlier_mask):
                ax.scatter(time_axis[outlier_mask], error_data[outlier_mask], 
                          color='red', s=50, alpha=0.8, label='Outliers', zorder=5)
        except Exception as e:
            print(f"[RelativeErrorTimeSeries] Error highlighting outliers: {e}")
    
    def _add_threshold_line(self, ax, stats_results: Dict[str, Any]) -> None:
        """Add error threshold line."""
        try:
            if 'descriptive' in stats_results:
                mean_val = stats_results['descriptive'].get('mean', np.nan)
                std_val = stats_results['descriptive'].get('std', np.nan)
                
                if not np.isnan(mean_val) and not np.isnan(std_val):
                    threshold = mean_val + 2 * std_val
                    ax.axhline(y=threshold, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label=f'Threshold: {threshold:.3f}')
        except Exception as e:
            print(f"[RelativeErrorTimeSeries] Error adding threshold line: {e}")
    
    def _add_smoothed_line(self, ax, time_axis: np.ndarray, smoothed_results: Dict[str, Any]) -> None:
        """Add smoothed error line."""
        try:
            smoothed_values = smoothed_results.get('smoothed_values', [])
            window_size = smoothed_results.get('window_size', 10)
            
            if smoothed_values:
                # Adjust time axis for smoothed values
                smoothed_time = time_axis[window_size-1:]
                ax.plot(smoothed_time, smoothed_values, 'g-', linewidth=2, 
                       label=f'Smoothed (window={window_size})')
        except Exception as e:
            print(f"[RelativeErrorTimeSeries] Error adding smoothed line: {e}")
    
    def _add_trend_line(self, ax, time_axis: np.ndarray, trend_results: Dict[str, Any]) -> None:
        """Add trend line to the plot."""
        try:
            slope = trend_results.get('slope', np.nan)
            intercept = trend_results.get('intercept', np.nan)
            
            if not np.isnan(slope) and not np.isnan(intercept):
                trend_line = slope * time_axis + intercept
                ax.plot(time_axis, trend_line, 'orange', linestyle='--', linewidth=2,
                       label=f'Trend (slope={slope:.4f})')
        except Exception as e:
            print(f"[RelativeErrorTimeSeries] Error adding trend line: {e}")
    
    def _add_confidence_bands(self, ax, time_axis: np.ndarray, error_data: np.ndarray, 
                             stats_results: Dict[str, Any]) -> None:
        """Add confidence bands around the error series."""
        try:
            if 'descriptive' in stats_results:
                mean_val = stats_results['descriptive'].get('mean', np.nan)
                std_val = stats_results['descriptive'].get('std', np.nan)
                
                if not np.isnan(mean_val) and not np.isnan(std_val):
                    confidence_level = self.kwargs.get('confidence_level', 0.95)
                    z_score = stats.norm.ppf((1 + confidence_level) / 2)
                    
                    upper_bound = mean_val + z_score * std_val
                    lower_bound = mean_val - z_score * std_val
                    
                    ax.fill_between(time_axis, lower_bound, upper_bound, 
                                   alpha=0.2, color='gray', 
                                   label=f'{confidence_level*100:.0f}% Confidence')
        except Exception as e:
            print(f"[RelativeErrorTimeSeries] Error adding confidence bands: {e}")
    
    def _add_statistical_text_box(self, ax, stats_results: Dict[str, Any], plot_config: Dict[str, Any]) -> None:
        """Add statistical results text box to the plot."""
        text_lines = self._format_statistical_text(stats_results, plot_config)
        if text_lines:
            text_str = '\n'.join(text_lines)
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _format_statistical_text(self, stats_results: Dict[str, Any], plot_config: Dict[str, Any] = None) -> List[str]:
        """Format statistical results for display on plot."""
        text_lines = []
        
        if plot_config is None:
            plot_config = {}
        
        # Descriptive statistics
        if 'descriptive' in stats_results:
            desc_stats = stats_results['descriptive']
            mean_val = desc_stats.get('mean', np.nan)
            std_val = desc_stats.get('std', np.nan)
            median_val = desc_stats.get('median', np.nan)
            
            if not np.isnan(mean_val):
                text_lines.append(f"Mean: {mean_val:.4f}")
            if not np.isnan(std_val):
                text_lines.append(f"Std: {std_val:.4f}")
            if not np.isnan(median_val):
                text_lines.append(f"Median: {median_val:.4f}")
        
        # Trend analysis
        if 'trend' in stats_results:
            trend_stats = stats_results['trend']
            slope = trend_stats.get('slope', np.nan)
            r_squared = trend_stats.get('r_squared', np.nan)
            trend_class = trend_stats.get('trend_classification', 'Unknown')
            
            if not np.isnan(slope):
                text_lines.append(f"Trend: {trend_class}")
            if not np.isnan(r_squared):
                text_lines.append(f"R²: {r_squared:.3f}")
        
        # Outlier information
        if 'outliers' in stats_results:
            outlier_stats = stats_results['outliers']
            n_outliers = outlier_stats.get('n_outliers', 0)
            outlier_pct = outlier_stats.get('outlier_percentage', 0)
            text_lines.append(f"Outliers: {n_outliers} ({outlier_pct:.1f}%)")
        
        return text_lines
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Relative Error Time Series",
            "description": "Analyzes how relative error between datasets varies over time",
            "interpretation": {
                "stable_error": "Consistent error levels suggest stable measurement differences",
                "increasing_trend": "Growing error over time may indicate drift or degradation",
                "decreasing_trend": "Decreasing error suggests improving agreement",
                "outliers": "Sudden spikes may indicate measurement artifacts or events",
                "smoothing": "Smoothed trends help identify underlying patterns"
            },
            "use_cases": [
                "Instrument calibration drift monitoring",
                "Algorithm performance tracking over time",
                "Sensor degradation analysis",
                "Temporal stability assessment"
            ],
            "tips": [
                "Use relative error for scale-independent comparison",
                "Look for systematic trends that might indicate drift",
                "Outliers may indicate specific events or measurement problems",
                "Smoothing helps identify underlying trends in noisy data",
                "Compare different time periods to assess stability"
            ]
        }