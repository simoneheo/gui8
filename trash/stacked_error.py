"""
Stacked Error Comparison Method

This module implements stacked error analysis between two data channels,
showing different error components as stacked areas over time.
"""

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class StackedErrorComparison(BaseComparison):
    """
    Stacked error analysis comparison method.
    
    Decomposes errors into different components and visualizes them as
    stacked areas over time or sample indices.
    """
    
    name = "stacked_error"
    description = "Decompose and visualize error components as stacked areas over time"
    category = "Error Analysis"
    tags = ["stacked_area", "error", "decomposition", "time_series", "components"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "error_components", "type": "str", "default": "bias_precision", "options": ["bias_precision", "systematic_random", "absolute_relative"], "help": "How to decompose error components"},
        {"name": "time_window", "type": "int", "default": 100, "min": 10, "max": 1000, "help": "Window size for rolling error calculation"},
        {"name": "overlap", "type": "float", "default": 0.5, "min": 0.0, "max": 0.9, "step": 0.1, "help": "Overlap between windows (0 = no overlap)"},
        {"name": "normalize_by_total", "type": "bool", "default": False, "help": "Normalize each stack to show proportional errors"},
        {"name": "smooth_errors", "type": "bool", "default": True, "help": "Apply smoothing to error components"},
        {"name": "smoothing_window", "type": "int", "default": 10, "min": 3, "max": 50, "help": "Window size for smoothing"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before stacking"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "min": 0.5, "max": 0.99, "step": 0.01, "help": "Confidence level for error bounds"}
    ]
    
    # Plot configuration
    plot_type = "stacked_area"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'total_error_line': {'default': True, 'label': 'Total Error Line', 'tooltip': 'Line showing total error magnitude', 'type': 'line'},
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'tooltip': 'Display error component statistics', 'type': 'text'},
        'legend': {'default': True, 'label': 'Component Legend', 'tooltip': 'Show legend for error components', 'type': 'text'},
        'zero_reference': {'default': True, 'label': 'Zero Reference', 'tooltip': 'Reference line at zero error', 'type': 'line'},
        'confidence_bands': {'default': False, 'label': 'Confidence Bands', 'tooltip': 'Show confidence intervals around components', 'type': 'fill'}
    }
    
    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for stacked error analysis
        
        This defines what gets plotted for stacked area visualization.
        
        Args:
            ref_data: Reference measurements (cleaned of NaN/infinite values)
            test_data: Test measurements (cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (time_data, error_components, total_error, metadata)
                time_data: Time or sample indices for X-axis
                error_components: Dictionary of error components for stacking
                total_error: Total error magnitude
                metadata: Plot configuration dictionary
        """
        
        # Handle outlier removal if requested
        if params.get("remove_outliers", False):
            ref_data, test_data = self._remove_outliers(ref_data, test_data, params)
        
        # Get parameters
        error_components = params.get("error_components", "bias_precision")
        time_window = params.get("time_window", 100)
        overlap = params.get("overlap", 0.5)
        normalize_by_total = params.get("normalize_by_total", False)
        smooth_errors = params.get("smooth_errors", True)
        smoothing_window = params.get("smoothing_window", 10)
        
        # Calculate rolling error components
        time_data, component_data = self._calculate_rolling_components(
            ref_data, test_data, error_components, time_window, overlap
        )
        
        # Apply smoothing if requested
        if smooth_errors:
            component_data = self._smooth_components(component_data, smoothing_window)
        
        # Calculate total error
        total_error = np.sum(list(component_data.values()), axis=0)
        
        # Normalize if requested
        if normalize_by_total:
            for component in component_data:
                component_data[component] = component_data[component] / (total_error + 1e-10) * 100
            total_error = np.ones_like(total_error) * 100
        
        # Create metadata
        metadata = {
            'error_components': error_components,
            'time_window': time_window,
            'overlap': overlap,
            'normalize_by_total': normalize_by_total,
            'smooth_errors': smooth_errors,
            'smoothing_window': smoothing_window,
            'n_windows': len(time_data),
            'component_names': list(component_data.keys()),
            'xlabel': 'Time/Sample Index',
            'ylabel': 'Error Magnitude (%)' if normalize_by_total else 'Error Magnitude'
        }
        
        return time_data, component_data, total_error, metadata
    
    def stats_script(self, time_data: np.ndarray, error_components: Dict[str, np.ndarray], 
                    total_error: np.ndarray, ref_data: np.ndarray, params: dict) -> dict:
        """
        Statistical analysis for stacked error components
        
        Args:
            time_data: Time indices from plot_script
            error_components: Error components from plot_script
            total_error: Total error from plot_script
            ref_data: Reference data (unused but required for interface)
            params: Method parameters
            
        Returns:
            Dictionary containing statistical results
        """
        
        # Calculate statistics for each component
        component_stats = {}
        for component_name, component_values in error_components.items():
            component_stats[component_name] = {
                'mean': np.mean(component_values),
                'std': np.std(component_values),
                'max': np.max(component_values),
                'min': np.min(component_values),
                'median': np.median(component_values),
                'contribution_percent': np.mean(component_values) / np.mean(total_error) * 100
            }
        
        # Calculate total error statistics
        total_stats = {
            'mean_total_error': np.mean(total_error),
            'std_total_error': np.std(total_error),
            'max_total_error': np.max(total_error),
            'min_total_error': np.min(total_error),
            'median_total_error': np.median(total_error)
        }
        
        # Find dominant component
        mean_contributions = {name: stats['contribution_percent'] 
                            for name, stats in component_stats.items()}
        if mean_contributions:
            dominant_component = max(mean_contributions.keys(), key=lambda x: mean_contributions[x])
        else:
            dominant_component = "unknown"
        
        # Calculate error stability (coefficient of variation)
        error_cv = np.std(total_error) / np.mean(total_error) * 100
        
        # Calculate trend analysis
        if len(total_error) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_data, total_error)
            trend_direction = "increasing" if slope > 0 else "decreasing"
            trend_significant = p_value < 0.05
        else:
            slope = 0
            r_value = 0
            p_value = 1
            trend_direction = "stable"
            trend_significant = False
        
        stats_results = {
            'component_statistics': component_stats,
            'total_error_statistics': total_stats,
            'dominant_component': dominant_component,
            'dominant_contribution': mean_contributions[dominant_component],
            'error_stability_cv': error_cv,
            'trend_slope': slope,
            'trend_r_squared': r_value**2,
            'trend_p_value': p_value,
            'trend_direction': trend_direction,
            'trend_significant': trend_significant,
            'n_windows': len(time_data),
            'component_names': list(error_components.keys())
        }
        
        return stats_results
    
    def _calculate_rolling_components(self, ref_data: np.ndarray, test_data: np.ndarray,
                                    error_components: str, time_window: int, 
                                    overlap: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Calculate rolling error components"""
        
        # Calculate step size based on overlap
        step_size = int(time_window * (1 - overlap))
        if step_size < 1:
            step_size = 1
        
        # Calculate number of windows
        n_samples = len(ref_data)
        n_windows = (n_samples - time_window) // step_size + 1
        
        # Initialize arrays
        time_data = np.zeros(n_windows)
        component_data = {}
        
        # Initialize component arrays based on decomposition type
        if error_components == "bias_precision":
            component_data = {
                'bias_error': np.zeros(n_windows),
                'precision_error': np.zeros(n_windows)
            }
        elif error_components == "systematic_random":
            component_data = {
                'systematic_error': np.zeros(n_windows),
                'random_error': np.zeros(n_windows)
            }
        else:  # absolute_relative
            component_data = {
                'absolute_error': np.zeros(n_windows),
                'relative_error': np.zeros(n_windows)
            }
        
        # Calculate components for each window
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + time_window
            
            # Get window data
            ref_window = ref_data[start_idx:end_idx]
            test_window = test_data[start_idx:end_idx]
            
            # Calculate time point (center of window)
            time_data[i] = (start_idx + end_idx) / 2
            
            # Calculate error components based on type
            if error_components == "bias_precision":
                bias, precision = self._calculate_bias_precision(ref_window, test_window)
                component_data['bias_error'][i] = abs(bias)
                component_data['precision_error'][i] = precision
                
            elif error_components == "systematic_random":
                systematic, random = self._calculate_systematic_random(ref_window, test_window)
                component_data['systematic_error'][i] = systematic
                component_data['random_error'][i] = random
                
            else:  # absolute_relative
                absolute, relative = self._calculate_absolute_relative(ref_window, test_window)
                component_data['absolute_error'][i] = absolute
                component_data['relative_error'][i] = relative
        
        return time_data, component_data
    
    def _calculate_bias_precision(self, ref_window: np.ndarray, test_window: np.ndarray) -> Tuple[float, float]:
        """Calculate bias and precision error components"""
        differences = test_window - ref_window
        
        # Bias: mean difference
        bias = float(np.mean(differences))
        
        # Precision: standard deviation of differences
        precision = float(np.std(differences))
        
        return bias, precision
    
    def _calculate_systematic_random(self, ref_window: np.ndarray, test_window: np.ndarray) -> Tuple[float, float]:
        """Calculate systematic and random error components"""
        differences = test_window - ref_window
        
        # Systematic error: absolute mean difference
        systematic = float(abs(np.mean(differences)))
        
        # Random error: standard deviation after removing systematic bias
        random = float(np.std(differences - np.mean(differences)))
        
        return systematic, random
    
    def _calculate_absolute_relative(self, ref_window: np.ndarray, test_window: np.ndarray) -> Tuple[float, float]:
        """Calculate absolute and relative error components"""
        differences = test_window - ref_window
        
        # Absolute error: mean absolute difference
        absolute = float(np.mean(np.abs(differences)))
        
        # Relative error: mean absolute percentage error
        # Avoid division by zero
        mask = np.abs(ref_window) > 1e-10
        if np.any(mask):
            relative_errors = np.abs(differences[mask] / ref_window[mask]) * 100
            relative = float(np.mean(relative_errors))
        else:
            relative = 0.0
        
        return absolute, relative
    
    def _smooth_components(self, component_data: Dict[str, np.ndarray], 
                          smoothing_window: int) -> Dict[str, np.ndarray]:
        """Apply smoothing to error components"""
        smoothed_data = {}
        
        for component_name, component_values in component_data.items():
            # Apply uniform filter (moving average)
            smoothed_values = uniform_filter1d(component_values, size=smoothing_window)
            smoothed_data[component_name] = smoothed_values
        
        return smoothed_data
    
    def _remove_outliers(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """Remove outliers from the data"""
        outlier_method = params.get("outlier_method", "iqr")
        
        if outlier_method == "iqr":
            # IQR method
            iqr_factor = params.get("iqr_factor", 1.5)
            
            # Calculate IQR for both datasets
            ref_q25, ref_q75 = np.percentile(ref_data, [25, 75])
            ref_iqr = ref_q75 - ref_q25
            ref_lower = ref_q25 - iqr_factor * ref_iqr
            ref_upper = ref_q75 + iqr_factor * ref_iqr
            
            test_q25, test_q75 = np.percentile(test_data, [25, 75])
            test_iqr = test_q75 - test_q25
            test_lower = test_q25 - iqr_factor * test_iqr
            test_upper = test_q75 + iqr_factor * test_iqr
            
            mask = ((ref_data >= ref_lower) & (ref_data <= ref_upper) & 
                   (test_data >= test_lower) & (test_data <= test_upper))
            
        else:  # zscore
            # Z-score method
            z_threshold = params.get("z_threshold", 3.0)
            
            ref_z = np.abs(stats.zscore(ref_data))
            test_z = np.abs(stats.zscore(test_data))
            
            mask = (ref_z < z_threshold) & (test_z < z_threshold)
        
        return ref_data[mask], test_data[mask] 