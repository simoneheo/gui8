"""
Error Histogram Comparison Method

This module implements error histogram analysis for examining the distribution
of errors between two methods. Shows frequency distribution of differences.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class ErrorHistogramComparison(BaseComparison):
    """
    Error histogram analysis comparison method.
    
    Implements histogram analysis of errors between two methods to examine
    the distribution and characteristics of differences.
    """
    
    name = "error_histogram"
    description = "Histogram analysis of errors between two methods"
    category = "Statistical"
    tags = ["histogram", "error", "distribution"]
    
    # Parameters following mixer/steps pattern
    params = [
        {
            "name": "error_type", 
            "type": "str", 
            "default": "simple", 
            "options": ["simple", "percentage", "normalized", "absolute"], 
            "help": "Type of error calculation: simple (test-ref), percentage ((test-ref)/ref*100), normalized ((test-ref)/std(ref)), absolute (|test-ref|)"
        },
        {
            "name": "bins", 
            "type": "int", 
            "default": 30, 
            "help": "Number of histogram bins"
        },
        {
            "name": "range_percentiles", 
            "type": "bool", 
            "default": True, 
            "help": "Use 1st-99th percentiles for range instead of min-max"
        },
        {
            "name": "density", 
            "type": "bool", 
            "default": False, 
            "help": "Normalize histogram to show density instead of counts"
        },    
    ]
    
    # Plot configuration
    plot_type = "histogram"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'normal_fit': {
            'default': True, 
            'label': 'Normal Distribution Fit', 
            'tooltip': 'Overlay fitted normal distribution curve', 
            'type': 'line'
        },
        'std_bands': {
            'default': True, 
            'label': 'Standard Deviation Bands', 
            'tooltip': 'Show vertical lines at ±1σ, ±2σ, ±3σ from mean', 
            'type': 'line'
        },
        'mean_line': {
            'default': True, 
            'label': 'Mean Error Line', 
            'tooltip': 'Show vertical line at mean error', 
            'type': 'line'
        },
        'median_line': {
            'default': True, 
            'label': 'Median Error Line', 
            'tooltip': 'Show vertical line at median error', 
            'type': 'line'
        },
        'statistical_results': {
            'default': True, 
            'label': 'Statistical Results', 
            'tooltip': 'Display statistical results on the plot', 
            'type': 'text'
        }
    }
    
    def __init__(self, error_type="simple", bins=30, range_percentiles=True, density=False,
                 show_distribution_fit=True, show_std_bands=True, show_mean_line=True, 
                 show_median_line=False, **kwargs):
        """
        Initialize Error Histogram comparison.
        
        Args:
            error_type: Type of error calculation
            bins: Number of histogram bins
            range_percentiles: Use percentiles for range
            density: Show density instead of counts
            show_distribution_fit: Overlay normal distribution fit
            show_std_bands: Show standard deviation bands
            show_mean_line: Show mean error line
            show_median_line: Show median error line
            **kwargs: Additional keyword arguments passed to parent class
        """
        # Call parent constructor to initialize self.kwargs
        super().__init__(**kwargs)
        
        self.error_type = error_type
        self.bins = bins
        self.range_percentiles = range_percentiles
        self.density = density
    
    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate histogram data for error distribution.
        
        Args:
            ref_data: Reference method data
            test_data: Test method data  
            params: Analysis parameters
            
        Returns:
            Tuple of (bin_centers, counts, metadata)
        """
        # Calculate errors based on error_type
        errors = self._calculate_errors(ref_data, test_data, params.get('error_type', 'simple'))
        
        # Remove invalid values
        errors = errors[np.isfinite(errors)]
        
        if len(errors) == 0:
            return np.array([]), np.array([]), {}
        
        # Determine histogram range
        if params.get('range_percentiles', True):
            range_min = np.percentile(errors, 1)
            range_max = np.percentile(errors, 99)
        else:
            range_min = np.min(errors)
            range_max = np.max(errors)
        
        # Create histogram
        bin_count = params.get('bins', 30)
        counts, bin_edges = np.histogram(errors, bins=bin_count, range=(float(range_min), float(range_max)), 
                                       density=params.get('density', False))
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]
        
        # Generate dynamic labels based on parameters
        error_type = params.get('error_type', 'simple')
        histogram_type = 'density' if params.get('density', False) else 'counts'
        
        # X-axis label based on error type
        x_label_map = {
            'simple': 'Error (Test - Reference)',
            'percentage': 'Percentage Error (%)',
            'normalized': 'Normalized Error',
            'absolute': 'Absolute Error'
        }
        x_label = x_label_map.get(error_type, 'Error')
        
        # Y-axis label based on histogram type
        y_label_map = {
            'counts': 'Frequency',
            'density': 'Density', 
            'probability': 'Probability'
        }
        y_label = y_label_map.get(histogram_type, 'Frequency')
        
        # Title based on error type
        title_map = {
            'simple': 'Error Distribution',
            'percentage': 'Percentage Error Distribution',
            'normalized': 'Normalized Error Distribution', 
            'absolute': 'Absolute Error Distribution'
        }
        title = title_map.get(error_type, 'Error Distribution')
        
        # Store metadata for overlays and styling
        metadata = {
            'x_label': x_label,
            'y_label': y_label,
            'title': title,
            'histogram_type': histogram_type,
            'bin_edges': bin_edges.tolist(),
            'bin_widths': bin_widths.tolist(),
            'error_data': errors,
            'range_min': range_min,
            'range_max': range_max,
            'error_type': error_type,
            'sample_size': len(errors)
        }
        
        return errors, np.array([]), metadata
           
    
    def stats_script(self, x_data: np.ndarray, y_data: np.ndarray, ref_data: np.ndarray, 
                    test_data: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistical measures for error histogram.
        
        Args:
            x_data: Bin centers
            y_data: Counts/density values
            ref_data: Reference method data
            test_data: Test method data
            params: Analysis parameters
            
        Returns:
            Dictionary of statistical results
        """
        # Calculate errors
        errors = self._calculate_errors(ref_data, test_data, params.get('error_type', 'simple'))
        errors = errors[np.isfinite(errors)]
        
        if len(errors) == 0:
            return {}
        
        # Basic statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors, ddof=1)
        
        # Distribution statistics
        skewness = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)
        
        # Normality test
        if len(errors) > 8:  # Minimum sample size for Shapiro-Wilk
            shapiro_stat, shapiro_p = stats.shapiro(errors)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # Percentiles
        percentiles = np.percentile(errors, [5, 25, 50, 75, 95])
        
        # Distribution fit parameters (for overlays)
        fit_params = {
            'mean': mean_error,
            'std': std_error,
            'range_min': np.min(errors),
            'range_max': np.max(errors)
        }
        
        return {
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'percentiles': {
                'p5': percentiles[0],
                'p25': percentiles[1],
                'p50': percentiles[2],
                'p75': percentiles[3],
                'p95': percentiles[4]
            },
            'fit_params': fit_params,
            'n_points': len(errors),
            'error_type': params.get('error_type', 'simple')
        }
    

    
    def _create_overlays(self, ref_data: np.ndarray, test_data: np.ndarray, 
                        stats_results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Create overlay definitions for error histogram analysis.
        
        Returns a dictionary of overlay definitions that will be rendered by the base class.
        Each overlay definition contains type, main data, and style information.
        
        Args:
            ref_data: Reference data arrays
            test_data: Test data array
            stats_results: Statistical results from stats_script method
            params: Overlay configuration dictionary
            
        Returns:
            Dictionary of overlay definitions
        """
        
        mean_line = {
            'type': 'vline',
            'show': params.get('mean_line', True),
            'label': 'Mean Error Line',
            'main': self._get_mean_line(stats_results)
        }
        
        median_line = {
            'type': 'vline',
            'show': params.get('median_line', True),
            'label': 'Median Error Line',
            'main': self._get_median_line(stats_results)
        }
        
        std_bands = {
            'type': 'vline',  # Changed from 'fill' to 'vline' for multiple vertical lines
            'show': params.get('std_bands', True),
            'label': 'Standard Deviation Bands',
            'main': self._get_std_bands(stats_results)
        }
        
        normal_fit = {
            'type': 'line',
            'show': params.get('normal_fit', True),
            'label': 'Normal Distribution Fit',
            'main': self._get_normal_fit_curve(stats_results)
        }
        
        statistical_results = {
            'type': 'text',
            'show': params.get('statistical_results', True),
            'label': 'Statistical Results',
            'main': self._get_statistical_results(stats_results)
        }
        
        overlays = {
            'mean_line': mean_line,
            'median_line': median_line,
            'std_bands': std_bands,
            'normal_fit': normal_fit,
            'statistical_results': statistical_results
        }
        
        return overlays
    
    def _calculate_errors(self, ref_data: np.ndarray, test_data: np.ndarray, error_type: str) -> np.ndarray:
        """
        Calculate errors between reference and test data.
        
        Args:
            ref_data: Reference method data
            test_data: Test method data
            error_type: Type of error calculation
            
        Returns:
            Array of error values
        """
        if error_type == "simple":
            return test_data - ref_data
        elif error_type == "percentage":
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                return ((test_data - ref_data) / ref_data) * 100
        elif error_type == "normalized":
            ref_std = np.std(ref_data, ddof=1)
            if ref_std == 0:
                return np.zeros_like(test_data)
            return (test_data - ref_data) / ref_std
        elif error_type == "absolute":
            return np.abs(test_data - ref_data)
        else:
            # Default to simple difference
            return test_data - ref_data
    
    def _get_normal_fit_curve(self, stats_results):
        """Get data for normal distribution fit curve overlay."""
        normal_fit_params = stats_results.get('normal_fit_params', (0, 1))
        return {
            'distribution': 'norm',
            'params': normal_fit_params,
            'color': 'red',
            'linestyle': '--',
            'alpha': 0.8
        }

    def _get_std_bands(self, stats_results):
        """Get data for standard deviation bands overlay."""
        mean_error = stats_results.get('mean_error', 0)
        std_error = stats_results.get('std_error', 1)
        
        # Return x values as a list for vline overlay
        return {
            'x': [
                mean_error - 3*std_error,  # -3σ
                mean_error - 2*std_error,  # -2σ
                mean_error - 1*std_error,  # -1σ
                mean_error + 1*std_error,  # +1σ
                mean_error + 2*std_error,  # +2σ
                mean_error + 3*std_error   # +3σ
            ],
            'color': 'red',
            'linestyle': ':',
            'linewidth': 1,
            'alpha': 0.5
        }

    def _get_mean_line(self, stats_results):
        """Get data for mean error line overlay."""
        mean_error = stats_results.get('mean_error', 0)
        return {
            'x': [mean_error],  # Use list format for vline consistency
            'color': 'blue',
            'linestyle': '-',
            'linewidth': 2
        }
    
    def _get_median_line(self, stats_results):
        """Get data for median error line overlay."""
        median_error = stats_results.get('median_error', 0)
        return {
            'x': [median_error],  # Use list format for vline consistency
            'color': 'green',
            'linestyle': '-',
            'linewidth': 2
        }

    def _get_statistical_results(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for statistical results text overlay."""
        # Only return the most informative statistics for text overlay
        essential_stats = {
            'mean_error': stats_results.get('mean_error'),
            'std_error': stats_results.get('std_error'),
            'median_error': stats_results.get('median_error'),
            'skewness': stats_results.get('skewness'),
            'sample_size': stats_results.get('sample_size')
        }
        return essential_stats
    
    @classmethod
    def get_description(cls) -> str:
        """
        Get a description of this comparison method for display in the wizard console.
        
        Returns:
            String description explaining what this comparison method does
        """
        return """Error Histogram Analysis: Examines the distribution of errors between two methods.

• Simple: Test - Reference (raw differences)
• Percentage: ((Test - Reference) / Reference) × 100
• Normalized: (Test - Reference) / std(Reference)
• Absolute: |Test - Reference| 
""" 