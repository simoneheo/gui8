"""
Difference Waterfall Comparison Module

This module provides a comprehensive waterfall chart analysis showing signed errors
over time, helping visualize positive and negative deviations in a cumulative format.

Author: Auto-generated
Date: 2024
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison


@register_comparison
class DifferenceWaterfallComparison(BaseComparison):
    """
    Difference waterfall comparison method.
    
    This comparison creates a waterfall-style plot showing signed errors 
    over time, with bars colored by positive/negative deviation. This
    visualization helps identify systematic biases, drift patterns, and
    temporal variations in agreement.
    
    The analysis helps identify:
    - Systematic bias (consistent over/under estimation)
    - Temporal drift patterns
    - Periods of high/low agreement
    - Error magnitude distribution over time
    
    Statistical measures calculated:
    - Positive/negative error counts and percentages
    - Cumulative error progression
    - Error magnitude statistics by time segments
    - Bias trend analysis
    
    Attributes:
        name (str): Unique identifier for the comparison method
        description (str): Human-readable description
        category (str): Method category for organization
        version (str): Version identifier
        tags (List[str]): Descriptive tags for search/filtering
        plot_type (str): Type of plot generated ('bar')
        
    Example:
        >>> comparison = DifferenceWaterfallComparison(
        ...     segment_size=50,
        ...     show_cumulative=True
        ... )
        >>> results = comparison.apply(ref_data, test_data, ref_time)
        >>> stats = comparison.calculate_stats()
        >>> fig, ax = comparison.generate_plot()
    """
    
    name = "difference_waterfall"
    description = "Waterfall-style plot of signed error over time"
    category = "Visual"
    version = "1.0.0"
    tags = ["waterfall", "signed error", "bar", "drift", "temporal"]
    plot_type = "bar"

    # Rich parameter definitions following mixer/steps pattern
    params = [
        {
            "name": "segment_size",
            "type": "int",
            "default": 50,
            "help": "Number of samples per segment for aggregation",
            "tooltip": "Group data into segments for better visualization of trends",
            "advanced": False,
            "min_value": 1,
            "max_value": 1000
        },
        {
            "name": "show_cumulative",
            "type": "bool",
            "default": False,
            "help": "Show cumulative error progression",
            "tooltip": "Display cumulative sum of errors as overlay line",
            "advanced": False
        },
        {
            "name": "normalize_by_reference",
            "type": "bool",
            "default": False,
            "help": "Normalize errors by reference values",
            "tooltip": "Convert to percentage error for better comparison",
            "advanced": True
        },
        {
            "name": "exclude_outliers",
            "type": "bool",
            "default": False,
            "help": "Exclude outlier errors from display",
            "tooltip": "Remove extreme values that might skew visualization",
            "advanced": True
        },
        {
            "name": "outlier_threshold",
            "type": "float",
            "default": 3.0,
            "help": "Z-score threshold for outlier detection",
            "tooltip": "Standard deviations from mean for outlier exclusion",
            "advanced": True,
            "min_value": 1.0,
            "max_value": 5.0
        },
        {
            "name": "aggregation_method",
            "type": "choice",
            "default": "mean",
            "help": "Method for aggregating errors within segments",
            "tooltip": "How to combine multiple errors within each segment",
            "advanced": True,
            "choices": ["mean", "median", "sum", "max", "min", "std"]
        }
    ]

    # Rich overlay options for wizard controls
    overlay_options = {
        'color_by_sign': {
            'default': True,
            'label': 'Color by Sign',
            'tooltip': 'Color bars red (positive) or blue (negative)',
            'type': 'bool'
        },
        'show_zero_line': {
            'default': True,
            'label': 'Show Zero Line',
            'tooltip': 'Display horizontal line at zero error',
            'type': 'bool'
        },
        'show_trend_line': {
            'default': False,
            'label': 'Show Trend Line',
            'tooltip': 'Add linear trend line to identify systematic drift',
            'type': 'bool'
        },
        'bar_width': {
            'default': 0.8,
            'label': 'Bar Width',
            'tooltip': 'Width of waterfall bars (0.1 to 1.0)',
            'type': 'float',
            'min_value': 0.1,
            'max_value': 1.0
        },
        'transparency': {
            'default': 0.7,
            'label': 'Bar Transparency',
            'tooltip': 'Transparency of waterfall bars (0.1 to 1.0)',
            'type': 'float',
            'min_value': 0.1,
            'max_value': 1.0
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize the Difference Waterfall comparison.
        
        Args:
            **kwargs: Keyword arguments for parameter configuration
                - segment_size (int): Number of samples per segment
                - show_cumulative (bool): Show cumulative error progression
                - normalize_by_reference (bool): Use percentage errors
                - exclude_outliers (bool): Remove outlier errors
                - outlier_threshold (float): Z-score threshold for outliers
                - aggregation_method (str): Method for segment aggregation
        """
        super().__init__(**kwargs)
        self.results = None
        self.statistics = None
        self.plot_data = None
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        segment_size = self.kwargs.get("segment_size", 50)
        if not isinstance(segment_size, int) or segment_size <= 0:
            raise ValueError("segment_size must be a positive integer")
            
        outlier_threshold = self.kwargs.get("outlier_threshold", 3.0)
        if not isinstance(outlier_threshold, (int, float)) or outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be a positive number")
            
        aggregation_method = self.kwargs.get("aggregation_method", "mean")
        valid_methods = ["mean", "median", "sum", "max", "min", "std"]
        if aggregation_method not in valid_methods:
            raise ValueError(f"aggregation_method must be one of {valid_methods}")

    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Apply difference waterfall analysis to the data.
        
        Args:
            ref_data (np.ndarray): Reference data array
            test_data (np.ndarray): Test data array  
            ref_time (Optional[np.ndarray]): Reference time array
            test_time (Optional[np.ndarray]): Test time array (not used)
            
        Returns:
            Dict[str, Any]: Analysis results containing:
                - method: Method name
                - n_samples: Number of samples analyzed
                - plot_data: Data for plotting (time, error, segments)
                - error_statistics: Basic error statistics
                - temporal_analysis: Time-based analysis results
                
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If analysis fails
        """
        try:
            # Validate input data
            ref_data, test_data = self._validate_input_data(ref_data, test_data)
            
            # Calculate signed error
            error = self._calculate_error(ref_data, test_data)
            
            # Create time axis if not provided
            if ref_time is None:
                time_axis = np.arange(len(error))
            else:
                time_axis = ref_time
                
            # Handle outlier exclusion
            if self.kwargs.get("exclude_outliers", False):
                error, time_axis = self._remove_outliers(error, time_axis)
                
            if len(error) == 0:
                raise ValueError("No valid data points after outlier removal")
            
            # Segment data for aggregation
            segment_size = self.kwargs.get("segment_size", 50)
            segments = self._create_segments(error, time_axis, segment_size)
            
            # Calculate cumulative error if requested
            cumulative_error = np.cumsum(error) if self.kwargs.get("show_cumulative", False) else None
            
            # Store plot data
            self.plot_data = {
                "time": time_axis.tolist(),
                "error": error.tolist(),
                "segments": segments,
                "cumulative_error": cumulative_error.tolist() if cumulative_error is not None else None,
                "total_samples": len(error),
                "segment_size": segment_size
            }
            
            # Prepare results
            self.results = {
                "method": self.name,
                "n_samples": len(error),
                "plot_data": self.plot_data,
                "error_data": error.tolist(),
                "time_data": time_axis.tolist(),
                "parameters": dict(self.kwargs)
            }
            
            return self.results
            
        except Exception as e:
            raise RuntimeError(f"Difference waterfall analysis failed: {str(e)}")
            
    def _calculate_error(self, ref_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        """Calculate signed error based on configuration."""
        if self.kwargs.get("normalize_by_reference", False):
            # Percentage error (signed)
            with np.errstate(divide='ignore', invalid='ignore'):
                error = (test_data - ref_data) / np.abs(ref_data) * 100
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Absolute error (signed)
            error = test_data - ref_data
            
        return error
        
    def _remove_outliers(self, error: np.ndarray, time_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outlier errors based on Z-score threshold."""
        threshold = self.kwargs.get("outlier_threshold", 3.0)
        
        # Calculate Z-scores
        mean_error = np.mean(error)
        std_error = np.std(error)
        
        if std_error > 0:
            z_scores = np.abs((error - mean_error) / std_error)
            mask = z_scores <= threshold
            return error[mask], time_axis[mask]
        else:
            return error, time_axis
            
    def _create_segments(self, error: np.ndarray, time_axis: np.ndarray, segment_size: int) -> Dict[str, List]:
        """Create segments for aggregated visualization."""
        n_segments = int(np.ceil(len(error) / segment_size))
        aggregation_method = self.kwargs.get("aggregation_method", "mean")
        
        segments = {
            "segment_times": [],
            "segment_errors": [],
            "segment_indices": [],
            "segment_sizes": []
        }
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, len(error))
            
            segment_error = error[start_idx:end_idx]
            segment_time = time_axis[start_idx:end_idx]
            
            # Calculate aggregated error
            if aggregation_method == "mean":
                agg_error = np.mean(segment_error)
            elif aggregation_method == "median":
                agg_error = np.median(segment_error)
            elif aggregation_method == "sum":
                agg_error = np.sum(segment_error)
            elif aggregation_method == "max":
                agg_error = np.max(segment_error)
            elif aggregation_method == "min":
                agg_error = np.min(segment_error)
            elif aggregation_method == "std":
                agg_error = np.std(segment_error)
            else:
                agg_error = np.mean(segment_error)
                
            segments["segment_times"].append(np.mean(segment_time))
            segments["segment_errors"].append(agg_error)
            segments["segment_indices"].append(i)
            segments["segment_sizes"].append(len(segment_error))
            
        return segments

    def calculate_stats(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the waterfall analysis.
        
        Returns:
            Dict[str, Any]: Statistical measures including:
                - error_distribution: Error distribution statistics
                - bias_analysis: Bias and trend analysis
                - temporal_stats: Time-based statistics
                - segment_analysis: Segment-based statistics
                
        Raises:
            RuntimeError: If no results available or calculation fails
        """
        if self.results is None:
            raise RuntimeError("No results available. Run apply() first.")
            
        try:
            plot_data = self.plot_data
            error_data = np.array(plot_data["error"])
            
            # Error distribution statistics
            positive_errors = error_data[error_data > 0]
            negative_errors = error_data[error_data < 0]
            zero_errors = error_data[error_data == 0]
            
            error_distribution = {
                "total_samples": len(error_data),
                "positive_count": len(positive_errors),
                "negative_count": len(negative_errors),
                "zero_count": len(zero_errors),
                "positive_percentage": (len(positive_errors) / len(error_data)) * 100,
                "negative_percentage": (len(negative_errors) / len(error_data)) * 100,
                "zero_percentage": (len(zero_errors) / len(error_data)) * 100,
                "mean_error": np.mean(error_data),
                "median_error": np.median(error_data),
                "std_error": np.std(error_data),
                "min_error": np.min(error_data),
                "max_error": np.max(error_data)
            }
            
            # Bias analysis
            bias_analysis = {
                "systematic_bias": np.mean(error_data),
                "bias_magnitude": np.abs(np.mean(error_data)),
                "bias_direction": "positive" if np.mean(error_data) > 0 else "negative" if np.mean(error_data) < 0 else "neutral",
                "mean_absolute_error": np.mean(np.abs(error_data)),
                "root_mean_square_error": np.sqrt(np.mean(error_data**2))
            }
            
            # Temporal analysis (if enough data points)
            if len(error_data) > 2:
                # Linear trend analysis
                time_indices = np.arange(len(error_data))
                trend_coef = np.polyfit(time_indices, error_data, 1)[0]
                
                temporal_stats = {
                    "trend_slope": trend_coef,
                    "trend_direction": "increasing" if trend_coef > 0 else "decreasing" if trend_coef < 0 else "stable",
                    "trend_magnitude": abs(trend_coef),
                    "first_half_mean": np.mean(error_data[:len(error_data)//2]),
                    "second_half_mean": np.mean(error_data[len(error_data)//2:]),
                    "drift_magnitude": abs(np.mean(error_data[len(error_data)//2:]) - np.mean(error_data[:len(error_data)//2]))
                }
            else:
                temporal_stats = {
                    "trend_slope": 0.0,
                    "trend_direction": "insufficient_data",
                    "trend_magnitude": 0.0,
                    "first_half_mean": np.mean(error_data),
                    "second_half_mean": np.mean(error_data),
                    "drift_magnitude": 0.0
                }
                
            # Segment analysis
            if plot_data["segments"]:
                segment_errors = np.array(plot_data["segments"]["segment_errors"])
                segment_analysis = {
                    "n_segments": len(segment_errors),
                    "segment_mean_error": np.mean(segment_errors),
                    "segment_std_error": np.std(segment_errors),
                    "most_positive_segment": np.argmax(segment_errors),
                    "most_negative_segment": np.argmin(segment_errors),
                    "max_segment_error": np.max(segment_errors),
                    "min_segment_error": np.min(segment_errors),
                    "segment_range": np.max(segment_errors) - np.min(segment_errors)
                }
            else:
                segment_analysis = {"n_segments": 0}
            
            self.statistics = {
                "error_distribution": error_distribution,
                "bias_analysis": bias_analysis,
                "temporal_stats": temporal_stats,
                "segment_analysis": segment_analysis
            }
            
            return self.statistics
            
        except Exception as e:
            raise RuntimeError(f"Statistics calculation failed: {str(e)}")

    def generate_plot(self, fig: Optional[plt.Figure] = None, 
                     ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate difference waterfall plot.
        
        Args:
            fig (Optional[plt.Figure]): Existing figure to use
            ax (Optional[plt.Axes]): Existing axes to use
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: Figure and axes objects
            
        Raises:
            RuntimeError: If no results available or plotting fails
        """
        if self.results is None:
            raise RuntimeError("No results available. Run apply() first.")
            
        try:
            # Create figure if not provided
            if fig is None or ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
            
            plot_data = self.plot_data
            
            # Get overlay options
            color_by_sign = self.kwargs.get('color_by_sign', True)
            show_zero_line = self.kwargs.get('show_zero_line', True)
            show_trend_line = self.kwargs.get('show_trend_line', False)
            bar_width = self.kwargs.get('bar_width', 0.8)
            transparency = self.kwargs.get('transparency', 0.7)
            
            # Prepare data
            time_data = np.array(plot_data["time"])
            error_data = np.array(plot_data["error"])
            
            # Use segments if available and segment_size > 1
            if plot_data["segments"] and plot_data["segment_size"] > 1:
                x_data = plot_data["segments"]["segment_times"]
                y_data = plot_data["segments"]["segment_errors"]
                bar_width = bar_width * plot_data["segment_size"]
            else:
                x_data = time_data
                y_data = error_data
            
            # Color bars by sign
            if color_by_sign:
                colors = ['red' if y > 0 else 'blue' if y < 0 else 'gray' for y in y_data]
            else:
                colors = 'steelblue'
            
            # Create waterfall bars
            bars = ax.bar(x_data, y_data, width=bar_width, color=colors, alpha=transparency, edgecolor='black', linewidth=0.5)
            
            # Add zero line
            if show_zero_line:
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
            
            # Add trend line
            if show_trend_line and len(x_data) > 2:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(x_data, p(x_data), 'r--', alpha=0.8, linewidth=2, label='Trend Line')
                ax.legend()
            
            # Add cumulative error line if available
            if plot_data["cumulative_error"] is not None:
                ax2 = ax.twinx()
                ax2.plot(time_data, plot_data["cumulative_error"], 'g-', alpha=0.7, linewidth=2, label='Cumulative Error')
                ax2.set_ylabel('Cumulative Error', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.legend(loc='upper right')
            
            # Styling
            unit = "%" if self.kwargs.get("normalize_by_reference", False) else ""
            ax.set_xlabel('Time' if plot_data["time"][0] != 0 else 'Sample Index')
            ax.set_ylabel(f'Error {unit}')
            ax.set_title(f'Difference Waterfall Analysis\n(n={plot_data["total_samples"]} samples)')
            ax.grid(True, alpha=0.3)
            
            # Add color legend if using sign-based coloring
            if color_by_sign:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='red', alpha=transparency, label='Positive Error'),
                    Patch(facecolor='blue', alpha=transparency, label='Negative Error')
                ]
                if any(y == 0 for y in y_data):
                    legend_elements.append(Patch(facecolor='gray', alpha=transparency, label='Zero Error'))
                ax.legend(handles=legend_elements, loc='upper left')
            
            plt.tight_layout()
            return fig, ax
            
        except Exception as e:
            raise RuntimeError(f"Plot generation failed: {str(e)}")

    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about this comparison method.
        
        Returns:
            Dict[str, Any]: Method information including parameters, description, and capabilities
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "tags": self.tags,
            "plot_type": self.plot_type,
            "parameters": self.params,
            "overlay_options": self.overlay_options,
            "capabilities": {
                "handles_time_series": True,
                "handles_missing_data": True,
                "statistical_analysis": True,
                "visual_analysis": True,
                "batch_processing": True
            },
            "output_types": ["waterfall_chart", "statistics", "temporal_analysis"]
        }