"""
Stacked Error Time Band Comparison Module

This module provides a comprehensive stacked area chart analysis showing the proportion 
of different error tiers over time, helping visualize error distribution patterns.

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
class StackedErrorTimeBandComparison(BaseComparison):
    """
    Stacked error time band comparison method.
    
    This comparison creates a stacked area plot showing the proportion of
    different error tiers over time segments, providing insights into how
    error distribution changes temporally.
    
    The analysis helps identify:
    - Temporal patterns in error distribution
    - Periods of high/low error concentration
    - Error tier stability over time
    - Seasonal or cyclical error patterns
    
    Statistical measures calculated:
    - Error tier percentages by time segment
    - Temporal stability metrics
    - Error concentration indices
    - Trend analysis for each error tier
    
    Attributes:
        name (str): Unique identifier for the comparison method
        description (str): Human-readable description
        category (str): Method category for organization
        version (str): Version identifier
        tags (List[str]): Descriptive tags for search/filtering
        plot_type (str): Type of plot generated ('stacked_area')
        
    Example:
        >>> comparison = StackedErrorTimeBandComparison(
        ...     segment_size=100,
        ...     error_thresholds="5,10,20"
        ... )
        >>> results = comparison.apply(ref_data, test_data, ref_time)
        >>> stats = comparison.calculate_stats()
        >>> fig, ax = comparison.generate_plot()
    """
    
    name = "stacked_error_time_band"
    description = "Stacked area plot showing proportion of error tiers over time"
    category = "Visual"
    version = "1.0.0"
    tags = ["stacked", "error", "time", "band", "temporal", "distribution"]
    plot_type = "stacked_area"

    # Rich parameter definitions following mixer/steps pattern
    params = [
        {
            "name": "segment_size",
            "type": "int",
            "default": 100,
            "help": "Number of samples per time segment",
            "tooltip": "Size of each time segment for error analysis",
            "advanced": False,
            "min_value": 10,
            "max_value": 1000
        },
        {
            "name": "error_thresholds",
            "type": "str",
            "default": "5,10,20",
            "help": "Comma-separated percentage thresholds for error tiers",
            "tooltip": "Define error bands: '5,10,20' creates tiers [0-5], (5-10], (10-20], >20%",
            "advanced": False,
            "validation": {
                "pattern": r"^[\d\.,\s]+$",
                "message": "Must be comma-separated numbers"
            }
        },
        {
            "name": "use_percentage",
            "type": "bool",
            "default": True,
            "help": "Use percentage error instead of absolute error",
            "tooltip": "Calculate error as percentage of reference value",
            "advanced": False
        },
        {
            "name": "normalize_by_reference",
            "type": "bool",
            "default": True,
            "help": "Normalize error by reference value (for percentage mode)",
            "tooltip": "Controls whether percentage error uses reference or test value",
            "advanced": True
        },
        {
            "name": "overlap_segments",
            "type": "bool",
            "default": False,
            "help": "Use overlapping time segments",
            "tooltip": "Create overlapping segments for smoother temporal analysis",
            "advanced": True
        },
        {
            "name": "overlap_ratio",
            "type": "float",
            "default": 0.5,
            "help": "Overlap ratio for segments (0-1)",
            "tooltip": "Fraction of segment size for overlap (0.5 = 50% overlap)",
            "advanced": True,
            "min_value": 0.1,
            "max_value": 0.9
        },
        {
            "name": "exclude_zeros",
            "type": "bool",
            "default": False,
            "help": "Exclude zero reference values from analysis",
            "tooltip": "Prevents division by zero in percentage calculations",
            "advanced": True
        }
    ]

    # Rich overlay options for wizard controls
    overlay_options = {
        'show_legend': {
            'default': True,
            'label': 'Show Legend',
            'tooltip': 'Display error band labels',
            'type': 'bool'
        },
        'show_total_line': {
            'default': False,
            'label': 'Show Total Error Line',
            'tooltip': 'Add line showing total error per segment',
            'type': 'bool'
        },
        'color_scheme': {
            'default': 'viridis',
            'label': 'Color Scheme',
            'tooltip': 'Color scheme for error bands',
            'type': 'choice',
            'choices': ['viridis', 'plasma', 'inferno', 'magma', 'cool', 'warm', 'traffic_light']
        },
        'transparency': {
            'default': 0.7,
            'label': 'Band Transparency',
            'tooltip': 'Transparency of stacked areas (0.1 to 1.0)',
            'type': 'float',
            'min_value': 0.1,
            'max_value': 1.0
        },
        'smooth_bands': {
            'default': False,
            'label': 'Smooth Bands',
            'tooltip': 'Apply smoothing to stacked areas',
            'type': 'bool'
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize the Stacked Error Time Band comparison.
        
        Args:
            **kwargs: Keyword arguments for parameter configuration
                - segment_size (int): Number of samples per segment
                - error_thresholds (str): Comma-separated error thresholds
                - use_percentage (bool): Use percentage error calculation
                - normalize_by_reference (bool): Normalize by reference values
                - overlap_segments (bool): Use overlapping segments
                - overlap_ratio (float): Overlap ratio for segments
                - exclude_zeros (bool): Exclude zero reference values
        """
        super().__init__(**kwargs)
        self.results = None
        self.statistics = None
        self.plot_data = None
        
        # Validate and parse parameters
        self._validate_parameters()
        self._parse_error_thresholds()
        
    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        segment_size = self.kwargs.get("segment_size", 100)
        if not isinstance(segment_size, int) or segment_size <= 0:
            raise ValueError("segment_size must be a positive integer")
            
        # Validate error_thresholds format
        error_thresholds_str = self.kwargs.get("error_thresholds", "5,10,20")
        try:
            thresholds = [float(x.strip()) for x in error_thresholds_str.split(",")]
            if len(thresholds) == 0:
                raise ValueError("At least one error threshold required")
            if any(t <= 0 for t in thresholds):
                raise ValueError("Error thresholds must be positive")
        except ValueError as e:
            raise ValueError(f"Invalid error_thresholds format: {e}")
            
        overlap_ratio = self.kwargs.get("overlap_ratio", 0.5)
        if not isinstance(overlap_ratio, (int, float)) or not (0 <= overlap_ratio < 1):
            raise ValueError("overlap_ratio must be a number between 0 and 1")
            
    def _parse_error_thresholds(self) -> List[float]:
        """Parse and validate error thresholds."""
        error_thresholds_str = self.kwargs.get("error_thresholds", "5,10,20")
        try:
            thresholds = [float(x.strip()) for x in error_thresholds_str.split(",")]
            thresholds = sorted(thresholds)  # Ensure ascending order
            self.thresholds = [0] + thresholds + [np.inf]
            return thresholds
        except ValueError as e:
            raise ValueError(f"Error parsing error_thresholds: {e}")

    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Apply stacked error time band analysis to the data.
        
        Args:
            ref_data (np.ndarray): Reference data array
            test_data (np.ndarray): Test data array  
            ref_time (Optional[np.ndarray]): Reference time array
            test_time (Optional[np.ndarray]): Test time array (not used)
            
        Returns:
            Dict[str, Any]: Analysis results containing:
                - method: Method name
                - n_samples: Number of samples analyzed
                - plot_data: Data for plotting (segments, labels, percent_matrix)
                - temporal_analysis: Time-based analysis results
                - error_statistics: Error distribution statistics
                
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If analysis fails
        """
        try:
            # Validate input data
            ref_data, test_data = self._validate_input_data(ref_data, test_data)
            
            # Handle zero exclusion
            if self.kwargs.get("exclude_zeros", False):
                mask = ref_data != 0
                ref_data = ref_data[mask]
                test_data = test_data[mask]
                if ref_time is not None:
                    ref_time = ref_time[mask]
                    
            if len(ref_data) == 0:
                raise ValueError("No valid data points after filtering")
            
            # Calculate error
            error = self._calculate_error(ref_data, test_data)
            
            # Create time segments
            segment_size = self.kwargs.get("segment_size", 100)
            segments = self._create_time_segments(error, ref_time, segment_size)
            
            # Generate tier labels
            tier_labels = self._generate_tier_labels()
            
            # Calculate error tier matrix
            band_matrix = self._calculate_band_matrix(segments, tier_labels)
            
            # Create time axis for segments
            segment_times = self._create_segment_times(segments, ref_time)
            
            # Store plot data
            self.plot_data = {
                "segments": list(range(len(segments))),
                "segment_times": segment_times,
                "labels": tier_labels,
                "percent_matrix": band_matrix.tolist(),
                "n_segments": len(segments),
                "total_samples": len(error),
                "segment_size": segment_size
            }
            
            # Prepare results
            self.results = {
                "method": self.name,
                "n_samples": len(error),
                "plot_data": self.plot_data,
                "error_data": error.tolist(),
                "thresholds": self.thresholds,
                "parameters": dict(self.kwargs)
            }
            
            return self.results
            
        except Exception as e:
            raise RuntimeError(f"Stacked error time band analysis failed: {str(e)}")
            
    def _calculate_error(self, ref_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        """Calculate error based on configuration."""
        if self.kwargs.get("use_percentage", True):
            # Percentage error (absolute)
            denominator = ref_data if self.kwargs.get("normalize_by_reference", True) else test_data
            with np.errstate(divide='ignore', invalid='ignore'):
                error = np.abs(test_data - ref_data) / np.abs(denominator) * 100
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Absolute error
            error = np.abs(test_data - ref_data)
            
        return error
        
    def _create_time_segments(self, error: np.ndarray, ref_time: Optional[np.ndarray], 
                            segment_size: int) -> List[np.ndarray]:
        """Create time segments for analysis."""
        segments = []
        
        if self.kwargs.get("overlap_segments", False):
            # Overlapping segments
            overlap_ratio = self.kwargs.get("overlap_ratio", 0.5)
            step_size = int(segment_size * (1 - overlap_ratio))
            
            for i in range(0, len(error) - segment_size + 1, step_size):
                segments.append(error[i:i + segment_size])
        else:
            # Non-overlapping segments
            for i in range(0, len(error), segment_size):
                segment = error[i:i + segment_size]
                if len(segment) > 0:  # Include partial segments
                    segments.append(segment)
                    
        return segments
        
    def _generate_tier_labels(self) -> List[str]:
        """Generate tier labels from thresholds."""
        unit = "%" if self.kwargs.get("use_percentage", True) else ""
        labels = []
        
        for i in range(len(self.thresholds) - 1):
            if self.thresholds[i+1] == np.inf:
                labels.append(f">{self.thresholds[i]:.0f}{unit}")
            else:
                labels.append(f"≤{self.thresholds[i+1]:.0f}{unit}")
                
        return labels
        
    def _calculate_band_matrix(self, segments: List[np.ndarray], tier_labels: List[str]) -> np.ndarray:
        """Calculate the percentage matrix for stacked bands."""
        n_tiers = len(tier_labels)
        n_segments = len(segments)
        band_matrix = np.zeros((n_tiers, n_segments))
        
        for i, segment in enumerate(segments):
            if len(segment) > 0:
                # Calculate histogram for this segment
                bin_counts, _ = np.histogram(segment, bins=self.thresholds)
                total = len(segment)
                
                # Convert to percentages
                if total > 0:
                    band_matrix[:, i] = (bin_counts / total) * 100
                    
        return band_matrix
        
    def _create_segment_times(self, segments: List[np.ndarray], ref_time: Optional[np.ndarray]) -> List[float]:
        """Create time axis for segments."""
        if ref_time is None:
            # Use segment indices as time
            return list(range(len(segments)))
        else:
            # Use actual time values
            segment_times = []
            segment_size = self.kwargs.get("segment_size", 100)
            
            if self.kwargs.get("overlap_segments", False):
                overlap_ratio = self.kwargs.get("overlap_ratio", 0.5)
                step_size = int(segment_size * (1 - overlap_ratio))
                
                for i in range(0, len(ref_time) - segment_size + 1, step_size):
                    segment_times.append(np.mean(ref_time[i:i + segment_size]))
            else:
                for i in range(0, len(ref_time), segment_size):
                    end_idx = min(i + segment_size, len(ref_time))
                    segment_times.append(np.mean(ref_time[i:end_idx]))
                    
            return segment_times

    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the stacked error time band analysis.
        
        Args:
            ref_data (np.ndarray): Reference data array
            test_data (np.ndarray): Test data array
            ref_time (Optional[np.ndarray]): Reference time array
            test_time (Optional[np.ndarray]): Test time array (not used)
        
        Returns:
            Dict[str, Any]: Statistical measures including:
                - tier_statistics: Statistics for each error tier
                - temporal_stability: Temporal stability metrics
                - concentration_indices: Error concentration measures
                - trend_analysis: Trend analysis for each tier
                
        Raises:
            RuntimeError: If no results available or calculation fails
        """
        # If no results available, run apply() first
        if self.results is None:
            try:
                self.apply(ref_data, test_data, ref_time, test_time)
            except Exception as e:
                raise RuntimeError(f"Failed to apply analysis before calculating stats: {str(e)}")
            
        try:
            plot_data = self.plot_data
            percent_matrix = np.array(plot_data["percent_matrix"])
            tier_labels = plot_data["labels"]
            n_segments = plot_data["n_segments"]
            
            # Per-tier statistics
            tier_statistics = {}
            for i, label in enumerate(tier_labels):
                tier_data = percent_matrix[i, :]
                tier_statistics[label] = {
                    "mean_percentage": np.mean(tier_data),
                    "std_percentage": np.std(tier_data),
                    "min_percentage": np.min(tier_data),
                    "max_percentage": np.max(tier_data),
                    "median_percentage": np.median(tier_data),
                    "range_percentage": np.max(tier_data) - np.min(tier_data)
                }
            
            # Temporal stability metrics
            temporal_stability = {}
            for i, label in enumerate(tier_labels):
                tier_data = percent_matrix[i, :]
                if len(tier_data) > 1:
                    # Calculate coefficient of variation
                    cv = np.std(tier_data) / np.mean(tier_data) if np.mean(tier_data) > 0 else 0
                    temporal_stability[label] = {
                        "coefficient_of_variation": cv,
                        "stability_score": 1 / (1 + cv),  # Higher score = more stable
                        "temporal_range": np.max(tier_data) - np.min(tier_data)
                    }
                else:
                    temporal_stability[label] = {
                        "coefficient_of_variation": 0,
                        "stability_score": 1,
                        "temporal_range": 0
                    }
            
            # Error concentration indices
            concentration_indices = {}
            for i in range(n_segments):
                segment_data = percent_matrix[:, i]
                # Calculate entropy (lower entropy = more concentrated)
                non_zero_probs = segment_data[segment_data > 0] / 100  # Convert to probabilities
                if len(non_zero_probs) > 0:
                    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs + 1e-10))
                    concentration_indices[f"segment_{i}"] = {
                        "entropy": entropy,
                        "concentration_score": 1 / (1 + entropy),  # Higher score = more concentrated
                        "dominant_tier": tier_labels[np.argmax(segment_data)],
                        "dominant_percentage": np.max(segment_data)
                    }
                else:
                    concentration_indices[f"segment_{i}"] = {
                        "entropy": 0,
                        "concentration_score": 1,
                        "dominant_tier": tier_labels[0],
                        "dominant_percentage": 0
                    }
            
            # Trend analysis for each tier
            trend_analysis = {}
            if n_segments > 2:
                time_indices = np.arange(n_segments)
                for i, label in enumerate(tier_labels):
                    tier_data = percent_matrix[i, :]
                    trend_coef = np.polyfit(time_indices, tier_data, 1)[0]
                    
                    trend_analysis[label] = {
                        "trend_slope": trend_coef,
                        "trend_direction": "increasing" if trend_coef > 0 else "decreasing" if trend_coef < 0 else "stable",
                        "trend_magnitude": abs(trend_coef),
                        "first_half_mean": np.mean(tier_data[:n_segments//2]),
                        "second_half_mean": np.mean(tier_data[n_segments//2:]),
                        "change_magnitude": abs(np.mean(tier_data[n_segments//2:]) - np.mean(tier_data[:n_segments//2]))
                    }
            else:
                for label in tier_labels:
                    trend_analysis[label] = {
                        "trend_slope": 0,
                        "trend_direction": "insufficient_data",
                        "trend_magnitude": 0,
                        "first_half_mean": 0,
                        "second_half_mean": 0,
                        "change_magnitude": 0
                    }
            
            # Overall statistics
            overall_stats = {
                "total_segments": n_segments,
                "total_samples": plot_data["total_samples"],
                "segment_size": plot_data["segment_size"],
                "most_stable_tier": min(temporal_stability.keys(), key=lambda x: temporal_stability[x]["coefficient_of_variation"]),
                "most_variable_tier": max(temporal_stability.keys(), key=lambda x: temporal_stability[x]["coefficient_of_variation"]),
                "average_entropy": np.mean([concentration_indices[f"segment_{i}"]["entropy"] for i in range(n_segments)]),
                "average_concentration": np.mean([concentration_indices[f"segment_{i}"]["concentration_score"] for i in range(n_segments)])
            }
            
            self.statistics = {
                "tier_statistics": tier_statistics,
                "temporal_stability": temporal_stability,
                "concentration_indices": concentration_indices,
                "trend_analysis": trend_analysis,
                "overall_stats": overall_stats
            }
            
            return self.statistics
            
        except Exception as e:
            raise RuntimeError(f"Statistics calculation failed: {str(e)}")

    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate stacked error time band plot.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data (np.ndarray): Reference data array
            test_data (np.ndarray): Test data array
            plot_config (Dict[str, Any]): Plot configuration dictionary
            stats_results (Dict[str, Any]): Statistical results from calculate_stats method
            
        Raises:
            RuntimeError: If no results available or plotting fails
        """
        # If no results available, run apply() first
        if self.results is None:
            try:
                self.apply(ref_data, test_data)
            except Exception as e:
                raise RuntimeError(f"Failed to apply analysis before plotting: {str(e)}")
            
        try:
            plot_data = self.plot_data
            
            # Merge plot_config with overlay options if provided
            if plot_config is None:
                plot_config = {}
            
            # Get overlay options (prefer plot_config over kwargs)
            show_legend = plot_config.get('show_legend', self.kwargs.get('show_legend', True))
            show_total_line = plot_config.get('show_total_line', self.kwargs.get('show_total_line', False))
            color_scheme = plot_config.get('color_scheme', self.kwargs.get('color_scheme', 'viridis'))
            transparency = plot_config.get('transparency', self.kwargs.get('transparency', 0.7))
            smooth_bands = plot_config.get('smooth_bands', self.kwargs.get('smooth_bands', False))
            
            # Prepare data
            x_data = plot_data["segment_times"]
            percent_matrix = np.array(plot_data["percent_matrix"])
            tier_labels = plot_data["labels"]
            
            # Get colors
            colors = self._get_color_scheme(color_scheme, len(tier_labels))
            
            # Apply smoothing if requested
            if smooth_bands and len(x_data) > 3:
                try:
                    from scipy.ndimage import gaussian_filter1d
                    for i in range(len(tier_labels)):
                        percent_matrix[i, :] = gaussian_filter1d(percent_matrix[i, :], sigma=1)
                except ImportError:
                    print("[Warning] scipy not available, skipping smoothing")
            
            # Create stacked area plot
            ax.stackplot(x_data, *percent_matrix, labels=tier_labels, colors=colors, alpha=transparency)
            
            # Add total error line if requested
            if show_total_line:
                total_error = np.sum(percent_matrix, axis=0)
                ax.plot(x_data, total_error, 'k-', linewidth=2, label='Total Error')
            
            # Styling
            ax.set_xlabel('Time' if isinstance(x_data[0], (int, float)) and x_data[0] > 10 else 'Segment')
            ax.set_ylabel('Percentage (%)')
            ax.set_title(f'Stacked Error Time Band Analysis\n(n={plot_data["total_samples"]} samples, {plot_data["n_segments"]} segments)')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            
            # Add legend
            if show_legend:
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
        except Exception as e:
            raise RuntimeError(f"Plot generation failed: {str(e)}")
            
    def _get_color_scheme(self, scheme: str, n_colors: int) -> List[str]:
        """Get color scheme for bands."""
        if scheme == 'traffic_light':
            # Green -> Yellow -> Red progression
            if n_colors <= 3:
                return ['green', 'yellow', 'red'][:n_colors]
            else:
                colors = ['green'] + ['yellow'] * (n_colors - 2) + ['red']
                return colors
        elif scheme == 'viridis':
            return plt.cm.viridis(np.linspace(0, 1, n_colors))
        elif scheme == 'plasma':
            return plt.cm.plasma(np.linspace(0, 1, n_colors))
        elif scheme == 'inferno':
            return plt.cm.inferno(np.linspace(0, 1, n_colors))
        elif scheme == 'magma':
            return plt.cm.magma(np.linspace(0, 1, n_colors))
        elif scheme == 'cool':
            return plt.cm.cool(np.linspace(0, 1, n_colors))
        elif scheme == 'warm':
            return plt.cm.warm(np.linspace(0, 1, n_colors))
        else:
            return plt.cm.tab10(np.arange(n_colors))

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
            "output_types": ["stacked_area_chart", "statistics", "temporal_analysis", "distribution_metrics"]
        }