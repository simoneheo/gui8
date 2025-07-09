"""
Agreement Breakdown Bar Chart Comparison Module

This module provides a comprehensive bar chart analysis showing the percentage of samples
in different error ranges, giving insights into agreement patterns between reference and test data.

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
class AgreementBreakdownBarChartComparison(BaseComparison):
    """
    Agreement breakdown bar chart comparison method.
    
    This comparison creates a bar chart showing the percentage of samples
    falling within different error ranges, providing insights into the 
    distribution of agreement between reference and test data.
    
    The analysis helps identify:
    - Overall agreement patterns
    - Proportion of samples with excellent/good/poor agreement
    - Error distribution characteristics
    
    Statistical measures calculated:
    - Sample counts per error bin
    - Percentage distribution
    - Cumulative agreement metrics
    - Error range statistics
    
    Attributes:
        name (str): Unique identifier for the comparison method
        description (str): Human-readable description
        category (str): Method category for organization
        version (str): Version identifier
        tags (List[str]): Descriptive tags for search/filtering
        plot_type (str): Type of plot generated ('bar')
        
    Example:
        >>> comparison = AgreementBreakdownBarChartComparison(
        ...     use_percentage=True,
        ...     error_bins="5,10,20"
        ... )
        >>> results = comparison.apply(ref_data, test_data)
        >>> stats = comparison.calculate_stats()
        >>> fig, ax = comparison.generate_plot()
    """
    
    name = "agreement_breakdown"
    description = "Bar chart showing percentage of samples in each error range"
    category = "Summary"
    version = "1.0.0"
    tags = ["agreement", "error", "bar", "summary", "distribution"]
    plot_type = "bar"

    # Rich parameter definitions following mixer/steps pattern
    params = [
        {
            "name": "use_percentage",
            "type": "bool",
            "default": True,
            "help": "Use percentage error instead of absolute error",
            "tooltip": "When enabled, error is calculated as percentage of reference value",
            "advanced": False
        },
        {
            "name": "error_bins", 
            "type": "str",
            "default": "5,10,20",
            "help": "Comma-separated thresholds for error bins (% or absolute)",
            "tooltip": "Define error ranges: '5,10,20' creates bins [0-5], (5-10], (10-20], >20",
            "advanced": False,
            "validation": {
                "pattern": r"^[\d\.,\s]+$",
                "message": "Must be comma-separated numbers"
            }
        },
        {
            "name": "normalize_by_reference",
            "type": "bool", 
            "default": True,
            "help": "Normalize error by reference value (for percentage mode)",
            "tooltip": "Controls whether percentage error uses reference or test value as denominator",
            "advanced": True
        },
        {
            "name": "exclude_zeros",
            "type": "bool",
            "default": False,
            "help": "Exclude zero reference values from analysis",
            "tooltip": "Prevents division by zero and infinite percentage errors",
            "advanced": True
        },
        {
            "name": "bin_labels",
            "type": "str",
            "default": "auto",
            "help": "Custom bin labels (comma-separated) or 'auto'",
            "tooltip": "Override automatic bin labels with custom names",
            "advanced": True
        }
    ]

    # Rich overlay options for wizard controls
    overlay_options = {
        'show_values': {
            'default': True,
            'label': 'Show Bar Values',
            'tooltip': 'Display count or percentage on each bar',
            'type': 'bool'
        },
        'show_percentages': {
            'default': True,
            'label': 'Show Percentages',
            'tooltip': 'Display percentage values instead of counts',
            'type': 'bool'
        },
        'color_scheme': {
            'default': 'traffic_light',
            'label': 'Color Scheme',
            'tooltip': 'Color scheme for bars',
            'type': 'choice',
            'choices': ['traffic_light', 'viridis', 'plasma', 'cool', 'warm']
        },
        'add_cumulative': {
            'default': False,
            'label': 'Add Cumulative Line',
            'tooltip': 'Add cumulative percentage line plot',
            'type': 'bool'
        },
        'horizontal_bars': {
            'default': False,
            'label': 'Horizontal Bars',
            'tooltip': 'Use horizontal bar chart layout',
            'type': 'bool'
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize the Agreement Breakdown Bar Chart comparison.
        
        Args:
            **kwargs: Keyword arguments for parameter configuration
                - use_percentage (bool): Use percentage error calculation
                - error_bins (str): Comma-separated error bin thresholds
                - normalize_by_reference (bool): Normalize by reference values
                - exclude_zeros (bool): Exclude zero reference values
                - bin_labels (str): Custom bin labels or 'auto'
        """
        super().__init__(**kwargs)
        self.results = None
        self.statistics = None
        self.plot_data = None
        
        # Validate and parse parameters
        self._validate_parameters()
        self._parse_error_bins()
        self._parse_bin_labels()
        
    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        # Validate error_bins format
        error_bins_str = self.kwargs.get("error_bins", "5,10,20")
        try:
            bins = [float(x.strip()) for x in error_bins_str.split(",")]
            if len(bins) == 0:
                raise ValueError("At least one error bin threshold required")
            if any(b <= 0 for b in bins):
                raise ValueError("Error bin thresholds must be positive")
        except ValueError as e:
            raise ValueError(f"Invalid error_bins format: {e}")
            
    def _parse_error_bins(self) -> List[float]:
        """Parse and validate error bin thresholds."""
        error_bins_str = self.kwargs.get("error_bins", "5,10,20")
        try:
            bins = [float(x.strip()) for x in error_bins_str.split(",")]
            bins = sorted(bins)  # Ensure ascending order
            self.bin_edges = [0] + bins + [np.inf]
            return bins
        except ValueError as e:
            raise ValueError(f"Error parsing error_bins: {e}")
            
    def _parse_bin_labels(self) -> List[str]:
        """Parse or generate bin labels."""
        bin_labels_str = self.kwargs.get("bin_labels", "auto")
        
        if bin_labels_str == "auto":
            # Generate automatic labels
            unit = "%" if self.kwargs.get("use_percentage", True) else ""
            labels = []
            for i in range(len(self.bin_edges) - 1):
                if self.bin_edges[i+1] == np.inf:
                    labels.append(f">{self.bin_edges[i]:.0f}{unit}")
                else:
                    labels.append(f"≤{self.bin_edges[i+1]:.0f}{unit}")
            return labels
        else:
            # Use custom labels
            labels = [x.strip() for x in bin_labels_str.split(",")]
            if len(labels) != len(self.bin_edges) - 1:
                raise ValueError(f"Number of custom labels ({len(labels)}) must match number of bins ({len(self.bin_edges) - 1})")
            return labels

    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Apply agreement breakdown bar chart analysis to the data.
        
        Args:
            ref_data (np.ndarray): Reference data array
            test_data (np.ndarray): Test data array  
            ref_time (Optional[np.ndarray]): Reference time array (not used)
            test_time (Optional[np.ndarray]): Test time array (not used)
            
        Returns:
            Dict[str, Any]: Analysis results containing:
                - method: Method name
                - n_samples: Number of samples analyzed
                - plot_data: Data for plotting (labels, counts, percentages)
                - bin_edges: Error bin boundaries
                - statistics: Basic statistics
                
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
                
            if len(ref_data) == 0:
                raise ValueError("No valid data points after filtering")
            
            # Calculate error
            error = self._calculate_error(ref_data, test_data)
            
            # Generate bin labels
            bin_labels = self._parse_bin_labels()
            
            # Count samples in each bin
            bin_counts, _ = np.histogram(error, bins=self.bin_edges)
            total = len(error)
            bin_percentages = (bin_counts / total * 100) if total > 0 else np.zeros_like(bin_counts)
            
            # Calculate cumulative percentages
            cumulative_percentages = np.cumsum(bin_percentages)
            
            # Store plot data
            self.plot_data = {
                "labels": bin_labels,
                "counts": bin_counts.tolist(),
                "percentages": bin_percentages.tolist(),
                "cumulative_percentages": cumulative_percentages.tolist(),
                "bin_edges": self.bin_edges[:-1],  # Exclude infinity
                "total_samples": total
            }
            
            # Prepare results
            self.results = {
                "method": self.name,
                "n_samples": total,
                "plot_data": self.plot_data,
                "bin_edges": self.bin_edges,
                "error_data": error.tolist(),
                "parameters": dict(self.kwargs)
            }
            
            return self.results
            
        except Exception as e:
            raise RuntimeError(f"Agreement breakdown analysis failed: {str(e)}")
            
    def _calculate_error(self, ref_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        """Calculate error based on configuration."""
        if self.kwargs.get("use_percentage", True):
            # Percentage error
            denominator = ref_data if self.kwargs.get("normalize_by_reference", True) else test_data
            with np.errstate(divide='ignore', invalid='ignore'):
                error = np.abs(test_data - ref_data) / np.abs(denominator) * 100
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Absolute error
            error = np.abs(test_data - ref_data)
            
        return error

    def calculate_stats(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for the agreement breakdown.
        
        Returns:
            Dict[str, Any]: Statistical measures including:
                - bin_statistics: Per-bin statistics
                - agreement_metrics: Overall agreement measures
                - distribution_stats: Error distribution characteristics
                - quality_metrics: Data quality indicators
                
        Raises:
            RuntimeError: If no results available or calculation fails
        """
        if self.results is None:
            raise RuntimeError("No results available. Run apply() first.")
            
        try:
            plot_data = self.plot_data
            
            # Per-bin statistics
            bin_stats = {}
            for i, label in enumerate(plot_data["labels"]):
                bin_stats[label] = {
                    "count": plot_data["counts"][i],
                    "percentage": plot_data["percentages"][i],
                    "cumulative_percentage": plot_data["cumulative_percentages"][i]
                }
            
            # Agreement metrics
            excellent_threshold = 0  # First bin (lowest errors)
            good_threshold = min(1, len(plot_data["percentages"]) - 1)  # Second bin
            
            excellent_agreement = plot_data["percentages"][excellent_threshold]
            good_agreement = sum(plot_data["percentages"][:good_threshold + 1])
            poor_agreement = 100 - good_agreement
            
            # Distribution characteristics
            total_samples = plot_data["total_samples"]
            most_common_bin = np.argmax(plot_data["percentages"])
            most_common_label = plot_data["labels"][most_common_bin]
            
            # Quality metrics
            data_coverage = (total_samples / total_samples) * 100  # Always 100% after filtering
            
            self.statistics = {
                "bin_statistics": bin_stats,
                "agreement_metrics": {
                    "excellent_agreement_pct": excellent_agreement,
                    "good_agreement_pct": good_agreement,
                    "poor_agreement_pct": poor_agreement,
                    "most_common_error_range": most_common_label,
                    "most_common_percentage": plot_data["percentages"][most_common_bin]
                },
                "distribution_stats": {
                    "total_samples": total_samples,
                    "number_of_bins": len(plot_data["labels"]),
                    "non_zero_bins": sum(1 for x in plot_data["counts"] if x > 0),
                    "max_bin_count": max(plot_data["counts"]),
                    "min_bin_count": min(plot_data["counts"])
                },
                "quality_metrics": {
                    "data_coverage_pct": data_coverage,
                    "analysis_method": "percentage" if self.kwargs.get("use_percentage", True) else "absolute",
                    "zero_exclusion": self.kwargs.get("exclude_zeros", False)
                }
            }
            
            return self.statistics
            
        except Exception as e:
            raise RuntimeError(f"Statistics calculation failed: {str(e)}")

    def generate_plot(self, fig: Optional[plt.Figure] = None, 
                     ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate agreement breakdown bar chart plot.
        
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
                fig, ax = plt.subplots(figsize=(10, 6))
            
            plot_data = self.plot_data
            
            # Get overlay options
            show_values = self.kwargs.get('show_values', True)
            show_percentages = self.kwargs.get('show_percentages', True)
            color_scheme = self.kwargs.get('color_scheme', 'traffic_light')
            add_cumulative = self.kwargs.get('add_cumulative', False)
            horizontal = self.kwargs.get('horizontal_bars', False)
            
            # Prepare data
            labels = plot_data["labels"]
            values = plot_data["percentages"] if show_percentages else plot_data["counts"]
            
            # Color scheme
            colors = self._get_color_scheme(color_scheme, len(labels))
            
            # Create bar chart
            if horizontal:
                bars = ax.barh(labels, values, color=colors)
                ax.set_xlabel('Percentage (%)' if show_percentages else 'Count')
                ax.set_ylabel('Error Range')
            else:
                bars = ax.bar(labels, values, color=colors)
                ax.set_ylabel('Percentage (%)' if show_percentages else 'Count')
                ax.set_xlabel('Error Range')
            
            # Add value labels on bars
            if show_values:
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if horizontal:
                        ax.text(value + max(values) * 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{value:.1f}%' if show_percentages else f'{int(value)}',
                               ha='left', va='center')
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.01,
                               f'{value:.1f}%' if show_percentages else f'{int(value)}',
                               ha='center', va='bottom')
            
            # Add cumulative line if requested
            if add_cumulative and not horizontal:
                ax2 = ax.twinx()
                ax2.plot(labels, plot_data["cumulative_percentages"], 'ro-', alpha=0.7, label='Cumulative %')
                ax2.set_ylabel('Cumulative Percentage (%)')
                ax2.set_ylim(0, 100)
                ax2.legend(loc='upper right')
            
            # Styling
            ax.set_title(f'Agreement Breakdown Analysis\n(n={plot_data["total_samples"]} samples)')
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            if not horizontal and len(labels) > 3:
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            return fig, ax
            
        except Exception as e:
            raise RuntimeError(f"Plot generation failed: {str(e)}")
            
    def _get_color_scheme(self, scheme: str, n_colors: int) -> List[str]:
        """Get color scheme for bars."""
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
                "handles_time_series": False,
                "handles_missing_data": True,
                "statistical_analysis": True,
                "visual_analysis": True,
                "batch_processing": True
            },
            "output_types": ["bar_chart", "statistics", "distribution_metrics"]
        }