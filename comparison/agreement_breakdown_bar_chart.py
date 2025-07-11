"""
Agreement Breakdown Bar Chart Comparison Method

This module implements agreement breakdown bar chart analysis for visualizing
the distribution of agreement between reference and test data across error ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
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
    ...     error_unit="percentage",
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
        {"name": "num_bands", "type": "int", "default": 5, "min": 2, "max": 20, "help": "Number of agreement bands."},
        {"name": "band_width", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0, "decimals": 2, "help": "Width of each band."}
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
        },
        'show_legend': {
            'default': False,
            'label': 'Show Legend',
            'tooltip': 'Show legend with plot elements',
            'type': 'legend'
        }
    }

    def __init__(self, **kwargs):
        """
        Initialize the Agreement Breakdown Bar Chart comparison.
        
        Args:
            **kwargs: Keyword arguments for parameter configuration
                - error_unit (str): Error unit type ('percentage' or 'absolute')
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
            unit = "%" if self.kwargs.get("error_unit", "percentage") == "percentage" else ""
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
        Main comparison method - orchestrates the agreement breakdown analysis.
        
        Streamlined 3-step workflow:
        1. Validate input data (basic validation + remove NaN/infinite values)
        2. plot_script (core transformation + data processing)
        3. stats_script (statistical calculations)
        
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
            # === STEP 1: VALIDATE INPUT DATA ===
            # Basic validation (shape, type, length compatibility)
            ref_data, test_data = self._validate_input_data(ref_data, test_data)
            # Remove NaN and infinite values
            ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
            
            # === STEP 2: PLOT SCRIPT (core transformation + data processing) ===
            x_data, y_data, plot_metadata = self.plot_script(ref_clean, test_clean, self.kwargs)
            
            # === STEP 3: STATS SCRIPT (statistical calculations) ===
            stats_results = self.stats_script(x_data, y_data, ref_clean, test_clean, self.kwargs)
            
            # Prepare plot data
            plot_data = {
                'labels': x_data,
                'counts': y_data,
                'percentages': plot_metadata['percentages'],
                'cumulative_percentages': plot_metadata['cumulative_percentages'],
                'bin_edges': plot_metadata['bin_edges'],
                'total_samples': plot_metadata['total_samples'],
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
            
        except Exception as e:
            raise RuntimeError(f"Agreement breakdown analysis failed: {str(e)}")

    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for agreement breakdown analysis
        
        This defines what gets plotted for the bar chart - bin labels and counts.
        
        Args:
            ref_data: Reference measurements (already cleaned of NaN/infinite values)
            test_data: Test measurements (already cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (x_data, y_data, metadata)
                x_data: Bin labels for X-axis
                y_data: Bin counts for Y-axis
                metadata: Plot configuration dictionary
        """
        # Handle zero exclusion
        if params.get("exclude_zeros", False):
            mask = ref_data != 0
            ref_data = ref_data[mask]
            test_data = test_data[mask]
            
        if len(ref_data) == 0:
            raise ValueError("No valid data points after filtering")
        
        # Calculate error
        error = self._calculate_error(ref_data, test_data, params)
        
        # Generate bin labels
        bin_labels = self._parse_bin_labels()
        
        # Count samples in each bin
        bin_counts, _ = np.histogram(error, bins=self.bin_edges)
        total = len(error)
        bin_percentages = (bin_counts / total * 100) if total > 0 else np.zeros_like(bin_counts)
        
        # Calculate cumulative percentages
        cumulative_percentages = np.cumsum(bin_percentages)
        
        # Prepare metadata for plotting
        metadata = {
            'percentages': bin_percentages.tolist(),
            'cumulative_percentages': cumulative_percentages.tolist(),
            'bin_edges': self.bin_edges[:-1],  # Exclude infinity
            'total_samples': total,
            'x_label': 'Error Range',
            'y_label': 'Percentage (%)' if params.get('show_percentages', True) else 'Count',
            'title': 'Agreement Breakdown Analysis',
            'plot_type': 'bar',
            'is_percentage': params.get('error_unit', 'percentage') == 'percentage',
            'error_data': error.tolist()
        }
        
        return bin_labels, bin_counts.tolist(), metadata

    def _calculate_error(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> np.ndarray:
        """Calculate error based on configuration."""
        if params.get("error_unit", "percentage") == "percentage":
            # Percentage error
            denominator = ref_data if params.get("normalize_by_reference", True) else test_data
            with np.errstate(divide='ignore', invalid='ignore'):
                error = np.abs(test_data - ref_data) / np.abs(denominator) * 100
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Absolute error
            error = np.abs(test_data - ref_data)
            
        return error

    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        BACKWARD COMPATIBILITY + SAFETY WRAPPER: Calculate agreement breakdown statistics.
        
        This method maintains compatibility with existing code and provides comprehensive
        validation and error handling around the core statistical calculations.
        
        Args:
            ref_data (np.ndarray): Reference data array
            test_data (np.ndarray): Test data array
            ref_time (Optional[np.ndarray]): Reference time array (not used)
            test_time (Optional[np.ndarray]): Test time array (not used)
        
        Returns:
            Dict[str, Any]: Statistical measures including:
                - bin_statistics: Per-bin statistics
                - agreement_metrics: Overall agreement measures
                - distribution_stats: Error distribution characteristics
                - quality_metrics: Data quality indicators
                
        Raises:
            RuntimeError: If no results available or calculation fails
        """
        # Get plot data using the script-based approach
        x_data, y_data, plot_metadata = self.plot_script(ref_data, test_data, self.kwargs)
        
        # === INPUT VALIDATION ===
        if len(x_data) != len(y_data):
            raise ValueError("X and Y data arrays must have the same length")
        
        if len(y_data) < 1:
            raise ValueError("Insufficient data for statistical analysis")
        
        # === PURE CALCULATIONS (delegated to stats_script) ===
        stats_results = self.stats_script(x_data, y_data, ref_data, test_data, self.kwargs)
        
        return stats_results

    def stats_script(self, x_data: List[str], y_data: List[int], 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for agreement breakdown analysis
        
        Args:
            x_data: Bin labels
            y_data: Bin counts
            ref_data: Original reference data
            test_data: Original test data
            params: Method parameters dictionary
            
        Returns:
            Dictionary containing statistical results
        """
        # Get metadata from plot_script
        _, _, plot_metadata = self.plot_script(ref_data, test_data, params)
        
        bin_percentages = plot_metadata['percentages']
        cumulative_percentages = plot_metadata['cumulative_percentages']
        total_samples = plot_metadata['total_samples']
        
        # Per-bin statistics
        bin_stats = {}
        for i, label in enumerate(x_data):
            bin_stats[label] = {
                "count": y_data[i],
                "percentage": bin_percentages[i],
                "cumulative_percentage": cumulative_percentages[i]
            }
        
        # Agreement metrics
        excellent_threshold = 0  # First bin (lowest errors)
        good_threshold = min(1, len(bin_percentages) - 1)  # Second bin
        
        excellent_agreement = bin_percentages[excellent_threshold]
        good_agreement = sum(bin_percentages[:good_threshold + 1])
        poor_agreement = 100 - good_agreement
        
        # Distribution characteristics
        most_common_bin = np.argmax(bin_percentages)
        most_common_label = x_data[most_common_bin]
        
        # Quality metrics
        data_coverage = (total_samples / total_samples) * 100  # Always 100% after filtering
        
        stats_results = {
            "bin_statistics": bin_stats,
            "agreement_metrics": {
                "excellent_agreement_pct": excellent_agreement,
                "good_agreement_pct": good_agreement,
                "poor_agreement_pct": poor_agreement,
                "most_common_error_range": most_common_label,
                "most_common_percentage": bin_percentages[most_common_bin]
            },
            "distribution_stats": {
                "total_samples": total_samples,
                "number_of_bins": len(x_data),
                "non_zero_bins": sum(1 for x in y_data if x > 0),
                "max_bin_count": max(y_data),
                "min_bin_count": min(y_data)
            },
            "quality_metrics": {
                "data_coverage_pct": data_coverage,
                "analysis_method": "percentage" if params.get("error_unit", "percentage") == "percentage" else "absolute",
                "zero_exclusion": params.get("exclude_zeros", False)
            }
        }
        
        return stats_results

    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate agreement breakdown bar chart plot.
        
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
            plot_data = self.results['plot_data']
            
            # Merge plot_config with overlay options if provided
            if plot_config is None:
                plot_config = {}
            
            # Get overlay options (prefer plot_config over kwargs)
            show_values = plot_config.get('show_values', self.kwargs.get('show_values', True))
            show_percentages = plot_config.get('show_percentages', self.kwargs.get('show_percentages', True))
            color_scheme = plot_config.get('color_scheme', self.kwargs.get('color_scheme', 'traffic_light'))
            add_cumulative = plot_config.get('add_cumulative', self.kwargs.get('add_cumulative', False))
            horizontal = plot_config.get('horizontal_bars', self.kwargs.get('horizontal_bars', False))
            
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

    def _format_statistical_text(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format statistical results for text overlay."""
        lines = []
        
        total_samples = stats_results.get('total_samples', 0)
        if total_samples > 0:
            lines.append(f"Total Samples: {total_samples}")
        
        excellent_pct = stats_results.get('excellent_agreement_percentage', 0)
        if excellent_pct > 0:
            lines.append(f"Excellent: {excellent_pct:.1f}%")
        
        return lines
    
    def _get_overlay_functional_properties(self, overlay_id: str, overlay_type: str, 
                                         stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get functional properties for bar chart overlays (no arbitrary styling)."""
        properties = {}
        
        if overlay_id == 'show_values' and overlay_type == 'text':
            properties.update({
                'show_bar_values': True,
                'label': 'Bar Values'
            })
        elif overlay_id == 'show_percentages' and overlay_type == 'text':
            properties.update({
                'show_percentages': True,
                'label': 'Percentages'
            })
        elif overlay_id == 'show_legend' and overlay_type == 'legend':
            properties.update({
                'label': 'Legend'
            })
        
        return properties