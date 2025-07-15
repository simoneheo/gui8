"""
Base Comparison Class

This module defines the base class for all comparison methods, providing a consistent
interface and common functionality for data comparison operations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overlay import Overlay

class BaseComparison(ABC):
    """
    Base class for all comparison methods.
    
    Provides common functionality for data validation, plotting, and overlay management.
    """
    
    # Class attributes that should be defined by subclasses
    name: str = ""
    description: str = ""
    category: str = ""
    version: str = "1.0.0"
    tags: List[str] = []
    params: List[Dict[str, Any]] = []
    plot_type: str = "scatter"  # Default plot type
    overlay_options: Dict[str, Dict[str, Any]] = {}
    
    # Default overlay styles - can be overridden by subclasses
    default_overlay_styles: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self, **kwargs):
        """Initialize comparison method with parameters."""
        self.kwargs = kwargs
        self.results = None
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Apply the comparison method to the data.
        
        This is the main entry point that orchestrates the comparison process:
        1. Validates input data
        2. Removes invalid data points
        3. Calls method-specific plot_script
        4. Calls method-specific stats_script
        5. Packages results in standard format
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            ref_time: Optional reference time array
            test_time: Optional test time array
            
        Returns:
            Dictionary containing comparison results
        """
        try:
            # === STEP 1: VALIDATE INPUT DATA ===
            # Basic validation (shape, type, length compatibility)
            ref_data, test_data = self._validate_input_data(ref_data, test_data)
            
            # Remove NaN and infinite values
            ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
            
            # === STEP 2: PLOT SCRIPT (method-specific transformation) ===
            x_data, y_data, plot_metadata = self.plot_script(ref_clean, test_clean, self.kwargs)
            
            # === STEP 3: STATS SCRIPT (method-specific statistical calculations) ===
            stats_results = self.stats_script(x_data, y_data, ref_clean, test_clean, self.kwargs)
            
            # === STEP 4: PACKAGE RESULTS ===
            # Let subclasses customize the plot_data structure if needed
            plot_data = self._prepare_plot_data(x_data, y_data, ref_clean, test_clean, valid_ratio, plot_metadata)
            
            # Combine results in standard format
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
            raise RuntimeError(f"{self.name} comparison failed: {str(e)}")
    
    def _prepare_plot_data(self, x_data, y_data, ref_clean, test_clean, valid_ratio, plot_metadata):
        """
        Prepare plot data structure. Can be overridden by subclasses for custom structure.
        
        Args:
            x_data: X-axis data from plot_script
            y_data: Y-axis data from plot_script
            ref_clean: Cleaned reference data
            test_clean: Cleaned test data
            valid_ratio: Ratio of valid data points
            plot_metadata: Metadata from plot_script
            
        Returns:
            Dictionary containing plot data
        """
        return {
            'x_data': x_data,
            'y_data': y_data,
            'ref_data': ref_clean,
            'test_data': test_clean,
            'valid_ratio': valid_ratio,
            'metadata': plot_metadata
        }
    
    @abstractmethod
    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """Transform data for plotting. Returns (x_data, y_data, metadata)."""
        pass
    
    @abstractmethod
    def stats_script(self, x_data: List[float], y_data: List[float], 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """Calculate statistics for the comparison."""
        pass
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Optional[Dict[str, Any]] = None, 
                     stats_results: Optional[Dict[str, Any]] = None,
                     overlay_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Generate comparison plot with performance optimization and overlay support.
        
        This is the main plotting method that:
        1. Applies performance optimizations
        2. Dispatches to appropriate plot type based on config
        3. Adds overlay elements
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from stats_script method
            overlay_config: Separate overlay configuration dictionary
        """
        if plot_config is None:
            plot_config = {}
        if overlay_config is None:
            overlay_config = {}
        
        # Step 1: Apply performance optimizations
        ref_plot, test_plot = self._apply_performance_optimizations(ref_data, test_data, plot_config)
        
        # Step 2: Dispatch to appropriate plot type
        plot_type = plot_config.get('type', self.plot_type)
        
        if plot_type == 'scatter':
            self._create_scatter_plot(ax, ref_plot, test_plot, plot_config)
        elif plot_type == 'density':
            self._create_density_plot(ax, ref_plot, test_plot, plot_config)
        elif plot_type == 'hexbin':
            self._create_hexbin_plot(ax, ref_plot, test_plot, plot_config)
        elif plot_type == 'histogram':
            self._create_histogram_plot(ax, ref_plot, test_plot, plot_config)
        elif plot_type == 'histogram2d':
            self._create_histogram2d_plot(ax, ref_plot, test_plot, plot_config)
        else:
            # Fallback to scatter plot
            self._create_scatter_plot(ax, ref_plot, test_plot, plot_config)
        
        # Step 3: Add overlay elements
        self._add_overlay_elements(ax, ref_plot, test_plot, plot_config, stats_results, overlay_config)
        
        # Step 4: Set labels and title
        self._set_plot_labels(ax, plot_config)
    
    def _apply_performance_optimizations(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                       plot_config: Dict[str, Any]) -> tuple:
        """
        Apply performance optimizations for large datasets.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            
        Returns:
            Tuple of (optimized_ref_data, optimized_test_data)
        """
        max_points = plot_config.get('max_points', 10000)
        
        if len(ref_data) <= max_points:
            return ref_data, test_data
        
        # Subsample data for performance
        subsample_method = plot_config.get('subsample_method', 'random')
        
        if subsample_method == 'random':
            indices = np.random.choice(len(ref_data), max_points, replace=False)
            indices = np.sort(indices)  # Maintain order
        elif subsample_method == 'uniform':
            indices = np.linspace(0, len(ref_data) - 1, max_points, dtype=int)
        else:
            # Default to uniform sampling
            indices = np.linspace(0, len(ref_data) - 1, max_points, dtype=int)
        
        return ref_data[indices], test_data[indices]
    
    def _create_scatter_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                           plot_config: Dict[str, Any]) -> None:
        """Create a scatter plot."""
        alpha = plot_config.get('alpha', 0.6)
        marker_size = plot_config.get('marker_size', 20)
        color = plot_config.get('color', 'blue')
        
        ax.scatter(ref_data, test_data, alpha=alpha, s=marker_size, c=color)
    
    def _create_density_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                           plot_config: Dict[str, Any]) -> None:
        """Create a density scatter plot."""
        try:
            # Try to create density plot with color mapping
            from matplotlib.colors import LinearSegmentedColormap
            
            # Calculate 2D histogram for density
            bins = plot_config.get('density_bins', 50)
            hist, xedges, yedges = np.histogram2d(ref_data, test_data, bins=bins)
            
            # Create density values for each point
            x_idx = np.digitize(ref_data, xedges) - 1
            y_idx = np.digitize(test_data, yedges) - 1
            
            # Clip indices to valid range
            x_idx = np.clip(x_idx, 0, hist.shape[0] - 1)
            y_idx = np.clip(y_idx, 0, hist.shape[1] - 1)
            
            density = hist[x_idx, y_idx]
            
            alpha = plot_config.get('alpha', 0.6)
            marker_size = plot_config.get('marker_size', 20)
            
            scatter = ax.scatter(ref_data, test_data, c=density, alpha=alpha, s=marker_size, cmap='viridis')
            
            # Add colorbar if requested
            if plot_config.get('show_colorbar', False):
                plt.colorbar(scatter, ax=ax, label='Density')
                
        except Exception:
            # Fallback to regular scatter plot
            self._create_scatter_plot(ax, ref_data, test_data, plot_config)
    
    def _create_hexbin_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                          plot_config: Dict[str, Any]) -> None:
        """Create a hexbin plot for large datasets."""
        gridsize = plot_config.get('hexbin_gridsize', 30)
        cmap = plot_config.get('cmap', 'Blues')
        
        hb = ax.hexbin(ref_data, test_data, gridsize=gridsize, cmap=cmap, mincnt=1)
        
        # Add colorbar if requested
        if plot_config.get('show_colorbar', False):
            plt.colorbar(hb, ax=ax, label='Count')
    
    def _create_histogram_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any]) -> None:
        """Create histogram plots (side by side or overlaid)."""
        bins = plot_config.get('bins', 50)
        alpha = plot_config.get('alpha', 0.7)
        
        ax.hist(ref_data, bins=bins, alpha=alpha, label='Reference', density=True)
        ax.hist(test_data, bins=bins, alpha=alpha, label='Test', density=True)
        ax.legend()
    
    def _create_histogram2d_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                               plot_config: Dict[str, Any]) -> None:
        """Create a 2D histogram plot."""
        bins = plot_config.get('bins', 50)
        cmap = plot_config.get('cmap', 'Blues')
        
        h = ax.hist2d(ref_data, test_data, bins=bins, cmap=cmap)
        
        # Add colorbar if requested
        if plot_config.get('show_colorbar', False):
            plt.colorbar(h[3], ax=ax, label='Count')
    
    def _add_overlay_elements(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                            plot_config: Dict[str, Any], 
                            stats_results: Optional[Dict[str, Any]] = None,
                            overlay_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Add overlay elements to the plot.
        
        This method can be overridden by subclasses for comparison-specific overlays,
        or extended to provide common overlay functionality.
        
        Args:
            ax: Matplotlib axes object
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from stats_script method
            overlay_config: Separate overlay configuration dictionary
        """
        # Base implementation - can be overridden by subclasses
        # Common overlays like grid, legend, etc. can be added here
        
        if plot_config.get('show_grid', False):
            ax.grid(True, alpha=0.3)
        
        # Let subclasses handle their specific overlays using the new declarative system
        if hasattr(self, '_create_overlays'):
            self._add_comparison_specific_overlays(ax, ref_data, test_data, overlay_config, stats_results)
    
    def _set_plot_labels(self, ax, plot_config: Dict[str, Any]) -> None:
        """Set plot labels and title."""
        x_label = plot_config.get('x_label', 'Reference Data')
        y_label = plot_config.get('y_label', 'Test Data')
        title = plot_config.get('title', f'{self.name.title()} Analysis')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    
    def _validate_input_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> tuple:
        """Validate input data arrays."""
        if len(ref_data) != len(test_data):
            raise ValueError("Reference and test data must have the same length")
        
        if len(ref_data) == 0:
            raise ValueError("Data arrays cannot be empty")
        
        return ref_data, test_data
    
    def _remove_invalid_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> tuple:
        """Remove NaN and infinite values from data."""
        # Find valid (finite) data points
        valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
        
        if not np.any(valid_mask):
            raise ValueError("No valid data points found (all NaN or infinite)")
        
        ref_clean = ref_data[valid_mask]
        test_clean = test_data[valid_mask]
        valid_ratio = np.sum(valid_mask) / len(ref_data)
        
        return ref_clean, test_clean, valid_ratio
    
    @classmethod
    def get_supported_plot_types(cls) -> List[str]:
        """Get list of supported plot types for this comparison method."""
        return ['scatter', 'density', 'hexbin', 'histogram', 'histogram2d']
    
    def _get_overlay_style(self, overlay_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get merged overlay style configuration.
        
        Args:
            overlay_type: Type of overlay ('statistical_text', 'identity_line', etc.)
            config: Configuration dictionary (plot_config or overlay_config)
            
        Returns:
            Merged style dictionary with defaults and user overrides
        """
        # Get default style for this overlay type
        default_style = self.default_overlay_styles.get(overlay_type, {}).copy()
        
        # Get user-provided style overrides
        user_styles = config.get('overlay_styles', {})
        user_style = user_styles.get(overlay_type, {})
        
        # Merge: defaults + user overrides
        merged_style = default_style.copy()
        merged_style.update(user_style)
        
        return merged_style
    
    def _add_comparison_specific_overlays(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                                        overlay_config: Optional[Dict[str, Any]] = None,
                                        stats_results: Optional[Dict[str, Any]] = None) -> None:
        """
        Add comparison-specific overlay elements using the declarative overlay system.
        
        This method creates Overlay objects from overlay definitions and renders them.
        
        Args:
            ax: Matplotlib axes object
            ref_data: Reference data array
            test_data: Test data array
            overlay_config: Overlay configuration dictionary
            stats_results: Statistical results from stats_script method
        """
        if overlay_config is None:
            overlay_config = {}
            
        # Get overlay definitions from subclass if method exists
        overlay_definitions = getattr(self, '_create_overlays', lambda *args, **kwargs: {})(ref_data, test_data, overlay_config, stats_results)
        
        if not overlay_definitions:
            return
            
        # Create Overlay objects and render them
        overlay_objects = self._create_overlay_objects(overlay_definitions)
        
        for overlay in overlay_objects:
            overlay.apply_to_plot(ax)
    
    def _create_overlay_objects(self, overlay_definitions: Dict[str, Dict[str, Any]]) -> List[Overlay]:
        """
        Create Overlay objects from overlay definitions.
        
        Args:
            overlay_definitions: Dictionary of overlay definitions from _create_overlays()
            
        Returns:
            List of Overlay objects
        """
        overlay_objects = []
        
        for overlay_id, overlay_def in overlay_definitions.items():
            if overlay_def.get('show', True):
                # Extract data based on overlay type
                overlay_type = overlay_def['type']
                main_data = overlay_def['main']
                overlay_data = self._extract_overlay_data(overlay_type, main_data)
                
                # Get style configuration
                style = overlay_def.get('style', {})
                default_style = self.default_overlay_styles.get(overlay_type, {})
                merged_style = default_style.copy()
                merged_style.update(style)
                
                # Create Overlay object
                overlay = Overlay(
                    id=overlay_id,
                    name=overlay_def.get('label', overlay_id),
                    type=overlay_type,
                    data=overlay_data,
                    style=merged_style,
                    show=overlay_def.get('show', True)
                )
                
                overlay_objects.append(overlay)
        
        return overlay_objects
    
    def _extract_overlay_data(self, overlay_type: str, main_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract overlay data based on type.
        
        Args:
            overlay_type: Type of overlay ('text', 'line', 'fill', etc.)
            main_data: Main data dictionary from overlay definition
            
        Returns:
            Dictionary containing extracted data for the Overlay object
        """
        if overlay_type == 'text':
            # Check if this is statistical results data (raw stats_results dict)
            if 'text' not in main_data and len(main_data) > 0:
                # This is likely raw stats_results, format it
                text_lines = self._parse_statistical_results(main_data)
                if text_lines:
                    return {
                        'text': '\n'.join(text_lines),
                        'text_lines': text_lines,  # Add text_lines for compatibility
                        'x': main_data.get('x', 0.02),
                        'y': main_data.get('y', 0.98),
                        'transform': main_data.get('transform', 'axes')
                    }
                else:
                    return {
                        'text': 'No statistical data available',
                        'text_lines': ['No statistical data available'],  # Add text_lines for compatibility
                        'x': main_data.get('x', 0.02),
                        'y': main_data.get('y', 0.98),
                        'transform': main_data.get('transform', 'axes')
                    }
            else:
                # This is regular text data
                text_content = main_data.get('text', '')
                text_lines = [text_content] if text_content else []
                return {
                    'text': text_content,
                    'text_lines': text_lines,  # Add text_lines for compatibility
                    'x': main_data.get('x', 0.02),
                    'y': main_data.get('y', 0.98),
                    'transform': main_data.get('transform', 'axes')
                }
        elif overlay_type == 'line':
            return {
                'x': main_data.get('x', []),
                'y': main_data.get('y', [])
            }
        elif overlay_type == 'vline':
            return {
                'x': main_data.get('x', [])
            }
        elif overlay_type == 'fill':
            return {
                'x': main_data.get('x', []),
                'y_lower': main_data.get('y_lower', []),
                'y_upper': main_data.get('y_upper', [])
            }
        elif overlay_type == 'marker':
            return {
                'x': main_data.get('x', []),
                'y': main_data.get('y', [])
            }
        else:
            # Return all data for unknown types
            return main_data.copy()
    

    
    def _get_stats_section(self, stats_results: Dict[str, Any], section_name: str, 
                          fallback_func=None, *args, **kwargs) -> Dict[str, Any]:
        """
        Safely get a section from stats_results with fallback computation.
        
        Args:
            stats_results: Statistical results dictionary
            section_name: Name of the section to retrieve
            fallback_func: Function to call if section not found
            *args, **kwargs: Arguments to pass to fallback function
            
        Returns:
            Dictionary containing the requested section
        """
        if stats_results and section_name in stats_results:
            section = stats_results[section_name]
            if isinstance(section, dict) and 'error' not in section:
                return section
        
        # Fallback to computing the section
        if fallback_func:
            return fallback_func(*args, **kwargs)
        
        return {}
    
    def _parse_statistical_results(self, stats_results: Dict[str, Any]) -> List[str]:
        """
        Generic parser for statistical results that handles nested dictionaries.
        
        Args:
            stats_results: Statistical results dictionary
            
        Returns:
            List of formatted text lines
        """
        text_lines = []
        
        def _format_value(value: Any) -> str:
            """Format a value for display."""
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if np.isnan(value):
                    return "N/A"
                elif abs(value) < 0.001 and value != 0:
                    return f"{value:.2e}"  # Scientific notation for very small values
                elif abs(value) < 1:
                    return f"{value:.3f}"  # 3 decimals for small values
                elif abs(value) < 100:
                    return f"{value:.2f}"  # 2 decimals for medium values
                else:
                    return f"{value:.1f}"  # 1 decimal for large values
            elif isinstance(value, bool):
                return "Yes" if value else "No"
            elif isinstance(value, str) and len(value) < 50:
                return value
            else:
                return str(value)
        
        def _format_key(key: str) -> str:
            """Format a key name for display."""
            # Convert snake_case to Title Case
            formatted = key.replace('_', ' ').title()
            # Handle special cases
            replacements = {
                'R Value': 'R',
                'P Value': 'p-value',
                'Pearson R': 'Pearson r',
                'Spearman R': 'Spearman ρ',
                'Kendall Tau': 'Kendall τ',
                'R Squared': 'R²',
                'Rmse': 'RMSE',
                'Mae': 'MAE',
                'Mse': 'MSE',
                'Std Err': 'Std Error',
                'LoA': 'LoA',
                'Bias': 'Bias',
                'N Samples': 'N',
                'Std Diff': 'Std Diff',
                'Upper Loa': 'Upper LoA',
                'Lower Loa': 'Lower LoA',
                'Agreement Multiplier': 'Agreement Multiplier',
                'Proportional Bias Slope': 'Prop Bias Slope',
                'Proportional Bias Intercept': 'Prop Bias Intercept',
                'Proportional Bias R': 'Prop Bias R',
                'Proportional Bias P Value': 'Prop Bias p-value',
                'Percent Outside Loa': '% Outside LoA',
                'Repeatability Coefficient': 'Repeatability Coef',
                'Confidence Level': 'Confidence Level',
                'Bias Ci Lower': 'Bias CI Lower',
                'Bias Ci Upper': 'Bias CI Upper',
                'Loa Lower Ci Lower': 'LoA Lower CI Lower',
                'Loa Lower Ci Upper': 'LoA Lower CI Upper',
                'Loa Upper Ci Lower': 'LoA Upper CI Lower',
                'Loa Upper Ci Upper': 'LoA Upper CI Upper'
            }
            for old, new in replacements.items():
                formatted = formatted.replace(old, new)
            return formatted
        
        # Iterate through all stats_results items
        for key, value in stats_results.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        # Handle double-nested dictionaries (e.g., correlations -> pearson -> coefficient)
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            if sub_sub_key not in ['error'] and sub_sub_value is not None:
                                formatted_value = _format_value(sub_sub_value)
                                if formatted_value != "N/A":
                                    label = f"{_format_key(sub_key)} {_format_key(sub_sub_key)}"
                                    text_lines.append(f"{label}: {formatted_value}")
                    else:
                        # Handle single-nested dictionaries
                        if sub_key not in ['error'] and sub_value is not None:
                            formatted_value = _format_value(sub_value)
                            if formatted_value != "N/A":
                                label = _format_key(sub_key)
                                text_lines.append(f"{label}: {formatted_value}")
            else:
                # Handle top-level values
                if key not in ['error'] and value is not None:
                    formatted_value = _format_value(value)
                    if formatted_value != "N/A":
                        label = _format_key(key)
                        text_lines.append(f"{label}: {formatted_value}")
        
        return text_lines
    
    def generate_overlays(self, stats_results: Dict[str, Any]) -> List[Overlay]:
        """
        Generate overlay objects from statistical results.
        
        This method bridges the gap between the declarative overlay system and
        global overlay generation needed by PairAnalyzer. It creates Overlay objects
        from the statistical results using the subclass's _create_overlays method.
        
        Args:
            stats_results: Combined statistical results from all pairs
            
        Returns:
            List of Overlay objects ready for rendering
        """
        try:
            # Get overlay definitions from subclass if method exists
            overlay_method = getattr(self, '_create_overlays', None)
            if overlay_method is not None:
                # Create dummy data arrays for overlay generation
                # The actual data isn't needed since we're working with combined stats
                dummy_ref = np.array([0, 1])  # Minimal dummy data
                dummy_test = np.array([0, 1])
                
                # Get overlay definitions from subclass
                overlay_definitions = overlay_method(
                    dummy_ref, dummy_test, stats_results, self.kwargs
                )
                
                if overlay_definitions:
                    # Convert overlay definitions to Overlay objects
                    overlay_objects = self._create_overlay_objects(overlay_definitions)
                    print(f"[BaseComparison] Generated {len(overlay_objects)} overlays from {self.name}")
                    return overlay_objects
                else:
                    print(f"[BaseComparison] No overlay definitions returned from {self.name}")
                    return []
            else:
                print(f"[BaseComparison] No _create_overlays method found in {self.name}")
                return []
                
        except Exception as e:
            print(f"[BaseComparison] Error generating overlays for {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _apply_overlay_style(self, ax, plot_func: str, *args, **kwargs) -> None:
        """
        Apply overlay style to matplotlib plotting function.
        
        Args:
            ax: Matplotlib axes object
            plot_func: Name of matplotlib function ('plot', 'text', 'fill_between', etc.)
            *args: Positional arguments for the plotting function
            **kwargs: Keyword arguments including style overrides
        """
        # Extract style-related kwargs
        style_kwargs = {}
        for key in ['color', 'linestyle', 'linewidth', 'alpha', 'marker', 'markersize', 
                   'fontsize', 'weight', 'bbox', 'verticalalignment', 'horizontalalignment']:
            if key in kwargs:
                style_kwargs[key] = kwargs.pop(key)
        
        # Call the appropriate matplotlib function
        if plot_func == 'plot':
            ax.plot(*args, **style_kwargs)
        elif plot_func == 'text':
            ax.text(*args, **style_kwargs)
        elif plot_func == 'fill_between':
            ax.fill_between(*args, **style_kwargs)
        elif plot_func == 'scatter':
            ax.scatter(*args, **style_kwargs)
        else:
            # Fallback to generic function call
            getattr(ax, plot_func)(*args, **style_kwargs)
    
    @classmethod
    def get_comparison_guidance(cls) -> Dict[str, Any]:
        """Get guidance information for this comparison method."""
        return {
            "title": cls.name.title(),
            "description": cls.description,
            "use_cases": [],
            "tips": []
        } 

    @classmethod
    def get_parameters(cls) -> List[Dict[str, Any]]:
        """
        Get parameters in the format expected by the comparison wizard.
        
        This method converts the class's params list to the format expected
        by the wizard's parameter table.
        
        Returns:
            List of parameter dictionaries with 'name', 'type', 'default', etc.
        """
        if not hasattr(cls, 'params') or not cls.params:
            return []
        
        # Convert params list to the format expected by the wizard
        parameters = []
        for param in cls.params:
            # Ensure all required fields are present
            param_dict = {
                'name': param.get('name', ''),
                'type': param.get('type', 'str'),
                'default': param.get('default', ''),
                'help': param.get('help', ''),
                'description': param.get('help', param.get('description', '')),
            }
            
            # Add optional fields if present
            if 'options' in param:
                param_dict['options'] = param['options']
            if 'min' in param:
                param_dict['min'] = param['min']
            if 'max' in param:
                param_dict['max'] = param['max']
            if 'step' in param:
                param_dict['step'] = param['step']
            
            parameters.append(param_dict)
        
        return parameters


    # Default overlay styles - can be overridden by wizard
    default_overlay_styles = {
        'statistical_text': {
            'position': (0.02, 0.98),
            'fontsize': 9,
            'color': 'black',
            'bbox': {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
            'verticalalignment': 'top',
            'horizontalalignment': 'left'
        },
        'regression_equation': {
            'position': (0.05, 0.95),
            'fontsize': 10,
            'color': 'red',
            'weight': 'bold',
            'verticalalignment': 'top',
            'horizontalalignment': 'left'
        },
        'identity_line': {
            'color': 'black',
            'linestyle': '--',
            'alpha': 0.8,
            'linewidth': 2
        },
        'regression_line': {
            'color': 'red',
            'alpha': 0.8,
            'linewidth': 2
        },
        'confidence_bands': {
            'color': 'red',
            'alpha': 0.2,
            'edgecolor': 'red',
            'linewidth': 1
        }
    }
    