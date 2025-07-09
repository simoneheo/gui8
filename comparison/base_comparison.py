"""
Base Comparison Class

This module defines the base class for all comparison methods, providing a consistent
interface and common functionality for data comparison operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import warnings
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import stats

class BaseComparison(ABC):
    """
    Abstract base class for all comparison methods.
    
    This class defines the interface that all comparison methods must implement,
    ensuring consistency across different comparison techniques.
    """
    
    # Class attributes that subclasses should override
    name = "Base Comparison"
    description = "Base class for comparison methods"
    category = "Base"
    version = "1.0.0"
    tags = ["comparison"]
    
    # Parameters that the comparison method accepts (like mixer/steps params)
    params = []
    
    # Plot configuration
    plot_type = "scatter"  # Default plot type
    requires_pairs = False  # Whether this comparison needs pair data instead of combined data
    
    def __init__(self, **kwargs):
        """
        Initialize the comparison method with parameters.
        
        Args:
            **kwargs: Parameter values for the comparison method
        """
        self.kwargs = kwargs
        self.results = {}
        self.metadata = {
            'method': self.name,
            'version': self.version,
            'parameters': kwargs.copy()
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Return all metadata about the comparison method.
        Useful for GUI display or parameter prompting.
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "tags": self.tags,
            "params": self.params
        }
    
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """
        Parse and validate user input parameters.
        
        Args:
            user_input: Dictionary of user-provided parameters
            
        Returns:
            Dictionary of validated parameters with defaults
        """
        parsed = {}
        for param in cls.params:
            name = param["name"]
            value = user_input.get(name, param.get("default"))
            
            # Handle type conversion based on parameter type
            param_type = param.get("type", "str")
            try:
                if param_type == "float":
                    parsed[name] = float(value)
                elif param_type == "int":
                    parsed[name] = int(value)
                elif param_type in ["bool", "boolean"]:
                    if isinstance(value, bool):
                        parsed[name] = value
                    elif isinstance(value, str):
                        parsed[name] = value.lower() in ['true', '1', 'yes', 'on']
                    else:
                        parsed[name] = bool(value)
                else:
                    parsed[name] = value
            except (ValueError, TypeError):
                # If conversion fails, use default or keep as string
                parsed[name] = param.get("default", value)
        return parsed
    
    @abstractmethod
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the comparison process.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing comparison results with statistics and plot data
        """
        pass
    
    @abstractmethod
    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate statistical measures for the comparison.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing statistical results
        """
        pass
    
    @abstractmethod
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate plot for this comparison method.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        pass
    
    def _validate_input_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and clean input data arrays.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            
        Returns:
            Tuple of cleaned reference and test data arrays
        """
        # Convert to numpy arrays
        ref_data = np.asarray(ref_data)
        test_data = np.asarray(test_data)
        
        # Check dimensions
        if ref_data.ndim != 1:
            raise ValueError("Reference data must be 1-dimensional")
        if test_data.ndim != 1:
            raise ValueError("Test data must be 1-dimensional")
        
        # Check length compatibility
        if len(ref_data) != len(test_data):
            warnings.warn(f"Data arrays have different lengths: ref={len(ref_data)}, test={len(test_data)}")
            # Truncate to shorter length
            min_len = min(len(ref_data), len(test_data))
            ref_data = ref_data[:min_len]
            test_data = test_data[:min_len]
        
        # Check for empty arrays
        if len(ref_data) == 0:
            raise ValueError("Data arrays are empty")
        
        return ref_data, test_data
    
    def _remove_invalid_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Remove NaN and infinite values from data arrays.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            
        Returns:
            Tuple of (cleaned_ref_data, cleaned_test_data, valid_ratio)
        """
        # Find valid data points
        valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
        
        # Calculate valid ratio
        valid_ratio = np.sum(valid_mask) / len(valid_mask)
        
        # Filter data
        ref_clean = ref_data[valid_mask]
        test_clean = test_data[valid_mask]
        
        if len(ref_clean) == 0:
            raise ValueError("No valid data points after removing NaN/infinite values")
        
        return ref_clean, test_clean, valid_ratio
    
    def _apply_performance_optimizations(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                       plot_config: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply performance optimizations like downsampling based on plot configuration.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            
        Returns:
            Tuple of (optimized_ref_data, optimized_test_data)
        """
        if plot_config is None:
            return ref_data, test_data
        
        # Apply downsampling if requested
        downsample_limit = plot_config.get('downsample', None)
        if downsample_limit and len(ref_data) > downsample_limit:
            # Use systematic sampling to preserve data distribution
            step = len(ref_data) // downsample_limit
            indices = np.arange(0, len(ref_data), step)[:downsample_limit]
            ref_data = ref_data[indices]
            test_data = test_data[indices]
            print(f"[Performance] Downsampled data from {len(indices)*step} to {len(ref_data)} points")
        
        return ref_data, test_data
    
    def _create_density_plot(self, ax, x_data: np.ndarray, y_data: np.ndarray, 
                           plot_config: Dict[str, Any] = None) -> None:
        """
        Create density plot based on configuration (scatter, hexbin, or KDE).
        
        Args:
            ax: Matplotlib axes object
            x_data: X-axis data
            y_data: Y-axis data
            plot_config: Plot configuration dictionary
        """
        if plot_config is None:
            plot_config = {}
        
        density_type = plot_config.get('density_display', 'scatter').lower()
        
        try:
            if density_type == 'hexbin':
                bin_size = plot_config.get('bin_size', 20)
                # Check for zero range (would cause ZeroDivisionError)
                if np.ptp(x_data) == 0 or np.ptp(y_data) == 0:
                    print("[Performance] Zero range detected, falling back to scatter plot")
                    ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                else:
                    hb = ax.hexbin(x_data, y_data, gridsize=bin_size, cmap='viridis', mincnt=1)
                    # Add colorbar
                    fig = ax.get_figure()
                    # Remove any existing colorbars to prevent duplication
                    if hasattr(fig, '_colorbar_list'):
                        for cb in fig._colorbar_list:
                            try:
                                cb.remove()
                            except:
                                pass
                        fig._colorbar_list = []
                    
                    cbar = plt.colorbar(hb, ax=ax)
                    cbar.set_label('Point Density', rotation=270, labelpad=15)
                    
                    # Keep track of colorbars for future cleanup
                    if not hasattr(fig, '_colorbar_list'):
                        fig._colorbar_list = []
                    fig._colorbar_list.append(cbar)
            
            elif density_type == 'kde':
                kde_bandwidth = plot_config.get('kde_bandwidth', 0.2)
                self._create_kde_plot(ax, x_data, y_data, kde_bandwidth)
            
            else:  # scatter (default)
                ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                
        except Exception as e:
            print(f"[Performance] Density plotting failed ({density_type}): {e}, falling back to scatter")
            ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
    
    def _create_kde_plot(self, ax, x_data: np.ndarray, y_data: np.ndarray, 
                        bandwidth: float = 0.2) -> None:
        """
        Create KDE density plot.
        
        Args:
            ax: Matplotlib axes object
            x_data: X-axis data
            y_data: Y-axis data
            bandwidth: KDE bandwidth parameter
        """
        try:
            if len(x_data) < 10:
                print(f"[Performance] Insufficient data for KDE ({len(x_data)} points), using scatter fallback")
                ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                return
            
            # Create KDE
            xy = np.vstack([x_data, y_data])
            kde = gaussian_kde(xy, bw_method=bandwidth)
            
            # Create grid for evaluation
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            
            # Add padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            
            # Create meshgrid
            xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate KDE
            f = np.reshape(kde(positions).T, xx.shape)
            
            # Plot contours
            ax.contourf(xx, yy, f, levels=10, cmap='viridis', alpha=0.6)
            ax.contour(xx, yy, f, levels=10, colors='black', alpha=0.3, linewidths=0.5)
            
            # Overlay scatter points
            ax.scatter(x_data, y_data, alpha=0.3, s=10, c='red')
            
        except Exception as e:
            print(f"[Performance] KDE plotting failed: {e}, falling back to scatter")
            ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
    
    def _add_overlay_elements(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                            plot_config: Dict[str, Any] = None, 
                            stats_results: Dict[str, Any] = None) -> None:
        """
        Add overlay elements to the plot based on configuration.
        
        Args:
            ax: Matplotlib axes object
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Add identity line
        if plot_config.get('show_identity_line', False):
            min_val = min(np.min(ref_data), np.min(test_data))
            max_val = max(np.max(ref_data), np.max(test_data))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
                   linewidth=2, label='Perfect Agreement')
        
        # Add regression line
        if plot_config.get('show_regression_line', False):
            self._add_regression_line(ax, ref_data, test_data)
        
        # Add confidence bands
        if plot_config.get('show_confidence_bands', False):
            confidence_level = plot_config.get('confidence_level', 0.95)
            self._add_confidence_bands(ax, ref_data, test_data, confidence_level)
        
        # Highlight outliers
        if plot_config.get('highlight_outliers', False):
            self._highlight_outliers(ax, ref_data, test_data)
        
        # Add custom line
        if plot_config.get('show_custom_line', False):
            custom_line_value = plot_config.get('custom_line_value', plot_config.get('custom_line', 0.0))
            try:
                line_value = float(custom_line_value)
                ax.axhline(y=line_value, color='red', linestyle=':', alpha=0.7, 
                          label=f'Custom: {line_value}')
            except (ValueError, TypeError):
                pass
        
        # Add statistical results text
        if plot_config.get('show_statistical_results', False) and stats_results:
            self._add_statistical_text(ax, stats_results, plot_config)
    
    def _add_regression_line(self, ax, ref_data: np.ndarray, test_data: np.ndarray) -> None:
        """Add regression line to plot."""
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(ref_data, test_data)
            
            x_line = np.array([np.min(ref_data), np.max(ref_data)])
            y_line = slope * x_line + intercept
            
            ax.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2, 
                   label=f'Regression (r={r_value:.3f})')
            
            # Add equation text
            ax.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Error adding regression line: {e}")
    
    def _add_confidence_bands(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                            confidence_level: float) -> None:
        """Add confidence bands to plot."""
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(ref_data, test_data)
            
            x_line = np.linspace(np.min(ref_data), np.max(ref_data), 100)
            y_line = slope * x_line + intercept
            
            n = len(ref_data)
            x_mean = np.mean(ref_data)
            x_std = np.std(ref_data)
            
            se_pred = std_err * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / (n * x_std**2))
            
            t_value = stats.t.ppf((1 + confidence_level) / 2, n - 2)
            ci_lower = y_line - t_value * se_pred
            ci_upper = y_line + t_value * se_pred
            
            ax.fill_between(x_line, ci_lower, ci_upper, alpha=0.2, color='red', 
                          label=f'{confidence_level*100:.0f}% Confidence')
        except Exception as e:
            print(f"Error adding confidence bands: {e}")
    
    def _highlight_outliers(self, ax, ref_data: np.ndarray, test_data: np.ndarray) -> None:
        """Highlight outliers on plot."""
        try:
            # Use IQR method to detect outliers
            q1, q3 = np.percentile(ref_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (ref_data < lower_bound) | (ref_data > upper_bound)
            
            if np.any(outlier_mask):
                ax.scatter(ref_data[outlier_mask], test_data[outlier_mask], 
                          color='red', s=50, alpha=0.8, label='Outliers')
        except Exception as e:
            print(f"Error highlighting outliers: {e}")
    
    def _add_statistical_text(self, ax, stats_results: Dict[str, Any], plot_config: Dict[str, Any] = None) -> None:
        """Add statistical results as text on plot."""
        try:
            text_lines = []
            
            # Method-specific text formatting - subclasses can override
            if hasattr(self, '_format_statistical_text'):
                # Check if method accepts plot_config parameter
                import inspect
                sig = inspect.signature(self._format_statistical_text)
                if 'plot_config' in sig.parameters:
                    text_lines = self._format_statistical_text(stats_results, plot_config)
                else:
                    text_lines = self._format_statistical_text(stats_results)
            else:
                # Generic formatting
                for key, value in stats_results.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        text_lines.append(f"{key}: {value:.3f}")
            
            if text_lines:
                text = '\n'.join(text_lines)
                ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='bottom', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Error adding statistical text: {e}")

    def get_results(self) -> Dict[str, Any]:
        """
        Get the results of the comparison.
        
        Returns:
            Dictionary containing comparison results
        """
        return self.results.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the comparison.
        
        Returns:
            Dictionary containing metadata
        """
        return self.metadata.copy()
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>" 