"""
Error Distribution Histogram Comparison Method

This module implements error distribution analysis between two data channels,
showing histogram of errors (test - reference) with optional statistical overlays.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class ErrorDistributionHistogramComparison(BaseComparison):
    """
    Error distribution histogram comparison method.
    
    Analyzes the distribution of errors between test and reference datasets
    using histograms with optional statistical overlays like KDE and Gaussian fits.
    """
    
    name = "error_distribution_histogram"
    description = "Histogram analysis of (test - reference) error distribution with statistical overlays"
    category = "Distribution"
    version = "1.0.0"
    tags = ["histogram", "distribution", "error", "statistical"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "use_percentage", "type": "bool", "default": False, "help": "Use percentage error instead of absolute error"},
        {"name": "bins", "type": "int", "default": 50, "help": "Number of histogram bins"},
        {"name": "density", "type": "bool", "default": True, "help": "Normalize histogram to show density"},
        {"name": "outlier_threshold", "type": "float", "default": 3.0, "help": "Z-score threshold for outlier detection"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for statistical tests"}
    ]
    
    # Plot configuration
    plot_type = "histogram"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_kde': {'default': True, 'label': 'Show KDE Curve', 'tooltip': 'Overlay kernel density estimate curve'},
        'show_gaussian_fit': {'default': True, 'label': 'Show Gaussian Fit', 'tooltip': 'Overlay normal distribution fit'},
        'show_mean_line': {'default': True, 'label': 'Show Mean Line', 'tooltip': 'Show vertical line at mean error'},
        'show_zero_line': {'default': True, 'label': 'Show Zero Line', 'tooltip': 'Show vertical line at zero error'},
        'highlight_outliers': {'default': False, 'label': 'Highlight Outliers', 'tooltip': 'Highlight outlier bins in the histogram'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display distribution statistics on the plot'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the error distribution analysis.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing error distribution results with statistics and plot data
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Calculate error values
        error_data = self._calculate_error_values(ref_clean, test_clean)
        
        # Calculate statistics
        stats_results = self.calculate_stats(ref_clean, test_clean, ref_time, test_time)
        
        # Prepare plot data
        plot_data = {
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
        Calculate error distribution statistics.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing distribution statistics
        """
        # Calculate error values
        error_data = self._calculate_error_values(ref_data, test_data)
        
        # Initialize results
        stats_results = {
            'distribution': {},
            'normality': {},
            'outliers': {}
        }
        
        # Basic distribution statistics
        stats_results['distribution'] = self._compute_distribution_stats(error_data)
        
        # Normality tests
        stats_results['normality'] = self._compute_normality_tests(error_data)
        
        # Outlier analysis
        stats_results['outliers'] = self._compute_outlier_analysis(error_data)
        
        # Gaussian fit if requested
        if self.kwargs.get('show_gaussian_fit', True):
            stats_results['gaussian_fit'] = self._fit_gaussian_distribution(error_data)
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate error distribution histogram with overlays.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Calculate error values
        error_data = self._calculate_error_values(ref_data, test_data)
        
        # Create histogram
        self._create_histogram(ax, error_data, plot_config)
        
        # Add overlay elements
        self._add_distribution_overlays(ax, error_data, plot_config, stats_results)
        
        # Set labels and title
        error_type = "Percentage Error (%)" if self.kwargs.get('use_percentage', False) else "Error"
        ax.set_xlabel(error_type)
        ax.set_ylabel('Density' if self.kwargs.get('density', True) else 'Frequency')
        ax.set_title('Error Distribution Analysis')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if there are overlays
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
    
    def _calculate_error_values(self, ref_data: np.ndarray, test_data: np.ndarray) -> np.ndarray:
        """Calculate error values based on configuration."""
        error = test_data - ref_data
        
        if self.kwargs.get("use_percentage", False):
            with np.errstate(divide='ignore', invalid='ignore'):
                error = error / np.abs(ref_data) * 100
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        
        return error
    
    def _compute_distribution_stats(self, error_data: np.ndarray) -> Dict[str, float]:
        """Compute basic distribution statistics."""
        try:
            return {
                'mean': np.mean(error_data),
                'std': np.std(error_data),
                'var': np.var(error_data),
                'skewness': stats.skew(error_data),
                'kurtosis': stats.kurtosis(error_data),
                'median': np.median(error_data),
                'mad': stats.median_abs_deviation(error_data),
                'min': np.min(error_data),
                'max': np.max(error_data),
                'range': np.ptp(error_data)
            }
        except Exception as e:
            return {
                'mean': np.nan, 'std': np.nan, 'var': np.nan,
                'skewness': np.nan, 'kurtosis': np.nan, 'median': np.nan,
                'mad': np.nan, 'min': np.nan, 'max': np.nan, 'range': np.nan,
                'error': str(e)
            }
    
    def _compute_normality_tests(self, error_data: np.ndarray) -> Dict[str, Any]:
        """Perform normality tests on error distribution."""
        try:
            # Shapiro-Wilk test (good for small samples)
            shapiro_stat, shapiro_p = stats.shapiro(error_data[:5000])  # Limit for performance
            
            # Kolmogorov-Smirnov test against normal
            ks_stat, ks_p = stats.kstest(error_data, 'norm', args=(np.mean(error_data), np.std(error_data)))
            
            # Anderson-Darling test
            ad_result = stats.anderson(error_data, dist='norm')
            
            return {
                'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p},
                'anderson_darling': {
                    'statistic': ad_result.statistic,
                    'critical_values': ad_result.critical_values.tolist(),
                    'significance_levels': ad_result.significance_level.tolist()
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _compute_outlier_analysis(self, error_data: np.ndarray) -> Dict[str, Any]:
        """Analyze outliers in error distribution."""
        try:
            threshold = self.kwargs.get('outlier_threshold', 3.0)
            z_scores = np.abs(stats.zscore(error_data))
            outlier_mask = z_scores > threshold
            
            return {
                'threshold': threshold,
                'n_outliers': np.sum(outlier_mask),
                'outlier_percentage': np.sum(outlier_mask) / len(error_data) * 100,
                'outlier_indices': np.where(outlier_mask)[0].tolist(),
                'outlier_values': error_data[outlier_mask].tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _fit_gaussian_distribution(self, error_data: np.ndarray) -> Dict[str, float]:
        """Fit Gaussian distribution to error data."""
        try:
            mu, sigma = stats.norm.fit(error_data)
            
            # Goodness of fit
            ks_stat, ks_p = stats.kstest(error_data, 'norm', args=(mu, sigma))
            
            return {
                'mu': mu,
                'sigma': sigma,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p
            }
        except Exception as e:
            return {
                'mu': np.nan,
                'sigma': np.nan,
                'ks_statistic': np.nan,
                'ks_p_value': np.nan,
                'error': str(e)
            }
    
    def _create_histogram(self, ax, error_data: np.ndarray, plot_config: Dict[str, Any]) -> None:
        """Create the main histogram plot."""
        bins = self.kwargs.get('bins', 50)
        density = self.kwargs.get('density', True)
        
        # Create histogram
        n, bins_edges, patches = ax.hist(
            error_data, 
            bins=bins, 
            density=density, 
            alpha=0.7, 
            color='skyblue', 
            edgecolor='black',
            label='Error Distribution'
        )
        
        # Highlight outlier bins if requested
        if plot_config.get('highlight_outliers', False):
            self._highlight_outlier_bins(ax, error_data, bins_edges, patches)
    
    def _add_distribution_overlays(self, ax, error_data: np.ndarray, 
                                 plot_config: Dict[str, Any], 
                                 stats_results: Dict[str, Any] = None) -> None:
        """Add overlay elements to the histogram."""
        if stats_results is None:
            return
        
        # KDE overlay
        if plot_config.get('show_kde', True):
            self._add_kde_overlay(ax, error_data)
        
        # Gaussian fit overlay
        if plot_config.get('show_gaussian_fit', True) and 'gaussian_fit' in stats_results:
            self._add_gaussian_fit_overlay(ax, error_data, stats_results['gaussian_fit'])
        
        # Mean line
        if plot_config.get('show_mean_line', True) and 'distribution' in stats_results:
            mean_val = stats_results['distribution'].get('mean', np.nan)
            if not np.isnan(mean_val):
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        # Zero line
        if plot_config.get('show_zero_line', True):
            ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
        
        # Statistical results text
        if plot_config.get('show_statistical_results', True):
            self._add_statistical_text_box(ax, stats_results, plot_config)
    
    def _add_kde_overlay(self, ax, error_data: np.ndarray) -> None:
        """Add KDE overlay to histogram."""
        try:
            kde = gaussian_kde(error_data)
            x_range = np.linspace(np.min(error_data), np.max(error_data), 200)
            kde_values = kde(x_range)
            ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
        except Exception as e:
            print(f"[ErrorDistribution] Error adding KDE overlay: {e}")
    
    def _add_gaussian_fit_overlay(self, ax, error_data: np.ndarray, fit_results: Dict[str, float]) -> None:
        """Add Gaussian fit overlay to histogram."""
        try:
            mu = fit_results.get('mu', np.nan)
            sigma = fit_results.get('sigma', np.nan)
            
            if not np.isnan(mu) and not np.isnan(sigma):
                x_range = np.linspace(np.min(error_data), np.max(error_data), 200)
                gaussian_values = stats.norm.pdf(x_range, mu, sigma)
                ax.plot(x_range, gaussian_values, 'orange', linestyle='--', linewidth=2, 
                       label=f'Gaussian Fit (μ={mu:.3f}, σ={sigma:.3f})')
        except Exception as e:
            print(f"[ErrorDistribution] Error adding Gaussian fit overlay: {e}")
    
    def _highlight_outlier_bins(self, ax, error_data: np.ndarray, bins_edges: np.ndarray, patches) -> None:
        """Highlight bins containing outliers."""
        try:
            threshold = self.kwargs.get('outlier_threshold', 3.0)
            z_scores = np.abs(stats.zscore(error_data))
            outlier_mask = z_scores > threshold
            outlier_values = error_data[outlier_mask]
            
            # Color outlier bins differently
            for i, patch in enumerate(patches):
                bin_left = bins_edges[i]
                bin_right = bins_edges[i + 1]
                if np.any((outlier_values >= bin_left) & (outlier_values < bin_right)):
                    patch.set_facecolor('red')
                    patch.set_alpha(0.8)
        except Exception as e:
            print(f"[ErrorDistribution] Error highlighting outlier bins: {e}")
    
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
        
        # Distribution statistics
        if 'distribution' in stats_results:
            dist_stats = stats_results['distribution']
            mean_val = dist_stats.get('mean', np.nan)
            std_val = dist_stats.get('std', np.nan)
            skew_val = dist_stats.get('skewness', np.nan)
            
            if not np.isnan(mean_val):
                text_lines.append(f"Mean: {mean_val:.4f}")
            if not np.isnan(std_val):
                text_lines.append(f"Std: {std_val:.4f}")
            if not np.isnan(skew_val):
                text_lines.append(f"Skewness: {skew_val:.3f}")
        
        # Normality test results
        if 'normality' in stats_results:
            norm_stats = stats_results['normality']
            if 'shapiro' in norm_stats:
                shapiro_p = norm_stats['shapiro'].get('p_value', np.nan)
                if not np.isnan(shapiro_p):
                    text_lines.append(f"Shapiro p: {shapiro_p:.4f}")
        
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
            "title": "Error Distribution Histogram",
            "description": "Analyzes the distribution of errors between test and reference datasets",
            "interpretation": {
                "normal_distribution": "Errors should follow normal distribution for unbiased measurements",
                "skewness": "Positive skew indicates systematic positive bias, negative skew indicates negative bias",
                "kurtosis": "High kurtosis indicates heavy tails (more outliers), low kurtosis indicates light tails",
                "outliers": "Large number of outliers may indicate measurement problems or model inadequacy"
            },
            "use_cases": [
                "Quality assessment of measurement instruments",
                "Model validation and residual analysis", 
                "Error characterization and bias detection",
                "Statistical assumption verification"
            ],
            "tips": [
                "Normal distribution suggests unbiased, random errors",
                "Use percentage errors for scale-independent comparison",
                "Check for outliers that might indicate data quality issues",
                "Compare skewness to detect systematic bias",
                "Use KDE overlay for smoother distribution visualization"
            ]
        }