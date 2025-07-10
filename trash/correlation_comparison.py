"""
Correlation Comparison Method

This module implements correlation analysis between two data channels,
including Pearson, Spearman, and Kendall correlation coefficients.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class CorrelationComparison(BaseComparison):
    """
    Correlation analysis comparison method.
    
    Computes various correlation coefficients and related statistics
    to assess the linear and monotonic relationships between datasets.
    """
    
    name = "correlation"
    description = "Compute Pearson, Spearman, and Kendall correlation coefficients with significance tests"
    category = "Statistical"
    version = "1.0.0"
    tags = ["scatter", "correlation", "statistical", "relationship", "linear", "monotonic"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "correlation_type", "type": "str", "default": "pearson", "help": "Correlation type: pearson, spearman, all"},
        {"name": "include_rmse", "type": "bool", "default": True, "help": "Include RMSE calculation"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "help": "Outlier detection method: iqr, zscore, modified_zscore"},
        {"name": "partial_correlation", "type": "bool", "default": False, "help": "Calculate partial correlation"},
        {"name": "detrend_method", "type": "str", "default": "none", "help": "Detrending method: none, linear, polynomial"},
        {"name": "bootstrap_samples", "type": "int", "default": 1000, "help": "Number of bootstrap samples for confidence intervals"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for statistical tests"}
    ]
    
    # Plot configuration
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_identity_line': {'default': True, 'label': 'Show y = x Line', 'tooltip': 'Show identity line for perfect correlation reference'},
        'show_regression_line': {'default': True, 'label': 'Show Regression Line', 'tooltip': 'Show best-fit regression line'},
        'show_confidence_bands': {'default': False, 'label': 'Show Confidence Bands', 'tooltip': 'Show confidence bands around regression line'},
        'show_r_squared': {'default': True, 'label': 'Show R² Value', 'tooltip': 'Display R² value on the plot'},
        'highlight_outliers': {'default': False, 'label': 'Highlight Outliers', 'tooltip': 'Highlight outlier points on the plot'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display correlation statistics on the plot'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the correlation analysis.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing correlation results with statistics and plot data
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Calculate statistics
        stats_results = self.calculate_stats(ref_clean, test_clean, ref_time, test_time)
        
        # Prepare plot data
        plot_data = {
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
        Calculate correlation statistics.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing correlation statistics
        """
        # Remove outliers if requested
        if self.kwargs.get('remove_outliers', False):
            ref_data, test_data = self._remove_outliers(ref_data, test_data)
        
        # Initialize results
        stats_results = {
            'correlations': {},
            'interpretation': {}
        }
        
        correlation_type = self.kwargs.get('correlation_type', 'pearson')
        
        # Compute correlations based on type
        if correlation_type in ['pearson', 'all']:
            stats_results['correlations']['pearson'] = self._compute_pearson(ref_data, test_data)
        
        if correlation_type in ['spearman', 'all']:
            stats_results['correlations']['spearman'] = self._compute_spearman(ref_data, test_data)
        
        if correlation_type in ['kendall', 'all']:
            stats_results['correlations']['kendall'] = self._compute_kendall(ref_data, test_data)
        
        # Calculate RMSE if requested
        if self.kwargs.get('include_rmse', True):
            stats_results['rmse_metrics'] = self._compute_rmse_metrics(ref_data, test_data)
        
        # Add interpretation
        stats_results['interpretation'] = self._interpret_correlations(stats_results['correlations'])
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate correlation plot with performance and overlay options.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Apply performance optimizations
        ref_plot, test_plot = self._apply_performance_optimizations(ref_data, test_data, plot_config)
        
        # Create density plot based on configuration
        self._create_density_plot(ax, ref_plot, test_plot, plot_config)
        
        # Add overlay elements
        self._add_overlay_elements(ax, ref_plot, test_plot, plot_config, stats_results)
        
        # Set labels and title
        ax.set_xlabel('Reference Data')
        ax.set_ylabel('Test Data')
        ax.set_title(f'{self.name.title()} Analysis')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', False):
            ax.legend()
    
    def _compute_pearson(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute Pearson correlation coefficient."""
        try:
            r, p_value = stats.pearsonr(ref_data, test_data)
            r_squared = r ** 2
            
            return {
                'coefficient': r,
                'p_value': p_value,
                'r_squared': r_squared
            }
        except Exception as e:
            return {
                'coefficient': np.nan,
                'p_value': np.nan,
                'r_squared': np.nan,
                'error': str(e)
            }
    
    def _compute_spearman(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute Spearman correlation coefficient."""
        try:
            rho, p_value = stats.spearmanr(ref_data, test_data)
            
            return {
                'coefficient': rho,
                'p_value': p_value
            }
        except Exception as e:
            return {
                'coefficient': np.nan,
                'p_value': np.nan,
                'error': str(e)
            }
    
    def _compute_kendall(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute Kendall correlation coefficient."""
        try:
            tau, p_value = stats.kendalltau(ref_data, test_data)
            
            return {
                'coefficient': tau,
                'p_value': p_value
            }
        except Exception as e:
            return {
                'coefficient': np.nan,
                'p_value': np.nan,
                'error': str(e)
            }
    
    def _compute_rmse_metrics(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Compute RMSE and related metrics."""
        try:
            differences = test_data - ref_data
            rmse = np.sqrt(np.mean(differences ** 2))
            mae = np.mean(np.abs(differences))
            mse = np.mean(differences ** 2)
            
            return {
                'rmse': rmse,
                'mae': mae,
                'mse': mse
            }
        except Exception as e:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mse': np.nan,
                'error': str(e)
            }
    
    def _interpret_correlations(self, correlations: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Interpret correlation results."""
        interpretation = {}
        
        for corr_type, corr_data in correlations.items():
            if 'coefficient' not in corr_data or np.isnan(corr_data['coefficient']):
                interpretation[corr_type] = "Could not compute correlation"
                continue
            
            r = corr_data['coefficient']
            p_value = corr_data.get('p_value', np.nan)
            
            # Strength interpretation
            if abs(r) >= 0.9:
                strength = "Very Strong"
            elif abs(r) >= 0.7:
                strength = "Strong"
            elif abs(r) >= 0.5:
                strength = "Moderate"
            elif abs(r) >= 0.3:
                strength = "Weak"
            else:
                strength = "Very Weak"
            
            # Direction
            direction = "Positive" if r > 0 else "Negative"
            
            # Significance
            if not np.isnan(p_value):
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                interpretation[corr_type] = f"{strength} {direction} ({r:.3f}, p={p_value:.3f}, {significance})"
            else:
                interpretation[corr_type] = f"{strength} {direction} ({r:.3f})"
        
        return interpretation
    
    def _remove_outliers(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from data."""
        method = self.kwargs.get('outlier_method', 'iqr')
        
        if method == 'iqr':
            # IQR method
            q1, q3 = np.percentile(ref_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (ref_data >= lower_bound) & (ref_data <= upper_bound)
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(ref_data))
            mask = z_scores < 3
        else:
            # No outlier removal
            mask = np.ones(len(ref_data), dtype=bool)
        
        return ref_data[mask], test_data[mask]
    
    def _format_statistical_text(self, stats_results: Dict[str, Any], plot_config: Dict[str, Any] = None) -> List[str]:
        """Format statistical results for display on plot."""
        text_lines = []
        
        if plot_config is None:
            plot_config = {}
        
        # Add correlation results
        if 'correlations' in stats_results:
            for corr_type, corr_data in stats_results['correlations'].items():
                if 'coefficient' in corr_data and not np.isnan(corr_data['coefficient']):
                    r = corr_data['coefficient']
                    p = corr_data.get('p_value', np.nan)
                    if not np.isnan(p):
                        text_lines.append(f"{corr_type.title()}: r={r:.3f}, p={p:.3f}")
                    else:
                        text_lines.append(f"{corr_type.title()}: r={r:.3f}")
        
        # Add R² if available and requested
        if (plot_config.get('show_r_squared', False) and 
            'correlations' in stats_results and 'pearson' in stats_results['correlations']):
            pearson_data = stats_results['correlations']['pearson']
            r_squared = pearson_data.get('r_squared', np.nan)
            if not np.isnan(r_squared):
                text_lines.append(f"R²: {r_squared:.3f}")
        
        # Add RMSE if available
        if 'rmse_metrics' in stats_results:
            rmse = stats_results['rmse_metrics'].get('rmse', np.nan)
            if not np.isnan(rmse):
                text_lines.append(f"RMSE: {rmse:.3f}")
        
        # Add interpretation
        if 'interpretation' in stats_results:
            for corr_type, interpretation in stats_results['interpretation'].items():
                if len(interpretation) < 50:  # Only show short interpretations
                    text_lines.append(f"{corr_type.title()}: {interpretation}")
        
        return text_lines
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Correlation Analysis",
            "description": "Measures relationships between datasets using correlation coefficients",
            "interpretation": {
                "pearson": "Linear correlation (r), assumes normal distribution",
                "spearman": "Rank-based correlation (ρ), non-parametric, robust to outliers",
                "kendall": "Rank-based correlation (τ), more robust but computationally intensive",
                "strength": "Strong: |r| > 0.7, Moderate: 0.3 < |r| < 0.7, Weak: |r| < 0.3"
            },
            "use_cases": [
                "Assessing how well two variables move together",
                "Method validation and agreement assessment",
                "Feature correlation analysis",
                "Signal similarity assessment"
            ],
            "tips": [
                "Use Pearson for linear relationships with normal data",
                "Use Spearman for monotonic relationships or non-normal data",
                "Use Kendall for small samples or when robustness is important",
                "Check p-values for statistical significance",
                "Consider RMSE for absolute agreement assessment"
            ]
        } 