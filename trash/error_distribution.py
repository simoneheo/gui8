"""
Error Distribution Comparison Method

This module implements error distribution analysis between two data channels,
showing the distribution of errors using histograms with optional distribution fitting.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class ErrorDistributionComparison(BaseComparison):
    """
    Error distribution analysis comparison method.
    
    Analyzes the distribution of errors between two datasets using histograms
    with optional theoretical distribution fitting.
    """
    
    name = "error_distribution"
    description = "Analyze error distribution between datasets using histograms with distribution fitting"
    category = "Error Analysis"
    tags = ["histogram", "error", "distribution", "fitting", "statistical"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "error_type", "type": "str", "default": "absolute", "options": ["absolute", "relative", "percentage"], "help": "Type of error to analyze"},
        {"name": "bins", "type": "int", "default": 50, "min": 10, "max": 200, "help": "Number of histogram bins"},
        {"name": "bin_method", "type": "str", "default": "auto", "options": ["auto", "sturges", "scott", "fd", "rice"], "help": "Method for determining bin size"},
        {"name": "normalize", "type": "bool", "default": True, "help": "Normalize histogram to show probability density"},
        {"name": "remove_outliers", "type": "bool", "default": False, "help": "Remove outliers before analysis"},
        {"name": "outlier_method", "type": "str", "default": "iqr", "options": ["iqr", "zscore"], "help": "Method for detecting outliers"},
        {"name": "show_distribution_fit", "type": "bool", "default": True, "help": "Fit and show theoretical distribution"},
        {"name": "distribution_type", "type": "str", "default": "normal", "options": ["normal", "lognormal", "gamma", "beta"], "help": "Type of distribution to fit"}
    ]
    
    # Plot configuration
    plot_type = "histogram"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'distribution_fit': {'default': True, 'label': 'Distribution Fit', 'tooltip': 'Show fitted theoretical distribution', 'type': 'line'},
        'mean_line': {'default': True, 'label': 'Mean Line', 'tooltip': 'Vertical line at mean error', 'type': 'line'},
        'median_line': {'default': True, 'label': 'Median Line', 'tooltip': 'Vertical line at median error', 'type': 'line'},
        'percentile_lines': {'default': False, 'label': 'Percentile Lines', 'tooltip': 'Show 25th and 75th percentiles', 'type': 'line'},
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'tooltip': 'Display distribution statistics', 'type': 'text'},
        'zero_line': {'default': True, 'label': 'Zero Error Line', 'tooltip': 'Reference line at zero error', 'type': 'line'},
        'confidence_interval': {'default': False, 'label': 'Confidence Interval', 'tooltip': 'Shaded confidence interval', 'type': 'fill'}
    }
    
    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for error distribution analysis
        
        This defines what gets plotted for histogram visualization.
        
        Args:
            ref_data: Reference measurements (cleaned of NaN/infinite values)
            test_data: Test measurements (cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (bin_edges, counts, errors, metadata)
                bin_edges: Bin edges for histogram
                counts: Count or density values for each bin
                errors: Raw error values for overlay calculations
                metadata: Plot configuration dictionary
        """
        
        # Calculate errors based on error type
        error_type = params.get("error_type", "absolute")
        
        if error_type == "absolute":
            errors = test_data - ref_data
        elif error_type == "relative":
            # Avoid division by zero
            mask = np.abs(ref_data) > 1e-10
            errors = np.full_like(ref_data, np.nan)
            errors[mask] = (test_data[mask] - ref_data[mask]) / ref_data[mask]
        else:  # percentage
            # Avoid division by zero
            mask = np.abs(ref_data) > 1e-10
            errors = np.full_like(ref_data, np.nan)
            errors[mask] = (test_data[mask] - ref_data[mask]) / ref_data[mask] * 100
        
        # Remove NaN values
        errors = errors[np.isfinite(errors)]
        
        # Handle outlier removal if requested
        if params.get("remove_outliers", False):
            errors = self._remove_outliers_from_errors(errors, params)
        
        # Get histogram parameters
        bins = params.get("bins", 50)
        bin_method = params.get("bin_method", "auto")
        normalize = params.get("normalize", True)
        
        # Determine bins based on method
        if bin_method == "auto":
            bin_edges = np.histogram_bin_edges(errors, bins=bins)
        elif bin_method == "sturges":
            n_bins = int(np.ceil(np.log2(len(errors))) + 1)
            bin_edges = np.histogram_bin_edges(errors, bins=n_bins)
        elif bin_method == "scott":
            h = 3.5 * np.std(errors) / (len(errors) ** (1/3))
            n_bins = int(np.ceil((np.max(errors) - np.min(errors)) / h))
            bin_edges = np.histogram_bin_edges(errors, bins=n_bins)
        elif bin_method == "fd":
            # Freedman-Diaconis rule
            iqr = np.percentile(errors, 75) - np.percentile(errors, 25)
            h = 2 * iqr / (len(errors) ** (1/3))
            n_bins = int(np.ceil((np.max(errors) - np.min(errors)) / h))
            bin_edges = np.histogram_bin_edges(errors, bins=n_bins)
        else:  # rice
            n_bins = int(np.ceil(2 * (len(errors) ** (1/3))))
            bin_edges = np.histogram_bin_edges(errors, bins=n_bins)
        
        # Calculate histogram
        counts, _ = np.histogram(errors, bins=bin_edges, density=normalize)
        
        # Create metadata
        metadata = {
            'error_type': error_type,
            'bins': len(bin_edges) - 1,
            'bin_method': bin_method,
            'normalize': normalize,
            'n_samples': len(errors),
            'error_range': [np.min(errors), np.max(errors)],
            'xlabel': f'{error_type.title()} Error',
            'ylabel': 'Probability Density' if normalize else 'Count'
        }
        
        return bin_edges, counts, errors, metadata
    
    def stats_script(self, bin_edges: np.ndarray, counts: np.ndarray, 
                    errors: np.ndarray, ref_data: np.ndarray, params: dict) -> dict:
        """
        Statistical analysis for error distribution
        
        Args:
            bin_edges: Bin edges from plot_script
            counts: Histogram counts from plot_script
            errors: Raw error values from plot_script
            ref_data: Reference data (unused but required for interface)
            params: Method parameters
            
        Returns:
            Dictionary containing statistical results
        """
        
        # Basic distribution statistics
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        skewness = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)
        
        # Percentiles
        percentiles = np.percentile(errors, [25, 75, 90, 95, 99])
        
        # Test for normality
        shapiro_stat, shapiro_p = stats.shapiro(errors[:5000] if len(errors) > 5000 else errors)
        
        # Fit distribution if requested
        distribution_stats = {}
        if params.get("show_distribution_fit", True):
            distribution_type = params.get("distribution_type", "normal")
            distribution_stats = self._fit_distribution(errors, distribution_type)
        
        # Calculate error metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Zero-crossing analysis
        zero_crossings = np.sum(np.diff(np.sign(errors)) != 0)
        
        stats_results = {
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'mae': mae,
            'rmse': rmse,
            'percentile_25': percentiles[0],
            'percentile_75': percentiles[1],
            'percentile_90': percentiles[2],
            'percentile_95': percentiles[3],
            'percentile_99': percentiles[4],
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05,
            'zero_crossings': zero_crossings,
            'n_samples': len(errors),
            'error_range': np.max(errors) - np.min(errors)
        }
        
        # Add distribution fitting results
        stats_results.update(distribution_stats)
        
        return stats_results
    
    def _remove_outliers_from_errors(self, errors: np.ndarray, params: dict) -> np.ndarray:
        """Remove outliers from error data"""
        outlier_method = params.get("outlier_method", "iqr")
        
        if outlier_method == "iqr":
            # IQR method
            iqr_factor = params.get("iqr_factor", 1.5)
            
            q25, q75 = np.percentile(errors, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - iqr_factor * iqr
            upper_bound = q75 + iqr_factor * iqr
            
            mask = (errors >= lower_bound) & (errors <= upper_bound)
            
        else:  # zscore
            # Z-score method
            z_threshold = params.get("z_threshold", 3.0)
            z_scores = np.abs(stats.zscore(errors))
            mask = z_scores < z_threshold
        
        return errors[mask]
    
    def _fit_distribution(self, errors: np.ndarray, distribution_type: str) -> dict:
        """Fit theoretical distribution to error data"""
        
        fit_stats = {}
        
        try:
            if distribution_type == "normal":
                # Fit normal distribution
                mu, sigma = stats.norm.fit(errors)
                fit_stats.update({
                    'fit_distribution': 'normal',
                    'fit_mu': mu,
                    'fit_sigma': sigma,
                    'fit_params': [mu, sigma]
                })
                
                # Goodness of fit test
                ks_stat, ks_p = stats.kstest(errors, lambda x: stats.norm.cdf(x, mu, sigma))
                fit_stats.update({
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'fit_quality': 'good' if ks_p > 0.05 else 'poor'
                })
                
            elif distribution_type == "lognormal":
                # Fit lognormal distribution (only for positive values)
                if np.all(errors > 0):
                    s, loc, scale = stats.lognorm.fit(errors)
                    fit_stats.update({
                        'fit_distribution': 'lognormal',
                        'fit_s': s,
                        'fit_loc': loc,
                        'fit_scale': scale,
                        'fit_params': [s, loc, scale]
                    })
                    
                    # Goodness of fit test
                    ks_stat, ks_p = stats.kstest(errors, lambda x: stats.lognorm.cdf(x, s, loc, scale))
                    fit_stats.update({
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'fit_quality': 'good' if ks_p > 0.05 else 'poor'
                    })
                else:
                    fit_stats.update({
                        'fit_distribution': 'lognormal',
                        'fit_error': 'Cannot fit lognormal to negative values'
                    })
                    
            elif distribution_type == "gamma":
                # Fit gamma distribution (only for positive values)
                if np.all(errors > 0):
                    a, loc, scale = stats.gamma.fit(errors)
                    fit_stats.update({
                        'fit_distribution': 'gamma',
                        'fit_a': a,
                        'fit_loc': loc,
                        'fit_scale': scale,
                        'fit_params': [a, loc, scale]
                    })
                    
                    # Goodness of fit test
                    ks_stat, ks_p = stats.kstest(errors, lambda x: stats.gamma.cdf(x, a, loc, scale))
                    fit_stats.update({
                        'ks_statistic': ks_stat,
                        'ks_p_value': ks_p,
                        'fit_quality': 'good' if ks_p > 0.05 else 'poor'
                    })
                else:
                    fit_stats.update({
                        'fit_distribution': 'gamma',
                        'fit_error': 'Cannot fit gamma to negative values'
                    })
                    
            elif distribution_type == "beta":
                # Fit beta distribution (normalize to [0,1] range)
                errors_normalized = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
                a, b, loc, scale = stats.beta.fit(errors_normalized)
                fit_stats.update({
                    'fit_distribution': 'beta',
                    'fit_a': a,
                    'fit_b': b,
                    'fit_loc': loc,
                    'fit_scale': scale,
                    'fit_params': [a, b, loc, scale]
                })
                
                # Goodness of fit test
                ks_stat, ks_p = stats.kstest(errors_normalized, lambda x: stats.beta.cdf(x, a, b, loc, scale))
                fit_stats.update({
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'fit_quality': 'good' if ks_p > 0.05 else 'poor'
                })
                
        except Exception as e:
            fit_stats.update({
                'fit_distribution': distribution_type,
                'fit_error': str(e)
            })
        
        return fit_stats 