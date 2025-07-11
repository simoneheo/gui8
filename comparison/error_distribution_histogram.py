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
        {"name": "num_bins", "type": "int", "default": 50, "min": 5, "max": 1000, "help": "Number of bins in histogram. Too many bins may make the plot hard to read."},
        {"name": "range_min", "type": "float", "default": -10.0, "min": -100.0, "max": 0.0, "decimals": 2, "help": "Minimum value for histogram range."},
        {"name": "range_max", "type": "float", "default": 10.0, "min": 0.0, "max": 100.0, "decimals": 2, "help": "Maximum value for histogram range."}
    ]
    
    # Plot configuration
    plot_type = "histogram"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_kde': {'default': True, 'label': 'Show KDE Curve', 'tooltip': 'Overlay kernel density estimate curve', 'type': 'line'},
        'show_zero_line': {'default': True, 'label': 'Show Zero Line', 'tooltip': 'Show vertical line at zero error', 'type': 'line'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the error distribution histogram analysis.
        
        Streamlined 3-step workflow:
        1. Validate input data (basic validation + remove NaN/infinite values)
        2. plot_script (core transformation + histogram computation)
        3. stats_script (statistical calculations)
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing error distribution histogram analysis results
        """
        try:
            # === STEP 1: VALIDATE INPUT DATA ===
            # Basic validation (shape, type, length compatibility)
            ref_data, test_data = self._validate_input_data(ref_data, test_data)
            # Remove NaN and infinite values
            ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
            # === STEP 2: PLOT SCRIPT (core transformation + histogram computation) ===
            x_data, y_data, plot_metadata = self.plot_script(ref_clean, test_clean, self.kwargs)
        
            # === STEP 3: STATS SCRIPT (statistical calculations) ===
            stats_results = self.stats_script(x_data, y_data, ref_clean, test_clean, self.kwargs)
        
            # Prepare plot data
            plot_data = {
                'bin_edges': x_data,
                'bin_counts': y_data,
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
            raise RuntimeError(f"Error distribution histogram analysis failed: {str(e)}")

    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for error distribution histogram analysis
        
        This defines what gets plotted - error bins and their frequencies.
        
        Args:
            ref_data: Reference measurements (already cleaned of NaN/infinite values)
            test_data: Test measurements (already cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (x_data, y_data, metadata)
                x_data: Bin edges for histogram
                y_data: Bin counts/frequencies
                metadata: Plot configuration dictionary
        """
        # Calculate errors
        errors = self._calculate_errors(ref_data, test_data, params)
        
        # Create histogram
        bin_edges, bin_counts = self._create_histogram(errors, params)
        
        # Prepare metadata for plotting
        metadata = {
            'x_label': self._get_error_label(params),
            'y_label': 'Frequency' if params.get("y_axis_type", "density") == "frequency" else 'Density',
            'title': 'Error Distribution Histogram',
            'plot_type': 'histogram',
            'error_unit': params.get("error_unit", "absolute"),
            'n_bins': params.get("bins", 50),
            'y_axis_type': params.get("y_axis_type", "density"),
            'total_samples': len(errors)
        }
        
        return bin_edges.tolist(), bin_counts.tolist(), metadata

    def _calculate_errors(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> np.ndarray:
        """Calculate errors based on configuration."""
        error_unit = params.get("error_unit", "absolute")
        
        if error_unit == "absolute":
            errors = test_data - ref_data
        elif error_unit == "percentage":
            with np.errstate(divide='ignore', invalid='ignore'):
                errors = (test_data - ref_data) / ref_data * 100
                errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Default to absolute
            errors = test_data - ref_data
            
        return errors

    def _create_histogram(self, errors: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create histogram of errors."""
        n_bins = params.get("bins", 50)
        bin_range = params.get("bin_range", None)
        y_axis_type = params.get("y_axis_type", "density")
        
        if bin_range is None:
            # Auto-determine range
            if params.get("exclude_outliers", False):
                # Use IQR-based range
                q25, q75 = np.percentile(errors, [25, 75])
                iqr = q75 - q25
                outlier_factor = params.get("outlier_iqr_factor", 1.5)
                bin_range = (q25 - outlier_factor * iqr, q75 + outlier_factor * iqr)
            else:
                # Use full range
                bin_range = (np.min(errors), np.max(errors))
        
        # Create histogram
        density = (y_axis_type == "density")
        bin_counts, bin_edges = np.histogram(errors, bins=n_bins, range=bin_range, density=density)
        
        return bin_edges, bin_counts

    def _get_error_label(self, params: dict) -> str:
        """Get appropriate label for error type."""
        error_unit = params.get("error_unit", "absolute")
        
        if error_unit == "absolute":
            return "Error"
        elif error_unit == "percentage":
            return "Percentage Error (%)"
        else:
            return "Error"
    
    def calculate_stats(self, ref_data: np.ndarray, test_data: np.ndarray, 
                       ref_time: Optional[np.ndarray] = None, 
                       test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        BACKWARD COMPATIBILITY + SAFETY WRAPPER: Calculate error distribution histogram statistics.
        
        This method maintains compatibility with existing code and provides comprehensive
        validation and error handling around the core statistical calculations.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing error distribution histogram statistics
        """
        # Get plot data using the script-based approach
        x_data, y_data, plot_metadata = self.plot_script(ref_data, test_data, self.kwargs)
        
        # === INPUT VALIDATION ===
        if len(x_data) != len(y_data) + 1:  # Bin edges vs bin counts
            raise ValueError("Bin edges and bin counts must have compatible lengths")
        
        if len(y_data) < 1:
            raise ValueError("Insufficient data for statistical analysis")
        
        # === PURE CALCULATIONS (delegated to stats_script) ===
        stats_results = self.stats_script(x_data, y_data, ref_data, test_data, self.kwargs)
        
        return stats_results

    def stats_script(self, x_data: List[float], y_data: List[float], 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for error distribution histogram analysis
        
        Args:
            x_data: Bin edges
            y_data: Bin counts/frequencies
            ref_data: Original reference data
            test_data: Original test data
            params: Method parameters dictionary
            
        Returns:
            Dictionary containing statistical results
        """
        bin_edges = np.array(x_data)
        bin_counts = np.array(y_data)
        
        # Calculate original errors for detailed statistics
        errors = self._calculate_errors(ref_data, test_data, params)
        
        # Basic distribution statistics
        distribution_stats = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'min': np.min(errors),
            'max': np.max(errors),
            'median': np.median(errors),
            'q25': np.percentile(errors, 25),
            'q75': np.percentile(errors, 75),
            'iqr': np.percentile(errors, 75) - np.percentile(errors, 25),
            'range': np.ptp(errors),
            'total_count': len(errors)
        }
        
        # Histogram-specific statistics
        histogram_stats = self._analyze_histogram_properties(bin_edges, bin_counts, params)
        
        # Shape analysis
        shape_analysis = self._analyze_distribution_shape(errors, params)
        
        # Normality tests
        normality_tests = self._perform_normality_tests(errors)
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(errors, params)
        
        # Symmetry analysis
        symmetry_analysis = self._analyze_symmetry(errors, bin_edges, bin_counts)
        
        stats_results = {
            'distribution_stats': distribution_stats,
            'histogram_stats': histogram_stats,
            'shape_analysis': shape_analysis,
            'normality_tests': normality_tests,
            'outlier_analysis': outlier_analysis,
            'symmetry_analysis': symmetry_analysis
        }
        
        return stats_results

    def _analyze_histogram_properties(self, bin_edges: np.ndarray, bin_counts: np.ndarray, params: dict) -> Dict[str, Any]:
        """Analyze histogram-specific properties."""
        try:
            # Bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Mode (bin with highest count)
            mode_idx = np.argmax(bin_counts)
            mode_value = bin_centers[mode_idx]
            mode_count = bin_counts[mode_idx]
            
            # Histogram entropy
            total_count = np.sum(bin_counts)
            if total_count > 0:
                probs = bin_counts / total_count
                probs = probs[probs > 0]  # Remove zero probabilities
                entropy = -np.sum(probs * np.log2(probs))
            else:
                entropy = 0
            
            # Effective number of bins (bins with non-zero counts)
            effective_bins = np.sum(bin_counts > 0)
            
            # Histogram spread
            weighted_mean = np.average(bin_centers, weights=bin_counts)
            weighted_std = np.sqrt(np.average((bin_centers - weighted_mean)**2, weights=bin_counts))
            
            return {
                'n_bins': len(bin_counts),
                'effective_bins': effective_bins,
                'mode_value': mode_value,
                'mode_count': mode_count,
                'entropy': entropy,
                'weighted_mean': weighted_mean,
                'weighted_std': weighted_std,
                'bin_width': bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 0
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_distribution_shape(self, errors: np.ndarray, params: dict) -> Dict[str, Any]:
        """Analyze the shape of the error distribution."""
        try:
            from scipy.stats import skew, kurtosis
            
            # Skewness and kurtosis
            skewness = skew(errors)
            kurt = kurtosis(errors)
            
            # Tail analysis
            q01 = np.percentile(errors, 1)
            q99 = np.percentile(errors, 99)
            tail_ratio = (q99 - np.median(errors)) / (np.median(errors) - q01)
            
            # Peak analysis
            from scipy.stats import mode
            mode_result = mode(errors, keepdims=True)
            modal_value = mode_result.mode[0]
            modal_count = mode_result.count[0]
            
            return {
                'skewness': skewness,
                'kurtosis': kurt,
                'is_symmetric': abs(skewness) < 0.5,
                'is_mesokurtic': abs(kurt) < 0.5,
                'tail_ratio': tail_ratio,
                'modal_value': modal_value,
                'modal_count': modal_count,
                'shape_classification': self._classify_distribution_shape(skewness, kurt)
            }
        except Exception as e:
            return {'error': str(e)}

    def _classify_distribution_shape(self, skewness: float, kurtosis: float) -> str:
        """Classify distribution shape based on skewness and kurtosis."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "approximately_normal"
        elif abs(skewness) < 0.5:
            return "symmetric_non_normal"
        elif skewness > 0.5:
            return "right_skewed"
        elif skewness < -0.5:
            return "left_skewed"
        else:
            return "irregular"

    def _perform_normality_tests(self, errors: np.ndarray) -> Dict[str, Any]:
        """Perform normality tests on error distribution."""
        try:
            from scipy.stats import shapiro, jarque_bera, normaltest, kstest
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = shapiro(errors)
            
            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(errors)
            
            # D'Agostino's normality test
            da_stat, da_p = normaltest(errors)
            
            # Kolmogorov-Smirnov test against normal distribution
            ks_stat, ks_p = kstest(errors, 'norm', args=(np.mean(errors), np.std(errors)))
            
            return {
                'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p, 'is_normal': shapiro_p > 0.05},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p, 'is_normal': jb_p > 0.05},
                'dagostino': {'statistic': da_stat, 'p_value': da_p, 'is_normal': da_p > 0.05},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p, 'is_normal': ks_p > 0.05},
                'consensus_normal': sum([shapiro_p > 0.05, jb_p > 0.05, da_p > 0.05, ks_p > 0.05]) >= 3
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_outliers(self, errors: np.ndarray, params: dict) -> Dict[str, Any]:
        """Analyze outliers in error distribution."""
        try:
            # Z-score method
            z_threshold = params.get('outlier_z_threshold', 3.0)
            z_scores = np.abs((errors - np.mean(errors)) / np.std(errors))
            z_outliers = np.where(z_scores > z_threshold)[0]
            
            # IQR method
            q25, q75 = np.percentile(errors, [25, 75])
            iqr = q75 - q25
            iqr_factor = params.get('outlier_iqr_factor', 1.5)
            iqr_lower = q25 - iqr_factor * iqr
            iqr_upper = q75 + iqr_factor * iqr
            iqr_outliers = np.where((errors < iqr_lower) | (errors > iqr_upper))[0]
            
            # Modified Z-score method (using median)
            median_abs_dev = np.median(np.abs(errors - np.median(errors)))
            modified_z_scores = 0.6745 * (errors - np.median(errors)) / median_abs_dev
            modified_z_outliers = np.where(np.abs(modified_z_scores) > z_threshold)[0]
            
            return {
                'z_score_outliers': {
                    'indices': z_outliers.tolist(),
                    'count': len(z_outliers),
                    'percentage': len(z_outliers) / len(errors) * 100
                },
                'iqr_outliers': {
                    'indices': iqr_outliers.tolist(),
                    'count': len(iqr_outliers),
                    'percentage': len(iqr_outliers) / len(errors) * 100,
                    'lower_bound': iqr_lower,
                    'upper_bound': iqr_upper
                },
                'modified_z_outliers': {
                    'indices': modified_z_outliers.tolist(),
                    'count': len(modified_z_outliers),
                    'percentage': len(modified_z_outliers) / len(errors) * 100
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_symmetry(self, errors: np.ndarray, bin_edges: np.ndarray, bin_counts: np.ndarray) -> Dict[str, Any]:
        """Analyze symmetry of error distribution."""
        try:
            # Statistical symmetry measures
            mean_error = np.mean(errors)
            median_error = np.median(errors)
            
            # Histogram-based symmetry
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            center_bin_idx = np.argmin(np.abs(bin_centers - median_error))
            
            # Compare left and right sides of histogram
            left_counts = bin_counts[:center_bin_idx]
            right_counts = bin_counts[center_bin_idx+1:]
            
            # Pad shorter side with zeros
            max_len = max(len(left_counts), len(right_counts))
            left_padded = np.pad(left_counts, (0, max_len - len(left_counts)), 'constant')
            right_padded = np.pad(right_counts[::-1], (0, max_len - len(right_counts)), 'constant')
            
            # Symmetry correlation
            if len(left_padded) > 0 and len(right_padded) > 0:
                symmetry_corr = np.corrcoef(left_padded, right_padded)[0, 1]
            else:
                symmetry_corr = 0
            
            return {
                'mean_median_diff': mean_error - median_error,
                'symmetry_correlation': symmetry_corr,
                'is_symmetric_statistical': abs(mean_error - median_error) < 0.1 * np.std(errors),
                'is_symmetric_visual': symmetry_corr > 0.8,
                'left_tail_weight': np.sum(left_counts),
                'right_tail_weight': np.sum(right_counts)
            }
        except Exception as e:
            return {'error': str(e)}
    
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
        error_data = self._calculate_errors(ref_data, test_data, self.kwargs)
        
        # Create histogram
        bins = self.kwargs.get('bins', 50)
        y_axis_type = self.kwargs.get('y_axis_type', 'density')
        density = (y_axis_type == 'density')
        
        # Create histogram
        if plot_config.get('show_legend', False):
            n, bins_edges, patches = ax.hist(
                error_data, 
                bins=bins, 
                density=density, 
                alpha=0.7, 
                color='skyblue', 
                edgecolor='black',
                label='Error Distribution'
            )
        else:
            n, bins_edges, patches = ax.hist(
                error_data, 
                bins=bins, 
                density=density, 
                alpha=0.7, 
                color='skyblue', 
                edgecolor='black'
            )
        
        # Add overlay elements
        self._add_distribution_overlays(ax, error_data, plot_config, stats_results)
        
        # Set labels and title
        error_unit = self.kwargs.get('error_unit', 'absolute')
        error_label = "Percentage Error (%)" if error_unit == "percentage" else "Error"
        ax.set_xlabel(error_label)
        ax.set_ylabel('Density' if y_axis_type == 'density' else 'Frequency')
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
        
        if self.kwargs.get("error_unit", "absolute") == "percentage":
            with np.errstate(divide='ignore', invalid='ignore'):
                error = error / np.abs(ref_data) * 100
                error = np.nan_to_num(error, nan=0.0, posinf=0.0, neginf=0.0)
        
        return error
    
    def _format_statistical_text(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format statistical results for text overlay."""
        lines = []
        
        mean_error = stats_results.get('mean_error', np.nan)
        if not np.isnan(mean_error):
            lines.append(f"Mean Error: {mean_error:.3f}")
        
        std_error = stats_results.get('std_error', np.nan)
        if not np.isnan(std_error):
            lines.append(f"Std Error: {std_error:.3f}")
        
        n_samples = stats_results.get('n_samples', 0)
        if n_samples > 0:
            lines.append(f"N: {n_samples}")
        
        return lines
    
    def _get_overlay_functional_properties(self, overlay_id: str, overlay_type: str, 
                                         stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get functional properties for histogram overlays (no arbitrary styling)."""
        properties = {}
        
        if overlay_id == 'show_kde' and overlay_type == 'line':
            properties.update({
                'label': 'KDE'
            })
        elif overlay_id == 'show_zero_line' and overlay_type == 'line':
            properties.update({
                'x_value': 0,
                'label': 'Zero Error'
            })
        elif overlay_id == 'show_legend' and overlay_type == 'legend':
            properties.update({
                'label': 'Legend'
            })
        
        return properties
    
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
    

    
    def _add_distribution_overlays(self, ax, error_data: np.ndarray, 
                                 plot_config: Dict[str, Any], 
                                 stats_results: Dict[str, Any] = None) -> None:
        """Add overlay elements to the histogram."""
        if stats_results is None:
            return
        
        # KDE overlay
        if plot_config.get('show_kde', True):
            self._add_kde_overlay(ax, error_data, plot_config.get('show_legend', False))
        
        # Zero line
        if plot_config.get('show_zero_line', True):
            if plot_config.get('show_legend', False):
                ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
            else:
                ax.axvline(0, color='green', linestyle='-', linewidth=2)
    
    def _add_kde_overlay(self, ax, error_data: np.ndarray, show_legend: bool = True) -> None:
        """Add KDE overlay to histogram."""
        try:
            kde = gaussian_kde(error_data)
            x_range = np.linspace(np.min(error_data), np.max(error_data), 200)
            kde_values = kde(x_range)
            if show_legend:
                ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
            else:
                ax.plot(x_range, kde_values, 'r-', linewidth=2)
        except Exception as e:
            print(f"[ErrorDistribution] Error adding KDE overlay: {e}")
    

    
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