"""
Statistical Comparison Method

This module implements comprehensive statistical testing for comparing two datasets,
including t-tests, Wilcoxon tests, Kolmogorov-Smirnov tests, and other statistical measures.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class StatisticalComparison(BaseComparison):
    """
    Statistical testing comparison method.
    
    Performs comprehensive statistical tests to compare two datasets,
    including parametric and non-parametric tests for differences.
    """
    
    name = "statistical"
    description = "Comprehensive statistical testing including t-tests, Wilcoxon tests, and distribution comparisons"
    category = "Statistical"
    version = "1.0.0"
    tags = ["statistical", "hypothesis", "testing", "parametric", "nonparametric"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "test_type", "type": "str", "default": "auto", "help": "Statistical test type: auto, ttest, wilcoxon, ks_test, all"},
        {"name": "alpha", "type": "float", "default": 0.05, "help": "Significance level for statistical tests"},
        {"name": "paired", "type": "bool", "default": True, "help": "Whether to perform paired tests"},
        {"name": "equal_var", "type": "bool", "default": True, "help": "Assume equal variances for t-test"},
        {"name": "normality_test", "type": "bool", "default": True, "help": "Test for normality before choosing test"},
        {"name": "effect_size", "type": "bool", "default": True, "help": "Calculate effect size measures"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for confidence intervals"},
        {"name": "bootstrap_samples", "type": "int", "default": 1000, "help": "Number of bootstrap samples for confidence intervals"}
    ]
    
    # Plot configuration
    plot_type = "statistical"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_effect_size': {'default': True, 'label': 'Show Effect Size', 'tooltip': 'Display effect size measures on the plot'},
        'show_confidence_intervals': {'default': True, 'label': 'Show Confidence Intervals', 'tooltip': 'Show confidence intervals for statistical tests'},
        'show_distribution_overlay': {'default': False, 'label': 'Show Distribution Overlay', 'tooltip': 'Show distribution histograms for both datasets'},
        'show_mean_lines': {'default': True, 'label': 'Show Mean Lines', 'tooltip': 'Show horizontal lines at mean values'},
        'highlight_outliers': {'default': False, 'label': 'Highlight Outliers', 'tooltip': 'Highlight outlier points on the plot'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the statistical analysis.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing statistical results with statistics and plot data
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
        Calculate comprehensive statistical measures.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing statistical test results
        """
        # Initialize results
        stats_results = {
            'descriptive': {},
            'tests': {},
            'effect_sizes': {},
            'confidence_intervals': {}
        }
        
        # Descriptive statistics
        stats_results['descriptive'] = self._calculate_descriptive_stats(ref_data, test_data)
        
        # Test for normality if requested
        normality_results = None
        if self.kwargs.get('normality_test', True):
            normality_results = self._test_normality(ref_data, test_data)
            stats_results['normality'] = normality_results
        
        # Determine which tests to perform
        test_type = self.kwargs.get('test_type', 'auto')
        
        if test_type == 'auto':
            # Choose test based on normality
            if normality_results and normality_results.get('both_normal', False):
                test_type = 'ttest'
            else:
                test_type = 'wilcoxon'
        
        # Perform statistical tests
        if test_type in ['ttest', 'all']:
            stats_results['tests']['t_test'] = self._perform_t_test(ref_data, test_data)
        
        if test_type in ['wilcoxon', 'all']:
            stats_results['tests']['wilcoxon'] = self._perform_wilcoxon_test(ref_data, test_data)
        
        if test_type in ['ks_test', 'all']:
            stats_results['tests']['ks_test'] = self._perform_ks_test(ref_data, test_data)
        
        # Calculate effect sizes if requested
        if self.kwargs.get('effect_size', True):
            stats_results['effect_sizes'] = self._calculate_effect_sizes(ref_data, test_data)
        
        # Calculate confidence intervals
        confidence_level = self.kwargs.get('confidence_level', 0.95)
        stats_results['confidence_intervals'] = self._calculate_confidence_intervals(
            ref_data, test_data, confidence_level)
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate statistical comparison plot with performance and overlay options.
        
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
        ax.set_title('Statistical Comparison')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', False):
            ax.legend()
    
    def _validate_input_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validates and cleans input data.
        Ensures data is 1D numpy array and handles potential NaNs.
        """
        if not isinstance(ref_data, np.ndarray) or not isinstance(test_data, np.ndarray):
            raise ValueError("Input data must be numpy arrays.")
        
        if ref_data.ndim != 1 or test_data.ndim != 1:
            raise ValueError("Input data must be 1D arrays.")
        
        # Remove NaNs from both arrays
        valid_mask = np.logical_and(np.isfinite(ref_data), np.isfinite(test_data))
        ref_data = ref_data[valid_mask]
        test_data = test_data[valid_mask]
        
        return ref_data, test_data
    
    def _remove_invalid_data(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Removes invalid data (NaNs) and calculates valid ratio.
        Returns cleaned data and valid ratio.
        """
        initial_len = len(ref_data)
        valid_mask = np.logical_and(np.isfinite(ref_data), np.isfinite(test_data))
        ref_data = ref_data[valid_mask]
        test_data = test_data[valid_mask]
        valid_ratio = len(ref_data) / initial_len if initial_len > 0 else 0.0
        return ref_data, test_data, valid_ratio
    
    def _apply_performance_optimizations(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                         plot_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies performance optimizations based on plot_config.
        Returns optimized data for plotting.
        """
        if plot_config.get('density_display', 'scatter') == 'scatter':
            return ref_data, test_data
        
        # For density plots, we might want to use hexbin or KDE
        # This is a placeholder; actual implementation would depend on the plot_config
        # For now, we'll just return the original data
        return ref_data, test_data
    
    def _create_density_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                             plot_config: Dict[str, Any]) -> None:
        """
        Creates a density plot (scatter, hexbin, KDE) based on plot_config.
        """
        display_type = plot_config.get('density_display', 'scatter')
        
        if display_type == 'scatter':
            ax.scatter(ref_data, test_data, alpha=0.5, s=10, label='Data Points')
        elif display_type == 'hexbin':
            ax.hexbin(ref_data, test_data, gridsize=30, cmap='Blues', label='Hexbin')
        elif display_type == 'kde':
            # KDE plot requires scipy.stats.gaussian_kde
            try:
                from scipy.stats import gaussian_kde
                kde_ref = gaussian_kde(ref_data)
                kde_test = gaussian_kde(test_data)
                
                x_vals = np.linspace(min(ref_data.min(), test_data.min()), 
                                     max(ref_data.max(), test_data.max()), 100)
                
                ax.plot(x_vals, kde_ref(x_vals), 'b-', label='Ref KDE')
                ax.plot(x_vals, kde_test(x_vals), 'g-', label='Test KDE')
            except ImportError:
                print("KDE plot requires scipy.stats.gaussian_kde. Please install scipy.")
                ax.scatter(ref_data, test_data, alpha=0.5, s=10, label='Data Points')
        else:
            ax.scatter(ref_data, test_data, alpha=0.5, s=10, label='Data Points')
    
    def _add_overlay_elements(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                              plot_config: Dict[str, Any], stats_results: Dict[str, Any]) -> None:
        """
        Adds overlay elements to the plot based on plot_config and stats_results.
        """
        # Add mean lines
        if plot_config.get('show_mean_lines', True):
            ref_mean = np.mean(ref_data)
            test_mean = np.mean(test_data)
            ax.axhline(y=ref_mean, color='blue', linestyle='--', alpha=0.7, label=f'Ref Mean: {ref_mean:.3f}')
            ax.axhline(y=test_mean, color='green', linestyle='--', alpha=0.7, label=f'Test Mean: {test_mean:.3f}')
        
        # Add confidence intervals
        # Check both parameter names for backward compatibility
        show_ci = plot_config.get('show_confidence_intervals', plot_config.get('confidence_interval', False))
        if show_ci:
            self._add_confidence_intervals(ax, ref_data, test_data, plot_config)
        
        # Add distribution overlay
        if plot_config.get('show_distribution_overlay', False):
            self._add_distribution_overlay(ax, ref_data, test_data)
        
        # Highlight outliers
        if plot_config.get('highlight_outliers', False):
            self._highlight_outliers(ax, ref_data, test_data)
        
        # Add statistical results to plot
        if plot_config.get('show_effect_size', True):
            self._add_statistical_text(ax, stats_results)
    
    def _calculate_descriptive_stats(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate descriptive statistics for both datasets."""
        return {
            'reference': {
                'mean': np.mean(ref_data),
                'std': np.std(ref_data, ddof=1),
                'median': np.median(ref_data),
                'min': np.min(ref_data),
                'max': np.max(ref_data),
                'q25': np.percentile(ref_data, 25),
                'q75': np.percentile(ref_data, 75),
                'skewness': stats.skew(ref_data),
                'kurtosis': stats.kurtosis(ref_data)
            },
            'test': {
                'mean': np.mean(test_data),
                'std': np.std(test_data, ddof=1),
                'median': np.median(test_data),
                'min': np.min(test_data),
                'max': np.max(test_data),
                'q25': np.percentile(test_data, 25),
                'q75': np.percentile(test_data, 75),
                'skewness': stats.skew(test_data),
                'kurtosis': stats.kurtosis(test_data)
            }
        }
    
    def _test_normality(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk test."""
        try:
            # Shapiro-Wilk test for reference data
            ref_stat, ref_p = stats.shapiro(ref_data)
            ref_normal = ref_p > self.kwargs.get('alpha', 0.05)
            
            # Shapiro-Wilk test for test data
            test_stat, test_p = stats.shapiro(test_data)
            test_normal = test_p > self.kwargs.get('alpha', 0.05)
            
            return {
                'reference': {
                    'statistic': ref_stat,
                    'p_value': ref_p,
                    'normal': ref_normal
                },
                'test': {
                    'statistic': test_stat,
                    'p_value': test_p,
                    'normal': test_normal
                },
                'both_normal': ref_normal and test_normal
            }
        except Exception as e:
            return {
                'error': str(e),
                'both_normal': False
            }
    
    def _perform_t_test(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Perform t-test (paired or independent)."""
        try:
            paired = self.kwargs.get('paired', True)
            equal_var = self.kwargs.get('equal_var', True)
            
            if paired:
                # Paired t-test
                statistic, p_value = stats.ttest_rel(ref_data, test_data)
                test_name = "Paired t-test"
            else:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(ref_data, test_data, equal_var=equal_var)
                test_name = "Independent t-test"
            
            alpha = self.kwargs.get('alpha', 0.05)
            significant = p_value < alpha
            
            return {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'significant': significant,
                'alpha': alpha,
                'interpretation': f"{'Significant' if significant else 'Not significant'} difference (p={p_value:.4f})"
            }
        except Exception as e:
            return {
                'error': str(e),
                'significant': False
            }
    
    def _perform_wilcoxon_test(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Perform Wilcoxon test (signed-rank or rank-sum)."""
        try:
            paired = self.kwargs.get('paired', True)
            
            if paired:
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(ref_data, test_data)
                test_name = "Wilcoxon signed-rank test"
            else:
                # Mann-Whitney U test (Wilcoxon rank-sum)
                statistic, p_value = stats.mannwhitneyu(ref_data, test_data, alternative='two-sided')
                test_name = "Mann-Whitney U test"
            
            alpha = self.kwargs.get('alpha', 0.05)
            significant = p_value < alpha
            
            return {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'significant': significant,
                'alpha': alpha,
                'interpretation': f"{'Significant' if significant else 'Not significant'} difference (p={p_value:.4f})"
            }
        except Exception as e:
            return {
                'error': str(e),
                'significant': False
            }
    
    def _perform_ks_test(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Perform Kolmogorov-Smirnov test."""
        try:
            statistic, p_value = stats.ks_2samp(ref_data, test_data)
            
            alpha = self.kwargs.get('alpha', 0.05)
            significant = p_value < alpha
            
            return {
                'test_name': 'Kolmogorov-Smirnov test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': significant,
                'alpha': alpha,
                'interpretation': f"{'Significant' if significant else 'Not significant'} difference in distributions (p={p_value:.4f})"
            }
        except Exception as e:
            return {
                'error': str(e),
                'significant': False
            }
    
    def _calculate_effect_sizes(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate effect size measures."""
        try:
            # Cohen's d
            pooled_std = np.sqrt(((len(ref_data) - 1) * np.var(ref_data, ddof=1) + 
                                 (len(test_data) - 1) * np.var(test_data, ddof=1)) / 
                                (len(ref_data) + len(test_data) - 2))
            cohens_d = (np.mean(test_data) - np.mean(ref_data)) / pooled_std
            
            # Interpret Cohen's d
            if abs(cohens_d) < 0.2:
                d_interpretation = "Small effect"
            elif abs(cohens_d) < 0.5:
                d_interpretation = "Medium effect"
            elif abs(cohens_d) < 0.8:
                d_interpretation = "Large effect"
            else:
                d_interpretation = "Very large effect"
            
            # Cliff's delta (non-parametric effect size)
            cliffs_delta = self._calculate_cliffs_delta(ref_data, test_data)
            
            return {
                'cohens_d': {
                    'value': cohens_d,
                    'interpretation': d_interpretation
                },
                'cliffs_delta': cliffs_delta
            }
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def _calculate_cliffs_delta(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate Cliff's delta effect size."""
        try:
            n1, n2 = len(ref_data), len(test_data)
            
            # Calculate all pairwise comparisons
            greater = 0
            less = 0
            
            for x in ref_data:
                for y in test_data:
                    if y > x:
                        greater += 1
                    elif y < x:
                        less += 1
            
            delta = (greater - less) / (n1 * n2)
            
            # Interpret Cliff's delta
            if abs(delta) < 0.147:
                interpretation = "Negligible effect"
            elif abs(delta) < 0.33:
                interpretation = "Small effect"
            elif abs(delta) < 0.474:
                interpretation = "Medium effect"
            else:
                interpretation = "Large effect"
            
            return {
                'value': delta,
                'interpretation': interpretation
            }
        except Exception as e:
            return {
                'error': str(e),
                'value': np.nan
            }
    
    def _calculate_confidence_intervals(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                      confidence_level: float) -> Dict[str, Any]:
        """Calculate confidence intervals for means."""
        try:
            alpha = 1 - confidence_level
            
            # Confidence interval for reference data mean
            ref_mean = np.mean(ref_data)
            ref_sem = stats.sem(ref_data)
            ref_ci = stats.t.interval(confidence_level, len(ref_data) - 1, 
                                     loc=ref_mean, scale=ref_sem)
            
            # Confidence interval for test data mean
            test_mean = np.mean(test_data)
            test_sem = stats.sem(test_data)
            test_ci = stats.t.interval(confidence_level, len(test_data) - 1, 
                                      loc=test_mean, scale=test_sem)
            
            # Confidence interval for difference in means
            diff_mean = test_mean - ref_mean
            diff_sem = np.sqrt(ref_sem**2 + test_sem**2)
            diff_df = len(ref_data) + len(test_data) - 2
            diff_ci = stats.t.interval(confidence_level, diff_df, 
                                      loc=diff_mean, scale=diff_sem)
            
            return {
                'reference_mean': ref_ci,
                'test_mean': test_ci,
                'difference': diff_ci,
                'confidence_level': confidence_level
            }
        except Exception as e:
            return {
                'error': str(e)
            }
    
    def _format_statistical_text(self, stats_results: Dict[str, Any]) -> List[str]:
        """Format statistical results for display on plot."""
        text_lines = []
        
        # Add test results
        if 'tests' in stats_results:
            for test_name, test_result in stats_results['tests'].items():
                if 'p_value' in test_result and not np.isnan(test_result['p_value']):
                    p_val = test_result['p_value']
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    text_lines.append(f"{test_name}: p={p_val:.4f} {sig}")
        
        # Add effect sizes
        if 'effect_sizes' in stats_results:
            effect_sizes = stats_results['effect_sizes']
            if 'cohens_d' in effect_sizes:
                d_val = effect_sizes['cohens_d'].get('value', np.nan)
                if not np.isnan(d_val):
                    text_lines.append(f"Cohen's d: {d_val:.3f}")
        
        # Add descriptive statistics
        if 'descriptive' in stats_results:
            desc = stats_results['descriptive']
            ref_mean = desc.get('reference', {}).get('mean', np.nan)
            test_mean = desc.get('test', {}).get('mean', np.nan)
            if not np.isnan(ref_mean) and not np.isnan(test_mean):
                text_lines.append(f"Ref mean: {ref_mean:.3f}")
                text_lines.append(f"Test mean: {test_mean:.3f}")
        
        return text_lines
    
    def _add_confidence_intervals(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                                plot_config: Dict[str, Any]) -> None:
        """Add confidence intervals to plot."""
        try:
            confidence_level = plot_config.get('confidence_level', 0.95)
            
            # Calculate confidence intervals
            ref_mean = np.mean(ref_data)
            ref_sem = stats.sem(ref_data)
            ref_ci = stats.t.interval(confidence_level, len(ref_data)-1, loc=ref_mean, scale=ref_sem)
            
            test_mean = np.mean(test_data)
            test_sem = stats.sem(test_data)
            test_ci = stats.t.interval(confidence_level, len(test_data)-1, loc=test_mean, scale=test_sem)
            
            # Plot confidence intervals
            ax.fill_between([0.8, 1.2], [ref_ci[0], ref_ci[0]], [ref_ci[1], ref_ci[1]], 
                           alpha=0.3, color='blue', label=f'Ref {confidence_level*100:.0f}% CI')
            ax.fill_between([1.8, 2.2], [test_ci[0], test_ci[0]], [test_ci[1], test_ci[1]], 
                           alpha=0.3, color='green', label=f'Test {confidence_level*100:.0f}% CI')
        except Exception as e:
            print(f"Error adding confidence intervals: {e}")
    
    def _add_distribution_overlay(self, ax, ref_data: np.ndarray, test_data: np.ndarray) -> None:
        """Add distribution overlay to plot."""
        try:
            # Create histogram overlay
            ax.hist(ref_data, alpha=0.3, color='blue', bins=20, density=True, label='Ref Distribution')
            ax.hist(test_data, alpha=0.3, color='green', bins=20, density=True, label='Test Distribution')
        except Exception as e:
            print(f"Error adding distribution overlay: {e}")
    
    def _highlight_outliers(self, ax, ref_data: np.ndarray, test_data: np.ndarray) -> None:
        """Highlight outliers on plot."""
        try:
            # Use IQR method to detect outliers
            def get_outliers(data):
                q1, q3 = np.percentile(data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                return data[(data < lower_bound) | (data > upper_bound)]
            
            ref_outliers = get_outliers(ref_data)
            test_outliers = get_outliers(test_data)
            
            if len(ref_outliers) > 0:
                ax.scatter([1] * len(ref_outliers), ref_outliers, color='red', s=50, alpha=0.8, marker='x')
            if len(test_outliers) > 0:
                ax.scatter([2] * len(test_outliers), test_outliers, color='red', s=50, alpha=0.8, marker='x')
        except Exception as e:
            print(f"Error highlighting outliers: {e}")
    
    def _add_statistical_text(self, ax, stats_results: Dict[str, Any]) -> None:
        """Add statistical results as text on plot."""
        try:
            text_lines = []
            
            # Add test results
            if 'tests' in stats_results:
                for test_name, test_result in stats_results['tests'].items():
                    if 'p_value' in test_result and not np.isnan(test_result['p_value']):
                        p_val = test_result['p_value']
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        text_lines.append(f"{test_name}: p={p_val:.4f} {sig}")
            
            # Add effect sizes
            if 'effect_sizes' in stats_results:
                effect_sizes = stats_results['effect_sizes']
                if 'cohens_d' in effect_sizes:
                    d_val = effect_sizes['cohens_d'].get('value', np.nan)
                    if not np.isnan(d_val):
                        text_lines.append(f"Cohen's d: {d_val:.3f}")
            
            # Add descriptive statistics
            if 'descriptive' in stats_results:
                desc = stats_results['descriptive']
                ref_mean = desc.get('reference', {}).get('mean', np.nan)
                test_mean = desc.get('test', {}).get('mean', np.nan)
                if not np.isnan(ref_mean) and not np.isnan(test_mean):
                    text_lines.append(f"Ref mean: {ref_mean:.3f}")
                    text_lines.append(f"Test mean: {test_mean:.3f}")
            
            # Display text
            if text_lines:
                text = '\n'.join(text_lines)
                ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Error adding statistical text: {e}")
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Statistical Testing",
            "description": "Comprehensive statistical tests for comparing two datasets",
            "interpretation": {
                "t_test": "Parametric test for mean differences (assumes normality)",
                "wilcoxon": "Non-parametric test for median differences",
                "ks_test": "Tests for differences in entire distributions",
                "effect_size": "Magnitude of difference (Cohen's d, Cliff's delta)",
                "p_value": "Probability of observing difference by chance"
            },
            "use_cases": [
                "Hypothesis testing for mean/median differences",
                "Comparing distributions between groups",
                "Validating measurement methods",
                "A/B testing and experimental comparisons"
            ],
            "tips": [
                "Check normality before choosing parametric vs non-parametric tests",
                "Consider effect size alongside p-values",
                "Use paired tests when measurements are related",
                "Interpret confidence intervals for practical significance",
                "Consider multiple testing corrections for multiple comparisons"
            ]
        } 