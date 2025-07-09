"""
Cross-Correlation Comparison Method

This module implements cross-correlation analysis for comparing two time series,
including lag analysis, peak detection, and correlation strength assessment.
"""

import numpy as np
from scipy import stats, signal
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class CrossCorrelationComparison(BaseComparison):
    """
    Cross-correlation analysis comparison method.
    
    Performs cross-correlation analysis between two time series to identify
    time delays, synchronization, and correlation patterns.
    """
    
    name = "cross_correlation"
    description = "Cross-correlation analysis for time series comparison with lag detection"
    category = "Time Series"
    version = "1.0.0"
    tags = ["cross-correlation", "time-series", "lag", "synchronization", "signal"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "max_lag", "type": "int", "default": 50, "help": "Maximum lag to compute (in samples)"},
        {"name": "normalize", "type": "bool", "default": True, "help": "Normalize cross-correlation values"},
        {"name": "find_peak", "type": "bool", "default": True, "help": "Find peak correlation and optimal lag"},
        {"name": "detrend", "type": "bool", "default": False, "help": "Detrend signals before correlation"},
        {"name": "window_type", "type": "str", "default": "none", "help": "Window function: none, hann, hamming, blackman"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for significance testing"},
        {"name": "bootstrap_samples", "type": "int", "default": 1000, "help": "Number of bootstrap samples for confidence intervals"},
        {"name": "plot_type", "type": "str", "default": "correlation", "help": "Plot type: correlation, scatter, both"}
    ]
    
    # Plot configuration
    plot_type = "cross_correlation"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_peak': {'default': True, 'label': 'Show Peak', 'tooltip': 'Highlight peak correlation point'},
        'show_confidence_bounds': {'default': False, 'label': 'Show Confidence Bounds', 'tooltip': 'Show confidence bounds for correlation values'},
        'show_lag_range': {'default': False, 'label': 'Show Lag Range', 'tooltip': 'Display lag range information on the plot'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display cross-correlation statistics on the plot'}
    }
    
    def apply(self, ref_data: np.ndarray, test_data: np.ndarray, 
              ref_time: Optional[np.ndarray] = None, 
              test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Main comparison method - orchestrates the cross-correlation analysis.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing cross-correlation results with statistics and plot data
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
            'valid_ratio': valid_ratio,
            'correlation_function': stats_results.get('correlation_function', {}),
            'lags': stats_results.get('lags', np.array([]))
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
        Calculate cross-correlation statistics.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing cross-correlation statistics
        """
        # Preprocess signals
        ref_processed, test_processed = self._preprocess_signals(ref_data, test_data)
        
        # Calculate cross-correlation
        correlation_results = self._calculate_cross_correlation(ref_processed, test_processed)
        
        # Initialize results
        stats_results = {
            'correlation_function': correlation_results,
            'peak_analysis': {},
            'lag_analysis': {},
            'significance_tests': {}
        }
        
        # Extract lags array
        stats_results['lags'] = correlation_results.get('lags', np.array([]))
        
        # Find peak correlation if requested
        if self.kwargs.get('find_peak', True):
            stats_results['peak_analysis'] = self._analyze_peak_correlation(correlation_results)
        
        # Perform lag analysis
        stats_results['lag_analysis'] = self._analyze_lags(correlation_results)
        
        # Test significance
        stats_results['significance_tests'] = self._test_significance(
            ref_processed, test_processed, correlation_results)
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate cross-correlation plot with performance and overlay options.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        plot_type_param = self.kwargs.get('plot_type', 'correlation')
        
        if plot_type_param == 'scatter':
            self._generate_scatter_plot(ax, ref_data, test_data, plot_config, stats_results)
        elif plot_type_param == 'both':
            # Create subplots for both correlation and scatter
            self._generate_combined_plot(ax, ref_data, test_data, plot_config, stats_results)
        else:  # correlation (default)
            self._generate_correlation_plot(ax, ref_data, test_data, plot_config, stats_results)
    
    def _generate_correlation_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                                  plot_config: Dict[str, Any], stats_results: Dict[str, Any]) -> None:
        """Generate cross-correlation function plot."""
        if stats_results is None:
            return
        
        correlation_function = stats_results.get('correlation_function', {})
        lags = correlation_function.get('lags', np.array([]))
        correlation = correlation_function.get('correlation', np.array([]))
        
        if len(lags) == 0 or len(correlation) == 0:
            ax.text(0.5, 0.5, 'No correlation data available', 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Plot correlation function
        ax.plot(lags, correlation, 'b-', linewidth=2, label='Cross-correlation')
        
        # Add zero line
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Highlight peak if available
        if plot_config.get('show_peak', True) and 'peak_analysis' in stats_results:
            peak_analysis = stats_results['peak_analysis']
            peak_lag = peak_analysis.get('peak_lag', 0)
            peak_correlation = peak_analysis.get('peak_correlation', 0)
            
            ax.scatter([peak_lag], [peak_correlation], color='red', s=100, 
                      zorder=5, label=f'Peak: lag={peak_lag}, r={peak_correlation:.3f}')
        
        # Add confidence bounds if requested
        if plot_config.get('show_confidence_bounds', False):
            self._add_confidence_bounds(ax, lags, correlation, stats_results)
        
        # Add statistical results text
        if plot_config.get('show_statistical_results', True):
            self._add_correlation_text(ax, stats_results)
        
        # Set labels and title
        ax.set_xlabel('Lag (samples)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title('Cross-Correlation Function')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', True):
            ax.legend()
    
    def _generate_scatter_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                              plot_config: Dict[str, Any], stats_results: Dict[str, Any]) -> None:
        """Generate scatter plot of the two signals."""
        # Apply performance optimizations
        ref_plot, test_plot = self._apply_performance_optimizations(ref_data, test_data, plot_config)
        
        # Create density plot based on configuration
        self._create_density_plot(ax, ref_plot, test_plot, plot_config)
        
        # Add overlay elements
        self._add_overlay_elements(ax, ref_plot, test_plot, plot_config, stats_results)
        
        # Set labels and title
        ax.set_xlabel('Reference Signal')
        ax.set_ylabel('Test Signal')
        ax.set_title('Signal Scatter Plot')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if requested
        if plot_config.get('show_legend', False):
            ax.legend()
    
    def _generate_combined_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                               plot_config: Dict[str, Any], stats_results: Dict[str, Any]) -> None:
        """Generate combined correlation and scatter plot."""
        # For now, just show correlation plot
        # In a real implementation, you might create subplots
        self._generate_correlation_plot(ax, ref_data, test_data, plot_config, stats_results)
    
    def _preprocess_signals(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess signals before cross-correlation."""
        ref_processed = ref_data.copy()
        test_processed = test_data.copy()
        
        # Detrend if requested
        if self.kwargs.get('detrend', False):
            ref_processed = signal.detrend(ref_processed)
            test_processed = signal.detrend(test_processed)
        
        # Apply window function if specified
        window_type = self.kwargs.get('window_type', 'none')
        if window_type != 'none':
            window = self._get_window(len(ref_processed), window_type)
            ref_processed *= window
            test_processed *= window
        
        return ref_processed, test_processed
    
    def _get_window(self, length: int, window_type: str) -> np.ndarray:
        """Get window function for preprocessing."""
        if window_type == 'hann':
            return signal.windows.hann(length)
        elif window_type == 'hamming':
            return signal.windows.hamming(length)
        elif window_type == 'blackman':
            return signal.windows.blackman(length)
        else:
            return np.ones(length)
    
    def _calculate_cross_correlation(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Calculate cross-correlation function."""
        try:
            max_lag = self.kwargs.get('max_lag', 50)
            normalize = self.kwargs.get('normalize', True)
            
            # Ensure max_lag doesn't exceed signal length
            max_lag = min(max_lag, len(ref_data) - 1)
            
            # Calculate cross-correlation using scipy
            correlation = signal.correlate(test_data, ref_data, mode='full')
            
            # Create lag array
            lags = signal.correlation_lags(len(test_data), len(ref_data), mode='full')
            
            # Limit to specified max_lag
            center_idx = len(correlation) // 2
            start_idx = max(0, center_idx - max_lag)
            end_idx = min(len(correlation), center_idx + max_lag + 1)
            
            correlation = correlation[start_idx:end_idx]
            lags = lags[start_idx:end_idx]
            
            # Normalize if requested
            if normalize:
                # Normalize by the product of signal norms
                norm_factor = np.sqrt(np.sum(ref_data**2) * np.sum(test_data**2))
                if norm_factor > 0:
                    correlation = correlation / norm_factor
            
            return {
                'correlation': correlation,
                'lags': lags,
                'normalized': normalize,
                'max_lag': max_lag
            }
        except Exception as e:
            return {
                'error': str(e),
                'correlation': np.array([]),
                'lags': np.array([])
            }
    
    def _analyze_peak_correlation(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze peak correlation and optimal lag."""
        try:
            correlation = correlation_results.get('correlation', np.array([]))
            lags = correlation_results.get('lags', np.array([]))
            
            if len(correlation) == 0:
                return {'error': 'No correlation data available'}
            
            # Find peak correlation
            peak_idx = np.argmax(np.abs(correlation))
            peak_correlation = correlation[peak_idx]
            peak_lag = lags[peak_idx]
            
            # Find all peaks above threshold
            threshold = 0.1 * np.max(np.abs(correlation))
            peaks, properties = signal.find_peaks(np.abs(correlation), height=threshold)
            
            return {
                'peak_correlation': peak_correlation,
                'peak_lag': peak_lag,
                'peak_index': peak_idx,
                'all_peaks': {
                    'indices': peaks.tolist(),
                    'lags': lags[peaks].tolist(),
                    'correlations': correlation[peaks].tolist()
                },
                'threshold': threshold
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_lags(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze lag characteristics."""
        try:
            correlation = correlation_results.get('correlation', np.array([]))
            lags = correlation_results.get('lags', np.array([]))
            
            if len(correlation) == 0:
                return {'error': 'No correlation data available'}
            
            # Find zero-lag correlation
            zero_lag_idx = np.argmin(np.abs(lags))
            zero_lag_correlation = correlation[zero_lag_idx]
            
            # Calculate lag statistics
            lag_stats = {
                'zero_lag_correlation': zero_lag_correlation,
                'zero_lag_index': zero_lag_idx,
                'max_positive_lag': np.max(lags),
                'max_negative_lag': np.min(lags),
                'lag_range': np.max(lags) - np.min(lags)
            }
            
            # Analyze correlation decay
            decay_analysis = self._analyze_correlation_decay(correlation, lags)
            lag_stats.update(decay_analysis)
            
            return lag_stats
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_correlation_decay(self, correlation: np.ndarray, lags: np.ndarray) -> Dict[str, Any]:
        """Analyze how correlation decays with lag."""
        try:
            # Find central peak
            center_idx = np.argmin(np.abs(lags))
            max_corr = np.abs(correlation[center_idx])
            
            # Calculate correlation at different lag distances
            lag_distances = np.abs(lags)
            
            # Find where correlation drops to certain thresholds
            thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
            decay_lags = {}
            
            for threshold in thresholds:
                target_corr = threshold * max_corr
                # Find first lag where correlation drops below threshold
                below_threshold = np.where(np.abs(correlation) < target_corr)[0]
                if len(below_threshold) > 0:
                    decay_lags[f'lag_{int(threshold*100)}pct'] = lag_distances[below_threshold[0]]
                else:
                    decay_lags[f'lag_{int(threshold*100)}pct'] = np.nan
            
            return {
                'decay_analysis': decay_lags,
                'max_correlation': max_corr
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _test_significance(self, ref_data: np.ndarray, test_data: np.ndarray, 
                          correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Test statistical significance of correlation."""
        try:
            correlation = correlation_results.get('correlation', np.array([]))
            
            if len(correlation) == 0:
                return {'error': 'No correlation data available'}
            
            # Simple significance test based on sample size
            n = len(ref_data)
            
            # Critical correlation for significance (approximate)
            confidence_level = self.kwargs.get('confidence_level', 0.95)
            alpha = 1 - confidence_level
            
            # Approximate critical value for correlation
            critical_r = stats.norm.ppf(1 - alpha/2) / np.sqrt(n - 2)
            
            # Find significant correlations
            significant_mask = np.abs(correlation) > critical_r
            n_significant = np.sum(significant_mask)
            
            # Peak significance
            peak_correlation = np.max(np.abs(correlation))
            peak_significant = peak_correlation > critical_r
            
            return {
                'critical_correlation': critical_r,
                'confidence_level': confidence_level,
                'n_significant_lags': n_significant,
                'peak_correlation': peak_correlation,
                'peak_significant': peak_significant,
                'significant_mask': significant_mask.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _add_confidence_bounds(self, ax, lags: np.ndarray, correlation: np.ndarray, 
                              stats_results: Dict[str, Any]) -> None:
        """Add confidence bounds to correlation plot."""
        try:
            significance_tests = stats_results.get('significance_tests', {})
            critical_r = significance_tests.get('critical_correlation', 0)
            
            if critical_r > 0:
                ax.fill_between(lags, -critical_r, critical_r, alpha=0.2, color='gray', 
                               label=f'95% Confidence Bounds (±{critical_r:.3f})')
        except Exception as e:
            print(f"Error adding confidence bounds: {e}")
    
    def _add_correlation_text(self, ax, stats_results: Dict[str, Any]) -> None:
        """Add correlation statistics as text on plot."""
        try:
            text_lines = self._format_statistical_text(stats_results)
            
            if text_lines:
                text = '\n'.join(text_lines)
                ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='white', alpha=0.8))
        except Exception as e:
            print(f"Error adding correlation text: {e}")
    
    def _format_statistical_text(self, stats_results: Dict[str, Any], plot_config: Dict[str, Any] = None) -> List[str]:
        """Format statistical results for display on plot."""
        text_lines = []
        
        if plot_config is None:
            plot_config = {}
        
        # Add peak correlation information
        if 'peak_analysis' in stats_results:
            peak_analysis = stats_results['peak_analysis']
            peak_corr = peak_analysis.get('peak_correlation', np.nan)
            peak_lag = peak_analysis.get('peak_lag', np.nan)
            
            if not np.isnan(peak_corr) and not np.isnan(peak_lag):
                text_lines.append(f"Peak: r={peak_corr:.3f} at lag={peak_lag}")
        
        # Add zero-lag correlation
        if 'lag_analysis' in stats_results:
            lag_analysis = stats_results['lag_analysis']
            zero_lag_corr = lag_analysis.get('zero_lag_correlation', np.nan)
            
            if not np.isnan(zero_lag_corr):
                text_lines.append(f"Zero-lag: r={zero_lag_corr:.3f}")
        
        # Add significance information
        if 'significance_tests' in stats_results:
            sig_tests = stats_results['significance_tests']
            peak_significant = sig_tests.get('peak_significant', False)
            critical_r = sig_tests.get('critical_correlation', np.nan)
            
            if not np.isnan(critical_r):
                text_lines.append(f"Critical r: ±{critical_r:.3f}")
                text_lines.append(f"Peak significant: {'Yes' if peak_significant else 'No'}")
        
        # Add lag range information if requested
        if plot_config.get('show_lag_range', False) and 'lag_analysis' in stats_results:
            lag_analysis = stats_results['lag_analysis']
            max_pos_lag = lag_analysis.get('max_positive_lag', np.nan)
            max_neg_lag = lag_analysis.get('max_negative_lag', np.nan)
            lag_range = lag_analysis.get('lag_range', np.nan)
            
            if not np.isnan(max_pos_lag) and not np.isnan(max_neg_lag):
                text_lines.append(f"Lag range: {max_neg_lag} to {max_pos_lag}")
            if not np.isnan(lag_range):
                text_lines.append(f"Total range: {lag_range}")
        
        # Add correlation function information
        if 'correlation_function' in stats_results:
            corr_func = stats_results['correlation_function']
            normalized = corr_func.get('normalized', False)
            max_lag = corr_func.get('max_lag', 0)
            
            text_lines.append(f"Normalized: {'Yes' if normalized else 'No'}")
            text_lines.append(f"Max lag: ±{max_lag}")
        
        return text_lines
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Cross-Correlation Analysis",
            "description": "Analyzes temporal relationships and synchronization between time series",
            "interpretation": {
                "peak_correlation": "Maximum correlation value indicates strength of relationship",
                "peak_lag": "Lag at peak correlation shows time delay between signals",
                "zero_lag": "Correlation at zero lag shows simultaneous relationship",
                "significance": "Statistical significance of correlation peaks",
                "decay": "How correlation decreases with increasing lag"
            },
            "use_cases": [
                "Time series synchronization analysis",
                "Signal delay detection and measurement",
                "Pattern matching in temporal data",
                "Lead-lag relationship identification"
            ],
            "tips": [
                "Use normalization for comparing different signal amplitudes",
                "Consider detrending for signals with trends",
                "Look for multiple peaks indicating periodic relationships",
                "Check significance bounds to identify meaningful correlations",
                "Optimal lag indicates time delay between signals"
            ]
        }