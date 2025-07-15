"""
Time Lag Cross Correlation Comparison Method

This module implements time lag cross correlation analysis between two data channels,
computing correlation coefficients at various time lags to find optimal alignment.
"""

import numpy as np
from scipy import stats
from scipy.signal import correlate
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class TimeLagCrossCorrelationComparison(BaseComparison):
    """
    Time lag cross correlation analysis comparison method.
    
    Computes correlation coefficients at various time lags to determine
    optimal temporal alignment between two signals.
    """
    
    name = "time_lag_cross_correlation"
    description = "Analyze time lag cross correlation to find optimal signal alignment"
    category = "Time Series"
    tags = ["scatter", "correlation", "time_lag", "cross_correlation", "alignment"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "max_lag", "type": "int", "default": 100, "min": 1, "max": 1000, "help": "Maximum lag to test (in samples)"},
        {"name": "correlation_type", "type": "str", "default": "pearson", "options": ["pearson", "spearman", "kendall"], "help": "Type of correlation to compute"},
        {"name": "normalize", "type": "bool", "default": True, "help": "Normalize cross-correlation by autocorrelation"},
        {"name": "detrend", "type": "bool", "default": True, "help": "Remove linear trend before correlation"},
        {"name": "window_size", "type": "int", "default": 0, "min": 0, "max": 1000, "help": "Window size for windowed correlation (0 = full signal)"},
        {"name": "overlap", "type": "float", "default": 0.5, "min": 0.0, "max": 0.9, "step": 0.1, "help": "Overlap between windows"},
        {"name": "significance_test", "type": "bool", "default": True, "help": "Perform significance testing"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "min": 0.5, "max": 0.99, "step": 0.01, "help": "Confidence level for significance"}
    ]
    
    # Plot configuration
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'max_correlation_point': {'default': True, 'label': 'Max Correlation Point', 'tooltip': 'Highlight point of maximum correlation', 'type': 'marker'},
        'significance_threshold': {'default': True, 'label': 'Significance Threshold', 'tooltip': 'Horizontal lines showing significance thresholds', 'type': 'line'},
        'zero_lag_line': {'default': True, 'label': 'Zero Lag Line', 'tooltip': 'Vertical line at zero lag', 'type': 'line'},
        'confidence_interval': {'default': True, 'label': 'Confidence Interval', 'tooltip': 'Shaded confidence interval around correlation', 'type': 'fill'},
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'tooltip': 'Display correlation statistics and optimal lag', 'type': 'text'}
    }
    
    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for time lag cross correlation analysis
        
        This defines what gets plotted for scatter visualization.
        
        Args:
            ref_data: Reference measurements (cleaned of NaN/infinite values)
            test_data: Test measurements (cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (lag_values, correlation_values, confidence_intervals, metadata)
                lag_values: Time lag values for X-axis
                correlation_values: Correlation coefficients for Y-axis
                confidence_intervals: Confidence intervals for correlations
                metadata: Plot configuration dictionary
        """
        
        # Get parameters
        max_lag = params.get("max_lag", 100)
        correlation_type = params.get("correlation_type", "pearson")
        normalize = params.get("normalize", True)
        detrend = params.get("detrend", True)
        window_size = params.get("window_size", 0)
        overlap = params.get("overlap", 0.5)
        significance_test = params.get("significance_test", True)
        confidence_level = params.get("confidence_level", 0.95)
        
        # Ensure data is same length
        min_length = min(len(ref_data), len(test_data))
        ref_data = ref_data[:min_length]
        test_data = test_data[:min_length]
        
        # Detrend if requested
        if detrend:
            ref_data = self._detrend_signal(ref_data)
            test_data = self._detrend_signal(test_data)
        
        # Calculate cross-correlation
        if window_size > 0:
            # Windowed cross-correlation
            lag_values, correlation_values, confidence_intervals = self._windowed_cross_correlation(
                ref_data, test_data, max_lag, correlation_type, window_size, overlap, 
                normalize, significance_test, confidence_level
            )
        else:
            # Full signal cross-correlation
            lag_values, correlation_values, confidence_intervals = self._full_cross_correlation(
                ref_data, test_data, max_lag, correlation_type, normalize, 
                significance_test, confidence_level
            )
        
        # Create metadata
        metadata = {
            'max_lag': max_lag,
            'correlation_type': correlation_type,
            'normalize': normalize,
            'detrend': detrend,
            'window_size': window_size,
            'overlap': overlap,
            'significance_test': significance_test,
            'confidence_level': confidence_level,
            'n_samples': len(ref_data),
            'n_lags': len(lag_values),
            'xlabel': 'Time Lag (samples)',
            'ylabel': f'{correlation_type.title()} Correlation'
        }
        
        return lag_values, correlation_values, confidence_intervals, metadata
    
    def stats_script(self, lag_values: np.ndarray, correlation_values: np.ndarray, 
                    confidence_intervals: np.ndarray, ref_data: np.ndarray, params: dict) -> dict:
        """
        Statistical analysis for time lag cross correlation
        
        Args:
            lag_values: Time lag values from plot_script
            correlation_values: Correlation coefficients from plot_script
            confidence_intervals: Confidence intervals from plot_script
            ref_data: Reference data (unused but required for interface)
            params: Method parameters
            
        Returns:
            Dictionary containing statistical results
        """
        
        # Find optimal lag and correlation
        max_corr_idx = np.argmax(np.abs(correlation_values))
        optimal_lag = lag_values[max_corr_idx]
        max_correlation = correlation_values[max_corr_idx]
        
        # Calculate correlation statistics
        mean_correlation = np.mean(correlation_values)
        std_correlation = np.std(correlation_values)
        
        # Find zero-lag correlation
        zero_lag_idx = np.argmin(np.abs(lag_values))
        zero_lag_correlation = correlation_values[zero_lag_idx]
        
        # Calculate significance threshold if requested
        significance_threshold = None
        if params.get("significance_test", True):
            confidence_level = params.get("confidence_level", 0.95)
            alpha = 1 - confidence_level
            n_samples = len(ref_data) if ref_data is not None else 100
            
            # Critical correlation value for significance
            if n_samples > 3:
                # Use t-distribution for significance test
                t_critical = stats.t.ppf(1 - alpha/2, n_samples - 2)
                significance_threshold = t_critical / np.sqrt(n_samples - 2 + t_critical**2)
            else:
                significance_threshold = 0.5
        
        # Calculate lag statistics
        positive_lags = lag_values[lag_values > 0]
        negative_lags = lag_values[lag_values < 0]
        
        positive_corr = correlation_values[lag_values > 0]
        negative_corr = correlation_values[lag_values < 0]
        
        # Asymmetry analysis
        if len(positive_corr) > 0 and len(negative_corr) > 0:
            positive_max = np.max(np.abs(positive_corr))
            negative_max = np.max(np.abs(negative_corr))
            asymmetry = (positive_max - negative_max) / (positive_max + negative_max)
        else:
            asymmetry = 0
        
        # Calculate confidence interval width
        if confidence_intervals is not None and len(confidence_intervals) > 0:
            mean_ci_width = np.mean(confidence_intervals)
        else:
            mean_ci_width = 0
        
        # Lag range analysis
        significant_lags = []
        if significance_threshold is not None:
            significant_mask = np.abs(correlation_values) > significance_threshold
            significant_lags = lag_values[significant_mask].tolist()
        
        stats_results = {
            'optimal_lag': int(optimal_lag),
            'max_correlation': float(max_correlation),
            'zero_lag_correlation': float(zero_lag_correlation),
            'mean_correlation': float(mean_correlation),
            'std_correlation': float(std_correlation),
            'significance_threshold': float(significance_threshold) if significance_threshold is not None else None,
            'asymmetry': float(asymmetry),
            'mean_ci_width': float(mean_ci_width),
            'significant_lags': significant_lags,
            'n_significant_lags': len(significant_lags),
            'correlation_range': float(np.max(correlation_values) - np.min(correlation_values)),
            'lag_range': [int(np.min(lag_values)), int(np.max(lag_values))],
            'is_significant': bool(significance_threshold is not None and abs(max_correlation) > significance_threshold)
        }
        
        return stats_results
    
    def _detrend_signal(self, signal: np.ndarray) -> np.ndarray:
        """Remove linear trend from signal"""
        x = np.arange(len(signal))
        p = np.polyfit(x, signal, 1)
        trend = np.polyval(p, x)
        return signal - trend
    
    def _full_cross_correlation(self, ref_data: np.ndarray, test_data: np.ndarray, 
                               max_lag: int, correlation_type: str, normalize: bool,
                               significance_test: bool, confidence_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate cross-correlation for full signal"""
        
        # Limit max_lag to reasonable value
        max_lag = min(max_lag, len(ref_data) // 2)
        
        # Create lag array
        lag_values = np.arange(-max_lag, max_lag + 1)
        correlation_values = np.zeros(len(lag_values))
        confidence_intervals = np.zeros(len(lag_values))
        
        # Calculate correlation for each lag
        for i, lag in enumerate(lag_values):
            if lag == 0:
                # Zero lag case
                ref_segment = ref_data
                test_segment = test_data
            elif lag > 0:
                # Positive lag: test leads ref
                ref_segment = ref_data[lag:]
                test_segment = test_data[:-lag]
            else:
                # Negative lag: ref leads test
                ref_segment = ref_data[:lag]
                test_segment = test_data[-lag:]
            
            # Calculate correlation
            if len(ref_segment) > 1 and len(test_segment) > 1:
                if correlation_type == "pearson":
                    corr, p_value = stats.pearsonr(ref_segment, test_segment)
                elif correlation_type == "spearman":
                    corr, p_value = stats.spearmanr(ref_segment, test_segment)
                elif correlation_type == "kendall":
                    corr, p_value = stats.kendalltau(ref_segment, test_segment)
                else:
                    corr, p_value = stats.pearsonr(ref_segment, test_segment)
                
                correlation_values[i] = corr
                
                # Calculate confidence interval
                if significance_test:
                    n = len(ref_segment)
                    if n > 3:
                        # Fisher z-transformation for confidence interval
                        z_corr = np.arctanh(corr)
                        z_std = 1 / np.sqrt(n - 3)
                        z_alpha = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                        
                        z_lower = z_corr - z_alpha * z_std
                        z_upper = z_corr + z_alpha * z_std
                        
                        ci_lower = np.tanh(z_lower)
                        ci_upper = np.tanh(z_upper)
                        
                        confidence_intervals[i] = (ci_upper - ci_lower) / 2
                    else:
                        confidence_intervals[i] = 0.5
                else:
                    confidence_intervals[i] = 0
            else:
                correlation_values[i] = 0
                confidence_intervals[i] = 0
        
        # Normalize if requested
        if normalize:
            # Normalize by autocorrelation at zero lag
            ref_auto = np.corrcoef(ref_data, ref_data)[0, 1]
            test_auto = np.corrcoef(test_data, test_data)[0, 1]
            norm_factor = np.sqrt(ref_auto * test_auto)
            if norm_factor > 0:
                correlation_values = correlation_values / norm_factor
        
        return lag_values, correlation_values, confidence_intervals
    
    def _windowed_cross_correlation(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                   max_lag: int, correlation_type: str, window_size: int, 
                                   overlap: float, normalize: bool, significance_test: bool,
                                   confidence_level: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate cross-correlation using windowed approach"""
        
        # Calculate step size based on overlap
        step_size = int(window_size * (1 - overlap))
        if step_size < 1:
            step_size = 1
        
        # Calculate number of windows
        n_samples = len(ref_data)
        n_windows = (n_samples - window_size) // step_size + 1
        
        # Limit max_lag to reasonable value
        max_lag = min(max_lag, window_size // 2)
        
        # Create lag array
        lag_values = np.arange(-max_lag, max_lag + 1)
        
        # Initialize arrays to store results
        all_correlations = np.zeros((n_windows, len(lag_values)))
        
        # Calculate correlation for each window
        for w in range(n_windows):
            start_idx = w * step_size
            end_idx = start_idx + window_size
            
            ref_window = ref_data[start_idx:end_idx]
            test_window = test_data[start_idx:end_idx]
            
            # Calculate correlation for each lag in this window
            for i, lag in enumerate(lag_values):
                if lag == 0:
                    ref_segment = ref_window
                    test_segment = test_window
                elif lag > 0:
                    ref_segment = ref_window[lag:]
                    test_segment = test_window[:-lag]
                else:
                    ref_segment = ref_window[:lag]
                    test_segment = test_window[-lag:]
                
                if len(ref_segment) > 1 and len(test_segment) > 1:
                    if correlation_type == "pearson":
                        corr, _ = stats.pearsonr(ref_segment, test_segment)
                    elif correlation_type == "spearman":
                        corr, _ = stats.spearmanr(ref_segment, test_segment)
                    elif correlation_type == "kendall":
                        corr, _ = stats.kendalltau(ref_segment, test_segment)
                    else:
                        corr, _ = stats.pearsonr(ref_segment, test_segment)
                    
                    all_correlations[w, i] = corr
        
        # Average correlations across windows
        correlation_values = np.mean(all_correlations, axis=0)
        
        # Calculate confidence intervals
        if significance_test and n_windows > 1:
            confidence_intervals = stats.sem(all_correlations, axis=0) * stats.t.ppf(1 - (1 - confidence_level) / 2, n_windows - 1)
        else:
            confidence_intervals = np.zeros(len(lag_values))
        
        # Normalize if requested
        if normalize:
            # Normalize by average zero-lag correlation
            zero_lag_idx = np.argmin(np.abs(lag_values))
            zero_lag_corr = correlation_values[zero_lag_idx]
            if zero_lag_corr > 0:
                correlation_values = correlation_values / zero_lag_corr
        
        return lag_values, correlation_values, confidence_intervals 