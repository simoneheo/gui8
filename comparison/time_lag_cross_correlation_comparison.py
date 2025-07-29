"""
Time Lag Cross-Correlation Comparison Method

This module implements time lag cross-correlation analysis between two data channels,
finding optimal time delays and measuring correlation strength at different lags.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class TimeLagCrossCorrelationComparison(BaseComparison):
    """
    Time lag cross-correlation analysis comparison method.
    
    Computes cross-correlation between two signals at different time lags
    to find optimal alignment delays and assess signal relationships.
    """
    
    name = "time_lag_cross_correlation"
    description = "Compute time lag cross-correlation to find optimal delays and signal relationships"
    category = "Signal Processing"
    tags = ["cross-correlation", "time-delay", "signal-alignment", "lag-analysis"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "max_lag", "type": "int", "default": 100, "help": "Maximum lag in samples to analyze"},
        {"name": "normalize", "type": "bool", "default": True, "help": "Normalize cross-correlation coefficients"},
        {"name": "mode", "type": "str", "default": "full", "options": ["full", "valid", "same"], "help": "Cross-correlation mode"},
        {"name": "detrend", "type": "bool", "default": True, "help": "Remove DC component before correlation"},
        {"name": "lag_units", "type": "str", "default": "samples", "options": ["samples", "time"], "help": "Units for lag axis"},
        {"name": "sampling_rate", "type": "float", "default": 1.0, "help": "Sampling rate for time conversion (Hz)"},
    ]
    
    # Plot configuration
    plot_type = "scatter"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay = {
        'peak_marker': {'default': True, 'label': 'Peak Marker', 'help': 'Mark optimal lag position', 'type': 'vline'},
        'zero_lag_line': {'default': True, 'label': 'Zero Lag Line', 'help': 'Reference line at zero lag', 'type': 'vline'},
        'correlation_thresholds': {'default': False, 'label': 'Correlation Thresholds', 'help': 'Horizontal lines at correlation levels', 'type': 'hline'},
        'statistical_results': {'default': True, 'label': 'Statistical Results', 'help': 'Cross-correlation statistics on the plot', 'type': 'text'},
    }

    def plot_script(self, ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> tuple:
        """
        Core plotting transformation for time lag cross-correlation analysis
        
        This computes cross-correlation at different time lags between the signals.
        
        Args:
            ref_data: Reference signal (cleaned of NaN/infinite values)
            test_data: Test signal (cleaned of NaN/infinite values)
            params: Method parameters dictionary
            
        Returns:
            tuple: (x_data, y_data, metadata)
                x_data: Time lags (samples or time units)
                y_data: Cross-correlation coefficients
                metadata: Plot configuration dictionary
        """
        
        # Get parameters
        max_lag = params.get("max_lag", 100)
        normalize = params.get("normalize", True)
        mode = params.get("mode", "full")
        detrend = params.get("detrend", True)
        lag_units = params.get("lag_units", "samples")
        sampling_rate = params.get("sampling_rate", 1.0)
        
        # Ensure equal length signals
        min_len = min(len(ref_data), len(test_data))
        ref_signal = ref_data[:min_len]
        test_signal = test_data[:min_len]
        
        # Detrend signals if requested
        if detrend:
            ref_signal = signal.detrend(ref_signal)
            test_signal = signal.detrend(test_signal)
        
        # Compute cross-correlation
        if normalize:
            # Normalized cross-correlation
            correlation = signal.correlate(test_signal, ref_signal, mode=mode)
            # Normalize by the product of signal norms
            norm_factor = np.sqrt(np.sum(ref_signal**2) * np.sum(test_signal**2))
            if norm_factor > 0:
                correlation = correlation / norm_factor
        else:
            # Raw cross-correlation
            correlation = signal.correlate(test_signal, ref_signal, mode=mode)
        
        # Generate lag array
        if mode == "full":
            lags = signal.correlation_lags(len(test_signal), len(ref_signal), mode=mode)
        else:
            # For other modes, generate appropriate lag array
            correlation_len = len(correlation)
            center = correlation_len // 2
            lags = np.arange(-center, correlation_len - center)
        
        # Limit to max_lag if specified
        if max_lag > 0:
            valid_indices = np.abs(lags) <= max_lag
            lags = lags[valid_indices]
            correlation = correlation[valid_indices]
        
        # Convert lag units if requested
        if lag_units == "time" and sampling_rate > 0:
            lags = lags / sampling_rate
            x_label = f"Time Lag (seconds, fs={sampling_rate}Hz)"
        else:
            x_label = "Lag (samples)"
        
        # Prepare metadata for plotting
        metadata = {
            'x_label': x_label,
            'y_label': 'Cross-Correlation Coefficient' if normalize else 'Cross-Correlation',
            'title': 'Time Lag Cross-Correlation',
            'lag_units': lag_units,
            'sampling_rate': sampling_rate,
            'max_lag': max_lag,
            'signal_length': min_len
        }

        x_data = lags
        y_data = correlation

        return x_data, y_data, metadata
    
    def stats_script(self, x_data: List[float], y_data: List[float], 
                    ref_data: np.ndarray, test_data: np.ndarray, params: dict) -> Dict[str, Any]:
        """
        Statistical calculations for time lag cross-correlation analysis
        
        Args:
            x_data: Lag values
            y_data: Cross-correlation coefficients
            ref_data: Original reference data
            test_data: Original test data
            params: Method parameters dictionary
            
        Returns:
            Dictionary containing statistical results
        """
        
        stats_results = {}
        
        # Convert to numpy arrays
        lags = np.array(x_data)
        correlation = np.array(y_data)
        
        if len(correlation) == 0:
            return {
                'peak_lag': 0,
                'peak_correlation': 0,
                'sample_size': 0
            }
        
        # Find peak correlation
        peak_idx = np.argmax(np.abs(correlation))
        peak_lag = lags[peak_idx]
        peak_correlation = correlation[peak_idx]
        
        # Find zero lag correlation
        zero_lag_idx = np.argmin(np.abs(lags))
        zero_lag_correlation = correlation[zero_lag_idx]
        
        # Statistical measures
        mean_correlation = np.mean(correlation)
        std_correlation = np.std(correlation)
        max_correlation = np.max(correlation)
        min_correlation = np.min(correlation)
        
        # Correlation statistics
        stats_results.update({
            'peak_lag': float(peak_lag),
            'peak_correlation': float(peak_correlation),
            'zero_lag_correlation': float(zero_lag_correlation),
            'mean_correlation': float(mean_correlation),
            'std_correlation': float(std_correlation),
            'max_correlation': float(max_correlation),
            'min_correlation': float(min_correlation),
            'correlation_range': float(max_correlation - min_correlation),
            'sample_size': len(correlation)
        })
        
        # Significance analysis
        # Simple threshold-based significance (can be improved with proper statistical tests)
        significant_threshold = 0.3  # Common threshold for correlation significance
        significant_lags = lags[np.abs(correlation) > significant_threshold]
        
        stats_results.update({
            'significant_threshold': significant_threshold,
            'significant_lag_count': len(significant_lags),
            'significant_percentage': (len(significant_lags) / len(lags)) * 100 if len(lags) > 0 else 0
        })
        
        # Delay analysis
        delay_direction = "Test leads" if peak_lag > 0 else "Reference leads" if peak_lag < 0 else "Synchronized"
        
        stats_results.update({
            'delay_direction': delay_direction,
            'delay_magnitude': abs(float(peak_lag)),
            'lag_range': [float(np.min(lags)), float(np.max(lags))]
        })
        
        return stats_results
    
    def _get_peak_marker(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for peak marker overlay."""
        try:
            peak_lag = stats_results.get('peak_lag', 0)
            
            return {
                'x': [peak_lag],
                'label': f'Peak at lag={peak_lag:.3f}'
            }
        except Exception as e:
            print(f"[TimeLagCrossCorrelationOverlay] Error getting peak marker data: {e}")
            return {}
    
    def _get_zero_lag_line(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for zero lag line overlay."""
        try:
            return {
                'x': [0],
                'label': 'Zero Lag'
            }
        except Exception as e:
            print(f"[TimeLagCrossCorrelationOverlay] Error getting zero lag line data: {e}")
            return {}
    
    def _get_correlation_thresholds(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for correlation threshold lines overlay."""
        try:
            thresholds = [0.3, 0.5, 0.7, 0.9, -0.3, -0.5, -0.7, -0.9]
            
            return {
                'y': thresholds,
                'label': 'Correlation Thresholds'
            }
        except Exception as e:
            print(f"[TimeLagCrossCorrelationOverlay] Error getting correlation thresholds data: {e}")
            return {}
    
    def _get_statistical_results(self, stats_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get data for statistical results text overlay."""
        
        # Only return the most informative statistics for text overlay
        essential_stats = {
            'peak_lag': stats_results.get('peak_lag'),
            'peak_correlation': stats_results.get('peak_correlation'),
            'zero_lag_correlation': stats_results.get('zero_lag_correlation'),
            'delay_direction': stats_results.get('delay_direction'),
        }
        
        return essential_stats

    def _create_overlays(self, ref_data: np.ndarray, test_data: np.ndarray, 
                        stats_results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Create overlay definitions for time lag cross-correlation analysis.
        
        Returns a dictionary of overlay definitions that will be rendered by the base class.
        Each overlay definition contains type, main data, and style information.
        
        Args:
            ref_data: Reference data arrays
            test_data: Test data array
            stats_results: Statistical results from stats_script method
            params: Overlay configuration dictionary
            
        Returns:
            Dictionary of overlay definitions
        """
        
        peak_marker = {
            'type': 'vline',
            'show': params.get('peak_marker', True),
            'label': 'Peak Marker',
            'main': self._get_peak_marker(stats_results)
        }

        zero_lag_line = {
            'type': 'vline',
            'show': params.get('zero_lag_line', True),
            'label': 'Zero Lag Line',
            'main': self._get_zero_lag_line(stats_results)
        }

        correlation_thresholds = {
            'type': 'hline',
            'show': params.get('correlation_thresholds', False),
            'label': 'Correlation Thresholds',
            'main': self._get_correlation_thresholds(stats_results)
        }

        statistical_results = {
            'type': 'text',
            'show': params.get('statistical_results', True),
            'label': 'Statistical Results',
            'main': self._get_statistical_results(stats_results)
        }

        overlays = {
            'peak_marker': peak_marker,
            'zero_lag_line': zero_lag_line,
            'correlation_thresholds': correlation_thresholds,
            'statistical_results': statistical_results
        }
                
        return overlays
    

    @classmethod
    def get_description(cls) -> str:
        """
        Get a description of this comparison method for display in the wizard console.
        
        Returns:
            String description explaining what this comparison method does
        """
        return """Time Lag Cross-Correlation: Measures correlation between signals at different time delays.

• Computes cross-correlation at various time lags between signals
• Finds optimal delay for maximum correlation strength
• Determines which signal leads or lags the other
• Normalized correlation removes amplitude dependence
""" 