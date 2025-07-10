"""
Time Lag Cross Correlation Comparison Method

This module implements time lag cross-correlation analysis between two data channels,
identifying optimal time alignment and temporal relationships between signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy import stats
from typing import Dict, Any, Optional, Tuple, List
from comparison.base_comparison import BaseComparison
from comparison.comparison_registry import register_comparison

@register_comparison
class TimeLagCrossCorrelationComparison(BaseComparison):
    """
    Time lag cross-correlation comparison method.
    
    Analyzes temporal relationships between signals by computing cross-correlation
    at different time lags to identify optimal alignment and phase relationships.
    """
    
    name = "time_lag_cross_correlation"
    description = "Cross-correlation analysis for temporal alignment and phase relationship detection"
    category = "Alignment"
    version = "1.0.0"
    tags = ["alignment", "cross-correlation", "lag", "temporal", "phase"]
    
    # Parameters following mixer/steps pattern
    params = [
        {"name": "max_lag_seconds", "type": "float", "default": 5.0, "help": "Maximum lag to compute (in seconds)"},
        {"name": "sampling_rate", "type": "float", "default": 100.0, "help": "Sampling rate (Hz)"},
        {"name": "normalize", "type": "bool", "default": True, "help": "Normalize cross-correlation values"},
        {"name": "detrend", "type": "bool", "default": True, "help": "Remove mean from signals before correlation"},
        {"name": "confidence_level", "type": "float", "default": 0.95, "help": "Confidence level for statistical significance"}
    ]
    
    # Plot configuration
    plot_type = "line"
    
    # Overlay options - defines which overlay controls should be shown in the wizard
    overlay_options = {
        'show_peak_lag': {'default': True, 'label': 'Show Peak Lag', 'tooltip': 'Show vertical line at peak correlation lag'},
        'show_zero_lag': {'default': True, 'label': 'Show Zero Lag', 'tooltip': 'Show vertical line at zero lag reference'},
        'show_significance_threshold': {'default': True, 'label': 'Show Significance Threshold', 'tooltip': 'Show statistical significance threshold'},
        'show_confidence_bands': {'default': False, 'label': 'Show Confidence Bands', 'tooltip': 'Show confidence bands around correlation'},
        'highlight_significant_lags': {'default': True, 'label': 'Highlight Significant Lags', 'tooltip': 'Highlight statistically significant correlation peaks'},
        'show_statistical_results': {'default': True, 'label': 'Show Statistical Results', 'tooltip': 'Display correlation statistics on the plot'}
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
        
        # Compute cross-correlation
        lags_seconds, xcorr_values = self._compute_cross_correlation(ref_clean, test_clean)
        
        # Calculate statistics
        stats_results = self.calculate_stats(ref_clean, test_clean, ref_time, test_time)
        
        # Prepare plot data
        plot_data = {
            'lags': lags_seconds,
            'xcorr': xcorr_values,
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
        Calculate cross-correlation statistics.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array  
            ref_time: Optional time array for reference data
            test_time: Optional time array for test data
            
        Returns:
            Dictionary containing cross-correlation statistics
        """
        # Compute cross-correlation
        lags_seconds, xcorr_values = self._compute_cross_correlation(ref_data, test_data)
        
        # Initialize results
        stats_results = {
            'peak_analysis': {},
            'correlation_stats': {},
            'significance': {}
        }
        
        # Peak analysis
        stats_results['peak_analysis'] = self._analyze_correlation_peaks(lags_seconds, xcorr_values)
        
        # Overall correlation statistics
        stats_results['correlation_stats'] = self._compute_correlation_statistics(xcorr_values)
        
        # Statistical significance
        stats_results['significance'] = self._compute_significance_analysis(
            ref_data, test_data, lags_seconds, xcorr_values)
        
        # Lag distribution analysis
        stats_results['lag_distribution'] = self._analyze_lag_distribution(lags_seconds, xcorr_values)
        
        return stats_results
    
    def generate_plot(self, ax, ref_data: np.ndarray, test_data: np.ndarray, 
                     plot_config: Dict[str, Any] = None, 
                     stats_results: Dict[str, Any] = None) -> None:
        """
        Generate cross-correlation plot with overlays.
        
        Args:
            ax: Matplotlib axes object to plot on
            ref_data: Reference data array
            test_data: Test data array
            plot_config: Plot configuration dictionary
            stats_results: Statistical results from calculate_stats method
        """
        if plot_config is None:
            plot_config = {}
        
        # Compute cross-correlation
        lags_seconds, xcorr_values = self._compute_cross_correlation(ref_data, test_data)
        
        # Create main cross-correlation plot
        self._create_cross_correlation_plot(ax, lags_seconds, xcorr_values, plot_config)
        
        # Add overlay elements
        self._add_correlation_overlays(ax, lags_seconds, xcorr_values, plot_config, stats_results)
        
        # Set labels and title
        ax.set_xlabel('Time Lag (seconds)')
        ax.set_ylabel('Cross-Correlation')
        ax.set_title('Time Lag Cross-Correlation Analysis')
        
        # Add grid if requested
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
        
        # Add legend if there are overlays
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best')
    
    def _compute_cross_correlation(self, ref_data: np.ndarray, test_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cross-correlation between reference and test data."""
        # Get parameters
        max_lag_s = self.kwargs.get("max_lag_seconds", 5.0)
        fs = self.kwargs.get("sampling_rate", 100.0)
        normalize = self.kwargs.get("normalize", True)
        detrend = self.kwargs.get("detrend", True)
        
        # Prepare signals
        ref_signal = ref_data.copy()
        test_signal = test_data.copy()
        
        if detrend:
            ref_signal = ref_signal - np.mean(ref_signal)
            test_signal = test_signal - np.mean(test_signal)
        
        # Compute cross-correlation
        n = len(ref_signal)
        max_lag_samples = int(max_lag_s * fs)
        
        # Use scipy's correlate function
        xcorr_full = correlate(test_signal, ref_signal, mode="full")
        
        # Create lag array
        lags = np.arange(-n + 1, n)
        
        # Limit to specified lag range
        valid_mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
        lags = lags[valid_mask]
        xcorr = xcorr_full[valid_mask]
        
        # Convert lags to seconds
        lags_seconds = lags / fs
        
        # Normalize if requested
        if normalize:
            xcorr = xcorr / (np.std(ref_signal) * np.std(test_signal) * len(ref_signal))
        
        return lags_seconds, xcorr
    
    def _analyze_correlation_peaks(self, lags_seconds: np.ndarray, xcorr_values: np.ndarray) -> Dict[str, Any]:
        """Analyze correlation peaks and find optimal lag."""
        try:
            # Find peak correlation
            peak_idx = np.argmax(np.abs(xcorr_values))
            peak_lag = lags_seconds[peak_idx]
            peak_correlation = xcorr_values[peak_idx]
            
            # Find all significant peaks
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(np.abs(xcorr_values), 
                                         height=0.1 * np.max(np.abs(xcorr_values)),
                                         distance=max(1, len(xcorr_values) // 20))
            
            # Analyze multiple peaks
            peak_lags = lags_seconds[peaks]
            peak_values = xcorr_values[peaks]
            
            return {
                'peak_lag_seconds': peak_lag,
                'peak_correlation': peak_correlation,
                'peak_correlation_abs': np.abs(peak_correlation),
                'zero_lag_correlation': xcorr_values[np.argmin(np.abs(lags_seconds))],
                'n_significant_peaks': len(peaks),
                'all_peak_lags': peak_lags.tolist(),
                'all_peak_values': peak_values.tolist()
            }
        except Exception as e:
            return {
                'peak_lag_seconds': np.nan,
                'peak_correlation': np.nan,
                'peak_correlation_abs': np.nan,
                'zero_lag_correlation': np.nan,
                'n_significant_peaks': 0,
                'all_peak_lags': [],
                'all_peak_values': [],
                'error': str(e)
            }
    
    def _compute_correlation_statistics(self, xcorr_values: np.ndarray) -> Dict[str, float]:
        """Compute overall correlation statistics."""
        try:
            return {
                'max_correlation': np.max(xcorr_values),
                'min_correlation': np.min(xcorr_values),
                'max_abs_correlation': np.max(np.abs(xcorr_values)),
                'mean_correlation': np.mean(xcorr_values),
                'std_correlation': np.std(xcorr_values),
                'rms_correlation': np.sqrt(np.mean(xcorr_values**2)),
                'energy': np.sum(xcorr_values**2)
            }
        except Exception as e:
            return {
                'max_correlation': np.nan, 'min_correlation': np.nan,
                'max_abs_correlation': np.nan, 'mean_correlation': np.nan,
                'std_correlation': np.nan, 'rms_correlation': np.nan,
                'energy': np.nan, 'error': str(e)
            }
    
    def _compute_significance_analysis(self, ref_data: np.ndarray, test_data: np.ndarray,
                                     lags_seconds: np.ndarray, xcorr_values: np.ndarray) -> Dict[str, Any]:
        """Compute statistical significance of correlation values."""
        try:
            n_samples = len(ref_data)
            confidence_level = self.kwargs.get('confidence_level', 0.95)
            
            # Theoretical significance threshold for white noise
            # Based on the assumption that under null hypothesis, correlation follows normal distribution
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            significance_threshold = z_score / np.sqrt(n_samples)
            
            # Count significant correlations
            significant_mask = np.abs(xcorr_values) > significance_threshold
            n_significant = np.sum(significant_mask)
            
            # Effective degrees of freedom (accounting for autocorrelation)
            # Simplified estimation
            effective_n = n_samples / (1 + 2 * np.sum(np.abs(xcorr_values)))
            effective_threshold = z_score / np.sqrt(effective_n)
            
            return {
                'significance_threshold': significance_threshold,
                'effective_threshold': effective_threshold,
                'n_significant': n_significant,
                'significance_percentage': n_significant / len(xcorr_values) * 100,
                'confidence_level': confidence_level,
                'effective_dof': effective_n
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_lag_distribution(self, lags_seconds: np.ndarray, xcorr_values: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of correlation values across lags."""
        try:
            # Weighted statistics using correlation values as weights
            weights = np.abs(xcorr_values)
            weights = weights / np.sum(weights)
            
            weighted_mean_lag = np.average(lags_seconds, weights=weights)
            weighted_std_lag = np.sqrt(np.average((lags_seconds - weighted_mean_lag)**2, weights=weights))
            
            # Symmetry analysis
            positive_lags = lags_seconds[lags_seconds > 0]
            negative_lags = lags_seconds[lags_seconds < 0]
            
            if len(positive_lags) > 0 and len(negative_lags) > 0:
                pos_corr = xcorr_values[lags_seconds > 0]
                neg_corr = xcorr_values[lags_seconds < 0]
                
                asymmetry = np.mean(np.abs(pos_corr)) - np.mean(np.abs(neg_corr))
            else:
                asymmetry = np.nan
            
            return {
                'weighted_mean_lag': weighted_mean_lag,
                'weighted_std_lag': weighted_std_lag,
                'lag_range': np.ptp(lags_seconds),
                'asymmetry': asymmetry,
                'positive_lag_dominance': np.mean(np.abs(xcorr_values[lags_seconds > 0])) if len(positive_lags) > 0 else np.nan,
                'negative_lag_dominance': np.mean(np.abs(xcorr_values[lags_seconds < 0])) if len(negative_lags) > 0 else np.nan
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _create_cross_correlation_plot(self, ax, lags_seconds: np.ndarray, xcorr_values: np.ndarray, 
                                      plot_config: Dict[str, Any]) -> None:
        """Create the main cross-correlation plot."""
        # Main correlation plot
        ax.plot(lags_seconds, xcorr_values, 'b-', linewidth=1.5, label='Cross-Correlation')
        
        # Fill positive and negative areas with different colors
        ax.fill_between(lags_seconds, 0, xcorr_values, where=(xcorr_values >= 0), 
                       color='blue', alpha=0.3, interpolate=True)
        ax.fill_between(lags_seconds, 0, xcorr_values, where=(xcorr_values < 0), 
                       color='red', alpha=0.3, interpolate=True)
    
    def _add_correlation_overlays(self, ax, lags_seconds: np.ndarray, xcorr_values: np.ndarray, 
                                 plot_config: Dict[str, Any], 
                                 stats_results: Dict[str, Any] = None) -> None:
        """Add overlay elements to the correlation plot."""
        if stats_results is None:
            return
        
        # Peak lag line
        if plot_config.get('show_peak_lag', True) and 'peak_analysis' in stats_results:
            self._add_peak_lag_line(ax, stats_results['peak_analysis'])
        
        # Zero lag reference line
        if plot_config.get('show_zero_lag', True):
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Zero Lag')
        
        # Significance threshold
        if plot_config.get('show_significance_threshold', True) and 'significance' in stats_results:
            self._add_significance_threshold(ax, stats_results['significance'])
        
        # Highlight significant lags
        if plot_config.get('highlight_significant_lags', True) and 'significance' in stats_results:
            self._highlight_significant_lags(ax, lags_seconds, xcorr_values, stats_results['significance'])
        
        # Confidence bands
        if plot_config.get('show_confidence_bands', False) and 'significance' in stats_results:
            self._add_confidence_bands(ax, lags_seconds, stats_results['significance'])
        
        # Statistical results text
        if plot_config.get('show_statistical_results', True):
            self._add_statistical_text_box(ax, stats_results, plot_config)
    
    def _add_peak_lag_line(self, ax, peak_analysis: Dict[str, Any]) -> None:
        """Add vertical line at peak correlation lag."""
        try:
            peak_lag = peak_analysis.get('peak_lag_seconds', np.nan)
            peak_correlation = peak_analysis.get('peak_correlation', np.nan)
            
            if not np.isnan(peak_lag) and not np.isnan(peak_correlation):
                ax.axvline(x=peak_lag, color='red', linestyle='-', linewidth=2, 
                          label=f'Peak Lag: {peak_lag:.3f}s')
                ax.plot(peak_lag, peak_correlation, 'ro', markersize=8, zorder=5)
        except Exception as e:
            print(f"[TimeLagCrossCorrelation] Error adding peak lag line: {e}")
    
    def _add_significance_threshold(self, ax, significance: Dict[str, Any]) -> None:
        """Add significance threshold lines."""
        try:
            threshold = significance.get('significance_threshold', np.nan)
            
            if not np.isnan(threshold):
                ax.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, 
                          label=f'Significance: ±{threshold:.3f}')
                ax.axhline(y=-threshold, color='green', linestyle='--', alpha=0.7)
        except Exception as e:
            print(f"[TimeLagCrossCorrelation] Error adding significance threshold: {e}")
    
    def _highlight_significant_lags(self, ax, lags_seconds: np.ndarray, xcorr_values: np.ndarray, 
                                   significance: Dict[str, Any]) -> None:
        """Highlight statistically significant correlation values."""
        try:
            threshold = significance.get('significance_threshold', np.nan)
            
            if not np.isnan(threshold):
                significant_mask = np.abs(xcorr_values) > threshold
                if np.any(significant_mask):
                    ax.scatter(lags_seconds[significant_mask], xcorr_values[significant_mask], 
                              color='orange', s=30, alpha=0.8, label='Significant', zorder=4)
        except Exception as e:
            print(f"[TimeLagCrossCorrelation] Error highlighting significant lags: {e}")
    
    def _add_confidence_bands(self, ax, lags_seconds: np.ndarray, significance: Dict[str, Any]) -> None:
        """Add confidence bands around zero correlation."""
        try:
            threshold = significance.get('significance_threshold', np.nan)
            
            if not np.isnan(threshold):
                ax.fill_between(lags_seconds, -threshold, threshold, 
                               alpha=0.2, color='gray', label='Confidence Band')
        except Exception as e:
            print(f"[TimeLagCrossCorrelation] Error adding confidence bands: {e}")
    
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
        
        # Peak analysis
        if 'peak_analysis' in stats_results:
            peak_stats = stats_results['peak_analysis']
            peak_lag = peak_stats.get('peak_lag_seconds', np.nan)
            peak_corr = peak_stats.get('peak_correlation', np.nan)
            zero_lag_corr = peak_stats.get('zero_lag_correlation', np.nan)
            
            if not np.isnan(peak_lag):
                text_lines.append(f"Peak Lag: {peak_lag:.3f}s")
            if not np.isnan(peak_corr):
                text_lines.append(f"Peak Corr: {peak_corr:.3f}")
            if not np.isnan(zero_lag_corr):
                text_lines.append(f"Zero Lag: {zero_lag_corr:.3f}")
        
        # Correlation statistics
        if 'correlation_stats' in stats_results:
            corr_stats = stats_results['correlation_stats']
            max_abs_corr = corr_stats.get('max_abs_correlation', np.nan)
            
            if not np.isnan(max_abs_corr):
                text_lines.append(f"Max |Corr|: {max_abs_corr:.3f}")
        
        # Significance
        if 'significance' in stats_results:
            sig_stats = stats_results['significance']
            n_significant = sig_stats.get('n_significant', 0)
            sig_pct = sig_stats.get('significance_percentage', 0)
            text_lines.append(f"Significant: {n_significant} ({sig_pct:.1f}%)")
        
        return text_lines
    
    @classmethod
    def get_comparison_guidance(cls):
        """Get guidance for this comparison method."""
        return {
            "title": "Time Lag Cross-Correlation",
            "description": "Analyzes temporal relationships and optimal alignment between signals",
            "interpretation": {
                "positive_lag": "Test signal leads reference signal (test occurs earlier)",
                "negative_lag": "Reference signal leads test signal (reference occurs earlier)",
                "zero_lag": "Signals are optimally aligned at zero lag",
                "peak_correlation": "Strength of relationship at optimal lag",
                "significance": "Statistical confidence in correlation values"
            },
            "use_cases": [
                "Signal alignment and synchronization",
                "Lag detection between related measurements",
                "Phase relationship analysis",
                "Temporal causality assessment"
            ],
            "tips": [
                "Positive lag means test signal leads reference",
                "Use appropriate max_lag_seconds to avoid edge effects",
                "High correlation at zero lag suggests good synchronization",
                "Multiple peaks may indicate periodic relationships",
                "Check significance threshold to avoid false positives"
            ]
        }