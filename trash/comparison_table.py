# comparison_table.py

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ConstantInputWarning
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

class AlignmentMethod(Enum):
    """Enumeration of alignment methods."""
    INDEX = "index"
    TIME_INTERP_REF = "time_interp_ref"
    TIME_INTERP_TEST = "time_interp_test"
    CROSS_CORRELATION = "cross_correlation"
    DTW = "dynamic_time_warping"

class InterpolationMethod(Enum):
    """Enumeration of interpolation methods."""
    LINEAR = "linear"
    CUBIC = "cubic"
    NEAREST = "nearest"
    ZERO = "zero"
    SLINEAR = "slinear"
    QUADRATIC = "quadratic"

@dataclass
class AlignmentResult:
    """Container for alignment results."""
    aligned_x: np.ndarray
    ref_y_aligned: np.ndarray
    test_y_aligned: np.ndarray
    method: str
    parameters: Dict
    alignment_offset: Optional[float] = None
    correlation_score: Optional[float] = None

@dataclass
class ComparisonStats:
    """Container for comparison statistics."""
    # Basic statistics
    n_samples: int
    ref_mean: float
    test_mean: float
    ref_std: float
    test_std: float
    
    # Correlation statistics
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    
    # Agreement statistics
    mean_bias: float
    bias_std: float
    limits_of_agreement: Tuple[float, float]
    
    # Residual statistics
    rmse: float
    mae: float
    mape: float
    
    # Regression statistics
    slope: float
    intercept: float
    r_squared: float
    
    # Outlier statistics
    outlier_indices: List[int]
    outlier_percentage: float

class ComparisonTable:
    """
    A comprehensive table for signal comparison analysis.
    Provides methods for alignment, statistical comparison, and overlay analysis.
    """
    
    def __init__(self, ref_channel, test_channel, signal_bus=None):
        """
        Initialize comparison table with reference and test channels.
        
        Args:
            ref_channel: Reference channel (ChannelInfo object)
            test_channel: Test channel (ChannelInfo object)
            signal_bus: Optional signal bus for logging
        """
        self.ref_channel = ref_channel
        self.test_channel = test_channel
        self.signal_bus = signal_bus
        
        # Cached results
        self.alignment_result = None
        self.comparison_stats = None
        
        # Configuration
        self.outlier_threshold = 2.0  # Standard deviations for outlier detection
        
    def log(self, event: str, level="info", **kwargs):
        """Log events if signal bus is available."""
        if self.signal_bus:
            from format_message import log_event
            log_event(self.signal_bus, event, level=level, **kwargs)
    
    # ==================== ALIGNMENT METHODS ====================
    
    def align_signals(self, 
                     method: AlignmentMethod = AlignmentMethod.INDEX,
                     start_idx: int = 0,
                     width: Optional[int] = None,
                     interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
                     **kwargs) -> AlignmentResult:
        """
        Align two signals using the specified method.
        
        Args:
            method: Alignment method to use
            start_idx: Starting index for alignment window
            width: Width of alignment window (None = use full signal)
            interpolation: Interpolation method for time-based alignment
            **kwargs: Additional parameters for specific alignment methods
            
        Returns:
            AlignmentResult object containing aligned signals and metadata
        """
        # Defensive checks to prevent memory corruption
        if self.ref_channel is None or self.test_channel is None:
            raise ValueError("Reference or test channel is None")
        
        if self.ref_channel.ydata is None or self.test_channel.ydata is None:
            raise ValueError("Reference or test channel has no ydata")
        
        if len(self.ref_channel.ydata) == 0 or len(self.test_channel.ydata) == 0:
            raise ValueError("Reference or test channel has empty ydata")
        
        # Additional checks for time-based alignment
        if method in [AlignmentMethod.TIME_INTERP_REF, AlignmentMethod.TIME_INTERP_TEST]:
            if self.ref_channel.xdata is None or self.test_channel.xdata is None:
                raise ValueError("Time-based alignment requires xdata for both channels")
            if len(self.ref_channel.xdata) == 0 or len(self.test_channel.xdata) == 0:
                raise ValueError("Time-based alignment requires non-empty xdata for both channels")
        
        self.log("alignment_start", method=method.value, start_idx=start_idx, width=width)
        
        try:
            if method == AlignmentMethod.INDEX:
                result = self._align_by_index(start_idx, width)
            elif method == AlignmentMethod.TIME_INTERP_REF:
                result = self._align_by_time_interpolation(start_idx, width, 
                                                         interpolation, use_ref_as_base=True)
            elif method == AlignmentMethod.TIME_INTERP_TEST:
                result = self._align_by_time_interpolation(start_idx, width, 
                                                         interpolation, use_ref_as_base=False)
            elif method == AlignmentMethod.CROSS_CORRELATION:
                result = self._align_by_cross_correlation(start_idx, width, **kwargs)
            elif method == AlignmentMethod.DTW:
                result = self._align_by_dtw(start_idx, width, **kwargs)
            else:
                raise ValueError(f"Unsupported alignment method: {method}")
            
            self.alignment_result = result
            self.log("alignment_success", method=method.value, 
                    n_samples=len(result.aligned_x), 
                    correlation=result.correlation_score)
            return result
            
        except Exception as e:
            self.log("alignment_failed", level="error", method=method.value, error=str(e))
            raise
    
    def _align_by_index(self, start_idx: int, width: Optional[int]) -> AlignmentResult:
        """Align signals by common index range."""
        ref_len = len(self.ref_channel.ydata)
        test_len = len(self.test_channel.ydata)
        
        if width is None:
            width = min(ref_len - start_idx, test_len - start_idx)
        
        end_idx = start_idx + width
        ref_end = min(end_idx, ref_len)
        test_end = min(end_idx, test_len)
        
        actual_width = min(ref_end - start_idx, test_end - start_idx)
        
        aligned_x = np.arange(actual_width)
        ref_y = self.ref_channel.ydata[start_idx:start_idx + actual_width]
        test_y = self.test_channel.ydata[start_idx:start_idx + actual_width]
        
        # Calculate correlation for quality metric
        correlation = np.corrcoef(ref_y, test_y)[0, 1] if len(ref_y) > 1 else 0.0
        
        return AlignmentResult(
            aligned_x=aligned_x,
            ref_y_aligned=ref_y,
            test_y_aligned=test_y,
            method="index",
            parameters={"start_idx": start_idx, "width": actual_width},
            correlation_score=correlation
        )
    
    def _align_by_time_interpolation(self, start_idx: int, width: Optional[int], 
                                   interpolation: InterpolationMethod, 
                                   use_ref_as_base: bool) -> AlignmentResult:
        """Align signals by time-based interpolation."""
        ref_x = self.ref_channel.xdata
        ref_y = self.ref_channel.ydata
        test_x = self.test_channel.xdata
        test_y = self.test_channel.ydata
        
        if ref_x is None or test_x is None:
            raise ValueError("Time-based alignment requires xdata (time) for both channels")
        
        if use_ref_as_base:
            # Use reference time base
            if width is None:
                base_x = ref_x[start_idx:]
                base_y = ref_y[start_idx:]
            else:
                end_idx = min(start_idx + width, len(ref_x))
                base_x = ref_x[start_idx:end_idx]
                base_y = ref_y[start_idx:end_idx]
            
            # Interpolate test signal to reference time base
            interp_func = interp1d(test_x, test_y, kind=interpolation.value, 
                                 bounds_error=False, fill_value=np.nan)
            aligned_test_y = interp_func(base_x)
            
            aligned_x = base_x
            aligned_ref_y = base_y
            
        else:
            # Use test time base
            if width is None:
                base_x = test_x[start_idx:]
                base_y = test_y[start_idx:]
            else:
                end_idx = min(start_idx + width, len(test_x))
                base_x = test_x[start_idx:end_idx]
                base_y = test_y[start_idx:end_idx]
            
            # Interpolate reference signal to test time base
            interp_func = interp1d(ref_x, ref_y, kind=interpolation.value, 
                                 bounds_error=False, fill_value=np.nan)
            aligned_ref_y = interp_func(base_x)
            aligned_test_y = base_y
            
            aligned_x = base_x
        
        # Remove NaN values
        valid_mask = ~(np.isnan(aligned_ref_y) | np.isnan(aligned_test_y))
        aligned_x = aligned_x[valid_mask]
        aligned_ref_y = aligned_ref_y[valid_mask]
        aligned_test_y = aligned_test_y[valid_mask]
        
        # Calculate correlation
        correlation = np.corrcoef(aligned_ref_y, aligned_test_y)[0, 1] if len(aligned_ref_y) > 1 else 0.0
        
        return AlignmentResult(
            aligned_x=aligned_x,
            ref_y_aligned=aligned_ref_y,
            test_y_aligned=aligned_test_y,
            method=f"time_interp_{'ref' if use_ref_as_base else 'test'}",
            parameters={"start_idx": start_idx, "width": width, 
                       "interpolation": interpolation.value, "use_ref_as_base": use_ref_as_base},
            correlation_score=correlation
        )
    
    def _align_by_cross_correlation(self, start_idx: int, width: Optional[int], 
                                  max_lag: int = 100) -> AlignmentResult:
        """Align signals using cross-correlation to find optimal offset."""
        ref_y = self.ref_channel.ydata
        test_y = self.test_channel.ydata
        
        if width is None:
            width = min(len(ref_y) - start_idx, len(test_y) - start_idx)
        
        # Extract segments for cross-correlation
        ref_segment = ref_y[start_idx:start_idx + width]
        test_segment = test_y[start_idx:start_idx + width]
        
        # Perform cross-correlation
        correlation = np.correlate(ref_segment, test_segment, mode='full')
        lags = np.arange(-len(test_segment) + 1, len(ref_segment))
        
        # Limit search to reasonable lag range
        center = len(correlation) // 2
        search_start = max(0, center - max_lag)
        search_end = min(len(correlation), center + max_lag + 1)
        
        search_corr = correlation[search_start:search_end]
        search_lags = lags[search_start:search_end]
        
        # Find optimal lag
        optimal_idx = np.argmax(search_corr)
        optimal_lag = search_lags[optimal_idx]
        max_correlation = search_corr[optimal_idx]
        
        # Apply optimal alignment
        if optimal_lag >= 0:
            # Test signal leads reference
            aligned_ref_y = ref_segment[optimal_lag:]
            aligned_test_y = test_segment[:len(aligned_ref_y)]
            aligned_x = np.arange(len(aligned_ref_y))
        else:
            # Reference signal leads test
            aligned_test_y = test_segment[-optimal_lag:]
            aligned_ref_y = ref_segment[:len(aligned_test_y)]
            aligned_x = np.arange(len(aligned_ref_y))
        
        return AlignmentResult(
            aligned_x=aligned_x,
            ref_y_aligned=aligned_ref_y,
            test_y_aligned=aligned_test_y,
            method="cross_correlation",
            parameters={"start_idx": start_idx, "width": width, 
                       "max_lag": max_lag, "optimal_lag": optimal_lag},
            alignment_offset=optimal_lag,
            correlation_score=max_correlation
        )
    
    def _align_by_dtw(self, start_idx: int, width: Optional[int], 
                     window_size: Optional[int] = None) -> AlignmentResult:
        """Align signals using Dynamic Time Warping (DTW)."""
        try:
            from dtaidistance import dtw
        except ImportError:
            raise ImportError("DTW alignment requires 'dtaidistance' package. Install with: pip install dtaidistance")
        
        ref_y = self.ref_channel.ydata
        test_y = self.test_channel.ydata
        
        if width is None:
            width = min(len(ref_y) - start_idx, len(test_y) - start_idx)
        
        # Extract segments
        ref_segment = ref_y[start_idx:start_idx + width]
        test_segment = test_y[start_idx:start_idx + width]
        
        # Perform DTW alignment
        if window_size:
            path = dtw.warping_path(ref_segment, test_segment, window=window_size)
        else:
            path = dtw.warping_path(ref_segment, test_segment)
        
        # Extract aligned sequences
        ref_indices, test_indices = zip(*path)
        aligned_ref_y = ref_segment[list(ref_indices)]
        aligned_test_y = test_segment[list(test_indices)]
        aligned_x = np.arange(len(aligned_ref_y))
        
        # Calculate correlation
        correlation = np.corrcoef(aligned_ref_y, aligned_test_y)[0, 1] if len(aligned_ref_y) > 1 else 0.0
        
        return AlignmentResult(
            aligned_x=aligned_x,
            ref_y_aligned=aligned_ref_y,
            test_y_aligned=aligned_test_y,
            method="dynamic_time_warping",
            parameters={"start_idx": start_idx, "width": width, "window_size": window_size},
            correlation_score=correlation
        )
    
    # ==================== STATISTICAL COMPARISON METHODS ====================
    
    def compute_comparison_stats(self, alignment_result: Optional[AlignmentResult] = None) -> ComparisonStats:
        """
        Compute comprehensive comparison statistics.
        
        Args:
            alignment_result: Optional alignment result. If None, uses last alignment.
            
        Returns:
            ComparisonStats object with all statistical measures
        """
        if alignment_result is None:
            alignment_result = self.alignment_result
        
        if alignment_result is None:
            raise ValueError("No alignment result available. Run align_signals() first.")
        
        self.log("stats_computation_start")
        
        ref_y = alignment_result.ref_y_aligned
        test_y = alignment_result.test_y_aligned
        
        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(ref_y) | np.isnan(test_y))
        ref_y = ref_y[valid_mask]
        test_y = test_y[valid_mask]
        
        if len(ref_y) < 2:
            raise ValueError("Insufficient valid data points for statistical analysis")
        
        # Basic statistics
        n_samples = len(ref_y)
        ref_mean = np.mean(ref_y)
        test_mean = np.mean(test_y)
        ref_std = np.std(ref_y, ddof=1)
        test_std = np.std(test_y, ddof=1)
        
        # Correlation statistics
        pearson_r, pearson_p = stats.pearsonr(ref_y, test_y)
        spearman_r, spearman_p = stats.spearmanr(ref_y, test_y)
        
        # Bland-Altman statistics
        mean_vals = (ref_y + test_y) / 2
        diff_vals = ref_y - test_y
        mean_bias = np.mean(diff_vals)
        bias_std = np.std(diff_vals, ddof=1)
        limits_of_agreement = (mean_bias - 1.96 * bias_std, mean_bias + 1.96 * bias_std)
        
        # Residual statistics
        residuals = test_y - ref_y
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        # MAPE (Mean Absolute Percentage Error) - handle division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape_vals = np.abs((ref_y - test_y) / ref_y) * 100
            mape = np.mean(mape_vals[np.isfinite(mape_vals)])
            if not np.isfinite(mape):
                mape = np.inf
        
        # Linear regression statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(ref_y, test_y)
        r_squared = r_value**2
        
        # Outlier detection (using standardized residuals)
        z_scores = np.abs(stats.zscore(residuals))
        outlier_indices = np.where(z_scores > self.outlier_threshold)[0].tolist()
        outlier_percentage = (len(outlier_indices) / n_samples) * 100
        
        stats_result = ComparisonStats(
            n_samples=n_samples,
            ref_mean=ref_mean,
            test_mean=test_mean,
            ref_std=ref_std,
            test_std=test_std,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            mean_bias=mean_bias,
            bias_std=bias_std,
            limits_of_agreement=limits_of_agreement,
            rmse=rmse,
            mae=mae,
            mape=mape,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            outlier_indices=outlier_indices,
            outlier_percentage=outlier_percentage
        )
        
        self.comparison_stats = stats_result
        self.log("stats_computation_success", n_samples=n_samples, 
                pearson_r=pearson_r, rmse=rmse, outlier_percentage=outlier_percentage)
        
        return stats_result
    
    # ==================== OVERLAY STATISTICS ====================
    
    def compute_overlay_stats(self, overlay_type: str = "all") -> Dict:
        """
        Compute statistics for overlay visualizations.
        
        Args:
            overlay_type: Type of overlay statistics to compute
                         ("identity", "bias", "regression", "outliers", "all")
        
        Returns:
            Dictionary containing overlay statistics
        """
        if self.comparison_stats is None:
            self.compute_comparison_stats()
        
        stats = self.comparison_stats
        overlay_stats = {}
        
        if overlay_type in ["identity", "all"]:
            # Identity line statistics
            ref_y = self.alignment_result.ref_y_aligned
            test_y = self.alignment_result.test_y_aligned
            
            # Distance from identity line
            identity_distances = np.abs(test_y - ref_y)
            overlay_stats["identity"] = {
                "mean_distance": np.mean(identity_distances),
                "max_distance": np.max(identity_distances),
                "distance_std": np.std(identity_distances),
                "line_params": {"slope": 1.0, "intercept": 0.0}
            }
        
        if overlay_type in ["bias", "all"]:
            # Bias line statistics
            overlay_stats["bias"] = {
                "mean_bias": stats.mean_bias,
                "bias_std": stats.bias_std,
                "upper_loa": stats.limits_of_agreement[1],
                "lower_loa": stats.limits_of_agreement[0],
                "line_params": {"y_value": stats.mean_bias}
            }
        
        if overlay_type in ["regression", "all"]:
            # Regression line statistics
            overlay_stats["regression"] = {
                "slope": stats.slope,
                "intercept": stats.intercept,
                "r_squared": stats.r_squared,
                "std_error": np.sqrt(stats.rmse),
                "line_params": {"slope": stats.slope, "intercept": stats.intercept}
            }
        
        if overlay_type in ["outliers", "all"]:
            # Outlier statistics
            overlay_stats["outliers"] = {
                "indices": stats.outlier_indices,
                "count": len(stats.outlier_indices),
                "percentage": stats.outlier_percentage,
                "threshold": self.outlier_threshold
            }
        
        return overlay_stats
    
    # ==================== BLAND-ALTMAN SPECIFIC METHODS ====================
    
    def compute_bland_altman_stats(self) -> Dict:
        """Compute detailed Bland-Altman statistics."""
        if self.alignment_result is None:
            raise ValueError("No alignment result available. Run align_signals() first.")
        
        ref_y = self.alignment_result.ref_y_aligned
        test_y = self.alignment_result.test_y_aligned
        
        # Remove NaN values
        valid_mask = ~(np.isnan(ref_y) | np.isnan(test_y))
        ref_y = ref_y[valid_mask]
        test_y = test_y[valid_mask]
        
        mean_vals = (ref_y + test_y) / 2
        diff_vals = ref_y - test_y
        
        mean_bias = np.mean(diff_vals)
        bias_std = np.std(diff_vals, ddof=1)
        
        # 95% limits of agreement
        upper_loa = mean_bias + 1.96 * bias_std
        lower_loa = mean_bias - 1.96 * bias_std
        
        # Proportional bias test (correlation between differences and means)
        # Handle constant input arrays gracefully
        try:
            if np.std(mean_vals) == 0 or np.std(diff_vals) == 0:
                # If either array is constant, correlation is undefined
                prop_bias_r, prop_bias_p = 0.0, 1.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=stats.ConstantInputWarning)
                    prop_bias_r, prop_bias_p = stats.pearsonr(mean_vals, diff_vals)
                    # Handle NaN results from constant arrays
                    if np.isnan(prop_bias_r) or np.isnan(prop_bias_p):
                        prop_bias_r, prop_bias_p = 0.0, 1.0
        except Exception:
            # Fallback for any correlation calculation issues
            prop_bias_r, prop_bias_p = 0.0, 1.0
        
        # Percentage within limits
        within_limits = np.sum((diff_vals >= lower_loa) & (diff_vals <= upper_loa))
        percentage_within = (within_limits / len(diff_vals)) * 100
        
        return {
            "mean_bias": mean_bias,
            "bias_std": bias_std,
            "upper_loa": upper_loa,
            "lower_loa": lower_loa,
            "proportional_bias_r": prop_bias_r,
            "proportional_bias_p": prop_bias_p,
            "percentage_within_limits": percentage_within,
            "mean_values": mean_vals,
            "differences": diff_vals
        }
    
    # ==================== UTILITY METHODS ====================
    
    def get_summary_report(self) -> str:
        """Generate a comprehensive text summary of the comparison analysis."""
        if self.comparison_stats is None:
            return "No comparison statistics available. Run compute_comparison_stats() first."
        
        stats = self.comparison_stats
        
        report = []
        report.append("=== SIGNAL COMPARISON SUMMARY ===")
        report.append(f"Reference: {self.ref_channel.legend_label or self.ref_channel.channel_id}")
        report.append(f"Test: {self.test_channel.legend_label or self.test_channel.channel_id}")
        report.append("")
        
        report.append("BASIC STATISTICS:")
        report.append(f"  Samples: {stats.n_samples}")
        report.append(f"  Reference Mean ± SD: {stats.ref_mean:.4f} ± {stats.ref_std:.4f}")
        report.append(f"  Test Mean ± SD: {stats.test_mean:.4f} ± {stats.test_std:.4f}")
        report.append("")
        
        report.append("CORRELATION ANALYSIS:")
        report.append(f"  Pearson r: {stats.pearson_r:.4f} (p = {stats.pearson_p:.4e})")
        report.append(f"  Spearman r: {stats.spearman_r:.4f} (p = {stats.spearman_p:.4e})")
        report.append("")
        
        report.append("AGREEMENT ANALYSIS:")
        report.append(f"  Mean Bias: {stats.mean_bias:.4f}")
        report.append(f"  95% Limits of Agreement: [{stats.limits_of_agreement[0]:.4f}, {stats.limits_of_agreement[1]:.4f}]")
        report.append("")
        
        report.append("ERROR METRICS:")
        report.append(f"  RMSE: {stats.rmse:.4f}")
        report.append(f"  MAE: {stats.mae:.4f}")
        report.append(f"  MAPE: {stats.mape:.2f}%")
        report.append("")
        
        report.append("REGRESSION ANALYSIS:")
        report.append(f"  Slope: {stats.slope:.4f}")
        report.append(f"  Intercept: {stats.intercept:.4f}")
        report.append(f"  R²: {stats.r_squared:.4f}")
        report.append("")
        
        report.append("OUTLIER ANALYSIS:")
        report.append(f"  Outliers: {len(stats.outlier_indices)} ({stats.outlier_percentage:.2f}%)")
        report.append(f"  Threshold: {self.outlier_threshold} standard deviations")
        
        return "\n".join(report)
    
    def export_results(self) -> Dict:
        """Export all results as a dictionary for external use."""
        results = {
            "alignment_result": self.alignment_result,
            "comparison_stats": self.comparison_stats,
            "overlay_stats": self.compute_overlay_stats() if self.comparison_stats else None,
            "bland_altman_stats": self.compute_bland_altman_stats() if self.alignment_result else None,
            "summary_report": self.get_summary_report()
        }
        return results 