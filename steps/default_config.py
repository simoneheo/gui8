import numpy as np

def get_intelligent_defaults(step_name, channel):
    """
    Calculate intelligent defaults for a step based on channel properties.
    
    Args:
        step_name: Name of the step (e.g., "count_samples", "area_envelope")
        channel: Channel object with properties like fs_median, xdata, ydata
        
    Returns:
        dict: Parameter overrides, or None if can't calculate
    """
    if not channel:
        return None
    
    try:
        # Get basic channel properties with extra validation
        fs = getattr(channel, 'fs_median', None)
        if not fs or not isinstance(fs, (int, float)) or fs <= 0 or not np.isfinite(fs):
            return None
        
        # Safely get total samples with comprehensive validation
        total_samples = 0
        if hasattr(channel, 'xdata') and channel.xdata is not None:
            try:
                # Try to convert to numpy array first to ensure it's valid
                x_data = np.array(channel.xdata)
                if not np.isfinite(x_data).any():
                    print(f"[DefaultConfig] Channel xdata contains no finite values")
                    return None
                
                total_samples = len(x_data)
                if not isinstance(total_samples, int) or total_samples <= 0:
                    print(f"[DefaultConfig] Invalid total_samples: {total_samples}")
                    return None
                    
                # Additional validation: check for reasonable data size
                if total_samples > 10000000:  # 10M samples seems excessive
                    print(f"[DefaultConfig] Data size too large: {total_samples} samples")
                    return None
                    
            except (TypeError, AttributeError, ValueError, MemoryError) as e:
                print(f"[DefaultConfig] Error accessing channel data: {e}")
                return None
        else:
            print(f"[DefaultConfig] No xdata available in channel")
            return None
        
        # Additional safety check: ensure channel has reasonable properties
        try:
            if hasattr(channel, 'ydata') and channel.ydata is not None:
                y_data = np.array(channel.ydata)
                if len(y_data) != total_samples:
                    print(f"[DefaultConfig] xdata and ydata length mismatch: {total_samples} vs {len(y_data)}")
                    return None
                if not np.isfinite(y_data).any():
                    print(f"[DefaultConfig] ydata contains no finite values")
                    # Don't return None here, some steps might not need ydata
        except Exception as y_e:
            print(f"[DefaultConfig] Error validating ydata: {y_e}")
            # Don't return None here, some steps might not need ydata
        
        # Step-specific intelligent defaults
        if step_name == "count_samples":
            return _get_count_samples_defaults(fs, total_samples)
        elif step_name == "area_envelope":
            return _get_area_envelope_defaults(fs, total_samples)
        elif step_name == "moving_average":
            return _get_moving_average_defaults(fs, total_samples)
        elif step_name == "moving_mean":
            return _get_moving_mean_defaults(fs, total_samples)
        elif step_name == "gaussian_smooth":
            return _get_gaussian_smooth_defaults(fs, total_samples)
        elif step_name == "median_smooth":
            return _get_median_smooth_defaults(fs, total_samples)
        elif step_name == "resample":
            return _get_resample_defaults(fs, total_samples)
        elif step_name == "bandpass_butter":
            return _get_bandpass_butter_defaults(fs, total_samples)
        elif step_name == "lowpass_butter":
            return _get_lowpass_butter_defaults(fs, total_samples)
        elif step_name == "highpass_butter":
            return _get_highpass_butter_defaults(fs, total_samples)
        elif step_name == "detect_peaks":
            return _get_detect_peaks_defaults(fs, total_samples, channel)
        elif step_name == "detect_valleys":
            return _get_detect_valleys_defaults(fs, total_samples, channel)
        elif step_name == "envelope_peaks":
            return _get_envelope_peaks_defaults(fs, total_samples)
        elif step_name == "hilbert_envelope":
            return _get_hilbert_envelope_defaults(fs, total_samples)
        elif step_name == "welch_spectrogram":
            return _get_welch_spectrogram_defaults(fs, total_samples)
        elif step_name == "stft_spectrogram":
            return _get_stft_spectrogram_defaults(fs, total_samples)
        elif step_name == "cwt_spectrogram":
            return _get_cwt_spectrogram_defaults(fs, total_samples)
        elif step_name == "windowed_energy":
            return _get_windowed_energy_defaults(fs, total_samples)
        elif step_name == "moving_rms":
            return _get_moving_rms_defaults(fs, total_samples)
        elif step_name == "cumulative_max":
            return _get_cumulative_max_defaults(fs, total_samples)
        elif step_name == "cumulative_sum":
            return _get_cumulative_sum_defaults(fs, total_samples)
        elif step_name == "derivative":
            return _get_derivative_defaults(fs, total_samples)
        elif step_name == "detect_extrema":
            return _get_detect_extrema_defaults(fs, total_samples, channel)
        elif step_name == "energy_sliding":
            return _get_energy_sliding_defaults(fs, total_samples)
        elif step_name == "zscore_sliding":
            return _get_zscore_sliding_defaults(fs, total_samples)
        elif step_name == "percentile_clip_sliding":
            return _get_percentile_clip_sliding_defaults(fs, total_samples)
        elif step_name == "top_percentile_sliding":
            return _get_top_percentile_sliding_defaults(fs, total_samples)
        elif step_name == "median_sliding":
            return _get_median_sliding_defaults(fs, total_samples)
        elif step_name == "moving_absmax":
            return _get_moving_absmax_defaults(fs, total_samples)
        elif step_name == "savitzky_golay":
            return _get_savitzky_golay_defaults(fs, total_samples)
        elif step_name == "exp_smooth":
            return _get_exp_smooth_defaults(fs, total_samples)
        elif step_name == "loess_smooth":
            return _get_loess_smooth_defaults(fs, total_samples)
        elif step_name == "moving_skewness":
            return _get_moving_skewness_defaults(fs, total_samples)
        elif step_name == "moving_kurtosis":
            return _get_moving_kurtosis_defaults(fs, total_samples)
        # Add more steps as needed
        
        # Add comprehensive intelligent defaults for all missing steps
        elif step_name == "abs_transform":
            return {}  # No parameters needed
        elif step_name == "add_constant":
            return _get_add_constant_defaults(fs, total_samples, channel)
        elif step_name == "bandpass_bessel":
            return _get_bandpass_bessel_defaults(fs, total_samples, channel)
        elif step_name == "bandpass_fir":
            return _get_bandpass_fir_defaults(fs, total_samples, channel)
        elif step_name == "boxcox_transform":
            return _get_boxcox_transform_defaults(fs, total_samples, channel)
        elif step_name == "clip_values":
            return _get_clip_values_defaults(fs, total_samples, channel)
        elif step_name == "count_samples":
            return _get_count_samples_defaults(fs, total_samples, channel)
        elif step_name == "detrend_polynomial":
            return _get_detrend_polynomial_defaults(fs, total_samples, channel)
        elif step_name == "envelope_peaks":
            return _get_envelope_peaks_defaults(fs, total_samples, channel)
        elif step_name == "exp_transform":
            return {}  # No parameters needed
        elif step_name == "highpass_bessel":
            return _get_highpass_bessel_defaults(fs, total_samples, channel)
        elif step_name == "highpass_fir":
            return _get_highpass_fir_defaults(fs, total_samples, channel)
        elif step_name == "impute_missing":
            return _get_impute_missing_defaults(fs, total_samples, channel)
        elif step_name == "linear_detrend":
            return {}  # No parameters needed
        elif step_name == "lowpass_bessel":
            return _get_lowpass_bessel_defaults(fs, total_samples, channel)
        elif step_name == "lowpass_fir":
            return _get_lowpass_fir_defaults(fs, total_samples, channel)
        elif step_name == "median_subtract":
            return {}  # No parameters needed
        elif step_name == "modulo":
            return _get_modulo_defaults(fs, total_samples, channel)
        elif step_name == "multiply_constant":
            return _get_multiply_constant_defaults(fs, total_samples, channel)
        elif step_name == "normalize":
            return _get_normalize_defaults(fs, total_samples, channel)
        elif step_name == "percentile_clip":
            return _get_percentile_clip_defaults(fs, total_samples, channel)
        elif step_name == "power":
            return _get_power_defaults(fs, total_samples, channel)
        elif step_name == "quantize":
            return _get_quantize_defaults(fs, total_samples, channel)
        elif step_name == "rank_transform":
            return {}  # No parameters needed
        elif step_name == "reciprocal":
            return _get_reciprocal_defaults(fs, total_samples, channel)
        elif step_name == "rolling_mean_subtract":
            return _get_rolling_mean_subtract_defaults(fs, total_samples, channel)
        elif step_name == "sign_only":
            return {}  # No parameters needed
        elif step_name == "standardize":
            return _get_standardize_defaults(fs, total_samples, channel)
        elif step_name == "threshold_binary":
            return _get_threshold_binary_defaults(fs, total_samples, channel)
        elif step_name == "threshold_clip":
            return _get_threshold_clip_defaults(fs, total_samples, channel)
        elif step_name == "detect_zero_crossings":
            return {}  # No parameters needed
        elif step_name == "wavelet_decompose":
            return _get_wavelet_decompose_defaults(fs, total_samples, channel)
        elif step_name == "wavelet_denoise":
            return _get_wavelet_denoise_defaults(fs, total_samples, channel)
        elif step_name == "wavelet_filter_band":
            return _get_wavelet_filter_band_defaults(fs, total_samples, channel)
        elif step_name == "wavelet_reconstruct":
            return _get_wavelet_reconstruct_defaults(fs, total_samples, channel)
        elif step_name == "minmax_normalize":
            return _get_minmax_normalize_defaults(fs, total_samples, channel)
        elif step_name == "zscore_global":
            return {}  # No parameters needed
        elif step_name == "custom":
            return _get_custom_defaults(fs, total_samples, channel)
        # Add more steps as needed
        
    except Exception as e:
        print(f"[DefaultConfig] Failed to calculate defaults for {step_name}: {e}")
        return None
    
    return None

def _get_detect_extrema_defaults(fs, total_samples, channel):
    """Generate intelligent defaults for DetectExtremaStep (sample-based window)"""
    defaults = {}

    # Window: 2 seconds worth of samples or 1/10 of signal length, whichever is smaller
    window = min(int(2 * fs), total_samples // 10)
    window = max(window, 10)  # Ensure minimum reasonable window

    # Overlap: 50% of window
    overlap = window // 2

    # Height threshold as fraction of signal range
    if hasattr(channel, 'ydata') and channel.ydata is not None:
        y = channel.ydata
        if len(y) > 1:
            signal_range = np.nanmax(y) - np.nanmin(y)
            if signal_range > 0:
                min_height = 0.1  # Fraction (as specified in the UI)
            else:
                min_height = 0.0
        else:
            min_height = 0.0
    else:
        min_height = 0.1

    defaults["window"] = str(window)
    defaults["overlap"] = str(overlap)
    defaults["min_height"] = str(round(min_height, 3))

    return defaults


def _get_derivative_defaults(fs, total_samples):
    """Defaults for derivative: full signal range, no windowing needed"""
    return {}

def _get_cumulative_max_defaults(fs, total_samples):
    """Defaults for cumulative_max: full signal range, no windowing needed"""
    return {}  # No params needed for pure cumulative max

def _get_cumulative_sum_defaults(fs, total_samples):
    """Defaults for cumulative_sum: full signal range, no windowing needed"""
    return {}


def _get_count_samples_defaults(fs, total_samples):
    """Calculate defaults for count_samples step"""
    # Window: 2 seconds worth of samples, but not more than 1/10 of signal
    window = min(int(2 * fs), total_samples // 10)
    window = max(window, 100)  # Minimum window size
    
    # Overlap: 50% of window
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap)
    }

def _get_area_envelope_defaults(fs, total_samples):
    """Calculate defaults for area_envelope step"""
    # Window: 3 seconds worth of samples, but not more than 1/5 of signal
    window = min(int(3 * fs), total_samples // 5)
    window = max(window, 50)  # Minimum window size
    
    # Overlap: window/50 (2% overlap)
    overlap = max(window // 50, 1)
    
    return {
        "window": str(window),
        "overlap": str(overlap)
    }

def _get_moving_average_defaults(fs, total_samples):
    """Calculate defaults for moving_average step"""
    # Window: 0.1 seconds worth of samples for smoothing
    window = max(int(0.1 * fs), 5)
    window = min(window, total_samples // 20)  # Not more than 5% of signal
    
    return {
        "window": str(window)
    }

def _get_moving_mean_defaults(fs, total_samples):
    """Calculate defaults for moving_mean step"""
    # Window: 0.1 seconds worth of samples for smoothing
    window = max(int(0.1 * fs), 5)
    window = min(window, total_samples // 20)  # Not more than 5% of signal
    
    return {
        "window": str(window)
    }

def _get_gaussian_smooth_defaults(fs, total_samples):
    """Calculate defaults for gaussian_smooth step"""
    # Sigma: 0.05 seconds worth of samples
    sigma = max(int(0.05 * fs), 2)
    sigma = min(sigma, total_samples // 50)  # Not more than 2% of signal
    
    return {
        "sigma": str(sigma)
    }

def _get_median_smooth_defaults(fs, total_samples):
    """Calculate defaults for median_smooth step"""
    # Window: 0.05 seconds worth of samples
    window = max(int(0.05 * fs), 3)
    window = min(window, total_samples // 50)  # Not more than 2% of signal
    # Ensure odd number for median filter
    if window % 2 == 0:
        window += 1
    
    return {
        "window": str(window)
    }

def _get_resample_defaults(fs, total_samples):
    """Calculate defaults for resample step"""
    # Target fs: Smart downsampling based on original fs
    if fs > 1000:
        # High frequency: downsample by 10-20x
        target_fs = fs / 15
    elif fs > 100:
        # Medium frequency: downsample by 4-10x
        target_fs = fs / 6
    else:
        # Low frequency: downsample by 2-4x
        target_fs = fs / 2.5
    
    # Round to convenient values
    target_fs = round(target_fs, 1)
    
    return {
        "target_fs": str(target_fs)
    }

def _get_bandpass_butter_defaults(fs, total_samples):
    """Calculate defaults for bandpass_butter step"""
    nyquist = fs / 2
    
    # Low cutoff: 10% of Nyquist
    low_cutoff = nyquist * 0.1
    # High cutoff: 80% of Nyquist
    high_cutoff = nyquist * 0.8
    
    return {
        "low_cutoff": str(round(low_cutoff, 2)),
        "high_cutoff": str(round(high_cutoff, 2)),
        "fs": str(round(fs, 3))
    }

def _get_lowpass_butter_defaults(fs, total_samples):
    """Calculate defaults for lowpass_butter step"""
    nyquist = fs / 2
    
    # Cutoff: 40% of Nyquist
    cutoff = nyquist * 0.4
    
    return {
        "cutoff": str(round(cutoff, 2)),
        "fs": str(round(fs, 3))
    }

def _get_highpass_butter_defaults(fs, total_samples):
    """Calculate defaults for highpass_butter step"""
    nyquist = fs / 2
    
    # Cutoff: 5% of Nyquist (remove very low frequencies)
    cutoff = nyquist * 0.05
    
    return {
        "cutoff": str(round(cutoff, 2)),
        "fs": str(round(fs, 3))
    }

def _get_detect_peaks_defaults(fs, total_samples, channel):
    """Calculate defaults for detect_peaks step"""
    defaults = {}
    
    # Try to estimate height threshold from signal statistics
    try:
        if (hasattr(channel, 'ydata') and channel.ydata is not None and 
            len(channel.ydata) > 0 and np.isfinite(channel.ydata).any()):
            y_data = np.array(channel.ydata)
            # Filter out non-finite values
            y_data = y_data[np.isfinite(y_data)]
            if len(y_data) > 0:
                # Height: median + 2 * std (captures prominent peaks)
                height = np.median(y_data) + 2 * np.std(y_data)
                if np.isfinite(height):
                    defaults["height"] = str(round(height, 6))
    except Exception as e:
        print(f"[DefaultConfig] Error calculating peak height: {e}")
    
    # Distance: 0.1 seconds worth of samples (prevent duplicate peaks)
    distance = max(int(0.1 * fs), 1)
    defaults["distance"] = str(distance)
    
    return defaults

def _get_detect_valleys_defaults(fs, total_samples, channel):
    """Calculate defaults for detect_valleys step"""
    defaults = {}
    
    # Try to estimate height threshold from signal statistics
    try:
        if (hasattr(channel, 'ydata') and channel.ydata is not None and 
            len(channel.ydata) > 0 and np.isfinite(channel.ydata).any()):
            y_data = np.array(channel.ydata)
            # Filter out non-finite values
            y_data = y_data[np.isfinite(y_data)]
            if len(y_data) > 0:
                # Height: median - 2 * std (captures prominent valleys)
                height = np.median(y_data) - 2 * np.std(y_data)
                if np.isfinite(height):
                    defaults["height"] = str(round(height, 6))
    except Exception as e:
        print(f"[DefaultConfig] Error calculating valley height: {e}")
    
    # Distance: 0.1 seconds worth of samples (prevent duplicate valleys)
    distance = max(int(0.1 * fs), 1)
    defaults["distance"] = str(distance)
    
    return defaults

def _get_envelope_peaks_defaults(fs, total_samples):
    """Calculate defaults for envelope_peaks step"""
    # Window: 0.5 seconds worth of samples
    window = max(int(0.5 * fs), 10)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    return {
        "window": str(window)
    }

def _get_hilbert_envelope_defaults(fs, total_samples):
    """Calculate defaults for hilbert_envelope step"""
    # Usually no parameters needed for basic Hilbert transform
    # But if there are smoothing parameters, we can set them
    return {}

def _get_welch_spectrogram_defaults(fs, total_samples):
    """Calculate defaults for welch_spectrogram step"""
    # Window: 2 seconds worth of samples, good for frequency resolution
    window = min(int(2 * fs), total_samples // 10)
    window = max(window, 256)  # Minimum for reasonable frequency resolution
    
    # Overlap: 50% is standard for spectrograms
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap),
        "fs": str(round(fs, 3))
    }

def _get_stft_spectrogram_defaults(fs, total_samples):
    """Calculate defaults for stft_spectrogram step"""
    # Window: 1 second worth of samples, good balance of time/freq resolution
    window = min(int(1 * fs), total_samples // 20)
    window = max(window, 256)  # Minimum for reasonable frequency resolution
    
    # Overlap: 75% is common for STFT
    overlap = int(window * 0.75)
    
    return {
        "window": str(window),
        "overlap": str(overlap),
        "fs": str(round(fs, 3))
    }

def _get_cwt_spectrogram_defaults(fs, total_samples):
    """Calculate defaults for cwt_spectrogram step"""
    # Window: 1 second worth of samples, good balance of time/freq resolution
    window = min(int(1 * fs), total_samples // 20)
    window = max(window, 256)  # Minimum for reasonable frequency resolution
    
    # Overlap: 75% is common for STFT
    overlap = int(window * 0.75)
    
    return {
        "window": str(window),
        "overlap": str(overlap),
        "fs": str(round(fs, 3))
    }

def _get_windowed_energy_defaults(fs, total_samples):
    """Calculate defaults for windowed_energy step"""
    # Window: 0.5 seconds worth of samples
    window = max(int(0.5 * fs), 10)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    # Overlap: 50% of window
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap)
    }

def _get_moving_rms_defaults(fs, total_samples):
    """Calculate defaults for moving_rms step"""
    # Window: 0.2 seconds worth of samples
    window = max(int(0.2 * fs), 10)
    window = min(window, total_samples // 20)  # Not more than 5% of signal
    
    return {
        "window": str(window)
    }

def _get_energy_sliding_defaults(fs, total_samples):
    """Calculate defaults for energy_sliding step"""
    # Window: 0.5 seconds worth of samples
    window = max(int(0.5 * fs), 10)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    # Overlap: 50% of window
    overlap = 0.5
    
    return {
        "window": str(window),
        "overlap": str(overlap)
    }

def _get_zscore_sliding_defaults(fs, total_samples):
    """Calculate defaults for zscore_sliding step"""
    # Window: 2 seconds worth of samples for local normalization
    window = max(int(2 * fs), 100)
    window = min(window, total_samples // 5)  # Not more than 20% of signal
    
    return {
        "window": str(window)
    }

def _get_percentile_clip_sliding_defaults(fs, total_samples):
    """Calculate defaults for percentile_clip_sliding step"""
    # Window: 3 seconds worth of samples for robust clipping
    window = max(int(3 * fs), 50)
    window = min(window, total_samples // 5)  # Not more than 20% of signal
    
    return {
        "window": str(window),
        "lower": "5.0",  # 5th percentile
        "upper": "95.0"  # 95th percentile
    }

def _get_top_percentile_sliding_defaults(fs, total_samples):
    """Calculate defaults for top_percentile_sliding step"""
    # Window: 1 second worth of samples for envelope tracking
    window = max(int(1 * fs), 25)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    return {
        "window": str(window),
        "percentile": "95.0"  # 95th percentile
    }

def _get_median_sliding_defaults(fs, total_samples):
    """Calculate defaults for median_sliding step"""
    # Window: 0.5 seconds worth of samples
    window = max(int(0.5 * fs), 10)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    return {
        "window": str(window)
    }

def _get_moving_absmax_defaults(fs, total_samples):
    """Calculate defaults for moving_absmax step"""
    # Window: 0.5 seconds worth of samples for envelope tracking
    window = max(int(0.5 * fs), 10)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    return {
        "window": str(window)
    }

def _get_savitzky_golay_defaults(fs, total_samples):
    """Calculate defaults for savitzky_golay step"""
    # Window: 0.1 seconds worth of samples for smoothing
    window = max(int(0.1 * fs), 5)
    window = min(window, total_samples // 20)  # Not more than 5% of signal
    # Ensure odd number for Savitzky-Golay
    if window % 2 == 0:
        window += 1
    
    return {
        "window": str(window),
        "order": "3"  # Cubic polynomial is common
    }

def _get_exp_smooth_defaults(fs, total_samples):
    """Calculate defaults for exp_smooth step"""
    # Alpha based on 0.1 second time constant
    time_constant_seconds = 0.1
    alpha = 1 - np.exp(-1 / (time_constant_seconds * fs))
    alpha = max(0.01, min(0.3, alpha))  # Clamp to reasonable range
    
    return {
        "alpha": str(round(alpha, 4))
    }

def _get_loess_smooth_defaults(fs, total_samples):
    """Calculate defaults for loess_smooth step"""
    # Span: 10% of the data for local regression
    span = 0.1
    
    return {
        "span": str(span)
    }

def _get_moving_skewness_defaults(fs, total_samples):
    """Calculate defaults for moving_skewness step"""
    # Window: 2 seconds worth of samples for statistical calculations
    window = max(int(2 * fs), 50)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    return {
        "window": str(window)
    }

def _get_moving_kurtosis_defaults(fs, total_samples):
    """Calculate defaults for moving_kurtosis step"""
    # Window: 2 seconds worth of samples for statistical calculations
    window = max(int(2 * fs), 50)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    return {
        "window": str(window)
    }

def format_intelligent_default_info(step_name, channel, defaults):
    """
    Format information about intelligent defaults for user display.
    
    Args:
        step_name: Name of the step
        channel: Channel object used for calculation
        defaults: Dictionary of calculated defaults
        
    Returns:
        str: Formatted information string
    """
    if not defaults or not channel:
        return ""
    
    try:
        fs = getattr(channel, 'fs_median', None)
        if not fs:
            return ""
        
        info_parts = []
        info_parts.append(f"Intelligent defaults calculated for {step_name}:")
        info_parts.append(f"  • Sampling rate: {fs:.1f} Hz")
        
        for param_name, param_value in defaults.items():
            if param_name == "window" and param_value.isdigit():
                samples = int(param_value)
                duration = samples / fs
                info_parts.append(f"  • {param_name}: {param_value} samples ({duration:.3f}s)")
            elif param_name == "overlap" and param_value.isdigit():
                samples = int(param_value)
                duration = samples / fs
                info_parts.append(f"  • {param_name}: {param_value} samples ({duration:.3f}s)")
            elif param_name in ["cutoff", "low_cutoff", "high_cutoff"]:
                freq = float(param_value)
                nyquist = fs / 2
                percent = (freq / nyquist) * 100
                info_parts.append(f"  • {param_name}: {param_value} Hz ({percent:.1f}% of Nyquist)")
            else:
                info_parts.append(f"  • {param_name}: {param_value}")
        
        return "\n".join(info_parts)
        
    except Exception as e:
        return f"Error formatting intelligent defaults info: {e}"

# Add all missing helper functions for intelligent defaults

def _get_add_constant_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for add_constant step."""
    try:
        # Default to adding 0 (no change)
        return {"constant": "0.0"}
    except Exception as e:
        return {"constant": "0.0"}

def _get_bandpass_bessel_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for bandpass_bessel step."""
    try:
        # Use same logic as bandpass_butter
        return _get_bandpass_butter_defaults(fs, total_samples, channel)
    except Exception as e:
        return {"low_cutoff": "0.5", "high_cutoff": "4.0", "order": "2"}

def _get_bandpass_fir_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for bandpass_fir step."""
    try:
        # Use same logic as bandpass_butter but with different default order
        defaults = _get_bandpass_butter_defaults(fs, total_samples, channel)
        # FIR filters typically need higher order
        defaults["order"] = "21"
        return defaults
    except Exception as e:
        return {"low_cutoff": "0.5", "high_cutoff": "4.0", "order": "21"}

def _get_boxcox_transform_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for boxcox_transform step."""
    try:
        # Default lambda for Box-Cox transform
        return {"lmbda": "0.5"}
    except Exception as e:
        return {"lmbda": "0.5"}

def _get_clip_values_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for clip_values step."""
    try:
        if channel and hasattr(channel, 'ydata') and channel.ydata is not None:
            data = channel.ydata
            if len(data) > 0 and np.any(np.isfinite(data)):
                finite_data = data[np.isfinite(data)]
                if len(finite_data) > 0:
                    p5, p95 = np.percentile(finite_data, [5, 95])
                    return {
                        "min_val": str(p5),
                        "max_val": str(p95)
                    }
        return {"min_val": "-1.0", "max_val": "1.0"}
    except Exception as e:
        return {"min_val": "-1.0", "max_val": "1.0"}

def _get_count_samples_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for count_samples step."""
    try:
        # Calculate reasonable window size (1 second of data)
        window_samples = min(int(fs), total_samples // 4) if fs > 0 else min(1000, total_samples // 4)
        window_samples = max(window_samples, 100)  # Minimum 100 samples
        
        # Default overlap is 50% of window
        overlap_samples = window_samples // 2
        
        return {
            "window": str(window_samples),
            "overlap": str(overlap_samples),
            "unit": "count/window"
        }
    except Exception as e:
        return {"window": "1000", "overlap": "500", "unit": "count/window"}

def _get_detrend_polynomial_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for detrend_polynomial step."""
    try:
        # Default polynomial order based on signal length
        if total_samples < 1000:
            order = 1
        elif total_samples < 10000:
            order = 2
        else:
            order = 3
        return {"order": str(order)}
    except Exception as e:
        return {"order": "2"}

def _get_envelope_peaks_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for envelope_peaks step."""
    try:
        # Default interpolation method
        return {"method": "linear"}
    except Exception as e:
        return {"method": "linear"}

def _get_highpass_bessel_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for highpass_bessel step."""
    try:
        # Conservative highpass cutoff - remove DC and very low frequencies
        cutoff = min(0.1, fs / 100) if fs > 0 else 0.1
        return {"cutoff": str(cutoff), "order": "2"}
    except Exception as e:
        return {"cutoff": "0.1", "order": "2"}

def _get_highpass_fir_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for highpass_fir step."""
    try:
        # Same cutoff as Bessel but higher order for FIR
        cutoff = min(0.1, fs / 100) if fs > 0 else 0.1
        return {"cutoff": str(cutoff), "order": "21"}
    except Exception as e:
        return {"cutoff": "0.1", "order": "21"}

def _get_impute_missing_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for impute_missing step."""
    try:
        # Default to zero fill for simplicity
        return {"method": "zero"}
    except Exception as e:
        return {"method": "zero"}

def _get_lowpass_bessel_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for lowpass_bessel step."""
    try:
        # Conservative lowpass cutoff - remove high frequency noise
        cutoff = min(fs / 4, 10.0) if fs > 0 else 10.0
        return {"cutoff": str(cutoff), "order": "2"}
    except Exception as e:
        return {"cutoff": "10.0", "order": "2"}

def _get_lowpass_fir_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for lowpass_fir step."""
    try:
        # Same cutoff as Bessel but higher order for FIR
        cutoff = min(fs / 4, 10.0) if fs > 0 else 10.0
        return {"cutoff": str(cutoff), "order": "21"}
    except Exception as e:
        return {"cutoff": "10.0", "order": "21"}

def _get_modulo_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for modulo step."""
    try:
        # Default modulo value
        return {"divisor": "2.0"}
    except Exception as e:
        return {"divisor": "2.0"}

def _get_multiply_constant_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for multiply_constant step."""
    try:
        # Default to no change
        return {"constant": "1.0"}
    except Exception as e:
        return {"constant": "1.0"}

def _get_normalize_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for normalize step."""
    try:
        # Default normalization method
        return {"method": "minmax"}
    except Exception as e:
        return {"method": "minmax"}

def _get_percentile_clip_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for percentile_clip step."""
    try:
        # Default percentile clipping - remove outliers
        return {"lower_percentile": "5.0", "upper_percentile": "95.0"}
    except Exception as e:
        return {"lower_percentile": "5.0", "upper_percentile": "95.0"}

def _get_power_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for power step."""
    try:
        # Default power (square)
        return {"exponent": "2.0"}
    except Exception as e:
        return {"exponent": "2.0"}

def _get_quantize_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for quantize step."""
    try:
        # Default number of levels
        return {"levels": "16"}
    except Exception as e:
        return {"levels": "16"}

def _get_reciprocal_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for reciprocal step."""
    try:
        # Default epsilon to avoid division by zero
        return {"epsilon": "1e-10"}
    except Exception as e:
        return {"epsilon": "1e-10"}

def _get_rolling_mean_subtract_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for rolling_mean_subtract step."""
    try:
        # Default window size (0.1 seconds or 100 samples)
        window_samples = min(int(fs * 0.1), 100) if fs > 0 else 100
        window_samples = max(window_samples, 3)  # Minimum 3 samples
        return {"window": str(window_samples)}
    except Exception as e:
        return {"window": "100"}

def _get_standardize_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for standardize step."""
    try:
        # Default standardization method
        return {"method": "zscore"}
    except Exception as e:
        return {"method": "zscore"}

def _get_threshold_binary_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for threshold_binary step."""
    try:
        threshold = 0.0
        if channel and hasattr(channel, 'ydata') and channel.ydata is not None:
            data = channel.ydata
            if len(data) > 0 and np.any(np.isfinite(data)):
                finite_data = data[np.isfinite(data)]
                if len(finite_data) > 0:
                    # Use median as threshold
                    threshold = np.median(finite_data)
        
        return {"threshold": str(threshold), "operator": ">"}
    except Exception as e:
        return {"threshold": "0.0", "operator": ">"}

def _get_threshold_clip_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for threshold_clip step."""
    try:
        threshold = 0.0
        if channel and hasattr(channel, 'ydata') and channel.ydata is not None:
            data = channel.ydata
            if len(data) > 0 and np.any(np.isfinite(data)):
                finite_data = data[np.isfinite(data)]
                if len(finite_data) > 0:
                    # Use 95th percentile as threshold
                    threshold = np.percentile(finite_data, 95)
        
        return {"threshold": str(threshold), "clip_value": "0.0"}
    except Exception as e:
        return {"threshold": "0.0", "clip_value": "0.0"}

def _get_wavelet_decompose_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for wavelet_decompose step."""
    try:
        # Default wavelet and number of levels
        levels = min(6, int(np.log2(total_samples)) - 2) if total_samples > 0 else 4
        return {"wavelet": "db4", "levels": str(levels)}
    except Exception as e:
        return {"wavelet": "db4", "levels": "4"}

def _get_wavelet_denoise_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for wavelet_denoise step."""
    try:
        # Default wavelet denoising parameters
        return {"wavelet": "db4", "threshold": "0.1", "mode": "soft"}
    except Exception as e:
        return {"wavelet": "db4", "threshold": "0.1", "mode": "soft"}

def _get_wavelet_filter_band_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for wavelet_filter_band step."""
    try:
        # Default wavelet band filtering parameters
        return {"wavelet": "db4", "level": "3"}
    except Exception as e:
        return {"wavelet": "db4", "level": "3"}

def _get_wavelet_reconstruct_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for wavelet_reconstruct step."""
    try:
        # Default wavelet reconstruction parameters
        return {"wavelet": "db4"}
    except Exception as e:
        return {"wavelet": "db4"}

def _get_minmax_normalize_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for minmax_normalize step."""
    try:
        # Default range for min-max normalization
        return {"feature_range_min": "0.0", "feature_range_max": "1.0"}
    except Exception as e:
        return {"feature_range_min": "0.0", "feature_range_max": "1.0"}

def _get_custom_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for custom step."""
    try:
        # Default custom function - simple example
        return {"function": "y_new = y * 2  # Double the signal"}
    except Exception as e:
        return {"function": "y_new = y * 2  # Double the signal"} 