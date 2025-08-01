import numpy as np
from steps.base_step import BaseStep

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
        # Use BaseStep validation methods for consistency
        BaseStep.validate_channel_input(channel)
        
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
                BaseStep.validate_signal_data(x_data, channel.ydata)
                
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
        
        # Use the new comprehensive step defaults function
        return get_step_defaults(step_name, fs, total_samples, channel)
        
    except Exception as e:
        print(f"[DefaultConfig] Failed to calculate defaults for {step_name}: {e}")
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


def _get_count_samples_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for count_samples step."""
    try:
        # Calculate reasonable window size (1 second of data)
        window_samples = min(int(fs)*3, total_samples // 4) if fs > 0 else min(1000, total_samples // 4)
        window_samples = max(window_samples, 100)  # Minimum 100 samples
        
        # Ensure overlap is less than window size
        # Use 50% overlap but ensure it's valid
        overlap_samples = window_samples // 2
        overlap_samples = min(overlap_samples, window_samples - 1)  # Must be less than window
        overlap_samples = max(overlap_samples, 1)  # At least 1 sample overlap
        
        return {
            "window": str(window_samples),
            "overlap": str(overlap_samples),
            "unit": "count/window"
        }
    except Exception as e:
        return {"window": "1000", "overlap": "500", "unit": "count/window"}

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
    """Calculate improved defaults for moving_average step"""
    try:
        # Calculate window size based on sampling rate and signal length
        # Rule: 0.1 seconds worth of samples for good smoothing
        
        # Base window size on sampling rate
        if fs > 0:
            base_window = max(int(0.1 * fs), 3)  # 0.1 seconds or minimum 3 samples
        else:
            base_window = 10  # Safe fallback
        
        # Ensure window doesn't exceed reasonable bounds
        # Maximum: 5% of signal length to avoid over-smoothing
        max_window = max(total_samples // 20, 3)  # At least 3 samples
        
        # Conservative minimum for effective smoothing
        min_window = 3
        
        # Final window size calculation
        window = max(min_window, min(base_window, max_window))
        
        # Additional safety check for very short signals
        if window >= total_samples:
            window = max(total_samples // 3, 1)  # Use 1/3 of signal length
        
        return {
            "window": str(window)
        }
    except Exception as e:
        # Safe fallback
        return {
            "window": "5"
        }

def _get_moving_mean_defaults(fs, total_samples):
    """Calculate improved defaults for moving_mean step"""
    try:
        # Calculate window size based on sampling rate and signal length
        # Rule: 0.1 seconds worth of samples for good smoothing
        
        # Base window size on sampling rate
        if fs > 0:
            base_window = max(int(0.1 * fs), 3)  # 0.1 seconds or minimum 3 samples
        else:
            base_window = 10  # Safe fallback
        
        # Ensure window doesn't exceed reasonable bounds
        # Maximum: 5% of signal length to avoid over-smoothing
        max_window = max(total_samples // 20, 3)  # At least 3 samples
        
        # Conservative minimum for effective smoothing
        min_window = 3
        
        # Final window size calculation
        window = max(min_window, min(base_window, max_window))
        
        # Additional safety check for very short signals
        if window >= total_samples:
            window = max(total_samples // 3, 1)  # Use 1/3 of signal length
        
        return {
            "window": str(window)
        }
    except Exception as e:
        # Safe fallback
        return {
            "window": "5"
        }

def _get_moving_mean_windowed_defaults(fs, total_samples):
    """Calculate improved defaults for moving_mean_windowed step"""
    try:
        # Calculate window size based on sampling rate and signal length
        # Rule: 0.2 seconds worth of samples for windowed mean analysis
        
        # Base window size on sampling rate (larger than simple moving average)
        if fs > 0:
            base_window = max(int(0.2 * fs), 5)  # 0.2 seconds or minimum 5 samples
        else:
            base_window = 20  # Safe fallback
        
        # Ensure window doesn't exceed reasonable bounds
        # Maximum: 10% of signal length to maintain meaningful analysis
        max_window = max(total_samples // 10, 5)  # At least 5 samples
        
        # Conservative minimum for effective windowed analysis
        min_window = 5
        
        # Final window size calculation
        window = max(min_window, min(base_window, max_window))
        
        # Additional safety check for very short signals
        if window >= total_samples:
            window = max(total_samples // 4, 1)  # Use 1/4 of signal length
        
        # Calculate intelligent overlap in samples
        # 50% overlap is good balance between resolution and computation
        overlap = window // 2
        
        # For very short signals, reduce overlap to ensure we get meaningful output
        if total_samples < 100:
            overlap = window // 4  # 25% overlap for short signals
        
        # Ensure overlap doesn't make step size too small
        step_size = window - overlap
        if step_size < 1:
            overlap = max(0, window - 1)  # Ensure step size >= 1
        
        return {
            "window": str(window),
            "overlap": str(overlap)
        }
    except Exception as e:
        # Safe fallback
        return {
            "window": "20",
            "overlap": "10"
        }

def _get_gaussian_smooth_defaults(fs, total_samples):
    """Calculate improved defaults for gaussian_smooth step"""
    try:
        # Calculate sigma based on sampling rate and signal length
        # Rule: 0.05 seconds worth of samples for gentle smoothing
        
        # Base sigma on sampling rate
        if fs > 0:
            base_sigma = max(int(0.05 * fs), 1)  # 0.05 seconds or minimum 1 sample
        else:
            base_sigma = 3  # Safe fallback
        
        # Ensure sigma doesn't exceed reasonable bounds
        # Maximum: 2% of signal length to avoid over-smoothing
        max_sigma = max(total_samples // 50, 1)  # At least 1 sample
        
        # Conservative minimum for effective smoothing
        min_sigma = 1
        
        # Final sigma calculation
        sigma = max(min_sigma, min(base_sigma, max_sigma))
        
        # Additional safety check for very short signals
        if sigma >= total_samples // 3:
            sigma = max(total_samples // 10, 1)  # Use 1/10 of signal length
        
        return {
            "window_std": str(sigma),
            "window_size": str(sigma * 6 + 1 if sigma * 6 + 1 > 3 else 5)  # 6-sigma window, minimum 5
        }
    except Exception as e:
        # Safe fallback
        return {
            "window_std": "2",
            "window_size": "13"  # 6-sigma window
        }

def _get_median_smooth_defaults(fs, total_samples):
    """Calculate improved defaults for median_smooth step"""
    try:
        # Calculate window size based on sampling rate and signal length
        # Rule: 0.05 seconds worth of samples for gentle median smoothing
        
        # Base window size on sampling rate
        if fs > 0:
            base_window = max(int(0.05 * fs), 3)  # 0.05 seconds or minimum 3 samples
        else:
            base_window = 5  # Safe fallback
        
        # Ensure window doesn't exceed reasonable bounds
        # Maximum: 2% of signal length to avoid over-smoothing
        max_window = max(total_samples // 50, 3)  # At least 3 samples
        
        # Conservative minimum for effective median filtering
        min_window = 3
        
        # Final window size calculation
        window = max(min_window, min(base_window, max_window))
        
        # Additional safety check for very short signals
        if window >= total_samples:
            window = max(total_samples // 5, 3)  # Use 1/5 of signal length
        
        # Ensure odd number for median filter (required for proper centering)
        if window % 2 == 0:
            window += 1
        
        # Final check to ensure window is still within bounds after making it odd
        if window > total_samples:
            window = max(total_samples - 1, 3)  # Subtract 1 and ensure minimum
            if window % 2 == 0:
                window -= 1  # Make it odd
        
        return {
            "window": str(window)
        }
    except Exception as e:
        # Safe fallback
        return {
            "window_size": "3",
            "overlap": "1"
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
        "nperseg": str(window),
        "noverlap": str(overlap),
        "window": "hann",
        "reduction": "max_intensity"
    }

def _get_cwt_spectrogram_defaults(fs, total_samples):
    """Calculate defaults for cwt_spectrogram step"""
    # Default scale range for CWT - covers typical frequencies
    min_scale = 1
    max_scale = 64
    
    # Adjust max scale based on sampling rate to avoid meaningless high frequencies
    if fs > 0:
        max_scale = min(max_scale, int(fs / 2))  # Don't exceed half sampling rate
    
    return {
        "wavelet": "morl",
        "scales": f"{min_scale}-{max_scale}",
        "reduction": "max_intensity"
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
    
    # Overlap: 50% of window (in samples)
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap)
    }

def _get_energy_sliding_defaults(fs, total_samples):
    """Calculate defaults for energy_sliding step"""
    # Window: 0.5 seconds worth of samples
    window = max(int(0.5 * fs), 10)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    # Overlap: 50% of window (in samples)
    overlap = window // 2
    
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
    
    # Overlap: 50% of window (in samples)
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap),
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
    
    # Overlap: 50% of window (in samples)
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap)
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
        "polyorder": "3"  # Cubic polynomial is common
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
        "frac": str(span)
    }

def _get_moving_skewness_defaults(fs, total_samples):
    """Calculate defaults for moving_skewness step"""
    # Window: 2 seconds worth of samples for statistical calculations
    window = max(int(2 * fs), 50)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    # Overlap: 50% of window (in samples)
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap)
    }

def _get_moving_kurtosis_defaults(fs, total_samples):
    """Calculate defaults for moving_kurtosis step"""
    # Window: 2 seconds worth of samples for statistical calculations
    window = max(int(2 * fs), 50)
    window = min(window, total_samples // 10)  # Not more than 10% of signal
    
    # Overlap: 50% of window (in samples)
    overlap = window // 2
    
    return {
        "window": str(window),
        "overlap": str(overlap)
    }

def get_step_defaults(step_name, fs, total_samples, channel):
    """
    Get intelligent defaults for any step based on its parameters.
    
    Args:
        step_name: Name of the step
        fs: Sampling frequency
        total_samples: Total number of samples
        channel: Channel object for additional context
        
    Returns:
        dict: Parameter defaults
    """
    # Import step registry to get step class
    try:
        from steps.process_registry import ProcessRegistry
        step_cls = ProcessRegistry.get(step_name)
        params = step_cls.params
    except (ImportError, KeyError, AttributeError):
        # Fallback to old step-specific functions
        return _get_legacy_step_defaults(step_name, fs, total_samples, channel)
    
    defaults = {}
    
    for param in params:
        param_name = param["name"]
        param_type = param.get("type", "str")
        param_default = param.get("default")
        
        # Generate intelligent default based on parameter name and type
        intelligent_default = _generate_param_default(
            param_name, param_type, param_default, fs, total_samples, channel
        )
        
        if intelligent_default is not None:
            defaults[param_name] = intelligent_default
    
    return defaults

def _generate_param_default(param_name, param_type, param_default, fs, total_samples, channel):
    """
    Generate intelligent default for a parameter based on its name, type, and context.
    """
    # Window-based parameters (computation time optimized)
    if param_name in ["window", "window_size"]:
        return _get_window_default(fs, total_samples, param_name)
    
    # Overlap parameters
    elif param_name in ["overlap", "noverlap"]:
        return _get_overlap_default(fs, total_samples)
    
    # Spectrogram parameters
    elif param_name in ["nperseg", "nfft"]:
        return _get_spectrogram_window_default(fs, total_samples)
    
    # FIR filter parameters
    elif param_name == "numtaps":
        return _get_fir_numtaps_default(fs, total_samples)
    
    # Frequency cutoff parameters
    elif param_name in ["cutoff", "low_cutoff", "high_cutoff"]:
        return _get_frequency_cutoff_default(param_name, fs)
    
    # Clustering/ML parameters
    elif param_name == "n_clusters":
        return _get_n_clusters_default(total_samples)
    elif param_name == "n_components":
        return _get_n_components_default(total_samples)
    elif param_name == "random_state":
        return "42"
    
    # Threshold parameters
    elif param_name == "threshold":
        return _get_threshold_default(channel)
    elif param_name == "eps":
        return _get_eps_default(channel)
    
    # Box-Cox specific parameters
    elif param_name == "shift" and step_name == "boxcox_transform":
        return _get_boxcox_shift_default(channel)
    
    # Percentile parameters
    elif param_name in ["lower_percentile", "lower"]:
        return "5.0"
    elif param_name in ["upper_percentile", "upper"]:
        return "95.0"
    elif param_name == "percentile":
        return "90.0"
    
    # Boolean parameters
    elif param_type in ["bool", "boolean"]:
        return "True" if param_default is None else str(param_default)
    
    # Numeric parameters with defaults
    elif param_type in ["int", "float"] and param_default is not None:
        return str(param_default)
    
    # String parameters with options
    elif param_type == "str" and param_default is not None:
        return str(param_default)
    
    # Fallback to original default
    return str(param_default) if param_default is not None else None

def _get_window_default(fs, total_samples, param_name):
    """Generate intelligent window size based on step type and signal properties."""
    # Base window time (seconds) - optimized for computation time
    if "smooth" in param_name or "filter" in param_name:
        target_time = 0.2  # 200ms for smoothing
        max_factor = 10    # 10% of signal length max
    elif "feature" in param_name or "energy" in param_name:
        target_time = 0.5  # 500ms for feature extraction
        max_factor = 5     # 20% of signal length max
    elif "spectrogram" in param_name:
        target_time = 1.0  # 1s for spectrograms
        max_factor = 4     # 25% of signal length max
    elif "sliding" in param_name:
        target_time = 0.5  # 500ms for sliding operations
        max_factor = 8     # 12.5% of signal length max
    else:
        target_time = 0.3  # 300ms default
        max_factor = 8     # 12.5% of signal length max
    
    # Calculate window size
    window_samples = int(target_time * fs)
    max_window = total_samples // max_factor
    
    # Apply limits
    window_samples = min(window_samples, max_window, 10000)  # Max 10k samples
    window_samples = max(window_samples, 3)  # Min 3 samples
    
    # Ensure odd for median filters
    if "median" in param_name and window_samples % 2 == 0:
        window_samples += 1
    
    return str(window_samples)

def _get_overlap_default(fs, total_samples):
    """Generate intelligent overlap based on signal properties."""
    # Calculate a reasonable window size first
    target_time = 0.3  # 300ms default window
    window_samples = int(target_time * fs)
    max_window = total_samples // 8
    window_samples = min(window_samples, max_window, 10000)
    window_samples = max(window_samples, 3)
    
    # Base overlap is 50% of window
    overlap_samples = window_samples // 2
    
    # Reduce overlap for high-frequency signals to save computation
    if fs > 1000:
        overlap_samples = window_samples // 4  # 25% overlap for high-frequency signals
    
    # Reduce overlap for very long signals
    if total_samples > 100000:
        overlap_samples = window_samples // 5  # 20% overlap for very long signals
    
    # Ensure overlap is less than window
    overlap_samples = min(overlap_samples, window_samples - 1)
    overlap_samples = max(overlap_samples, 1)  # At least 1 sample overlap
    
    return str(overlap_samples)

def _get_spectrogram_window_default(fs, total_samples):
    """Generate intelligent spectrogram window size."""
    # Target 1-2 seconds for good frequency resolution
    target_time = 1.0
    window_samples = int(target_time * fs)
    
    # Ensure power of 2 for FFT efficiency
    window_samples = 2 ** int(np.log2(window_samples))
    
    # Apply limits
    window_samples = min(window_samples, total_samples // 4, 4096)  # Max 4k samples
    window_samples = max(window_samples, 64)  # Min 64 samples
    
    return str(window_samples)

def _get_fir_numtaps_default(fs, total_samples):
    """Generate intelligent FIR filter length."""
    # Base: 50ms of data minimum
    base_taps = max(51, int(fs * 0.05))
    
    # Ensure odd
    if base_taps % 2 == 0:
        base_taps += 1
    
    # Limit based on signal length
    max_taps = total_samples // 6
    numtaps = min(base_taps, max_taps)
    
    # Apply bounds
    numtaps = max(21, numtaps)  # Min 21 taps
    numtaps = min(501, numtaps)  # Max 501 taps
    
    # Ensure odd
    if numtaps % 2 == 0:
        numtaps -= 1
    
    return str(numtaps)

def _get_frequency_cutoff_default(param_name, fs):
    """Generate intelligent frequency cutoff based on parameter name."""
    nyquist = fs / 2
    
    if param_name == "low_cutoff":
        # Low cutoff: 5% of Nyquist (remove very low frequencies)
        cutoff = nyquist * 0.05
    elif param_name == "high_cutoff":
        # High cutoff: 80% of Nyquist (preserve most signal)
        cutoff = nyquist * 0.8
    elif param_name == "cutoff":
        # Single cutoff: 40% of Nyquist (middle ground)
        cutoff = nyquist * 0.4
    else:
        cutoff = nyquist * 0.1  # Default 10% of Nyquist
    
    # Ensure reasonable bounds
    cutoff = max(cutoff, 0.1)  # Min 0.1 Hz
    cutoff = min(cutoff, nyquist * 0.95)  # Max 95% of Nyquist
    
    return str(round(cutoff, 2))

def _get_n_clusters_default(total_samples):
    """Generate intelligent number of clusters."""
    # Base on signal length, but keep reasonable
    n_clusters = min(3, total_samples // 100)
    n_clusters = max(n_clusters, 2)  # Min 2 clusters
    return str(n_clusters)

def _get_n_components_default(total_samples):
    """Generate intelligent number of components."""
    # Base on signal length, but keep reasonable
    n_components = min(2, total_samples // 1000)
    n_components = max(n_components, 1)  # Min 1 component
    return str(n_components)

def _get_threshold_default(channel):
    """Generate intelligent threshold based on signal statistics."""
    try:
        if hasattr(channel, 'ydata') and channel.ydata is not None:
            y = channel.ydata
            if len(y) > 0 and np.any(np.isfinite(y)):
                # Use median as threshold
                threshold = np.median(y)
                return str(round(threshold, 6))
    except:
        pass
    return "0.0"

def _get_eps_default(channel):
    """Generate intelligent eps for DBSCAN."""
    try:
        if hasattr(channel, 'ydata') and channel.ydata is not None:
            y = channel.ydata
            if len(y) > 1:
                # Use 10% of signal range as default eps
                signal_range = np.nanmax(y) - np.nanmin(y)
                eps = signal_range * 0.1
                eps = max(eps, 0.01)  # Minimum eps
                return str(round(eps, 3))
    except:
        pass
    return "0.5"

def _get_boxcox_shift_default(channel):
    """Generate intelligent shift for Box-Cox transform."""
    try:
        if hasattr(channel, 'ydata') and channel.ydata is not None:
            y_data = np.array(channel.ydata)
            if len(y_data) > 0 and np.any(np.isfinite(y_data)):
                finite_data = y_data[np.isfinite(y_data)]
                if len(finite_data) > 0:
                    min_val = np.min(finite_data)
                    if min_val <= 0:
                        # Calculate shift to make all values positive
                        # Add small epsilon for numerical stability
                        shift = abs(min_val) + 1e-6
                        return str(round(shift, 6))
                    else:
                        # If all values are already positive, use small shift for safety
                        return str(1e-6)
    except:
        pass
    return "0.0"

def _get_legacy_step_defaults(step_name, fs, total_samples, channel):
    """Fallback to old step-specific default functions."""
    # Step-specific intelligent defaults (legacy support)
    if step_name == "count_samples":
        return _get_count_samples_defaults(fs, total_samples, channel)
    elif step_name == "area_envelope":
        return _get_area_envelope_defaults(fs, total_samples)
    elif step_name == "moving_average":
        return _get_moving_average_defaults(fs, total_samples)
    elif step_name == "moving_mean":
        return _get_moving_mean_defaults(fs, total_samples)
    elif step_name == "moving_mean_windowed":
        return _get_moving_mean_windowed_defaults(fs, total_samples)
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
    elif step_name == "boxcox_transform":
        return _get_boxcox_transform_defaults(fs, total_samples, channel)
    # Add more legacy steps as needed
    else:
        return {}  # Return empty dict for unknown steps

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
        # Calculate Nyquist frequency
        nyquist = fs / 2
        
        # Low cutoff: 10% of Nyquist
        low_cutoff = nyquist * 0.1
        # High cutoff: 80% of Nyquist
        high_cutoff = nyquist * 0.8
        
        # Ensure both cutoffs are well below Nyquist (safety margin)
        if high_cutoff >= nyquist * 0.9:
            high_cutoff = nyquist * 0.8
        if low_cutoff >= nyquist * 0.9:
            low_cutoff = nyquist * 0.1
        
        return {
            "low_cutoff": str(round(low_cutoff, 2)),
            "high_cutoff": str(round(high_cutoff, 2)),
            "order": "2"
        }
    except Exception as e:
        return {"low_cutoff": "0.5", "high_cutoff": "4.0", "order": "2"}

def _get_bandpass_fir_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for bandpass_fir step."""
    try:
        # Calculate Nyquist frequency
        nyquist = fs / 2
        
        # Calculate intelligent numtaps based on sampling rate and signal length
        # Rule: numtaps should be odd and provide good frequency resolution
        # But not too large to avoid computational issues
        
        # Base numtaps on sampling rate - higher fs needs more taps for same resolution
        base_taps = max(51, int(fs * 0.05))  # 50ms of data minimum
        
        # Ensure it's odd (required for bandpass FIR)
        if base_taps % 2 == 0:
            base_taps += 1
        
        # Limit based on signal length - FIR filter needs signal > 3*numtaps
        max_taps = total_samples // 6  # Conservative: signal length / 6
        numtaps = min(base_taps, max_taps)
        
        # Ensure minimum and maximum bounds
        numtaps = max(21, numtaps)  # Minimum 21 taps
        numtaps = min(501, numtaps)  # Maximum 501 taps
        
        # Ensure odd
        if numtaps % 2 == 0:
            numtaps -= 1
        
        # Default frequency range based on Nyquist frequency
        # Low cutoff: remove DC and very low frequencies (1% of Nyquist)
        low_cutoff = min(nyquist * 0.01, 0.5)  # 1% of Nyquist or 0.5 Hz
        
        # High cutoff: remove high frequency noise, keep main signal (40% of Nyquist)
        high_cutoff = min(nyquist * 0.4, 20.0)  # 40% of Nyquist or 20 Hz
        
        # Ensure low < high
        if low_cutoff >= high_cutoff:
            low_cutoff = high_cutoff * 0.1
        
        # Ensure both cutoffs are well below Nyquist (safety margin)
        if high_cutoff >= nyquist * 0.9:
            high_cutoff = nyquist * 0.4
        if low_cutoff >= nyquist * 0.9:
            low_cutoff = nyquist * 0.01
        
        return {
            "low_cutoff": f"{low_cutoff:.3f}",
            "high_cutoff": f"{high_cutoff:.1f}",
            "numtaps": str(numtaps),
            "window": "hamming"
        }
    except Exception as e:
        # Safe fallback
        return {
            "low_cutoff": "0.5",
            "high_cutoff": "4.0", 
            "numtaps": "101",
            "window": "hamming"
        }

def _get_boxcox_transform_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for boxcox_transform step."""
    try:
        # Calculate intelligent shift based on signal minimum value
        shift = 0.0
        if channel and hasattr(channel, 'ydata') and channel.ydata is not None:
            y_data = np.array(channel.ydata)
            if len(y_data) > 0 and np.any(np.isfinite(y_data)):
                finite_data = y_data[np.isfinite(y_data)]
                if len(finite_data) > 0:
                    min_val = np.min(finite_data)
                    if min_val <= 0:
                        # Calculate shift to make all values positive
                        # Add small epsilon for numerical stability
                        shift = abs(min_val) + 1e-6
                    else:
                        # If all values are already positive, use small shift for safety
                        shift = 1e-6
        
        return {
            "lambda": "0.5",
            "shift": str(round(shift, 6))
        }
    except Exception as e:
        # Safe fallback
        return {
            "lambda": "0.5", 
            "shift": "0.0"
        }

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
        
        # Ensure overlap is less than window size
        # Use 50% overlap but ensure it's valid
        overlap_samples = window_samples // 2
        overlap_samples = min(overlap_samples, window_samples - 1)  # Must be less than window
        overlap_samples = max(overlap_samples, 1)  # At least 1 sample overlap
        
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
        # Calculate Nyquist frequency
        nyquist = fs / 2
        
        # Conservative highpass cutoff - remove DC and very low frequencies
        # Use 1% of Nyquist frequency, but ensure it's well below Nyquist
        cutoff = min(nyquist * 0.01, 0.1) if fs > 0 else 0.1
        
        # Ensure cutoff is well below Nyquist (safety margin)
        if cutoff >= nyquist * 0.9:
            cutoff = nyquist * 0.01
        
        return {"cutoff": str(round(cutoff, 3)), "order": "2"}
    except Exception as e:
        return {"cutoff": "0.1", "order": "2"}

def _get_highpass_fir_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for highpass_fir step."""
    try:
        # Calculate Nyquist frequency
        nyquist = fs / 2
        
        # Calculate intelligent numtaps similar to other FIR filters
        base_taps = max(51, int(fs * 0.05))  # 50ms of data minimum
        
        # Ensure it's odd
        if base_taps % 2 == 0:
            base_taps += 1
        
        # Limit based on signal length
        max_taps = total_samples // 6
        numtaps = min(base_taps, max_taps)
        
        # Bounds
        numtaps = max(21, numtaps)
        numtaps = min(501, numtaps)
        
        # Ensure odd
        if numtaps % 2 == 0:
            numtaps -= 1
        
        # Conservative highpass cutoff - remove DC and very low frequencies
        # Use 1% of Nyquist frequency, but ensure it's well below Nyquist
        cutoff = min(nyquist * 0.01, 0.1)  # 1% of Nyquist or 0.1 Hz
        
        # Ensure cutoff is well below Nyquist (safety margin)
        if cutoff >= nyquist * 0.9:
            cutoff = nyquist * 0.01
        
        return {
            "cutoff": f"{cutoff:.3f}",
            "numtaps": str(numtaps),
            "window": "hamming"
        }
    except Exception as e:
        return {
            "cutoff": "0.1",
            "numtaps": "101",
            "window": "hamming"
        }

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
        # Calculate Nyquist frequency
        nyquist = fs / 2
        
        # Conservative lowpass cutoff - remove high frequency noise
        # Use 40% of Nyquist frequency, but ensure it's well below Nyquist
        cutoff = min(nyquist * 0.4, 10.0) if fs > 0 else 10.0
        
        # Ensure cutoff is well below Nyquist (safety margin)
        if cutoff >= nyquist * 0.9:
            cutoff = nyquist * 0.4
        
        return {"cutoff": str(round(cutoff, 2)), "order": "2"}
    except Exception as e:
        return {"cutoff": "10.0", "order": "2"}

def _get_lowpass_fir_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for lowpass_fir step."""
    try:
        # Calculate Nyquist frequency
        nyquist = fs / 2
        
        # Calculate intelligent numtaps similar to bandpass
        base_taps = max(51, int(fs * 0.05))  # 50ms of data minimum
        
        # Ensure it's odd (good practice for FIR)
        if base_taps % 2 == 0:
            base_taps += 1
        
        # Limit based on signal length
        max_taps = total_samples // 6
        numtaps = min(base_taps, max_taps)
        
        # Bounds
        numtaps = max(21, numtaps)
        numtaps = min(501, numtaps)
        
        # Ensure odd
        if numtaps % 2 == 0:
            numtaps -= 1
        
        # Conservative lowpass cutoff - remove high frequency noise
        # Use 40% of Nyquist frequency, but ensure it's well below Nyquist
        cutoff = min(nyquist * 0.4, 10.0)  # 40% of Nyquist or 10 Hz
        
        # Ensure cutoff is well below Nyquist (safety margin)
        if cutoff >= nyquist * 0.9:
            cutoff = nyquist * 0.4
        
        return {
            "cutoff": f"{cutoff:.1f}",
            "numtaps": str(numtaps),
            "window": "hamming"
        }
    except Exception as e:
        return {
            "cutoff": "10.0",
            "numtaps": "101", 
            "window": "hamming"
        }

def _get_median_subtract_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for median_subtract step."""
    try:
        # Default window size (0.1 seconds or 100 samples)
        window_samples = min(int(fs * 0.1), 100) if fs > 0 else 100
        window_samples = max(window_samples, 3)  # Minimum 3 samples
        return {"window": str(window_samples)}
    except Exception as e:
        return {"window": "100"}

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
        return {"lower": "5.0", "upper": "95.0"}
    except Exception as e:
        return {"lower": "5.0", "upper": "95.0"}

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

def _get_sign_only_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for sign_only step."""
    try:
        # Default to no change
        return {}
    except Exception as e:
        return {}

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
        # Calculate optimal decomposition level based on signal length and sampling rate
        # Rule: Each level reduces frequency resolution by half
        
        # Maximum useful levels based on signal length
        max_levels_by_length = int(np.log2(total_samples)) - 2
        max_levels_by_length = max(1, max_levels_by_length)
        
        # Maximum useful levels based on sampling rate
        # Don't decompose beyond frequencies that would be too low to be meaningful
        if fs > 0:
            min_useful_freq = 0.01  # 0.01 Hz minimum meaningful frequency
            max_levels_by_freq = int(np.log2(fs / (2 * min_useful_freq)))
            max_levels_by_freq = max(1, max_levels_by_freq)
        else:
            max_levels_by_freq = 6
        
        # Choose the more conservative limit
        optimal_levels = min(max_levels_by_length, max_levels_by_freq, 8)  # Cap at 8 levels
        optimal_levels = max(2, optimal_levels)  # Minimum 2 levels
        
        # Select appropriate wavelet based on signal characteristics
        # For most biological/physical signals, db4 or db6 are good choices
        # For smoother signals, higher order wavelets work better
        if total_samples > 10000:
            # For longer signals, can use higher order wavelets
            wavelet = "db6"
        else:
            # For shorter signals, use lower order wavelets
            wavelet = "db4"
        
        # Intelligent decompose_levels selection
        # Default to levels that capture meaningful frequency bands
        if optimal_levels >= 4:
            # For signals with many levels, decompose middle levels that often contain features
            decompose_levels = f"2,3,4"
        elif optimal_levels >= 3:
            decompose_levels = f"2,3"
        else:
            decompose_levels = f"1,2"
        
        return {
            "wavelet": wavelet,
            "level": str(optimal_levels),
            "decompose_levels": decompose_levels
        }
    except Exception as e:
        return {
            "wavelet": "db4",
            "level": "4",
            "decompose_levels": "1,2"
        }

def _get_wavelet_denoise_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for wavelet_denoise step."""
    try:
        # Calculate optimal denoising parameters based on signal characteristics
        
        # Select wavelet based on signal length and characteristics
        if total_samples > 10000:
            # For longer signals, can use higher order wavelets for better denoising
            wavelet = "db6"
        else:
            # For shorter signals, use lower order wavelets
            wavelet = "db4"
        
        # Calculate optimal decomposition level
        max_levels_by_length = int(np.log2(total_samples)) - 2
        max_levels_by_length = max(1, max_levels_by_length)
        
        if fs > 0:
            # Don't decompose beyond frequencies that would remove important signal content
            min_useful_freq = 0.1  # 0.1 Hz minimum for denoising
            max_levels_by_freq = int(np.log2(fs / (2 * min_useful_freq)))
            max_levels_by_freq = max(1, max_levels_by_freq)
        else:
            max_levels_by_freq = 6
        
        optimal_levels = min(max_levels_by_length, max_levels_by_freq, 6)  # Cap at 6 for denoising
        optimal_levels = max(2, optimal_levels)  # Minimum 2 levels
        
        # Calculate intelligent threshold based on signal characteristics
        threshold = 0.1  # Default conservative threshold
        
        # If channel data is available, estimate noise level and adjust threshold
        if channel and hasattr(channel, 'ydata') and channel.ydata is not None:
            try:
                y_data = np.array(channel.ydata)
                if len(y_data) > 0 and np.any(np.isfinite(y_data)):
                    finite_data = y_data[np.isfinite(y_data)]
                    if len(finite_data) > 10:
                        # Estimate noise level using median absolute deviation
                        # This is robust to outliers
                        mad = np.median(np.abs(finite_data - np.median(finite_data)))
                        noise_estimate = mad / 0.6745  # Convert MAD to standard deviation
                        
                        # Calculate threshold as multiple of noise estimate
                        # Higher threshold for noisier signals
                        signal_std = np.std(finite_data)
                        if signal_std > 0:
                            snr_estimate = signal_std / noise_estimate
                            if snr_estimate > 10:
                                # High SNR: use lower threshold for gentle denoising
                                threshold = max(0.05, noise_estimate * 0.5)
                            elif snr_estimate > 5:
                                # Medium SNR: moderate threshold
                                threshold = max(0.1, noise_estimate * 1.0)
                            else:
                                # Low SNR: higher threshold for aggressive denoising
                                threshold = max(0.2, noise_estimate * 2.0)
                        
                        # Normalize threshold to typical range
                        threshold = min(threshold, 1.0)  # Cap at 1.0
                        threshold = max(threshold, 0.02)  # Minimum 0.02
                        
            except Exception as e:
                print(f"[WaveletDenoise] Could not estimate noise level: {e}")
                threshold = 0.1
        
        # Default to soft thresholding (more gentle, preserves signal shape)
        mode = "soft"
        
        return {
            "wavelet": wavelet,
            "level": str(optimal_levels),
            "threshold": f"{threshold:.3f}",
            "mode": mode
        }
    except Exception as e:
        return {
            "wavelet": "db4",
            "level": "4",
            "threshold": "0.1",
            "mode": "soft"
        }

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
        # Calculate intelligent window size based on sampling rate and signal length
        # Rule: 0.5 seconds worth of samples for adaptive normalization
        
        # Base window size on sampling rate
        if fs > 0:
            base_window = max(int(0.5 * fs), 10)  # 0.5 seconds or minimum 10 samples
        else:
            base_window = 50  # Safe fallback
        
        # Ensure window doesn't exceed reasonable bounds
        # Maximum: 20% of signal length to maintain local adaptation
        max_window = max(total_samples // 5, 10)  # At least 10 samples
        
        # Conservative minimum for effective normalization
        min_window = 10
        
        # Final window size calculation
        window = max(min_window, min(base_window, max_window))
        
        # Additional safety check for very short signals
        if window >= total_samples:
            window = max(total_samples // 2, 1)  # Use 1/2 of signal length
        
        # Calculate intelligent overlap
        # 50% overlap provides good coverage while maintaining efficiency
        overlap = window // 2
        
        # Ensure overlap is reasonable
        overlap = max(overlap, 1)  # At least 1 sample overlap
        overlap = min(overlap, window - 1)  # Overlap must be less than window
        
        # For very short signals, reduce overlap
        if total_samples < 50:
            overlap = max(window // 4, 1)  # Quarter overlap for short signals
        
        return {
            "range_min": "0.0",
            "range_max": "1.0",
            "window": str(window),
            "overlap": str(overlap)
        }
    except Exception as e:
        return {
            "range_min": "0.0",
            "range_max": "1.0",
            "window": "50",
            "overlap": "25"
        }

def _get_zscore_global_defaults(fs, total_samples, channel):
    """Calculate intelligent defaults for zscore_global step."""
    try:
        # Default to no change
        return {}
    except Exception as e:
        return {}

def _get_custom_defaults(fs, total_samples, channel):
    """Get intelligent defaults for custom step"""
    return {
        "script": "# Custom processing script\n# Input: x, y, fs\n# Output: y_new\n\ny_new = y  # Example: pass-through"
    }

# === New BioSPPy Step Defaults ===

def _get_hamilton_segmenter_defaults(fs, total_samples, channel):
    """Get intelligent defaults for Hamilton segmenter step"""
    # Hamilton segmenter works well with standard ECG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_ecg_processing_defaults(fs, total_samples, channel):
    """Get intelligent defaults for ECG processing step"""
    # ECG processing works well with standard ECG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_emg_processing_defaults(fs, total_samples, channel):
    """Get intelligent defaults for EMG processing step"""
    # EMG processing works well with standard EMG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_eda_processing_defaults(fs, total_samples, channel):
    """Get intelligent defaults for EDA processing step"""
    # EDA processing works well with standard EDA sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_resp_processing_defaults(fs, total_samples, channel):
    """Get intelligent defaults for respiration processing step"""
    # Respiration processing works well with standard respiration sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_bandpower_defaults(fs, total_samples, channel):
    """Get intelligent defaults for bandpower step"""
    # Intelligent band selection based on sampling rate
    if fs >= 1000:
        # High sampling rate - include higher frequency bands
        bands = "0.5-4,4-8,8-12,12-30,30-100"
    elif fs >= 250:
        # Medium sampling rate - standard bands
        bands = "0.5-4,4-8,8-12,12-30"
    else:
        # Lower sampling rate - limit to lower frequencies
        nyquist = fs / 2
        if nyquist > 30:
            bands = "0.5-4,4-8,8-12,12-30"
        elif nyquist > 12:
            bands = "0.5-4,4-8,8-12"
        else:
            bands = "0.5-4,4-8"
    
    return {
        "bands": bands
    }

def _get_signal_stats_defaults(fs, total_samples, channel):
    """Get intelligent defaults for signal statistics step"""
    # Signal statistics don't require parameters
    return {}  # No parameters needed

def _get_smoother_defaults(fs, total_samples, channel):
    """Get intelligent defaults for BioSPPy smoother step"""
    # Intelligent window size based on sampling rate
    if fs >= 1000:
        # High sampling rate - larger window
        size = 20
    elif fs >= 250:
        # Medium sampling rate - medium window
        size = 10
    else:
        # Lower sampling rate - smaller window
        size = 5
    
    return {
        "method": "moving_average",
        "size": size
    }

# === Updated Envelope Step Defaults ===

def _get_cumulative_max_defaults(fs, total_samples, channel):
    """Get intelligent defaults for cumulative max step"""
    # Intelligent window size based on sampling rate and signal length
    if fs >= 1000:
        # High sampling rate - larger window
        window = min(500, total_samples // 10)
    elif fs >= 250:
        # Medium sampling rate - medium window
        window = min(200, total_samples // 10)
    else:
        # Lower sampling rate - smaller window
        window = min(100, total_samples // 10)
    
    # Ensure minimum window size
    window = max(10, window)
    
    # Intelligent overlap - typically 50% of window
    overlap = window // 2
    
    return {
        "window": window,
        "overlap": overlap
    }

def _get_area_envelope_defaults(fs, total_samples, channel):
    """Get intelligent defaults for area envelope step"""
    # Intelligent window size based on sampling rate
    if fs >= 1000:
        # High sampling rate - larger window
        window = min(100, total_samples // 20)
    elif fs >= 250:
        # Medium sampling rate - medium window
        window = min(50, total_samples // 20)
    else:
        # Lower sampling rate - smaller window
        window = min(25, total_samples // 20)
    
    # Ensure minimum window size
    window = max(5, window)
    
    # No overlap by default for area envelope
    overlap = 0
    
    return {
        "window": window,
        "overlap": overlap
    }

# === Missing Step Defaults ===

def _get_detect_zero_crossings_defaults(fs, total_samples, channel):
    """Get intelligent defaults for zero crossings detection step"""
    # Zero crossing detection typically doesn't need parameters
    return {}  # No parameters needed

def _get_log_transform_defaults(fs, total_samples, channel):
    """Get intelligent defaults for log transform step"""
    # Try to determine appropriate offset based on signal statistics
    try:
        if hasattr(channel, 'ydata') and channel.ydata is not None:
            y_data = np.array(channel.ydata)
            
            # Filter out non-finite values
            finite_mask = np.isfinite(y_data)
            if np.any(finite_mask):
                finite_data = y_data[finite_mask]
                
                # Get minimum and maximum of finite data
                y_min = np.min(finite_data)
                y_max = np.max(finite_data)
                
                # Calculate offset to ensure all values are positive after offset
                if y_min <= 0:
                    # If signal has negative or zero values, add offset
                    # Use abs(min) + small epsilon to ensure positive
                    offset = abs(y_min) + 1e-6
                else:
                    # If signal is already positive, add small offset for numerical stability
                    # Use a small fraction of the data range or minimum value
                    data_range = y_max - y_min
                    if data_range > 0:
                        # Use 1% of data range as offset
                        offset = max(1e-6, data_range * 0.01)
                    else:
                        # If no range (constant signal), use small offset
                        offset = max(1e-6, abs(y_min) * 0.01)
                
                # Additional safety check: ensure offset is reasonable
                if offset < 1e-6:
                    offset = 1e-6
                elif offset > 1e6:
                    offset = 1.0  # Cap at reasonable value
            else:
                # No finite data available, use safe default
                offset = 1.0
        else:
            # No channel data available, use safe default
            offset = 1.0
    except Exception as e:
        # Fallback to safe default
        offset = 1.0
    
    return {
        "offset": str(round(offset, 6))
    }

def _get_ppg_processing_defaults(fs, total_samples, channel):
    """Generate intelligent defaults for PPG processing with HeartPy"""
    defaults = {}
    
    # Use channel sampling rate if available
    if hasattr(channel, 'fs_median') and channel.fs_median:
        defaults["sample_rate"] = str(float(channel.fs_median))
    elif hasattr(channel, 'fs') and channel.fs:
        defaults["sample_rate"] = str(float(channel.fs))
    else:
        defaults["sample_rate"] = str(100.0)  # Default for PPG
    
    # Window size: adaptive based on sampling rate
    # Default 0.75 seconds, but adjust for very low or high sampling rates
    if fs > 0:
        if fs < 50:
            windowsize = 1.5  # Longer window for low sampling rates
        elif fs > 500:
            windowsize = 0.5  # Shorter window for high sampling rates
        else:
            windowsize = 0.75  # Standard window
    else:
        windowsize = 0.75
    
    defaults["windowsize"] = str(windowsize)
    
    # Enable both time and frequency domain analysis by default
    defaults["report_time"] = "True"
    defaults["report_freq"] = "True"
    
    # Use FFT for frequency analysis (fastest and most reliable)
    defaults["freq_method"] = "fft"
    
    # Enable artifact removal for robust processing
    defaults["interp_clipping"] = "True"
    defaults["clipping_scale"] = "1.3"
    defaults["interp_threshold"] = "1020"
    
    # Enable Hampel correction for outlier removal
    defaults["hampel_correct"] = "True"
    defaults["hampel_threshold"] = "3.0"
    
    return defaults

def _get_ppg_cleaned_baseline_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PPG cleaned baseline step"""
    # PPG baseline removal works well with standard PPG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_ppg_filtered_bandpass_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PPG filtered bandpass step"""
    defaults = {}
    
    # Intelligent cutoff frequency based on PPG characteristics
    # PPG signals typically have cardiac components around 0.8-3 Hz
    # We want to preserve these while removing low-frequency baseline drift and high-frequency noise
    
    if fs > 0:
        # Calculate intelligent cutoff based on sampling rate
        # For PPG, we typically want to preserve frequencies around 0.8-3 Hz
        # Use 0.8 Hz as the cutoff to preserve cardiac information
        cutoff = 0.8
        
        # Ensure cutoff is reasonable for the sampling rate
        nyquist = fs / 2
        if cutoff >= nyquist:
            cutoff = nyquist * 0.1  # Use 10% of Nyquist frequency
        
        # Ensure minimum reasonable cutoff
        if cutoff < 0.1:
            cutoff = 0.1
    else:
        # Default cutoff for unknown sampling rate
        cutoff = 0.8
    
    defaults["cutoff"] = str(round(cutoff, 2))
    
    return defaults

def _get_ppg_hrv_features_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PPG HRV features step"""
    # PPG HRV feature extraction works well with standard PPG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_ppg_nn_intervals_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PPG NN intervals step"""
    # PPG NN interval extraction works well with standard PPG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_ppg_rr_intervals_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PPG RR intervals step"""
    # PPG RR interval extraction works well with standard PPG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_ppg_scaled_signal_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PPG scaled signal step"""
    # PPG signal scaling works well with standard PPG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_ppg_smoothed_signal_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PPG smoothed signal step"""
    # PPG signal smoothing works well with standard PPG sampling rates
    return {}  # No parameters needed, fs is auto-detected

def _get_power_spectral_density_defaults(fs, total_samples, channel):
    """Get intelligent defaults for power spectral density step"""
    defaults = {}
    
    # Intelligent window size based on sampling rate and signal length
    if fs > 0:
        # Aim for ~1-2 seconds of data in each window for good frequency resolution
        target_window_time = 1.0  # seconds
        target_window_samples = int(target_window_time * fs)
        
        # Ensure window is reasonable size and power of 2 for efficiency
        window_size = min(target_window_samples, total_samples // 4)
        window_size = max(window_size, 64)  # Minimum window size
        
        # Round to nearest power of 2 for FFT efficiency
        window_size = 2 ** int(np.log2(window_size))
        window_size = min(window_size, 4096)  # Maximum reasonable window size
        
        # Set overlap to 50% of window size (standard practice)
        overlap = window_size // 2
        
        defaults["window"] = str(window_size)
        defaults["overlap"] = str(overlap)
    else:
        # Fallback defaults
        defaults["window"] = "1024"
        defaults["overlap"] = "512"
    
    return defaults 

def _get_standard_scaler_defaults(fs, total_samples, channel):
    """Get intelligent defaults for standard scaler step"""
    # Standard scaler typically doesn't need parameters beyond defaults
    return {}  # No parameters needed

def _get_robust_scaler_defaults(fs, total_samples, channel):
    """Get intelligent defaults for robust scaler step"""
    # Robust scaler typically doesn't need parameters beyond defaults
    return {}  # No parameters needed

def _get_quantile_transformer_defaults(fs, total_samples, channel):
    """Get intelligent defaults for quantile transformer step"""
    # Quantile transformer typically doesn't need parameters beyond defaults
    return {}  # No parameters needed

def _get_agglomerative_clustering_defaults(fs, total_samples, channel):
    """Get intelligent defaults for agglomerative clustering step"""
    # Default to 2 clusters for most cases
    return {
        "n_clusters": "2",
        "linkage": "ward"
    }

def _get_dbscan_defaults(fs, total_samples, channel):
    """Get intelligent defaults for DBSCAN clustering step"""
    # Try to estimate eps based on signal characteristics
    try:
        if hasattr(channel, 'ydata') and channel.ydata is not None:
            y = channel.ydata
            if len(y) > 1:
                # Use 10% of signal range as default eps
                signal_range = np.nanmax(y) - np.nanmin(y)
                eps = signal_range * 0.1
                eps = max(eps, 0.01)  # Minimum eps
            else:
                eps = 0.5
        else:
            eps = 0.5
    except:
        eps = 0.5
    
    return {
        "eps": str(round(eps, 3)),
        "min_samples": "5"
    }

def _get_kmeans_defaults(fs, total_samples, channel):
    """Get intelligent defaults for K-Means clustering step"""
    # Default to 3 clusters for most cases
    return {
        "n_clusters": "3",
        "random_state": "42"
    }

def _get_pca_defaults(fs, total_samples, channel):
    """Get intelligent defaults for PCA step"""
    # Default to 2 components for visualization
    return {
        "n_components": "2",
        "random_state": "42"
    }

def _get_power_transformer_defaults(fs, total_samples, channel):
    """Get intelligent defaults for power transformer step"""
    # Default to Yeo-Johnson method (more robust)
    return {
        "method": "yeo-johnson",
        "standardize": "True"
    }

def _get_select_kbest_defaults(fs, total_samples, channel):
    """Get intelligent defaults for SelectKBest step"""
    # Default to selecting top 10 features
    return {
        "k": "10",
        "target_column": ""
    }

def _get_tsne_defaults(fs, total_samples, channel):
    """Get intelligent defaults for t-SNE step"""
    # Default perplexity based on data size
    if total_samples < 30:
        perplexity = 5.0
    elif total_samples < 100:
        perplexity = 15.0
    else:
        perplexity = 30.0
    
    return {
        "perplexity": str(perplexity),
        "random_state": "42"
    }

def _get_variance_threshold_defaults(fs, total_samples, channel):
    """Get intelligent defaults for variance threshold step"""
    # Default to removing only constant features
    return {
        "threshold": "0.0"
    }

def _get_bollinger_bands_defaults(fs, total_samples, channel):
    """Get intelligent defaults for Bollinger Bands step"""
    defaults = {}
    
    # Calculate intelligent window size based on sampling rate
    if fs > 0:
        # Use 20 seconds worth of data for Bollinger Bands (typical for financial analysis)
        target_window_time = 20.0  # seconds
        target_window_samples = int(target_window_time * fs)
        
        # Ensure window is reasonable size
        window_size = min(target_window_samples, total_samples // 4)
        window_size = max(window_size, 10)  # Minimum window size
        window_size = min(window_size, 1000)  # Maximum reasonable window size
    else:
        # Fallback for unknown sampling rate
        window_size = min(100, total_samples // 4)
        window_size = max(window_size, 10)
    
    defaults["window"] = str(window_size)
    defaults["n_std"] = "2.0"  # Standard 2 standard deviations
    
    return defaults

def _get_garch_forecast_defaults(fs, total_samples, channel):
    """Get intelligent defaults for GARCH forecast step"""
    defaults = {}
    
    # Calculate intelligent forecast horizon based on signal length
    if total_samples > 1000:
        horizon = 20  # Longer horizon for longer signals
    elif total_samples > 500:
        horizon = 15
    elif total_samples > 200:
        horizon = 10
    else:
        horizon = 5  # Shorter horizon for short signals
    
    # Ensure horizon is reasonable
    horizon = min(horizon, total_samples // 10)  # Don't exceed 10% of signal length
    horizon = max(horizon, 1)  # Minimum horizon of 1
    
    defaults["horizon"] = str(horizon)
    
    return defaults

def _get_ruptures_changepoint_defaults(fs, total_samples, channel):
    """Get intelligent defaults for ruptures change point detection step"""
    defaults = {}
    
    # Calculate intelligent penalty based on signal characteristics
    try:
        if hasattr(channel, 'ydata') and channel.ydata is not None:
            y = channel.ydata
            if len(y) > 1:
                # Use signal variance as a base for penalty calculation
                signal_variance = np.nanvar(y)
                if signal_variance > 0:
                    # Scale penalty with signal variance and length
                    base_penalty = np.sqrt(signal_variance) * np.log(len(y))
                    penalty = max(base_penalty, 1.0)  # Minimum penalty
                    penalty = min(penalty, 50.0)  # Maximum reasonable penalty
                else:
                    penalty = 5.0
            else:
                penalty = 5.0
        else:
            penalty = 5.0
    except:
        penalty = 5.0
    
    defaults["pen"] = str(round(penalty, 2))
    
    return defaults

def _get_volatility_estimation_defaults(fs, total_samples, channel):
    """Get intelligent defaults for volatility estimation step"""
    defaults = {}
    
    # Calculate intelligent window size based on sampling rate
    if fs > 0:
        # Use 10 seconds worth of data for volatility estimation
        target_window_time = 10.0  # seconds
        target_window_samples = int(target_window_time * fs)
        
        # Ensure window is reasonable size
        window_size = min(target_window_samples, total_samples // 4)
        window_size = max(window_size, 10)  # Minimum window size
        window_size = min(window_size, 500)  # Maximum reasonable window size
    else:
        # Fallback for unknown sampling rate
        window_size = min(50, total_samples // 4)
        window_size = max(window_size, 10)
    
    defaults["window"] = str(window_size)
    defaults["method"] = "std"  # Default to standard deviation method
    
    return defaults

def _get_autocorrelation_defaults(fs, total_samples, channel):
    """Get intelligent defaults for autocorrelation step"""
    defaults = {}
    
    # Calculate intelligent max_lag based on signal length
    # Use 10% of signal length or 100 samples, whichever is smaller
    max_lag = min(total_samples // 10, 100)
    max_lag = max(max_lag, 10)  # Minimum lag of 10
    max_lag = min(max_lag, 500)  # Maximum lag of 500
    
    defaults["max_lag"] = str(max_lag)
    
    return defaults

def _get_isolation_forest_defaults(fs, total_samples, channel):
    """Get intelligent defaults for isolation forest step"""
    defaults = {}
    
    # Default contamination based on signal characteristics
    # Use 5% as default, which is reasonable for most anomaly detection tasks
    contamination = 0.05
    
    defaults["contamination"] = str(contamination)
    
    return defaults

def _get_random_forest_feature_importance_defaults(fs, total_samples, channel):
    """Get intelligent defaults for random forest feature importance step"""
    # This step requires features and labels in metadata, no additional parameters needed
    return {}

def _get_svc_classifier_defaults(fs, total_samples, channel):
    """Get intelligent defaults for SVC classifier step"""
    # This step requires features and labels in metadata, no additional parameters needed
    return {}

def _get_minmax_scaler_defaults(fs, total_samples, channel):
    """Get intelligent defaults for minmax scaler step"""
    # MinMax scaler typically doesn't need parameters beyond defaults
    return {}

def _get_tsfresh_features_defaults(fs, total_samples, channel):
    """Get intelligent defaults for tsfresh features step"""
    defaults = {}
    
    # Calculate intelligent defaults based on signal length
    if total_samples > 10000:
        # For long signals, use more conservative settings
        defaults["minimal_features"] = "True"
        defaults["n_jobs"] = "4"
    elif total_samples > 5000:
        # For medium signals, use balanced settings
        defaults["minimal_features"] = "False"
        defaults["n_jobs"] = "2"
    else:
        # For short signals, can use more features
        defaults["minimal_features"] = "False"
        defaults["n_jobs"] = "1"
    
    return defaults

def _get_tsfresh_features_heatmap_defaults(fs, total_samples, channel):
    """Get intelligent defaults for tsfresh features heatmap step"""
    defaults = {}
    
    # Calculate intelligent defaults based on signal length
    if total_samples > 10000:
        # For long signals, use more conservative settings
        defaults["minimal_features"] = "True"
        defaults["n_jobs"] = "4"
    elif total_samples > 5000:
        # For medium signals, use balanced settings
        defaults["minimal_features"] = "False"
        defaults["n_jobs"] = "2"
    else:
        # For short signals, can use more features
        defaults["minimal_features"] = "False"
        defaults["n_jobs"] = "1"
    
    return defaults