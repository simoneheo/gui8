
import numpy as np
import pandas as pd
import warnings

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

# Suppress deprecated pkg_resources warning from tsfresh
warnings.filterwarnings("ignore", category=UserWarning, module="tsfresh")

@register_step
class tsfresh_features_step(BaseStep):
    name = "tsfresh_features_step"
    category = "Features"
    description = """Extract sliding window statistical features using tsfresh.

This step segments the input signal into overlapping windows and extracts a subset of useful tsfresh features.
It outputs:
• Multiple time-series channels (one per feature)
• A single bar-chart channel for global feature summary

Features extracted per window (varies by feature set):
• Basic: Mean, StdDev, Variance, Max, Min, Median
• Extended: + MeanAbsChange, LongestStrikeAboveMean, Skewness, Kurtosis
• Entropy: SampleEntropy, ApproxEntropy, BinnedEntropy + basic stats
• Spectral: FFTAggregated, CID_CE + basic stats

Useful for:
• **Feature engineering**
• **ML preprocessing**
• **Anomaly detection**
• **Pattern recognition**
• **Signal characterization**
"""

    tags = ["feature-extraction", "tsfresh", "statistical", "time-series", "sliding-window", "machine-learning"]
    
    # Define feature presets
    FEATURE_SETS = {
        "basic": {
            "mean": None,
            "standard_deviation": None,
            "variance": None,
            "maximum": None,
            "minimum": None,
            "median": None
        },
        "extended": {
            "mean_abs_change": None,
            "longest_strike_above_mean": None,
            "skewness": None,
            "kurtosis": None,
            "mean": None,
            "standard_deviation": None,
            "variance": None,
            "maximum": None,
            "minimum": None,
            "median": None
        },
        "entropy": {
            "sample_entropy": None,
            "approximate_entropy": None,
            "binned_entropy": {"max_bins": 10},
            "mean": None,
            "standard_deviation": None
        },
        "spectral": {
            "fft_aggregated": {"aggtype": "variance"},
            "cid_ce": {"normalize": True},
            "mean": None,
            "standard_deviation": None
        },
        "custom": {}  # Leave empty for now, enable later
    }
    
    params = [
        {
            "name": "window_size",
            "type": "int",
            "default": 200,
            "description": "Window size in samples for sliding window feature extraction",
            "help": "Larger windows capture more global patterns, smaller windows provide better temporal resolution"
        },
        {
            "name": "overlap",
            "type": "int",
            "default": 100,
            "description": "Overlap in samples between consecutive windows",
            "help": "Higher overlap provides smoother feature evolution but increases computation time"
        },
        {
            "name": "feature_set",
            "type": "select",
            "options": ["basic", "extended", "entropy", "spectral", "custom"],
            "default": "basic",
            "description": "Which feature group to compute",
            "help": "Basic = mean/std/max/etc. Extended = more time-domain stats. Entropy = complexity. Spectral = FFT-based."
        },
        {
            "name": "top_n_features",
            "type": "int",
            "default": 6,
            "description": "Number of top features to return",
            "help": "Only return the top-N features with highest variance"
        },
        {
            "name": "fs",
            "type": "float",
            "default": "",
            "help": "Sampling frequency (Hz) - automatically detected from channel if not provided"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Extracts statistical features using tsfresh (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _get_channel_fs(cls, channel: Channel) -> float:
        """Get sampling frequency from channel"""
        if hasattr(channel, 'fs_median') and channel.fs_median:
            return float(channel.fs_median)
        elif hasattr(channel, 'fs') and channel.fs:
            return float(channel.fs)
        else:
            return 1000.0  # Default fallback

    @classmethod
    def _get_feature_display_name(cls, feature_name: str) -> str:
        """Map tsfresh feature names to readable display names"""
        # Extract the feature type from tsfresh naming convention
        # tsfresh names are typically like: "value__mean", "value__standard_deviation", etc.
        if "__" in feature_name:
            feature_type = feature_name.split("__")[-1]
        else:
            feature_type = feature_name
        
        # Map to readable names
        feature_mapping = {
            "mean": "Mean",
            "standard_deviation": "StdDev",
            "variance": "Variance", 
            "maximum": "Max",
            "minimum": "Min",
            "median": "Median",
            "mean_abs_change": "MeanAbsChange",
            "longest_strike_above_mean": "LongestStrikeAboveMean",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "sample_entropy": "SampleEntropy",
            "approximate_entropy": "ApproxEntropy",
            "binned_entropy": "BinnedEntropy",
            "fft_aggregated": "FFTAggregated",
            "cid_ce": "CID_CE"
        }
        
        return feature_mapping.get(feature_type, feature_type.title())

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if len(y) < 32:
            raise ValueError("Signal too short for feature extraction (minimum 32 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate tsfresh feature extraction parameters"""
        window_size = params.get("window_size", 200)
        overlap = params.get("overlap", 100)
        feature_set = params.get("feature_set", "basic")
        top_n_features = params.get("top_n_features", 6)
        fs = params.get("fs")
        
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer")
        if window_size < 16:
            raise ValueError("Window size too small (minimum 16 samples)")
        if window_size > 10000:
            raise ValueError("Window size too large (maximum 10000 samples)")
            
        if not isinstance(overlap, int) or overlap < 0:
            raise ValueError("Overlap must be a non-negative integer")
        if overlap >= window_size:
            raise ValueError("Overlap must be less than window size")
        
        valid_feature_sets = ["basic", "extended", "entropy", "spectral", "custom"]
        if feature_set not in valid_feature_sets:
            raise ValueError(f"Feature set must be one of {valid_feature_sets}")
            
        if not isinstance(top_n_features, int) or top_n_features <= 0:
            raise ValueError("Top N features must be a positive integer")
        if top_n_features > 50:
            raise ValueError("Top N features too large (maximum 50)")
            
        if fs is not None and (not isinstance(fs, (int, float)) or fs <= 0):
            raise ValueError("Sampling frequency must be a positive number")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, features: pd.DataFrame, segment_times: np.ndarray) -> None:
        """Validate output feature data"""
        if features.empty:
            raise ValueError("Feature extraction produced no results")
        if len(segment_times) == 0:
            raise ValueError("No time segments generated")
        if len(segment_times) != len(features):
            raise ValueError("Number of time segments doesn't match number of feature rows")
        if np.any(np.isnan(features.values)) and not np.any(np.isnan(y_original)):
            raise ValueError("Feature extraction produced unexpected NaN values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            if name == "fs":
                continue  # Skip fs as it's injected from channel
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = param.get("default")
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "select":
                    parsed[name] = str(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        """Apply tsfresh feature extraction to the channel data and return multiple channels"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel
            fs = cls._get_channel_fs(channel)
            if fs is None:
                fs = 1000.0  # Default sampling rate
            
            # Inject sampling frequency into params
            params["fs"] = fs
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            features, segment_times = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, features, segment_times)
            
            # Create output channels
            output_channels = []
            
            # Create time-series channels for each feature
            for feature_name in features.columns:
                y_feature = features[feature_name].values
                
                # Map tsfresh feature names to readable names
                feature_display_name = cls._get_feature_display_name(feature_name)
                
                channel_out = cls.create_new_channel(
                    parent=channel,
                    xdata=np.array(segment_times),
                    ydata=y_feature,
                    params=params,
                    suffix=feature_display_name
                )
                channel_out.tags = ["feature", "tsfresh", "time-series"]
                channel_out.xlabel = "Time (s)" if hasattr(channel, 'xlabel') and 's' in str(channel.xlabel) else "Time"
                channel_out.ylabel = feature_display_name
                output_channels.append(channel_out)

            # Create global bar chart for feature summary
            summary_values = features.mean().values
            feature_names = [cls._get_feature_display_name(col) for col in features.columns]
            
            bar_chart = cls.create_new_channel(
                parent=channel,
                xdata=np.arange(len(features.columns)),
                ydata=summary_values,
                params=params,
                suffix="tsfresh_summary"
            )
            bar_chart.tags = ["bar-chart", "tsfresh", "summary"]
            bar_chart.xlabel = "Feature Type"
            bar_chart.ylabel = "Mean Value"
            
            # Store feature names in metadata for reference
            bar_chart.metadata = {
                "feature_names": feature_names,
                "feature_types": list(features.columns)
            }
            
            output_channels.append(bar_chart)

            return output_channels
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Tsfresh feature extraction failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core tsfresh feature extraction logic"""
        from tsfresh import extract_features, select_features
        from tsfresh.utilities.dataframe_functions import impute

        # Get parameters
        window_size = int(params.get('window_size', 200))
        overlap = int(params.get('overlap', 100))
        feature_set = params.get('feature_set', 'basic')
        top_n_features = int(params.get('top_n_features', 6))
        
        # Calculate hop size
        hop_size = window_size - overlap
        
        # Ensure window size doesn't exceed signal length
        if window_size > len(y):
            window_size = len(y) // 2
            hop_size = max(1, window_size // 2)
        
        # Create segments
        segments = []
        segment_times = []
        
        # Calculate time offset from original data
        time_offset = 0.0
        if len(x) > 0:
            # If x data exists, use the time of the first sample as offset
            time_offset = x[0]
        
        for i in range(0, len(y) - window_size + 1, hop_size):
            segments.append(y[i:i+window_size])
            
            # Time point is center of window
            center_sample = i + window_size // 2
            if len(x) > center_sample:
                # Use original time data (preserves offset naturally)
                segment_times.append(x[center_sample])
            else:
                # Fallback: Calculate time with preserved offset
                segment_times.append((center_sample / fs) + time_offset)

        # Create DataFrame for tsfresh
        df = pd.DataFrame({
            "id": np.repeat(range(len(segments)), window_size),
            "time": np.tile(range(window_size), len(segments)),
            "value": np.concatenate(segments)
        })

        # Get feature parameters based on selected feature set
        fc_params = cls.FEATURE_SETS.get(feature_set, cls.FEATURE_SETS["basic"])
        
        # Extract features using tsfresh
        features = extract_features(df, column_id="id", column_sort="time", 
                                  default_fc_parameters=fc_params, 
                                  disable_progressbar=True)
        
        # Impute any missing values
        impute(features)
        
        # Select top-N features based on variance
        if not features.empty and len(features.columns) > top_n_features:
            # Calculate feature importance (variance)
            feature_variance = features.var()
            # Sort by variance and select top-N
            top_features = feature_variance.nlargest(top_n_features).index
            features = features[top_features]
        
        return features, segment_times
