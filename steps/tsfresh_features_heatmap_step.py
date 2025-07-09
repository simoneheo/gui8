
import numpy as np
import pandas as pd
import warnings


from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class tsfresh_heatmap_features_step(BaseStep):
    name = "tsfresh_heatmap_features_step"
    category = "Features"
    description = """Extract statistical features using tsfresh in sliding windows and return as a heatmap channel.

This step computes tsfresh features over sliding windows of a signal and returns a single 2D heatmap-style channel:
• x = time segments (center of window)
• y = features (mean, std, etc.)
• metadata['matrix'] = feature values (shape: [features x time])

Useful for:
• Feature evolution visualization
• Anomaly detection or trend detection
• Temporal pattern interpretation"""
    tags = ["tsfresh", "heatmap", "feature-extraction", "time-frequency", "evolution"]
    params = [
        {
            "name": "window_size",
            "type": "int",
            "default": 200,
            "help": "Window size in samples (must be positive)"
        },
        {
            "name": "overlap",
            "type": "int",
            "default": 100,
            "help": "Overlap in samples (must be < window_size)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Heatmap feature evolution (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 10:
            raise ValueError("Signal too short for tsfresh heatmap (minimum 10 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window_size = params.get("window_size")
        overlap = params.get("overlap")
        
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("Overlap must be non-negative and less than window size")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, matrix: np.ndarray, feature_names: list) -> None:
        """Validate output data"""
        if matrix.size == 0:
            raise ValueError("tsfresh computation produced empty feature matrix")
        if len(feature_names) == 0:
            raise ValueError("No features were extracted")
        if np.any(np.isnan(matrix)) and not np.any(np.isnan(y_original)):
            raise ValueError("tsfresh computation produced unexpected NaN values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        """Apply tsfresh heatmap feature extraction to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', 1000.0)
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Validate window size vs signal length
            window_size = params.get("window_size", 200)
            if window_size > len(y):
                raise ValueError(f"Window size ({window_size}) is larger than signal length ({len(y)})")
            
            # Process the data
            matrix, feature_names, segment_times = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, matrix, feature_names)
            
            # Create heatmap channel
            heatmap_channel = cls.create_new_channel(
                parent=channel,
                xdata=segment_times,
                ydata=np.arange(matrix.shape[0]),
                params=params,
                suffix="tsfresh_heatmap"
            )
            
            # Set heatmap-specific properties
            heatmap_channel.tags = ["spectrogram", "feature-evolution"]
            heatmap_channel.xlabel = "Time (s)"
            heatmap_channel.ylabel = "Feature Index"
            heatmap_channel.legend_label = f"{channel.legend_label} - tsfresh Heatmap"
            
            # Store heatmap data in metadata
            heatmap_channel.metadata = {
                "matrix": matrix,
                "feature_names": feature_names,
            }
            
            return [heatmap_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"tsfresh heatmap computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core processing logic for tsfresh heatmap feature extraction"""
        from tsfresh import extract_features, select_features
        from tsfresh.utilities.dataframe_functions import impute

        # Suppress tsfresh warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            window_size = params.get("window_size", 200)
            overlap = params.get("overlap", 100)
            hop = window_size - overlap

            # Generate sliding windows
            segments = []
            segment_times = []
            for i in range(0, len(y) - window_size + 1, hop):
                segments.append(y[i:i+window_size])
                center = i + window_size // 2
                segment_times.append(x[center] if len(x) > center else center / fs)

            if len(segments) == 0:
                raise ValueError("No valid windows found")

            # Prepare data for tsfresh
            df = pd.DataFrame({
                "id": np.repeat(range(len(segments)), window_size),
                "time": np.tile(np.arange(window_size), len(segments)),
                "value": np.concatenate(segments)
            })

            # Extract features using tsfresh
            features = extract_features(df, column_id="id", column_sort="time", default_fc_parameters={
                "mean": None,
                "standard_deviation": None,
                "variance": None,
                "maximum": None,
                "minimum": None,
                "median": None
            }, disable_progressbar=True)
            
            # Handle missing values
            impute(features)

            # Convert to matrix format
            matrix = features.to_numpy().T
            feature_names = list(features.columns)
            segment_times = np.array(segment_times)

            return matrix, feature_names, segment_times
