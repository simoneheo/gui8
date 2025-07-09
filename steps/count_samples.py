import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class count_samples_step(BaseStep):
    name = "count_samples"
    category = "Features"
    description = "Count samples within sliding windows and report counts in various units (count/window, count/s, count/min)"
    tags = ["time-series", "feature", "peak", "count", "threshold", "detection", "events"]
    params = [
        {
            "name": "window", 
            "type": "int", 
            "default": "1000", 
            "help": "Window size in number of samples. Must be positive and smaller than total signal length."
        },
        {
            "name": "overlap", 
            "type": "int", 
            "default": "500", 
            "help": "Window overlap in number of samples. Must be less than window size. Higher overlap gives smoother results but more computation."
        },
        {
            "name": "unit", 
            "type": "str", 
            "default": "count/window", 
            "options": ["count/window", "count/min", "count/s"],
            "help": "Output unit for sample counts. 'count/window' gives raw counts, others extrapolate to rates."
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"
    
    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, x: np.ndarray, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input signal is empty")
        if len(x) != len(y):
            raise ValueError(f"Time and signal data length mismatch: {len(x)} vs {len(y)}")
        if len(x) < 2:
            raise ValueError("Channel must have at least 2 data points for sample counting")
        if np.any(np.isnan(x)):
            raise ValueError("Time data contains NaN values")
        if np.any(np.isinf(x)):
            raise ValueError("Time data contains infinite values")
        if not np.all(np.diff(x) > 0):
            raise ValueError("Time data must be monotonically increasing")

    @classmethod
    def _validate_parameters(cls, params: dict, total_samples: int) -> None:
        """Validate parameters and business rules"""
        window_samples = params.get("window")
        overlap_samples = params.get("overlap")
        unit = params.get("unit")
        
        if window_samples is None or window_samples <= 0:
            raise ValueError("Window size must be positive")
        if window_samples > 1000000:
            raise ValueError(f"Window size seems too large ({window_samples} samples). Maximum allowed is 1,000,000 samples")
        if window_samples > total_samples:
            raise ValueError(
                f"Window size ({window_samples} samples) is larger than signal length ({total_samples} samples). "
                f"Try a smaller window or use a longer signal."
            )
        
        if overlap_samples is None or overlap_samples < 0:
            raise ValueError("Overlap must be non-negative")
        if overlap_samples >= window_samples:
            raise ValueError(f"Overlap ({overlap_samples}) must be less than window size ({window_samples})")
        
        valid_units = ["count/window", "count/min", "count/s"]
        if unit not in valid_units:
            raise ValueError(f"Unit must be one of {valid_units}")
        
        # Check if we'll have enough windows
        step_samples = window_samples - overlap_samples
        if step_samples <= 0:
            raise ValueError(f"Step size must be positive (window_samples - overlap_samples = {step_samples})")
        
        estimated_windows = int((total_samples - window_samples) / step_samples) + 1
        if estimated_windows < 1:
            raise ValueError(
                f"Configuration would produce no valid windows. "
                f"Window: {window_samples} samples, Overlap: {overlap_samples} samples, Signal: {total_samples} samples"
            )
        if estimated_windows > 100000:
            raise ValueError(
                f"Configuration would produce too many windows ({estimated_windows}). "
                f"Consider using a larger window or less overlap."
            )

    @classmethod
    def _validate_output_data(cls, x_output: np.ndarray, y_output: np.ndarray) -> None:
        """Validate output signal data"""
        if len(x_output) == 0 or len(y_output) == 0:
            raise ValueError("No output data generated")
        if len(x_output) != len(y_output):
            raise ValueError("Output time and signal data length mismatch")
        if np.any(np.isnan(y_output)):
            raise ValueError("Output contains NaN values")
        if np.any(np.isinf(y_output)):
            raise ValueError("Output contains infinite values")

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
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "float":
                    parsed[name] = float(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply sample counting to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(x, y)
            cls._validate_parameters(params, len(x))
            
            # Process the data
            x_output, y_output = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(x_output, y_output)
            
            # Determine appropriate ylabel based on unit
            unit = params["unit"]
            ylabel_map = {
                "count/window": "Sample Count",
                "count/s": "Samples/Second",
                "count/min": "Samples/Minute"
            }
            ylabel = ylabel_map.get(unit, "Sample Count")
            
            # Create new channel
            new_channel = cls.create_new_channel(
                parent=channel, 
                xdata=x_output, 
                ydata=y_output, 
                params=params
            )
            
            # Set channel properties
            new_channel.xlabel = "Time (s)"
            new_channel.ylabel = ylabel
            new_channel.legend_label = f"{channel.legend_label} - Sample Count ({unit})"
            
            # Add metadata for debugging/analysis
            new_channel.metadata = {
                'original_samples': len(channel.xdata),
                'window_samples': params["window"],
                'overlap_samples': params["overlap"],
                'num_windows': len(x_output),
                'unit': unit,
                'mean_count': float(np.mean(y_output)),
                'std_count': float(np.std(y_output))
            }
            
            return new_channel
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Sample counting failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for sample counting"""
        window_samples = params["window"]
        overlap_samples = params["overlap"]
        unit = params["unit"]
        
        total_samples = len(x)
        step_samples = window_samples - overlap_samples
        
        # Generate sliding windows
        window_starts = []
        current_idx = 0
        
        while current_idx + window_samples <= total_samples:
            window_starts.append(current_idx)
            current_idx += step_samples
        
        # Estimate sampling frequency for rate calculations
        time_diffs = np.diff(x)
        median_dt = np.median(time_diffs)
        sampling_freq = 1.0 / median_dt if median_dt > 0 else 1.0
        
        # Calculate window duration in seconds
        window_duration = window_samples / sampling_freq
        
        # Count samples in each window
        x_new = []
        y_new = []
        
        for start_idx in window_starts:
            end_idx = start_idx + window_samples
            
            # Extract window data
            window_x = x[start_idx:end_idx]
            window_y = y[start_idx:end_idx]
            
            # Count actual samples in window
            sample_count = len(window_y)
            
            # Calculate center time for this window
            center_time = window_x[len(window_x) // 2] if len(window_x) > 0 else x[start_idx]
            
            # Convert count based on selected unit
            if unit == "count/window":
                output_value = sample_count
            elif unit == "count/s":
                output_value = sample_count / window_duration
            elif unit == "count/min":
                output_value = (sample_count / window_duration) * 60.0
            else:
                raise ValueError(f"Unknown unit: {unit}")
            
            x_new.append(center_time)
            y_new.append(output_value)
        
        # Convert to numpy arrays
        x_new = np.array(x_new)
        y_new = np.array(y_new)
        
        return x_new, y_new 
