import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class count_samples_step(BaseStep):
    name = "count samples"
    category = "Features"
    description = """Count samples within sliding windows and report counts in various units.
Supports counting based on thresholds, peaks, or all samples with flexible output units (count/window, count/s, count/min)."""
    tags = ["feature", "peak", "count", "threshold", "detection", "events"]
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
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        # Validate window parameter
        window_samples = cls.validate_integer_parameter(
            "window", 
            params.get("window"), 
            min_val=1,
            max_val=1000000
        )
        
        # Validate overlap parameter
        overlap_samples = cls.validate_integer_parameter(
            "overlap", 
            params.get("overlap"), 
            min_val=0,
            max_val=window_samples-1
        )
        
        # Validate unit parameter
        cls.validate_string_parameter(
            "unit",
            params.get("unit", "count/window"),
            valid_options=["count/window", "count/min", "count/s"]
        )

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
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
        
        return [
            {
                'tags': ['time-series'],
                'x': x_new,
                'y': y_new
            }
        ] 