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
            "type": "float", 
            "default": "1.0", 
            "help": "Window size in seconds. Must be positive and smaller than total signal duration."
        },
        {
            "name": "overlap", 
            "type": "float", 
            "default": "0.5", 
            "help": "Window overlap in seconds. Must be less than window size. Higher overlap gives smoother results but more computation."
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
        window_time = cls.validate_float_parameter(
            "window", 
            params.get("window"), 
            min_val=0.001,
            max_val=3600.0
        )
        
        # Validate overlap parameter
        overlap_time = cls.validate_float_parameter(
            "overlap", 
            params.get("overlap"), 
            min_val=0.0,
            max_val=window_time-0.001
        )
        
        # Validate unit parameter
        cls.validate_string_parameter(
            "unit",
            params.get("unit", "count/window"),
            valid_options=["count/window", "count/min", "count/s"]
        )

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        window_time = float(params["window"])
        overlap_time = float(params["overlap"])
        unit = params["unit"]

        # Define time step between windows
        step_time = window_time - overlap_time
        if step_time <= 0:
            raise ValueError("Overlap must be smaller than window size.")

        # Determine total duration
        t_start = float(x[0])
        t_end = float(x[-1])
        time_points = []
        counts = []

        current_start = t_start
        while current_start + window_time <= t_end:
            current_end = current_start + window_time

            # Count number of peaks within the current window
            in_window = (x >= current_start) & (x < current_end)
            count = np.count_nonzero(in_window)

            duration = current_end - current_start
            center_time = current_start + (duration / 2)

            # Convert to desired unit
            if unit == "count/window":
                output_value = count
            elif unit == "count/s":
                output_value = count / duration
            elif unit == "count/min":
                output_value = (count / duration) * 60
            else:
                raise ValueError(f"Unknown unit: {unit}")

            time_points.append(center_time)
            counts.append(output_value)

            current_start += step_time

        return [{
            'tags': ['time-series'],
            'x': np.array(time_points),
            'y': np.array(counts)
        }]
