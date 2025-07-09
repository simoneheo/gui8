import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel
from scipy.signal import find_peaks

@register_step
class detect_peaks_step(BaseStep):
    name = "detect_peaks"
    category = "Features"
    description = """Detects peaks in the signal using height, distance, and prominence criteria.
Automatically adjusts parameters if not specified to ensure robust peak detection."""
    tags = ["time-series", "event", "peaks", "scipy", "find_peaks", "maxima", "detection"]
    params = [
        {
            "name": "height", 
            "type": "float", 
            "default": "", 
            "help": "Minimum peak height (amplitude). Leave blank for auto-detection based on signal statistics."
        },
        {
            "name": "distance", 
            "type": "int", 
            "default": "1", 
            "help": "Minimum distance between peaks in samples. Use larger values to avoid detecting multiple peaks in noisy regions."
        },
        {
            "name": "prominence", 
            "type": "float", 
            "default": "", 
            "help": "Minimum peak prominence (how much a peak stands out from surrounding baseline). Leave blank for auto-detection."
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"
    
    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")
        if len(y) < 3:
            raise ValueError("Signal too short for peak detection (minimum 3 samples)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        distance = params.get("distance")
        
        if distance is not None and distance < 1:
            raise ValueError("Distance must be at least 1")
        
        # Height and prominence can be None (auto-detection)
        height = params.get("height")
        if height is not None and np.isnan(height):
            raise ValueError("Height cannot be NaN")
        
        prominence = params.get("prominence")
        if prominence is not None and prominence < 0:
            raise ValueError("Prominence must be non-negative")

    @classmethod
    def _validate_output_data(cls, x_output: np.ndarray, y_output: np.ndarray) -> None:
        """Validate output signal data"""
        if len(x_output) != len(y_output):
            raise ValueError("Output time and signal data length mismatch")

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
        """Apply peak detection to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            x_output, y_output = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(x_output, y_output)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x_output, 
                ydata=y_output, 
                params=params,
                suffix="Peaks"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Peak detection failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic for peak detection"""
        # Prepare parameters for find_peaks (only include non-None parameters)
        kwargs = {}
        if params.get("height") is not None:
            kwargs["height"] = params["height"]
        if params.get("distance") is not None:
            kwargs["distance"] = params["distance"]
        if params.get("prominence") is not None:
            kwargs["prominence"] = params["prominence"]
        
        indices, _ = find_peaks(y, **kwargs)
        
        if len(indices) == 0:
            raise ValueError("No peaks detected. Try adjusting height, distance, or prominence parameters.")
        
        x_new = x[indices]
        y_new = y[indices] 
        return x_new, y_new
