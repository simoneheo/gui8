import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class gaussian_smooth_step(BaseStep):
    name = "gaussian_smooth"
    category = "Filter"
    description = """Applies a 1D Gaussian smoother to the signal.
Uses scipy's gaussian_filter1d for efficient convolution-based smoothing."""
    tags = ["time-series", "smoothing", "filter", "scipy", "gaussian", "gaussian_filter1d", "kernel"]
    params = [
        {
            "name": "window_std", 
            "type": "float", 
            "default": "2.0", 
            "help": "Standard deviation of Gaussian kernel (larger = more smoothing)"
        },
        {
            "name": "window_size", 
            "type": "int", 
            "default": "21", 
            "help": "Window size in samples (odd integer > 3, larger = wider smoothing)"
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

    @classmethod
    def _validate_parameters(cls, params: dict, total_samples: int) -> None:
        """Validate parameters and business rules"""
        window_std = params.get("window_std")
        window_size = params.get("window_size")
        
        if window_std is None or window_std <= 0:
            raise ValueError("Standard deviation must be positive")
        if np.isnan(window_std):
            raise ValueError("Standard deviation cannot be NaN")
        
        if window_size is None or window_size <= 3:
            raise ValueError("Window size must be > 3")
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        if window_size > total_samples:
            raise ValueError(
                f"Window size ({window_size} samples) is larger than signal length ({total_samples} samples). "
                f"Try a smaller window or use a longer signal."
            )

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Gaussian smoothing produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Gaussian smoothing produced unexpected infinite values")

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
        """Apply Gaussian smoothing to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params, len(y))
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="GaussianSmooth"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Gaussian smoothing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for Gaussian smoothing"""
        from scipy.ndimage import gaussian_filter1d
        
        window_std = params["window_std"]
        window_size = params["window_size"]
        
        y_new = gaussian_filter1d(y, sigma=window_std, truncate=(window_size / (2 * window_std)))
        
        return y_new
