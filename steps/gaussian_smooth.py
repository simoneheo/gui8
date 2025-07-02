import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def gaussian_smooth(y, window_std=2.0, window_size=21):
    from scipy.ndimage import gaussian_filter1d

    if window_std <= 0:
        raise ValueError(f"Standard deviation must be positive, got {window_std}")
    if window_size <= 3 or window_size % 2 == 0:
        raise ValueError(f"Window size must be an odd integer > 3, got {window_size}")
    if len(y) < window_size:
        raise ValueError(f"Signal too short for smoothing: requires signal length > {window_size}, got {len(y)}")

    try:
        return gaussian_filter1d(y, sigma=window_std, truncate=(window_size / (2 * window_std)))
    except Exception as e:
        raise ValueError(f"Gaussian smoothing failed: {str(e)}")

@register_step
class gaussian_smooth_step(BaseStep):
    name = "gaussian_smooth"
    category = "Smoother"
    description = "Applies a 1D Gaussian smoother to the signal."
    tags = ["time-series"]
    params = [
        {
            "name": "window_std", 
            "type": "float", 
            "default": "2.0", 
            "help": "Standard deviation of Gaussian kernel (larger = more smoothing)",
        },
        {
            "name": "window_size", 
            "type": "int", 
            "default": "21", 
            "help": "Window size in samples (odd integer > 3, larger = wider smoothing)",
        }
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            name = param["name"]
            value = user_input.get(name, param["default"])
            
            try:
                if param["type"] == "float":
                    parsed_val = float(value)
                    if name == "window_std" and parsed_val <= 0:
                        raise ValueError("Standard deviation must be positive")
                    parsed[name] = parsed_val
                elif param["type"] == "int":
                    parsed_val = int(value)
                    if name == "window_size":
                        if parsed_val <= 3:
                            raise ValueError("Window size must be > 3")
                        if parsed_val % 2 == 0:
                            raise ValueError("Window size must be odd")
                    parsed[name] = parsed_val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
                
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Validate input data
        if len(channel.ydata) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaN values")
            
        y_new = gaussian_smooth(channel.ydata, **params)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
