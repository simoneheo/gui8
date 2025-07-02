import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel


def moving_average(y, window_size=5):
    if window_size < 1:
        raise ValueError(f"Window size must be >= 1, got {window_size}")
    if window_size % 2 == 0:
        raise ValueError(f"Window size must be an odd number, got {window_size}")
    if len(y) < window_size:
        raise ValueError(f"Signal too short for smoothing: requires signal length > {window_size}, got {len(y)}")
    kernel = np.ones(window_size) / window_size
    return np.convolve(y, kernel, mode='same')


@register_step
class moving_average_step(BaseStep):
    name = "moving_average"
    category = "Smoother"
    description = "Applies a moving average smoother to the signal."
    tags = ["time-series"]
    params = [
        {
            'name': 'window_size', 
            'type': 'int', 
            'default': '5', 
            'help': 'Window size in samples (odd integer >= 1, larger = more smoothing)',
        }
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        try:
            window_size = int(user_input.get("window_size", "5"))
            
            if window_size < 1:
                raise ValueError("Window size must be at least 1")
            if window_size % 2 == 0:
                raise ValueError("Window size must be odd")
                
            parsed["window_size"] = window_size
            
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Window size must be a valid integer")
            raise e
            
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Validate input data
        if len(channel.ydata) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaN values")
        
        window_size = params.get("window_size", 5)
        if len(channel.ydata) < window_size:
            raise ValueError(f"Signal too short: requires at least {window_size} samples but got {len(channel.ydata)}")
            
        try:
            y_new = moving_average(channel.ydata, **params)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Moving average failed: {str(e)}")
