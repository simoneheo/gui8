import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel


def savitzky_golay(y, window_size=5, polyorder=2):
    from scipy.signal import savgol_filter
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"Window size must be odd and >= 3, got {window_size}")
    if polyorder >= window_size:
        raise ValueError(f"Polynomial order must be less than window size. Got order={polyorder}, window={window_size}")
    if len(y) < window_size:
        raise ValueError(f"Signal too short for smoothing: requires signal length > {window_size}, got {len(y)}")
    return savgol_filter(y, window_length=window_size, polyorder=polyorder)


@register_step
class savitzky_golay_step(BaseStep):
    name = "savitzky_golay"
    category = "Smoother"
    description = "Applies a Savitzky-Golay smoother to the signal."
    tags = ["time-series"]
    params = [
        {
            'name': 'window_size', 
            'type': 'int', 
            'default': '5', 
            'help': 'Window size in samples (odd integer >= 3, larger = more smoothing)',
        }, 
        {
            'name': 'polyorder', 
            'type': 'int', 
            'default': '2', 
            'help': 'Polynomial order (must be < window size, higher = preserves peaks better)',
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
            polyorder = int(user_input.get("polyorder", "2"))
            
            # Validation
            if window_size < 3:
                raise ValueError("Window size must be at least 3")
            if window_size % 2 == 0:
                raise ValueError("Window size must be odd")
            if polyorder < 1:
                raise ValueError("Polynomial order must be at least 1")
            if polyorder >= window_size:
                raise ValueError(f"Polynomial order ({polyorder}) must be less than window size ({window_size})")
            
            parsed["window_size"] = window_size
            parsed["polyorder"] = polyorder
            
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Window size and polynomial order must be valid integers")
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
            y_new = savitzky_golay(channel.ydata, **params)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Savitzky-Golay smoothing failed: {str(e)}")
