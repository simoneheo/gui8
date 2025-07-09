import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class savitzky_golay_step(BaseStep):
    name = "savitzky_golay"
    category = "Filter"
    description = """Apply Savitzky-Golay smoothing filter to reduce noise while preserving signal features.
    
This step applies a Savitzky-Golay filter, which uses polynomial fitting within a sliding window
to smooth the signal. Unlike simple averaging, this method preserves peaks, valleys, and other
important signal features while reducing noise.

• **Window size**: Size of the smoothing window (odd integer >= 3)
• **Polynomial order**: Order of the polynomial fit (must be < window size)

Useful for:
• **Noise reduction**: Smooth noisy signals while preserving features
• **Peak preservation**: Maintain important signal peaks and valleys
• **Signal enhancement**: Improve signal quality for analysis
• **Feature extraction**: Prepare signals for peak detection algorithms"""
    tags = ["time-series", "smoothing", "noise-reduction", "scipy", "polynomial", "savgol", "peaks"]
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
    def get_info(cls): 
        return f"{cls.name} — Apply Savitzky-Golay smoothing filter (Category: {cls.category})"

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

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window_size = params.get("window_size")
        polyorder = params.get("polyorder")
        
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(f"Window size must be odd and >= 3, got {window_size}")
        if polyorder >= window_size:
            raise ValueError(f"Polynomial order must be less than window size. Got order={polyorder}, window={window_size}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Savitzky-Golay smoothing produced unexpected NaN values")

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
        """Apply Savitzky-Golay smoothing to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Validate signal length vs window size
            window_size = params.get("window_size", 5)
            if len(y) < window_size:
                raise ValueError(f"Signal too short for smoothing: requires signal length > {window_size}, got {len(y)}")
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="SavitzkyGolay"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Savitzky-Golay smoothing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for Savitzky-Golay smoothing"""
        window_size = params.get("window_size", 5)
        polyorder = params.get("polyorder", 2)
        
        from scipy.signal import savgol_filter
        y_new = savgol_filter(y, window_length=window_size, polyorder=polyorder)
        return y_new
