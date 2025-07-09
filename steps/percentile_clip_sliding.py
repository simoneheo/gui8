import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class percentile_clip_sliding_step(BaseStep):
    name = "percentile_clip_sliding"
    category = "General"
    description = """Apply percentile-based clipping within sliding windows to remove outliers.
    
This step applies percentile clipping within overlapping sliding windows, where each window
defines its own clipping bounds based on the specified percentiles. This provides adaptive
outlier removal that adjusts to local signal characteristics.

• **Lower/Upper percentiles**: Percentile bounds for clipping (0-100)
• **Window size**: Number of samples in each clipping window
• **Overlap**: Number of overlapping samples between windows (must be < window size)

Useful for:
• **Adaptive outlier removal**: Remove outliers based on local statistics
• **Noise reduction**: Clip extreme values that vary by region
• **Signal cleaning**: Remove artifacts while preserving local structure
• **Data quality improvement**: Handle non-stationary noise patterns"""
    tags = ["time-series", "outlier-removal", "data-cleaning", "window", "sliding", "percentile", "robust"]
    params = [
        {'name': 'lower', 'type': 'float', 'default': '1.0', 'help': 'Lower percentile (0–100)'},
        {'name': 'upper', 'type': 'float', 'default': '99.0', 'help': 'Upper percentile (0–100)'},
        {'name': 'window', 'type': 'int', 'default': '100', 'help': 'Window size in samples (must be >= 2)'},
        {'name': 'overlap', 'type': 'int', 'default': '50', 'help': 'Overlap in samples (must be < window size)'}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Apply percentile clipping within sliding windows (Category: {cls.category})"

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
        lower = params.get("lower")
        upper = params.get("upper")
        window = params.get("window")
        overlap = params.get("overlap")
        
        if not (0 <= lower < upper <= 100):
            raise ValueError(f"Invalid percentile range: {lower}–{upper}")
        if window < 2:
            raise ValueError("Window size must be >= 2")
        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap must be non-negative and less than window size")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Percentile clipping produced unexpected NaN values")

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
        """Apply percentile clipping within sliding windows to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_new,
                params=params,
                suffix="PercentileClip"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Percentile clipping failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for percentile clipping within sliding windows"""
        lower = params['lower']
        upper = params['upper']
        window = params['window']
        overlap = params['overlap']
        
        step = max(1, window - overlap)
        y_new = np.copy(y)
        
        for start in range(0, len(y) - window + 1, step):
            end = start + window
            segment = y[start:end]
            lo = np.percentile(segment, lower)
            hi = np.percentile(segment, upper)
            y_new[start:end] = np.clip(segment, lo, hi)

        return y_new
