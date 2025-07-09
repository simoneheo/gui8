import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class top_percentile_sliding_step(BaseStep):
    name = "top_percentile_sliding"
    category = "Envelope"
    description = """Compute the top percentile value in each sliding window for envelope extraction.
    
This step calculates the specified percentile (e.g., 95th percentile) within each sliding
window to create an envelope that follows the upper trend of the signal. This is useful
for extracting the upper envelope while being robust to outliers.

• **Window size**: Number of samples in each sliding window
• **Percentile**: Percentile value to compute (0-100, higher values follow upper envelope)

Useful for:
• **Envelope extraction**: Create upper envelope that follows signal peaks
• **Outlier robustness**: Percentile-based approach is less sensitive to outliers than max
• **Trend analysis**: Identify upper trends in noisy signals
• **Peak detection**: Prepare signals for peak detection algorithms"""
    tags = ["time-series", "envelope", "percentile", "sliding-window", "window", "sliding", "robust"]
    params = [
        {
            "name": "window", 
            "type": "int", 
            "default": "25", 
            "help": "Sliding window size in samples (must be positive)"
        },
        {
            "name": "percentile", 
            "type": "float", 
            "default": "95.0", 
            "help": "Top percentile to compute in each window (0-100, higher = upper envelope)"
        },
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Compute top percentile in sliding windows for envelope extraction (Category: {cls.category})"

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
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window = params.get("window")
        percentile = params.get("percentile")
        
        if window <= 0:
            raise ValueError("Window size must be positive")
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("Percentile must be between 0 and 100")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Top percentile sliding produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Top percentile sliding produced unexpected infinite values")

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
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply top percentile sliding window processing to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Validate signal length vs window size
            if len(y) < params["window"]:
                raise ValueError(f"Signal too short for window: signal length={len(y)}, window={params['window']}")
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="TopPercentileSliding"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Top percentile envelope computation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for top percentile sliding window"""
        window = int(params['window'])
        q = params['percentile']
        
        y_new = []
        for i in range(len(y)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(y), i + window // 2 + 1)
            window_data = np.abs(y[start_idx:end_idx])
            
            if len(window_data) == 0:
                raise ValueError(f"Empty window at index {i}")
            
            percentile_val = np.percentile(window_data, q)
            if np.isnan(percentile_val) or np.isinf(percentile_val):
                raise ValueError(f"Invalid percentile value at index {i}")
            
            y_new.append(percentile_val)
        y_new = np.array(y_new)     
        return y_new
