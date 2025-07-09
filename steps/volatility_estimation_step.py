
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class volatility_estimation_step(BaseStep):
    name = "volatility_estimation"
    category = "Statistical"
    description = """Compute rolling volatility using rolling standard deviation or MAD.

This step calculates time-varying volatility measures using either:
• Standard deviation (std): Traditional volatility measure
• Median Absolute Deviation (mad): Robust volatility measure

Useful for:
• Financial time series volatility analysis
• Risk assessment and monitoring
• Signal variability tracking
• Time-varying statistical analysis"""
    tags = ["volatility", "rolling", "finance", "statistical", "mad"]
    params = [
        {
            "name": "window",
            "type": "int",
            "default": 20,
            "help": "Rolling window size (must be positive)"
        },
        {
            "name": "method",
            "type": "str",
            "default": "std",
            "options": ["std", "mad"],
            "help": "Volatility metric: 'std' for standard deviation, 'mad' for median absolute deviation"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Rolling volatility estimation (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for volatility estimation (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window = params.get("window")
        method = params.get("method")
        
        if window <= 0:
            raise ValueError("window must be positive")
        if method not in ["std", "mad"]:
            raise ValueError("method must be either 'std' or 'mad'")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, volatility: np.ndarray) -> None:
        """Validate output data"""
        if volatility.size == 0:
            raise ValueError("Volatility estimation produced empty results")
        if len(volatility) != len(y_original):
            raise ValueError("Volatility output length doesn't match input signal length")
        if np.any(volatility < 0):
            raise ValueError("Volatility values should be non-negative")

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
    def apply(cls, channel: Channel, params: dict) -> list:
        """Apply volatility estimation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            volatility = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, volatility)
            
            # Create volatility channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=volatility,
                params=params,
                suffix="Volatility"
            )
            
            # Set channel properties
            method = params.get("method", "std")
            new_channel.tags = ["volatility", "rolling", "finance", method]
            new_channel.legend_label = f"{channel.legend_label} (Volatility {method.upper()})"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Volatility estimation failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for volatility estimation"""
        window = params.get("window", 20)
        method = params.get("method", "std")
        
        # Ensure window doesn't exceed signal length
        window = min(window, len(y))
        
        if method == "std":
            # Standard deviation method
            y_out = np.array([np.std(y[max(0, i - window + 1):i + 1]) for i in range(len(y))])
        elif method == "mad":
            # Median Absolute Deviation method
            y_out = np.array([
                np.median(np.abs(y[max(0, i - window + 1):i + 1] - np.median(y[max(0, i - window + 1):i + 1]))) 
                for i in range(len(y))
            ])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return y_out
