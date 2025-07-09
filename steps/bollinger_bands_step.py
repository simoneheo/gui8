
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class bollinger_bands_step(BaseStep):
    name = "bollinger_bands"
    category = "Features"
    description = "Generate Bollinger Bands (rolling mean ± n_std × rolling standard deviation) for volatility analysis"
    tags = ["finance", "envelope", "bollinger", "rolling", "statistical"]
    params = [
        {
            "name": "window",
            "type": "int",
            "default": 20,
            "help": "Window size for rolling statistics (must be positive)"
        },
        {
            "name": "n_std",
            "type": "float",
            "default": 2.0,
            "help": "Number of standard deviations for band width (must be positive)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Bollinger Bands (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for Bollinger Bands (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        window = params.get("window")
        n_std = params.get("n_std")
        
        if window <= 0:
            raise ValueError("window must be positive")
        if n_std <= 0:
            raise ValueError("n_std must be positive")
        if window > len(y) if 'y' in locals() else False:
            raise ValueError("window size cannot exceed signal length")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, upper: np.ndarray, mean: np.ndarray, lower: np.ndarray) -> None:
        """Validate output data"""
        if upper.size == 0 or mean.size == 0 or lower.size == 0:
            raise ValueError("Bollinger Bands computation produced empty results")
        if len(upper) != len(y_original) or len(mean) != len(y_original) or len(lower) != len(y_original):
            raise ValueError("Bollinger Bands output length doesn't match input signal length")
        if np.any(upper < lower):
            raise ValueError("Upper band should not be below lower band")

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
        """Apply Bollinger Bands computation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            upper, mean, lower = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, upper, mean, lower)
            
            # Create Bollinger Bands channels
            upper_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=upper,
                params=params,
                suffix="BollingerUpper"
            )
            upper_channel.tags = ["finance", "envelope", "bollinger", "upper"]
            upper_channel.legend_label = f"{channel.legend_label} (Bollinger Upper)"
            
            mean_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=mean,
                params=params,
                suffix="BollingerMean"
            )
            mean_channel.tags = ["finance", "envelope", "bollinger", "mean"]
            mean_channel.legend_label = f"{channel.legend_label} (Bollinger Mean)"
            
            lower_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=lower,
                params=params,
                suffix="BollingerLower"
            )
            lower_channel.tags = ["finance", "envelope", "bollinger", "lower"]
            lower_channel.legend_label = f"{channel.legend_label} (Bollinger Lower)"
            
            return [upper_channel, mean_channel, lower_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Bollinger Bands computation failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> tuple:
        """Core processing logic for Bollinger Bands computation"""
        window = params.get("window", 20)
        n_std = params.get("n_std", 2.0)
        
        # Ensure window doesn't exceed signal length
        window = min(window, len(y))
        
        # Compute rolling mean and standard deviation
        mean = np.array([np.mean(y[max(0, i - window + 1):i + 1]) for i in range(len(y))])
        std = np.array([np.std(y[max(0, i - window + 1):i + 1]) for i in range(len(y))])
        
        # Compute Bollinger Bands
        upper = mean + n_std * std
        lower = mean - n_std * std
        
        return upper, mean, lower
