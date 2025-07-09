import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class standardize_step(BaseStep):
    name = "standardize"
    category = "Transform"
    description = """Scale signal to custom mean and standard deviation.
    
This step standardizes the signal by first centering it around zero (subtracting the mean)
and scaling it to unit standard deviation, then scaling to the target standard deviation
and shifting to the target mean.

• **Target mean**: Desired mean value for the standardized signal
• **Target std**: Desired standard deviation for the standardized signal (must be > 0)

Useful for:
• **Signal normalization**: Make signals comparable across different scales
• **Feature scaling**: Prepare signals for machine learning algorithms
• **Statistical analysis**: Standardize signals for statistical comparisons
• **Data preprocessing**: Normalize data before further processing"""
    tags = ["time-series", "normalization", "scaling", "standardize", "zscore", "mean", "std"]
    params = [
        {"name": "mean", "type": "float", "default": "0.0", "help": "Target mean (must be finite)"},
        {"name": "std", "type": "float", "default": "1.0", "help": "Target std (must be > 0)"},
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Scale signal to custom mean and standard deviation (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.any(np.isnan(y)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(y)):
            raise ValueError("Input signal contains infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        mean = params.get("mean")
        std = params.get("std")
        
        if np.isnan(mean) or np.isinf(mean):
            raise ValueError(f"Target mean must be a finite number, got {mean}")
        if np.isnan(std) or np.isinf(std):
            raise ValueError(f"Target std must be a finite number, got {std}")
        if std <= 0:
            raise ValueError(f"Target std must be positive, got {std}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
            raise ValueError("Standardization resulted in invalid values")

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
        """Apply standardization to the channel data."""
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
                suffix="Standardize"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Standardization failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for standardization"""
        target_mean = params['mean']
        target_std = params['std']
        
        # Compute current statistics
        current_mean = np.mean(y)
        current_std = np.std(y)
        
        if current_std == 0:
            raise ValueError("Input signal has zero standard deviation (constant signal)")
        
        if np.isnan(current_mean) or np.isinf(current_mean):
            raise ValueError("Failed to compute signal mean")
        if np.isnan(current_std) or np.isinf(current_std):
            raise ValueError("Failed to compute signal standard deviation")
        
        # Apply standardization
        y_new = ((y - current_mean) / current_std) * target_std + target_mean
        return y_new
