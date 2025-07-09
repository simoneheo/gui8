import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class boxcox_transform_step(BaseStep):
    name = "boxcox_transform"
    category = "Transform"
    description = "Apply Box-Cox transformation to make data more normally distributed and stabilize variance."
    tags = ["time-series", "transform", "normalization", "scipy", "boxcox", "lambda", "power"]
    params = [
        {
            'name': 'lmbda', 
            'type': 'float', 
            'default': '0.0', 
            'help': 'Lambda parameter: 0=log transform, 0.5=square root, 1=no transform, -1=reciprocal',
            'options': ['0.0', '0.5', '1.0', '-0.5', '-1.0', '2.0']
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
        if np.any(y <= 0):
            raise ValueError("Box-Cox transform requires all values to be positive")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        lmbda = params.get("lmbda")
        
        if lmbda is None:
            raise ValueError("Lambda parameter is required")
        if np.isnan(lmbda):
            raise ValueError("Lambda cannot be NaN")
        if np.isinf(lmbda):
            raise ValueError("Lambda cannot be infinite")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Processing produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Processing produced unexpected infinite values")
        if np.all(np.isnan(y_new)):
            raise ValueError("Processing produced only NaN values")
        if np.all(np.isinf(y_new)):
            raise ValueError("Processing produced only infinite values")

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
                elif param["type"] == "bool":
                    parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the processing step to a channel"""
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', None)
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_final = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_final,
                params=params,
                suffix="BoxCox"
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Box-Cox transformation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """Core processing logic"""
        from scipy.stats import boxcox
        
        lmbda = params["lmbda"]
        
        # Ensure all values are positive for Box-Cox transform
        y_shifted = y - np.min(y) + 1e-4
        
        y_new = boxcox(y_shifted, lmbda=lmbda)
        return y_new
