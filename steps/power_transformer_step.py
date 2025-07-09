
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class power_transformer_step(BaseStep):
    name = "power_transformer"
    category = "scikit-learn"
    description = """Apply PowerTransformer (Box-Cox or Yeo-Johnson) to scale data for Gaussian-like distribution.

This step transforms data to follow a more Gaussian-like distribution using power transformations.
Useful for:
• Making non-Gaussian data more Gaussian-like
• Improving performance of algorithms that assume normality
• Handling skewed distributions
• Preparing data for statistical modeling"""
    tags = ["preprocessing", "scaling", "transformation", "scikit-learn", "normalization"]
    params = [
        {
            "name": "method",
            "type": "str",
            "default": "yeo-johnson",
            "options": ["yeo-johnson", "box-cox"],
            "help": "Method to use for transformation"
        },
        {
            "name": "standardize",
            "type": "bool",
            "default": True,
            "help": "Whether to standardize output (zero mean, unit variance)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Power transformation for Gaussian-like distribution (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for power transformation (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal contains constant values (cannot transform)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        method = params.get("method")
        standardize = params.get("standardize")
        
        if method not in ["yeo-johnson", "box-cox"]:
            raise ValueError("method must be 'yeo-johnson' or 'box-cox'")
        if not isinstance(standardize, bool):
            raise ValueError("standardize must be a boolean")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_transformed: np.ndarray) -> None:
        """Validate output data"""
        if y_transformed.size == 0:
            raise ValueError("Power transformation produced empty output")
        if len(y_transformed) != len(y_original):
            raise ValueError("Transformed signal length doesn't match input signal length")
        if np.any(np.isnan(y_transformed)) and not np.any(np.isnan(y_original)):
            raise ValueError("Power transformation produced unexpected NaN values")

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
                    if isinstance(val, str):
                        parsed[name] = val.lower() in ['true', '1', 'yes', 'on']
                    else:
                        parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        """Apply power transformation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_transformed = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_transformed)
            
            # Create transformed channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_transformed,
                params=params,
                suffix="PowerTransformed"
            )
            
            # Set channel properties
            new_channel.tags = ["transformed", "power-transform", "normalized"]
            new_channel.legend_label = f"{channel.legend_label} (Power Transformed)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Power transformation failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for power transformation"""
        from sklearn.preprocessing import PowerTransformer
        
        method = params.get("method", "yeo-johnson")
        standardize = params.get("standardize", True)
        
        # Check for NaN values
        if np.any(np.isnan(y)):
            raise ValueError("NaNs present in input signal")
        
        # Reshape for sklearn
        y_reshaped = y.reshape(-1, 1)
        
        # Create and fit power transformer
        transformer = PowerTransformer(
            method=method,
            standardize=standardize
        )
        
        y_transformed = transformer.fit_transform(y_reshaped).flatten()
        
        return y_transformed
