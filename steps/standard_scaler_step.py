import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class standard_scaler_step(BaseStep):
    name = "standard_scaler"
    category = "scikit-learn"
    description = """Scale signal to zero mean and unit variance using StandardScaler from scikit-learn.

This step standardizes the signal by:
• Centering the data by subtracting the mean
• Scaling to unit variance by dividing by standard deviation

Useful for:
• Normalizing data for machine learning algorithms
• Making features comparable across different scales
• Preparing data for algorithms sensitive to feature scales
• Standard preprocessing for many ML pipelines"""
    tags = ["scaling", "normalization", "standardization", "scikit-learn", "z-score"]
    params = [
        {
            "name": "with_mean",
            "type": "bool",
            "default": True,
            "help": "Whether to center the data before scaling"
        },
        {
            "name": "with_std",
            "type": "bool",
            "default": True,
            "help": "Whether to scale the data to unit variance"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Standard scaling to zero mean and unit variance (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for standard scaling (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal contains constant values (cannot scale)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        with_mean = params.get("with_mean")
        with_std = params.get("with_std")
        
        if not isinstance(with_mean, bool):
            raise ValueError("with_mean must be a boolean")
        if not isinstance(with_std, bool):
            raise ValueError("with_std must be a boolean")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_scaled: np.ndarray) -> None:
        """Validate output data"""
        if y_scaled.size == 0:
            raise ValueError("Standard scaling produced empty output")
        if np.any(np.isnan(y_scaled)) and not np.any(np.isnan(y_original)):
            raise ValueError("Standard scaling produced unexpected NaN values")

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
        """Apply standard scaling to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_scaled = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_scaled)
            
            # Create scaled channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_scaled,
                params=params,
                suffix="StandardScaled"
            )
            
            # Set channel properties
            new_channel.tags = ["scaled", "standard-scaled"]
            new_channel.legend_label = f"{channel.legend_label} (Standard Scaled)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Standard scaling failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for standard scaling"""
        from sklearn.preprocessing import StandardScaler
        
        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)
        
        # Create and fit standard scaler
        scaler = StandardScaler(
            with_mean=with_mean,
            with_std=with_std
        )
        
        # Reshape for sklearn and transform
        y_reshaped = y.reshape(-1, 1)
        y_scaled = scaler.fit_transform(y_reshaped).flatten()
        
        return y_scaled