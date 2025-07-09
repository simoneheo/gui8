import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class robust_scaler_step(BaseStep):
    name = "robust_scaler"
    category = "scikit-learn"
    description = """Rescale signal using median and IQR (robust to outliers).

This step scales the signal using statistics that are robust to outliers:
• Centers the data by subtracting the median
• Scales by the interquartile range (IQR)

Useful for:
• Handling signals with outliers
• Robust normalization
• Preprocessing for machine learning algorithms
• Maintaining signal characteristics in presence of extreme values"""
    tags = ["scaling", "robust", "iqr", "scikit-learn", "normalization"]
    params = [
        {
            "name": "quantile_range",
            "type": "str",
            "default": "25.0,75.0",
            "help": "Quantile range for IQR calculation (format: 'q1,q3')"
        },
        {
            "name": "with_centering",
            "type": "bool",
            "default": True,
            "help": "Whether to center the data before scaling"
        },
        {
            "name": "with_scaling",
            "type": "bool",
            "default": True,
            "help": "Whether to scale the data to unit IQR"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Robust scaling using median and IQR (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for robust scaling (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        quantile_range = params.get("quantile_range")
        with_centering = params.get("with_centering")
        with_scaling = params.get("with_scaling")
        
        # Parse quantile range
        try:
            q1, q3 = map(float, quantile_range.split(','))
            if not (0 <= q1 < q3 <= 100):
                raise ValueError("Quantile range must be 0 <= q1 < q3 <= 100")
        except (ValueError, AttributeError):
            raise ValueError("quantile_range must be in format 'q1,q3' (e.g., '25.0,75.0')")
        
        if not isinstance(with_centering, bool):
            raise ValueError("with_centering must be a boolean")
        if not isinstance(with_scaling, bool):
            raise ValueError("with_scaling must be a boolean")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_scaled: np.ndarray) -> None:
        """Validate output data"""
        if y_scaled.size == 0:
            raise ValueError("Robust scaling produced empty output")
        if np.any(np.isnan(y_scaled)) and not np.any(np.isnan(y_original)):
            raise ValueError("Robust scaling produced unexpected NaN values")

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
        """Apply robust scaling to the channel data."""
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
                suffix="RobustScaled"
            )
            
            # Set channel properties
            new_channel.tags = ["scaled", "robust-scaled"]
            new_channel.legend_label = f"{channel.legend_label} (Robust Scaled)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Robust scaling failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for robust scaling"""

        from sklearn.preprocessing import RobustScaler
        
        quantile_range = params.get("quantile_range", "25.0,75.0")
        with_centering = params.get("with_centering", True)
        with_scaling = params.get("with_scaling", True)
        
        # Parse quantile range
        q1, q3 = map(float, quantile_range.split(','))
        quantile_range = (q1, q3)
        
        # Create and fit robust scaler
        scaler = RobustScaler(
            quantile_range=quantile_range,
            with_centering=with_centering,
            with_scaling=with_scaling
        )
        
        # Reshape for sklearn and transform
        y_reshaped = y.reshape(-1, 1)
        y_scaled = scaler.fit_transform(y_reshaped).flatten()
        
        return y_scaled