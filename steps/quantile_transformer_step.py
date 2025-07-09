import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class quantile_transformer_step(BaseStep):
    name = "quantile_transformer"
    category = "scikit-learn"
    description = """Transform signal to follow a uniform or normal distribution using quantile transformation.

This step maps the signal values to a specified distribution (uniform or normal) based on quantiles.
Useful for:
• Making non-Gaussian data more Gaussian-like
• Handling outliers robustly
• Normalizing data distributions
• Preparing data for algorithms that assume normal distributions"""
    tags = ["scaling", "quantile", "distribution", "scikit-learn", "normalization"]
    params = [
        {
            "name": "output_distribution",
            "type": "str",
            "default": "uniform",
            "options": ["uniform", "normal"],
            "help": "Target distribution for the transformed data"
        },
        {
            "name": "n_quantiles",
            "type": "int",
            "default": 1000,
            "help": "Number of quantiles to compute (must be positive)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Quantile transformation (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for quantile transformation (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal contains constant values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        n_quantiles = params.get("n_quantiles")
        output_distribution = params.get("output_distribution")
        
        if n_quantiles <= 0:
            raise ValueError("n_quantiles must be positive")
        if output_distribution not in ["uniform", "normal"]:
            raise ValueError("output_distribution must be 'uniform' or 'normal'")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_transformed: np.ndarray) -> None:
        """Validate output data"""
        if y_transformed.size == 0:
            raise ValueError("Quantile transformation produced empty output")
        if np.any(np.isnan(y_transformed)) and not np.any(np.isnan(y_original)):
            raise ValueError("Quantile transformation produced unexpected NaN values")

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
        """Apply quantile transformation to the channel data."""
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
                suffix="QuantileTransformed"
            )
            
            # Set channel properties
            new_channel.tags = ["scaled", "quantile-transformed"]
            new_channel.legend_label = f"{channel.legend_label} (Quantile Transformed)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Quantile transformation failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for quantile transformation"""
        from sklearn.preprocessing import QuantileTransformer
        
        output_distribution = params.get("output_distribution", "uniform")
        n_quantiles = params.get("n_quantiles", 1000)
        
        # Ensure n_quantiles doesn't exceed signal length
        n_quantiles = min(n_quantiles, len(y))
        
        # Create and fit quantile transformer
        transformer = QuantileTransformer(
            output_distribution=output_distribution,
            n_quantiles=n_quantiles,
            random_state=42
        )
        
        # Reshape for sklearn and transform
        y_reshaped = y.reshape(-1, 1)
        y_transformed = transformer.fit_transform(y_reshaped).flatten()
        
        return y_transformed
