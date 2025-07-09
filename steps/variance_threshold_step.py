
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class variance_threshold_step(BaseStep):
    name = "variance_threshold"
    category = "scikit-learn"
    description = """Remove low-variance features using VarianceThreshold.

This step removes features with variance below a specified threshold.
Useful for:
• Removing constant or near-constant features
• Feature selection based on variance
• Data preprocessing for machine learning
• Reducing dimensionality by removing uninformative features"""
    tags = ["feature-selection", "variance", "scikit-learn", "unsupervised"]
    params = [
        {
            "name": "threshold",
            "type": "float",
            "default": 0.0,
            "help": "Minimum variance threshold (features with lower variance will be removed)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Remove low-variance features (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for variance threshold (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if y.ndim == 1:
            raise ValueError("VarianceThreshold requires 2D input (multi-feature signal)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        threshold = params.get("threshold")
        
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_selected: np.ndarray) -> None:
        """Validate output data"""
        if y_selected.size == 0:
            raise ValueError("Variance threshold produced empty output")
        if len(y_selected) != len(y_original):
            raise ValueError("Selected features length doesn't match input signal length")

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
        """Apply variance threshold feature selection to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_selected = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_selected)
            
            # Create feature selection channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_selected,
                params=params,
                suffix="VarThresh"
            )
            
            # Set channel properties
            new_channel.tags = ["feature-selection", "variance", "selected"]
            new_channel.legend_label = f"{channel.legend_label} (Variance Threshold)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Variance threshold feature selection failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for variance threshold feature selection"""
        from sklearn.feature_selection import VarianceThreshold
        
        threshold = params.get("threshold", 0.0)
        
        # Create and fit variance threshold selector
        selector = VarianceThreshold(threshold=threshold)
        
        y_selected = selector.fit_transform(y)
        
        return y_selected
