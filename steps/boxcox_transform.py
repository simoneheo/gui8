import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class boxcox_transform_step(BaseStep):
    name = "boxcox transform"
    category = "Transform"
    description = """Apply Box-Cox transformation to make data more normally distributed.
Useful for stabilizing variance and normalizing skewed distributions."""
    tags = ["boxcox", "transform", "normalization", "skewness", "scipy"]
    params = [
        {"name": "lambda", "type": "float", "default": "0.5", "help": "Lambda parameter for Box-Cox transform"},
        {"name": "shift", "type": "float", "default": "0.0", "help": "Shift to add before transformation"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        lambda_param = cls.validate_numeric_parameter("lambda", params.get("lambda"))
        shift = cls.validate_numeric_parameter("shift", params.get("shift"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from scipy.stats import boxcox
        
        lambda_param = params["lambda"]
        shift = params["shift"]
        
        # Apply shift to ensure positive values
        y_shifted = y + shift
        
        # Check if all values are positive (required for Box-Cox)
        if np.any(y_shifted <= 0):
            # Calculate suggested shift based on minimum value
            min_val = np.min(y)
            suggested_shift = abs(min_val) + 1e-6  # Add small epsilon for numerical stability
            
            raise ValueError(
                f"All values must be positive after shift for Box-Cox transform. "
                f"Current shift: {shift}, minimum value: {min_val}. "
                f"Suggested shift: {suggested_shift:.6f} (or larger)"
            )
        
        # Apply Box-Cox transformation
        if lambda_param == 0:
            # Log transform case
            y_transformed = np.log(y_shifted)
        else:
            # General Box-Cox case
            y_transformed = (y_shifted**lambda_param - 1) / lambda_param
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_transformed
            }
        ]
