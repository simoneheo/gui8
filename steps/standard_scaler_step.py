import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class standard_scaler_step(BaseStep):
    name = "standard scaler"
    category = "Transform"
    description = """Scale signal to zero mean and unit variance using StandardScaler from scikit-learn."""
    tags = ["time-series", "scaling", "normalization", "standardization", "z-score"]
    params = [
        {"name": "with_mean", "type": "bool", "default": "True", "help": "Whether to center the data before scaling"},
        {"name": "with_std", "type": "bool", "default": "True", "help": "Whether to scale the data to unit variance"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)
        
        if not isinstance(with_mean, bool):
            raise ValueError("with_mean must be a boolean")
        if not isinstance(with_std, bool):
            raise ValueError("with_std must be a boolean")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from sklearn.preprocessing import StandardScaler
        
        with_mean = params.get("with_mean", True)
        with_std = params.get("with_std", True)
        
        # Create and fit standard scaler
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        
        # Reshape for sklearn and transform
        y_reshaped = y.reshape(-1, 1)
        y_scaled = scaler.fit_transform(y_reshaped).flatten()
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_scaled
            }
        ]