import numpy as np

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class robust_scaler_step(BaseStep):
    name = "robust scaler"
    category = "ML"
    description = """Rescale signal using median and IQR (robust to outliers)."""
    tags = ["scaling", "robust", "iqr", "normalization"]
    params = [
        {"name": "quantile_range", "type": "str", "default": "25.0,75.0", "help": "Quantile range for IQR calculation (format: 'q1,q3')"},
        {"name": "with_centering", "type": "bool", "default": "True", "help": "Whether to center the data before scaling"},
        {"name": "with_scaling", "type": "bool", "default": "True", "help": "Whether to scale the data to unit IQR"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        quantile_range = params.get("quantile_range", "25.0,75.0")
        with_centering = params.get("with_centering", True)
        with_scaling = params.get("with_scaling", True)
        
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
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
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
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_scaled
            }
        ]