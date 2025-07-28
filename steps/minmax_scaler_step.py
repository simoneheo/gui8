import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class minmax_scaler_step(BaseStep):
    name = "minmax scaler"
    category = "ML"
    description = """Rescale signal to the [0, 1] range using MinMaxScaler."""
    tags = ["time-series", "scaling", "normalization", "minmax"]
    params = []

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from sklearn.preprocessing import MinMaxScaler
        
        # Reshape for sklearn
        y_reshaped = y.reshape(-1, 1)
        
        # Apply MinMaxScaler
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y_reshaped).flatten()
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_scaled
            }
        ]