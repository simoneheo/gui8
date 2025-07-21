import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class abs_val_step(BaseStep):
    name = "abs_val"
    category = "Arithmetic"
    description = """Apply absolute value transformation to convert all signal values to their absolute magnitude"""
    tags = ["rectification", "absolute", "magnitude"]
    params = [
        # No parameters needed for absolute value transformation
    ]
   
    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # No parameters to validate for absolute value transformation
        pass
  
    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:

        y_new = np.abs(y)        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_new
            }
        ]
