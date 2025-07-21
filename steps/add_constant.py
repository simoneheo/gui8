import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class add_constant_step(BaseStep):
    name = "add_constant"
    category = "Arithmetic"
    description = "Add a constant value to all signal values for baseline adjustment"
    tags = ["time-series", "arithmetic", "offset", "bias", "constant", "add"]
    params = [
        {
            "name": "constant", 
            "type": "float", 
            "default": "0.0", 
            "help": "Constant value to add to all signal values. Positive values shift up, negative values shift down."
        }
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        cls.validate_numeric_parameter("constant", params.get("constant"), 
                                     min_val=-1000, max_val=1000, 
                                     allow_nan=False, allow_inf=False)

    def script(self, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
    
        constant = params["constant"]
        y_new = y + constant
        return y_new