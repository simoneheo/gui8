import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class quantize_step(BaseStep):
    name = "quantize"
    category = "Transform"
    description = """Quantize signal to discrete levels.
Reduces signal precision by mapping values to nearest quantization levels."""
    tags = ["time-series", "quantize", "discrete", "levels", "precision", "mapping"]
    params = [
        {"name": "levels", "type": "int", "default": "8", "help": "Number of quantization levels"},
        {"name": "method", "type": "str", "default": "uniform", "options": ["uniform", "percentile"], "help": "Quantization method"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        levels = cls.validate_integer_parameter("levels", params.get("levels"), min_val=2, max_val=1000)
        method = cls.validate_string_parameter("method", params.get("method"), 
                                              valid_options=["uniform", "percentile"])

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        levels = params["levels"]
        method = params["method"]
        
        # Check if signal has enough unique values
        unique_vals = np.unique(y)
        if len(unique_vals) < 2:
            raise ValueError("Signal must have at least 2 unique values for quantization")
        
        if method == "uniform":
            # Uniform quantization across the full range
            y_min, y_max = np.min(y), np.max(y)
            if y_max == y_min:
                # All values are the same, return original
                y_quantized = y.copy()
            else:
                # Create uniform quantization levels
                quant_levels = np.linspace(y_min, y_max, levels)
                # Find nearest quantization level for each value
                y_quantized = np.zeros_like(y)
                for i, val in enumerate(y):
                    distances = np.abs(quant_levels - val)
                    nearest_idx = np.argmin(distances)
                    y_quantized[i] = quant_levels[nearest_idx]
        
        elif method == "percentile":
            # Quantization based on percentiles
            percentiles = np.linspace(0, 100, levels)
            quant_levels = np.percentile(y, percentiles)
            # Find nearest quantization level for each value
            y_quantized = np.zeros_like(y)
            for i, val in enumerate(y):
                distances = np.abs(quant_levels - val)
                nearest_idx = np.argmin(distances)
                y_quantized[i] = quant_levels[nearest_idx]
        
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_quantized
            }
        ]
