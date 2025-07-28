import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class rank_transform_step(BaseStep):
    name = "rank transform"
    category = "Transform"
    description = """Transform signal values to their ranks.
Converts values to their position in the sorted array."""
    tags = ["time-series", "rank", "transform", "non-parametric", "order-statistics"]
    params = [
        {"name": "method", "type": "str", "default": "average", "options": ["average", "min", "max", "dense"], "help": "Ranking method for ties"},
        {"name": "ascending", "type": "bool", "default": "True", "help": "Sort in ascending order (True) or descending (False)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        method = cls.validate_string_parameter("method", params.get("method"), 
                                              valid_options=["average", "min", "max", "dense"])
        ascending = params.get("ascending", True)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        method = params["method"]
        ascending = params["ascending"]
        
        # Check if signal has enough unique values
        unique_vals = np.unique(y)
        if len(unique_vals) < 2:
            raise ValueError("Signal must have at least 2 unique values for ranking")
        
        # Apply rank transformation
        if method == "average":
            # Average rank for ties
            from scipy.stats import rankdata
            y_ranked = rankdata(y, method='average')
            if not ascending:
                y_ranked = len(y) + 1 - y_ranked
                
        elif method == "min":
            # Minimum rank for ties
            from scipy.stats import rankdata
            y_ranked = rankdata(y, method='min')
            if not ascending:
                y_ranked = len(y) + 1 - y_ranked
                
        elif method == "max":
            # Maximum rank for ties
            from scipy.stats import rankdata
            y_ranked = rankdata(y, method='max')
            if not ascending:
                y_ranked = len(y) + 1 - y_ranked
                
        elif method == "dense":
            # Dense ranking (no gaps)
            from scipy.stats import rankdata
            y_ranked = rankdata(y, method='dense')
            if not ascending:
                y_ranked = len(np.unique(y)) + 1 - y_ranked
                
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_ranked
            }
        ]
