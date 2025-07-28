import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class impute_missing_step(BaseStep):
    name = "impute missing"
    category = "Transform"
    description = """Fill missing values (NaN) in the signal using various interpolation methods.
Useful for handling gaps and missing data points."""
    tags = ["time-series", "imputation", "missing", "interpolation", "nan", "fill"]
    params = [
        {"name": "method", "type": "str", "default": "linear", "options": ["linear", "cubic", "nearest", "mean"], "help": "Interpolation method for filling missing values"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        method = cls.validate_string_parameter("method", params.get("method"), 
                                               valid_options=["linear", "cubic", "nearest", "mean"])

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        method = params["method"]
        
        # Check if there are any missing values
        if not np.any(np.isnan(y)):
            # No missing values, return original signal
            return [
                {
                    'tags': ['time-series'],
                    'x': x,
                    'y': y
                }
            ]
        
        # Create mask for non-missing values
        valid_mask = ~np.isnan(y)
        
        if method == "mean":
            # Fill with mean of valid values
            mean_val = np.nanmean(y)
            y_imputed = y.copy()
            y_imputed[np.isnan(y)] = mean_val
        else:
            # Interpolation methods
            from scipy.interpolate import interp1d
            
            # Get valid indices and values
            valid_indices = np.where(valid_mask)[0]
            valid_values = y[valid_mask]
            
            if len(valid_indices) < 2:
                raise ValueError("Need at least 2 valid points for interpolation")
            
            # Create interpolation function
            if method == "linear":
                interp_func = interp1d(valid_indices, valid_values, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
            elif method == "cubic":
                if len(valid_indices) < 4:
                    # Fall back to linear if not enough points for cubic
                    interp_func = interp1d(valid_indices, valid_values, kind='linear', 
                                          bounds_error=False, fill_value='extrapolate')
                else:
                    interp_func = interp1d(valid_indices, valid_values, kind='cubic', 
                                          bounds_error=False, fill_value='extrapolate')
            elif method == "nearest":
                interp_func = interp1d(valid_indices, valid_values, kind='nearest', 
                                      bounds_error=False, fill_value='extrapolate')
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            
            # Interpolate all indices
            all_indices = np.arange(len(y))
            y_imputed = interp_func(all_indices)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_imputed
            }
        ]
