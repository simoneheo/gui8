import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def resample(y, fs, target_fs=10.0, method='linear'):
    # Input validation
    if len(y) == 0:
        raise ValueError("Input signal is empty")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    if target_fs <= 0:
        raise ValueError(f"Target sampling frequency must be positive, got {target_fs}")
    
    # Check for NaN values
    if np.any(np.isnan(y)):
        raise ValueError("Input signal contains NaN values")
    if np.any(np.isinf(y)):
        raise ValueError("Input signal contains infinite values")
    
    try:
        from scipy.interpolate import interp1d
        duration = len(y) / fs
        
        if duration <= 0:
            raise ValueError(f"Invalid signal duration: {duration}")
        
        t_old = np.linspace(0, duration, len(y))
        target_samples = int(duration * target_fs)
        
        if target_samples <= 0:
            raise ValueError(f"Target sample count must be positive, got {target_samples}")
        
        t_new = np.linspace(0, duration, target_samples)
        
        # Create interpolator based on method
        if method == "linear":
            interpolator = interp1d(t_old, y, kind="linear", fill_value="extrapolate")
        elif method == "spline":
            if len(y) < 3:
                raise ValueError("Spline interpolation requires at least 3 data points")
            interpolator = interp1d(t_old, y, kind="quadratic", fill_value="extrapolate")
        elif method == "nearest":
            interpolator = interp1d(t_old, y, kind="nearest", fill_value="extrapolate")
        else:
            raise ValueError(f"Unknown interpolation method: {method}. Use 'linear', 'spline', or 'nearest'")
        
        result = interpolator(t_new)
        
        # Validate result
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("Resampling resulted in invalid values")
        
        return result
        
    except ImportError:
        raise ValueError("SciPy is required for resampling")
    except Exception as e:
        raise ValueError(f"Resampling failed: {str(e)}")

@register_step
class resample_step(BaseStep):
    name = "resample"
    category = "General"
    description = "Resample the signal to a target sampling rate."
    tags = ["time-series"]
    params = [
        {
            'name': 'target_fs', 
            'type': 'float', 
            'default': '10.0', 
            'help': 'Target sampling frequency (Hz). Smart default is calculated based on the original sampling rate: high-frequency signals (>1000 Hz) suggest 10x-20x downsampling, medium-frequency (100-1000 Hz) suggest 4x-10x downsampling, low-frequency (<100 Hz) suggest 2x-4x downsampling, rounded to convenient values.'
        }, 
        {
            'name': 'method', 
            'type': 'str', 
            'default': 'linear', 
            'options': ['linear', 'spline', 'nearest'], 
            'help': 'Interpolation method: linear (fast, good for most signals), spline (smooth, good for curves), nearest (preserves exact values, good for step functions)'
        }
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        try:
            parsed = {}
            for param in cls.params:
                if param["name"] == "fs": continue
                value = user_input.get(param["name"], param.get("default"))
                parsed[param["name"]] = float(value) if param["type"] == "float" else value
            
            # Parameter validation
            target_fs = parsed.get("target_fs")
            method = parsed.get("method", "linear")
            
            if target_fs is not None:
                if target_fs <= 0:
                    raise ValueError(f"Target sampling frequency must be positive, got {target_fs}")
            
            valid_methods = ['linear', 'spline', 'nearest']
            if method not in valid_methods:
                raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
                
            return parsed
        except ValueError as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse input parameters: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Input validation
        if channel is None:
            raise ValueError("Channel is None")
        if channel.ydata is None or len(channel.ydata) == 0:
            raise ValueError("Channel has no data")
        if channel.xdata is None or len(channel.xdata) == 0:
            raise ValueError("Channel has no x-axis data")
        if len(channel.xdata) != len(channel.ydata):
            raise ValueError("X and Y data lengths don't match")
        
        try:
            params = cls._inject_fs_if_needed(channel, params, resample)
            y_new = resample(channel.ydata, **params)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
        except Exception as e:
            raise ValueError(f"Resampling step failed: {str(e)}")
