import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class resample_step(BaseStep):
    name = "resample"
    category = "General"
    description = """Resample the signal to a target sampling rate using interpolation.
    
This step changes the sampling rate of the signal by interpolating between existing
samples to create new samples at the target rate. The signal duration remains the same,
but the number of samples changes.

• **Target sampling rate**: Desired sampling frequency in Hz
• **Interpolation method**: Method for computing intermediate values

Useful for:
• **Data reduction**: Downsample high-frequency signals to reduce storage
• **Standardization**: Convert signals to a common sampling rate
• **Processing optimization**: Adjust sampling rate for specific algorithms
• **Format conversion**: Prepare signals for systems with different requirements"""
    tags = ["time-series", "interpolation", "sampling", "resample", "frequency", "upsample", "downsample"]
    params = [
        {
            'name': 'target_fs', 
            'type': 'float', 
            'default': '10.0', 
            'help': 'Target sampling frequency (Hz, must be > 0)'
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
    def get_info(cls): 
        return f"{cls.name} — Resample signal to target sampling rate (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.any(np.isnan(y)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(y)):
            raise ValueError("Input signal contains infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        target_fs = params.get("target_fs")
        method = params.get("method", "linear")
        
        if target_fs <= 0:
            raise ValueError(f"Target sampling frequency must be positive, got {target_fs}")
        
        valid_methods = ['linear', 'spline', 'nearest']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) == 0:
            raise ValueError("Resampling produced empty output")
        if np.any(np.isnan(y_new)) or np.any(np.isinf(y_new)):
            raise ValueError("Resampling resulted in invalid values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "float":
                    parsed[name] = float(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply resampling to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel using the helper method
            fs = cls._get_channel_fs(channel)
            
            # Validate that we have a sampling frequency
            if fs is None or fs <= 0:
                raise ValueError("No valid sampling frequency available from channel. Cannot perform resampling.")
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Validate spline method requirements
            method = params.get("method", "linear")
            if method == "spline" and len(y) < 3:
                raise ValueError("Spline interpolation requires at least 3 data points")
            
            print(f"[{cls.name}] Using fs={fs:.2f} from channel for resampling")
            
            # Process the data
            y_final = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=np.linspace(x[0], x[-1], len(y_final)), 
                ydata=y_final, 
                params=params,
                suffix="Resampled"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Resampling failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """Core processing logic for resampling"""
        target_fs = params.get("target_fs")
        method = params.get("method", "linear")
        
        from scipy.interpolate import interp1d
        duration = len(y) / fs
        
        t_old = np.linspace(0, duration, len(y))
        target_samples = int(duration * target_fs)
        
        t_new = np.linspace(0, duration, target_samples)
        
        # Create interpolator based on method
        if method == "linear":
            interpolator = interp1d(t_old, y, kind="linear", fill_value="extrapolate")
        elif method == "spline":
            interpolator = interp1d(t_old, y, kind="quadratic", fill_value="extrapolate")
        elif method == "nearest":
            interpolator = interp1d(t_old, y, kind="nearest", fill_value="extrapolate")
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        y_new = interpolator(t_new)
        return y_new
