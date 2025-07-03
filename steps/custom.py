import numpy as np
import scipy
from scipy import signal, stats, optimize, integrate, interpolate
import math
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class custom_step(BaseStep):
    name = "custom"
    category = "Custom"
    description = "Execute custom Python function on channel data"
    tags = ["time-series"]
    params = [
        {
            "name": "function", 
            "type": "string", 
            "default": "y_new = y * 2", 
            "help": """Custom Python function to apply to the data.

Note: Always assign result to 'y_new' variable.

Available variables:
• x - time/index array (1D numpy array)
• y - signal values (1D numpy array) 
• fs - sampling frequency (float)
• y_new - output signal (must be assigned)

Available packages:
• numpy as np - numerical computing
• scipy - scientific computing library
• scipy.signal - signal processing
• scipy.stats - statistical functions
• scipy.optimize - optimization algorithms
• scipy.integrate - numerical integration
• scipy.interpolate - interpolation functions
• math - basic math functions

Example functions:
• y_new = y * 2  # Double the signal
• y_new = np.sqrt(np.abs(y))  # Square root of absolute values
• y_new = y - np.mean(y)  # Remove DC offset
• y_new = scipy.signal.detrend(y)  # Remove linear trend
• y_new = np.diff(y)  # First derivative
• y_new = np.cumsum(y)  # Cumulative sum
• y_new = scipy.signal.hilbert(y).real  # Hilbert transform (real part)

"""
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        info_text = f"""{cls.description}

IMPORTANT: Always assign your result to the 'y_new' variable.

AVAILABLE VARIABLES:
• x - time/index array (1D numpy array)
• y - signal values (1D numpy array) 
• fs - sampling frequency (float)
• y_new - output signal (must be assigned)

AVAILABLE PACKAGES:
• numpy as np - numerical computing
• scipy - scientific computing library
• scipy.signal - signal processing
• scipy.stats - statistical functions
• scipy.optimize - optimization algorithms
• scipy.integrate - numerical integration
• scipy.interpolate - interpolation functions
• math - basic math functions

EXAMPLE FUNCTIONS:
• y_new = y * 2                           # Double the signal
• y_new = np.sqrt(np.abs(y))              # Square root of absolute values
• y_new = y - np.mean(y)                  # Remove DC offset
• y_new = scipy.signal.detrend(y)         # Remove linear trend
• y_new = np.diff(y)                      # First derivative
• y_new = np.cumsum(y)                    # Cumulative sum
• y_new = scipy.signal.hilbert(y).real    # Hilbert transform (real part)
• y_new = np.convolve(y, np.ones(5)/5, mode='same')  # 5-point moving average
• y_new = np.fft.fft(y)                   # Fast Fourier Transform
• y_new = scipy.signal.butter(4, 0.1, output='sos')  # Design filter coefficients

"""
        
        return {"info": info_text, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            value = user_input.get(param["name"], param["default"])
            parsed[param["name"]] = value
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x = channel.xdata
        y = channel.ydata
        fs = channel.fs_median if hasattr(channel, 'fs_median') and channel.fs_median else 1.0
        
        # Get the custom function code
        function_code = params['function']
        
        # Prepare the execution environment with available packages and variables
        exec_globals = {
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'signal': signal,
            'stats': stats,
            'optimize': optimize,
            'integrate': integrate,
            'interpolate': interpolate,
            'math': math,
            'x': x,
            'y': y,
            'fs': fs,
            'y_new': None  # Will be set by user function
        }
        
        try:
            # Execute the custom function
            exec(function_code, exec_globals)
            
            # Get the result
            y_new = exec_globals.get('y_new')
            
            if y_new is None:
                raise ValueError("Custom function must assign result to 'y_new' variable")
            
            # Convert to numpy array if needed
            y_new = np.asarray(y_new)
            
            # Handle different output lengths
            if len(y_new) == len(x):
                # Same length - use original x
                x_new = x
            elif len(y_new) == len(x) - 1:
                # One less (e.g., from np.diff) - adjust x
                x_new = x[:-1]
            elif len(y_new) == len(x) + 1:
                # One more (e.g., from cumsum with initial value) - extend x
                if len(x) > 1:
                    dx = x[1] - x[0]
                    x_new = np.concatenate([[x[0] - dx], x])
                else:
                    x_new = np.arange(len(y_new))
            else:
                # Different length - create new x array
                if len(x) > 1:
                    # Scale to match original time range
                    x_new = np.linspace(x[0], x[-1], len(y_new))
                else:
                    x_new = np.arange(len(y_new))
            
            # Create descriptive label
            function_preview = function_code.split('\n')[0][:50]
            if len(function_code.split('\n')[0]) > 50:
                function_preview += "..."
            
            return Channel.from_parent(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                legend_label=f"{channel.legend_label} - Custom",
                description=f"Custom function: {function_preview}",
                tags=cls.tags,
                params=params
            )
            
        except Exception as e:
            raise ValueError(f"Error executing custom function: {str(e)}") 