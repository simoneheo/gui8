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
    tags = ["time-series", "custom", "user-defined", "script", "python", "flexible"]
    params = [
        {
            "name": "function", 
            "type": "multiline", 
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

Multi-line example:
# Initialize output
y_new = np.zeros_like(y)

# Custom processing loop
for i in range(len(y)):
    if y[i] > 0:
        y_new[i] = y[i] * 2
    else:
        y_new[i] = 0
        
# Apply smoothing
y_new = np.convolve(y_new, np.ones(3)/3, mode='same')

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
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        BaseStep.validate_signal_data(None, y)  # Use base validation

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate custom function parameters"""
        function_code = params.get("function")
        
        if not function_code or not isinstance(function_code, str):
            raise ValueError("Custom function code must be a non-empty string")
        
        if len(function_code.strip()) == 0:
            raise ValueError("Custom function code cannot be empty")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data from custom function"""
        if y_new is None:
            raise ValueError("Custom function must assign result to 'y_new' variable")
        
        y_new = np.asarray(y_new)
        
        if len(y_new) == 0:
            raise ValueError("Custom function produced empty output")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            value = user_input.get(name, param.get("default"))
            
            # For multiline text, preserve whitespace and newlines
            if param["type"] == "multiline":
                parsed[name] = str(value).strip()
            else:
                parsed[name] = value
                
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the custom processing step to a channel"""
        try:
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs_median', None) or 1.0
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Get processed data from script method
            y_new, x_new = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            # Create descriptive label
            function_preview = params['function'].split('\n')[0][:50]
            if len(params['function'].split('\n')[0]) > 50:
                function_preview += "..."
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Custom processing failed: {str(e)}")

    @classmethod
    def _get_example_function_code(cls) -> str:
        """Get example function code for users to modify"""
        return """
# Custom function example - modify this code to process your signal
# Available variables: x (time), y (signal), fs (sampling frequency)
# Always assign result to y_new

# Example 1: Simple scaling
y_new = y * 2

# Example 2: Remove DC offset
# y_new = y - np.mean(y)

# Example 3: Apply moving average smoothing
# window_size = 5
# y_new = np.convolve(y, np.ones(window_size)/window_size, mode='same')

# Example 4: Apply Butterworth lowpass filter
# from scipy.signal import butter, filtfilt
# cutoff = 10  # Hz
# nyq = fs / 2
# normal_cutoff = cutoff / nyq
# b, a = butter(4, normal_cutoff, btype='low', analog=False)
# y_new = filtfilt(b, a, y)

# Example 5: Calculate envelope using Hilbert transform
# from scipy.signal import hilbert
# y_new = np.abs(hilbert(y))

# Example 6: Apply thresholding
# threshold = np.percentile(y, 90)
# y_new = np.where(y > threshold, y, 0)

# Example 7: Calculate first derivative
# y_new = np.diff(y)
# Note: This changes signal length, time array will be adjusted automatically

# Example 8: Apply Savitzky-Golay smoothing
# from scipy.signal import savgol_filter
# y_new = savgol_filter(y, window_length=11, polyorder=3)

# Example 9: Normalize signal to [0, 1] range
# y_new = (y - np.min(y)) / (np.max(y) - np.min(y))

# Example 10: Apply exponential smoothing
# alpha = 0.1
# y_new = np.zeros_like(y)
# y_new[0] = y[0]
# for i in range(1, len(y)):
#     y_new[i] = alpha * y[i] + (1 - alpha) * y_new[i-1]
"""

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core processing logic for custom function execution"""
        # Get function code - use user's code or fall back to example
        function_code = params.get("function", "").strip()
        if not function_code:
            function_code = cls._get_example_function_code()
        
        # Prepare the execution environment
        exec_globals = {
            'np': np,
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
            'y_new': None
        }
        
        try:
            # Execute the custom function
            exec(function_code, exec_globals)
            
            # Get the result
            y_new = exec_globals.get('y_new')
            
            if y_new is None:
                # Fallback: return original data
                return y, x
            
            # Convert to numpy array
            y_new = np.asarray(y_new)
            
            # Handle different output lengths
            if len(y_new) == len(x):
                x_new = x
            elif len(y_new) == len(x) - 1:
                x_new = x[:-1]
            elif len(y_new) == len(x) + 1:
                if len(x) > 1:
                    dx = x[1] - x[0]
                    x_new = np.concatenate([[x[0] - dx], x])
                else:
                    x_new = np.arange(len(y_new))
            else:
                # Different length - create new x array
                if len(x) > 1:
                    x_new = np.linspace(x[0], x[-1], len(y_new))
                else:
                    x_new = np.arange(len(y_new))
            
            return y_new, x_new
            
        except Exception:
            # Fallback: return original data if custom function fails
            return y, x 