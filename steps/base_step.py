# steps/base_step.py

from abc import ABC, abstractmethod
from channel import Channel  # Import the Channel class
import numpy as np
from typing import Optional, Dict, Any, List


class BaseStep(ABC):
    """
    Abstract base class for all processing steps.
    Each step must implement `apply()` and provide metadata like `name`, `category`, `description`, and `params`.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs  # Store parameters passed by user

    @abstractmethod
    def apply(self, channel: Channel) -> Channel:
        """
        Apply this step to the given channel. Must return a new ChannelInfo.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Return all metadata about the step.
        Useful for GUI display or parameter prompting.
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "tags": self.tags,
            "params": self.params
        }

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            name = param["name"]
            if name == "fs":
                # Skip parsing fs - it will be injected from parent channel
                continue
            value = user_input.get(name, param.get("default"))
            
            # Handle type conversion based on parameter type
            param_type = param.get("type", "str")
            try:
                if param_type == "float":
                    parsed[name] = float(value)
                elif param_type == "int":
                    parsed[name] = int(value)
                elif param_type in ["bool", "boolean"]:
                    if isinstance(value, bool):
                        parsed[name] = value
                    elif isinstance(value, str):
                        parsed[name] = value.lower() in ['true', '1', 'yes', 'on']
                    else:
                        parsed[name] = bool(value)
                else:
                    parsed[name] = value
            except (ValueError, TypeError):
                # If conversion fails, use default or keep as string
                parsed[name] = param.get("default", value)
        return parsed

    @classmethod
    def _inject_fs_if_needed(cls, channel: Channel, params: dict, func) -> dict:
        """
        Automatically inject fs from parent channel if the processing function requires it.
        """
        if "fs" in func.__code__.co_varnames:
            fs = cls._get_channel_fs(channel)
            if fs is not None:
                params["fs"] = fs
                print(f"[{cls.name}Step] Injected fs={fs:.2f} from parent channel")
            else:
                print(f"[{cls.name}Step] Warning: Could not extract sampling frequency from channel")
        return params

    @classmethod
    def _get_channel_fs(cls, channel: Channel) -> Optional[float]:
        """
        Extract sampling frequency from a channel using the correct attribute hierarchy.
        
        Args:
            channel: Channel object to extract fs from
            
        Returns:
            float: Sampling frequency in Hz, or None if not available
        """
        # Priority 1: Use fs_median (primary source in Channel class)
        fs = getattr(channel, 'fs_median', None)
        if fs is not None and fs > 0:
            return fs
        
        # Priority 2: Fallback to fs attribute (backwards compatibility)
        fs = getattr(channel, 'fs', None)
        if fs is not None and fs > 0:
            return fs
        
        # Priority 3: Try to calculate from sampling_stats
        if hasattr(channel, 'sampling_stats') and channel.sampling_stats:
            fs = getattr(channel.sampling_stats, 'median_fs', None)
            if fs is not None and fs > 0:
                return fs
        
        # Priority 4: Try to calculate from time data if available
        if hasattr(channel, 'xdata') and channel.xdata is not None and len(channel.xdata) > 1:
            try:
                time_diffs = np.diff(channel.xdata)
                valid_diffs = time_diffs[time_diffs > 0]
                if len(valid_diffs) > 0:
                    median_dt = np.median(valid_diffs)
                    if median_dt > 0:
                        return 1.0 / median_dt
            except Exception as e:
                print(f"[{cls.name}Step] Error calculating fs from time data: {e}")
        
        return None

    @classmethod
    def create_new_channel(cls, parent: Channel, xdata: np.ndarray, ydata: np.ndarray, params: dict, suffix: str = None) -> Channel:
        """
        Helper method to create a new channel with consistent parameter handling.
        
        Args:
            parent: Parent channel to inherit properties from
            xdata: Time/index data for the new channel
            ydata: Signal data for the new channel
            params: Parameters used for processing
            suffix: Optional suffix for the channel name (defaults to step name)
        """
        # Use provided suffix or default to step name
        name_suffix = suffix if suffix else cls.name
        
        return Channel.from_parent(
            parent=parent,
            xdata=xdata,
            ydata=ydata,
            legend_label=f"{parent.legend_label} - {name_suffix}",
            description=cls.description,
            tags=cls.tags,
            params=params  # Pass the parameters to the new channel
        )

    # ============================================================================
    # VALIDATION HELPER METHODS
    # ============================================================================

    @classmethod
    def validate_channel_input(cls, channel) -> None:
        """
        Validate that the input channel is valid for processing.
        
        Args:
            channel: Input channel object to validate
            
        Raises:
            ValueError: If channel is invalid
        """
        if channel is None:
            raise ValueError("Input channel cannot be None")
        
        if not hasattr(channel, 'ydata') or channel.ydata is None:
            raise ValueError("Channel must have valid signal data (ydata)")
        
        if not hasattr(channel, 'xdata') or channel.xdata is None:
            raise ValueError("Channel must have valid time data (xdata)")

    @classmethod
    def validate_signal_data(cls, xdata, ydata) -> None:
        """
        Validate signal data arrays.
        
        Args:
            xdata: Time/index array
            ydata: Signal values array
            
        Raises:
            ValueError: If data is invalid
        """
        # Check for empty data
        if len(ydata) == 0:
            raise ValueError("Signal data cannot be empty")
        
        if len(xdata) == 0:
            raise ValueError("Time data cannot be empty")
        
        # Check length mismatch
        if len(xdata) != len(ydata):
            raise ValueError(f"Time and signal data length mismatch: {len(xdata)} vs {len(ydata)}")
        
        # Check for all NaN values
        if np.all(np.isnan(ydata)):
            raise ValueError("Signal contains only NaN values")
        
        # Check for all infinite values
        if np.all(np.isinf(ydata)):
            raise ValueError("Signal contains only infinite values")
        
        # Check time data validity
        if np.any(np.isnan(xdata)):
            raise ValueError("Time data contains NaN values")
        
        if np.any(np.isinf(xdata)):
            raise ValueError("Time data contains infinite values")

    @classmethod
    def validate_numeric_parameter(cls, param_name: str, value: Any, 
                                 min_val: Optional[float] = None,
                                 max_val: Optional[float] = None,
                                 allow_nan: bool = False,
                                 allow_inf: bool = False) -> float:
        """
        Validate and convert a numeric parameter.
        
        Args:
            param_name: Name of the parameter for error messages
            value: Input value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            
        Returns:
            float: Validated numeric value
            
        Raises:
            ValueError: If parameter is invalid
        """
        # Handle string input
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError(f"{param_name} parameter cannot be empty")
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"{param_name} must be a valid number, got '{value}'")
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            raise ValueError(f"{param_name} must be a number, got {type(value).__name__}: {value}")
        
        # Check for NaN
        if np.isnan(value) and not allow_nan:
            raise ValueError(f"{param_name} cannot be NaN")
        
        # Check for infinity
        if np.isinf(value) and not allow_inf:
            raise ValueError(f"{param_name} cannot be infinite")
        
        # Check range
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
        
        return value

    @classmethod
    def validate_integer_parameter(cls, param_name: str, value: Any,
                                 min_val: Optional[int] = None,
                                 max_val: Optional[int] = None) -> int:
        """
        Validate and convert an integer parameter.
        
        Args:
            param_name: Name of the parameter for error messages
            value: Input value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            
        Returns:
            int: Validated integer value
            
        Raises:
            ValueError: If parameter is invalid
        """
        # Handle string input
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError(f"{param_name} parameter cannot be empty")
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"{param_name} must be a valid integer, got '{value}'")
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{param_name} must be an integer, got {value}")
            value = int(value)
        else:
            raise ValueError(f"{param_name} must be an integer, got {type(value).__name__}: {value}")
        
        # Check range
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
        
        return value

    @classmethod
    def validate_string_parameter(cls, param_name: str, value: Any,
                                valid_options: Optional[List[str]] = None,
                                allow_empty: bool = False) -> str:
        """
        Validate a string parameter.
        
        Args:
            param_name: Name of the parameter for error messages
            value: Input value to validate
            valid_options: List of valid string options
            allow_empty: Whether to allow empty strings
            
        Returns:
            str: Validated string value
            
        Raises:
            ValueError: If parameter is invalid
        """
        if not isinstance(value, str):
            raise ValueError(f"{param_name} must be a string, got {type(value).__name__}: {value}")
        
        value = value.strip()
        
        if not value and not allow_empty:
            raise ValueError(f"{param_name} parameter cannot be empty")
        
        if valid_options is not None and value not in valid_options:
            raise ValueError(f"{param_name} must be one of {valid_options}, got '{value}'")
        
        return value

    @classmethod
    def validate_output_data(cls, y_input: np.ndarray, y_output: np.ndarray,
                           allow_length_change: bool = False) -> None:
        """
        Validate output data from processing.
        
        Args:
            y_input: Original input signal
            y_output: Processed output signal
            allow_length_change: Whether to allow output length to differ from input
            
        Raises:
            ValueError: If output data is invalid
        """
        if y_output is None:
            raise ValueError("Processing produced no output")
        
        y_output = np.asarray(y_output)
        
        if len(y_output) == 0:
            raise ValueError("Processing produced empty output")
        
        if not allow_length_change and len(y_output) != len(y_input):
            raise ValueError(f"Processing changed signal length: {len(y_input)} -> {len(y_output)}")
        
        # Check for unexpected NaN values
        if np.any(np.isnan(y_output)) and not np.any(np.isnan(y_input)):
            raise ValueError("Processing produced unexpected NaN values")
        
        # Check for unexpected infinite values
        if np.any(np.isinf(y_output)) and not np.any(np.isinf(y_input)):
            raise ValueError("Processing produced unexpected infinite values")

    @classmethod
    def validate_window_parameter(cls, window: int, signal_length: int,
                                min_window: int = 3) -> int:
        """
        Validate a window size parameter.
        
        Args:
            window: Window size to validate
            signal_length: Length of the signal being processed
            min_window: Minimum allowed window size
            
        Returns:
            int: Validated window size
            
        Raises:
            ValueError: If window size is invalid
        """
        if window < min_window:
            raise ValueError(f"Window size must be >= {min_window}, got {window}")
        
        if window > signal_length:
            raise ValueError(f"Window size ({window}) cannot be larger than signal length ({signal_length})")
        
        # Warn if window is very large relative to signal
        if window > signal_length * 0.5:
            import warnings
            warnings.warn(f"Window size ({window}) is more than 50% of signal length ({signal_length})")
        
        return window

    @classmethod
    def validate_frequency_parameter(cls, frequency: float, sampling_rate: float,
                                  param_name: str = "frequency") -> float:
        """
        Validate a frequency parameter against sampling rate.
        
        Args:
            frequency: Frequency value to validate
            sampling_rate: Sampling rate of the signal
            param_name: Name of the parameter for error messages
            
        Returns:
            float: Validated frequency value
            
        Raises:
            ValueError: If frequency is invalid
        """
        if frequency <= 0:
            raise ValueError(f"{param_name} must be positive, got {frequency}")
        
        nyquist = sampling_rate / 2
        if frequency >= nyquist:
            raise ValueError(f"{param_name} ({frequency} Hz) must be less than Nyquist frequency ({nyquist:.1f} Hz)")
        
        return frequency
