# steps/base_step.py

from abc import ABC, abstractmethod
from channel import Channel  # Import the Channel class
import numpy as np
from typing import Optional, Dict, Any, List


class BaseStep(ABC):
    """
    Abstract base class for all processing steps.
    Each step must implement `script()` and provide metadata like `name`, `category`, `description`, and `params`.
    """
    
    # Class attributes that subclasses should override
    name: str = "base_step"
    category: str = "Base"
    description: str = "Base processing step"
    tags: List[str] = []
    params: List[Dict[str, Any]] = []
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs  # Store parameters passed by user

    @abstractmethod
    def script(self, x: np.ndarray, y: np.ndarray, fs: Optional[float], params: dict) -> list:
        """
        Core processing logic that subclasses must implement.
        
        Args:
            x: Time/index data
            y: Signal data
            fs: Sampling frequency (may be None)
            params: Validated parameters
            
        Returns:
            list: List of dictionaries, each containing channel data with 'type', 'x', 'y' keys
                 Example: [{'type': 'main', 'x': x_data, 'y': y_data}]
        """
        pass

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel | list[Channel]:
        """
        Standard apply method that handles all common processing steps.
        Subclasses should not override this method.
        """
        try:
            step_name = cls.name.replace('_', ' ').title()
            print(f"[{step_name}] Starting apply with params: {params}")
            
            # 1. Validate channel input
            cls.validate_channel_input(channel)
            
            # 2. Extract data
            x = channel.xdata
            y = channel.ydata
            
            # Validate that we have valid data arrays
            if x is None or y is None:
                raise ValueError("Channel must have valid xdata and ydata")
            
            # 3. Validate input data
            cls.validate_signal_data(x, y)
            
            # 4. Get sampling frequency
            fs = cls._get_channel_fs(channel)
            
            print(f"[{step_name}] Input data shape: {y.shape if hasattr(y, 'shape') else len(y)}")
            
            # 5. Validate parameters (declarative + custom)
            try:
                validate_method = getattr(cls, 'validate_parameters', None)
                if validate_method and callable(validate_method):
                    validate_method(params)
            except AttributeError:
                pass  # No custom validation method
            
            # 6. Call core processing logic
            # Create a temporary instance to call the script method
            temp_instance = cls()
            channels_data = temp_instance.script(x, y, fs, params)
            
            # 7. Process each channel in the result
            created_channels = []
            for i, channel_info in enumerate(channels_data):
                # Validate channel structure first
                cls.validate_channel_structure(channel_info, i)
                
                tags = channel_info['tags']
                # Use the first tag as the channel type, or 'main' as default
                channel_type = tags[0] if tags else 'main'
                
                # Extract data based on channel type
                if channel_type == 'spectrogram':
                    # Spectrogram channels use 't', 'f', 'z' fields
                    x_data = channel_info['t']  # Time axis
                    y_data = channel_info['f']  # Frequency axis
                    z_data = channel_info['z']  # Spectrogram data
                else:
                    # Regular channels use 'x', 'y' fields
                    x_data = channel_info['x']
                    y_data = channel_info['y']
                    z_data = None
                
                # Validate output data for this channel based on its type
                # Allow length changes for time-series from STFT/spectrogram processing
                allow_length_change = channel_type in ['time-series', 'spectrogram', 'reduced']
                cls.validate_output_data(y, y_data, channel_type=channel_type, allow_length_change=allow_length_change)
                
                print(f"[{step_name}] Channel {i+1} ({channel_type}) data shape: {y_data.shape if hasattr(y_data, 'shape') else len(y_data)}")
                
                # 8. Generate channel suffix automatically
                suffix = cls._generate_channel_suffix(params)
                if len(channels_data) > 1:
                    suffix = f"{suffix}_{channel_type}"
                
                print(f"[{step_name}] Creating {channel_type} channel with suffix: {suffix}")
                
                # 9. Create channel based on type
                # Extract tags from channel_info if available
                channel_tags = channel_info.get('tags', None)
                
                if channel_type == 'time-series':
                    new_channel = cls.create_new_channel(
                        parent=channel, 
                        xdata=x_data, 
                        ydata=y_data, 
                        params=params,
                        suffix=suffix,
                        channel_tags=channel_tags
                    )
              
                elif channel_type == 'spectrogram':
                    # Handle spectrogram-specific properties
                    new_channel = cls.create_new_channel(
                        parent=channel, 
                        xdata=x_data,      # Time axis (t)
                        ydata=y_data,      # Frequency axis (f)
                        params=params,
                        suffix=suffix,
                        channel_tags=channel_tags
                    )
                    
                    # Add spectrogram data to metadata
                    if z_data is not None:
                        new_channel.metadata = {'Zxx': z_data}
                    else:
                        new_channel.metadata = {}
                else:
                    # Default channel creation
                    new_channel = cls.create_new_channel(
                        parent=channel, 
                        xdata=x_data, 
                        ydata=y_data, 
                        params=params,
                        suffix=suffix,
                        channel_tags=channel_tags
                    )
                
                created_channels.append(new_channel)
            
            # Configure marker properties for detection steps
            if 'marker' in cls.tags:
                for new_channel in created_channels:
                    # Set marker properties for detection steps
                    new_channel.marker = 'o'  # Circle marker
                    new_channel.style = 'none'  # No line connecting points (matplotlib expects 'none')
                    new_channel.color = '#ff4444'  # Red color for detection points
            
            # Return single channel if only one, otherwise return list
            if len(created_channels) == 1:
                return created_channels[0]
            else:
                return created_channels
            
        except Exception as e:
            step_name = cls.name.replace('_', ' ').title()
            print(f"[{step_name}] Error in apply: {e}")
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"{step_name} processing failed: {str(e)}")

    @classmethod
    def _generate_channel_suffix(cls, params: dict) -> str:
        """
        Generate an appropriate suffix for the new channel based on parameters.
        Subclasses can override this for custom naming.
        """
        # Try to generate smart suffix based on common parameter patterns
        
        # Handle constant addition/subtraction
        if 'constant' in params and params['constant'] is not None:
            constant = float(params['constant'])
            if constant >= 0:
                return f"Plus{constant:g}"
            else:
                return f"Minus{abs(constant):g}"
        
        # Handle window-based operations
        if 'window' in params and params['window'] is not None:
            window = params['window']
            # Handle both numeric and string window types
            try:
                window_int = int(window)
                if cls.name.startswith('moving_'):
                    operation = cls.name.replace('moving_', '').upper()
                    return f"{operation}{window_int}"
                else:
                    return f"Win{window_int}"
            except (ValueError, TypeError):
                # Handle string window types (like 'hann', 'hamming', etc.)
                if cls.name.startswith('moving_'):
                    operation = cls.name.replace('moving_', '').upper()
                    return f"{operation}{window.capitalize()}"
                else:
                    return f"Win{window.capitalize()}"
        
        # Handle frequency-based operations
        if 'low_freq' in params and 'high_freq' in params and params['low_freq'] is not None and params['high_freq'] is not None:
            low = float(params['low_freq'])
            high = float(params['high_freq'])
            return f"BP{low:g}-{high:g}Hz"
        elif 'frequency' in params and params['frequency'] is not None:
            freq = float(params['frequency'])
            if 'lowpass' in cls.name:
                return f"LP{freq:g}Hz"
            elif 'highpass' in cls.name:
                return f"HP{freq:g}Hz"
            else:
                return f"F{freq:g}Hz"
        
        # Handle threshold operations
        if 'threshold' in params and params['threshold'] is not None:
            thresh = float(params['threshold'])
            return f"Thresh{thresh:g}"
        
        # Handle method-based operations
        if 'method' in params and params['method'] is not None:
            method = str(params['method'])
            return f"{method.capitalize()}"
        
        # Default: use step name
        return cls.name.replace('_step', '').replace('_', ' ').title()

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
    def get_prompt(cls) -> Dict[str, Any]:
        """
        Return prompt information for GUI parameter collection.
        Standard method used by ProcessWizardManager.
        """
        return {"info": cls.description, "params": cls.params}

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
                    parsed[name] = float(value) if value is not None else None
                elif param_type == "int":
                    parsed[name] = int(value) if value is not None else None
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
    def create_new_channel(cls, parent: Channel, xdata: np.ndarray, ydata: np.ndarray, params: dict, suffix: Optional[str] = None, channel_tags: Optional[list] = None) -> Channel:
        """
        Helper method to create a new channel with consistent parameter handling.
        
        Args:
            parent: Parent channel to inherit properties from
            xdata: Time/index data for the new channel
            ydata: Signal data for the new channel
            params: Parameters used for processing
            suffix: Optional suffix for the channel name (defaults to step name)
            channel_tags: Optional tags for this specific channel (overrides cls.tags)
        """
        # Use provided suffix or default to step name
        name_suffix = suffix if suffix else cls.name
        
        # Use channel-specific tags if provided, otherwise use step class tags
        tags_to_use = channel_tags if channel_tags is not None else cls.tags
        
        return Channel.from_parent(
            parent=parent,
            xdata=xdata,
            ydata=ydata,
            legend_label=f"{parent.legend_label} - {name_suffix}",
            description=cls.description,
            tags=tags_to_use,
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
                           allow_length_change: bool = False, channel_type: str = "main") -> None:
        """
        Validate output data from processing.
        
        Args:
            y_input: Original input signal
            y_output: Processed output signal
            allow_length_change: Whether to allow output length to differ from input
            channel_type: Type of channel being validated (e.g., 'time-series', 'spectrogram')
            
        Raises:
            ValueError: If output data is invalid
        """
        if y_output is None:
            raise ValueError("Processing produced no output")
        
        y_output = np.asarray(y_output)
        
        if len(y_output) == 0:
            raise ValueError("Processing produced empty output")
        
        # Channel-type specific validation
        if channel_type == "time-series":
            # Time-series should maintain same length as input unless explicitly allowed
            if not allow_length_change and len(y_output) != len(y_input):
                raise ValueError(f"Time-series processing changed signal length: {len(y_input)} -> {len(y_output)}")
        elif channel_type == "spectrogram":
            # Spectrograms can have different time resolution
            # Just ensure it's not empty and has reasonable dimensions
            if y_output.ndim == 1:
                if len(y_output) == 0:
                    raise ValueError("Spectrogram frequency axis is empty")
            elif y_output.ndim == 2:
                if y_output.shape[0] == 0 or y_output.shape[1] == 0:
                    raise ValueError("Spectrogram data has zero dimensions")
            else:
                raise ValueError(f"Spectrogram data has unexpected dimensions: {y_output.shape}")
        elif channel_type == "reduced":
            # Reduced data (like envelope, peaks) can be shorter
            if len(y_output) == 0:
                raise ValueError("Reduced data is empty")
            if len(y_output) > len(y_input):
                raise ValueError(f"Reduced data cannot be longer than input: {len(y_input)} -> {len(y_output)}")
        else:
            # Default validation for unknown channel types
            if not allow_length_change and len(y_output) != len(y_input):
                raise ValueError(f"Processing changed signal length: {len(y_input)} -> {len(y_output)}")
        
        # Check for unexpected NaN values (skip for spectrograms as they may have NaN regions)
        if channel_type != "spectrogram" and np.any(np.isnan(y_output)) and not np.any(np.isnan(y_input)):
            raise ValueError("Processing produced unexpected NaN values")
        
        # Check for unexpected infinite values
        if np.any(np.isinf(y_output)) and not np.any(np.isinf(y_input)):
            raise ValueError("Processing produced unexpected infinite values")

    @classmethod
    def validate_channel_structure(cls, channel_info: dict, channel_index: int) -> None:
        """
        Validate the structure of a channel info dictionary.
        
        Args:
            channel_info: Dictionary containing channel data
            channel_index: Index of the channel for error messages
            
        Raises:
            ValueError: If channel structure is invalid
        """
        # Check required fields
        required_fields = ['tags']
        for field in required_fields:
            if field not in channel_info:
                raise ValueError(f"Channel {channel_index + 1} missing required field '{field}'")
        
        # Validate tags field
        tags = channel_info['tags']
        if not isinstance(tags, list):
            raise ValueError(f"Channel {channel_index + 1} 'tags' must be a list, got {type(tags)}")
        if len(tags) == 0:
            raise ValueError(f"Channel {channel_index + 1} 'tags' cannot be empty")
        
        # Check if this is a spectrogram channel
        is_spectrogram = 'spectrogram' in tags
        
        if is_spectrogram:
            # Spectrogram channels require 't', 'f', and 'z' fields
            spectrogram_fields = ['t', 'f', 'z']
            for field in spectrogram_fields:
                if field not in channel_info:
                    raise ValueError(f"Spectrogram channel {channel_index + 1} missing '{field}' field")
            
            # Validate spectrogram data fields
            t_data = channel_info['t']
            f_data = channel_info['f']
            z_data = channel_info['z']
            
            if t_data is None:
                raise ValueError(f"Spectrogram channel {channel_index + 1} 't' data cannot be None")
            if f_data is None:
                raise ValueError(f"Spectrogram channel {channel_index + 1} 'f' data cannot be None")
            if z_data is None:
                raise ValueError(f"Spectrogram channel {channel_index + 1} 'z' data cannot be None")
        else:
            # Regular channels require 'x' and 'y' fields
            regular_fields = ['x', 'y']
            for field in regular_fields:
                if field not in channel_info:
                    raise ValueError(f"Channel {channel_index + 1} missing required field '{field}'")
            
            # Validate regular data fields
            x_data = channel_info['x']
            y_data = channel_info['y']
            
            if x_data is None:
                raise ValueError(f"Channel {channel_index + 1} 'x' data cannot be None")
            if y_data is None:
                raise ValueError(f"Channel {channel_index + 1} 'y' data cannot be None")

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
