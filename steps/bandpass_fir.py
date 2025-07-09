import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class bandpass_fir_step(BaseStep):
    name = "bandpass_fir"
    category = "Filter"
    description = "Apply bandpass FIR filter using window method to remove frequencies outside the specified range."
    tags = ["time-series", "filter", "bandpass", "scipy", "fir", "finite", "frequency", "passband"]
    params = [
        {"name": "low_cutoff", "type": "float", "default": "0.5", "help": "Low cutoff frequency (Hz)"},
        {"name": "high_cutoff", "type": "float", "default": "4.0", "help": "High cutoff frequency (Hz)"},
        {"name": "numtaps", "type": "int", "default": "101", "help": "Number of filter taps (kernel size)"},
        {"name": "window", "type": "str", "default": "hamming", "options": ["hamming", "hann", "blackman", "bartlett"], "help": "Window function to use"},
        {"name": "fs", "type": "float", "default": "", "help": "Sampling frequency (injected from parent channel)"}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        fs = params.get("fs")
        numtaps = params.get("numtaps")
        low_cutoff = params.get("low_cutoff")
        high_cutoff = params.get("high_cutoff")
        window = params.get("window")
        
        if fs is None or fs <= 0:
            raise ValueError("Sampling frequency must be positive")
        if numtaps is None or numtaps <= 0:
            raise ValueError("Number of taps must be positive")
        if numtaps % 2 == 0:
            raise ValueError("Number of taps must be odd for bandpass filter")
        if low_cutoff is None or low_cutoff <= 0:
            raise ValueError("Low cutoff frequency must be positive")
        if high_cutoff is None or high_cutoff <= 0:
            raise ValueError("High cutoff frequency must be positive")
        if low_cutoff >= high_cutoff:
            raise ValueError(f"Low cutoff ({low_cutoff}) must be less than high cutoff ({high_cutoff})")
        
        nyq = 0.5 * fs
        if high_cutoff >= nyq:
            raise ValueError(f"High cutoff frequency ({high_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        if low_cutoff >= nyq:
            raise ValueError(f"Low cutoff frequency ({low_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        
        # Validate filter design parameters
        normal_cutoff = [low_cutoff / nyq, high_cutoff / nyq]
        if any(f >= 1.0 for f in normal_cutoff):
            raise ValueError("Cutoff frequencies too high relative to sampling rate")
        
        # Validate window parameter
        valid_windows = ["hamming", "hann", "blackman", "bartlett"]
        if window not in valid_windows:
            raise ValueError(f"Window must be one of {valid_windows}")

    @classmethod
    def _validate_filter_design(cls, params: dict, y: np.ndarray) -> None:
        """Validate filter design and signal compatibility"""
        from scipy.signal import firwin
        
        numtaps = params["numtaps"]
        low_cutoff = params["low_cutoff"]
        high_cutoff = params["high_cutoff"]
        window = params["window"]
        fs = params["fs"]
        
        cutoff = [low_cutoff, high_cutoff]
        nyq = 0.5 * fs
        normal_cutoff = [f / nyq for f in cutoff]
        
        try:
            b = firwin(numtaps, normal_cutoff, window=window, pass_zero=False)
        except ValueError as e:
            raise ValueError(f"FIR bandpass filter design failed: {str(e)}")
        
        # Check if signal is long enough for the filter
        padlen = 3 * len(b)
        if len(y) <= padlen:
            raise ValueError(
                f"Signal too short for FIR bandpass filter: "
                f"requires signal length > {padlen} but got {len(y)}. "
                f"Try reducing 'numtaps' (currently {numtaps})."
            )

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Processing produced unexpected NaN values")
        if np.any(np.isinf(y_new)) and not np.any(np.isinf(y_original)):
            raise ValueError("Processing produced unexpected infinite values")
        if np.all(np.isnan(y_new)):
            raise ValueError("Processing produced only NaN values")
        if np.all(np.isinf(y_new)):
            raise ValueError("Processing produced only infinite values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            if name == "fs": 
                continue
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "bool":
                    parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the processing step to a channel"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel using the helper method
            fs = cls._get_channel_fs(channel)
            
            # Inject sampling frequency if not provided in params
            if fs is not None and "fs" not in params:
                params["fs"] = fs
                print(f"[{cls.name}] Injected fs={fs:.2f} from channel")
            elif "fs" not in params:
                # If no fs available from channel, raise an error
                raise ValueError("No sampling frequency available from channel. Please provide 'fs' parameter.")
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            cls._validate_filter_design(params, y)
            
            # Process the data
            y_final = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_final,
                params=params,
                suffix="BandpassFIR"
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"FIR bandpass filter processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """Core processing logic"""
        from scipy.signal import firwin, filtfilt
        
        low_cutoff = params["low_cutoff"]
        high_cutoff = params["high_cutoff"]
        numtaps = params["numtaps"]
        window = params["window"]
        fs = params["fs"]
        
        cutoff = [low_cutoff, high_cutoff]
        nyq = 0.5 * fs
        normal_cutoff = [f / nyq for f in cutoff]
        
        b = firwin(numtaps, normal_cutoff, window=window, pass_zero=False)
        y_new = filtfilt(b, [1.0], y)
        
        return y_new
