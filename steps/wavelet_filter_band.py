import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

import pywt

@register_step
class wavelet_filter_band_step(BaseStep):
    name = "wavelet_filter_band"
    category = "pywt"
    description = """Reconstruct signal using selected wavelet frequency bands.
    
This step performs wavelet decomposition and reconstructs the signal using only
specified frequency bands (levels). This allows selective filtering of different
frequency components in the signal.

• **Wavelet type**: Type of wavelet for decomposition (e.g., db4, sym4)
• **Decomposition level**: Number of decomposition levels (auto if not specified)
• **Keep levels**: Comma-separated list of levels to preserve in reconstruction

Useful for:
• **Frequency filtering**: Selectively remove or preserve frequency bands
• **Signal decomposition**: Analyze different frequency components
• **Noise removal**: Remove high-frequency noise while preserving signal structure
• **Feature extraction**: Focus on specific frequency ranges of interest"""
    tags = ["time-series", "wavelet", "frequency-filtering", "band-selection", "pywt", "filter", "band"]
    params = [
        { 
            "name": "wavelet", 
            "type": "str", 
            "default": "db4", 
            "help": "Wavelet type for decomposition",
            "options": ["db1", "db2", "db4", "db8", "db10", "haar", "sym4", "sym5", "sym8", "coif2", "coif4", "coif6", "bior2.2", "bior4.4", "dmey"]
        },
        { 
            "name": "level", 
            "type": "int", 
            "default": "", 
            "help": "Decomposition level (optional, auto if blank)"
        },
        { 
            "name": "keep_levels", 
            "type": "str", 
            "default": "1,2", 
            "help": "Comma-separated levels to keep or reconstruct"
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Reconstruct signal using selected wavelet bands (Category: {cls.category})"

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
        if len(y) < 4:
            raise ValueError("Signal too short for wavelet decomposition (minimum 4 samples)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        keep_levels = params.get("keep_levels", [])
        
        if not keep_levels:
            raise ValueError("At least one level must be specified in keep_levels")
        
        if any(l < 0 for l in keep_levels):
            raise ValueError("Levels must be non-negative")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Wavelet band filtering produced unexpected NaN values")

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
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    if name == "keep_levels":
                        # Parse and validate keep_levels format
                        try:
                            levels = [int(l.strip()) for l in str(val).split(',')]
                            if not levels:
                                raise ValueError("At least one level must be specified")
                            if any(l < 0 for l in levels):
                                raise ValueError("Levels must be non-negative")
                        except ValueError as e:
                            if "invalid literal" in str(e):
                                raise ValueError("keep_levels must be comma-separated integers")
                            raise e
                        parsed[name] = levels
                    else:
                        parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply wavelet band filtering to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix="WaveletBandFilter"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Wavelet band filtering failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for wavelet band filtering"""
        wavelet = params.get("wavelet", "db4")
        level = params.get("level")
        keep_levels = params.get("keep_levels", [1, 2])
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(y, wavelet, mode='periodization', level=level)
        
        # Zero out coefficients for levels not in keep_levels
        for i in range(1, len(coeffs)):
            if i not in keep_levels:
                coeffs[i] = np.zeros_like(coeffs[i])
        
        # Reconstruct signal
        y_new = pywt.waverec(coeffs, wavelet, mode='periodization')[:len(y)]
        return y_new
