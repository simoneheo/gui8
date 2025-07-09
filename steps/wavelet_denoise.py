import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

import pywt

@register_step
class wavelet_denoise_step(BaseStep):
    name = "wavelet_denoise"
    category = "pywt"
    description = """Denoise signal using wavelet thresholding to remove noise while preserving important features.
    
This step applies wavelet decomposition, thresholds the coefficients to remove noise,
and reconstructs the signal. The process preserves important signal features while
reducing random noise and artifacts.

• **Wavelet type**: Type of wavelet for decomposition (e.g., db4, sym4)
• **Decomposition level**: Number of decomposition levels (auto if not specified)
• **Threshold**: Threshold value for coefficient shrinkage (0-1 range typical)
• **Thresholding mode**: Soft (gradual) or hard (abrupt) thresholding

Useful for:
• **Noise reduction**: Remove random noise from signals
• **Signal enhancement**: Improve signal quality for analysis
• **Feature preservation**: Maintain important peaks and patterns
• **Preprocessing**: Prepare signals for further analysis"""
    tags = ["time-series", "wavelet", "denoising", "noise-reduction", "pywt", "thresholding", "clean"]
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
            "help": "Decomposition level (leave blank for auto)"
        },
        { 
            "name": "threshold", 
            "type": "float", 
            "default": "0.2", 
            "help": "Threshold for coefficient shrinkage (0-1 range typical)",
        },
        { 
            "name": "mode", 
            "type": "str", 
            "default": "soft", 
            "help": "Thresholding mode",
            "options": ["soft", "hard"]
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Denoise signal using wavelet thresholding (Category: {cls.category})"

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
        threshold = params.get("threshold")
        mode = params.get("mode", "soft")
        
        if threshold < 0 or threshold > 10:
            raise ValueError("Threshold should typically be between 0 and 10")
        
        valid_modes = ["soft", "hard"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}, got '{mode}'")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)) and not np.any(np.isnan(y_original)):
            raise ValueError("Wavelet denoising produced unexpected NaN values")

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
                    parsed_val = int(val)
                    if name == "level" and parsed_val < 1:
                        raise ValueError("Decomposition level must be at least 1")
                    parsed[name] = parsed_val
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply wavelet denoising to the channel data."""
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
                suffix="WaveletDenoise"
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Wavelet denoising failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for wavelet denoising"""
        wavelet = params.get("wavelet", "db4")
        level = params.get("level")
        threshold = params.get("threshold", 0.2)
        mode = params.get("mode", "soft")
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(y, wavelet, mode='periodization', level=level)
        
        # Apply thresholding to detail coefficients
        thresholded = [pywt.threshold(c, threshold, mode=mode) if i > 0 else c for i, c in enumerate(coeffs)]
        
        # Reconstruct signal
        y_new = pywt.waverec(thresholded, wavelet, mode='periodization')[:len(y)]
        return y_new
