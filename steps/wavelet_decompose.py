import numpy as np
import pywt
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class wavelet_decompose_step(BaseStep):
    name = "wavelet decompose"
    category = "Transform"
    description = """Decompose signal using discrete wavelet transform (DWT) into approximation and detail coefficients.
Creates multiple channels for different frequency components of the signal."""
    tags = ["time-series", "wavelet", "decomposition", "dwt", "pywt", "frequency", "multiresolution"]
    params = [
        {
            "name": "wavelet", 
            "type": "str", 
            "default": "db4", 
            "options": ["haar", "db4", "db8", "bior2.2", "coif2"], 
            "help": "Wavelet function to use for decomposition"
        },
        {
            "name": "mode", 
            "type": "str", 
            "default": "symmetric", 
            "options": ["symmetric", "periodization", "zero", "constant"], 
            "help": "Signal extension mode for boundary conditions"
        },
        {
            "name": "levels", 
            "type": "str", 
            "default": "1,2", 
            "help": "Comma-separated levels to decompose (e.g., '1,2,3')"
        }
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        wavelet = cls.validate_string_parameter("wavelet", params.get("wavelet"), 
                                               valid_options=["haar", "db4", "db8", "bior2.2", "coif2"])
        mode = cls.validate_string_parameter("mode", params.get("mode"),
                                           valid_options=["symmetric", "periodization", "zero", "constant"])
        levels = cls.validate_string_parameter("levels", params.get("levels"))
        
        # Validate levels format
        try:
            level_list = [int(x.strip()) for x in levels.split(',')]
            for level in level_list:
                if level < 1:
                    raise ValueError("All levels must be positive integers")
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid levels format '{levels}': must be comma-separated positive integers (e.g., '1,2,3')")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic for wavelet decomposition"""
        wavelet = params["wavelet"]
        mode = params["mode"]
        levels = params["levels"]
        
        # Parse levels
        level_list = [int(x.strip()) for x in levels.split(',')]
        max_level = max(level_list)
        
        # Check if signal is long enough for decomposition
        min_len = pywt.dwt_max_level(len(y), wavelet)
        if max_level > min_len:
            raise ValueError(f"Signal too short for level {max_level} decomposition (max possible: {min_len})")
        
        # Perform multi-level DWT
        coeffs = pywt.wavedec(y, wavelet, mode=mode, level=max_level)
        
        # Create output channels
        result_channels = []
        
        # Add approximation coefficients (lowest frequency)
        if max_level in level_list:
            approx_coeffs = coeffs[0]
            # Create time axis for approximation coefficients
            approx_time = np.linspace(x[0], x[-1], len(approx_coeffs))
            
            result_channels.append({
                'tags': ['time-series'],
                'x': approx_time,
                'y': approx_coeffs
            })
        
        # Add detail coefficients (higher frequencies)
        for level in level_list:
            if 1 <= level <= max_level:
                detail_coeffs = coeffs[level]
                # Create time axis for detail coefficients
                detail_time = np.linspace(x[0], x[-1], len(detail_coeffs))
                
                result_channels.append({
                    'tags': ['time-series'],
                    'x': detail_time,
                    'y': detail_coeffs
                })
        
        if not result_channels:
            raise ValueError("No valid decomposition levels found")
        
        return result_channels 
