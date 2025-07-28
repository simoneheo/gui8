import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class wavelet_filter_band_step(BaseStep):
    name = "wavelet filter band"
    category = "Filter"
    description = """Filter signal to specific frequency band using wavelet decomposition.
Extracts frequency components within specified band."""
    tags = ["time-series", "wavelet", "filter", "bandpass", "frequency-band"]
    params = [
        {"name": "wavelet", "type": "str", "default": "db4", "help": "Wavelet type (e.g., db4, haar, sym4)"},
        {"name": "level", "type": "int", "default": "3", "help": "Decomposition level"},
        {"name": "band", "type": "str", "default": "detail", "options": ["approximation", "detail", "specific"], "help": "Frequency band to extract"},
        {"name": "specific_level", "type": "int", "default": "1", "help": "Specific detail level (when band=specific)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        wavelet = cls.validate_string_parameter("wavelet", params.get("wavelet"))
        level = cls.validate_integer_parameter("level", params.get("level"), min_val=1, max_val=10)
        band = cls.validate_string_parameter("band", params.get("band"), 
                                            valid_options=["approximation", "detail", "specific"])
        specific_level = cls.validate_integer_parameter("specific_level", params.get("specific_level"), min_val=1, max_val=10)
        
        # Validate specific_level constraint
        if band == "specific" and specific_level > level:
            raise ValueError("specific_level cannot exceed decomposition level")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        import pywt
        
        wavelet = params["wavelet"]
        level = params["level"]
        band = params["band"]
        specific_level = params["specific_level"]
        
        # Check if signal is long enough for wavelet decomposition
        if len(y) < 2**level:
            raise ValueError(f"Signal too short for level {level} decomposition (minimum {2**level} samples)")
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(y, wavelet, level=level)
        
        # Create filtered coefficients based on band selection
        filtered_coeffs = [np.zeros_like(c) for c in coeffs]
        
        if band == "approximation":
            # Keep only approximation coefficients
            filtered_coeffs[0] = coeffs[0]
        elif band == "detail":
            # Keep only detail coefficients (all levels)
            for i in range(1, len(coeffs)):
                filtered_coeffs[i] = coeffs[i]
        elif band == "specific":
            # Keep only specific detail level
            if specific_level < len(coeffs):
                filtered_coeffs[specific_level] = coeffs[specific_level]
            else:
                raise ValueError(f"Specific level {specific_level} exceeds decomposition depth")
        
        # Reconstruct signal from filtered coefficients
        y_filtered = pywt.waverec(filtered_coeffs, wavelet)
        
        # Ensure output length matches input
        if len(y_filtered) != len(y):
            y_filtered = y_filtered[:len(y)]
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_filtered
            }
        ]
