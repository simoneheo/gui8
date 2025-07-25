import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class highpass_fir_step(BaseStep):
    name = "highpass_fir"
    category = "Filter"
    description = "Apply highpass FIR filter with linear phase response and precise frequency control to remove low frequencies below the cutoff."
    tags = ["time-series", "filter", "highpass", "scipy", "fir", "linear-phase", "frequency", "cutoff"]
    params = [
        {"name": "cutoff", "type": "float", "default": "0.5", "help": "Cutoff frequency in Hz"},
        {"name": "numtaps", "type": "int", "default": "101", "help": "Filter length (odd number, higher = sharper cutoff)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        numtaps = cls.validate_integer_parameter("numtaps", params.get("numtaps"), min_val=3)
        cutoff = cls.validate_numeric_parameter("cutoff", params.get("cutoff"), min_val=0.0)
        
        if numtaps % 2 == 0:
            raise ValueError("numtaps must be odd for FIR filter")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic"""
        from scipy.signal import firwin, filtfilt
        
        cutoff = params["cutoff"]
        numtaps = params["numtaps"]
        
        # Validate frequency against sampling rate
        nyq = 0.5 * fs
        if cutoff >= nyq:
            raise ValueError(f"Cutoff frequency ({cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        
        # Validate filter length vs signal length
        if numtaps > len(y):
            raise ValueError(f"Filter length ({numtaps}) cannot be longer than signal length ({len(y)})")
        
        # Design FIR highpass filter
        try:
            b = firwin(numtaps, cutoff, pass_zero=False, fs=fs)
        except ValueError as e:
            raise ValueError(f"FIR highpass filter design failed: {str(e)}")
        
        # Apply filter
        y_new = filtfilt(b, 1.0, y)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_new
            }
        ]
