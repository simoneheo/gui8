import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class bandpass_fir_step(BaseStep):
    name = "bandpass fir"
    category = "Filter"
    description = """Apply bandpass FIR filter with linear phase response and precise frequency control.
Removes frequencies outside the specified range while maintaining linear phase characteristics."""
    tags = [ "filter", "bandpass", "scipy", "fir", "linear-phase", "frequency", "passband"]
    params = [
        {"name": "low_cutoff", "type": "float", "default": "0.5", "help": "Low cutoff frequency (Hz)"},
        {"name": "high_cutoff", "type": "float", "default": "4.0", "help": "High cutoff frequency (Hz)"},
        {"name": "numtaps", "type": "int", "default": "101", "help": "Filter length (odd number, higher = sharper cutoff)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        numtaps = cls.validate_integer_parameter("numtaps", params.get("numtaps"), min_val=3)
        low_cutoff = cls.validate_numeric_parameter("low_cutoff", params.get("low_cutoff"), min_val=0.0)
        high_cutoff = cls.validate_numeric_parameter("high_cutoff", params.get("high_cutoff"), min_val=0.0)
        
        if numtaps % 2 == 0:
            raise ValueError("numtaps must be odd for FIR filter")
        if low_cutoff >= high_cutoff:
            raise ValueError(f"Low cutoff ({low_cutoff}) must be less than high cutoff ({high_cutoff})")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic"""
        from scipy.signal import firwin, filtfilt
        
        low_cutoff = params["low_cutoff"]
        high_cutoff = params["high_cutoff"]
        numtaps = params["numtaps"]
        
        # Validate frequencies against sampling rate
        nyq = 0.5 * fs
        if high_cutoff >= nyq:
            raise ValueError(f"High cutoff frequency ({high_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        if low_cutoff >= nyq:
            raise ValueError(f"Low cutoff frequency ({low_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        
        # Validate filter length vs signal length
        if numtaps > len(y):
            raise ValueError(f"Filter length ({numtaps}) cannot be longer than signal length ({len(y)})")
        
        # Design FIR bandpass filter
        try:
            b = firwin(numtaps, [low_cutoff, high_cutoff], pass_zero=False, fs=fs)
        except ValueError as e:
            raise ValueError(f"FIR bandpass filter design failed: {str(e)}")
        
        # Apply filter
        y_new = filtfilt(b, 1.0, y)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_new
            }
        ]
