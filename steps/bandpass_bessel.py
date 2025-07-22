import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class bandpass_bessel_step(BaseStep):
    name = "bandpass_bessel"
    category = "Filter"
    description = "Apply bandpass Bessel filter with linear phase response to remove frequencies outside the specified range while preserving waveform shape."
    tags = ["filter", "bandpass", "scipy", "bessel", "linear-phase", "frequency", "passband"]
    params = [
        {"name": "low_cutoff", "type": "float", "default": "0.5", "help": "Low cutoff frequency (Hz)"},
        {"name": "high_cutoff", "type": "float", "default": "4.0", "help": "High cutoff frequency (Hz)"},
        {"name": "order", "type": "int", "default": "2", "help": "Order of the Bessel filter"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        order = cls.validate_integer_parameter("order", params.get("order"), min_val=1)
        low_cutoff = cls.validate_numeric_parameter("low_cutoff", params.get("low_cutoff"), min_val=0.0)
        high_cutoff = cls.validate_numeric_parameter("high_cutoff", params.get("high_cutoff"), min_val=0.0)
        
        if low_cutoff >= high_cutoff:
            raise ValueError(f"Low cutoff ({low_cutoff}) must be less than high cutoff ({high_cutoff})")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic"""
        from scipy.signal import bessel, filtfilt
        
        low_cutoff = params["low_cutoff"]
        high_cutoff = params["high_cutoff"]
        order = params["order"]
        
        # Validate frequencies against sampling rate
        nyq = 0.5 * fs
        if high_cutoff >= nyq:
            raise ValueError(f"High cutoff frequency ({high_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        if low_cutoff >= nyq:
            raise ValueError(f"Low cutoff frequency ({low_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        
        # Validate filter design and signal compatibility
        cutoff = [low_cutoff, high_cutoff]
        normal_cutoff = [f / nyq for f in cutoff]
        
        try:
            b, a = bessel(N=order, Wn=normal_cutoff, btype='band', analog=False)
        except ValueError as e:
            raise ValueError(f"Bessel bandpass filter design failed: {str(e)}")
        
        # Check if signal is long enough for the filter
        padlen = 3 * max(len(a), len(b))
        if len(y) <= padlen:
            raise ValueError(
                f"Signal too short for Bessel bandpass filter: "
                f"requires signal length > {padlen} but got {len(y)}. "
                f"Try reducing filter 'order' (currently {order})."
            )
        
        y_new = filtfilt(b, a, y)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_new
            }
        ]
