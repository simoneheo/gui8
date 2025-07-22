import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class lowpass_bessel_step(BaseStep):
    name = "lowpass_bessel"
    category = "Filter"
    description = "Apply lowpass Bessel filter with linear phase response to remove high frequencies while preserving waveform shape."
    tags = ["time-series", "filter", "lowpass", "scipy", "bessel", "linear-phase", "frequency", "cutoff"]
    params = [
        {"name": "cutoff", "type": "float", "default": "2.0", "help": "Cutoff frequency in Hz"},
        {"name": "order", "type": "int", "default": "2", "help": "Order of the Bessel filter"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        order = cls.validate_integer_parameter("order", params.get("order"), min_val=1)
        cutoff = cls.validate_numeric_parameter("cutoff", params.get("cutoff"), min_val=0.0)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic"""
        from scipy.signal import bessel, filtfilt
        
        cutoff = params["cutoff"]
        order = params["order"]
        
        # Validate frequency against sampling rate
        nyq = 0.5 * fs
        if cutoff >= nyq:
            raise ValueError(f"Cutoff frequency ({cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
        
        normal_cutoff = cutoff / nyq
        
        try:
            b, a = bessel(N=order, Wn=normal_cutoff, btype='low', analog=False)
        except ValueError as e:
            raise ValueError(f"Bessel lowpass filter design failed: {str(e)}")
        
        # Check if signal is long enough for the filter
        padlen = 3 * max(len(a), len(b))
        if len(y) <= padlen:
            raise ValueError(
                f"Signal too short for Bessel lowpass filter: "
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
