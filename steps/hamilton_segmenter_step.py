import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class hamilton_segmenter_step(BaseStep):
    name = "hamilton_segmenter"
    category = "BioSignal"
    description = """Detect QRS complexes in ECG signal using BioSPPy's Hamilton segmenter algorithm."""
    tags = ["time-series", "biosignal", "ecg", "qrs", "peak-detection", "r-peaks"]
    params = []

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from biosppy.signals.ecg import hamilton_segmenter
        
        # Check if signal is long enough
        if len(y) < 100:
            raise ValueError("ECG signal too short for QRS detection (minimum 100 samples)")
        
        # Apply Hamilton segmenter
        rpeaks = hamilton_segmenter(signal=y, sampling_rate=fs)[0]
        
        # Create binary marker signal
        rpeak_marker = np.zeros_like(y, dtype=float)
        
        # Ensure R-peak indices are within bounds
        valid_rpeaks = rpeaks[rpeaks < len(y)]
        rpeak_marker[valid_rpeaks] = 1.0
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': rpeak_marker
            }
        ]