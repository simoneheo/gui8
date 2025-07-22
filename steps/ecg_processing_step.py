import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class ecg_processing_step(BaseStep):
    name = "ecg_processing"
    category = "BioSignal"
    description = """Apply BioSPPy ECG processing pipeline for R-peak detection and heart rate estimation."""
    tags = ["biosignal", "ecg", "heart-rate", "r-peaks", "hrv"]
    params = []

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        pass
    
    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from biosppy.signals import ecg

        if len(y) < 500:
            raise ValueError("ECG signal too short for heart rate analysis (minimum 500 samples)")

        # Run ECG analysis
        output = ecg.ecg(signal=y, sampling_rate=fs, show=False)

        # Extract timestamps and heart rate values
        hr_times = output['heart_rate_ts']   # Timestamps (seconds)
        hr_values = output['heart_rate']     # Heart rate values (bpm)

        # Validate
        if hr_times is None or hr_values is None or len(hr_times) != len(hr_values):
            raise ValueError("BioSPPy ECG output invalid or empty")

        return [
            {
                'tags': ['time-series'],
                'x': hr_times,
                'y': hr_values
            }
        ]
