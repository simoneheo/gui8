import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class eda_processing_step(BaseStep):
    name = "eda_processing"
    category = "BioSignal"
    description = """Apply BioSPPy EDA processing pipeline for electrodermal activity filtering and analysis."""
    tags = ["biosignal", "eda", "skin-conductance", "arousal", "stress"]
    params = []

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        pass

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from biosppy.signals import eda
        
        # Check if signal is long enough
        if len(y) < 100:
            raise ValueError("EDA signal too short for processing (minimum 100 samples)")
        
        # Apply BioSPPy EDA processing pipeline
        output = eda.eda(signal=y, sampling_rate=fs, show=False)
        
        # Extract filtered EDA signal
        y_filtered = output['filtered']
        
        # Create new time axis
        x_new = np.linspace(0, len(y_filtered)/fs, len(y_filtered))
        
        return [
            {
                'tags': ['time-series'],
                'x': x_new,
                'y': y_filtered
            }
        ]