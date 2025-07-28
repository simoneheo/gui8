import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel
from scipy.signal import find_peaks

@register_step
class detect_peaks_step(BaseStep):
    name = "detect peaks"
    category = "Features"
    description = """Detects peaks in the signal using height, distance, and prominence criteria.
Automatically adjusts parameters if not specified to ensure robust peak detection."""
    tags = ["peak", "detection", "events", "feature", "marker"]
    params = [
        {
            "name": "height", 
            "type": "float", 
            "default": "", 
            "help": "Minimum peak height (amplitude). Leave blank for auto-detection based on signal statistics."
        },
        {
            "name": "distance", 
            "type": "int", 
            "default": "1", 
            "help": "Minimum distance between peaks in samples. Use larger values to avoid detecting multiple peaks in noisy regions."
        },
        {
            "name": "prominence", 
            "type": "float", 
            "default": "", 
            "help": "Minimum peak prominence (how much a peak stands out from surrounding baseline). Leave blank for auto-detection."
        }
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        # Height can be empty (auto-detection) or a positive number
        if params.get("height") not in [None, "", "auto"]:
            cls.validate_numeric_parameter("height", params.get("height"), min_val=0.0)
        
        distance = cls.validate_integer_parameter("distance", params.get("distance"), min_val=1)
        
        # Prominence can be empty (auto-detection) or a positive number
        if params.get("prominence") not in [None, "", "auto"]:
            cls.validate_numeric_parameter("prominence", params.get("prominence"), min_val=0.0)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        """Core processing logic for peak detection"""
        height = params.get("height", "")
        distance = params["distance"]
        prominence = params.get("prominence", "")
        
        # Auto-detect height if not specified
        if height in [None, "", "auto"]:
            height = np.mean(y) + 0.5 * np.std(y)
        else:
            height = float(height)
            
        # Auto-detect prominence if not specified
        if prominence in [None, "", "auto"]:
            prominence = 0.1 * np.ptp(y)  # 10% of peak-to-peak range
        else:
            prominence = float(prominence)
        
        # Find peaks using scipy
        peaks, properties = find_peaks(
            y, 
            height=height, 
            distance=distance, 
            prominence=prominence
        )

        # Return actual peak positions and amplitudes
        return [
            {
                'tags': ['time-series'],
                'x': x[peaks],
                'y': y[peaks]
            }
        ]
