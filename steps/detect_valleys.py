import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel
from scipy.signal import find_peaks

@register_step
class detect_valleys_step(BaseStep):
    name = "detect valleys"
    category = "Features"
    description = """Detect valleys (minima) in the signal by inverting and finding peaks.
Creates a binary signal marking valley locations."""
    tags = ["valley", "detection", "events", "feature", "marker"]
    params = [
        {"name": "height", "type": "float", "default": "", "help": "Minimum valley depth (leave blank for auto)"},
        {"name": "distance", "type": "int", "default": "1", "help": "Minimum distance between valleys in samples"},
        {"name": "prominence", "type": "float", "default": "", "help": "Minimum valley prominence (leave blank for auto)"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # Height can be empty (auto-detection) or a numeric value
        if params.get("height") not in [None, "", "auto"]:
            cls.validate_numeric_parameter("height", params.get("height"))
        
        distance = cls.validate_integer_parameter("distance", params.get("distance"), min_val=1)
        
        # Prominence can be empty (auto-detection) or a positive number
        if params.get("prominence") not in [None, "", "auto"]:
            cls.validate_numeric_parameter("prominence", params.get("prominence"), min_val=0.0)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        height = params.get("height", "")
        distance = params["distance"]
        prominence = params.get("prominence", "")

        # Invert signal to find valleys as peaks
        y_inverted = -y

        # Auto-detect height if not specified
        if height in [None, "", "auto"]:
            height = np.mean(y_inverted) + 0.5 * np.std(y_inverted)
        else:
            height = -float(height)  # Invert because we inverted the signal

        # Auto-detect prominence if not specified
        if prominence in [None, "", "auto"]:
            prominence = 0.1 * np.ptp(y_inverted)  # 10% of peak-to-peak range
        else:
            prominence = float(prominence)

        # Find valleys as peaks in inverted signal
        valleys, properties = find_peaks(
            y_inverted,
            height=height,
            distance=distance,
            prominence=prominence
        )

        # Return actual valley positions and amplitudes
        return [
            {
                'tags': ['time-series'],
                'x': x[valleys],
                'y': y[valleys]
            }
        ]
