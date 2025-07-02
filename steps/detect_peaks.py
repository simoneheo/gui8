import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel
from scipy.signal import find_peaks

@register_step
class DetectPeaksStep(BaseStep):
    name = "detect_peaks"
    category = "Event"
    description = "Detects peaks in the signal using height, distance, and prominence criteria."
    tags = ["time-series", "event"]
    params = [
        {
            "name": "height", 
            "type": "float", 
            "default": "", 
            "help": "Minimum peak height (amplitude). Peaks below this value are ignored. Leave blank for auto-detection based on signal statistics."
        },
        {
            "name": "distance", 
            "type": "int", 
            "default": "1", 
            "help": "Minimum distance between peaks in samples. Use larger values to avoid detecting multiple peaks in noisy regions. Typical: 10-50 samples."
        },
        {
            "name": "prominence", 
            "type": "float", 
            "default": "", 
            "help": "Minimum peak prominence (how much a peak stands out from surrounding baseline). Higher values = more selective. Leave blank for auto."
        }
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        # Parse height parameter
        height_val = user_input.get("height", "").strip()
        if height_val:
            try:
                parsed["height"] = float(height_val)
            except ValueError:
                raise ValueError("Height must be a valid number")
        
        # Parse distance parameter
        try:
            distance = int(user_input.get("distance", "1"))
            if distance < 1:
                raise ValueError("Distance must be at least 1")
            parsed["distance"] = distance
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Distance must be a valid integer")
            raise e
        
        # Parse prominence parameter
        prominence_val = user_input.get("prominence", "").strip()
        if prominence_val:
            try:
                parsed["prominence"] = float(prominence_val)
            except ValueError:
                raise ValueError("Prominence must be a valid number")

        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata

        # Validate input data
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if len(y) < 3:
            raise ValueError("Signal too short for peak detection (minimum 3 samples)")

        try:
            # Build find_peaks kwargs, only include non-None parameters
            kwargs = {}
            if "height" in params:
                kwargs["height"] = params["height"]
            if "distance" in params:
                kwargs["distance"] = params["distance"]
            if "prominence" in params:
                kwargs["prominence"] = params["prominence"]
            
            indices, _ = find_peaks(y, **kwargs)
            
            if len(indices) == 0:
                raise ValueError("No peaks detected. Try adjusting height, distance, or prominence parameters.")
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x[indices], 
                ydata=y[indices], 
                params=params
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Peak detection failed: {str(e)}")
