import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class WindowedMedianStep(BaseStep):
    name = "windowed_median"
    category = "Features"
    description = "Computes windowed median over sliding time windows."
    tags = ["time-series", "feature"]
    params = [
        {"name": "window", "type": "float", "default": "1.0", "help": "Window duration in seconds"},
        {"name": "overlap", "type": "float", "default": "0.5", "help": "Overlap fraction [0.0 - 0.9]"}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        return {
            "window": float(user_input.get("window", 1.0)),
            "overlap": float(user_input.get("overlap", 0.5))
        }

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata
        window_duration = params["window"]
        overlap = params["overlap"]
        
        if window_duration <= 0:
            raise ValueError(f"Window duration must be > 0, got {window_duration}")
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"Overlap must be between 0.0 and 0.9, got {overlap}")
        
        time_span = x[-1] - x[0]
        if window_duration > time_span:
            raise ValueError(f"Window duration ({window_duration}s) is larger than signal duration ({time_span:.3f}s)")
        
        step_duration = window_duration * (1 - overlap)
        
        # Generate window start times
        window_starts = []
        current_time = x[0]
        while current_time + window_duration <= x[-1]:
            window_starts.append(current_time)
            current_time += step_duration
        
        if len(window_starts) == 0:
            raise ValueError("No valid windows found")
        
        x_new = []
        y_new = []
        
        for start_time in window_starts:
            end_time = start_time + window_duration
            
            # Find all samples within this time window
            mask = (x >= start_time) & (x <= end_time)
            window_indices = np.where(mask)[0]
            
            if len(window_indices) < 2:
                continue  # Skip windows with insufficient data
            
            window_y = y[window_indices]
            
            try:
                result = np.median(window_y)
                if np.isnan(result) or np.isinf(result):
                    continue
                
                center_time = start_time + window_duration / 2
                x_new.append(center_time)
                y_new.append(result)
            except Exception as e:
                continue  # Skip this window on error

        if len(y_new) == 0:
            raise ValueError("No valid windows found")

        return cls.create_new_channel(parent=channel, xdata=np.array(x_new), ydata=np.array(y_new), params=params)
