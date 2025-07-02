import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel
from scipy.signal import argrelextrema

@register_step
class DetectExtremaStep(BaseStep):
    name = "detect_extrema"
    category = "Event"
    description = "Detects local extrema (both peaks and valleys) in the signal."
    tags = ["time-series", "event"]
    params = [
        {'name': 'min_height', 'type': 'float', 'default': '0.1', 'help': 'Minimum height/depth for extrema detection as fraction of signal range (0.0-1.0).'},
        {'name': 'window', 'type': 'float', 'default': '0.1', 'help': 'Time window duration in seconds for local extrema detection. Larger windows find more significant extrema.'}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        return {
            'min_height': float(user_input.get('min_height', 0.1)),
            'window': float(user_input.get('window', 0.1))
        }

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        x, y = channel.xdata, channel.ydata
        min_height = params['min_height']
        window_duration = params['window']
        
        if window_duration <= 0:
            raise ValueError(f"Window duration must be > 0, got {window_duration}")
        if not (0.0 <= min_height <= 1.0):
            raise ValueError(f"min_height must be between 0.0 and 1.0, got {min_height}")

        try:
            # Calculate signal range for height threshold
            signal_range = np.max(y) - np.min(y)
            height_threshold = min_height * signal_range
            
            # Convert time window to approximate sample order
            # Find average time spacing to estimate sample order
            if len(x) < 2:
                raise ValueError("Need at least 2 data points")
                
            time_diffs = np.diff(x)
            avg_time_spacing = np.median(time_diffs)  # Use median to be robust to outliers
            
            if avg_time_spacing <= 0:
                raise ValueError("Invalid time spacing detected")
                
            # Convert window duration to approximate sample order
            sample_order = max(1, int(window_duration / avg_time_spacing))
            
            # Find local maxima and minima using scipy
            max_idx = argrelextrema(y, np.greater, order=sample_order)[0]
            min_idx = argrelextrema(y, np.less, order=sample_order)[0]
            
            # Filter extrema by height threshold
            valid_max_idx = []
            valid_min_idx = []
            
            signal_mean = np.mean(y)
            
            # Filter maxima: must be above mean + threshold
            for idx in max_idx:
                if y[idx] - signal_mean >= height_threshold:
                    valid_max_idx.append(idx)
            
            # Filter minima: must be below mean - threshold  
            for idx in min_idx:
                if signal_mean - y[idx] >= height_threshold:
                    valid_min_idx.append(idx)
            
            # Combine and sort all valid extrema
            all_indices = np.sort(np.concatenate([valid_max_idx, valid_min_idx]))
            
            if len(all_indices) == 0:
                # Return empty result if no extrema found
                return cls.create_new_channel(parent=channel, xdata=np.array([]), ydata=np.array([]), params=params)
                
        except Exception as e:
            raise ValueError(f"Failed during extrema detection: {str(e)}")

        return cls.create_new_channel(parent=channel, xdata=x[all_indices], ydata=y[all_indices], params=params)
