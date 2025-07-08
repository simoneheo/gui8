import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class top_percentile_sliding_step(BaseStep):
    name = "top_percentile_sliding"
    category = "Envelope"
    description = "Top percentile in overlapping sliding windows"
    tags = ["time-series"]
    params = [
        {"name": "window", "type": "int", "default": "25", "help": "Sliding window size (samples)"},
        {"name": "percentile", "type": "float", "default": "95.0", "help": "Top percentile to compute in each window (0-100)"},
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        try:
            parsed = {}
            for param in cls.params:
                value = user_input.get(param["name"], param["default"])
                if param["type"] == "float":
                    parsed[param["name"]] = float(value)
                else:
                    parsed[param["name"]] = int(value)
            
            # Parameter validation
            window = parsed["window"]
            percentile = parsed["percentile"]
            
            if window <= 0:
                raise ValueError(f"Window size must be positive, got {window}")
            if not (0.0 <= percentile <= 100.0):
                raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
            
            return parsed
        except ValueError as e:
            raise ValueError(f"Parameter validation failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to parse input parameters: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Input validation
        if channel is None:
            raise ValueError("Channel is None")
        if channel.ydata is None or len(channel.ydata) == 0:
            raise ValueError("Channel has no data")
        if channel.xdata is None or len(channel.xdata) == 0:
            raise ValueError("Channel has no x-axis data")
        if len(channel.xdata) != len(channel.ydata):
            raise ValueError("X and Y data lengths don't match")
        
        # Check for NaN values
        if np.any(np.isnan(channel.ydata)):
            raise ValueError("Input signal contains NaN values")
        if np.any(np.isinf(channel.ydata)):
            raise ValueError("Input signal contains infinite values")
        
        y = channel.ydata
        x = channel.xdata
        window = int(params['window'])
        q = params['percentile']
        
        # Additional validation
        if len(y) < window:
            raise ValueError(f"Signal too short for window: signal length={len(y)}, window={window}")
        
        try:
            y_new = []
            for i in range(len(y)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(y), i + window // 2 + 1)
                window_data = np.abs(y[start_idx:end_idx])
                
                if len(window_data) == 0:
                    raise ValueError(f"Empty window at index {i}")
                    
                try:
                    percentile_val = np.percentile(window_data, q)
                    if np.isnan(percentile_val) or np.isinf(percentile_val):
                        raise ValueError(f"Invalid percentile value at index {i}")
                    y_new.append(percentile_val)
                except Exception as e:
                    raise ValueError(f"Failed to compute percentile at index {i}: {str(e)}")
            
            if len(y_new) == 0:
                raise ValueError("No valid percentile values computed")
                
            return cls.create_new_channel(parent=channel, xdata=x, ydata=np.array(y_new), params=params)
            
        except Exception as e:
            raise ValueError(f"Top percentile envelope computation failed: {str(e)}")
