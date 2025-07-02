import numpy as np
import scipy.signal
import scipy.stats
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class WindowedEnergyStep(BaseStep):
    name = "windowed_energy"
    category = "Features"
    description = "Computes windowed energy over sliding time windows."
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
        try:
            window = float(user_input.get("window", 1.0))
            overlap = float(user_input.get("overlap", 0.5))
            
            # Parameter validation
            if window <= 0:
                raise ValueError(f"Window duration must be > 0, got {window}")
            if not (0.0 <= overlap < 1.0):
                raise ValueError(f"Overlap must be between 0.0 and 0.9, got {overlap}")
            
            return {"window": window, "overlap": overlap}
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
        
        x, y = channel.xdata, channel.ydata
        window_duration = params["window"]
        overlap = params["overlap"]
        
        # Calculate time span and step
        time_span = x[-1] - x[0]
        if time_span <= 0:
            raise ValueError("Time span must be positive")
        if window_duration > time_span:
            raise ValueError(f"Window duration ({window_duration}s) is larger than signal duration ({time_span:.3f}s)")
        
        step_duration = window_duration * (1 - overlap)
        if step_duration <= 0:
            raise ValueError(f"Step duration is too small: {step_duration}")
        
        try:
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
                window_x = x[window_indices]
                
                # Compute energy
                try:
                    energy = np.sum(window_y**2)
                    if np.isnan(energy) or np.isinf(energy):
                        continue  # Skip invalid results
                    
                    # Use center time of window
                    center_time = start_time + window_duration / 2
                    x_new.append(center_time)
                    y_new.append(energy)
                    
                except Exception as e:
                    continue  # Skip this window on error
            
            if len(y_new) == 0:
                raise ValueError("No valid windows found")
            
            return cls.create_new_channel(parent=channel, xdata=np.array(x_new), ydata=np.array(y_new), params=params)
            
        except Exception as e:
            raise ValueError(f"Windowed energy computation failed: {str(e)}")
