import numpy as np
from steps.base_step import BaseStep
from steps.process_registry import register_step
from channel import Channel

@register_step
class CountSamplesStep(BaseStep):
    name = "count_samples"
    category = "Features"
    description = """Counts the number of samples within sliding time windows and reports the count in various units.
    
This step is particularly useful for:
- Analyzing data density variations over time
- Detecting sampling rate changes or gaps in data
- Quality control of data acquisition
- Preprocessing for rate-based analysis

The step uses overlapping sliding windows to compute sample counts continuously over time.
Window centers are used as time stamps for the output time series.

Output units:
- count/window: Raw number of samples per window
- count/min: Samples per minute (extrapolated from window rate)
- count/s: Samples per second (extrapolated from window rate)

Note: For units other than 'count/window', the count is extrapolated based on the window duration.
For example, if a 2-second window contains 1000 samples, count/s would be 500."""
    tags = ["time-series", "feature"]
    params = [
        {
            "name": "window", 
            "type": "float", 
            "default": "1.0", 
            "help": "Window duration in seconds. Must be positive and smaller than total signal duration."
        },
        {
            "name": "overlap", 
            "type": "float", 
            "default": "50.0", 
            "help": "Window overlap as percentage (0-90). Higher overlap gives smoother results but more computation."
        },
        {
            "name": "unit", 
            "type": "str", 
            "default": "count/window", 
            "options": ["count/window", "count/min", "count/s"],
            "help": "Output unit for sample counts. 'count/window' gives raw counts, others extrapolate to rates."
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"
    
    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}
    
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters."""
        try:
            parsed = {}
            
            # Parse window parameter
            window_val = user_input.get("window", 1.0)
            
            # Handle both string and numeric inputs
            if isinstance(window_val, str):
                window_val = window_val.strip()
                if not window_val:
                    raise ValueError("Window parameter cannot be empty")
                try:
                    window = float(window_val)
                except ValueError:
                    raise ValueError(f"Window must be a valid number, got '{window_val}'")
            elif isinstance(window_val, (int, float)):
                window = float(window_val)
            else:
                raise ValueError(f"Window must be a number, got {type(window_val).__name__}: {window_val}")
            
            if window <= 0:
                raise ValueError(f"Window duration must be positive, got {window}")
            if window > 3600:  # Sanity check: no more than 1 hour
                raise ValueError(f"Window duration seems too large ({window}s). Maximum allowed is 3600s")
            
            parsed["window"] = window
            
            # Parse overlap parameter
            overlap_val = user_input.get("overlap", 50.0)
            
            # Handle both string and numeric inputs
            if isinstance(overlap_val, str):
                overlap_val = overlap_val.strip()
                if not overlap_val:
                    raise ValueError("Overlap parameter cannot be empty")
                try:
                    overlap = float(overlap_val)
                except ValueError:
                    raise ValueError(f"Overlap must be a valid number, got '{overlap_val}'")
            elif isinstance(overlap_val, (int, float)):
                overlap = float(overlap_val)
            else:
                raise ValueError(f"Overlap must be a number, got {type(overlap_val).__name__}: {overlap_val}")
            
            if not (0.0 <= overlap < 90.0):
                raise ValueError(f"Overlap must be between 0.0 and 90.0 percent, got {overlap}")
            
            parsed["overlap"] = overlap / 100.0  # Convert percentage to fraction
            
            # Parse unit parameter
            unit = user_input.get("unit", "count/window")
            
            # Handle string input (strip if it's a string)
            if isinstance(unit, str):
                unit = unit.strip()
            else:
                raise ValueError(f"Unit must be a string, got {type(unit).__name__}: {unit}")
            
            valid_units = ["count/window", "count/min", "count/s"]
            if unit not in valid_units:
                raise ValueError(f"Unit must be one of {valid_units}, got '{unit}'")
            
            parsed["unit"] = unit
            
            return parsed
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to parse input parameters: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply sample counting to the channel data."""
        
        # Input validation
        try:
            if channel is None:
                raise ValueError("Channel is None")
            if channel.xdata is None or len(channel.xdata) == 0:
                raise ValueError("Channel has no time data")
            if channel.ydata is None or len(channel.ydata) == 0:
                raise ValueError("Channel has no signal data")
            if len(channel.xdata) != len(channel.ydata):
                raise ValueError(f"Time and signal data length mismatch: {len(channel.xdata)} vs {len(channel.ydata)}")
            
            # Check for minimum data requirements
            if len(channel.xdata) < 2:
                raise ValueError("Channel must have at least 2 data points for sample counting")
            
            # Validate time data
            if np.any(np.isnan(channel.xdata)):
                raise ValueError("Time data contains NaN values")
            if np.any(np.isinf(channel.xdata)):
                raise ValueError("Time data contains infinite values")
            
            # Check if time data is monotonically increasing
            if not np.all(np.diff(channel.xdata) > 0):
                raise ValueError("Time data must be monotonically increasing")
            
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}")
        
        # Extract parameters
        window_duration = params["window"]
        overlap = params["overlap"]
        unit = params["unit"]
        
        x, y = channel.xdata, channel.ydata
        
        # Signal validation
        try:
            total_duration = x[-1] - x[0]
            if total_duration <= 0:
                raise ValueError("Signal has zero or negative duration")
            
            if window_duration > total_duration:
                raise ValueError(
                    f"Window duration ({window_duration:.3f}s) is larger than signal duration ({total_duration:.3f}s). "
                    f"Try a smaller window or use a longer signal."
                )
            
            # Check if we'll have enough windows
            step_duration = window_duration * (1 - overlap)
            estimated_windows = int((total_duration - window_duration) / step_duration) + 1
            
            if estimated_windows < 1:
                raise ValueError(
                    f"Configuration would produce no valid windows. "
                    f"Window: {window_duration}s, Overlap: {overlap*100}%, Signal: {total_duration:.3f}s"
                )
            
            if estimated_windows > 100000:  # Sanity check
                raise ValueError(
                    f"Configuration would produce too many windows ({estimated_windows}). "
                    f"Consider using a larger window or less overlap."
                )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Signal validation failed: {str(e)}")
        
        # Generate sliding windows
        try:
            step_duration = window_duration * (1 - overlap)
            
            window_starts = []
            current_time = x[0]
            
            while current_time + window_duration <= x[-1]:
                window_starts.append(current_time)
                current_time += step_duration
            
            if len(window_starts) == 0:
                raise ValueError("No valid windows could be generated")
            
            print(f"[{cls.name}] Processing {len(window_starts)} windows with {overlap*100:.1f}% overlap")
            
        except Exception as e:
            raise ValueError(f"Window generation failed: {str(e)}")
        
        # Count samples in each window
        try:
            x_output = []
            y_output = []
            
            for i, start_time in enumerate(window_starts):
                end_time = start_time + window_duration
                
                # Find samples within this time window
                mask = (x >= start_time) & (x <= end_time)
                sample_count = np.sum(mask)
                
                # Calculate center time for this window
                center_time = start_time + window_duration / 2
                
                # Convert count based on selected unit
                if unit == "count/window":
                    output_value = sample_count
                elif unit == "count/s":
                    # Extrapolate to samples per second
                    output_value = sample_count / window_duration
                elif unit == "count/min":
                    # Extrapolate to samples per minute
                    output_value = (sample_count / window_duration) * 60.0
                else:
                    raise ValueError(f"Unknown unit: {unit}")
                
                x_output.append(center_time)
                y_output.append(output_value)
            
            # Convert to numpy arrays
            x_output = np.array(x_output)
            y_output = np.array(y_output)
            
            # Validate output
            if len(x_output) == 0 or len(y_output) == 0:
                raise ValueError("No output data generated")
            
            if np.any(np.isnan(y_output)):
                raise ValueError("Output contains NaN values")
            
            if np.any(np.isinf(y_output)):
                raise ValueError("Output contains infinite values")
            
            # Log statistics
            print(f"[{cls.name}] Output statistics:")
            print(f"  - Windows processed: {len(x_output)}")
            print(f"  - Count range: {np.min(y_output):.1f} - {np.max(y_output):.1f}")
            print(f"  - Mean count: {np.mean(y_output):.1f}")
            print(f"  - Unit: {unit}")
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Sample counting computation failed: {str(e)}")
        
        # Create output channel
        try:
            # Determine appropriate ylabel based on unit
            ylabel_map = {
                "count/window": "Sample Count",
                "count/s": "Samples/Second",
                "count/min": "Samples/Minute"
            }
            ylabel = ylabel_map.get(unit, "Sample Count")
            
            # Create new channel
            new_channel = cls.create_new_channel(
                parent=channel, 
                xdata=x_output, 
                ydata=y_output, 
                params=params
            )
            
            # Set channel properties
            new_channel.xlabel = "Time (s)"
            new_channel.ylabel = ylabel
            new_channel.legend_label = f"{channel.legend_label} - Sample Count ({unit})"
            
            # Add metadata for debugging/analysis
            new_channel.metadata = {
                'original_samples': len(channel.xdata),
                'window_duration': window_duration,
                'overlap_fraction': overlap,
                'num_windows': len(x_output),
                'unit': unit,
                'mean_count': float(np.mean(y_output)),
                'std_count': float(np.std(y_output))
            }
            
            return new_channel
            
        except Exception as e:
            raise ValueError(f"Failed to create output channel: {str(e)}") 
