# steps/base_step.py

from abc import ABC, abstractmethod
from channel import Channel  # Import the Channel class
import numpy as np
from typing import Optional, Dict, Any, List


class BaseStep(ABC):
    """
    Abstract base class for all processing steps.
    Each step must implement `script()` and provide metadata like `name`, `category`, `description`, and `params`.
    """
    
    # Class attributes that subclasses should override
    name: str = "base_step"
    category: str = "Base"
    description: str = "Base processing step"
    tags: List[str] = []
    params: List[Dict[str, Any]] = []
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs  # Store parameters passed by user

    @abstractmethod
    def script(self, x: np.ndarray, y: np.ndarray, fs: Optional[float], params: dict) -> list:
        """
        Core processing logic that subclasses must implement.
        
        Args:
            x: Time/index data
            y: Signal data
            fs: Sampling frequency (may be None)
            params: Validated parameters
            
        Returns:
            list: List of dictionaries, each containing channel data with 'type', 'x', 'y' keys
                 Example: [{'type': 'main', 'x': x_data, 'y': y_data}]
        """
        pass

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel | list[Channel]:
        """
        Standard apply method that handles all common processing steps.
        Subclasses should not override this method.
        """
        try:
            step_name = cls.name.replace('_', ' ').title()
            print(f"[{step_name}] Starting apply with params: {params}")
            
            # 1. Validate channel input
            cls.validate_channel_input(channel)
            
            # 2. Extract data
            x = channel.xdata
            y = channel.ydata
            
            # Validate that we have valid data arrays
            if x is None or y is None:
                raise ValueError("Channel must have valid xdata and ydata")
            
            # 3. Validate input data with repair capability
            data_repair_info = None
            try:
                cls.validate_signal_data(x, y)
            except ValueError as e:
                print(f"[{step_name}] Data validation failed: {e}")
                print(f"[{step_name}] Attempting data repair...")
                x, y, repair_summary = cls.repair_signal_data(x, y, step_name)
                data_repair_info = repair_summary
                print(f"[{step_name}] {repair_summary}")
                print(f"[{step_name}] Retrying validation with repaired data...")
                cls.validate_signal_data(x, y)
                print(f"[{step_name}] Validation passed after repair")
            
            # 4. Get sampling frequency
            fs = cls._get_channel_fs(channel)
            
            print(f"[{step_name}] Input data shape: {y.shape if hasattr(y, 'shape') else len(y)}")
            
            # 5. Validate parameters (declarative + custom)
            try:
                validate_method = getattr(cls, 'validate_parameters', None)
                if validate_method and callable(validate_method):
                    validate_method(params)
            except AttributeError:
                pass  # No custom validation method
            
            # 6. Call core processing logic
            # Create a temporary instance to call the script method
            temp_instance = cls()
            channels_data = temp_instance.script(x, y, fs, params)
            
            # 7. Process each channel in the result
            created_channels = []
            for i, channel_info in enumerate(channels_data):
                # Validate channel structure first
                cls.validate_channel_structure(channel_info, i)
                
                tags = channel_info['tags']
                # Use the first tag as the channel type, or 'main' as default
                channel_type = tags[0] if tags else 'main'
                
                # Extract data based on channel type
                if channel_type == 'spectrogram':
                    # Spectrogram channels use 't', 'f', 'z' fields
                    x_data = channel_info['t']  # Time axis
                    y_data = channel_info['f']  # Frequency axis
                    z_data = channel_info['z']  # Spectrogram data
                else:
                    # Regular channels use 'x', 'y' fields
                    x_data = channel_info['x']
                    y_data = channel_info['y']
                    z_data = None
                
                # Validate output data for this channel - generous validation
                if channel_type == 'spectrogram':
                    # Spectrogram validation: just check that data exists and is not empty
                    cls.validate_spectrogram_data(x_data, y_data, z_data)
                else:
                    # Regular time-series validation: check x/y length match
                    cls.validate_output_data(y, y_data, x_data=x_data)
                
                print(f"[{step_name}] Channel {i+1} ({channel_type}) data shape: {y_data.shape if hasattr(y_data, 'shape') else len(y_data)}")
                
                # 8. Generate channel suffix automatically
                suffix = cls._generate_channel_suffix(params)
                if len(channels_data) > 1:
                    suffix = f"{suffix}_{channel_type}"
                
                print(f"[{step_name}] Creating {channel_type} channel with suffix: {suffix}")
                
                # 9. Create channel based on type
                # Extract tags from channel_info if available
                channel_tags = channel_info.get('tags', None)
                
                if channel_type == 'time-series':
                    new_channel = cls.create_new_channel(
                        parent=channel, 
                        xdata=x_data, 
                        ydata=y_data, 
                        params=params,
                        suffix=suffix,
                        channel_tags=channel_tags
                    )
              
                elif channel_type == 'spectrogram':
                    # Handle spectrogram-specific properties
                    new_channel = cls.create_new_channel(
                        parent=channel, 
                        xdata=x_data,      # Time axis (t)
                        ydata=y_data,      # Frequency axis (f)
                        params=params,
                        suffix=suffix,
                        channel_tags=channel_tags
                    )
                    
                    # Add spectrogram data to metadata
                    if z_data is not None:
                        new_channel.metadata = {'Zxx': z_data}
                    else:
                        new_channel.metadata = {}
                else:
                    # Default channel creation
                    new_channel = cls.create_new_channel(
                        parent=channel, 
                        xdata=x_data, 
                        ydata=y_data, 
                        params=params,
                        suffix=suffix,
                        channel_tags=channel_tags
                    )
                
                # Add data repair information to channel metadata if repairs were made
                if data_repair_info is not None:
                    if not hasattr(new_channel, 'metadata') or new_channel.metadata is None:
                        new_channel.metadata = {}
                    new_channel.metadata['data_repair_info'] = data_repair_info
                
                created_channels.append(new_channel)
            
            # Configure marker properties for detection steps
            if 'marker' in cls.tags:
                for new_channel in created_channels:
                    # Set marker properties for detection steps
                    new_channel.marker = 'o'  # Circle marker
                    new_channel.style = 'none'  # No line connecting points (matplotlib expects 'none')
                    new_channel.color = '#ff4444'  # Red color for detection points
            
            # Return single channel if only one, otherwise return list
            if len(created_channels) == 1:
                return created_channels[0]
            else:
                return created_channels
            
        except Exception as e:
            step_name = cls.name.replace('_', ' ').title()
            print(f"[{step_name}] Error in apply: {e}")
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"{step_name} processing failed: {str(e)}")

    @classmethod
    def _generate_channel_suffix(cls, params: dict) -> str:
        """
        Generate an appropriate suffix for the new channel based on parameters.
        Subclasses can override this for custom naming.
        """
        # Try to generate smart suffix based on common parameter patterns
        
        # Handle constant addition/subtraction
        if 'constant' in params and params['constant'] is not None:
            constant = float(params['constant'])
            if constant >= 0:
                return f"Plus{constant:g}"
            else:
                return f"Minus{abs(constant):g}"
        
        # Handle window-based operations
        if 'window' in params and params['window'] is not None:
            window = params['window']
            # Handle both numeric and string window types
            try:
                window_int = int(window)
                if cls.name.startswith('moving_'):
                    operation = cls.name.replace('moving_', '').upper()
                    return f"{operation}{window_int}"
                else:
                    return f"Win{window_int}"
            except (ValueError, TypeError):
                # Handle string window types (like 'hann', 'hamming', etc.)
                if cls.name.startswith('moving_'):
                    operation = cls.name.replace('moving_', '').upper()
                    return f"{operation}{window.capitalize()}"
                else:
                    return f"Win{window.capitalize()}"
        
        # Handle frequency-based operations
        if 'low_freq' in params and 'high_freq' in params and params['low_freq'] is not None and params['high_freq'] is not None:
            low = float(params['low_freq'])
            high = float(params['high_freq'])
            return f"BP{low:g}-{high:g}Hz"
        elif 'frequency' in params and params['frequency'] is not None:
            freq = float(params['frequency'])
            if 'lowpass' in cls.name:
                return f"LP{freq:g}Hz"
            elif 'highpass' in cls.name:
                return f"HP{freq:g}Hz"
            else:
                return f"F{freq:g}Hz"
        
        # Handle threshold operations
        if 'threshold' in params and params['threshold'] is not None:
            thresh = float(params['threshold'])
            return f"Thresh{thresh:g}"
        
        # Handle method-based operations
        if 'method' in params and params['method'] is not None:
            method = str(params['method'])
            return f"{method.capitalize()}"
        
        # Default: use step name
        return cls.name.replace('_step', '').replace('_', ' ').title()

    def get_info(self) -> Dict[str, Any]:
        """
        Return all metadata about the step.
        Useful for GUI display or parameter prompting.
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "tags": self.tags,
            "params": self.params
        }

    @classmethod
    def get_prompt(cls) -> Dict[str, Any]:
        """
        Return prompt information for GUI parameter collection.
        Standard method used by ProcessWizardManager.
        """
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            name = param["name"]
            if name == "fs":
                # Skip parsing fs - it will be injected from parent channel
                continue
            value = user_input.get(name, param.get("default"))
            
            # Handle type conversion based on parameter type
            param_type = param.get("type", "str")
            try:
                if param_type == "float":
                    parsed[name] = float(value) if value is not None else None
                elif param_type == "int":
                    parsed[name] = int(value) if value is not None else None
                elif param_type in ["bool", "boolean"]:
                    if isinstance(value, bool):
                        parsed[name] = value
                    elif isinstance(value, str):
                        parsed[name] = value.lower() in ['true', '1', 'yes', 'on']
                    else:
                        parsed[name] = bool(value)
                else:
                    parsed[name] = value
            except (ValueError, TypeError):
                # If conversion fails, use default or keep as string
                parsed[name] = param.get("default", value)
        return parsed

    @classmethod
    def _inject_fs_if_needed(cls, channel: Channel, params: dict, func) -> dict:
        """
        Automatically inject fs from parent channel if the processing function requires it.
        """
        if "fs" in func.__code__.co_varnames:
            fs = cls._get_channel_fs(channel)
            if fs is not None:
                params["fs"] = fs
                print(f"[{cls.name}Step] Injected fs={fs:.2f} from parent channel")
            else:
                print(f"[{cls.name}Step] Warning: Could not extract sampling frequency from channel")
        return params

    @classmethod
    def _get_channel_fs(cls, channel: Channel) -> Optional[float]:
        """
        Extract sampling frequency from a channel using the correct attribute hierarchy.
        
        Args:
            channel: Channel object to extract fs from
            
        Returns:
            float: Sampling frequency in Hz, or None if not available
        """
        # Priority 1: Use fs_median (primary source in Channel class)
        fs = getattr(channel, 'fs_median', None)
        if fs is not None and fs > 0:
            return fs
        
        # Priority 2: Fallback to fs attribute (backwards compatibility)
        fs = getattr(channel, 'fs', None)
        if fs is not None and fs > 0:
            return fs
        
        # Priority 3: Try to calculate from sampling_stats
        if hasattr(channel, 'sampling_stats') and channel.sampling_stats:
            fs = getattr(channel.sampling_stats, 'median_fs', None)
            if fs is not None and fs > 0:
                return fs
        
        # Priority 4: Try to calculate from time data if available
        if hasattr(channel, 'xdata') and channel.xdata is not None and len(channel.xdata) > 1:
            try:
                time_diffs = np.diff(channel.xdata)
                valid_diffs = time_diffs[time_diffs > 0]
                if len(valid_diffs) > 0:
                    median_dt = np.median(valid_diffs)
                    if median_dt > 0:
                        return 1.0 / median_dt
            except Exception as e:
                print(f"[{cls.name}Step] Error calculating fs from time data: {e}")
        
        return None

    @classmethod
    def create_new_channel(cls, parent: Channel, xdata: np.ndarray, ydata: np.ndarray, params: dict, suffix: Optional[str] = None, channel_tags: Optional[list] = None) -> Channel:
        """
        Helper method to create a new channel with consistent parameter handling.
        
        Args:
            parent: Parent channel to inherit properties from
            xdata: Time/index data for the new channel
            ydata: Signal data for the new channel
            params: Parameters used for processing
            suffix: Optional suffix for the channel name (defaults to step name)
            channel_tags: Optional tags for this specific channel (overrides cls.tags)
        """
        # Use provided suffix or default to step name
        name_suffix = suffix if suffix else cls.name
        
        # Use channel-specific tags if provided, otherwise use step class tags
        tags_to_use = channel_tags if channel_tags is not None else cls.tags
        
        # Debug output for channel creation
        final_legend_label = f"{parent.legend_label} - {name_suffix}"
        print(f"[{cls.name}] DEBUG: Creating new channel with:")
        print(f"[{cls.name}] DEBUG:   - Parent legend_label: '{parent.legend_label}'")
        print(f"[{cls.name}] DEBUG:   - Name suffix: '{name_suffix}'")
        print(f"[{cls.name}] DEBUG:   - Final legend_label: '{final_legend_label}'")
        print(f"[{cls.name}] DEBUG:   - Tags: {tags_to_use}")
        print(f"[{cls.name}] DEBUG:   - xdata shape: {xdata.shape}")
        print(f"[{cls.name}] DEBUG:   - ydata shape: {ydata.shape}")
        print(f"[{cls.name}] DEBUG:   - ydata range: [{np.min(ydata):.3f}, {np.max(ydata):.3f}]")
        
        new_channel = Channel.from_parent(
            parent=parent,
            xdata=xdata,
            ydata=ydata,
            legend_label=final_legend_label,
            description=cls.description,
            tags=tags_to_use,
            params=params  # Pass the parameters to the new channel
        )
        
        print(f"[{cls.name}] DEBUG: New channel created successfully with legend_label: '{new_channel.legend_label}'")
        return new_channel

    # ============================================================================
    # VALIDATION HELPER METHODS
    # ============================================================================

    @classmethod
    def validate_channel_input(cls, channel) -> None:
        """
        Validate that the input channel is valid for processing.
        
        Args:
            channel: Input channel object to validate
            
        Raises:
            ValueError: If channel is invalid
        """
        if channel is None:
            raise ValueError("Input channel cannot be None")
        
        if not hasattr(channel, 'ydata') or channel.ydata is None:
            raise ValueError("Channel must have valid signal data (ydata)")
        
        if not hasattr(channel, 'xdata') or channel.xdata is None:
            raise ValueError("Channel must have valid time data (xdata)")

    @classmethod
    def validate_signal_data(cls, xdata, ydata) -> None:
        """
        Validate signal data arrays.
        
        Args:
            xdata: Time/index array
            ydata: Signal values array
            
        Raises:
            ValueError: If data is invalid
        """
        # Check for empty data
        if len(ydata) == 0:
            raise ValueError("Signal data cannot be empty")
        
        if len(xdata) == 0:
            raise ValueError("Time data cannot be empty")
        
        # Check length mismatch
        if len(xdata) != len(ydata):
            raise ValueError(f"Time and signal data length mismatch: {len(xdata)} vs {len(ydata)}")
        
        # Check for all NaN values
        if np.all(np.isnan(ydata)):
            raise ValueError("Signal contains only NaN values")
        
        # Check for any NaN values (triggers repair)
        if np.any(np.isnan(ydata)):
            raise ValueError("Signal contains NaN values")
        
        # Check for all infinite values
        if np.all(np.isinf(ydata)):
            raise ValueError("Signal contains only infinite values")
        
        # Check for any infinite values (triggers repair)
        if np.any(np.isinf(ydata)):
            raise ValueError("Signal contains infinite values")
        
        # Check time data validity
        if np.any(np.isnan(xdata)):
            raise ValueError("Time data contains NaN values")
        
        if np.any(np.isinf(xdata)):
            raise ValueError("Time data contains infinite values")

    @classmethod
    def repair_signal_data(cls, xdata: np.ndarray, ydata: np.ndarray, step_name: str) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Attempts to repair invalid signal data using the comprehensive approach.
        
        Args:
            xdata: Time/index data
            ydata: Signal data
            step_name: Name of the step that encountered the validation error
            
        Returns:
            tuple: (repaired_xdata, repaired_ydata, repair_summary)
        """
        print(f"[{step_name}] === DEBUG: Starting data repair ===")
        print(f"[{step_name}] Original xdata: shape={xdata.shape}, dtype={xdata.dtype}")
        print(f"[{step_name}] Original xdata range: [{np.min(xdata) if len(xdata) > 0 else 'empty'}] to [{np.max(xdata) if len(xdata) > 0 else 'empty'}]")
        print(f"[{step_name}] Original xdata stats: NaN={np.sum(np.isnan(xdata))}, Inf={np.sum(np.isinf(xdata))}, Finite={np.sum(np.isfinite(xdata))}")
        print(f"[{step_name}] Original ydata: shape={ydata.shape}, dtype={ydata.dtype}")
        print(f"[{step_name}] Original ydata range: [{np.min(ydata) if len(ydata) > 0 else 'empty'}] to [{np.max(ydata) if len(ydata) > 0 else 'empty'}]")
        print(f"[{step_name}] Original ydata stats: NaN={np.sum(np.isnan(ydata))}, Inf={np.sum(np.isinf(ydata))}, Finite={np.sum(np.isfinite(ydata))}")
        
        repaired_xdata = np.copy(xdata)
        repaired_ydata = np.copy(ydata)
        repairs_made = []
        original_length = len(ydata)
        
        # 1. Handle length mismatch first
        if len(xdata) != len(ydata):
            min_len = min(len(xdata), len(ydata))
            repaired_xdata = repaired_xdata[:min_len]
            repaired_ydata = repaired_ydata[:min_len]
            repairs_made.append(f"Length mismatch: x={len(xdata)}, y={len(ydata)} → truncated to {min_len} samples")
            print(f"[{step_name}] - Length mismatch detected: x={len(xdata)}, y={len(ydata)} → truncated to {min_len} samples")
            print(f"[{step_name}] DEBUG: After length fix - xdata.shape={repaired_xdata.shape}, ydata.shape={repaired_ydata.shape}")

        # 2. Handle time data issues (critical - delete problematic points)
        time_issues_mask = np.isnan(repaired_xdata) | np.isinf(repaired_xdata)
        if np.any(time_issues_mask):
            num_time_issues = np.sum(time_issues_mask)
            print(f"[{step_name}] DEBUG: Time issues mask: {time_issues_mask}")
            print(f"[{step_name}] DEBUG: Invalid time indices: {np.where(time_issues_mask)[0]}")
            valid_mask = ~time_issues_mask
            
            if np.any(valid_mask):  # Some valid time points remain
                print(f"[{step_name}] DEBUG: Before time repair - xdata shape: {repaired_xdata.shape}, ydata shape: {repaired_ydata.shape}")
                repaired_xdata = repaired_xdata[valid_mask]
                repaired_ydata = repaired_ydata[valid_mask]
                repairs_made.append(f"Removed {num_time_issues} time points with NaN/inf values")
                print(f"[{step_name}] - Found {num_time_issues} invalid time values → removed corresponding points")
                print(f"[{step_name}] DEBUG: After time repair - xdata shape: {repaired_xdata.shape}, ydata shape: {repaired_ydata.shape}")
                print(f"[{step_name}] DEBUG: After time repair - xdata range: [{np.min(repaired_xdata):.3f}, {np.max(repaired_xdata):.3f}]")
            else:
                raise ValueError("All time data points are invalid (NaN or infinite) - cannot repair")

        # 3. Handle signal NaN values (interpolate)
        nan_mask = np.isnan(repaired_ydata)
        if np.any(nan_mask):
            num_nan = np.sum(nan_mask)
            print(f"[{step_name}] DEBUG: NaN mask: {nan_mask}")
            print(f"[{step_name}] DEBUG: NaN indices: {np.where(nan_mask)[0]}")
            print(f"[{step_name}] DEBUG: Before NaN repair - ydata: {repaired_ydata}")
            
            if np.all(nan_mask):
                raise ValueError("All signal values are NaN - cannot repair")
            
            nan_percentage = (num_nan / len(repaired_ydata)) * 100
            if nan_percentage > 50:
                raise ValueError(f"Too many NaN values in signal ({nan_percentage:.1f}%) - data quality too poor for repair")
            
            # Interpolate over NaN values
            valid_indices = np.where(~nan_mask)[0]
            valid_values = repaired_ydata[~nan_mask]
            nan_indices = np.where(nan_mask)[0]
            print(f"[{step_name}] DEBUG: Valid indices: {valid_indices}, Valid values: {valid_values}")
            print(f"[{step_name}] DEBUG: NaN indices for interpolation: {nan_indices}")
            
            if len(valid_indices) > 1:
                # Use linear interpolation for small gaps, or clamp to nearest for edge cases
                interpolated_values = np.interp(nan_indices, valid_indices, valid_values)
                print(f"[{step_name}] DEBUG: Interpolated values: {interpolated_values}")
                repaired_ydata[nan_mask] = interpolated_values
                repairs_made.append(f"Interpolated {num_nan} NaN values ({nan_percentage:.1f}% of signal)")
                print(f"[{step_name}] - Found {num_nan} NaN values → interpolated using linear method")
                print(f"[{step_name}] DEBUG: After NaN repair - ydata: {repaired_ydata}")
            else:
                raise ValueError("Not enough valid signal points for interpolation")

        # 4. Handle signal infinite values (clip)
        inf_mask = np.isinf(repaired_ydata)
        if np.any(inf_mask):
            num_inf = np.sum(inf_mask)
            print(f"[{step_name}] DEBUG: Inf mask: {inf_mask}")
            print(f"[{step_name}] DEBUG: Inf indices: {np.where(inf_mask)[0]}")
            print(f"[{step_name}] DEBUG: Before inf repair - ydata: {repaired_ydata}")
            
            if np.all(inf_mask):
                raise ValueError("All signal values are infinite - cannot repair")
            
            inf_percentage = (num_inf / len(repaired_ydata)) * 100
            if inf_percentage > 50:
                raise ValueError(f"Too many infinite values in signal ({inf_percentage:.1f}%) - data quality too poor for repair")
            
            # Calculate clipping bounds from finite values
            finite_values = repaired_ydata[np.isfinite(repaired_ydata)]
            print(f"[{step_name}] DEBUG: Finite values for clipping bounds: {finite_values}")
            
            if len(finite_values) > 0:
                # Use 99th percentile as clipping bounds
                upper_bound = np.percentile(finite_values, 99)
                lower_bound = np.percentile(finite_values, 1)
                print(f"[{step_name}] DEBUG: Clipping bounds: [{lower_bound:.3g}, {upper_bound:.3g}]")
                
                # Clip infinite values
                pos_inf_mask = np.isposinf(repaired_ydata)
                neg_inf_mask = np.isneginf(repaired_ydata)
                print(f"[{step_name}] DEBUG: Positive inf indices: {np.where(pos_inf_mask)[0]}")
                print(f"[{step_name}] DEBUG: Negative inf indices: {np.where(neg_inf_mask)[0]}")
                
                repaired_ydata[pos_inf_mask] = upper_bound
                repaired_ydata[neg_inf_mask] = lower_bound
                
                repairs_made.append(f"Clipped {num_inf} infinite values to bounds [{lower_bound:.3g}, {upper_bound:.3g}]")
                print(f"[{step_name}] - Found {num_inf} infinite values → clipped to ±{upper_bound:.3g} (99th percentile)")
                print(f"[{step_name}] DEBUG: After inf repair - ydata: {repaired_ydata}")
            else:
                raise ValueError("No finite values available for clipping bounds")

        # 5. Final check - ensure we still have some data
        if len(repaired_ydata) == 0:
            raise ValueError("Data repair resulted in empty arrays")
        
        # Calculate data preservation statistics
        final_length = len(repaired_ydata)
        preservation_percentage = (final_length / original_length) * 100
        
        # Generate summary
        if repairs_made:
            repair_summary = f"Repair complete: {', '.join(repairs_made)} - {preservation_percentage:.1f}% of original data preserved"
            if preservation_percentage < 80:
                print(f"[{step_name}] Warning: Only {preservation_percentage:.1f}% of original data preserved after repair")
        else:
            repair_summary = "No repairs needed"
        
        # Final debug output
        print(f"[{step_name}] === DEBUG: Final repaired data ===")
        print(f"[{step_name}] Final xdata: shape={repaired_xdata.shape}, dtype={repaired_xdata.dtype}")
        print(f"[{step_name}] Final xdata range: [{np.min(repaired_xdata):.3f}, {np.max(repaired_xdata):.3f}]")
        print(f"[{step_name}] Final xdata: {repaired_xdata}")
        print(f"[{step_name}] Final ydata: shape={repaired_ydata.shape}, dtype={repaired_ydata.dtype}")
        print(f"[{step_name}] Final ydata range: [{np.min(repaired_ydata):.3f}, {np.max(repaired_ydata):.3f}]")
        print(f"[{step_name}] Final ydata: {repaired_ydata}")
        print(f"[{step_name}] Final data stats: NaN={np.sum(np.isnan(repaired_ydata))}, Inf={np.sum(np.isinf(repaired_ydata))}, Finite={np.sum(np.isfinite(repaired_ydata))}")
        print(f"[{step_name}] Data preservation: {preservation_percentage:.1f}% ({final_length}/{original_length} samples)")
        print(f"[{step_name}] === DEBUG: Repair completed ===")
        
        return repaired_xdata, repaired_ydata, repair_summary

    @classmethod
    def validate_numeric_parameter(cls, param_name: str, value: Any, 
                                 min_val: Optional[float] = None,
                                 max_val: Optional[float] = None,
                                 allow_nan: bool = False,
                                 allow_inf: bool = False) -> float:
        """
        Validate and convert a numeric parameter.
        
        Args:
            param_name: Name of the parameter for error messages
            value: Input value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            
        Returns:
            float: Validated numeric value
            
        Raises:
            ValueError: If parameter is invalid
        """
        # Handle string input
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError(f"{param_name} parameter cannot be empty")
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"{param_name} must be a valid number, got '{value}'")
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            raise ValueError(f"{param_name} must be a number, got {type(value).__name__}: {value}")
        
        # Check for NaN
        if np.isnan(value) and not allow_nan:
            raise ValueError(f"{param_name} cannot be NaN")
        
        # Check for infinity
        if np.isinf(value) and not allow_inf:
            raise ValueError(f"{param_name} cannot be infinite")
        
        # Check range
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
        
        return value

    @classmethod
    def validate_integer_parameter(cls, param_name: str, value: Any,
                                 min_val: Optional[int] = None,
                                 max_val: Optional[int] = None) -> int:
        """
        Validate and convert an integer parameter.
        
        Args:
            param_name: Name of the parameter for error messages
            value: Input value to validate
            min_val: Minimum allowed value (inclusive)
            max_val: Maximum allowed value (inclusive)
            
        Returns:
            int: Validated integer value
            
        Raises:
            ValueError: If parameter is invalid
        """
        # Handle string input
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError(f"{param_name} parameter cannot be empty")
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"{param_name} must be a valid integer, got '{value}'")
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(f"{param_name} must be an integer, got {value}")
            value = int(value)
        else:
            raise ValueError(f"{param_name} must be an integer, got {type(value).__name__}: {value}")
        
        # Check range
        if min_val is not None and value < min_val:
            raise ValueError(f"{param_name} must be >= {min_val}, got {value}")
        
        if max_val is not None and value > max_val:
            raise ValueError(f"{param_name} must be <= {max_val}, got {value}")
        
        return value

    @classmethod
    def validate_string_parameter(cls, param_name: str, value: Any,
                                valid_options: Optional[List[str]] = None,
                                allow_empty: bool = False) -> str:
        """
        Validate a string parameter.
        
        Args:
            param_name: Name of the parameter for error messages
            value: Input value to validate
            valid_options: List of valid string options
            allow_empty: Whether to allow empty strings
            
        Returns:
            str: Validated string value
            
        Raises:
            ValueError: If parameter is invalid
        """
        if not isinstance(value, str):
            raise ValueError(f"{param_name} must be a string, got {type(value).__name__}: {value}")
        
        value = value.strip()
        
        if not value and not allow_empty:
            raise ValueError(f"{param_name} parameter cannot be empty")
        
        if valid_options is not None and value not in valid_options:
            raise ValueError(f"{param_name} must be one of {valid_options}, got '{value}'")
        
        return value

    @classmethod
    def validate_output_data(cls, y_input: np.ndarray, y_output: np.ndarray,
                           x_data: np.ndarray = None, **kwargs) -> None:
        """
        Generous validation: only check if data is fundamentally unplottable.
        
        Args:
            y_input: Original input signal (unused, kept for compatibility)
            y_output: Processed output signal  
            x_data: Optional x-axis data to check length match
            **kwargs: Other unused args kept for compatibility
            
        Raises:
            ValueError: Only if data cannot be plotted by matplotlib
        """
        # Check for None output
        if y_output is None:
            raise ValueError("Processing produced no output")
        
        # Convert to numpy array for consistent handling
        try:
            y_output = np.asarray(y_output)
        except Exception as e:
            raise ValueError(f"Output data cannot be converted to array: {e}")
        
        # Check for empty output
        if len(y_output) == 0:
            raise ValueError("Processing produced empty output")
        
        # Check x/y dimension mismatch (critical for plotting)
        if x_data is not None:
            if len(x_data) != len(y_output):
                raise ValueError(f"x and y data have different lengths: {len(x_data)} vs {len(y_output)}")
        
        # That's it! Everything else is matplotlib's problem.
        # Let the plotting system handle NaN, inf, weird ranges, etc.

    @classmethod
    def validate_spectrogram_data(cls, t_data: np.ndarray, f_data: np.ndarray, z_data: np.ndarray) -> None:
        """
        Validate spectrogram data structure (t, f, z format).
        
        Args:
            t_data: Time axis data
            f_data: Frequency axis data  
            z_data: Spectrogram matrix data
            
        Raises:
            ValueError: Only if data cannot be used for spectrogram plotting
        """
        # Check for None data
        if t_data is None:
            raise ValueError("Spectrogram time axis cannot be None")
        if f_data is None:
            raise ValueError("Spectrogram frequency axis cannot be None")  
        if z_data is None:
            raise ValueError("Spectrogram matrix data cannot be None")
        
        # Convert to numpy arrays
        try:
            t_data = np.asarray(t_data)
            f_data = np.asarray(f_data)
            z_data = np.asarray(z_data)
        except Exception as e:
            raise ValueError(f"Spectrogram data cannot be converted to arrays: {e}")
        
        # Check for empty data
        if len(t_data) == 0:
            raise ValueError("Spectrogram time axis is empty")
        if len(f_data) == 0:
            raise ValueError("Spectrogram frequency axis is empty")
        if z_data.size == 0:
            raise ValueError("Spectrogram matrix is empty")
        
        # Check matrix dimensions match axes (if 2D)
        if z_data.ndim == 2:
            if z_data.shape[1] != len(t_data):
                raise ValueError(f"Spectrogram matrix time dimension ({z_data.shape[1]}) doesn't match time axis ({len(t_data)})")
            if z_data.shape[0] != len(f_data):
                raise ValueError(f"Spectrogram matrix frequency dimension ({z_data.shape[0]}) doesn't match frequency axis ({len(f_data)})")
        
        # That's it! Matplotlib can handle the rest.

    @classmethod
    def validate_channel_structure(cls, channel_info: dict, channel_index: int) -> None:
        """
        Validate the structure of a channel info dictionary.
        
        Args:
            channel_info: Dictionary containing channel data
            channel_index: Index of the channel for error messages
            
        Raises:
            ValueError: If channel structure is invalid
        """
        # Check required fields
        required_fields = ['tags']
        for field in required_fields:
            if field not in channel_info:
                raise ValueError(f"Channel {channel_index + 1} missing required field '{field}'")
        
        # Validate tags field
        tags = channel_info['tags']
        if not isinstance(tags, list):
            raise ValueError(f"Channel {channel_index + 1} 'tags' must be a list, got {type(tags)}")
        if len(tags) == 0:
            raise ValueError(f"Channel {channel_index + 1} 'tags' cannot be empty")
        
        # Check if this is a spectrogram channel
        is_spectrogram = 'spectrogram' in tags
        
        if is_spectrogram:
            # Spectrogram channels require 't', 'f', and 'z' fields
            spectrogram_fields = ['t', 'f', 'z']
            for field in spectrogram_fields:
                if field not in channel_info:
                    raise ValueError(f"Spectrogram channel {channel_index + 1} missing '{field}' field")
            
            # Validate spectrogram data fields
            t_data = channel_info['t']
            f_data = channel_info['f']
            z_data = channel_info['z']
            
            if t_data is None:
                raise ValueError(f"Spectrogram channel {channel_index + 1} 't' data cannot be None")
            if f_data is None:
                raise ValueError(f"Spectrogram channel {channel_index + 1} 'f' data cannot be None")
            if z_data is None:
                raise ValueError(f"Spectrogram channel {channel_index + 1} 'z' data cannot be None")
        else:
            # Regular channels require 'x' and 'y' fields
            regular_fields = ['x', 'y']
            for field in regular_fields:
                if field not in channel_info:
                    raise ValueError(f"Channel {channel_index + 1} missing required field '{field}'")
            
            # Validate regular data fields
            x_data = channel_info['x']
            y_data = channel_info['y']
            
            if x_data is None:
                raise ValueError(f"Channel {channel_index + 1} 'x' data cannot be None")
            if y_data is None:
                raise ValueError(f"Channel {channel_index + 1} 'y' data cannot be None")

    @classmethod
    def validate_window_parameter(cls, window: int, signal_length: int,
                                min_window: int = 3) -> int:
        """
        Validate a window size parameter.
        
        Args:
            window: Window size to validate
            signal_length: Length of the signal being processed
            min_window: Minimum allowed window size
            
        Returns:
            int: Validated window size
            
        Raises:
            ValueError: If window size is invalid
        """
        if window < min_window:
            raise ValueError(f"Window size must be >= {min_window}, got {window}")
        
        if window > signal_length:
            raise ValueError(f"Window size ({window}) cannot be larger than signal length ({signal_length})")
        
        # Warn if window is very large relative to signal
        if window > signal_length * 0.5:
            import warnings
            warnings.warn(f"Window size ({window}) is more than 50% of signal length ({signal_length})")
        
        return window

    @classmethod
    def validate_frequency_parameter(cls, frequency: float, sampling_rate: float,
                                  param_name: str = "frequency") -> float:
        """
        Validate a frequency parameter against sampling rate.
        
        Args:
            frequency: Frequency value to validate
            sampling_rate: Sampling rate of the signal
            param_name: Name of the parameter for error messages
            
        Returns:
            float: Validated frequency value
            
        Raises:
            ValueError: If frequency is invalid
        """
        if frequency <= 0:
            raise ValueError(f"{param_name} must be positive, got {frequency}")
        
        nyquist = sampling_rate / 2
        if frequency >= nyquist:
            raise ValueError(f"{param_name} ({frequency} Hz) must be less than Nyquist frequency ({nyquist:.1f} Hz)")
        
        return frequency
