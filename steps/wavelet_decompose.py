import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

import pywt

@register_step
class wavelet_decompose_step(BaseStep):
    name = "wavelet_decompose"
    category = "pywt"
    description = "Decomposes a signal into separate wavelet levels, creating multiple channels."
    tags = ["time-series", "wavelet", "decomposition", "pywt", "multiresolution", "levels"]
    params = [
        { 
            "name": "wavelet", 
            "type": "str", 
            "default": "db4", 
            "help": "Wavelet type for decomposition",
            "options": ["db1", "db2", "db4", "db8", "db10", "haar", "sym4", "sym5", "sym8", "coif2", "coif4", "coif6", "bior2.2", "bior4.4", "dmey"]
        },
        { 
            "name": "level", 
            "type": "int", 
            "default": "", 
            "help": "Decomposition level (leave blank for auto)"
        },
        { 
            "name": "decompose_levels", 
            "type": "str", 
            "default": "1,2", 
            "help": "Comma-separated levels to decompose (e.g., '1,2,3')"
        },
        { "name": "fs", "type": "float", "default": "", "help": "Sampling frequency (injected from parent channel)" }
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return { "info": cls.description, "params": cls.params }
    
    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if len(y) < 4:
            raise ValueError("Signal too short for wavelet decomposition (minimum 4 samples)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate cross-field logic and business rules"""
        # Check if decompose_levels has duplicate values
        decompose_levels = params.get("decompose_levels", [])
        if len(set(decompose_levels)) != len(decompose_levels):
            raise ValueError("Duplicate levels are not allowed in decompose_levels")
        
        # Check if level is specified and decompose_levels contains levels beyond it
        level = params.get("level")
        if level is not None:
            max_requested_level = max(decompose_levels) if decompose_levels else 0
            if max_requested_level >= level:
                raise ValueError(f"Requested levels {decompose_levels} exceed or equal specified decomposition level {level}")

    @classmethod
    def _validate_output(cls, y_original: np.ndarray, y_processed_list: list) -> None:
        """Validate processed output data"""
        if not y_processed_list:
            raise ValueError("No output channels were generated")
        
        for i, y_processed in enumerate(y_processed_list):
            if len(y_processed) != len(y_original):
                raise ValueError(f"Output length mismatch for level {i}: expected {len(y_original)}, got {len(y_processed)}")
            
            if np.all(np.isnan(y_processed)):
                raise ValueError(f"Processing produced only NaN values for level {i}")
            
            if np.all(np.isinf(y_processed)):
                raise ValueError(f"Processing produced only infinite values for level {i}")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        for param in cls.params:
            name = param["name"]
            if name == "fs": continue
            
            val = user_input.get(name, param["default"])
            
            try:
                if val == "": 
                    parsed[name] = None
                elif param["type"] == "float": 
                    parsed[name] = float(val)
                elif param["type"] == "int": 
                    parsed_val = int(val)
                    if name == "level" and parsed_val < 1:
                        raise ValueError("Decomposition level must be at least 1")
                    parsed[name] = parsed_val
                else: 
                    if name == "decompose_levels":
                        # Parse and validate decompose_levels format
                        try:
                            levels = [int(l.strip()) for l in str(val).split(',')]
                            if not levels:
                                raise ValueError("At least one level must be specified")
                            if any(l < 0 for l in levels):
                                raise ValueError("Levels must be non-negative")
                        except ValueError as e:
                            if "invalid literal" in str(e):
                                raise ValueError("decompose_levels must be comma-separated integers")
                            raise e
                        parsed[name] = levels
                    else:
                        parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
                
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        # Validate input data and parameters
        cls._validate_input_data(channel.ydata)
        cls._validate_parameters(params)
        
        try:
            params = cls._inject_fs_if_needed(channel, params, cls.script)
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', None)
            
            # Get processed data from script method
            processed_data = cls.script(x, y, fs, params)
            
            # Validate output
            cls._validate_output(y, processed_data)
            
            # Create multiple channels, one for each decomposition level
            new_channels = []
            for i, y_new in enumerate(processed_data):
                x_new = np.linspace(x[0], x[-1], len(y_new))
                # Add level information to the channel name/params
                level_params = params.copy()
                level_params['level_extracted'] = i
                new_channel = cls.create_new_channel(
                    parent=channel, 
                    xdata=x_new, 
                    ydata=y_new, 
                    params=level_params
                )
                # Modify the name to indicate which level this is
                if hasattr(new_channel, 'name'):
                    new_channel.name = f"{new_channel.name}_level_{i}"
                new_channels.append(new_channel)
            
            if not new_channels:
                raise ValueError("No channels were created from decomposition")
            
            return new_channels
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Wavelet decomposition failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list[np.ndarray]:
        wavelet = params["wavelet"]
        level = params.get("level")
        decompose_levels = params["decompose_levels"]
        
        # Perform wavelet decomposition
        try:
            coeffs = pywt.wavedec(y, wavelet, mode='periodization', level=level)
        except Exception as e:
            raise ValueError(f"Wavelet decomposition failed: {str(e)}")
        
        # Validate that requested levels exist
        max_available_level = len(coeffs) - 1
        invalid_levels = [l for l in decompose_levels if l >= len(coeffs)]
        if invalid_levels:
            raise ValueError(f"Requested levels {invalid_levels} exceed maximum available level {max_available_level}")
        
        # Create reconstructed signals for each requested level
        reconstructed_levels = []
        for level_idx in decompose_levels:
            if level_idx < len(coeffs):
                try:
                    # Create coefficients array with only this level
                    level_coeffs = [np.zeros_like(c) for c in coeffs]
                    level_coeffs[level_idx] = coeffs[level_idx]
                    # Reconstruct from this level only
                    reconstructed = pywt.waverec(level_coeffs, wavelet, mode='periodization')[:len(y)]
                    reconstructed_levels.append((level_idx, reconstructed))
                except Exception as e:
                    raise ValueError(f"Failed to reconstruct level {level_idx}: {str(e)}")
        
        if not reconstructed_levels:
            raise ValueError("No valid decomposition levels could be reconstructed")
        
        # Extract just the processed arrays (without level indices)
        y_new_arrays = [y_new for level_idx, y_new in reconstructed_levels]
        
        return y_new_arrays 
