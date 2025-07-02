import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

import pywt

def wavelet_decompose(y, wavelet='db4', level=None, decompose_levels='1,2'):
    # Validate decompose_levels parameter
    try:
        decompose_levels = [int(l.strip()) for l in str(decompose_levels).split(',')]
    except ValueError:
        raise ValueError("decompose_levels must be comma-separated integers")
    
    if not decompose_levels:
        raise ValueError("At least one decomposition level must be specified")
    
    if any(level < 0 for level in decompose_levels):
        raise ValueError("Decomposition levels must be non-negative")
    
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
    
    return reconstructed_levels

@register_step
class wavelet_decompose_step(BaseStep):
    name = "wavelet_decompose"
    category = "Wavelet"
    description = "Decomposes a signal into separate wavelet levels, creating multiple channels."
    tags = ["time-series"]
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
                        # Validate decompose_levels format
                        try:
                            levels = [int(l.strip()) for l in str(val).split(',')]
                            if not levels:
                                raise ValueError("At least one level must be specified")
                            if any(l < 0 for l in levels):
                                raise ValueError("Levels must be non-negative")
                            if len(set(levels)) != len(levels):
                                raise ValueError("Duplicate levels are not allowed")
                        except ValueError as e:
                            if "invalid literal" in str(e):
                                raise ValueError("decompose_levels must be comma-separated integers")
                            raise e
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
                
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        # Validate input data
        if len(channel.ydata) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaN values")
        if len(channel.ydata) < 4:
            raise ValueError("Signal too short for wavelet decomposition (minimum 4 samples)")
        
        try:
            params = cls._inject_fs_if_needed(channel, params, wavelet_decompose)
            level_data = wavelet_decompose(channel.ydata, **params)
            
            # Create multiple channels, one for each decomposition level
            new_channels = []
            for level_idx, y_new in level_data:
                x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
                # Add level information to the channel name/params
                level_params = params.copy()
                level_params['level_extracted'] = level_idx
                new_channel = cls.create_new_channel(
                    parent=channel, 
                    xdata=x_new, 
                    ydata=y_new, 
                    params=level_params
                )
                # Modify the name to indicate which level this is
                if hasattr(new_channel, 'name'):
                    new_channel.name = f"{new_channel.name}_level_{level_idx}"
                new_channels.append(new_channel)
            
            if not new_channels:
                raise ValueError("No channels were created from decomposition")
            
            return new_channels
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Wavelet decomposition failed: {str(e)}") 
