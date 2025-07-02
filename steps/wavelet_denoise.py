import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

import pywt
from scipy.signal import resample

def wavelet_denoise(y, wavelet='db4', level=None, threshold=0.2, mode='soft'):
    coeffs = pywt.wavedec(y, wavelet, mode='periodization', level=level)
    thresholded = [pywt.threshold(c, threshold, mode=mode) if i > 0 else c for i, c in enumerate(coeffs)]
    return pywt.waverec(thresholded, wavelet, mode='periodization')[:len(y)]

@register_step
class wavelet_denoise_step(BaseStep):
    name = "wavelet_denoise"
    category = "Wavelet"
    description = "Denoises a signal using wavelet thresholding."
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
            "name": "threshold", 
            "type": "float", 
            "default": "0.2", 
            "help": "Threshold for coefficient shrinkage (0-1 range typical)",
        },
        { 
            "name": "mode", 
            "type": "str", 
            "default": "soft", 
            "help": "Thresholding mode",
            "options": ["soft", "hard"]
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
                    parsed_val = float(val)
                    if name == "threshold" and (parsed_val < 0 or parsed_val > 10):
                        raise ValueError("Threshold should typically be between 0 and 10")
                    parsed[name] = parsed_val
                elif param["type"] == "int": 
                    parsed_val = int(val)
                    if name == "level" and parsed_val < 1:
                        raise ValueError("Decomposition level must be at least 1")
                    parsed[name] = parsed_val
                else: 
                    if name == "mode" and val not in ["soft", "hard"]:
                        raise ValueError("Mode must be 'soft' or 'hard'")
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
                
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Validate input data
        if len(channel.ydata) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaN values")
        if len(channel.ydata) < 4:
            raise ValueError("Signal too short for wavelet decomposition (minimum 4 samples)")
            
        try:
            params = cls._inject_fs_if_needed(channel, params, wavelet_denoise)
            y_new = wavelet_denoise(channel.ydata, **params)
            x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
            return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Wavelet denoising failed: {str(e)}")
