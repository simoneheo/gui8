import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel


import pywt

def wavelet_filter_band(y, wavelet='db4', level=None, keep_levels='1,2,3'):
    keep_levels = [int(l.strip()) for l in str(keep_levels).split(',')]
    coeffs = pywt.wavedec(y, wavelet, mode='periodization', level=level)
    for i in range(1, len(coeffs)):
        if i not in keep_levels:
            coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wavelet, mode='periodization')[:len(y)]


@register_step
class wavelet_filter_band_step(BaseStep):
    name = "wavelet_filter_band"
    category = "Wavelet"
    description = "Reconstructs a signal using selected wavelet bands."
    tags = ["time-series"]
    params = [
        { "name": "wavelet", "type": "str", "default": "db4", "help": "Wavelet type (e.g., db4, sym5, coif1)", "options": ["db1", "db2", "db4", "db8", "db10", "haar", "sym4", "sym5", "sym8", "coif2", "coif4", "coif6", "bior2.2", "bior4.4", "dmey"] },
        { "name": "level", "type": "int", "default": "", "help": "Decomposition level (optional, auto if blank)" },
        { "name": "keep_levels", "type": "str", "default": "1,2", "help": "Comma-separated levels to keep or reconstruct" },
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
            if param["name"] == "fs": continue
            val = user_input.get(param["name"], param["default"])
            if val == "": parsed[param["name"]] = None
            elif param["type"] == "float": parsed[param["name"]] = float(val)
            elif param["type"] == "int": parsed[param["name"]] = int(val)
            else: parsed[param["name"]] = val
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        params = cls._inject_fs_if_needed(channel, params, wavelet_filter_band)
        y_new = wavelet_filter_band(channel.ydata, **params)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
