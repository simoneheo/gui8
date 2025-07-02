
import numpy as np
import pywt

from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class cwt_spectrogram(BaseStep):
    name = "cwt_spectrogram"
    category = "Spectrogram"
    description = """Computes a spectrogram using Continuous Wavelet Transform (CWT) and outputs both:
1. A 2D spectrogram channel (tag='spectrogram') for visualizing frequency evolution using wavelets.
2. A 1D time-series channel (tag='time-series') summarizing the spectrogram with a reduction method.

Reduction methods:
- max_intensity: Maximum amplitude in each time slice.
- sum_intensity: Total wavelet energy in each time slice.
- centroid_freq: Weighted average scale index (interpretable as pseudo-frequency).
"""
    tags = ["spectrogram"]
    params = [
        {"name": "wavelet", "type": "str", "default": "morl", "options": ["morl", "cmor", "mexh"], "help": "Wavelet type to use for CWT."},
        {"name": "scales", "type": "str", "default": "1-64", "help": "Scale range as 'min-max' (e.g., '1-64')."},
        {"name": "reduction", "type": "str", "default": "max_intensity", "options": ["max_intensity", "sum_intensity", "centroid_freq"], "help": "Reduction method for summarizing the CWT."},
        {"name": "fs", "type": "float", "default": "", "help": "Sampling frequency (injected from parent channel)."}
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param["default"])
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                raise ValueError(f"Invalid input for '{name}': {str(e)}")
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        if len(channel.ydata) < 10:
            raise ValueError("Signal too short for CWT.")
        if np.all(np.isnan(channel.ydata)):
            raise ValueError("Signal contains only NaNs.")

        wavelet = params.get("wavelet", "morl")
        scale_range = params.get("scales", "1-64")
        reduction = params.get("reduction", "max_intensity")
        fs = params.get("fs", 1.0)

        try:
            scale_parts = scale_range.split("-")
            s_min, s_max = int(scale_parts[0]), int(scale_parts[1])
            if s_min < 1 or s_max <= s_min:
                raise ValueError("Invalid scale range")
            scales = np.arange(s_min, s_max + 1)
        except Exception as e:
            raise ValueError(f"Invalid scales '{scale_range}': {str(e)}")

        try:
            coeffs, freqs = pywt.cwt(channel.ydata, scales, wavelet, sampling_period=1/fs)
            power = np.abs(coeffs)
        except Exception as e:
            raise ValueError(f"CWT failed: {str(e)}")

        t = np.linspace(channel.xdata[0], channel.xdata[-1], power.shape[1])
        spectrogram_channel = cls.create_new_channel(
            parent=channel, xdata=t, ydata=freqs, params=params
        )
        spectrogram_channel.tags = ["spectrogram"]
        spectrogram_channel.xlabel = "Time (s)"
        spectrogram_channel.ylabel = "Scale (pseudo-freq)"
        spectrogram_channel.legend_label = f"{channel.legend_label} - CWT Spectrogram"
        spectrogram_channel.metadata = {"Zxx": power, "colormap": "plasma"}

        try:
            if reduction == "max_intensity":
                y_red = np.max(power, axis=0)
                ylabel = "Max Amplitude"
            elif reduction == "sum_intensity":
                y_red = np.sum(power, axis=0)
                ylabel = "Total Amplitude"
            elif reduction == "centroid_freq":
                norm = np.sum(power, axis=0)
                norm[norm == 0] = 1e-10
                y_red = np.sum(freqs[:, None] * power, axis=0) / norm
                ylabel = "Spectral Centroid (pseudo-freq)"
            else:
                raise ValueError(f"Unknown reduction method: {reduction}")
        except Exception as e:
            raise ValueError(f"Reduction failed: {str(e)}")

        summary_channel = cls.create_new_channel(
            parent=channel, xdata=t, ydata=y_red, params=params
        )
        summary_channel.tags = ["time-series"]
        summary_channel.xlabel = "Time (s)"
        summary_channel.ylabel = ylabel
        summary_channel.legend_label = f"{channel.legend_label} - {reduction.replace("_", " ").title()}"

        return [spectrogram_channel, summary_channel]
