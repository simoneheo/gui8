import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def bandpass_fir(y, fs, cutoff=(0.5, 4.0), numtaps=101, window='hamming'):
    from scipy.signal import firwin, filtfilt
    
    # Parameter validation
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs} Hz")
    if numtaps <= 0:
        raise ValueError(f"Number of taps must be positive, got {numtaps}")
    if numtaps % 2 == 0:
        raise ValueError(f"Number of taps must be odd for bandpass filter, got {numtaps}")
    
    # Validate cutoff frequencies
    if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
        raise ValueError(f"Cutoff must be a tuple/list of 2 frequencies, got {cutoff}")
    
    low_cutoff, high_cutoff = cutoff
    if low_cutoff <= 0 or high_cutoff <= 0:
        raise ValueError(f"Cutoff frequencies must be positive, got {cutoff}")
    if low_cutoff >= high_cutoff:
        raise ValueError(f"Low cutoff ({low_cutoff}) must be less than high cutoff ({high_cutoff})")
    
    nyq = 0.5 * fs
    if high_cutoff >= nyq:
        raise ValueError(f"High cutoff frequency ({high_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
    if low_cutoff >= nyq:
        raise ValueError(f"Low cutoff frequency ({low_cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
    
    normal_cutoff = [f / nyq for f in cutoff]
    
    try:
        b = firwin(numtaps, normal_cutoff, window=window, pass_zero=False)
    except ValueError as e:
        if "must be less than 1" in str(e):
            raise ValueError(f"Cutoff frequencies too high relative to sampling rate")
        else:
            raise ValueError(f"Filter design failed: {str(e)}")
    
    try:
        return filtfilt(b, [1.0], y)
    except ValueError as e:
        if "padlen" in str(e):
            padlen = 3 * len(b)
            msg = (
                f"Signal too short for FIR bandpass filter: "
                f"requires signal length > {padlen} but got {len(y)}. "
                f"Try reducing 'numtaps' (currently {numtaps})."
            )
            raise ValueError(msg) from e
        else:
            raise

@register_step
class bandpass_fir_step(BaseStep):
    name = "bandpass_fir"
    category = "Filter"
    description = "Applies a band-pass FIR filter using a window method."
    tags = ["time-series"]
    params = [
        {"name": "cutoff", "type": "str", "default": "(0.5, 4.0)", "help": "Tuple of (low, high) cutoff frequencies"},
        {"name": "numtaps", "type": "int", "default": "101", "help": "Number of filter taps (kernel size)"},
        {"name": "window", "type": "str", "default": "hamming", "options": ["hamming", "hann", "blackman", "bartlett"], "help": "Window function to use"},
        {"name": "fs", "type": "float", "default": "", "help": "Sampling frequency (injected from parent channel)"}
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
            if name == "fs": continue
            value = user_input.get(name, param.get("default"))
            if name == "cutoff":
                parsed[name] = eval(value) if isinstance(value, str) else value
            else:
                parsed[name] = value if "options" in param else float(value) if param["type"] == "float" else int(value)
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y, x = channel.ydata, channel.xdata
        params = cls._inject_fs_if_needed(channel, params, bandpass_fir)
        y_new = bandpass_fir(y, **params)
        x_new = np.linspace(x[0], x[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
