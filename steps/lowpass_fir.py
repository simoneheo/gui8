import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def lowpass_fir(y, fs, cutoff=2.0, numtaps=101, window='hamming'):
    from scipy.signal import firwin, filtfilt
    
    # Parameter validation
    if cutoff <= 0:
        raise ValueError(f"Cutoff frequency must be positive, got {cutoff} Hz")
    if numtaps <= 0:
        raise ValueError(f"Number of taps must be positive, got {numtaps}")
    if numtaps % 2 == 0:
        raise ValueError(f"Number of taps must be odd for FIR filter, got {numtaps}")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs} Hz")

    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(f"Cutoff frequency ({cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")

    normal_cutoff = cutoff / nyq
    
    try:
        b = firwin(numtaps, normal_cutoff, window=window, pass_zero=True)
    except ValueError as e:
        if "must be less than 1" in str(e):
            raise ValueError(f"Cutoff frequency too high: {cutoff} Hz >= Nyquist ({nyq:.1f} Hz)")
        else:
            raise ValueError(f"FIR filter design failed: {str(e)}")
    
    try:
        return filtfilt(b, [1.0], y)
    except ValueError as e:
        if "padlen" in str(e):
            padlen = 3 * max(len(b), 1)  # For FIR filter, a=[1.0], so len(a)=1
            msg = (
                f"Signal too short for FIR filter: "
                f"requires signal length > {padlen} but got {len(y)}. "
                f"Try reducing 'numtaps' (currently {numtaps})."
            )
            raise ValueError(msg) from e
        else:
            raise

@register_step
class lowpass_fir_step(BaseStep):
    name = "lowpass_fir"
    category = "Filter"
    description = "Applies a low-pass FIR filter using a window method."
    tags = ["time-series"]
    params = [
        {"name": "cutoff", "type": "float", "default": "2.0", "help": "Cutoff frequency in Hz"},
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
            parsed[name] = value if "options" in param else float(value) if param["type"] == "float" else int(value)
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y, x = channel.ydata, channel.xdata
        params = cls._inject_fs_if_needed(channel, params, lowpass_fir)
        y_new = lowpass_fir(y, **params)
        x_new = np.linspace(x[0], x[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
