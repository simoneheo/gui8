import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def lowpass_bessel(y, fs, cutoff=2.0, order=2):
    from scipy.signal import bessel, filtfilt
    
    # Parameter validation
    if cutoff <= 0:
        raise ValueError(f"Cutoff frequency must be positive, got {cutoff} Hz")
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs} Hz")
    
    nyq = 0.5 * fs
    if cutoff >= nyq:
        raise ValueError(f"Cutoff frequency ({cutoff} Hz) must be less than Nyquist frequency ({nyq:.1f} Hz)")
    
    normal_cutoff = cutoff / nyq
    
    try:
        b, a = bessel(N=order, Wn=normal_cutoff, btype='low', analog=False, norm='phase')
    except ValueError as e:
        if "must be less than 1" in str(e):
            raise ValueError(f"Cutoff frequency too high: {cutoff} Hz >= Nyquist ({nyq:.1f} Hz)")
        else:
            raise ValueError(f"Bessel lowpass filter design failed: {str(e)}")
    
    try:
        return filtfilt(b, a, y)
    except ValueError as e:
        if "padlen" in str(e):
            padlen = 3 * max(len(a), len(b))
            msg = (
                f"Signal too short for Bessel lowpass filter: "
                f"requires signal length > {padlen} but got {len(y)}. "
                f"Try reducing filter 'order' (currently {order})."
            )
            raise ValueError(msg) from e
        else:
            raise

@register_step
class lowpass_bessel_step(BaseStep):
    name = "lowpass_bessel"
    category = "Filter"
    description = "Applies a low-pass Bessel filter to the signal (preserves waveform shape)."
    tags = ["time-series"]
    params = [
        {"name": "cutoff", "type": "float", "default": "2.0", "help": "Cutoff frequency in Hz"},
        {"name": "order", "type": "int", "default": "2", "help": "Order of the Bessel filter"},
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
            parsed[name] = float(value) if param["type"] == "float" else int(value)
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y, x = channel.ydata, channel.xdata
        params = cls._inject_fs_if_needed(channel, params, lowpass_bessel)
        y_new = lowpass_bessel(y, **params)
        x_new = np.linspace(x[0], x[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
