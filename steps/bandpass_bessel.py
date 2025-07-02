import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def bandpass_bessel(y, fs, cutoff=(0.5, 4.0), order=2):
    from scipy.signal import bessel, filtfilt
    
    # Parameter validation
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs} Hz")
    if order <= 0:
        raise ValueError(f"Filter order must be positive, got {order}")
    
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
        b, a = bessel(N=order, Wn=normal_cutoff, btype='band', analog=False, norm='phase')
    except ValueError as e:
        if "must be less than 1" in str(e):
            raise ValueError(f"Cutoff frequencies too high relative to sampling rate")
        else:
            raise ValueError(f"Bessel bandpass filter design failed: {str(e)}")
    
    try:
        return filtfilt(b, a, y)
    except ValueError as e:
        if "padlen" in str(e):
            padlen = 3 * max(len(a), len(b))
            msg = (
                f"Signal too short for Bessel bandpass filter: "
                f"requires signal length > {padlen} but got {len(y)}. "
                f"Try reducing filter 'order' (currently {order})."
            )
            raise ValueError(msg) from e
        else:
            raise

@register_step
class bandpass_bessel_step(BaseStep):
    name = "bandpass_bessel"
    category = "Filter"
    description = "Applies a band-pass Bessel filter to the signal (preserves waveform shape)."
    tags = ["time-series"]
    params = [
        {"name": "low_cutoff", "type": "float", "default": "0.5", "help": "Low cutoff frequency (Hz)"},
        {"name": "high_cutoff", "type": "float", "default": "4.0", "help": "High cutoff frequency (Hz)"},
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
            if name in ["low_cutoff", "high_cutoff"]:
                parsed[name] = float(value)
            else:
                parsed[name] = float(value) if param["type"] == "float" else int(value)
        
        # Combine into cutoff tuple for the function
        if "low_cutoff" in parsed and "high_cutoff" in parsed:
            parsed["cutoff"] = (parsed["low_cutoff"], parsed["high_cutoff"])
            del parsed["low_cutoff"]
            del parsed["high_cutoff"]
        
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y, x = channel.ydata, channel.xdata
        params = cls._inject_fs_if_needed(channel, params, bandpass_bessel)
        y_new = bandpass_bessel(y, **params)
        x_new = np.linspace(x[0], x[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
