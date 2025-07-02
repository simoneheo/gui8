import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def impute_missing(y, method='zero'):

    if not np.any(np.isnan(y)):
        return y
    if method == "zero":
        return np.nan_to_num(y, nan=0.0)
    elif method == "mean":
        return np.nan_to_num(y, nan=np.nanmean(y))
    elif method == "ffill":
        import pandas as pd
        return pd.Series(y).fillna(method='ffill').fillna(0).values
    elif method == "bfill":
        import pandas as pd
        return pd.Series(y).fillna(method='bfill').fillna(0).values
    else:
        raise ValueError(f"Unknown imputation method: {method}")

@register_step
class impute_missing_step(BaseStep):
    name = "impute_missing"
    category = "General"
    description = "Fill missing values using simple imputation method."
    tags = ["time-series"]
    params = [{'name': 'method', 'type': 'str', 'default': 'zero', 'options': ['zero', 'ffill', 'bfill', 'mean'], 'help': 'Imputation method'}]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"
    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}
    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            if param["name"] == "fs": continue
            value = user_input.get(param["name"], param.get("default"))
            parsed[param["name"]] = float(value) if param["type"] == "float" else value
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        
        y_new = impute_missing(channel.ydata)
        x_new = np.linspace(channel.xdata[0], channel.xdata[-1], len(y_new))
        return cls.create_new_channel(parent=channel, xdata=x_new, ydata=y_new, params=params)
