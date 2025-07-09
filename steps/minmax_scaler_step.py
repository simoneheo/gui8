import numpy as np
from sklearn.preprocessing import MinMaxScaler
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class minmax_scaler_step(BaseStep):
    name = "minmax_scaler"
    category = "scikit-learn"
    description = "Rescale signal to the [0, 1] range using MinMaxScaler."
    tags = ["scaling", "normalization", "minmax", "scikit-learn"]
    params = []

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        x = channel.xdata
        y = channel.ydata.reshape(-1, 1)
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y).flatten()
        new_channel = cls.create_new_channel(parent=channel, xdata=x, ydata=y_scaled, params=params, suffix="MinMaxScaled")
        new_channel.legend_label = f"{channel.legend_label} (MinMax Scaled)"
        return [new_channel]