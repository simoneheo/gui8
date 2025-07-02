# steps/base_step.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from channel import Channel  # Import the Channel class
import numpy as np


class BaseStep(ABC):
    """
    Abstract base class for all processing steps.
    Each step must implement `apply()` and provide metadata like `name`, `category`, `description`, and `params`.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs  # Store parameters passed by user

    @abstractmethod
    def apply(self, channel: Channel) -> Channel:
        """
        Apply this step to the given channel. Must return a new ChannelInfo.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        """
        Return all metadata about the step.
        Useful for GUI display or parameter prompting.
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "tags": self.tags,
            "params": self.params
        }

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        for param in cls.params:
            name = param["name"]
            if name == "fs":
                # Skip parsing fs - it will be injected from parent channel
                continue
            value = user_input.get(name, param.get("default"))
            parsed[name] = float(value) if param["type"] == "float" else value
        return parsed

    @classmethod
    def _inject_fs_if_needed(cls, channel: Channel, params: dict, func) -> dict:
        """
        Automatically inject fs from parent channel if the processing function requires it.
        """
        if "fs" in func.__code__.co_varnames:
            params["fs"] = channel.fs_median
            print(f"[{cls.name}Step] Injected fs={channel.fs_median:.2f} from parent channel")
        return params

    @classmethod
    def create_new_channel(cls, parent: Channel, xdata: np.ndarray, ydata: np.ndarray, params: dict) -> Channel:
        """
        Helper method to create a new channel with consistent parameter handling.
        """
        return Channel.from_parent(
            parent=parent,
            xdata=xdata,
            ydata=ydata,
            legend_label=f"{parent.legend_label} - {cls.name}",
            description=cls.description,
            tags=cls.tags,
            params=params  # Pass the parameters to the new channel
        )
