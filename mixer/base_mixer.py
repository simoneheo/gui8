from abc import ABC, abstractmethod
try:
    from channel import Channel, SourceType
except ImportError:
    # Create dummy classes if not available
    class Channel:
        @classmethod
        def from_parent(cls, **kwargs):
            return cls()
    
    class SourceType:
        COMPOSED = "composed"
import numpy as np
from typing import Any, Dict

class BaseMixer(ABC):
    """
    Base class for signal mixers (composed channels).
    Supports operations like A+B, A*B, abs(A), A*(B>0.5), etc.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @abstractmethod
    def apply(self, channels: dict[str, Channel]) -> Channel:
        """Perform mixing using provided channel dict (e.g., {"A": ch1, "B": ch2})"""
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "tags": self.tags,
            "params": self.params,
        }

    @classmethod
    def create_channel(cls, xdata, ydata, parents: list[Channel], label: str, expr: str, params: dict):
        parent = parents[0]
        new_channel = Channel.from_parent(
            parent=parent,
            xdata=xdata,
            ydata=ydata,
            legend_label=label,
            description=f"Composed signal: {expr}",
            tags=["composed", "time-series"],
            type=SourceType.COMPOSED,
            params=params,
            metadata={"expression": expr}
        )
        new_channel.parent_ids = [p.channel_id for p in parents]
        return new_channel
