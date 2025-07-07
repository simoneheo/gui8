import numpy as np
try:
    from channel_info import ChannelInfo
except ImportError:
    # Fallback to Channel if ChannelInfo is not available
    from channel import Channel as ChannelInfo
from mixer.base_mixer import BaseMixer
from mixer.mixer_registry import register_mixer

@register_mixer
class ArithmeticMixer(BaseMixer):
    name = "arithmetic"
    category = "basic"
    description = "Performs arithmetic operations between two signals."
    tags = ["mixed", "arithmetic"]
    params = [
        {"name": "operation", "type": "str", "default": "add", "help": "add, sub, mul, div, mod"},
        {"name": "label", "type": "str", "default": "C", "help": "Output channel label"}
    ]

    def apply(self, channels: dict[str, ChannelInfo]) -> ChannelInfo:
        A = channels["A"]
        B = channels["B"]
        op = self.kwargs.get("operation", "add")
        label = self.kwargs.get("label", "C")

        x = A.xdata  # Use xdata instead of current_x
        a, b = A.ydata, B.ydata  # Use ydata instead of current_y

        if op == "add":
            y = a + b
            expr = f"{label} = A + B"
        elif op == "sub":
            y = a - b
            expr = f"{label} = A - B"
        elif op == "mul":
            y = a * b
            expr = f"{label} = A * B"
        elif op == "div":
            y = np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
            expr = f"{label} = A / B"
        elif op == "mod":
            y = np.mod(a, b, out=np.zeros_like(a), where=(b != 0))
            expr = f"{label} = A % B"
        else:
            raise ValueError(f"Unknown operation '{op}'")

        return self.create_channel(x, y, [A, B], label=label, expr=expr, params=self.kwargs)

    @classmethod
    def get_expression_templates(cls):
        """Get common arithmetic expression templates for UI."""
        return [
            "A + B",
            "A - B", 
            "A * B",
            "A / B",
            "A + B + C",
            "A - B - C",
            "(A + B) / 2",
            "A * 2 + B",
            "A + B * C",
            "(A - B) * C",
            "2 * A - B",
            "A + 0.5 * B",
            "A * B / C",
            "(A + B + C) / 3"
        ]

    @classmethod
    def get_mixer_guidance(cls):
        """Get specific guidance text for this mixer type."""
        return {
            "title": "Arithmetic Operations",
            "description": "Combines multiple channels using basic mathematical operations (+, -, *, /).",
            "use_cases": [
                "Signal addition and subtraction",
                "Channel scaling and normalization", 
                "Computing signal averages",
                "Creating composite signals",
                "Differential analysis (A - B)"
            ],
            "tips": [
                "Use parentheses to control operation order: (A + B) / 2",
                "Mix constants with channels: A * 2 + B",
                "Combine multiple channels: A + B + C",
                "Create ratios: A / B (handles division by zero safely)"
            ],
            "common_patterns": [
                "Signal averaging: (A + B) / 2",
                "Difference analysis: A - B", 
                "Scaling: A * constant",
                "Composite signals: A + B * weight"
            ]
        }
