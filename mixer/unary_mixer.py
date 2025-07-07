import numpy as np
try:
    from channel import Channel
except ImportError:
    Channel = None
from mixer.base_mixer import BaseMixer
from mixer.mixer_registry import register_mixer

@register_mixer
class UnaryMixer(BaseMixer):
    name = "unary"
    category = "transform"
    description = "Applies unary operations to a single signal."
    tags = ["mixed", "unary"]
    params = [
        {"name": "operation", "type": "str", "default": "abs", "help": "abs, normalize, zscore, invert, square, sqrt"},
        {"name": "label", "type": "str", "default": "C", "help": "Output channel label"}
    ]

    def apply(self, channels: dict[str, Channel]) -> Channel:
        A = channels["A"]
        op = self.kwargs.get("operation", "abs")
        label = self.kwargs.get("label", "C")

        x = A.xdata  # Use xdata instead of current_x
        y = A.ydata  # Use ydata instead of current_y

        if op == "abs":
            result = np.abs(y)
            expr = f"{label} = abs(A)"
        elif op == "normalize":
            max_val = np.max(np.abs(y))
            result = y / max_val if max_val != 0 else y
            expr = f"{label} = A / max(abs(A))"
        elif op == "zscore":
            result = (y - np.mean(y)) / np.std(y)
            expr = f"{label} = zscore(A)"
        elif op == "invert":
            result = -y
            expr = f"{label} = -A"
        elif op == "square":
            result = y ** 2
            expr = f"{label} = A**2"
        elif op == "sqrt":
            result = np.sqrt(np.abs(y))
            expr = f"{label} = sqrt(abs(A))"
        else:
            raise ValueError(f"Unknown unary operation '{op}'")

        return self.create_channel(x, result, [A], label=label, expr=expr, params=self.kwargs)

    @classmethod
    def get_expression_templates(cls):
        """Get common unary expression templates for UI."""
        return [
            "abs(A)",
            "sqrt(abs(A))",
            "-A",
            "A**2",
            "1/A",
            "A - mean(A)",
            "(A - mean(A)) / std(A)",
            "A / max(A)",
            "A / max(abs(A))",
            "sin(A)",
            "cos(A)",
            "exp(A)",
            "log(abs(A))",
            "sqrt(A**2 + B**2)"
        ]

    @classmethod
    def get_mixer_guidance(cls):
        """Get specific guidance text for this mixer type."""
        return {
            "title": "Unary Transform Operations",
            "description": "Applies mathematical transformations to a single channel, including normalization, statistical operations, and mathematical functions.",
            "use_cases": [
                "Signal rectification and absolute values",
                "Normalization and standardization",
                "Mathematical transformations (square, sqrt, log)",
                "Trigonometric functions for periodic analysis",
                "Statistical centering and scaling"
            ],
            "tips": [
                "Use abs(A) for signal rectification",
                "A**2 creates power signals (always positive)",
                "-A inverts signal polarity",
                "A - mean(A) removes DC offset",
                "(A - mean(A)) / std(A) creates z-scores"
            ],
            "common_patterns": [
                "Rectification: abs(A)",
                "Normalization: A / max(A)",
                "Z-score: (A - mean(A)) / std(A)",
                "Power signal: A**2",
                "DC removal: A - mean(A)"
            ]
        } 