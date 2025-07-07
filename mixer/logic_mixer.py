import numpy as np
try:
    from channel import Channel
except ImportError:
    Channel = None
from mixer.base_mixer import BaseMixer
from mixer.mixer_registry import register_mixer

@register_mixer
class LogicMixer(BaseMixer):
    name = "logic"
    category = "logical"
    description = "Performs logic operation between two signals (e.g., A * (B > threshold))."
    tags = ["mixed", "logical"]
    params = [
        {"name": "operator", "type": "str", "default": "gt", "help": "gt, lt, ge, le, eq, ne"},
        {"name": "threshold", "type": "float", "default": 0.5, "help": "Threshold for B"},
        {"name": "label", "type": "str", "default": "C", "help": "Output channel label"},
    ]

    def apply(self, channels: dict[str, Channel]) -> Channel:
        A = channels["A"]
        B = channels["B"]
        op = self.kwargs.get("operator", "gt")
        thresh = self.kwargs.get("threshold", 0.5)
        label = self.kwargs.get("label", "C")

        b = B.ydata
        if op == "gt":
            mask = b > thresh
        elif op == "lt":
            mask = b < thresh
        elif op == "ge":
            mask = b >= thresh
        elif op == "le":
            mask = b <= thresh
        elif op == "eq":
            mask = b == thresh
        elif op == "ne":
            mask = b != thresh
        else:
            raise ValueError(f"Unsupported logical operator: {op}")

        result = A.ydata * mask
        expr = f"{label} = A * (B {op} {thresh})"

        return self.create_channel(A.xdata, result, [A, B], label, expr, self.kwargs)

    @classmethod
    def get_expression_templates(cls):
        """Get common logic expression templates for UI."""
        return [
            "A > B",
            "A < B",
            "A >= B",
            "A <= B",
            "A == B",
            "A != B",
            "(A > 0) & (B > 0)",
            "(A > B) | (B > C)",
            "A * (B > 0.5)",
            "A * (B < 0)",
            "A if A > B else 0",
            "A if B > 0.5 else B",
            "(A > mean(A)) & (B > mean(B))",
            "A * (abs(B) > std(B))"
        ]

    @classmethod
    def get_mixer_guidance(cls):
        """Get specific guidance text for this mixer type."""
        return {
            "title": "Logic & Boolean Operations",
            "description": "Creates boolean masks and conditional results using logical comparisons between channels.",
            "use_cases": [
                "Signal detection and event identification",
                "Threshold-based signal gating",
                "Boolean masking and filtering",
                "Conditional signal processing",
                "Cross-channel relationship analysis"
            ],
            "tips": [
                "Use & for AND, | for OR operations: (A > 0) & (B > 0)",
                "Create masks with comparisons: A > B returns True/False",
                "Apply conditional logic: A if condition else B",
                "Combine with multiplication: A * (B > threshold)",
                "Statistical thresholds: A > mean(A)"
            ],
            "common_patterns": [
                "Event detection: A > threshold",
                "Signal gating: A * (B > 0.5)",
                "Conditional selection: A if A > B else 0",
                "Boolean masking: (A > 0) & (B > 0)",
                "Statistical filtering: A > mean(A) + std(A)"
            ]
        }
