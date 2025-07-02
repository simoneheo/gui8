import numpy as np
try:
    from channel import Channel
except ImportError:
    Channel = None
from mixer.base_mixer import BaseMixer
from mixer.mixer_registry import register_mixer

@register_mixer
class ThresholdMaskMixer(BaseMixer):
    name = "threshold_mask"
    category = "masking"
    description = "Applies a binary mask to A based on thresholding A itself."
    tags = ["composed", "masking"]
    params = [
        {"name": "threshold", "type": "float", "default": 0.0, "help": "Threshold value"},
        {"name": "mode", "type": "str", "default": "gt", "help": "gt, lt, ge, le, eq, ne"},
        {"name": "label", "type": "str", "default": "C", "help": "Output channel label"},
    ]

    def apply(self, channels: dict[str, Channel]) -> Channel:
        A = channels["A"]
        y = A.ydata
        mode = self.kwargs.get("mode", "gt")
        thresh = self.kwargs.get("threshold", 0.0)
        label = self.kwargs.get("label", "C")

        if mode == "gt":
            mask = y > thresh
        elif mode == "lt":
            mask = y < thresh
        elif mode == "ge":
            mask = y >= thresh
        elif mode == "le":
            mask = y <= thresh
        elif mode == "eq":
            mask = y == thresh
        elif mode == "ne":
            mask = y != thresh
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")

        result = y * mask
        expr = f"{label} = A * (A {mode} {thresh})"

        return self.create_channel(A.xdata, result, [A], label, expr, self.kwargs)

    @classmethod
    def get_expression_templates(cls):
        """Get common threshold/mask expression templates for UI."""
        return [
            "A * (A > 0.5)",
            "A * (B > threshold)",
            "A if A > B else 0",
            "A if B > 0.5 else B",
            "A * (A > mean(A))",
            "A * (abs(A) > std(A))",
            "A * (A > 0) + B * (A <= 0)",
            "clip(A, 0, 1)",
            "A * (B > percentile(B, 90))",
            "A * (abs(A - mean(A)) < 2*std(A))",
            "A * ((A > 0.2) & (A < 0.8))",
            "A * (B > max(B) * 0.5)",
            "A * (A > median(A))",
            "A * (sign(A) == sign(B))"
        ]

    @classmethod
    def get_mixer_guidance(cls):
        """Get specific guidance text for this mixer type."""
        return {
            "title": "Threshold & Masking Operations",
            "description": "Applies conditional filtering, clipping, and masking based on threshold values and statistical criteria.",
            "use_cases": [
                "Noise removal and signal cleaning",
                "Amplitude limiting and clipping",
                "Statistical outlier filtering",
                "Conditional signal gating",
                "Range-based signal selection"
            ],
            "tips": [
                "Use A * (A > threshold) to gate signals above threshold",
                "clip(A, min, max) constrains signal to a range",
                "Statistical thresholds: A > mean(A) + 2*std(A)",
                "Conditional replacement: A if condition else fallback",
                "Combine conditions: A * ((A > low) & (A < high))"
            ],
            "common_patterns": [
                "Amplitude gating: A * (A > 0.5)",
                "Noise filtering: A * (abs(A) > std(A))",
                "Range clipping: clip(A, 0, 1)",
                "Statistical filtering: A * (A > mean(A))",
                "Outlier removal: A * (abs(A - mean(A)) < 2*std(A))"
            ]
        }
