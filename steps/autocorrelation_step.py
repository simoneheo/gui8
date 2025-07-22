import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class autocorrelation_step(BaseStep):
    name = "autocorrelation"
    category = "Features"
    description = (
        "Compute autocorrelation to analyze signal periodicity.\n"
        "If time axis is uniformly sampled, lags are returned in seconds; "
        "otherwise, lags are returned in samples."
    )
    tags = ["autocorrelation", "periodicity", "correlation", "lag", "scipy"]
    params = [
        {"name": "max_lag", "type": "int", "default": "100", "help": "Maximum lag in samples for autocorrelation"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        max_lag = cls.validate_integer_parameter("max_lag", params.get("max_lag"), min_val=1)

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from scipy.signal import correlate

        max_lag = params["max_lag"]

        if len(y) < max_lag:
            raise ValueError(f"Signal too short: requires at least {max_lag} samples")

        # Compute autocorrelation
        autocorr = correlate(y, y, mode='full')
        autocorr = autocorr / autocorr[len(y) - 1]  # Normalize at lag 0

        # Extract positive lags only
        start_idx = len(y) - 1
        end_idx = min(start_idx + max_lag, len(autocorr))
        autocorr_positive = autocorr[start_idx:end_idx]
        lag_samples = np.arange(len(autocorr_positive))

        # Try to convert lags to time if x is uniformly sampled
        dx = np.diff(x)
        if len(dx) > 0 and np.allclose(dx, dx[0], atol=1e-6):
            # Uniform sampling → convert to seconds
            lag_axis = lag_samples * dx[0]
        else:
            # Non-uniform sampling → return lag in samples
            lag_axis = lag_samples

        return [
            {
                'tags': ['time-series'],
                'x': lag_axis,
                'y': autocorr_positive
            }
        ]
