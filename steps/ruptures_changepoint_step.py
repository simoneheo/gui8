
import numpy as np
import ruptures as rpt
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class ruptures_changepoint_step(BaseStep):
    name = "ruptures_changepoint"
    category = "Segmentation"
    description = """Detect change points using the ruptures library (Pelt, RBF cost).

This step identifies change points in time series data using the PELT (Pruned Exact Linear Time)
algorithm with RBF (Radial Basis Function) cost function.

Useful for:
• Detecting structural breaks in time series
• Segmenting signals into homogeneous regions
• Identifying regime changes in data
• Time series segmentation and analysis"""
    tags = ["segmentation", "ruptures", "changepoint", "pelt", "rbf"]
    params = [
        {
            "name": "pen",
            "type": "float",
            "default": 5.0,
            "help": "Penalty value to control number of change points (must be positive)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Change point detection using ruptures (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 10:
            raise ValueError("Signal too short for change point detection (minimum 10 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal has no variation (constant values)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        pen = params.get("pen")
        
        if pen <= 0:
            raise ValueError("pen must be positive")
        if pen > 1000:
            raise ValueError("pen too large (maximum 1000)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, labels: np.ndarray) -> None:
        """Validate output data"""
        if labels.size == 0:
            raise ValueError("Change point detection produced empty results")
        if len(labels) != len(y_original):
            raise ValueError("Output labels length doesn't match input signal length")
        if np.any(labels < 0):
            raise ValueError("Change point labels should be non-negative")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        """Apply change point detection to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            labels = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, labels)
            
            # Create change point channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=labels,
                params=params,
                suffix="Ruptures"
            )
            
            # Set channel properties
            new_channel.tags = ["segmentation", "ruptures", "changepoint", "labels"]
            new_channel.legend_label = f"{channel.legend_label} (Change Points)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Change point detection failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for change point detection"""
        pen = params.get("pen", 5.0)
        
        # Remove any NaN values for change point detection
        y_clean = y[~np.isnan(y)]
        if len(y_clean) < 10:
            raise ValueError("Insufficient non-NaN data for change point detection")
        
        # Fit PELT model with RBF cost
        model = rpt.Pelt(model="rbf").fit(y_clean)
        bkpts = model.predict(pen=pen)
        
        # Create labels array
        y_out = np.zeros_like(y)
        for i, bk in enumerate(bkpts[:-1]):
            start_idx = bkpts[i - 1] if i > 0 else 0
            y_out[start_idx:bk] = i + 1
        
        return y_out
