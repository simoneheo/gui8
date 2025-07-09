
import numpy as np
from sklearn.ensemble import IsolationForest
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class isolation_forest_step(BaseStep):
    name = "isolation_forest"
    category = "Anomaly Detection"
    description = "Detect anomalies using Isolation Forest"
    tags = ["sklearn", "anomaly", "unsupervised"]
    params = [
        {"name": "contamination", "type": "float", "default": "0.05", "help": "Proportion of outliers"}
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Isolation Forest anomaly detection (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")
        if len(y) < 10:
            raise ValueError("Signal too short for anomaly detection (minimum 10 samples)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        contamination = params.get("contamination")
        
        if contamination is None:
            raise ValueError("Contamination parameter is required")
        if contamination < 0.0 or contamination > 0.5:
            raise ValueError("Contamination must be between 0.0 and 0.5")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)):
            raise ValueError("Isolation Forest produced NaN values")
        if np.any(np.isinf(y_new)):
            raise ValueError("Isolation Forest produced infinite values")

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
                elif param["type"] == "bool":
                    parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the processing step to a channel"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_new,
                params=params,
                suffix="AnomalyScore"
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Isolation Forest processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic"""
        contamination = params["contamination"]
        
        # Reshape data for sklearn
        y_reshaped = y.reshape(-1, 1)
        
        # Fit Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(y_reshaped)
        
        # Get anomaly scores
        scores = model.decision_function(y_reshaped)
        
        return scores
