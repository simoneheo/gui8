
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class dbscan_step(BaseStep):
    name = "dbscan"
    category = "scikit-learn"
    description = "Apply density-based clustering using DBSCAN algorithm to group closely packed points"
    tags = ["clustering", "dbscan", "density", "scikit-learn", "unsupervised"]
    params = [
        {
            "name": "eps",
            "type": "float",
            "default": 0.5,
            "help": "Maximum distance for two points to be considered neighbors (must be positive)"
        },
        {
            "name": "min_samples",
            "type": "int",
            "default": 5,
            "help": "Minimum number of samples in a neighborhood (must be positive)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Density-based clustering (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for DBSCAN clustering (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        eps = params.get("eps")
        min_samples = params.get("min_samples")
        
        if eps <= 0:
            raise ValueError("eps must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, labels: np.ndarray) -> None:
        """Validate output data"""
        if labels.size == 0:
            raise ValueError("DBSCAN clustering produced empty labels")
        if len(labels) != len(y_original):
            raise ValueError("Number of labels doesn't match input signal length")

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
        """Apply DBSCAN clustering to the channel data."""
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
            
            # Create clustering channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=labels,
                params=params,
                suffix="DBSCAN"
            )
            
            # Set channel properties
            new_channel.tags = ["clustering", "density", "labels"]
            new_channel.legend_label = f"{channel.legend_label} (DBSCAN Clustering)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"DBSCAN clustering failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for DBSCAN clustering"""
        from sklearn.cluster import DBSCAN
        
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)
        
        # Ensure min_samples doesn't exceed number of samples
        min_samples = min(min_samples, len(y))
        
        # Reshape for sklearn if needed
        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y
        
        # Create and fit clustering model
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        
        labels = model.fit_predict(y_reshaped)
        
        return labels
