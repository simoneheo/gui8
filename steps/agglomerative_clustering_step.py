
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class agglomerative_clustering_step(BaseStep):
    name = "agglomerative_clustering"
    category = "scikit-learn"
    description = "Apply hierarchical agglomerative clustering to group signal data into clusters"
    tags = ["clustering", "hierarchical", "scikit-learn", "unsupervised"]
    params = [
        {
            "name": "n_clusters",
            "type": "int",
            "default": 2,
            "help": "Number of clusters to form (must be positive)"
        },
        {
            "name": "linkage",
            "type": "str",
            "default": "ward",
            "options": ["ward", "complete", "average", "single"],
            "help": "Linkage criterion for hierarchical clustering"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Hierarchical agglomerative clustering (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for clustering (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal contains constant values (cannot cluster)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        n_clusters = params.get("n_clusters")
        linkage = params.get("linkage")
        
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if linkage not in ["ward", "complete", "average", "single"]:
            raise ValueError("linkage must be one of: ward, complete, average, single")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, labels: np.ndarray) -> None:
        """Validate output data"""
        if labels.size == 0:
            raise ValueError("Clustering produced empty labels")
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
        """Apply agglomerative clustering to the channel data."""
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
                suffix="AggloClustering"
            )
            
            # Set channel properties
            new_channel.tags = ["clustering", "hierarchical", "labels"]
            new_channel.legend_label = f"{channel.legend_label} (Agglomerative Clustering)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Agglomerative clustering failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for agglomerative clustering"""
        from sklearn.cluster import AgglomerativeClustering
        
        n_clusters = params.get("n_clusters", 2)
        linkage = params.get("linkage", "ward")
        
        # Ensure n_clusters doesn't exceed number of samples
        n_clusters = min(n_clusters, len(y))
        
        # Reshape for sklearn if needed
        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y
        
        # Create and fit clustering model
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = model.fit_predict(y_reshaped)
        
        return labels
