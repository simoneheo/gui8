
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class kmeans_clustering_step(BaseStep):
    name = "kmeans"
    category = "Features"
    description = """Apply K-Means clustering to signal or feature set."""
    tags = ["clustering", "kmeans", "unsupervised"]
    params = [
        {"name": "n_clusters", "type": "int", "default": "3", "help": "Number of clusters to form"},
        {"name": "random_state", "type": "int", "default": "42", "help": "Random state for reproducible results"}
    ]

    @classmethod
    def validate_parameters(cls, params: dict) -> None:
        n_clusters = cls.validate_integer_parameter("n_clusters", params.get("n_clusters"), min_val=1)
        random_state = cls.validate_integer_parameter("random_state", params.get("random_state"))

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> list:
        from sklearn.cluster import KMeans
        
        n_clusters = params["n_clusters"]
        random_state = params["random_state"]
        
        # Ensure n_clusters doesn't exceed number of samples
        n_clusters = min(n_clusters, len(y))
        
        # Reshape for sklearn if needed
        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y
        
        # Create and fit clustering model
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        y_clustered = model.fit_predict(y_reshaped)
        
        return [
            {
                'tags': ['time-series'],
                'x': x,
                'y': y_clustered
            }
        ]
