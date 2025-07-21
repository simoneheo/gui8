
import numpy as np
from typing import Optional
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class agglomerative_clustering_step(BaseStep):
    name = "agglomerative_clustering"
    category = "scikit-learn"
    description = "Apply hierarchical agglomerative clustering to group signal data into clusters"
    tags = ["time-series", "clustering", "hierarchical", "scikit-learn", "unsupervised"]
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
    def validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        # Validate n_clusters parameter
        n_clusters = cls.validate_integer_parameter(
            "n_clusters", 
            params.get("n_clusters"), 
            min_val=1
        )
        
        # Validate linkage parameter
        linkage = cls.validate_string_parameter(
            "linkage", 
            params.get("linkage"), 
            valid_options=["ward", "complete", "average", "single"]
        )


    def script(self, x: np.ndarray, y: np.ndarray, fs: Optional[float], params: dict) -> np.ndarray:
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
        
        y_new = model.fit_predict(y_reshaped)
        
        return y_new
