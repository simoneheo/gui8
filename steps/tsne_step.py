
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class tsne_step(BaseStep):
    name = "tsne"
    category = "scikit-learn"
    description = """Visualize high-dimensional features in 2D using t-SNE (t-Distributed Stochastic Neighbor Embedding).

This step reduces high-dimensional data to 2D for visualization while preserving local structure.
Useful for:
• Visualizing high-dimensional feature spaces
• Exploring data clusters and patterns
• Dimensionality reduction for visualization
• Understanding data structure and relationships"""
    tags = ["tsne", "visualization", "embedding", "scikit-learn", "dimensionality-reduction"]
    params = [
        {
            "name": "perplexity",
            "type": "float",
            "default": 30.0,
            "help": "Perplexity parameter for t-SNE (must be positive)"
        },
        {
            "name": "random_state",
            "type": "int",
            "default": 42,
            "help": "Random state for reproducible results"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — t-SNE dimensionality reduction for visualization (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for t-SNE (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal contains constant values (cannot perform t-SNE)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        perplexity = params.get("perplexity")
        
        if perplexity <= 0:
            raise ValueError("perplexity must be positive")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_tsne: np.ndarray) -> None:
        """Validate output data"""
        if y_tsne.size == 0:
            raise ValueError("t-SNE produced empty output")
        if len(y_tsne) != len(y_original):
            raise ValueError("t-SNE output length doesn't match input signal length")

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
        """Apply t-SNE to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_tsne = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_tsne)
            
            # Create t-SNE channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_tsne,
                params=params,
                suffix="TSNE"
            )
            
            # Set channel properties
            new_channel.tags = ["tsne", "embedding", "visualization"]
            new_channel.legend_label = f"{channel.legend_label} (t-SNE)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"t-SNE transformation failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for t-SNE"""
        from sklearn.manifold import TSNE
        
        perplexity = params.get("perplexity", 30.0)
        random_state = params.get("random_state", 42)
        
        # Ensure perplexity doesn't exceed number of samples
        perplexity = min(perplexity, len(y) - 1)
        
        # Reshape for sklearn if needed
        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y
        
        # Create and fit t-SNE model
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state
        )
        
        y_tsne = model.fit_transform(y_reshaped)
        
        return y_tsne
