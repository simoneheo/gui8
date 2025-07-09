
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class pca_step(BaseStep):
    name = "pca"
    category = "scikit-learn"
    description = """Apply Principal Component Analysis (PCA) for dimensionality reduction.

This step reduces the dimensionality of the data while preserving the most important variance.
Useful for:
• Dimensionality reduction for visualization
• Feature extraction and compression
• Noise reduction
• Pattern discovery in high-dimensional data"""
    tags = ["pca", "dimensionality-reduction", "scikit-learn", "feature-extraction"]
    params = [
        {
            "name": "n_components",
            "type": "int",
            "default": 2,
            "help": "Number of principal components to keep (must be positive)"
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
        return f"{cls.name} — Principal Component Analysis (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for PCA (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal contains constant values (cannot perform PCA)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        n_components = params.get("n_components")
        
        if n_components <= 0:
            raise ValueError("n_components must be positive")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_pca: np.ndarray) -> None:
        """Validate output data"""
        if y_pca.size == 0:
            raise ValueError("PCA produced empty output")
        if len(y_pca) != len(y_original):
            raise ValueError("PCA output length doesn't match input signal length")

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
        """Apply PCA to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_pca = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_pca)
            
            # Create PCA channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_pca,
                params=params,
                suffix="PCA"
            )
            
            # Set channel properties
            new_channel.tags = ["pca", "dimensionality-reduction", "transformed"]
            new_channel.legend_label = f"{channel.legend_label} (PCA Transformed)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"PCA transformation failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for PCA"""
        from sklearn.decomposition import PCA
        
        n_components = params.get("n_components", 2)
        random_state = params.get("random_state", 42)
        
        # Ensure n_components doesn't exceed number of samples or features
        if y.ndim == 1:
            max_components = 1
        else:
            max_components = min(y.shape[0], y.shape[1])
        
        n_components = min(n_components, max_components)
        
        # Reshape for sklearn if needed
        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y
        
        # Create and fit PCA model
        pca = PCA(
            n_components=n_components,
            random_state=random_state
        )
        
        y_transformed = pca.fit_transform(y_reshaped)
        
        # If single component, flatten the result
        if n_components == 1:
            y_transformed = y_transformed.flatten()
        
        return y_transformed
