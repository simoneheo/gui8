
import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class select_kbest_step(BaseStep):
    name = "select_kbest"
    category = "scikit-learn"
    description = """Select K best features using ANOVA F-value (supervised feature selection).

This step selects the K most informative features based on their statistical relationship with target labels.
Useful for:
• Supervised feature selection
• Reducing dimensionality while preserving predictive power
• Identifying most relevant features for classification/regression
• Improving model performance by removing irrelevant features"""
    tags = ["feature-selection", "kbest", "scikit-learn", "supervised"]
    params = [
        {
            "name": "k",
            "type": "int",
            "default": 10,
            "help": "Number of top features to select (must be positive)"
        },
        {
            "name": "target_column",
            "type": "str",
            "default": "",
            "help": "Column name or index of target labels (optional)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — Select K best features using ANOVA F-value (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray, target: np.ndarray = None) -> None:
        """Validate input signal data"""
        if len(y) < 2:
            raise ValueError("Signal too short for feature selection (minimum 2 samples)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if target is not None and len(target) != len(y):
            raise ValueError("Target labels length doesn't match signal length")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        k = params.get("k")
        
        if k <= 0:
            raise ValueError("k must be positive")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_selected: np.ndarray) -> None:
        """Validate output data"""
        if y_selected.size == 0:
            raise ValueError("Feature selection produced empty output")
        if len(y_selected) != len(y_original):
            raise ValueError("Selected features length doesn't match input signal length")

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
        """Apply SelectKBest feature selection to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get target labels from channel
            target = getattr(channel, 'target', None)
            if target is None:
                raise ValueError("Target labels are required for SelectKBest. Set channel.target attribute.")
            
            # Validate input data and parameters
            cls._validate_input_data(y, target)
            cls._validate_parameters(params)
            
            # Process the data
            y_selected = cls.script(y, target, params)
            
            # Validate output data
            cls._validate_output_data(y, y_selected)
            
            # Create feature selection channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x,
                ydata=y_selected,
                params=params,
                suffix="SelectKBest"
            )
            
            # Set channel properties
            new_channel.tags = ["feature-selection", "kbest", "selected"]
            new_channel.legend_label = f"{channel.legend_label} (SelectKBest)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"SelectKBest feature selection failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, target: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for SelectKBest feature selection"""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        k = params.get("k", 10)
        
        # Ensure k doesn't exceed number of features
        if y.ndim == 1:
            max_features = 1
        else:
            max_features = y.shape[1] if y.shape[1] > 0 else 1
        
        k = min(k, max_features)
        
        # Reshape for sklearn if needed
        if y.ndim == 1:
            y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y
        
        # Create and fit feature selector
        selector = SelectKBest(
            score_func=f_classif,
            k=k
        )
        
        y_selected = selector.fit_transform(y_reshaped, target)
        
        # If single feature, flatten the result
        if k == 1:
            y_selected = y_selected.flatten()
        
        return y_selected
