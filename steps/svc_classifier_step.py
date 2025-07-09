
import numpy as np
from sklearn.svm import SVC
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class svc_classifier_step(BaseStep):
    name = "svc_classifier"
    category = "Classification"
    description = "Binary classification using Support Vector Classifier"
    tags = ["sklearn", "classification", "svc"]
    params = []

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — SVC binary classification (Category: {cls.category})"

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

    @classmethod
    def _validate_metadata(cls, channel: Channel) -> None:
        """Validate that required metadata is present"""
        features = channel.metadata.get("features", None)
        labels = channel.metadata.get("labels", None)
        
        if features is None:
            raise ValueError("Channel metadata must contain 'features' for SVC classification")
        if labels is None:
            raise ValueError("Channel metadata must contain 'labels' for SVC classification")
        
        if not isinstance(features, np.ndarray) and not isinstance(features, list):
            raise ValueError("Features must be a numpy array or list")
        if not isinstance(labels, np.ndarray) and not isinstance(labels, list):
            raise ValueError("Labels must be a numpy array or list")
        
        features = np.array(features)
        labels = np.array(labels)
        
        if len(features) == 0:
            raise ValueError("Features array is empty")
        if len(labels) == 0:
            raise ValueError("Labels array is empty")
        if len(features) != len(labels):
            raise ValueError(f"Features and labels length mismatch: {len(features)} vs {len(labels)}")
        if features.ndim != 2:
            raise ValueError("Features must be a 2D array")
        if features.shape[1] < 1:
            raise ValueError("At least 1 feature required for SVC classification")
        
        # Check for binary classification
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            raise ValueError(f"SVC classifier requires exactly 2 classes, got {len(unique_labels)}")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        # No parameters to validate for this step
        pass

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) == 0:
            raise ValueError("SVC classifier produced empty probability scores")
        if np.any(np.isnan(y_new)):
            raise ValueError("SVC classifier produced NaN probability scores")
        if np.any(np.isinf(y_new)):
            raise ValueError("SVC classifier produced infinite probability scores")
        if np.any(y_new < 0) or np.any(y_new > 1):
            raise ValueError("SVC classifier produced probability scores outside [0, 1] range")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        # No parameters to parse for this step
        return {}

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the processing step to a channel"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and metadata
            cls._validate_input_data(y)
            cls._validate_metadata(channel)
            cls._validate_parameters(params)
            
            # Process the data
            x_new, y_new = cls.script(x, y, channel, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            return cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params,
                suffix="SVCProbs"
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"SVC classifier processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, channel: Channel, params: dict) -> tuple[np.ndarray, np.ndarray]:
        """Core processing logic"""
        features = np.array(channel.metadata["features"])
        labels = np.array(channel.metadata["labels"])
        
        # Fit SVC model with probability estimation
        model = SVC(probability=True, random_state=42)
        model.fit(features, labels)
        
        # Get probability scores for positive class
        probs = model.predict_proba(features)[:, 1]
        
        # Create sample index array
        x_new = np.arange(len(probs))
        
        return x_new, probs
