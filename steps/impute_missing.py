import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class impute_missing_step(BaseStep):
    name = "impute_missing"
    category = "Data Quality"
    description = """Fill missing (NaN) values in signals using various imputation methods.
    
Missing values can occur due to sensor failures, data corruption, or preprocessing steps.
This step provides several strategies to handle missing data:

• **Zero fill**: Replace NaN values with 0 (simple and fast)
• **Mean fill**: Replace NaN values with the signal's mean (preserves average)
• **Forward fill**: Use the last valid value to fill gaps (maintains trends)
• **Backward fill**: Use the next valid value to fill gaps (anticipates trends)

Choose the method based on your signal characteristics and analysis requirements."""
    tags = ["time-series", "data-quality", "preprocessing", "impute", "fill", "missing", "nan"]
    params = [
        {
            'name': 'method', 
            'type': 'str', 
            'default': 'zero', 
            'options': ['zero', 'mean', 'ffill', 'bfill'], 
            'help': '''Imputation method to use:
• zero: Replace NaN with 0 (simple, fast)
• mean: Replace NaN with signal mean (preserves average)
• ffill: Forward fill - use last valid value (maintains trends)
• bfill: Backward fill - use next valid value (anticipates trends)'''
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Fill missing (NaN) values using various imputation strategies (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Cannot impute when all values are NaN")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        method = params.get("method")
        
        if method is None:
            raise ValueError("Method parameter is required")
        
        valid_methods = ["zero", "mean", "ffill", "bfill"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        if np.any(np.isnan(y_new)):
            raise ValueError("Imputation failed to fill all NaN values")
        
        # Validate that imputation was successful
        try:
            if not np.any(np.isnan(y_new)):
                return
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Output validation failed: {str(e)}")

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
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "float":
                    parsed[name] = float(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply missing value imputation to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_final = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            # Create descriptive suffix
            method_names = {
                'zero': 'ZeroFill',
                'mean': 'MeanFill', 
                'ffill': 'FFill',
                'bfill': 'BFill'
            }
            suffix = method_names.get(params["method"], 'Imputed')
            
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_final, 
                params=params,
                suffix=suffix
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Missing value imputation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for missing value imputation"""
        method = params["method"]
        
        # Check if there are any missing values
        if not np.any(np.isnan(y)):
            return y
        
        if method == "zero":
            return np.nan_to_num(y, nan=0.0)
        elif method == "mean":
            mean_val = np.nanmean(y)
            if np.isnan(mean_val):
                raise ValueError("Cannot compute mean for imputation - all finite values are NaN")
            return np.nan_to_num(y, nan=mean_val)
        elif method == "ffill":
            # Forward fill using pandas if available
            try:
                import pandas as pd
                filled = pd.Series(y).fillna(method='ffill')
                # If first values are NaN, fill with first valid value
                if pd.isna(filled.iloc[0]):
                    filled = filled.fillna(method='bfill')
                # If still NaN, fill with 0
                filled = filled.fillna(0)
                return filled.values
            except ImportError:
                raise ValueError("Forward fill imputation requires pandas library")
        elif method == "bfill":
            # Backward fill using pandas if available
            try:
                import pandas as pd
                filled = pd.Series(y).fillna(method='bfill')
                # If last values are NaN, fill with last valid value
                if pd.isna(filled.iloc[-1]):
                    filled = filled.fillna(method='ffill')
                # If still NaN, fill with 0
                filled = filled.fillna(0)
                return filled.values
            except ImportError:
                raise ValueError("Backward fill imputation requires pandas library")
        else:
            raise ValueError(f"Unknown imputation method: {method}")
