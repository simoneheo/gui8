import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def impute_missing(y, method='zero'):
    """
    Impute missing (NaN) values in the signal using various methods.
    
    Args:
        y: Input signal array
        method: Imputation method ('zero', 'mean', 'ffill', 'bfill')
    
    Returns:
        numpy array with missing values imputed
        
    Raises:
        ValueError: If method is unknown or imputation fails
    """
    # Input validation
    if y is None:
        raise ValueError("Input signal cannot be None")
    
    y = np.asarray(y)
    
    if len(y) == 0:
        raise ValueError("Input signal cannot be empty")
    
    # Check if there are any missing values
    if not np.any(np.isnan(y)):
        return y
    
    # Check if all values are NaN
    if np.all(np.isnan(y)):
        raise ValueError("Cannot impute when all values are NaN")
    
    try:
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
            
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        else:
            raise ValueError(f"Imputation failed with method '{method}': {str(e)}")

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
    tags = ["time-series", "data-quality", "preprocessing"]
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
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        try:
            # Parse method parameter
            method = user_input.get("method", "zero")
            
            # Handle string input (strip if it's a string)
            if isinstance(method, str):
                method = method.strip().lower()
            else:
                raise ValueError(f"Method must be a string, got {type(method).__name__}: {method}")
            
            # Validate method
            valid_methods = ["zero", "mean", "ffill", "bfill"]
            if method not in valid_methods:
                raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
            
            parsed["method"] = method
            return parsed
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to parse imputation method: {str(e)}")

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        # Input validation
        if channel is None:
            raise ValueError("Input channel cannot be None")
        
        if not hasattr(channel, 'ydata') or channel.ydata is None:
            raise ValueError("Channel must have valid signal data (ydata)")
        
        if not hasattr(channel, 'xdata') or channel.xdata is None:
            raise ValueError("Channel must have valid time data (xdata)")
        
        y = channel.ydata
        x = channel.xdata
        
        # Data validation
        if len(y) == 0:
            raise ValueError("Signal data cannot be empty")
        
        if len(x) != len(y):
            raise ValueError(f"Time and signal data length mismatch: {len(x)} vs {len(y)}")
        
        # Extract parameters
        method = params.get('method', 'zero')
        
        try:
            # Apply imputation
            y_new = impute_missing(y, method=method)
            
            # Validate output
            if len(y_new) != len(y):
                raise ValueError(f"Imputation changed signal length: {len(y)} -> {len(y_new)}")
            
            # Check that NaN values were actually filled
            if np.any(np.isnan(y_new)):
                raise ValueError(f"Imputation method '{method}' failed to fill all NaN values")
            
            # Create time array matching output length
            x_new = np.linspace(x[0], x[-1], len(y_new)) if len(x) > 1 else x
            
            # Create descriptive suffix
            method_names = {
                'zero': 'ZeroFill',
                'mean': 'MeanFill', 
                'ffill': 'FFill',
                'bfill': 'BFill'
            }
            suffix = method_names.get(method, 'Imputed')
            
            # Create output channel
            return cls.create_new_channel(
                parent=channel, 
                xdata=x_new, 
                ydata=y_new, 
                params=params,
                suffix=suffix
            )
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Missing value imputation failed: {str(e)}")
