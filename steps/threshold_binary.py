import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class threshold_binary_step(BaseStep):
    name = "threshold_binary"
    category = "Arithmetic"
    description = """Convert signal to binary (0/1) values based on threshold comparison.
    
This step creates a binary signal where:
• Values meeting the threshold condition become 1
• Values not meeting the threshold condition become 0

Common applications:
• **Event detection**: Identify when signals cross specific levels
• **State detection**: Convert continuous signals to on/off states  
• **Trigger generation**: Create digital triggers from analog signals
• **Signal conditioning**: Prepare signals for digital processing

The comparison operators work as follows:
• **>**: Values greater than threshold → 1
• **>=**: Values greater than or equal to threshold → 1  
• **<**: Values less than threshold → 1
• **<=**: Values less than or equal to threshold → 1
• **==**: Values equal to threshold (within tolerance) → 1
• **!=**: Values not equal to threshold → 1"""
    tags = ["time-series", "binary", "threshold", "detection", "decision", "classify"]
    params = [
        {
            "name": "threshold", 
            "type": "float", 
            "default": "0.0", 
            "help": "Threshold value for comparison. Choose based on your signal characteristics and detection requirements."
        },
        {
            "name": "operator", 
            "type": "str", 
            "default": ">", 
            "help": "Comparison operator to use for threshold evaluation.",
            "options": [">", ">=", "<", "<=", "==", "!="]
        },
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — Convert signal to binary (0/1) based on threshold comparison for event/state detection (Category: {cls.category})"

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

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        threshold = params.get("threshold")
        operator = params.get("operator", ">")
        
        if np.isnan(threshold):
            raise ValueError("Threshold cannot be NaN")
        if np.isinf(threshold):
            raise ValueError("Threshold cannot be infinite")
        
        valid_operators = [">", ">=", "<", "<=", "==", "!="]
        if operator not in valid_operators:
            raise ValueError(f"Operator must be one of {valid_operators}, got '{operator}'")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        if len(y_new) != len(y_original):
            raise ValueError("Output signal length differs from input")
        
        # Ensure output is actually binary
        unique_values = np.unique(y_new[~np.isnan(y_new)])
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(f"Threshold operation produced non-binary values: {unique_values}")

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
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply threshold binary conversion to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_new = cls.script(x, y, params)
            
            # Validate output data
            cls._validate_output_data(y, y_new)
            
            # Create descriptive suffix
            threshold = params.get("threshold", 0.0)
            operator = params.get("operator", ">")
            suffix = f"Thresh{operator}{threshold:g}"
            
            # Create output channel
            new_channel = cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_new, 
                params=params,
                suffix=suffix
            )
            
            # Set appropriate labels for binary signal
            new_channel.ylabel = "Binary State"
            new_channel.legend_label = f"{channel.legend_label} - Binary ({operator}{threshold:g})"
            
            return new_channel
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Threshold binary operation failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, params: dict) -> np.ndarray:
        """Core processing logic for threshold binary conversion"""
        threshold = params.get("threshold", 0.0)
        operator = params.get("operator", ">")
        
        # Apply threshold with specified operator
        if operator == ">":
            y_new = np.where(y > threshold, 1, 0)
        elif operator == ">=":
            y_new = np.where(y >= threshold, 1, 0)
        elif operator == "<":
            y_new = np.where(y < threshold, 1, 0)
        elif operator == "<=":
            y_new = np.where(y <= threshold, 1, 0)
        elif operator == "==":
            # Use tolerance for floating point comparison
            tolerance = np.finfo(float).eps * 100
            y_new = np.where(np.abs(y - threshold) <= tolerance, 1, 0)
        elif operator == "!=":
            # Use tolerance for floating point comparison
            tolerance = np.finfo(float).eps * 100
            y_new = np.where(np.abs(y - threshold) > tolerance, 1, 0)
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return y_new
