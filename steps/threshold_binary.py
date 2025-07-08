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
    tags = ["time-series", "binary", "threshold", "detection"]
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
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        try:
            # Parse threshold parameter
            threshold_val = user_input.get("threshold", "0.0")
            
            # Handle both string and numeric inputs
            if isinstance(threshold_val, str):
                threshold_val = threshold_val.strip()
                if not threshold_val:
                    raise ValueError("Threshold parameter cannot be empty")
                try:
                    threshold = float(threshold_val)
                except ValueError:
                    raise ValueError(f"Threshold must be a valid number, got '{threshold_val}'")
            elif isinstance(threshold_val, (int, float)):
                threshold = float(threshold_val)
            else:
                raise ValueError(f"Threshold must be a number, got {type(threshold_val).__name__}: {threshold_val}")
            
            # Validate threshold value
            if np.isnan(threshold):
                raise ValueError("Threshold cannot be NaN")
            if np.isinf(threshold):
                raise ValueError("Threshold cannot be infinite")
            
            parsed["threshold"] = threshold
            
            # Parse operator parameter
            operator = user_input.get("operator", ">")
            
            # Handle string input (strip if it's a string)
            if isinstance(operator, str):
                operator = operator.strip()
            else:
                raise ValueError(f"Operator must be a string, got {type(operator).__name__}: {operator}")
            
            # Validate operator
            valid_operators = [">", ">=", "<", "<=", "==", "!="]
            if operator not in valid_operators:
                raise ValueError(f"Operator must be one of {valid_operators}, got '{operator}'")
            
            parsed["operator"] = operator
            
            return parsed
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to parse threshold binary parameters: {str(e)}")

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
        
        # Check for all NaN values
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        
        # Extract parameters
        threshold = params.get('threshold', 0.0)
        operator = params.get('operator', '>')
        
        try:
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
            
            # Validate output
            if len(y_new) != len(y):
                raise ValueError(f"Threshold operation changed signal length: {len(y)} -> {len(y_new)}")
            
            # Ensure output is actually binary
            unique_values = np.unique(y_new[~np.isnan(y_new)])
            if not np.all(np.isin(unique_values, [0, 1])):
                raise ValueError(f"Threshold operation produced non-binary values: {unique_values}")
            
            # Create descriptive suffix
            suffix = f"Thresh{operator}{threshold:g}"
            
            # Create output channel with appropriate labels
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
