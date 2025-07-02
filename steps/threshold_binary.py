import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class threshold_binary_step(BaseStep):
    name = "threshold_binary"
    category = "Arithmetic"
    description = "Convert signal to binary (0/1) values based on threshold comparison"
    tags = ["time-series"]
    params = [
        {
            "name": "threshold", 
            "type": "float", 
            "default": "0.0", 
            "help": "Threshold value for comparison"
        },
        {
            "name": "operator", 
            "type": "str", 
            "default": ">", 
            "help": "Comparison operator",
            "options": [">", ">=", "<", "<=", "==", "!="]
        },
    ]

    @classmethod
    def get_info(cls): return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): return {"info": cls.description, "params": cls.params}

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        try:
            threshold = float(user_input.get("threshold", "0.0"))
            parsed["threshold"] = threshold
        except ValueError:
            raise ValueError("Threshold must be a valid number")
        
        operator = user_input.get("operator", ">")
        if operator not in [">", ">=", "<", "<=", "==", "!="]:
            raise ValueError("Operator must be one of: >, >=, <, <=, ==, !=")
        parsed["operator"] = operator
        
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        y = channel.ydata
        x = channel.xdata
        
        # Validate input data
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        
        threshold = params['threshold']
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
                y_new = np.where(np.isclose(y, threshold), 1, 0)
            elif operator == "!=":
                y_new = np.where(~np.isclose(y, threshold), 1, 0)
            else:
                raise ValueError(f"Unknown operator: {operator}")
                
            return cls.create_new_channel(parent=channel, xdata=x, ydata=y_new, params=params)
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Threshold binary operation failed: {str(e)}")
