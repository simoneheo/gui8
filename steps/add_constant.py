import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class add_constant_step(BaseStep):
    name = "add_constant"
    category = "Arithmetic"
    description = "Add a constant value to all signal values for baseline adjustment"
    tags = ["time-series", "arithmetic", "offset", "bias", "constant", "add"]
    params = [
        {
            "name": "constant", 
            "type": "float", 
            "default": "0.0", 
            "help": "Constant value to add to all signal values. Positive values shift up, negative values shift down."
        }
    ]

    @classmethod
    def get_info(cls): 
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls): 
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if y is None:
            raise ValueError("Input signal cannot be None")
        if len(y) == 0:
            raise ValueError("Input signal is empty")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(np.isinf(y)):
            raise ValueError("Signal contains only infinite values")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        constant = params.get("constant")
        if constant is None:
            raise ValueError("Constant parameter is required")
        if np.isnan(constant):
            raise ValueError("Constant cannot be NaN")
        if np.isinf(constant):
            raise ValueError("Constant cannot be infinite")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, y_new: np.ndarray) -> None:
        """Validate output signal data"""
        BaseStep.validate_output_data(y_original, y_new)

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param["default"])
            try:
                if val == "":
                    parsed[name] = 0.0  # Default to 0.0 for empty string
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                elif param["type"] == "bool":
                    parsed[name] = bool(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply the processing step to a channel"""
        try:
            print(f"[AddConstant] Starting apply with params: {params}")
            
            x = channel.xdata
            y = channel.ydata
            fs = getattr(channel, 'fs', None)
            
            print(f"[AddConstant] Input data shape: {y.shape if hasattr(y, 'shape') else len(y)}")
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            y_final = cls.script(x, y, fs, params)
            
            print(f"[AddConstant] Output data shape: {y_final.shape if hasattr(y_final, 'shape') else len(y_final)}")
            
            # Validate output data
            cls._validate_output_data(y, y_final)
            
            # Generate appropriate suffix for new channel name
            constant = params.get('constant', 0.0)
            if constant >= 0:
                suffix = f"Plus{constant:g}"
            else:
                suffix = f"Minus{abs(constant):g}"
            
            print(f"[AddConstant] Creating channel with suffix: {suffix}")
                
            return cls.create_new_channel(
                parent=channel, 
                xdata=x, 
                ydata=y_final, 
                params=params,
                suffix=suffix
            )
        except Exception as e:
            print(f"[AddConstant] Error in apply: {e}")
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Add constant processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> np.ndarray:
        """Core processing logic"""
        constant = params["constant"]
        y_new = y + constant  
        
        return y_new