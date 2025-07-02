import numpy as np
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

def derivative(y, method='gradient', order=1):
    """
    Computes the numerical derivative of the input signal `y`.

    Parameters:
    - method: 'gradient' (numpy.gradient) or 'diff' (numpy.diff) 
    - order: Order of derivative (1 for first derivative, 2 for second, etc.)

    Returns:
    - The derivative of `y` with respect to time
    """
    # Parameter validation
    if len(y) < 2:
        raise ValueError(f"Signal must have at least 2 samples for derivative computation, got {len(y)}")
    if order < 1:
        raise ValueError(f"Derivative order must be at least 1, got {order}")
    if len(y) <= order:
        raise ValueError(f"Signal length ({len(y)}) must be greater than derivative order ({order})")
    
    try:
        if method == 'gradient':
            result = y.copy()
            for _ in range(order):
                result = np.gradient(result)
            return result
        elif method == 'diff':
            result = y.copy()
            for _ in range(order):
                result = np.diff(result)
                # Pad to maintain length
                result = np.concatenate([result, [result[-1]]])
            return result
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        raise ValueError(f"Derivative computation failed: {str(e)}. Check input signal.")

@register_step
class derivative_step(BaseStep):
    name = "derivative"
    category = "Arithmetic"
    description = "Computes the numerical derivative of the input signal"
    tags = ["time-series"]
    params = [
        {
            "name": "method", 
            "type": "str", 
            "default": "gradient", 
            "help": "Numerical differentiation method",
            "options": ["gradient", "diff"]
        },
        {
            "name": "order", 
            "type": "int", 
            "default": "1", 
            "help": "Order of derivative (1=first, 2=second, etc.)",
            "options": ["1", "2", "3"]
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {
            "info": cls.description,
            "params": cls.params
        }

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        parsed = {}
        
        # Parse method
        method = user_input.get("method", "gradient")
        if method not in ["gradient", "diff"]:
            raise ValueError("Method must be 'gradient' or 'diff'")
        parsed["method"] = method
        
        # Parse order
        try:
            order = int(user_input.get("order", "1"))
            if order < 1:
                raise ValueError("Derivative order must be at least 1")
            if order > 5:
                raise ValueError("Derivative order too high (maximum 5)")
            parsed["order"] = order
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Order must be a valid integer")
            raise e
            
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
        
        order = params.get("order", 1)
        if len(y) <= order:
            raise ValueError(f"Signal too short: requires > {order} samples for order {order} derivative, got {len(y)}")

        try:
            y_new = derivative(y, **params)
            x_new = np.linspace(x[0], x[-1], len(y_new))

            return cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Derivative computation failed: {str(e)}")
