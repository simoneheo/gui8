import numpy as np
import ast
import operator
import re
try:
    from channel import Channel
except ImportError:
    # Fallback for compatibility
    Channel = None
from mixer.base_mixer import BaseMixer
from mixer.mixer_registry import register_mixer

@register_mixer
class ExpressionMixer(BaseMixer):
    name = "expression"
    category = "advanced"
    description = "Evaluates complex mathematical expressions with multiple channels and constants."
    tags = ["mixed", "expression", "advanced"]
    params = [
        {"name": "expression", "type": "str", "default": "A + B", "help": "Mathematical expression (e.g., A+B-3*D/5)"},
        {"name": "label", "type": "str", "default": "C", "help": "Output channel label"}
    ]

    # Safe operators allowed in expressions
    SAFE_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.BitAnd: operator.and_,  # For & operator
        ast.BitOr: operator.or_,    # For | operator
    }
    
    # Safe comparison operators
    SAFE_COMPARISONS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    # Safe functions allowed in expressions
    SAFE_FUNCTIONS = {
        'abs': np.abs,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'log': np.log,
        'log10': np.log10,
        'log2': np.log2,
        'exp': np.exp,
        'max': np.max,
        'min': np.min,
        'mean': np.mean,
        'std': np.std,
        'sum': np.sum,
        'var': np.var,
        'angle': np.angle,
        'arctan2': np.arctan2,
        'where': np.where,
        'logical_and': np.logical_and,
        'logical_or': np.logical_or,
        'logical_not': np.logical_not,
        'logical_xor': np.logical_xor,
        'isnan': np.isnan,
        'isinf': np.isinf,
        'isfinite': np.isfinite,
        'isclose': np.isclose,
        'cumsum': np.cumsum,
        'cumprod': np.cumprod,
        'diff': np.diff,
        'gradient': np.gradient,
        'clip': np.clip,
        'power': np.power,
        'reciprocal': np.reciprocal,
        'sign': np.sign,
        'floor': np.floor,
        'ceil': np.ceil,
        'round': np.round,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'percentile': np.percentile,
        'median': np.median,
    }

    def apply(self, channels: dict[str, Channel]) -> Channel:
        expression = self.kwargs.get("expression", "A + B")
        label = self.kwargs.get("label", "C")
        
        print(f"[ExpressionMixer] Evaluating: {expression}")
        
        # Parse and validate the expression
        try:
            parsed_expr = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")
        
        # Extract channel names from expression
        channel_names = self._extract_channel_names(expression)
        print(f"[ExpressionMixer] Found channel names: {channel_names}")
        
        # Validate all channels exist
        missing_channels = [name for name in channel_names if name not in channels]
        if missing_channels:
            raise ValueError(f"Missing channels: {missing_channels}")
        
        # Get reference channel for metadata (use first available channel)
        ref_channel = next(iter(channels.values()))
        
        # Validate all channels have compatible data
        for name in channel_names:
            channel = channels[name]
            if channel.xdata is None or channel.ydata is None:
                raise ValueError(f"Channel {name} has no data")
            if len(channel.ydata) != len(ref_channel.ydata):
                raise ValueError(f"Channel {name} has incompatible data length")
        
        # Evaluate the expression
        try:
            result = self._safe_eval(parsed_expr.body, channels)
            print(f"[ExpressionMixer] Expression evaluated successfully, result length: {len(result)}")
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")
        
        # Create the result channel
        expr_str = f"{label} = {expression}"
        return self.create_channel(
            ref_channel.xdata, 
            result, 
            list(channels.values()), 
            label=label, 
            expr=expr_str, 
            params=self.kwargs
        )

    def _extract_channel_names(self, expression):
        """Extract channel names (single uppercase letters) from expression."""
        # Find all single uppercase letters that could be channel names
        pattern = r'\b[A-Z]\b'
        channel_names = list(set(re.findall(pattern, expression)))
        return sorted(channel_names)

    def _safe_eval(self, node, channels):
        """Safely evaluate an AST node with channel data."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Name):
            # Channel name, numpy reference, function names, or special constants
            if node.id in channels:
                return channels[node.id].ydata
            elif node.id == 'np':
                # Return a namespace object for numpy
                return type('np', (), self.SAFE_FUNCTIONS)
            elif node.id in self.SAFE_FUNCTIONS:
                # Direct function name (e.g., sqrt, sin, cos)
                return self.SAFE_FUNCTIONS[node.id]
            elif node.id == 'j':
                # Imaginary unit for complex numbers
                return 1j
            else:
                raise ValueError(f"Unknown channel or variable: {node.id}")
        elif isinstance(node, ast.Attribute):
            # Handle np.function calls (e.g., np.where, np.angle) and np.constants (e.g., np.nan)
            if isinstance(node.value, ast.Name) and node.value.id == 'np':
                if node.attr in self.SAFE_FUNCTIONS:
                    return self.SAFE_FUNCTIONS[node.attr]
                elif node.attr == 'nan':
                    # Handle np.nan as a constant
                    return np.nan
                elif node.attr == 'inf':
                    # Handle np.inf as a constant
                    return np.inf
                elif node.attr == 'pi':
                    # Handle np.pi as a constant
                    return np.pi
                elif node.attr == 'e':
                    # Handle np.e as a constant
                    return np.e
                else:
                    raise ValueError(f"Unsupported numpy function or constant: np.{node.attr}")
            else:
                raise ValueError(f"Unsupported attribute access: {node.attr}")
        elif isinstance(node, ast.BinOp):
            # Binary operation (e.g., A + B, A + 1j * B for complex numbers)
            left = self._safe_eval(node.left, channels)
            right = self._safe_eval(node.right, channels)
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            
            # Handle complex number creation (e.g., A + 1j * B)
            if isinstance(right, complex) or isinstance(left, complex):
                # Convert arrays to complex if needed
                if hasattr(left, 'astype') and not np.iscomplexobj(left):
                    left = left.astype(complex)
                if hasattr(right, 'astype') and not np.iscomplexobj(right):
                    right = right.astype(complex)
            
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            # Unary operation (e.g., -A)
            operand = self._safe_eval(node.operand, channels)
            op = self.SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Call):
            # Function call (e.g., abs(A), np.where(A > B, A, B))
            func = self._safe_eval(node.func, channels)
            if not callable(func):
                raise ValueError(f"Not a callable function")
            
            args = [self._safe_eval(arg, channels) for arg in node.args]
            
            # Handle special cases
            if func == np.angle and len(args) == 1:
                # For np.angle(A + 1j * B) pattern, we need to handle complex numbers
                arg = args[0]
                if hasattr(arg, 'real') and hasattr(arg, 'imag'):
                    return func(arg)
                else:
                    # If it's not already complex, treat as real part only
                    return func(arg.astype(complex))
            elif func in [np.mean, np.sum] and len(args) == 1 and isinstance(args[0], list):
                # Handle mean([A, B]) and sum([A, B]) patterns
                array_list = args[0]
                if len(array_list) > 0:
                    # Stack arrays and apply function along axis 0
                    stacked = np.stack(array_list, axis=0)
                    return func(stacked, axis=0)
                else:
                    raise ValueError("Empty list provided to function")
            
            return func(*args)
        elif isinstance(node, ast.Compare):
            # Comparison operations (e.g., A > B, A == B)
            left = self._safe_eval(node.left, channels)
            
            # Handle multiple comparisons like a < b < c
            result = left
            for op, comp in zip(node.ops, node.comparators):
                right = self._safe_eval(comp, channels)
                comp_op = self.SAFE_COMPARISONS.get(type(op))
                if comp_op is None:
                    raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
                result = comp_op(result, right)
                result = result  # Keep the result for chained comparisons
            
            return result.astype(float)  # Convert boolean result to float (0.0 or 1.0)
        elif isinstance(node, ast.BoolOp):
            # Boolean operations (e.g., A & B, A | B)
            if isinstance(node.op, ast.And):
                result = self._safe_eval(node.values[0], channels)
                for value in node.values[1:]:
                    result = result & self._safe_eval(value, channels)
                return result.astype(float)
            elif isinstance(node.op, ast.Or):
                result = self._safe_eval(node.values[0], channels)
                for value in node.values[1:]:
                    result = result | self._safe_eval(value, channels)
                return result.astype(float)
        elif isinstance(node, ast.List):
            # Handle list literals [A, B] for functions like mean([A, B])
            return [self._safe_eval(item, channels) for item in node.elts]
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    @classmethod
    def parse_expression_for_mixer(cls, expression):
        """Parse an expression to determine if it should use ExpressionMixer."""
        # Check if expression contains multiple operators or functions
        operators = ['+', '-', '*', '/', '**']
        functions = list(cls.SAFE_FUNCTIONS.keys())
        
        op_count = sum(expression.count(op) for op in operators)
        func_count = sum(expression.count(f + '(') for f in functions)
        
        # Use ExpressionMixer if:
        # 1. Multiple operators (e.g., A+B-C)
        # 2. Contains functions (e.g., abs(A))
        # 3. Contains constants mixed with operations (e.g., A+3*B)
        has_constants = bool(re.search(r'\d', expression))
        
        return op_count > 1 or func_count > 0 or (op_count >= 1 and has_constants)

    @classmethod
    def get_expression_templates(cls):
        """Get common expression templates for UI."""
        return [
            "A + B",
            "A - B", 
            "A * B",
            "A / B",
            "A + B - C",
            "A * B + C",
            "(A + B) / 2",
            "A + 3 * B",
            "A - 2 * B + C",
            "abs(A - B)",
            "sqrt(A * A + B * B)",
            "A / max(A)",
            "(A - mean(A)) / std(A)",
            "A * (A > B)",
            "A * (abs(A) > std(A))",
            "A > mean(A)",
            "(A > B) * C"
        ]

    @classmethod
    def get_mixer_guidance(cls):
        """Get specific guidance text for this mixer type."""
        return {
            "title": "Advanced Expression Evaluation",
            "description": "Evaluates complex mathematical expressions with multiple channels, constants, and functions using safe expression parsing.",
            "use_cases": [
                "Complex multi-channel computations",
                "Mathematical modeling and transformations",
                "Statistical analysis combinations",
                "Custom formula implementations",
                "Multi-step calculations in single expressions"
            ],
            "tips": [
                "Use parentheses for operation precedence: (A + B) * C",
                "Mix channels with constants: A + 3 * B - 2",
                "Apply functions: sqrt(A**2 + B**2), abs(A - B)",
                "Statistical operations: (A - mean(A)) / std(A)",
                "Comparison masks: A * (abs(A) > std(A))",
                "Complex expressions: A * sin(B) + C * cos(D)"
            ],
            "common_patterns": [
                "Vector magnitude: sqrt(A**2 + B**2)",
                "Weighted average: (A * w1 + B * w2) / (w1 + w2)",
                "Z-score normalization: (A - mean(A)) / std(A)",
                "Signal envelope: abs(A + 1j*B)",
                "Threshold masking: A * (abs(A) > std(A))",
                "Binary thresholding: A > mean(A)",
                "Custom transforms: A * exp(-B/C)"
            ]
        } 