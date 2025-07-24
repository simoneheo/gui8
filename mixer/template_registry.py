"""
Template Registry for Signal Mixer Operations

This module centralizes all mixer templates that were previously hardcoded in the UI.
Templates are organized by category and can be retrieved by mixer type or category.
"""

from typing import Dict, List, Tuple, Optional

class TemplateRegistry:
    """Registry for mixer operation templates organized by category and mixer type."""
    
    # All templates organized by category
    # Format: (template_expression, mixer_type, description)
    TEMPLATES = {
        "Arithmetic": [
            ("A + B", "arithmetic", "Sum of both signals"),
            ("A - B", "arithmetic", "Difference (A minus B)"),
            ("A * B", "arithmetic", "Element-wise multiplication"),
            ("A / B", "arithmetic", "Element-wise division"),
            ("A % B", "arithmetic", "Modulo operation (great for cyclic signals)"),
            ("(A + B + C) / 3", "expression", "Three-signal average"),
            ("A * B + C", "expression", "Multiply A and B, then add C"),
            ("(A + B) * C", "expression", "Add A and B, then multiply by C"),
            ("A**2 + B**2 + C**2", "expression", "Sum of squares for three signals"),
            ("sqrt(A**2 + B**2 + C**2)", "expression", "3D vector magnitude"),
        ],
        
        "Expression": [
            ("A**2 + B**2", "expression", "Sum of squares"),
            ("sqrt(A**2 + B**2)", "expression", "Vector magnitude / Euclidean norm"),
            ("A * sin(B)", "expression", "Amplitude modulation with sine"),
            ("A * cos(B)", "expression", "Amplitude modulation with cosine"),
            ("A * tan(B)", "expression", "Amplitude modulation with tangent"),
            ("exp(A / max(abs(A)))", "expression", "Normalized exponential"),
            ("log(abs(A) + 1)", "expression", "Logarithmic transform"),
            ("log10(abs(A) + 1)", "expression", "Base-10 logarithmic transform"),
            ("np.where(A > B, A, B)", "expression", "Flexible maximum (same as max(A, B))"),
            ("np.where(A < B, A, np.nan)", "expression", "Keep lower values only (NaN the rest)"),
            ("np.where(A > 0, A, 0)", "expression", "Rectify signal (keep positive values)"),
            ("np.angle(A + 1j * B)", "expression", "Phase angle of (A, B) as complex vector"),
            ("np.arctan2(B, A)", "expression", "Quadrant-aware angle (e.g., rotation vector)"),
            ("mean([A, B])", "expression", "Element-wise mean"),
            ("std([A, B])", "expression", "Element-wise standard deviation"),
            ("var([A, B])", "expression", "Element-wise variance"),
            ("A * exp(-B)", "expression", "Exponential decay modulation"),
            ("A * (1 + sin(B))", "expression", "Sinusoidal amplitude modulation"),
            ("A * cos(B) + B * sin(A)", "expression", "Complex modulation pattern"),
            ("np.power(A, B)", "expression", "A raised to power B"),
            ("np.sign(A) * np.sqrt(abs(A))", "expression", "Signed square root"),
            ("A * np.cos(np.pi * B)", "expression", "Cosine modulation with π scaling"),
            ("A * np.sin(2 * np.pi * B)", "expression", "Sine modulation with 2π scaling"),
        ],
        
        "Logic": [
            ("A > B", "logic", "Greater than comparison"),
            ("A < B", "logic", "Less than comparison"),
            ("A >= B", "logic", "Greater than or equal comparison"),
            ("A <= B", "logic", "Less than or equal comparison"),
            ("A == B", "logic", "Equality comparison"),
            ("A != B", "logic", "Inequality comparison"),
            ("(A > 0.3) & (A < 0.7)", "logic", "Detect range bounds (binary mask)"),
            ("np.logical_and(A > 0.5, B < 0.2)", "logic", "Logical AND operation"),
            ("np.logical_or(A > 0.8, B < 0.1)", "logic", "Logical OR operation"),
            ("np.logical_xor(A > 0.5, B > 0.5)", "logic", "Logical XOR operation"),
            ("np.logical_not(A > 0.5)", "logic", "Logical NOT operation"),
            ("(A > B) & (B > C)", "logic", "Cascaded comparison (A > B > C)"),
            ("(A > 0.5) | (B > 0.5) | (C > 0.5)", "logic", "Any signal above threshold"),
            ("(A > 0.5) & (B > 0.5) & (C > 0.5)", "logic", "All signals above threshold"),
            ("np.isclose(A, B, atol=0.01)", "logic", "Approximate equality (tolerance 0.01)"),
            ("np.isnan(A) | np.isnan(B)", "logic", "Detect any NaN values"),
            ("np.isfinite(A) & np.isfinite(B)", "logic", "Detect finite values only"),
        ],
        
        "Threshold": [
            ("A * (A > 0.5)", "threshold", "Keep values above threshold"),
            ("A * (B > 0.5)", "threshold", "Mask A where B exceeds threshold"),
            ("A * (A > B)", "threshold", "Keep A where A exceeds B"),
            ("A * (abs(A) > 0.5)", "threshold", "Keep values with absolute value above threshold"),
            ("A * (B < 0.5)", "threshold", "Mask A where B is below threshold"),
            ("A * (A > 0.1) & (A < 0.9)", "threshold", "Keep A in range [0.1, 0.9]"),
            ("A * (abs(B) > 0.2)", "threshold", "Mask A where B magnitude exceeds threshold"),
            ("A * (B > np.mean(B))", "threshold", "Mask A where B exceeds its mean"),
            ("A * (B > np.percentile(B, 75))", "threshold", "Mask A where B exceeds 75th percentile"),
            ("A * (B < np.percentile(B, 25))", "threshold", "Mask A where B below 25th percentile"),
            ("A * (abs(A - B) < 0.1)", "threshold", "Keep A where A and B are similar"),
            ("A * (np.std([A, B]) < 0.05)", "threshold", "Keep A where signals have low variance"),
            ("A * (A > np.max(B) * 0.8)", "threshold", "Keep A above 80% of B's maximum"),
            ("A * (B > np.min(B) + 0.5 * (np.max(B) - np.min(B)))", "threshold", "Keep A where B is in upper half"),
        ],
        
        "Masking": [
            ("A * np.isnan(B)", "masking", "Show A where B is invalid"),
            ("A * (B > 0.2) & (B < 0.8)", "masking", "Mask A for range bounds on B"),
            ("A * (abs(B) < 0.1)", "masking", "Mask A where B magnitude is small"),
            ("A * np.isfinite(B)", "masking", "Mask A where B is finite"),
            ("A * (B != 0)", "masking", "Mask A where B is non-zero"),
            ("A * (abs(B - np.mean(B)) < np.std(B))", "masking", "Mask A where B is within 1 standard deviation"),
            ("A * (B > np.median(B))", "masking", "Mask A where B exceeds its median"),
            ("A * (B > np.percentile(B, 90))", "masking", "Mask A where B is in top 10%"),
            ("A * (B < np.percentile(B, 10))", "masking", "Mask A where B is in bottom 10%"),
            ("A * (np.abs(B - np.median(B)) < np.percentile(np.abs(B - np.median(B)), 50))", "masking", "Mask A where B is close to median"),
            ("A * (np.diff(B) > 0)", "masking", "Mask A where B is increasing"),
            ("A * (np.diff(B) < 0)", "masking", "Mask A where B is decreasing"),
        ],
        
        "Unary": [
            ("abs(A)", "unary", "Absolute value"),
            ("-A", "unary", "Negation"),
            ("A**2", "unary", "Square"),
            ("A / max(abs(A))", "unary", "Normalize to [-1, 1] range"),
            ("A / np.std(A)", "unary", "Standardize (z-score normalization)"),
            ("(A - np.min(A)) / (np.max(A) - np.min(A))", "unary", "Min-max normalization to [0, 1]"),
            ("np.sign(A)", "unary", "Sign function (-1, 0, 1)"),
            ("np.floor(A)", "unary", "Floor function (round down)"),
            ("np.ceil(A)", "unary", "Ceiling function (round up)"),
            ("np.round(A)", "unary", "Round to nearest integer"),
            ("np.sin(A)", "unary", "Sine function"),
            ("np.cos(A)", "unary", "Cosine function"),
            ("np.tan(A)", "unary", "Tangent function"),
            ("np.arcsin(A)", "unary", "Inverse sine function"),
            ("np.arccos(A)", "unary", "Inverse cosine function"),
            ("np.arctan(A)", "unary", "Inverse tangent function"),
            ("np.sinh(A)", "unary", "Hyperbolic sine function"),
            ("np.cosh(A)", "unary", "Hyperbolic cosine function"),
            ("np.tanh(A)", "unary", "Hyperbolic tangent function"),
            ("np.exp(A)", "unary", "Exponential function"),
            ("np.log(A + 1)", "unary", "Natural logarithm (with +1 offset)"),
            ("np.log10(A + 1)", "unary", "Base-10 logarithm (with +1 offset)"),
            ("np.log2(A + 1)", "unary", "Base-2 logarithm (with +1 offset)"),
            ("np.power(A, 3)", "unary", "Cube function"),
            ("np.power(A, 0.25)", "unary", "Fourth root function"),
            ("np.reciprocal(A + 1e-10)", "unary", "Reciprocal function (with small offset)"),
            ("np.cumsum(A)", "unary", "Cumulative sum"),
            ("np.cumprod(A + 1)", "unary", "Cumulative product (with +1 offset)"),
            ("np.diff(A)", "unary", "First difference"),
            ("np.gradient(A)", "unary", "Numerical gradient"),
            ("np.clip(A, -1, 1)", "unary", "Clip values to [-1, 1] range"),
            ("np.clip(A, 0, 1)", "unary", "Clip values to [0, 1] range"),
        ]
    }
    
    @classmethod
    def get_all_templates(cls) -> Dict[str, List[Tuple[str, str, str]]]:
        """Get all templates organized by category."""
        return cls.TEMPLATES.copy()
    
    @classmethod
    def get_templates_by_category(cls, category: str) -> List[Tuple[str, str, str]]:
        """Get templates for a specific category."""
        return cls.TEMPLATES.get(category, [])
    
    @classmethod
    def get_templates_by_mixer_type(cls, mixer_type: str) -> List[Tuple[str, str, str]]:
        """Get all templates that use a specific mixer type."""
        templates = []
        for category, template_list in cls.TEMPLATES.items():
            for template, mtype, description in template_list:
                if mtype == mixer_type:
                    templates.append((template, category, description))
        return templates
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """Get list of all available categories."""
        return list(cls.TEMPLATES.keys())
    
    @classmethod
    def get_templates_for_ui(cls, category_filter: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Get templates formatted for UI display.
        Returns list of (template_expression, category) tuples.
        """
        templates = []
        
        if category_filter and category_filter != "All":
            # Get templates for specific category
            if category_filter in cls.TEMPLATES:
                for template, mixer_type, description in cls.TEMPLATES[category_filter]:
                    templates.append((template, mixer_type))
        else:
            # Get all templates
            for category, template_list in cls.TEMPLATES.items():
                for template, mixer_type, description in template_list:
                    templates.append((template, mixer_type))
        
        return templates
    
    @classmethod
    def get_template_description(cls, template_expression: str) -> Optional[str]:
        """Get description for a specific template expression."""
        for category, template_list in cls.TEMPLATES.items():
            for template, mixer_type, description in template_list:
                if template == template_expression:
                    return description
        return None
    
    @classmethod
    def get_template_mixer_type(cls, template_expression: str) -> Optional[str]:
        """Get the recommended mixer type for a specific template expression."""
        for category, template_list in cls.TEMPLATES.items():
            for template, mixer_type, description in template_list:
                if template == template_expression:
                    return mixer_type
        return None
    
    @classmethod
    def add_custom_template(cls, category: str, template: str, mixer_type: str, description: str):
        """Add a custom template to the registry."""
        if category not in cls.TEMPLATES:
            cls.TEMPLATES[category] = []
        cls.TEMPLATES[category].append((template, mixer_type, description))
    
    @classmethod
    def get_template_guidance(cls, template_expression: str) -> Dict[str, str]:
        """Get guidance information for a specific template."""
        mixer_type = cls.get_template_mixer_type(template_expression)
        description = cls.get_template_description(template_expression)
        
        guidance = {
            "template": template_expression,
            "mixer_type": mixer_type,
            "description": description,
        }
        
        # Add specific guidance based on template patterns
        if "np.where" in template_expression:
            guidance["tip"] = "Conditional selection - flexible alternative to max/min operations"
        elif "np.angle" in template_expression:
            guidance["tip"] = "Phase angle calculation from complex vector representation"
        elif "np.arctan2" in template_expression:
            guidance["tip"] = "Quadrant-aware angle calculation (handles all four quadrants)"
        elif "np.logical_and" in template_expression:
            guidance["tip"] = "Logical AND operation for complex boolean conditions"
        elif "np.isnan" in template_expression:
            guidance["tip"] = "NaN detection and masking for data quality filtering"
        elif "%" in template_expression:
            guidance["tip"] = "Modulo operation - great for cyclic signals and phase wrapping"
        elif "mean([" in template_expression:
            guidance["tip"] = "Element-wise mean - clearer than manual averaging"
        elif "sum([" in template_expression:
            guidance["tip"] = "Element-wise sum of multiple signals"
        elif ">" in template_expression and "*" in template_expression:
            guidance["tip"] = "Threshold masking - zeros out values that don't meet condition"
        elif any(func in template_expression for func in ["sqrt", "sin", "cos", "exp", "log", "abs"]):
            guidance["tip"] = "Mathematical function application"
        elif any(op in template_expression for op in ["+", "-", "*", "/"]):
            guidance["tip"] = "Basic arithmetic operation"
        
        return guidance


# Convenience functions for backward compatibility
def get_all_templates():
    """Get all templates organized by category."""
    return TemplateRegistry.get_all_templates()

def get_templates_for_ui(category_filter=None):
    """Get templates formatted for UI display."""
    return TemplateRegistry.get_templates_for_ui(category_filter)

def get_template_description(template_expression):
    """Get description for a specific template expression."""
    return TemplateRegistry.get_template_description(template_expression)

def get_template_mixer_type(template_expression):
    """Get the recommended mixer type for a specific template expression."""
    return TemplateRegistry.get_template_mixer_type(template_expression) 