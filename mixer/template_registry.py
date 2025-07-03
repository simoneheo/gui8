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
            ("(A + B) / 2", "expression", "Element-wise average"),
            ("abs(A - B)", "expression", "Absolute difference"),
            ("A * (A >= B) + B * (B > A)", "expression", "Element-wise maximum"),
            ("A * (A <= B) + B * (B < A)", "expression", "Element-wise minimum"),
            ("A % B", "arithmetic", "Modulo operation (great for cyclic signals)"),
        ],
        
        "Expression": [
            ("A**2 + B**2", "expression", "Sum of squares"),
            ("sqrt(A**2 + B**2)", "expression", "Vector magnitude / Euclidean norm"),
            ("A * sin(B)", "expression", "Amplitude modulation with sine"),
            ("A * cos(B)", "expression", "Amplitude modulation with cosine"),
            ("exp(A / max(abs(A)))", "expression", "Normalized exponential"),
            ("log(abs(A) + 1)", "expression", "Logarithmic transform"),
            ("np.where(A > B, A, B)", "expression", "Flexible maximum (same as max(A, B))"),
            ("np.where(A < B, A, np.nan)", "expression", "Keep lower values only (NaN the rest)"),
            ("np.angle(A + 1j * B)", "expression", "Phase angle of (A, B) as complex vector"),
            ("np.arctan2(B, A)", "expression", "Quadrant-aware angle (e.g., rotation vector)"),
            ("mean([A, B])", "expression", "Element-wise mean (clearer than (A + B)/2)"),
            ("sum([A, B])", "expression", "Sum of both signals"),
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
        ],
        
        "Threshold": [
            ("A * (A > 0.5)", "threshold", "Keep values above threshold"),
            ("A * (B > 0.5)", "threshold", "Mask A where B exceeds threshold"),
            ("A * (A > B)", "threshold", "Keep A where A exceeds B"),
            ("A * (abs(A) > 0.5)", "threshold", "Keep values with absolute value above threshold"),
            ("A * (B < 0.3)", "threshold", "Mask A where B is below threshold"),
            ("A * (B < 0.5)", "threshold", "Mask A where B is below a threshold"),
        ],
        
        "Masking": [
            ("A * (B < 0.5)", "masking", "Mask A where B is below threshold"),
            ("A * np.isnan(B)", "masking", "Show A where B is invalid"),
            ("A * (B > 0.2) & (B < 0.8)", "masking", "Mask A for range bounds on B"),
            ("A * (abs(B) < 0.1)", "masking", "Mask A where B magnitude is small"),
        ],
        
        "Unary": [
            ("abs(A)", "unary", "Absolute value"),
            ("sqrt(abs(A))", "unary", "Square root of absolute value"),
            ("-A", "unary", "Negation"),
            ("A**2", "unary", "Square"),
            ("A / max(abs(A))", "unary", "Normalize to [-1, 1] range"),
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