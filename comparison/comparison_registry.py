"""
Comparison Registry

This module provides a registry system for managing comparison methods,
similar to the ProcessRegistry and MixerRegistry patterns used elsewhere
in the application.
"""

import os
import importlib
import inspect
from typing import Dict, List, Type, Optional, Any
from .base_comparison import BaseComparison

class ComparisonRegistry:
    """
    Registry for managing comparison methods.
    
    This class provides functionality to discover, register, and retrieve
    comparison methods dynamically.
    """
    
    _methods: Dict[str, Type[BaseComparison]] = {}
    _categories: Dict[str, List[str]] = {}
    _initialized = False
    
    @classmethod
    def register(cls, comparison_class: Type[BaseComparison]) -> None:
        """
        Register a comparison method.
        
        Args:
            comparison_class: Class that inherits from BaseComparison
        """
        if not issubclass(comparison_class, BaseComparison):
            raise ValueError(f"Class {comparison_class.__name__} must inherit from BaseComparison")
        
        method_name = comparison_class.name
        cls._methods[method_name] = comparison_class
        
        # Add to category
        category = getattr(comparison_class, 'category', 'Uncategorized')
        if category not in cls._categories:
            cls._categories[category] = []
        if method_name not in cls._categories[category]:
            cls._categories[category].append(method_name)
        
        print(f"[ComparisonRegistry] Registered: {method_name} (category: {category})")
    
    @classmethod
    def get(cls, method_name: str) -> Optional[Type[BaseComparison]]:
        """
        Get a comparison method by name.
        
        Args:
            method_name: Name of the comparison method
            
        Returns:
            Comparison method class or None if not found
        """
        return cls._methods.get(method_name)
    
    @classmethod
    def get_all_methods(cls) -> List[str]:
        """
        Get list of all registered comparison method names.
        
        Returns:
            List of method names
        """
        return list(cls._methods.keys())
    
    @classmethod
    def get_methods_by_category(cls, category: str) -> List[str]:
        """
        Get list of comparison methods in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of method names in the category
        """
        return cls._categories.get(category, [])
    
    @classmethod
    def get_all_categories(cls) -> List[str]:
        """
        Get list of all categories.
        
        Returns:
            List of category names
        """
        return list(cls._categories.keys())
    
    @classmethod
    def get_method_info(cls, method_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a comparison method.
        
        Args:
            method_name: Name of the comparison method
            
        Returns:
            Dictionary with method information or None if not found
        """
        method_class = cls.get(method_name)
        if method_class is None:
            return None
        
        return {
            'name': method_class.name,
            'description': method_class.description,
            'category': method_class.category,
            'version': method_class.version,
            'parameters': method_class.parameters,
            'output_types': method_class.output_types,
            'class_name': method_class.__name__,
            'module': method_class.__module__
        }
    
    @classmethod
    def create_method(cls, method_name: str, **kwargs) -> Optional[BaseComparison]:
        """
        Create an instance of a comparison method.
        
        Args:
            method_name: Name of the comparison method
            **kwargs: Parameters for the method
            
        Returns:
            Instance of the comparison method or None if not found
        """
        method_class = cls.get(method_name)
        if method_class is None:
            return None
        
        try:
            return method_class(**kwargs)
        except Exception as e:
            print(f"[ComparisonRegistry] Error creating {method_name}: {e}")
            return None
    
    @classmethod
    def load_methods_from_directory(cls, directory: str) -> int:
        """
        Load comparison methods from a directory.
        
        Args:
            directory: Path to directory containing comparison method files
            
        Returns:
            Number of methods loaded
        """
        if not os.path.exists(directory):
            print(f"[ComparisonRegistry] Directory not found: {directory}")
            return 0
        
        loaded_count = 0
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # Remove .py extension
                
                try:
                    # Import the module
                    module_path = f"comparison.{module_name}"
                    module = importlib.import_module(module_path)
                    
                    # Find comparison classes in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BaseComparison) and 
                            obj != BaseComparison and 
                            obj.__module__ == module_path):
                            cls.register(obj)
                            loaded_count += 1
                            
                except Exception as e:
                    print(f"[ComparisonRegistry] Error loading {module_name}: {e}")
        
        return loaded_count
    
    @classmethod
    def initialize(cls, comparison_dir: Optional[str] = None) -> None:
        """
        Initialize the registry by loading all comparison methods.
        
        Args:
            comparison_dir: Directory containing comparison methods (defaults to 'comparison')
        """
        if cls._initialized:
            return
        
        if comparison_dir is None:
            # Get the directory of this module
            current_dir = os.path.dirname(os.path.abspath(__file__))
            comparison_dir = current_dir
        
        print(f"[ComparisonRegistry] Initializing from: {comparison_dir}")
        loaded_count = cls.load_methods_from_directory(comparison_dir)
        
        cls._initialized = True
        print(f"[ComparisonRegistry] Loaded {loaded_count} comparison methods")
        print(f"[ComparisonRegistry] Available categories: {list(cls._categories.keys())}")
    
    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing purposes)."""
        cls._methods.clear()
        cls._categories.clear()
        cls._initialized = False
    
    @classmethod
    def get_registry_stats(cls) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            'total_methods': len(cls._methods),
            'categories': {cat: len(methods) for cat, methods in cls._categories.items()},
            'initialized': cls._initialized,
            'method_list': list(cls._methods.keys())
        }
    
    @classmethod
    def search_methods(cls, query: str) -> List[str]:
        """
        Search for comparison methods by name or description.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching method names
        """
        query_lower = query.lower()
        matches = []
        
        for method_name, method_class in cls._methods.items():
            # Search in name
            if query_lower in method_name.lower():
                matches.append(method_name)
                continue
            
            # Search in description
            if hasattr(method_class, 'description') and query_lower in method_class.description.lower():
                matches.append(method_name)
                continue
            
            # Search in category
            if hasattr(method_class, 'category') and query_lower in method_class.category.lower():
                matches.append(method_name)
        
        return sorted(matches)

# Convenience function for external use
def load_all_comparisons(comparison_dir: Optional[str] = None) -> None:
    """
    Load all comparison methods from the specified directory.
    
    Args:
        comparison_dir: Directory containing comparison methods
    """
    ComparisonRegistry.initialize(comparison_dir) 