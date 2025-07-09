"""
Comparison Registry

This module provides a registry system for managing comparison methods,
following the same pattern as MixerRegistry and ProcessRegistry.
"""

import os
import importlib
import inspect
from typing import Type, Dict, List
from comparison.base_comparison import BaseComparison

class _ComparisonRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseComparison]] = {}
        self._initialized = False

    def register(self, comparison_cls: Type[BaseComparison]):
        """Registers a comparison class by its name."""
        if not issubclass(comparison_cls, BaseComparison):
            raise ValueError(f"{comparison_cls} must inherit from BaseComparison")
        self._registry[comparison_cls.name] = comparison_cls

    def get(self, name: str) -> Type[BaseComparison]:
        if name not in self._registry:
            print(f"[ComparisonRegistry] Method '{name}' not found. Available methods: {list(self._registry.keys())}")
            return None
        return self._registry[name]

    def all(self) -> Dict[str, Type[BaseComparison]]:
        return self._registry.copy()

    def all_comparisons(self) -> List[str]:
        return list(self._registry.keys())

    def initialize(self):
        """Initialize the registry and auto-load all comparison methods."""
        if not self._initialized:
            self._initialized = True
            # Auto-load all comparison methods from the comparison folder
            comparison_folder = os.path.dirname(__file__)
            for filename in os.listdir(comparison_folder):
                if filename.endswith(".py") and filename not in ("__init__.py", "base_comparison.py", "comparison_registry.py"):
                    module_name = filename[:-3]
                    try:
                        # Import the module to trigger the @register_comparison decorator
                        importlib.import_module(f".{module_name}", package="comparison")
                        print(f"[ComparisonRegistry] Successfully loaded {module_name}")
                    except Exception as e:
                        print(f"[ComparisonRegistry] Error importing {module_name}: {e}")
            
            print(f"[ComparisonRegistry] Initialized with {len(self._registry)} comparison methods")

# Global singleton instance
ComparisonRegistry = _ComparisonRegistry()

def register_comparison(cls: Type[BaseComparison]):
    """Decorator to register a comparison class."""
    ComparisonRegistry.register(cls)
    return cls

def load_all_comparisons(folder: str):
    """
    Auto-import all .py comparison files from the given folder to populate the registry.
    """
    folder_path = os.path.abspath(folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename not in ("__init__.py", "base_comparison.py", "comparison_registry.py"):
            module_name = filename[:-3]
            import_path = f"{folder.replace(os.sep, '.')}.{module_name}"
            try:
                importlib.import_module(import_path)
            except Exception as e:
                print(f"[load_all_comparisons] Error importing {import_path}: {e}") 