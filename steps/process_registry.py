# steps/process_registry.py

import os
import importlib
import inspect
from typing import Type, Dict, List
from steps.base_step import BaseStep

class _ProcessRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseStep]] = {}

    def register(self, step_cls: Type[BaseStep]):
        """Registers a step class by its name."""
        if not issubclass(step_cls, BaseStep):
            raise ValueError(f"{step_cls} must inherit from BaseStep")
        self._registry[step_cls.name] = step_cls  # Use .name instead of class name

    def get(self, name: str) -> Type[BaseStep]:
        return self._registry[name]

    def has_script_method(self, name: str) -> bool:
        """Check if a step has a script method available for editing"""
        if name not in self._registry:
            return False
        step_cls = self._registry[name]
        return hasattr(step_cls, 'script') and callable(getattr(step_cls, 'script', None))

    def all(self) -> Dict[str, Type[BaseStep]]:
        return self._registry.copy()

    def all_steps(self) -> List[str]:
        return list(self._registry.keys())

# Global singleton instance
ProcessRegistry = _ProcessRegistry()

def register_step(cls: Type[BaseStep]):
    ProcessRegistry.register(cls)
    return cls  # enables decorator syntax

def load_all_steps(folder: str):
    """
    Auto-import all .py step files from the given folder to populate the registry.
    """
    folder_path = os.path.abspath(folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename not in ("__init__.py", "base_step.py", "process_registry.py"):
            module_name = filename[:-3]
            import_path = f"{folder.replace(os.sep, '.')}.{module_name}"
            try:
                importlib.import_module(import_path)
            except Exception as e:
                print(f"[load_all_steps] Error importing {import_path}: {e}")
