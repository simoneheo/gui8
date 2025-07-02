import importlib
import pkgutil
import inspect
from typing import Type, Dict

_STEP_REGISTRY: Dict[str, Type] = {}
_STEP_CATEGORY_MAP: Dict[str, str] = {}  # e.g., {"Resample": "Resampling"}


def register_step(display_name: str, cls: Type):
    _STEP_REGISTRY[display_name] = cls
    _STEP_CATEGORY_MAP[display_name] = getattr(cls, "category", "Uncategorized")


def get_step(name: str) -> Type:
    return _STEP_REGISTRY[name]

def all_steps() -> list[str]:
    return list(_STEP_REGISTRY.keys())

def load_all_steps(package_name: str = __name__):
    """Auto-import all modules in this package and register step classes."""
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{package_name}.{module_name}")

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, "get_info") and hasattr(obj, "apply"):
                display_name = getattr(obj, "name", obj.__name__)
                register_step(display_name, obj)

def get_category(name: str) -> str:
    return _STEP_CATEGORY_MAP.get(name, "Uncategorized")

def all_categories() -> list[str]:
    return sorted(set(_STEP_CATEGORY_MAP.values()))

def steps_in_category(category: str) -> list[str]:
    return [name for name, cat in _STEP_CATEGORY_MAP.items() if cat == category]
