import os
import importlib
import inspect
from typing import Type, Dict, List, Tuple, Optional
from mixer.base_mixer import BaseMixer  # Adjust if your folder structure differs

class _MixerRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseMixer]] = {}

    def register(self, mixer_cls: Type[BaseMixer]):
        if not issubclass(mixer_cls, BaseMixer):
            raise ValueError(f"{mixer_cls} must inherit from BaseMixer")
        self._registry[mixer_cls.name] = mixer_cls

    def get(self, name: str) -> Type[BaseMixer]:
        return self._registry[name]

    def all(self) -> Dict[str, Type[BaseMixer]]:
        return self._registry.copy()

    def all_mixers(self) -> List[str]:
        return list(self._registry.keys())
    
    def get_all_templates(self) -> List[Tuple[str, str]]:
        """Get all templates from the template registry."""
        try:
            from mixer.template_registry import TemplateRegistry
            return TemplateRegistry.get_templates_for_ui()
        except ImportError:
            print("[MixerRegistry] Warning: Template registry not available")
            return []
    
    def get_templates_by_category(self, category: str) -> List[Tuple[str, str]]:
        """Get templates for a specific category."""
        try:
            from mixer.template_registry import TemplateRegistry
            return TemplateRegistry.get_templates_for_ui(category)
        except ImportError:
            print("[MixerRegistry] Warning: Template registry not available")
            return []
    
    def get_all_categories(self) -> List[str]:
        """Get all available template categories."""
        try:
            from mixer.template_registry import TemplateRegistry
            return TemplateRegistry.get_all_categories()
        except ImportError:
            print("[MixerRegistry] Warning: Template registry not available")
            return []

# Global instance
MixerRegistry = _MixerRegistry()

def register_mixer(cls: Type[BaseMixer]):
    MixerRegistry.register(cls)
    return cls

def load_all_mixers(folder: str):
    """
    Auto-import all .py mixer files from the given folder to populate the registry.
    """
    print(f"[load_all_mixers] Loading mixers from folder: {folder}")
    folder_path = os.path.abspath(folder)
    print(f"[load_all_mixers] Absolute path: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"[load_all_mixers] Error: Folder does not exist: {folder_path}")
        return
    
    files = os.listdir(folder_path)
    print(f"[load_all_mixers] Found files: {files}")
    
    for filename in files:
        if filename.endswith(".py") and filename not in ("__init__.py", "base_mixer.py", "mixer_registry.py"):
            module_name = filename[:-3]
            import_path = f"{folder.replace(os.sep, '.')}.{module_name}"
            print(f"[load_all_mixers] Attempting to import: {import_path}")
            try:
                module = importlib.import_module(import_path)
                print(f"[load_all_mixers] Successfully imported: {import_path}")
                
                # Check what classes were loaded
                for name in dir(module):
                    obj = getattr(module, name)
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseMixer) and 
                        obj != BaseMixer):
                        print(f"[load_all_mixers] Found mixer class: {name}")
                        
            except Exception as e:
                print(f"[load_all_mixers] Error importing {import_path}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"[load_all_mixers] Registry now contains: {MixerRegistry.all_mixers()}")
