from PySide6.QtCore import QObject, Signal, QCoreApplication
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QColor, QFont
from comparison_wizard_window import ComparisonWizardWindow
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import warnings
import pandas as pd
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
from data_aligner import DataAligner, AlignmentResult
from pair_manager import PairManager
# Import Pair classes with error handling
try:
    print("[ComparisonWizardManager] Attempting to import Pair classes...")
    from pair import Pair, AlignmentConfig, AlignmentMethod
    print(f"[ComparisonWizardManager] Pair classes imported successfully")
    print(f"[ComparisonWizardManager] Pair class type: {type(Pair)}")
    print(f"[ComparisonWizardManager] AlignmentConfig class type: {type(AlignmentConfig)}")
    print(f"[ComparisonWizardManager] AlignmentMethod class type: {type(AlignmentMethod)}")
except Exception as e:
    print(f"[ComparisonWizardManager] Error importing Pair classes: {e}")
    import traceback
    traceback.print_exc()
    # Create fallback classes
    class Pair:
        pass
    class AlignmentConfig:
        pass
    class AlignmentMethod:
        pass

# Import comparison methods from the new comparison folder
try:
    from comparison.comparison_registry import ComparisonRegistry
    from comparison import load_all_comparisons
    from comparison.base_comparison import BaseComparison
    COMPARISON_AVAILABLE = True
    print("[ComparisonWizardManager] Comparison registry imported successfully")
except ImportError as e:
    print(f"[ComparisonWizardManager] Comparison registry not available: {e}")
    COMPARISON_AVAILABLE = False
    
    # Fallback registry for when comparison module is not available
    class ComparisonRegistry:
        @staticmethod
        def get_all_methods():
            return []
        
        @staticmethod
        def get_all_categories():
            return []
        
        @staticmethod
        def get_methods_by_category(category):
            return []
        
        @staticmethod
        def get_method_info(method_name):
            return None
        
        @staticmethod
        def create_method(method_name, **kwargs):
            return None
    
    def load_all_comparisons(directory=None):
        pass
    
    class BaseComparison:
        pass

# Import PairAnalyzer
try:
    from pair_analyzer import PairAnalyzer
    PAIR_ANALYZER_AVAILABLE = True
    print("[ComparisonWizardManager] PairAnalyzer imported successfully")
except ImportError as e:
    print(f"[ComparisonWizardManager] PairAnalyzer not available: {e}")
    PAIR_ANALYZER_AVAILABLE = False



class ComparisonPairCache:
    """
    Manages caching of comparison pair computation results with LRU eviction
    """
    def __init__(self, max_pairs: int = 50, max_memory_mb: int = 100):
        self.max_pairs = max_pairs
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()  # LRU cache
        self.matplotlib_artists = {}  # pair_id -> list of matplotlib artists
        
    def get_pair_result(self, pair_id: str) -> Optional[PairResult]:
        """Get cached result for a pair"""
        if pair_id in self.cache:
            result = self.cache[pair_id]
            result.touch()
            # Move to end (most recently used)
            self.cache.move_to_end(pair_id)
            return result
        return None
        
    def store_pair_result(self, pair_result: PairResult):
        """Store computation result for a pair"""
        pair_id = pair_result.pair_id
        
        # Remove existing entry if present
        if pair_id in self.cache:
            del self.cache[pair_id]
            
        # Check memory and pair limits
        self._enforce_limits()
        
        # Store new result
        self.cache[pair_id] = pair_result
        
        print(f"[ComparisonCache] Stored result for pair '{pair_id}' "
              f"(memory: {pair_result.memory_estimate//1024}KB, "
              f"compute time: {pair_result.computation_time:.3f}s)")
        
    def invalidate_pair(self, pair_id: str):
        """Invalidate cached result for a specific pair"""
        if pair_id in self.cache:
            del self.cache[pair_id]
            print(f"[ComparisonCache] Invalidated cache for pair '{pair_id}'")
            
        # Also remove matplotlib artists
        if pair_id in self.matplotlib_artists:
            self._remove_matplotlib_artists(pair_id)
            
    def invalidate_all(self):
        """Invalidate all cached results (method parameter change)"""
        print(f"[ComparisonCache] Invalidating all cached results ({len(self.cache)} pairs)")
        self.cache.clear()
        
        # Remove all matplotlib artists
        for pair_id in list(self.matplotlib_artists.keys()):
            self._remove_matplotlib_artists(pair_id)
            
    def invalidate_by_method_hash(self, old_hash: str):
        """Invalidate results with specific method hash"""
        to_remove = []
        for pair_id, result in self.cache.items():
            if result.method_hash == old_hash:
                to_remove.append(pair_id)
                
        for pair_id in to_remove:
            self.invalidate_pair(pair_id)
            
        print(f"[ComparisonCache] Invalidated {len(to_remove)} pairs with old method hash")
        
    def store_matplotlib_artists(self, pair_id: str, artists: List[Any]):
        """Store matplotlib artists for a pair"""
        self.matplotlib_artists[pair_id] = artists
        
    def get_matplotlib_artists(self, pair_id: str) -> List[Any]:
        """Get matplotlib artists for a pair"""
        return self.matplotlib_artists.get(pair_id, [])
        
    def set_pair_visibility(self, pair_id: str, visible: bool):
        """Show/hide matplotlib artists for a pair"""
        artists = self.matplotlib_artists.get(pair_id, [])
        for artist in artists:
            try:
                artist.set_visible(visible)
            except:
                pass  # Artist might have been removed
                
    def _remove_matplotlib_artists(self, pair_id: str):
        """Remove matplotlib artists for a pair"""
        artists = self.matplotlib_artists.get(pair_id, [])
        for artist in artists:
            try:
                artist.remove()
            except:
                pass  # Artist might have been removed already
                
        if pair_id in self.matplotlib_artists:
            del self.matplotlib_artists[pair_id]
            
    def _enforce_limits(self):
        """Enforce memory and pair count limits"""
        # Enforce pair count limit
        while len(self.cache) >= self.max_pairs:
            oldest_pair = next(iter(self.cache))
            print(f"[ComparisonCache] Evicting oldest pair: {oldest_pair}")
            self.invalidate_pair(oldest_pair)
            
        # Enforce memory limit
        total_memory = sum(result.memory_estimate for result in self.cache.values())
        max_memory_bytes = self.max_memory_mb * 1024 * 1024
        
        while total_memory > max_memory_bytes and self.cache:
            # Remove largest memory consumers first
            largest_pair = max(self.cache.keys(), 
                             key=lambda k: self.cache[k].memory_estimate)
            print(f"[ComparisonCache] Evicting largest pair: {largest_pair} "
                  f"({self.cache[largest_pair].memory_estimate//1024}KB)")
            self.invalidate_pair(largest_pair)
            total_memory = sum(result.memory_estimate for result in self.cache.values())
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_memory = sum(result.memory_estimate for result in self.cache.values())
        return {
            'pairs_cached': len(self.cache),
            'max_pairs': self.max_pairs,
            'memory_used_mb': total_memory / (1024 * 1024),
            'max_memory_mb': self.max_memory_mb,
            'memory_utilization': (total_memory / (1024 * 1024)) / self.max_memory_mb * 100,
            'matplotlib_artists': len(self.matplotlib_artists)
        }


def create_method_hash(method_name: str, method_params: Dict[str, Any]) -> str:
    """Create a hash for method + parameters to detect changes"""
    data = f"{method_name}|{sorted(method_params.items())}"
    return hashlib.md5(data.encode()).hexdigest()


class ComparisonWizardManager(QObject):
    """
    Manager for the comparison wizard that handles:
    - Data alignment between channels
    - Statistical calculations using comparison methods
    - Plot generation
    - State management and progress tracking
    """
    
    comparison_complete = Signal(dict)
    state_changed = Signal(str)  # Emit state changes
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store managers with validation
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Initialize state tracking
        self._stats = {
            'total_comparisons': 0,
            'successful_alignments': 0,
            'failed_alignments': 0,
            'last_comparison': None,
            'session_start': time.time()
        }
        
        # Legacy data structures removed - now using PairManager and PairAnalyzer
        # Initialize legacy pair_aligned_data for backward compatibility
        self.pair_aligned_data = {}
        
        # Initialize new caching system
        self.pair_cache = ComparisonPairCache(max_pairs=50, max_memory_mb=100)
        self.current_method_hash = ""  # Track current method parameters
        
        # Initialize data aligner and pair manager
        self.data_aligner = DataAligner()
        self.pair_manager = PairManager()
        
        # Initialize PairAnalyzer for computation
        self.pair_analyzer = None
        if PAIR_ANALYZER_AVAILABLE:
            self.pair_analyzer = PairAnalyzer()
            print("[ComparisonWizardManager] PairAnalyzer initialized successfully")
        
        # Initialize comparison methods
        self._initialize_comparison_methods()
        
        # Validate initialization
        if not self._validate_managers():
            raise ValueError("Required managers not available for ComparisonWizardManager")
        
        # Create window after validation
        self.window = ComparisonWizardWindow(
            file_manager=self.file_manager,
            channel_manager=self.channel_manager,
            signal_bus=self.signal_bus,
            parent=self.parent_window
        )
        
        # Set bidirectional reference
        self.window.comparison_manager = self
        
        # Connect signals first
        self._connect_signals()
        
        # Refresh window with comparison registry data after everything is set up
        try:
            self.window._refresh_comparison_data()
        except Exception as e:
            print(f"[ComparisonWizardManager] Warning: Could not refresh comparison data: {e}")
            # This is not critical, the window will use static controls
        
        # Log initialization
        self._log_state_change("Manager initialized successfully")
        
    def _get_script_configuration(self) -> Dict[str, Any]:
        """Extract script configuration from the comparison wizard window"""
        if not hasattr(self.window, 'script_tabs'):
            return {}
        
        script_config = {}
        
        # Get custom plot script
        if hasattr(self.window.script_tabs, 'plot_script_tab'):
            plot_script = self.window.script_tabs.plot_script_tab.get_script_content()
            if plot_script.strip():
                script_config['custom_plot_script'] = plot_script
                
        # Get custom statistics script
        if hasattr(self.window.script_tabs, 'stats_script_tab'):
            stats_script = self.window.script_tabs.stats_script_tab.get_script_content()
            if stats_script.strip():
                script_config['custom_stats_script'] = stats_script
                
        return script_config
        
    def _coordinate_cache_invalidation(self, pairs_to_invalidate: Optional[List[str]] = None):
        """Coordinate cache invalidation between PairAnalyzer and ComparisonPairCache"""
        if pairs_to_invalidate is None:
            # Invalidate all caches
            self.pair_cache.invalidate_all()
            if self.pair_analyzer:
                self.pair_analyzer.invalidate_cache(method_changed=True)
            print("[ComparisonWizardManager] Invalidated all caches")
        else:
            # Invalidate specific pairs
            for pair_name in pairs_to_invalidate:
                self.pair_cache.invalidate_pair(pair_name)
            # PairAnalyzer doesn't support pair-specific invalidation, so invalidate all
            if self.pair_analyzer:
                self.pair_analyzer.invalidate_cache(method_changed=False)
            print(f"[ComparisonWizardManager] Invalidated caches for pairs: {pairs_to_invalidate}")
            
    def _process_analysis_results(self, analysis_results: Dict[str, Any], pair_name: str) -> Dict[str, Any]:
        """Process analysis results from PairAnalyzer and prepare for existing plotting system"""
        if not analysis_results:
            return {}
            
        # Extract components from analysis results
        statistics = analysis_results.get('statistics', {})
        plot_data = analysis_results.get('plot_data', {})
        overlays = analysis_results.get('overlays', [])
        
        # Prepare plot data for existing plotting system
        processed_results = {
            'statistics': statistics,
            'plot_data': plot_data,
            'overlays': overlays,
            'pair_name': pair_name
        }
        
        # Add metadata
        if 'metadata' in analysis_results:
            processed_results['metadata'] = analysis_results['metadata']
            
        return processed_results
    
    def _initialize_comparison_methods(self):
        """Initialize comparison methods from the comparison folder"""
        try:
            if COMPARISON_AVAILABLE:
                # Set the comparison registry reference
                self.comparison_registry = ComparisonRegistry
                
                # Load all comparison methods from the comparison folder
                load_all_comparisons()
                
                # Log loaded methods
                methods = self.comparison_registry.all_comparisons()
                print(f"[ComparisonWizardManager] Using {len(methods)} comparison methods")
                print(f"  Methods: {', '.join(methods)}")
                
                self._log_state_change("Comparison methods loaded successfully")
            else:
                self._log_state_change("Comparison methods not available - using basic calculations only")
        except Exception as e:
            print(f"[ComparisonWizardManager] Warning: Could not load comparison methods: {e}")
            import traceback
            traceback.print_exc()
            self._log_state_change("Failed to load comparison methods - using fallback")
    

    
    def get_available_comparison_methods(self):
        """Get list of available comparison methods"""
        if COMPARISON_AVAILABLE:
            try:
                # Get registry names and convert to display names using the comparison classes
                registry_names = self.comparison_registry.all_comparisons()
                display_names = []
                for name in registry_names:
                    try:
                        comparison_cls = self.comparison_registry.get(name)
                        if comparison_cls:
                            # Generate clean display name
                            display_name = self._generate_clean_display_name(comparison_cls, name)
                            display_names.append(display_name)
                        else:
                            display_names.append(name.replace('_', ' ').title() + ' Analysis')
                    except Exception as e:
                        print(f"[ComparisonWizardManager] Error getting display name for {name}: {e}")
                        display_names.append(name.replace('_', ' ').title() + ' Analysis')
                return display_names
            except Exception as e:
                print(f"[ComparisonWizardManager] Error getting comparison methods: {e}")
        return ["Correlation Analysis", "Bland-Altman Analysis", "Residual Analysis"]
    
    def get_comparison_categories(self):
        """Get list of comparison method categories"""
        if COMPARISON_AVAILABLE:
            try:
                # Get categories from all registered comparison methods
                categories = set()
                for method_name in self.comparison_registry.all_comparisons():
                    try:
                        comparison_cls = self.comparison_registry.get(method_name)
                        if comparison_cls:
                            categories.add(comparison_cls.category)
                    except:
                        pass
                return list(categories)
            except:
                pass
        return ["Statistical", "Agreement", "Error Analysis"]
    
    def get_methods_by_category(self, category):
        """Get comparison methods in a specific category"""
        if COMPARISON_AVAILABLE:
            try:
                # Get methods by category from all registered comparison methods
                display_names = []
                for method_name in self.comparison_registry.all_comparisons():
                    try:
                        comparison_cls = self.comparison_registry.get(method_name)
                        if comparison_cls and comparison_cls.category == category:
                            # Generate clean display name
                            display_name = self._generate_clean_display_name(comparison_cls, method_name)
                            display_names.append(display_name)
                    except Exception as e:
                        print(f"[ComparisonWizardManager] Error getting method for category {category}: {e}")
                        pass
                return display_names
            except Exception as e:
                print(f"[ComparisonWizardManager] Error getting methods by category: {e}")
                pass
        # Fallback
        if category == "Statistical":
            return ["Correlation Analysis"]
        elif category == "Agreement":
            return ["Bland-Altman Analysis"]
        elif category == "Error Analysis":
            return ["Residual Analysis"]
        return []
    
    def get_method_info(self, method_name_or_display):
        """Get detailed information about a comparison method"""
        if COMPARISON_AVAILABLE:
            try:
                # Convert display name to registry name if needed
                registry_name = self.get_registry_name_from_display(method_name_or_display)
                
                # Try to get method information from the comparison registry
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls:
                    return comparison_cls().get_info()
                else:
                    print(f"[ComparisonWizardManager] Method {registry_name} not found in registry")
            except Exception as e:
                print(f"[ComparisonWizardManager] Error getting method info for {method_name_or_display}: {e}")
        
        # Fallback
        return {
            'name': method_name_or_display,
            'description': f'Description for {method_name_or_display}',
            'parameters': {},
            'category': 'Statistical'
        }
    
    def get_registry_name_from_display(self, display_name):
        """Convert display name to registry name"""
        if COMPARISON_AVAILABLE:
            try:
                # Check each registered method to find matching display name
                for registry_name in self.comparison_registry.all_comparisons():
                    comparison_cls = self.comparison_registry.get(registry_name)
                    if comparison_cls:
                        # Generate clean display name and check if it matches
                        method_display_name = self._generate_clean_display_name(comparison_cls, registry_name)
                        
                        if method_display_name == display_name:
                            return registry_name
                            
                # Fallback: convert display name to registry name
                return display_name.lower().replace(' analysis', '').replace(' ', '_')
            except Exception as e:
                print(f"[ComparisonWizardManager] Error converting display name {display_name}: {e}")
                return display_name.lower().replace(' analysis', '').replace(' ', '_')
        
        # Fallback conversion
        return display_name.lower().replace(' analysis', '').replace(' ', '_')
        
    def _generate_clean_display_name(self, comparison_cls, registry_name):
        """Generate a clean, user-friendly display name from comparison class"""
        if hasattr(comparison_cls, 'description') and comparison_cls.description:
            description = comparison_cls.description
            
            # If description contains ' - ', take the first part
            if ' - ' in description:
                return description.split(' - ')[0]
            
            # If description contains 'analysis' or 'comparison', extract the key part
            desc_lower = description.lower()
            
            # Common patterns to extract clean names
            if desc_lower.startswith('histogram analysis'):
                return 'Error Distribution Histogram'
            elif desc_lower.startswith('time series analysis'):
                return 'Relative Error Time Series'
            elif desc_lower.startswith('cross-correlation analysis'):
                return 'Time Lag Cross-Correlation'
            elif 'correlation' in desc_lower and 'coefficients' in desc_lower:
                return 'Correlation Analysis'
            elif 'bland-altman' in desc_lower or 'bland altman' in desc_lower:
                return 'Bland-Altman Analysis'
            elif 'residual' in desc_lower and 'analysis' in desc_lower:
                return 'Residual Analysis'
            else:
                # Try to extract first few meaningful words
                words = description.split()
                if len(words) >= 2:
                    # Take first 2-3 significant words and add 'Analysis'
                    significant_words = [w for w in words[:3] if len(w) > 2 and w.lower() not in ['of', 'the', 'and', 'with', 'for']]
                    if significant_words:
                        return ' '.join(significant_words[:2]).title() + ' Analysis'
        
        # Fallback: convert registry name to title case
        return registry_name.replace('_', ' ').title() + ' Analysis'
        
    def get_display_name_from_registry(self, registry_name):
        """Convert registry name to display name"""
        if COMPARISON_AVAILABLE:
            try:
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls:
                    return self._generate_clean_display_name(comparison_cls, registry_name)
            except Exception as e:
                print(f"[ComparisonWizardManager] Error converting registry name {registry_name}: {e}")
        
        # Fallback conversion
        return registry_name.replace('_', ' ').title() + ' Analysis'
    
    
    # REMOVED: _fallback_comparison - Not used in current implementation

    def _validate_managers(self) -> bool:
        """Validate that required managers are available and functional"""
        if not self.file_manager:
            print("[ComparisonWizardManager] ERROR: File manager not provided")
            return False
            
        if not self.channel_manager:
            print("[ComparisonWizardManager] ERROR: Channel manager not provided")
            return False
            
        # Validate manager functionality
        try:
            self.file_manager.get_file_count()
            self.channel_manager.get_channel_count()
            return True
        except Exception as e:
            print(f"[ComparisonWizardManager] ERROR: Manager validation failed: {e}")
            return False
            
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[ComparisonWizardManager {timestamp}] {message}")
        self.state_changed.emit(message)
        
    def _connect_signals(self):
        """Connect window signals to manager methods"""
        self.window.pair_added.connect(self._on_pair_added)
        self.window.pair_deleted.connect(self._on_pair_deleted)
        self.window.plot_generated.connect(self._on_plot_generated)
        
    def show(self):
        """Show the comparison wizard window"""
        self.window.show()
        
    def close(self):
        """Close the comparison wizard"""
        if self.window:
            self.window.close()
            
    def _on_pair_added(self, pair_config):
        """Handle when a new pair is added - now with Pair object creation and PairAnalyzer integration"""
        print(f"[ComparisonWizard] Pair added: {pair_config['name']}")
        print(f"[ComparisonWizard] DEBUG: pair_config keys: {list(pair_config.keys())}")
        
        # Get the channels for this pair - handle both old and new key formats
        if 'ref_file' in pair_config and 'ref_channel' in pair_config:
            # Old format
            ref_channel = self._get_channel(pair_config['ref_file'], pair_config['ref_channel'])
            test_channel = self._get_channel(pair_config['test_file'], pair_config['test_channel'])
        elif 'ref_file_id' in pair_config and 'ref_channel_name' in pair_config:
            # New format - need to resolve file names and channel names
            ref_file_id = pair_config['ref_file_id']
            test_file_id = pair_config['test_file_id']
            ref_channel_name = pair_config['ref_channel_name']
            test_channel_name = pair_config['test_channel_name']
            
            # Get file names from file manager
            ref_file = self.file_manager.get_file(ref_file_id) if self.file_manager else None
            test_file = self.file_manager.get_file(test_file_id) if self.file_manager else None
            
            ref_filename = ref_file.filename if ref_file else ref_file_id
            test_filename = test_file.filename if test_file else test_file_id
            
            ref_channel = self._get_channel(ref_filename, ref_channel_name)
            test_channel = self._get_channel(test_filename, test_channel_name)
        else:
            print(f"[ComparisonWizard] ERROR: Unknown pair config format. Keys: {list(pair_config.keys())}")
            ref_channel = None
            test_channel = None
        
        print(f"[ComparisonWizard] DEBUG: ref_channel found: {ref_channel is not None}")
        print(f"[ComparisonWizard] DEBUG: test_channel found: {test_channel is not None}")
        
        if ref_channel and test_channel:
            pair_name = pair_config['name']
            
            try:
                # STEP 1: Create Pair object and add to PairManager
                print(f"[ComparisonWizard] Creating Pair object for '{pair_name}'")
                print(f"[ComparisonWizard] DEBUG: About to check Pair class availability")
                print(f"[ComparisonWizard] DEBUG: Pair class available: {Pair is not None}")
                print(f"[ComparisonWizard] DEBUG: Pair class type: {type(Pair)}")
                print(f"[ComparisonWizard] DEBUG: Pair class module: {Pair.__module__ if hasattr(Pair, '__module__') else 'No module'}")
                print(f"[ComparisonWizard] DEBUG: Pair class name: {Pair.__name__ if hasattr(Pair, '__name__') else 'No name'}")
                print(f"[ComparisonWizard] DEBUG: Pair.from_config method available: {hasattr(Pair, 'from_config')}")
                
                # Prepare pair configuration with channel IDs - handle both old and new formats
                # Don't copy pair_id to ensure a new unique ID is generated
                pair_config_with_ids = {
                    key: value for key, value in pair_config.items() if key != 'pair_id'
                }
                pair_config_with_ids.update({
                    'ref_channel_id': ref_channel.channel_id,
                    'test_channel_id': test_channel.channel_id,
                    'ref_file_id': ref_channel.file_id if hasattr(ref_channel, 'file_id') else (pair_config.get('ref_file_id') or pair_config.get('ref_file', '')),
                    'test_file_id': test_channel.file_id if hasattr(test_channel, 'file_id') else (pair_config.get('test_file_id') or pair_config.get('test_file', '')),
                    'ref_channel_name': pair_config.get('ref_channel_name') or pair_config.get('ref_channel', ''),
                    'test_channel_name': pair_config.get('test_channel_name') or pair_config.get('test_channel', '')
                })
                
                print(f"[ComparisonWizard] DEBUG: pair_config_with_ids keys: {list(pair_config_with_ids.keys())}")
                print(f"[ComparisonWizard] DEBUG: alignment_config type: {type(pair_config_with_ids.get('alignment_config'))}")
                print(f"[ComparisonWizard] DEBUG: alignment_config content: {pair_config_with_ids.get('alignment_config')}")
                
                # Convert alignment_config dict to AlignmentConfig object if needed
                if 'alignment_config' in pair_config_with_ids and isinstance(pair_config_with_ids['alignment_config'], dict):
                    print(f"[ComparisonWizard] DEBUG: Converting alignment_config dict to object")
                    align_dict = pair_config_with_ids['alignment_config']
                    
                    # Convert method string to enum
                    method_str = align_dict.get('alignment_method', 'index')
                    print(f"[ComparisonWizard] DEBUG: method_str: {method_str}")
                    try:
                        method = AlignmentMethod(method_str)
                        print(f"[ComparisonWizard] DEBUG: AlignmentMethod created: {method}")
                    except ValueError as e:
                        print(f"[ComparisonWizard] DEBUG: AlignmentMethod ValueError: {e}")
                        method = AlignmentMethod.INDEX
                    
                    # Create AlignmentConfig object
                    try:
                        alignment_config = AlignmentConfig(
                            method=method,
                            mode=align_dict.get('mode', 'truncate'),
                            start_index=align_dict.get('start_index'),
                            end_index=align_dict.get('end_index'),
                            start_time=align_dict.get('start_time'),
                            end_time=align_dict.get('end_time'),
                            offset=align_dict.get('offset', 0.0),
                            interpolation=align_dict.get('interpolation'),
                            round_to=align_dict.get('round_to')
                        )
                        print(f"[ComparisonWizard] DEBUG: AlignmentConfig created successfully")
                        pair_config_with_ids['alignment_config'] = alignment_config
                    except Exception as e:
                        print(f"[ComparisonWizard] DEBUG: AlignmentConfig creation failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Create Pair object
                print(f"[ComparisonWizard] DEBUG: About to call Pair.from_config()")
                try:
                    pair = Pair.from_config(pair_config_with_ids)
                    print(f"[ComparisonWizard] DEBUG: Pair object created successfully: {pair}")
                except Exception as e:
                    print(f"[ComparisonWizard] DEBUG: Pair.from_config() failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Add to PairManager (this handles duplicate checking and validation)
                if self.pair_manager.add_pair(pair):
                    # Pair added successfully, continue with analysis
                    print(f"[ComparisonWizard] Successfully added pair '{pair_name}' to PairManager")
                else:
                    # PairManager couldn't add the pair - could be duplicate or validation error
                    # PairManager already handles duplicate detection and user interaction
                    print(f"[ComparisonWizard] PairManager could not add pair '{pair_name}' - likely duplicate or validation error")
                    return
                
                # STEP 2: Get current method and parameters
                current_method = self.window.method_list.currentItem().text() if self.window.method_list.currentItem() else "Correlation Analysis"
                method_params = self.window._get_method_parameters_from_controls() if hasattr(self.window, '_get_method_parameters_from_controls') else {}
                
                # Create method hash for cache invalidation
                method_hash = create_method_hash(current_method, method_params)
                
                # Check if we need to invalidate cache due to method change
                if self.current_method_hash != method_hash:
                    print(f"[ComparisonWizard] Method parameters changed, invalidating cache")
                    self._coordinate_cache_invalidation()
                    self.current_method_hash = method_hash
                
                # STEP 3: Check cache first
                cached_result = self.pair_cache.get_pair_result(pair_name)
                # STEP 4: Use PairAnalyzer for computation
                if self.pair_analyzer:
                    try:
                        # Prepare comparison configuration
                        comparison_config = {
                            'method': current_method,
                            'method_params': method_params,
                            'pair_data': pair_config
                        }
                        
                        # Get script configuration
                        script_config = self._get_script_configuration()
                        
                        # Run analysis through PairAnalyzer
                        analysis_session = self.pair_analyzer.analyze(comparison_config, self.pair_manager, script_config)
                        
                        # Process results for existing plotting system
                        if analysis_session.analysis_results:
                            processed_results = self._process_analysis_results(analysis_session.analysis_results, pair_name)
                            
                            # Extract aligned data and statistics
                            aligned_data = processed_results.get('statistics', {}).get('aligned_data', {})
                            stats = processed_results.get('statistics', {})
                            
                            # If PairAnalyzer provided good results, use them
                            if aligned_data and stats:
                                print(f"[ComparisonWizard] Using PairAnalyzer results for pair '{pair_name}'")
                                computation_time = 0.01  # PairAnalyzer handles timing internally
                            else:
                                raise ValueError("PairAnalyzer did not provide sufficient results")
                        else:
                            raise ValueError("PairAnalyzer returned empty results")
                            
                    except Exception as e:
                        print(f"[ComparisonWizard] PairAnalyzer failed for pair '{pair_name}': {e}")
                        # Show error to user
                        QMessageBox.critical(
                            self.window,
                            "Analysis Error",
                            f"Failed to analyze pair '{pair_name}': {str(e)}"
                        )
                        return
                else:
                    # No PairAnalyzer available
                    QMessageBox.warning(
                        self.window,
                        "PairAnalyzer Not Available",
                        "PairAnalyzer is not available. Please ensure it is properly installed."
                    )
                    return
                
                # STEP 5: Update table and plot
                self._update_table_with_pair_results(pair_name, stats)
                self._plot_pair_with_existing_system(pair_name, pair_config)
                
                # STEP 9: Log results
                self._log_individual_pair_stats(pair_name, stats)
                
                # STEP 10: Show alignment warnings if needed
                if self._has_alignment_warnings(aligned_data, stats):
                    self._show_alignment_summary(pair_name, aligned_data, stats)
                
                # Update overall statistics
                self._stats['total_comparisons'] += 1
                self._stats['successful_alignments'] += 1
                
                print(f"[ComparisonWizard] Pair '{pair_name}' added and plotted successfully")
                
            except Exception as e:
                print(f"[ComparisonWizard] Error processing pair '{pair_name}': {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Update failure statistics
                self._stats['failed_alignments'] += 1
                
                # Show error in console instead of blocking dialog for same-channel issues
                error_msg = str(e)
                if "Standard deviation calculation invalid" in error_msg or "zero variance" in error_msg.lower():
                    # Handle same-channel comparison gracefully
                    if hasattr(self, 'window') and hasattr(self.window, 'info_output'):
                        self.window.info_output.append(f"⚠️ Same-channel comparison detected for '{pair_name}' - results may not be meaningful")
                        self.window.info_output.append(f"💡 Consider comparing different channels for better analysis")
                    print(f"[ComparisonWizard] Same-channel comparison handled gracefully for '{pair_name}'")
                else:
                    # Show error message for other types of errors
                    QMessageBox.critical(
                        self.window,
                        "Pair Addition Error",
                        f"Failed to add pair '{pair_name}':\n{str(e)}"
                    )
        else:
            print(f"[ComparisonWizard] Could not find channels for pair '{pair_config['name']}'")
            
    # Legacy _compute_pair_fallback method removed - now using PairAnalyzer
            
    # Legacy _prepare_plot_data and _plot_pair_immediately methods removed - now using PairAnalyzer
            
    def _update_table_with_pair_results(self, pair_name: str, stats: Dict[str, Any]):
        """Update table with pair results from PairAnalyzer"""
        try:
            # Update the active pairs table to reflect new pair
            if hasattr(self.window, '_update_active_pairs_table'):
                self.window._update_active_pairs_table()
            
            # Force table refresh
            if hasattr(self.window, 'active_pair_table'):
                self.window.active_pair_table.repaint()
                QCoreApplication.processEvents()
                
            print(f"[ComparisonWizard] Updated table for pair '{pair_name}'")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error updating table: {str(e)}")
    
    def _plot_pair_with_existing_system(self, pair_name: str, pair_config: Dict[str, Any]):
        """Plot pair using existing plotting system"""
        try:
            # Get current plot configuration
            plot_config = self.window._get_plot_config() if hasattr(self.window, '_get_plot_config') else {}
            
            # Get all checked pairs for cumulative plotting
            checked_pairs = self.window.get_checked_pairs() if hasattr(self.window, 'get_checked_pairs') else []
            
            # Add current pair to checked pairs if not already there
            pair_in_checked = any(pair['name'] == pair_name for pair in checked_pairs)
            if not pair_in_checked:
                # Add pair with default visualization settings
                pair_config_extended = dict(pair_config)
                pair_config_extended.update({
                    'marker_type': '○ Circle',
                    'marker_color': '🔵 Blue',
                    'show': True
                })
                checked_pairs.append(pair_config_extended)
            
            # Generate cumulative plot using existing system
            print(f"[ComparisonWizard] Plotting pair '{pair_name}' with {len(checked_pairs)} total pairs")
            self._generate_scatter_plot(checked_pairs, plot_config)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error plotting pair: {str(e)}")
    
    def _log_individual_pair_stats(self, pair_name, stats):
        """Log individual pair statistics for debugging"""
        try:
            r_val = stats.get('r', np.nan)
            rms_val = stats.get('rms', np.nan)
            n_val = stats.get('n', 0)
            
            r_str = f"{r_val:.3f}" if not np.isnan(r_val) else "N/A"
            rms_str = f"{rms_val:.3f}" if not np.isnan(rms_val) else "N/A"
            
            print(f"[ComparisonWizard] Pair '{pair_name}': r={r_str}, RMS={rms_str}, N={n_val}")
            
            if 'error' in stats:
                print(f"  Error: {stats['error']}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error logging pair stats: {str(e)}")
    
    def _has_alignment_warnings(self, aligned_data, stats):
        """Check if alignment has warnings that should be shown to user"""
        try:
            # Check for data quality issues
            valid_ratio = aligned_data.get('valid_ratio', stats.get('valid_ratio', 1.0))
            if valid_ratio < 0.8:
                return True
                
            # Check for statistical issues
            if 'error' in stats:
                return True
                
            # Check for low correlation
            r_val = stats.get('r', np.nan)
            if not np.isnan(r_val) and abs(r_val) < 0.1:
                return True
                
            # Check for large RMS relative to data scale
            rms_val = stats.get('rms', np.nan)
            ref_std = stats.get('ref_std', np.nan)
            test_std = stats.get('test_std', np.nan)
            if not np.isnan(rms_val) and not np.isnan(ref_std) and not np.isnan(test_std):
                typical_scale = (ref_std + test_std) / 2
                if typical_scale > 0 and rms_val > 2 * typical_scale:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"[ComparisonWizard] Error checking alignment warnings: {str(e)}")
            return False
        
    def _on_pair_deleted(self):
        """Handle when a pair is deleted - now with PairManager integration and cache cleanup"""
        print("[ComparisonWizard] Pair deleted signal received")
        
        # Get currently selected pairs to identify what was deleted
        checked_pairs = self.window.get_checked_pairs() if hasattr(self.window, 'get_checked_pairs') else []
        current_pair_names = {pair['name'] for pair in checked_pairs}
        
        # Find pairs that were deleted (in cache but not in current pairs)
        cached_pair_names = set(self.pair_cache.cache.keys())
        deleted_pairs = cached_pair_names - current_pair_names
        
        # Also check legacy data structures
        legacy_pair_names = set(self.pair_aligned_data.keys())
        deleted_pairs.update(legacy_pair_names - current_pair_names)
        
        # Clean up cache and data for deleted pairs
        for pair_name in deleted_pairs:
            print(f"[ComparisonWizard] Cleaning up deleted pair: {pair_name}")
            
            # Remove from cache
            self.pair_cache.invalidate_pair(pair_name)
            
            # Remove from legacy data structures
            if pair_name in self.pair_aligned_data:
                del self.pair_aligned_data[pair_name]
            if pair_name in self.pair_statistics:
                del self.pair_statistics[pair_name]
            if pair_name in self._access_counts:
                del self._access_counts[pair_name]
            
            # Remove from PairManager if it exists there
            if self.pair_manager.has_pair(pair_name):
                self.pair_manager.remove_pair(pair_name)
                print(f"[ComparisonWizard] Removed pair '{pair_name}' from PairManager")
        
        # Regenerate plot with remaining pairs
        if checked_pairs:
            print(f"[ComparisonWizard] Regenerating plot with {len(checked_pairs)} remaining pairs")
            plot_config = self.window._get_plot_config() if hasattr(self.window, '_get_plot_config') else {}
            plot_type = self._determine_plot_type_from_pairs(checked_pairs)
            plot_config['plot_type'] = plot_type
            plot_config['checked_pairs'] = checked_pairs
            self._generate_scatter_plot(checked_pairs, plot_config)
        else:
            print("[ComparisonWizard] No pairs remaining, clearing plot")
            self._clear_all_plots()
        
        # Update cumulative display
        self._update_cumulative_display()
        
        print(f"[ComparisonWizard] Pair deletion complete. Cache contains {len(self.pair_cache.cache)} pairs, PairManager contains {self.pair_manager.get_pair_count()} pairs")
        
    def _clear_all_plots(self):
        """Clear all plots when no pairs remain"""
        if hasattr(self.window, 'canvas') and self.window.canvas:
            fig = self.window.canvas.figure
            self._clear_figure_completely(fig)
            self.window.canvas.draw()
            
        # Clear other plot tabs if they exist
        for attr_name in ['histogram_canvas', 'heatmap_canvas']:
            if hasattr(self.window, attr_name):
                canvas = getattr(self.window, attr_name)
                if canvas:
                    fig = canvas.figure
                    self._clear_figure_completely(fig)
                    canvas.draw()
                    
    def on_method_parameters_changed(self, method_name: str, method_params: Dict[str, Any]):
        """Handle when method or parameters change - invalidate cache if needed with PairAnalyzer integration"""
        new_method_hash = create_method_hash(method_name, method_params)
        
        if self.current_method_hash != new_method_hash:
            print(f"[ComparisonWizard] Method parameters changed, invalidating cache")
            print(f"[ComparisonWizard] Old hash: {self.current_method_hash}")
            print(f"[ComparisonWizard] New hash: {new_method_hash}")
            
            # Coordinate cache invalidation with PairAnalyzer
            self._coordinate_cache_invalidation()
            self.current_method_hash = new_method_hash
            
            # Trigger recomputation for all visible pairs
            self._recompute_all_visible_pairs()
            
    def _recompute_all_visible_pairs(self):
        """Recompute all visible pairs after parameter change"""
        checked_pairs = self.window.get_checked_pairs() if hasattr(self.window, 'get_checked_pairs') else []
        
        if not checked_pairs:
            return
            
        print(f"[ComparisonWizard] Recomputing {len(checked_pairs)} visible pairs")
        
        # Process each pair
        for pair_info in checked_pairs:
            pair_name = pair_info['name']
            
            # Get the channels for this pair
            ref_channel = self._get_channel(pair_info['ref_file'], pair_info['ref_channel'])
            test_channel = self._get_channel(pair_info['test_file'], pair_info['test_channel'])
            
            if ref_channel and test_channel:
                try:
                    # Recompute the pair using DataAligner
                    start_time = time.time()
                    
                    # Create AlignmentConfig from pair_info
                    alignment_config = pair_info.get('alignment_config')
                    if not alignment_config:
                        # Create a default alignment config if missing
                        from pair import AlignmentConfig, AlignmentMethod
                        alignment_config = AlignmentConfig(
                            method=AlignmentMethod.INDEX,
                            start_index=0,
                            end_index=500
                        )
                    
                    # Use DataAligner for alignment
                    alignment_result = self.data_aligner.align_channels(
                        ref_channel, test_channel, alignment_config
                    )
                    
                    if not alignment_result.success:
                        print(f"[ComparisonWizard] Alignment failed for pair '{pair_name}': {alignment_result.error_message}")
                        continue
                    
                    # Convert DataAligner result to legacy format for compatibility
                    aligned_data = {
                        'ref_data': alignment_result.ref_data,
                        'test_data': alignment_result.test_data,
                        'ref_label': ref_channel.legend_label or ref_channel.ylabel,
                        'test_label': test_channel.legend_label or test_channel.ylabel,
                        'alignment_method': alignment_config.method.value,
                        'n_points': len(alignment_result.ref_data)
                    }
                    
                    # Use simplified PairAnalyzer for statistics calculation
                    if self.pair_analyzer:
                        try:
                            # Get current comparison method
                            current_method = self.window.method_list.currentItem().text() if self.window.method_list.currentItem() else "Correlation Analysis"
                            method_params = self.window._get_method_parameters_from_controls() if hasattr(self.window, '_get_method_parameters_from_controls') else {}
                            
                            # Find the pair object in PairManager
                            pair_obj = None
                            if hasattr(self, 'pair_manager') and self.pair_manager:
                                for p in self.pair_manager.get_pairs_in_order():
                                    if p.name == pair_name:
                                        pair_obj = p
                                        break
                            
                            if pair_obj:
                                # Use simplified PairAnalyzer interface
                                stats = self.pair_analyzer.analyze_pair(pair_obj, current_method, method_params)
                            else:
                                print(f"[ComparisonWizard] Could not find pair object for '{pair_name}', using basic fallback")
                                stats = {'r': np.nan, 'rms': np.nan, 'n': len(alignment_result.ref_data), 'error': 'Pair not found'}
                        except Exception as e:
                            print(f"[ComparisonWizard] PairAnalyzer error for '{pair_name}': {e}")
                            stats = {'r': np.nan, 'rms': np.nan, 'n': len(alignment_result.ref_data), 'error': str(e)}
                    else:
                        print(f"[ComparisonWizard] No PairAnalyzer available, using basic fallback")
                        stats = {'r': np.nan, 'rms': np.nan, 'n': len(alignment_result.ref_data), 'error': 'No PairAnalyzer'}
                    
                    computation_time = time.time() - start_time
                    
                    # Update legacy data structures
                    self.pair_aligned_data[pair_name] = aligned_data
                    self.pair_statistics[pair_name] = stats
                    
                    # Prepare and cache new results
                    plot_data = self._prepare_plot_data(aligned_data, pair_info)
                    
                    computation_result = {
                        'aligned_data': aligned_data,
                        'statistics': stats,
                        'pair_config': pair_info
                    }
                    
                    pair_result = PairResult(
                        pair_id=pair_name,
                        computation_result=computation_result,
                        plot_data=plot_data,
                        method_hash=self.current_method_hash,
                        computation_time=computation_time
                    )
                    
                    self.pair_cache.store_pair_result(pair_result)
                    
                    # Update table
                    self._update_pair_statistics(pair_name, stats)
                    
                except Exception as e:
                    print(f"[ComparisonWizard] Error recomputing pair '{pair_name}': {str(e)}")
                    
        # Force table refresh
        self._force_table_refresh()
        
        # Regenerate plot with all pairs
        plot_config = self.window._get_plot_config() if hasattr(self.window, '_get_plot_config') else {}
        plot_type = self._determine_plot_type_from_pairs(checked_pairs)
        plot_config['plot_type'] = plot_type
        plot_config['checked_pairs'] = checked_pairs
        self._generate_scatter_plot(checked_pairs, plot_config)
        
        # Update cumulative display
        self._update_cumulative_display()
        
        print(f"[ComparisonWizard] Recomputation complete")
        
    def refresh_all_plots(self):
        """Refresh all plots and tables with current settings - used by refresh button with PairAnalyzer integration"""
        try:
            print("[ComparisonWizardManager] Refreshing all plots and tables")
            
            if not self.window:
                print("[ComparisonWizardManager] No window available for refresh")
                return
            
            # Get current checked pairs
            checked_pairs = self.window.get_checked_pairs()
            if not checked_pairs:
                print("[ComparisonWizardManager] No checked pairs to refresh")
                return
            
            # Clear all existing plots
            self._clear_all_plots()
            
            # Invalidate entire cache to force fresh computation
            self._coordinate_cache_invalidation()
            
            # Get current method and parameters
            current_method = self.window.method_list.currentItem().text() if self.window.method_list.currentItem() else "Correlation Analysis"
            method_params = self.window._get_method_parameters_from_controls() if hasattr(self.window, '_get_method_parameters_from_controls') else {}
            
            # Try PairAnalyzer for computation if available
            if self.pair_analyzer:
                try:
                    print("[ComparisonWizardManager] Using PairAnalyzer for refresh computation")
                    
                    # Prepare comparison configuration
                    comparison_config = {
                        'method': current_method,
                        'method_params': method_params,
                        'refresh_mode': True
                    }
                    
                    # Get script configuration
                    script_config = self._get_script_configuration()
                    
                    # Run analysis through PairAnalyzer
                    analysis_session = self.pair_analyzer.analyze(comparison_config, self.pair_manager, script_config)
                    
                    # Process results for existing plotting system
                    if analysis_session.analysis_results:
                        print(f"[ComparisonWizardManager] PairAnalyzer provided refresh results for {len(checked_pairs)} pairs")
                        
                        # Update pair statistics from analysis results
                        for pair in checked_pairs:
                            pair_name = pair['name']
                            if pair_name in analysis_session.pair_results:
                                pair_result = analysis_session.pair_results[pair_name]
                                if 'statistics' in pair_result:
                                    self.pair_statistics[pair_name] = pair_result['statistics']
                                if 'aligned_data' in pair_result:
                                    self.pair_aligned_data[pair_name] = pair_result['aligned_data']
                        
                        # Update table with new statistics
                        self._force_table_refresh()
                        
                    else:
                        print("[ComparisonWizardManager] PairAnalyzer returned empty results during refresh")
                        
                except Exception as e:
                    print(f"[ComparisonWizardManager] PairAnalyzer failed during refresh: {e}")
                    # Continue with existing plotting system
            
            # Get current plot configuration
            plot_config = self.window._get_plot_config()
            
            # Regenerate plots with current settings
            self._on_plot_generated(plot_config)
            
            # Update cumulative display
            self._update_cumulative_display()
            
            print(f"[ComparisonWizardManager] Successfully refreshed plots for {len(checked_pairs)} pairs")
            
        except Exception as e:
            print(f"[ComparisonWizardManager] Error refreshing plots: {e}")
            import traceback
            traceback.print_exc()
        
    def on_visibility_changed(self, pair_name: str, visible: bool):
        """Handle when pair visibility is toggled - visual only with PairAnalyzer integration"""
        print(f"[ComparisonWizard] Toggling visibility for pair '{pair_name}': {visible}")
        
        # Update matplotlib artists visibility
        self.pair_cache.set_pair_visibility(pair_name, visible)
        
        # Update PairAnalyzer visibility if available
        if self.pair_analyzer:
            try:
                # Find the current active session
                for session_id, session in self.pair_analyzer.active_sessions.items():
                    if session.pair_results and pair_name in session.pair_results:
                        self.pair_analyzer.update_pair_visibility(session_id, pair_name, visible)
                        break
            except Exception as e:
                print(f"[ComparisonWizard] Error updating PairAnalyzer visibility: {e}")
        
        # Redraw canvas
        if hasattr(self.window, 'canvas') and self.window.canvas:
            self.window.canvas.draw()
        
        # Update cumulative display
        self._update_cumulative_display()
        
    def _on_plot_generated(self, plot_config):
        """Handle plot generation request from step 2"""
        # Get checked pairs with marker types
        checked_pairs = plot_config.get('checked_pairs', [])
        
        if not checked_pairs:
            QMessageBox.warning(self.window, "No Pairs Selected", 
                              "Please select at least one pair to generate a plot.")
            return
        
        # Determine plot type from comparison method of first pair
        plot_type = self._determine_plot_type_from_pairs(checked_pairs)
        plot_config['plot_type'] = plot_type
        
        # Include overlay configurations in plot config
        self._enhance_plot_config_with_overlays(plot_config)
        
        print(f"[ComparisonWizard] Generating {plot_type} plot for {len(checked_pairs)} pairs")
        
        # Generate plots on all tabs
        self._generate_all_visualizations(checked_pairs, plot_config)
    
    def _determine_plot_type_from_pairs(self, checked_pairs):
        """Determine appropriate plot type based on comparison methods"""
        if not checked_pairs:
            return 'scatter'
        
        # Get the comparison method from the first pair
        first_pair = checked_pairs[0]
        comparison_method = first_pair.get('comparison_method', 'Correlation Analysis')
        
        # Convert to registry name and get plot type from comparison class
        if COMPARISON_AVAILABLE:
            try:
                registry_name = self.get_registry_name_from_display(comparison_method)
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls and hasattr(comparison_cls, 'plot_type'):
                    return comparison_cls.plot_type
            except Exception as e:
                print(f"[ComparisonWizardManager] Error determining plot type: {e}")
        
        # Fallback to scatter for all methods
        return 'scatter'
        
    def _generate_all_visualizations(self, checked_pairs, plot_config):
        """Generate plots for all tabs (scatter, histogram, heatmap)"""
        # Generate scatter plot (main comparison plot)
        self._generate_scatter_plot(checked_pairs, plot_config)
        
        # Generate histogram plot
        self._generate_histogram_plot(checked_pairs, plot_config)
        
        # Generate heatmap plot
        self._generate_heatmap_plot(checked_pairs, plot_config)
        
    def _generate_scatter_plot(self, checked_pairs, plot_config):
        """Generate scatter plot (renamed from _generate_multi_pair_plot)"""
        self._generate_multi_pair_plot(checked_pairs, plot_config)
        
    def _generate_histogram_plot(self, checked_pairs, plot_config):
        """Generate histogram plot for distribution comparison"""
        if not hasattr(self.window, 'histogram_canvas') or not self.window.histogram_canvas:
            return
            
        fig = self.window.histogram_canvas.figure
        self._clear_figure_completely(fig)
        
        # Collect all data for histogram
        all_ref_data = []
        all_test_data = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            
            # Find the pair in PairManager
            pair_obj = None
            if hasattr(self, 'pair_manager') and self.pair_manager:
                for p in self.pair_manager.get_pairs_in_order():
                    if p.name == pair_name:
                        pair_obj = p
                        break
            
            if not pair_obj or not pair_obj.has_aligned_data():
                continue
                
            aligned_data = pair_obj.get_aligned_data()
            ref_data = aligned_data['ref_data']
            test_data = aligned_data['test_data']
            
            if ref_data is None or test_data is None or len(ref_data) == 0:
                continue
                
            # Filter valid data
            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
            ref_clean = ref_data[valid_mask]
            test_clean = test_data[valid_mask]
            
            if len(ref_clean) > 0:
                all_ref_data.extend(ref_clean)
                all_test_data.extend(test_clean)
        
        if not all_ref_data:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No valid data for histogram', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Data Available')
            self.window.histogram_canvas.draw()
            return
        
        # Create histogram plot
        ax = fig.add_subplot(111)
        
        # Create side-by-side histograms
        ax.hist(all_ref_data, bins=50, alpha=0.7, label='Reference', color='blue', density=True)
        ax.hist(all_test_data, bins=50, alpha=0.7, label='Test', color='red', density=True)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        try:
            fig.tight_layout()
        except:
            pass
        self.window.histogram_canvas.draw()
        
    def _generate_heatmap_plot(self, checked_pairs, plot_config):
        """Generate heatmap plot for density visualization"""
        if not hasattr(self.window, 'heatmap_canvas') or not self.window.heatmap_canvas:
            return
            
        fig = self.window.heatmap_canvas.figure
        self._clear_figure_completely(fig)
        
        # Collect all data for heatmap
        all_ref_data = []
        all_test_data = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            
            # Find the pair in PairManager
            pair_obj = None
            if hasattr(self, 'pair_manager') and self.pair_manager:
                for p in self.pair_manager.get_pairs_in_order():
                    if p.name == pair_name:
                        pair_obj = p
                        break
            
            if not pair_obj or not pair_obj.has_aligned_data():
                continue
                
            aligned_data = pair_obj.get_aligned_data()
            ref_data = aligned_data['ref_data']
            test_data = aligned_data['test_data']
            
            if ref_data is None or test_data is None or len(ref_data) == 0:
                continue
                
            # Filter valid data
            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
            ref_clean = ref_data[valid_mask]
            test_clean = test_data[valid_mask]
            
            if len(ref_clean) > 0:
                all_ref_data.extend(ref_clean)
                all_test_data.extend(test_clean)
        
        if not all_ref_data or len(all_ref_data) < 10:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Insufficient Data')
            self.window.heatmap_canvas.draw()
            return
        
        # Create heatmap/density plot
        ax = fig.add_subplot(111)
        
        try:
            # Create 2D histogram/heatmap
            all_ref_data = np.array(all_ref_data)
            all_test_data = np.array(all_test_data)
            
            # Create heatmap based on plot type
            plot_type = plot_config.get('plot_type', 'scatter')
            
            if plot_type == 'bland_altman':
                # For Bland-Altman, show heatmap of differences vs means
                means = (all_ref_data + all_test_data) / 2
                diffs = all_test_data - all_ref_data
                
                h, xedges, yedges = np.histogram2d(means, diffs, bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                im = ax.imshow(h.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
                ax.set_xlabel('Mean of Methods')
                ax.set_ylabel('Difference (Test - Reference)')
                ax.set_title('Bland-Altman Density Heatmap')
                
            else:
                # For correlation/scatter, show heatmap of ref vs test
                h, xedges, yedges = np.histogram2d(all_ref_data, all_test_data, bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                im = ax.imshow(h.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
                ax.set_xlabel('Reference Data')
                ax.set_ylabel('Test Data')
                ax.set_title('Correlation Density Heatmap')
                
                # Add identity line for reference
                min_val = min(np.min(all_ref_data), np.min(all_test_data))
                max_val = max(np.max(all_ref_data), np.max(all_test_data))
                ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.8, linewidth=2, label='y=x')
                ax.legend()
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Point Density')
            
        except Exception as e:
            print(f"[ComparisonWizard] Error creating heatmap: {e}")
            # Show error in console, not on plot canvas
            if hasattr(self, 'window') and hasattr(self.window, 'info_output'):
                self.window.info_output.append(f"⚠️ Error creating heatmap: {str(e)}")
            ax.text(0.5, 0.5, 'Heatmap generation failed - check console for details', 
                   ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=10)
            ax.set_title('Heatmap')
        
        ax.grid(True, alpha=0.3)
        
        try:
            fig.tight_layout()
        except:
            pass
        self.window.heatmap_canvas.draw()
        
    def _generate_multi_pair_plot(self, checked_pairs, plot_config):
        """Generate plot for multiple pairs with enhanced styling and performance"""
        if not checked_pairs:
            return
            
        try:
            # Clear the figure first
            fig = self.window.canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Get plot type from config or determine from pairs
            plot_type = plot_config.get('plot_type', self._determine_plot_type_from_pairs(checked_pairs))
            
            print(f"[ComparisonWizard] Generating {plot_type} plot for {len(checked_pairs)} pairs")
            
            # Generate plot content
            success = self._generate_plot_content(ax, checked_pairs, plot_config)
            
            if success:
                # Create channel from comparison results after successful plot generation
                self._create_comparison_channel(checked_pairs, plot_config, ax)
                
                # Apply common styling and finalization
                self._apply_common_plot_config(ax, fig, plot_config, checked_pairs)
                
                print(f"[ComparisonWizard] Plot generated successfully with {len(checked_pairs)} pairs")
            else:
                print(f"[ComparisonWizard] Plot generation failed")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error generating multi-pair plot: {e}")
            import traceback
            traceback.print_exc()
            
            # Show error message on plot
            ax.text(0.5, 0.5, f'Plot generation failed: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # Always draw the canvas
        self.window.canvas.draw()

    def _create_comparison_channel(self, checked_pairs, plot_config, ax):
        """Create individual channels from comparison results (one per pair) and add to channel manager"""
        try:
            # Skip channel creation if no channel manager available
            if not self.channel_manager:
                print("[ComparisonWizard] No channel manager available - skipping channel creation")
                return []
            
            # Get comparison method info
            method_name = plot_config.get('comparison_method', 'Unknown')
            method_params = plot_config.get('method_parameters', {})
            
            # Create a unique group ID for this comparison session
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method_clean = method_name.lower().replace(' ', '_')
            group_id = f"comparison_{method_clean}_{timestamp}_{hash(tuple([p['name'] for p in checked_pairs])) % 10000}"
            
            created_channels = []
            
            # Create one channel per pair
            for pair in checked_pairs:
                try:
                    # Get parent channels for this pair
                    ref_channel = self._get_channel(pair['ref_file'], pair['ref_channel'])
                    test_channel = self._get_channel(pair['test_file'], pair['test_channel'])
                    
                    if not (ref_channel and test_channel):
                        print(f"[ComparisonWizard] Could not find parent channels for pair: {pair['name']}")
                        continue
                    
                    # Get aligned data for this pair from Pair object
                    pair_obj = None
                    if hasattr(self, 'pair_manager') and self.pair_manager:
                        for p in self.pair_manager.get_pairs_in_order():
                            if p.name == pair['name']:
                                pair_obj = p
                                break
                    
                    if not pair_obj or not pair_obj.has_aligned_data():
                        print(f"[ComparisonWizard] No aligned data for pair: {pair['name']}")
                        continue
                    
                    aligned_data = pair_obj.get_aligned_data()
                    ref_data = aligned_data.get('ref_data')
                    test_data = aligned_data.get('test_data')
                    
                    if ref_data is None or test_data is None:
                        print(f"[ComparisonWizard] Invalid data for pair: {pair['name']}")
                        continue
                    
                    # Filter valid data for this pair
                    valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                    ref_clean = ref_data[valid_mask]
                    test_clean = test_data[valid_mask]
                    
                    if len(ref_clean) == 0:
                        print(f"[ComparisonWizard] No valid data points for pair: {pair['name']}")
                        continue
                    
                    # For correlation/scatter plots: x=ref_data, y=test_data
                    # For other plot types, this might need adjustment
                    xdata = ref_clean
                    ydata = test_clean
                    
                    # Calculate individual pair statistics
                    pair_stats = self._calculate_pair_statistics(ref_clean, test_clean, method_name, method_params)
                    
                    # Create comprehensive metadata for this pair
                    pair_metadata = {
                        'comparison_group_id': group_id,
                        'comparison_method': method_name,
                        'method_parameters': method_params,
                        'group_timestamp': datetime.now().isoformat(),
                        'total_pairs_in_group': len(checked_pairs),
                        'pair_info': {
                            'name': pair['name'],
                            'ref_channel_id': ref_channel.channel_id,
                            'test_channel_id': test_channel.channel_id,
                            'ref_file': pair['ref_file'],
                            'test_file': pair['test_file'],
                            'ref_channel_name': pair['ref_channel'],
                            'test_channel_name': pair['test_channel'],
                            'alignment_config': pair.get('alignment_config', {}),
                            'data_points': len(ref_clean),
                            'r_squared': pair.get('r_squared'),
                            'marker_type': pair.get('marker_type', '○ Circle'),
                            'marker_color': pair.get('marker_color', '🔵 Blue')
                        },
                        'statistical_results': pair_stats,
                        'overlay_config': plot_config.get('overlay_config', {}),
                        'creation_timestamp': datetime.now().isoformat(),
                        'comparison_type': 'individual_pair'
                    }
                    
                    # Create the comparison channel for this pair
                    from channel import Channel
                    comparison_channel = Channel.from_comparison(
                        parent_channels=[ref_channel, test_channel],
                        comparison_method=method_name,
                        xdata=xdata,
                        ydata=ydata,
                        xlabel=plot_config.get('xlabel', 'Reference Data'),
                        ylabel=plot_config.get('ylabel', 'Test Data'),
                        legend_label=pair['name'],  # Use pair name as channel name
                        pairs_metadata=[pair_metadata['pair_info']],  # Single pair info
                        statistical_results=pair_stats,
                        method_parameters=method_params,
                        overlay_config=plot_config.get('overlay_config', {}),
                        tags=["comparison", method_clean, f"group_{group_id}"]
                    )
                    
                    # Override metadata with our comprehensive version
                    comparison_channel.metadata = pair_metadata
                    
                    # Set marker and color from pair styling
                    marker_map = {
                        '○ Circle': 'o', '□ Square': 's', '△ Triangle': '^', '◇ Diamond': 'D',
                        '▽ Inverted Triangle': 'v', '◁ Left Triangle': '<', '▷ Right Triangle': '>',
                        '⬟ Pentagon': 'p', '✦ Star': '*', '⬢ Hexagon': 'h'
                    }
                    color_map = {
                        '🔵 Blue': '#1f77b4', '🔴 Red': '#d62728', '🟢 Green': '#2ca02c',
                        '🟣 Purple': '#9467bd', '🟠 Orange': '#ff7f0e', '🟤 Brown': '#8c564b',
                        '🩷 Pink': '#e377c2', '⚫ Gray': '#7f7f7f', '🟡 Yellow': '#bcbd22',
                        '🔶 Cyan': '#17becf'
                    }
                    
                    comparison_channel.marker = marker_map.get(pair.get('marker_type', '○ Circle'), 'o')
                    comparison_channel.color = color_map.get(pair.get('marker_color', '🔵 Blue'), '#1f77b4')
                    comparison_channel.style = 'None'  # Comparison channels show only markers, no connecting lines
                    
                    # Apply additional marker properties from marker wizard
                    if 'marker_size' in pair:
                        comparison_channel.marker_size = pair.get('marker_size', 50)
                    if 'marker_alpha' in pair:
                        comparison_channel.alpha = pair.get('marker_alpha', 0.8)
                    if 'edge_color' in pair:
                        comparison_channel.edge_color = pair.get('edge_color', '#000000')
                    if 'edge_width' in pair:
                        comparison_channel.edge_width = pair.get('edge_width', 1.0)
                    if 'fill_style' in pair:
                        comparison_channel.fill_style = pair.get('fill_style', 'full')
                    
                    # Debug output for marker assignment
                    print(f"[ComparisonWizard] Pair '{pair['name']}' marker assignment:")
                    print(f"  - marker_type: {pair.get('marker_type', 'NOT SET')} -> {comparison_channel.marker}")
                    print(f"  - marker_color: {pair.get('marker_color', 'NOT SET')} -> {comparison_channel.color}")
                    print(f"  - marker_size: {pair.get('marker_size', 'DEFAULT')} -> {comparison_channel.marker_size}")
                    print(f"  - marker_alpha: {pair.get('marker_alpha', 'DEFAULT')} -> {comparison_channel.alpha}")
                    print(f"  - edge_color: {pair.get('edge_color', 'DEFAULT')} -> {comparison_channel.edge_color}")
                    print(f"  - edge_width: {pair.get('edge_width', 'DEFAULT')} -> {comparison_channel.edge_width}")
                    print(f"  - fill_style: {pair.get('fill_style', 'DEFAULT')} -> {comparison_channel.fill_style}")
                    print(f"  - style: {comparison_channel.style}")
                    
                    # Add to channel manager
                    self.channel_manager.add_channel(comparison_channel)
                    created_channels.append(comparison_channel)
                    
                    print(f"[ComparisonWizard] Created channel for pair: {pair['name']} (ID: {comparison_channel.channel_id})")
                    
                except Exception as e:
                    print(f"[ComparisonWizard] Error creating channel for pair {pair['name']}: {e}")
                    continue
            
            if created_channels:
                print(f"[ComparisonWizard] Successfully created {len(created_channels)} channels for comparison group: {group_id}")
                print(f"[ComparisonWizard] Channels can be grouped by metadata['comparison_group_id'] = '{group_id}'")
                
                # Create overlay objects for all comparison channels
                self._create_overlays_for_comparison_channels(created_channels, method_name, plot_config)
            else:
                print("[ComparisonWizard] No channels were created - check pair data and alignment")
            
            return created_channels
            
        except Exception as e:
            print(f"[ComparisonWizard] Error creating comparison channels: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_overlays_for_comparison_channel(self, comparison_channel, method_name, plot_config):
        """Create Overlay objects for a comparison channel based on the method's overlay options"""
        try:
            # Get method info to find overlay options
            method_info = self.get_method_info(method_name)
            if not method_info or 'overlay_options' not in method_info:
                print(f"[ComparisonWizard] No overlay options found for method: {method_name}")
                return
            
            overlay_options = method_info['overlay_options']
            created_overlays = []
            
            # Create one Overlay object for each overlay option
            for overlay_id, overlay_option in overlay_options.items():
                try:
                    # Get overlay configuration from plot config
                    overlay_config = plot_config.get('overlay_config', {}).get(overlay_id, {})
                    
                    # Determine overlay type based on overlay_id and method
                    overlay_type = self._determine_overlay_type(overlay_id, method_name)
                    
                    # Create style dictionary for the overlay
                    style = {
                        'color': overlay_config.get('color', '#3498db'),
                        'linestyle': overlay_config.get('linestyle', '-'),
                        'linewidth': overlay_config.get('linewidth', 2),
                        'alpha': overlay_config.get('alpha', 0.8),
                        'show': overlay_config.get('show', overlay_option.get('default', True))
                    }
                    
                    # Add method-specific style properties
                    if overlay_type == 'fill':
                        style['facecolor'] = overlay_config.get('facecolor', '#1f77b4')
                        style['alpha'] = overlay_config.get('alpha', 0.3)
                    elif overlay_type == 'text':
                        style['fontsize'] = overlay_config.get('fontsize', 9)
                        style['bbox'] = overlay_config.get('bbox', {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8})
                    
                    # Create the Overlay object
                    from overlay import Overlay
                    overlay = Overlay(
                        id=f"{comparison_channel.channel_id}_{overlay_id}",
                        name=overlay_option.get('label', overlay_id.replace('_', ' ').title()),
                        type=overlay_type,
                        style=style,
                        channel=comparison_channel.channel_id,
                        show=style['show'],
                        tags=[method_name.lower(), 'comparison', overlay_id]
                    )
                    
                    # Store overlay in comparison channel metadata
                    if 'overlays' not in comparison_channel.metadata:
                        comparison_channel.metadata['overlays'] = []
                    comparison_channel.metadata['overlays'].append(overlay.to_dict())
                    
                    created_overlays.append(overlay)
                    print(f"[ComparisonWizard] Created overlay '{overlay.name}' for channel {comparison_channel.channel_id}")
                    
                except Exception as e:
                    print(f"[ComparisonWizard] Error creating overlay {overlay_id}: {e}")
                    continue
            
            print(f"[ComparisonWizard] Created {len(created_overlays)} overlays for comparison channel {comparison_channel.channel_id}")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error creating overlays for comparison channel: {e}")
            import traceback
            traceback.print_exc()

    def _create_overlays_for_comparison_channels(self, comparison_channels, method_name, plot_config):
        """Create overlay objects for comparison channels based on the comparison method's overlay options"""
        try:
            # Get method info to determine overlay options
            method_info = self.get_method_info(method_name)
            if not method_info or 'overlay_options' not in method_info:
                print(f"[ComparisonWizard] No overlay options found for method: {method_name}")
                return
            
            overlay_options = method_info['overlay_options']
            total_created_overlays = 0
            
            # Create overlay objects for each overlay option
            for overlay_id, overlay_config in overlay_options.items():
                try:
                    # Determine overlay type based on overlay_id and config
                    overlay_type = self._determine_overlay_type(overlay_id, method_name)
                    
                    # Get current visibility state from plot config
                    is_visible = plot_config.get(f'show_{overlay_id}', overlay_config.get('default', True))
                    
                    # Get all channel IDs for this overlay (one per comparison pair)
                    channel_ids = [channel.channel_id for channel in comparison_channels]
                    
                    # Create overlay object with multiple channel IDs
                    from overlay import Overlay
                    from datetime import datetime
                    overlay = Overlay(
                        overlay_id=overlay_id,
                        overlay_type=overlay_type,
                        channel_id=channel_ids,  # Multiple channel IDs for all pairs
                        config=overlay_config,
                        is_visible=is_visible,
                        creation_timestamp=datetime.now().isoformat()
                    )
                    
                    # Store overlay in each channel's metadata
                    for channel in comparison_channels:
                        if 'overlays' not in channel.metadata:
                            channel.metadata['overlays'] = []
                        channel.metadata['overlays'].append(overlay)
                    
                    total_created_overlays += 1
                    print(f"[ComparisonWizard] Created overlay '{overlay_id}' for {len(channel_ids)} channels")
                    
                except Exception as e:
                    print(f"[ComparisonWizard] Error creating overlay '{overlay_id}': {e}")
                    continue
            
            print(f"[ComparisonWizard] Created {total_created_overlays} overlays for {len(comparison_channels)} comparison channels")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error creating overlays for comparison channels: {e}")
            import traceback
            traceback.print_exc()

    def _determine_overlay_type(self, overlay_id, method_name):
        """Determine the overlay type based on overlay_id and comparison method"""
        overlay_id_lower = overlay_id.lower()
        method_lower = method_name.lower()
        
        # Method-specific overlay type detection
        if 'bland' in method_lower:
            if 'bias' in overlay_id_lower:
                return 'line'
            elif 'limit' in overlay_id_lower:
                return 'line'
            elif 'confidence' in overlay_id_lower:
                return 'fill'
            elif 'statistical' in overlay_id_lower:
                return 'text'
        elif 'correlation' in method_lower:
            if 'regression' in overlay_id_lower or 'identity' in overlay_id_lower:
                return 'line'
            elif 'statistical' in overlay_id_lower:
                return 'text'
        
        # Generic overlay type detection
        if any(keyword in overlay_id_lower for keyword in ['line', 'bias', 'limit', 'regression', 'identity']):
            return 'line'
        elif any(keyword in overlay_id_lower for keyword in ['confidence', 'interval', 'fill', 'shading']):
            return 'fill'
        elif any(keyword in overlay_id_lower for keyword in ['text', 'statistical', 'result']):
            return 'text'
        elif any(keyword in overlay_id_lower for keyword in ['marker', 'point']):
            return 'marker'
        
        # Default to line type
        return 'line'

    def _calculate_pair_statistics(self, ref_data, test_data, method_name, method_params):
        """Calculate statistics for an individual pair"""
        try:
            # Basic statistics
            stats = {
                'n_points': len(ref_data),
                'ref_mean': float(np.mean(ref_data)),
                'test_mean': float(np.mean(test_data)),
                'ref_std': float(np.std(ref_data)),
                'test_std': float(np.std(test_data)),
                'method': method_name
            }
            
            # Calculate correlation if we have enough data
            if len(ref_data) > 2:
                from scipy.stats import pearsonr, spearmanr
                try:
                    pearson_r, pearson_p = pearsonr(ref_data, test_data)
                    stats['pearson_r'] = float(pearson_r)
                    stats['pearson_p'] = float(pearson_p)
                    stats['r_squared'] = float(pearson_r ** 2)
                except:
                    stats['pearson_r'] = np.nan
                    stats['pearson_p'] = np.nan
                    stats['r_squared'] = np.nan
                
                try:
                    spearman_r, spearman_p = spearmanr(ref_data, test_data)
                    stats['spearman_r'] = float(spearman_r)
                    stats['spearman_p'] = float(spearman_p)
                except Exception:
                    stats['spearman_r'] = np.nan
                    stats['spearman_p'] = np.nan
            
            # Calculate RMSE and MAE
            try:
                differences = test_data - ref_data
                stats['rmse'] = float(np.sqrt(np.mean(differences ** 2)))
                stats['mae'] = float(np.mean(np.abs(differences)))
                stats['mean_bias'] = float(np.mean(differences))
            except Exception:
                stats['rmse'] = np.nan
                stats['mae'] = np.nan
                stats['mean_bias'] = np.nan
            
            return stats
                        
        except Exception as e:
            print(f"[ComparisonWizard] Error calculating pair statistics: {e}")
            return {
                'n_points': len(ref_data) if ref_data is not None else 0,
                'method': method_name,
                'error': str(e)
            }

    def _extract_plot_data_from_axes(self, ax, method_name):
        """Extract plot data from matplotlib axes"""
        try:
            # Get the first scatter plot or line plot from the axes
            for child in ax.get_children():
                if hasattr(child, 'get_offsets'):  # Scatter plot
                    offsets = child.get_offsets()
                    if len(offsets) > 0:
                        xdata = offsets[:, 0]
                        ydata = offsets[:, 1]
                        return {
                            'xdata': xdata,
                            'ydata': ydata,
                            'xlabel': ax.get_xlabel(),
                            'ylabel': ax.get_ylabel()
                        }
                elif hasattr(child, 'get_xdata'):  # Line plot
                    xdata = child.get_xdata()
                    ydata = child.get_ydata()
                    if len(xdata) > 0 and len(ydata) > 0:
                        return {
                            'xdata': xdata,
                            'ydata': ydata,
                            'xlabel': ax.get_xlabel(),
                            'ylabel': ax.get_ylabel()
                        }
            
            print(f"[ComparisonWizard] Could not find plot data in axes children")
            return None
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error extracting plot data: {e}")
            return None

    def _get_combined_statistical_results(self, checked_pairs, method_name):
        """Get combined statistical results from all pairs"""
        try:
            combined_results = {
                'method': method_name,
                'n_pairs': len(checked_pairs),
                'total_data_points': 0,
                'pair_results': []
            }
            
            for pair in checked_pairs:
                pair_name = pair['name']
                if pair_name in self.pair_aligned_data:
                    aligned_data = self.pair_aligned_data[pair_name]
                    ref_data = aligned_data.get('ref_data')
                    test_data = aligned_data.get('test_data')
                    
                    if ref_data is not None and test_data is not None:
                        combined_results['total_data_points'] += len(ref_data)
                        
                        # Calculate basic statistics for this pair
                        pair_stats = {
                            'pair_name': pair_name,
                            'n_points': len(ref_data),
                            'r_squared': pair.get('r_squared'),
                            'ref_mean': float(np.mean(ref_data)) if len(ref_data) > 0 else 0,
                            'test_mean': float(np.mean(test_data)) if len(test_data) > 0 else 0,
                            'ref_std': float(np.std(ref_data)) if len(ref_data) > 0 else 0,
                            'test_std': float(np.std(test_data)) if len(test_data) > 0 else 0
                        }
                        combined_results['pair_results'].append(pair_stats)
            
            return combined_results
            
        except Exception as e:
            print(f"[ComparisonWizard] Error getting combined statistical results: {e}")
            return {'method': method_name, 'error': str(e)}

    def _get_pair_styling_info(self, checked_pairs):
        """Get styling information for individual pairs"""
        try:
            # Marker mapping
            marker_map = {
                '○ Circle': 'o',
                '□ Square': 's', 
                '△ Triangle': '^',
                '◇ Diamond': 'D',
                '▽ Inverted Triangle': 'v',
                '◁ Left Triangle': '<',
                '▷ Right Triangle': '>',
                '⬟ Pentagon': 'p',
                '✦ Star': '*',
                '⬢ Hexagon': 'h'
            }
            # Color mapping
            color_map = {
                '🔵 Blue': '#1f77b4',
                '🔴 Red': '#d62728',
                '🟢 Green': '#2ca02c',
                '🟣 Purple': '#9467bd',
                '🟠 Orange': '#ff7f0e',
                '🟤 Brown': '#8c564b',
                '🩷 Pink': '#e377c2',
                '⚫ Gray': '#7f7f7f',
                '🟡 Yellow': '#bcbd22',
                '🔶 Cyan': '#17becf'
            }
            pair_styling = []
            for pair in checked_pairs:
                pair_name = pair['name']
                
                # Find the pair in PairManager
                pair_obj = None
                if hasattr(self, 'pair_manager') and self.pair_manager:
                    for p in self.pair_manager.get_pairs_in_order():
                        if p.name == pair_name:
                            pair_obj = p
                            break
                
                if pair_obj and pair_obj.has_aligned_data():
                    aligned_data = pair_obj.get_aligned_data()
                    ref_data = aligned_data.get('ref_data')
                    test_data = aligned_data.get('test_data')
                    if ref_data is not None and test_data is not None:
                        # Use styling from Pair object
                        marker_text = pair_obj.marker_type
                        color_text = pair_obj.color
                        marker_size = pair_obj.marker_size
                        legend_label = pair_obj.legend_label or pair_name
                        
                        valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                        ref_clean = ref_data[valid_mask]
                        test_clean = test_data[valid_mask]
                        if len(ref_clean) > 0:
                            pair_styling.append({
                                'pair_name': pair_name,
                                'ref_data': ref_clean,
                                'test_data': test_clean,
                                'marker': marker_map.get(marker_text, 'o'),
                                'color': color_map.get(color_text, '#1f77b4'),
                                'marker_size': marker_size,
                                'legend_label': legend_label,
                                'n_points': len(ref_clean)
                            })
            return pair_styling
        except Exception as e:
            print(f"[ComparisonWizard] Error getting pair styling info: {e}")
            return []
    
    def _generate_plot_content(self, ax, checked_pairs, plot_config):
        """Generate plot content using plot generators from comparison folder"""
        try:
            # Collect all data from checked pairs
            all_ref_data = []
            all_test_data = []
            
            for pair in checked_pairs:
                pair_name = pair['name']
                
                # Get aligned data from Pair object
                try:
                    # Find the pair in PairManager
                    pair_obj = None
                    if hasattr(self, 'pair_manager') and self.pair_manager:
                        for p in self.pair_manager.get_pairs_in_order():
                            if p.name == pair_name:
                                pair_obj = p
                                break
                    
                    if pair_obj and pair_obj.has_aligned_data():
                        # Use aligned data stored in the Pair object
                        aligned_data = pair_obj.get_aligned_data()
                        ref_data = aligned_data['ref_data']
                        test_data = aligned_data['test_data']
                        
                        if ref_data is not None and test_data is not None:
                            # Filter valid data
                            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                            ref_clean = ref_data[valid_mask]
                            test_clean = test_data[valid_mask]
                            
                            if len(ref_clean) > 0:
                                all_ref_data.extend(ref_clean)
                                all_test_data.extend(test_clean)
                        else:
                            print(f"[ComparisonWizard] Invalid aligned data for pair '{pair_name}'")
                    else:
                        print(f"[ComparisonWizard] No aligned data found for pair '{pair_name}' - pair may need re-alignment")
                        
                except Exception as e:
                    print(f"[ComparisonWizard] Error getting aligned data for pair '{pair_name}': {e}")
                    continue
            
            if not all_ref_data:
                ax.text(0.5, 0.5, 'No valid data for plotting', 
                       ha='center', va='center', transform=ax.transAxes)
                return False
            
            # Convert to numpy arrays
            all_ref_data = np.array(all_ref_data)
            all_test_data = np.array(all_test_data)
            
            # Get method info
            method_name = plot_config.get('comparison_method', 'Correlation Analysis')
            if checked_pairs:
                method_name = checked_pairs[0].get('comparison_method', method_name)
            
            # Try to create the comparison method instance
            method_instance = None
            if COMPARISON_AVAILABLE and method_name:
                # Get method parameters from plot config first
                method_params = plot_config.get('method_parameters', {})
                
                # If no method parameters in plot config, try to get from checked pairs
                if not method_params and checked_pairs:
                    method_params = checked_pairs[0].get('method_parameters', {})
                
                # Also include overlay parameters
                overlay_params = {k: v for k, v in plot_config.items() if k.startswith('show_') or k in ['confidence_interval', 'custom_line']}
                method_params.update(overlay_params)
                
                print(f"[ComparisonWizard] Using method parameters: {method_params}")
                
                # Use dynamic display name to registry name conversion
                registry_name = self.get_registry_name_from_display(method_name)
                if registry_name:
                    comparison_cls = self.comparison_registry.get(registry_name)
                    if comparison_cls:
                        method_instance = comparison_cls(**method_params)
                        print(f"[ComparisonWizard] Successfully created {registry_name} method instance")
                    else:
                        print(f"[ComparisonWizard] Comparison class not found for registry name: {registry_name}")
                        method_instance = None
                else:
                    print(f"[ComparisonWizard] Could not convert display name to registry name: {method_name}")
                    method_instance = None
            
            if method_instance and hasattr(method_instance, 'generate_plot'):
                print(f"[ComparisonWizard] Using comparison method plot generation for {method_name}")
                
                try:
                    # Check if we should use custom scripts
                    use_custom_plot = False
                    use_custom_stat = False
                    
                    if self.window:
                        use_custom_plot = self.window._should_use_custom_plot_script()
                        use_custom_stat = self.window._should_use_custom_stat_script()
                        print(f"[ComparisonWizard] Custom script flags: plot={use_custom_plot}, stat={use_custom_stat}")
                    
                    # Calculate statistics first (with custom script support)
                    if use_custom_stat:
                        print(f"[ComparisonWizard] Attempting to use custom stat script for {method_name}")
                        try:
                            # First get plot data using the method's plot_script
                            x_data, y_data, plot_metadata = method_instance.plot_script(all_ref_data, all_test_data, method_params)
                            
                            # Execute custom stat script
                            stats_results = self.window._execute_custom_stat_script(x_data, y_data, all_ref_data, all_test_data, method_params)
                            
                            if stats_results:
                                print(f"[ComparisonWizard] Successfully used custom stat script for {method_name}")
                            else:
                                print(f"[ComparisonWizard] DEBUG: Custom stat script failed, falling back to original method")
                                stats_results = method_instance.calculate_stats(all_ref_data, all_test_data)
                        except Exception as e:
                            print(f"[ComparisonWizard] DEBUG: Custom stat script error: {e}, falling back to original method")
                            stats_results = method_instance.calculate_stats(all_ref_data, all_test_data)
                    else:
                        print(f"[ComparisonWizard] DEBUG: Using original stat method (no custom script)")
                        stats_results = method_instance.calculate_stats(all_ref_data, all_test_data)
                    
                    print(f"[ComparisonWizard] Successfully calculated stats for {method_name}")
                    
                    # Add pair styling information to plot config
                    plot_config_with_pairs = plot_config.copy()
                    plot_config_with_pairs['pair_styling'] = self._get_pair_styling_info(checked_pairs)
                    
                    # Enhanced plot config with overlay options
                    enhanced_config = self._enhance_plot_config_with_overlay_options(plot_config_with_pairs, checked_pairs)
                    
                    print(f"[ComparisonWizard] Plot config keys: {list(enhanced_config.keys())}")
                    
                    # Generate plot (with custom script support)
                    if use_custom_plot:
                        print(f"[ComparisonWizard] Attempting to use custom plot script for {method_name}")
                        try:
                            # Clear axes before custom script execution to prevent overlapping
                            ax.clear()
                            print(f"[ComparisonWizard] DEBUG: Axes cleared before custom script execution")
                            
                            # Execute custom plot script
                            x_data, y_data, plot_metadata = self.window._execute_custom_plot_script(all_ref_data, all_test_data, method_params)
                            
                            if x_data is not None and y_data is not None:
                                print(f"[ComparisonWizard] Successfully used custom plot script for {method_name}")
                                
                                # Create plot using custom script results with pair styling
                                pair_styling = enhanced_config.get('pair_styling', [])
                                if pair_styling:
                                    # Plot each pair with its own styling
                                    for pair_info in pair_styling:
                                        # Get the data for this pair from the custom script results
                                        pair_ref = pair_info['ref_data']
                                        pair_test = pair_info['test_data']
                                        
                                        # Apply the custom transformation to this pair's data
                                        pair_x_data, pair_y_data, _ = self.window._execute_custom_plot_script(pair_ref, pair_test, method_params)
                                        
                                        if pair_x_data is not None and pair_y_data is not None:
                                            ax.scatter(pair_x_data, pair_y_data, 
                                                     s=pair_info.get('marker_size', 50), 
                                                     color=pair_info['color'], 
                                                     marker=pair_info['marker'],
                                                     label=pair_info.get('legend_label', pair_info['pair_name']))
                                else:
                                    # Plot all data with default styling
                                    ax.scatter(x_data, y_data, alpha=0.6, s=50)
                                
                                # Set labels and title from custom script metadata
                                ax.set_xlabel(plot_metadata.get('x_label', 'X Data'))
                                ax.set_ylabel(plot_metadata.get('y_label', 'Y Data'))
                                ax.set_title(plot_metadata.get('title', method_name))
                                
                                # Add grid if requested
                                if enhanced_config.get('show_grid', True):
                                    ax.grid(True, alpha=0.3)
                                
                                # IMPORTANT: Add overlay generation after custom plot script execution
                                # This ensures that overlay options (show toggles) work with custom scripts
                                try:
                                    # After plotting the custom script results, add overlays using the original method
                                    # We need to call the method's overlay generation methods if they exist
                                    print(f"[ComparisonWizard] DEBUG: Adding overlays to custom plot for {method_name}")
                                    
                                    # Call the method's overlay generation methods if they exist
                                    if hasattr(method_instance, '_add_bland_altman_overlays'):
                                        # For Bland-Altman, we need means and differences
                                        method_instance._add_bland_altman_overlays(ax, x_data, y_data, enhanced_config, stats_results)
                                        print(f"[ComparisonWizard] DEBUG: Added Bland-Altman overlays to custom plot")
                                    elif hasattr(method_instance, '_add_overlay_elements'):
                                        # For generic overlay elements
                                        method_instance._add_overlay_elements(ax, x_data, y_data, enhanced_config, stats_results)
                                        print(f"[ComparisonWizard] DEBUG: Added generic overlay elements to custom plot")
                                    else:
                                        print(f"[ComparisonWizard] DEBUG: No specific overlay methods found for {method_name}")
                                        
                                except Exception as overlay_e:
                                    print(f"[ComparisonWizard] DEBUG: Error adding overlays to custom plot: {overlay_e}")
                                    # Don't fail the entire plot generation if overlay generation fails
                                    import traceback
                                    traceback.print_exc()
                                
                                print(f"[ComparisonWizard] Successfully generated plot using custom script for {method_name}")
                                return True
                            else:
                                print(f"[ComparisonWizard] DEBUG: Custom plot script failed, falling back to original method")
                                # Clear axes again before fallback to prevent overlapping
                                ax.clear()
                                method_instance.generate_plot(ax, all_ref_data, all_test_data, enhanced_config, stats_results)
                        except Exception as e:
                            print(f"[ComparisonWizard] DEBUG: Custom plot script error: {e}, falling back to original method")
                            import traceback
                            traceback.print_exc()
                            # Clear axes again before fallback to prevent overlapping
                            ax.clear()
                            method_instance.generate_plot(ax, all_ref_data, all_test_data, enhanced_config, stats_results)
                    else:
                        print(f"[ComparisonWizard] DEBUG: Using original plot method (no custom script)")
                        method_instance.generate_plot(ax, all_ref_data, all_test_data, enhanced_config, stats_results)
                    
                    print(f"[ComparisonWizard] Successfully generated plot for {method_name}")
                    return True
                    
                except Exception as e:
                    print(f"[ComparisonWizard] Error in {method_name} plot generation: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Show error message on plot instead of falling back
                    ax.text(0.5, 0.5, f'{method_name} plot generation failed:\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes, color='red', fontsize=10)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    return False
            else:
                # Ultimate fallback to hardcoded methods for compatibility
                print(f"[ComparisonWizard] No plot generator or method found, using fallback")
                plot_type = plot_config.get('plot_type', 'scatter')
                self._generate_fallback_plot_content(ax, all_ref_data, all_test_data, plot_config, plot_type, checked_pairs)
                return False
                
        except Exception as e:
            print(f"[ComparisonWizard] Error in plot generation: {e}")
            import traceback
            traceback.print_exc()
            # Show error in console, not on plot canvas
            if hasattr(self, 'window') and hasattr(self.window, 'info_output'):
                self.window.info_output.append(f"⚠️ Error generating plot: {str(e)}")
            # Generate empty plot instead of error message on canvas
            ax.text(0.5, 0.5, 'Plot generation failed - check console for details', 
                   ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return False
    
    def _enhance_plot_config_with_overlay_options(self, plot_config, checked_pairs):
        """Enhance plot config with overlay options from comparison methods and window overlay table"""
        enhanced_config = plot_config.copy()
        
        if not checked_pairs:
            return enhanced_config
        
        # Get overlay options from the first checked pair's comparison method
        first_pair = checked_pairs[0]
        method_name = first_pair.get('comparison_method')
        
        if method_name and COMPARISON_AVAILABLE:
            try:
                # Use dynamic display name to registry name conversion
                registry_name = self.get_registry_name_from_display(method_name)
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls:
                    method_info = comparison_cls().get_info()
                    if method_info and 'overlay_options' in method_info:
                        overlay_options = method_info['overlay_options']
                        
                        # First, set default values from the overlay options
                        for key, overlay_option in overlay_options.items():
                            if key not in enhanced_config:
                                # Use the default value from the overlay option
                                enhanced_config[key] = overlay_option.get('default', True)
                        
                        # Then, override with actual values from the window's overlay table
                        if self.window and hasattr(self.window, '_get_overlay_parameters'):
                            try:
                                overlay_params = self.window._get_overlay_parameters()
                                if overlay_params:
                                    enhanced_config.update(overlay_params)
                                    print(f"[ComparisonWizard] Updated plot config with overlay parameters from window: {overlay_params}")
                            except Exception as e:
                                print(f"[ComparisonWizard] Error getting overlay parameters from window: {e}")
                        
                        print(f"[ComparisonWizard] Enhanced plot config with {len(overlay_options)} overlay options from {method_name}")
                    
            except Exception as e:
                print(f"[ComparisonWizard] Error enhancing plot config with overlay options: {e}")
        
        return enhanced_config
    
    def _generate_fallback_plot_content(self, ax, all_ref_data, all_test_data, plot_config, plot_type, checked_pairs):
        """Fallback plot generation for backward compatibility"""
        import numpy as np
        from scipy import stats
        
        print(f"[ComparisonWizard] Fallback plot generated successfully with {len(checked_pairs)} pairs")
        
        # Simple fallback plot based on plot type
        if plot_type == 'scatter' or plot_type == 'correlation':
            # Create scatter plot
            ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
            
            # Add identity line if requested
            if plot_config.get('show_identity_line', True):
                min_val = min(np.min(all_ref_data), np.min(all_test_data))
                max_val = max(np.max(all_ref_data), np.max(all_test_data))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y = x')
            
            # Add regression line if requested
            if plot_config.get('show_regression_line', True):
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(all_ref_data, all_test_data)
                    line_x = np.array([np.min(all_ref_data), np.max(all_ref_data)])
                    line_y = slope * line_x + intercept
                    ax.plot(line_x, line_y, 'g-', alpha=0.8, label=f'Regression (R²={r_value**2:.3f})')
                except:
                    pass
            
            # Add statistical results if requested
            if plot_config.get('show_statistical_results', True):
                try:
                    r_val = np.corrcoef(all_ref_data, all_test_data)[0, 1]
                    rmse = np.sqrt(np.mean((all_test_data - all_ref_data)**2))
                    stats_text = f'R = {r_val:.3f}\nRMSE = {rmse:.3f}'
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                except:
                    pass
            
            ax.set_xlabel('Reference Data')
            ax.set_ylabel('Test Data')
            ax.set_title('Correlation Analysis')
            
        elif plot_type == 'bland_altman':
            differences = all_test_data - all_ref_data
            averages = (all_ref_data + all_test_data) / 2
            ax.scatter(averages, differences, alpha=0.6, s=20, c='blue')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Mean difference')
            
            # Add limits of agreement
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            ax.axhline(y=mean_diff + 1.96*std_diff, color='r', linestyle=':', alpha=0.7, label='+1.96 SD')
            ax.axhline(y=mean_diff - 1.96*std_diff, color='r', linestyle=':', alpha=0.7, label='-1.96 SD')
            
            ax.set_xlabel('Average')
            ax.set_ylabel('Difference')
            ax.set_title('Bland-Altman Plot')
            
        else:
            # Default scatter plot
            ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
            ax.set_xlabel('Reference Data')
            ax.set_ylabel('Test Data')
            ax.set_title(f'{plot_type.title()} Plot')
        
                # Add legend only if explicitly requested and there are labeled elements
        if plot_config.get('show_legend', False):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
        
    def _create_kde_plot(self, ax, x_data, y_data, bandwidth):
        """Create KDE density plot for multi-pair visualization"""
        try:
            if len(x_data) < 10:
                # Fallback to scatter for insufficient data
                print(f"[ComparisonWizard] Insufficient data for KDE ({len(x_data)} points), using scatter fallback")
                ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                return
            
            # Create KDE
            data_stack = np.vstack([x_data, y_data])
            kde = gaussian_kde(data_stack, bw_method=bandwidth)
            
            # Create grid for evaluation
            x_min, x_max = np.min(x_data), np.max(x_data)
            y_min, y_max = np.min(y_data), np.max(y_data)
            
            # Add padding to make sure all data is covered
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = 0.1 * x_range if x_range > 0 else 1.0
            y_padding = 0.1 * y_range if y_range > 0 else 1.0
            
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding
            
            # Create evaluation grid (adaptive size based on data)
            grid_size = min(100, max(30, int(np.sqrt(len(x_data)) / 2)))
            xi = np.linspace(x_min, x_max, grid_size)
            yi = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(xi, yi)
            
            # Evaluate KDE on grid
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)
            
            # Create filled contour plot
            levels = 15
            cs = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
            
            # Add colorbar
            try:
                # Remove any existing colorbars to prevent duplication
                fig = ax.get_figure()
                if hasattr(fig, '_colorbar_list'):
                    for cb in fig._colorbar_list:
                        try:
                            cb.remove()
                        except:
                            pass
                    fig._colorbar_list = []
                
                cbar = plt.colorbar(cs, ax=ax)
                cbar.set_label('Density', rotation=270, labelpad=15)
                
                # Keep track of colorbars for future cleanup
                if not hasattr(fig, '_colorbar_list'):
                    fig._colorbar_list = []
                fig._colorbar_list.append(cbar)
            except Exception as cbar_error:
                print(f"[ComparisonWizard] Could not add colorbar: {cbar_error}")
            
            # Overlay scatter points with transparency for reference
            # Downsample scatter points if too many
            n_scatter = min(1000, len(x_data))
            if n_scatter < len(x_data):
                indices = np.random.choice(len(x_data), n_scatter, replace=False)
                x_scatter = x_data[indices]
                y_scatter = y_data[indices]
            else:
                x_scatter = x_data
                y_scatter = y_data
                
            ax.scatter(x_scatter, y_scatter, alpha=0.3, s=8, c='white', 
                      edgecolors='black', linewidth=0.3)
            
            print(f"[ComparisonWizard] KDE plot created successfully with {len(x_data)} points")
            
        except Exception as e:
            print(f"[ComparisonWizard] KDE plot creation failed: {e}")
            # Ultimate fallback to simple scatter
            ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
        
    # REMOVED: _generate_pearson_plot_content - Now handled by PearsonPlotGenerator in comparison/plot_generators.py
        
    # REMOVED: _generate_bland_altman_plot_content - Now handled by BlandAltmanPlotGenerator in comparison/plot_generators.py
        
    # REMOVED: _generate_scatter_plot_content - Now handled by ScatterPlotGenerator in comparison/plot_generators.py
        
    # REMOVED: _generate_residual_plot_content - Now handled by ResidualPlotGenerator in comparison/plot_generators.py
        
    def _apply_common_plot_config(self, ax, fig, plot_config, checked_pairs):
        """Apply common plot configuration options"""
        # Apply plot configuration
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
            
        # Only show legend for scatter plots and if requested
        density_type = plot_config.get('density_display', 'scatter')
        if plot_config.get('show_legend', False) and density_type == 'scatter' and len(checked_pairs) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Apply axis ranges if specified
        x_range = plot_config.get('x_range', 'Auto')
        if x_range != 'Auto' and x_range.strip():
            try:
                x_min, x_max = map(float, x_range.split(','))
                ax.set_xlim(x_min, x_max)
            except:
                pass  # Invalid range format, use auto
                
        y_range = plot_config.get('y_range', 'Auto')
        if y_range != 'Auto' and y_range.strip():
            try:
                y_min, y_max = map(float, y_range.split(','))
                ax.set_ylim(y_min, y_max)
            except:
                pass  # Invalid range format, use auto
        
        # Apply tight layout with error handling
        try:
            fig.tight_layout()
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[ComparisonWizard] tight_layout failed: {e}, using subplots_adjust fallback")
            try:
                # Fallback to manual layout adjustment
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
            except Exception as fallback_error:
                print(f"[ComparisonWizard] Layout adjustment fallback also failed: {fallback_error}")
        
        self.window.canvas.draw()
    
    def _clear_preview_plot(self):
        """Clear the preview plot"""
        if not hasattr(self.window, 'canvas') or not self.window.canvas:
            return
            
        fig = self.window.canvas.figure
        
        # Use comprehensive clearing to prevent overlapping plots
        self._clear_figure_completely(fig)
        
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No pairs selected for preview', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Data Alignment Preview')
        try:
            fig.tight_layout()
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[ComparisonWizard] tight_layout failed: {e}, using subplots_adjust fallback")
            try:
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
            except Exception:
                pass  # If both fail, continue without layout adjustment
        self.window.canvas.draw()
    
    def _get_channel(self, filename, channel_name):
        """Get channel object by filename and channel name"""
        if not self.channel_manager or not self.file_manager:
            return None
            
        # Find file by filename
        file_info = None
        for f in self.file_manager.get_all_files():
            if f.filename == filename:
                file_info = f
                break
        
        if not file_info:
            return None
        
        # Get channels for this file using file_id
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        
        # Find matching channel by legend_label or channel_id
        for channel in channels:
            name = channel.legend_label or channel.channel_id
            if name == channel_name:
                return channel
        return None
    
    def _get_channel_by_id(self, channel_id):
        """Get channel by channel ID"""
        if not self.channel_manager:
            return None
        
        # Use the direct get_channel method if available
        if hasattr(self.channel_manager, 'get_channel'):
            return self.channel_manager.get_channel(channel_id)
        
        # Fallback: search through all channels
        if hasattr(self.channel_manager, 'get_all_channels'):
            for channel in self.channel_manager.get_all_channels():
                if hasattr(channel, 'channel_id') and channel.channel_id == channel_id:
                    return channel
        
        return None
        
    def _get_pair_config(self, pair_name):
        """Get configuration for a specific pair"""
        # This method can be simplified or removed since we now use PairManager
        for pair_config in getattr(self, 'active_pairs', []):
            if pair_config.get('name') == pair_name:
                return pair_config
        return None
        
    # Legacy _align_channels method removed - now using DataAligner directly
            
    # Legacy _align_by_index method removed - now using DataAligner directly
        
    # Legacy _align_by_time method removed - now using DataAligner directly
    
    # All legacy alignment and statistics methods removed - now using DataAligner and PairAnalyzer
    
    # Legacy corrupted code removed - all alignment and statistics now handled by DataAligner and PairAnalyzer
    
    def _update_pair_statistics(self, pair_name, statistics):
        """Update statistics in the active pairs table"""
        print(f"[ComparisonWizard] Updating table statistics for pair '{pair_name}'...")
        
        table = self.window.active_pair_table
        row = self._find_pair_row(pair_name)
        
        if row is None:
            print(f"[ComparisonWizard] WARNING: Could not find table row for pair '{pair_name}'")
            return
        
        print(f"[ComparisonWizard] Found pair '{pair_name}' at table row {row}")
        
        # Set tooltip for pair name
        self._set_pair_name_tooltip(row, pair_name)
        
        # Create and update table items
        items = self._create_statistics_table_items(statistics)
        self._set_table_items(row, items)
        
        # Set comprehensive tooltips
        self._set_detailed_tooltips(row, statistics)
        
        print(f"[ComparisonWizard] Updated row {row}: r={items['r_text']}, RMS={items['rms_text']}, N={items['n_text']}")
        
        # Refresh the updated cells
        self._refresh_table_cells(row, [2, 3, 4])
    
    def _find_pair_row(self, pair_name):
        """Find the table row for a given pair name"""
        table = self.window.active_pair_table
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            if name_item and name_item.text() == pair_name:
                return row
        return None
    
    def _create_statistics_table_items(self, statistics):
        """Create formatted table items for statistics display"""
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtCore import Qt
        
        # Format correlation
        r_val = statistics.get('r', np.nan)
        if np.isnan(r_val):
            r_text, r_color = "N/A", None
        else:
            r_text = f"{r_val:.3f}"
            r_color = self._get_correlation_color(r_val)
        
        # Format RMS
        rms_val = statistics.get('rms', np.nan)
        if np.isnan(rms_val):
            rms_text, rms_color = "N/A", None
        else:
            rms_text = f"{rms_val:.3f}"
            rms_color = QColor(70, 130, 180)  # Steel blue
        
        # Format sample size
        n_val = statistics.get('n', 0)
        n_text = f"{n_val:,}"
        n_color = QColor(105, 105, 105) if n_val > 0 else QColor(220, 20, 60)
        
        # Create table items
        items = {}
        for key, (text, color) in [('r', (r_text, r_color)), ('rms', (rms_text, rms_color)), ('n', (n_text, n_color))]:
            item = QTableWidgetItem(text)
            if color:
                item.setForeground(color)
                item.setData(Qt.FontRole, self._get_bold_font())
            items[f'{key}_item'] = item
            items[f'{key}_text'] = text
        
        return items
    
    def _get_correlation_color(self, r_val):
        """Get color for correlation value"""
        if abs(r_val) >= 0.7:
            return QColor(34, 139, 34)  # Forest green
        elif abs(r_val) >= 0.3:
            return QColor(255, 140, 0)  # Dark orange
        else:
            return QColor(220, 20, 60)  # Crimson
    
    def _set_table_items(self, row, items):
        """Set table items for statistics columns"""
        table = self.window.active_pair_table
        table.setItem(row, 2, items['r_item'])
        table.setItem(row, 3, items['rms_item'])
        table.setItem(row, 4, items['n_item'])
    
    def _refresh_table_cells(self, row, columns):
        """Refresh specific table cells"""
        table = self.window.active_pair_table
        for col in columns:
            table.update(table.model().index(row, col))
                    
    def _set_pair_name_tooltip(self, row, pair_name):
        """Set tooltip for pair name showing file and channel details"""
        try:
            table = self.window.active_pair_table
            name_item = table.item(row, 1)
            
            if not name_item:
                return
            
            # Find the pair configuration
            active_pairs = self.window.get_active_pairs()
            pair_config = None
            for pair in active_pairs:
                if pair['name'] == pair_name:
                    pair_config = pair
                    break
            
            if not pair_config:
                name_item.setToolTip(f"Pair: {pair_name}\n(Configuration not found)")
                return
            
            # Build detailed tooltip
            tooltip_lines = []
            tooltip_lines.append(f"Comparison Pair: {pair_name}")
            tooltip_lines.append("")  # Empty line for spacing
            
            # Reference channel info
            ref_file = pair_config.get('ref_file', 'Unknown')
            ref_channel = pair_config.get('ref_channel', 'Unknown')
            tooltip_lines.append(f"📊 Reference:")
            tooltip_lines.append(f"   File: {ref_file}")
            tooltip_lines.append(f"   Channel: {ref_channel}")
            
            # Test channel info
            test_file = pair_config.get('test_file', 'Unknown')
            test_channel = pair_config.get('test_channel', 'Unknown')
            tooltip_lines.append(f"📊 Test:")
            tooltip_lines.append(f"   File: {test_file}")
            tooltip_lines.append(f"   Channel: {test_channel}")
            
            # Alignment info
            alignment_mode = pair_config.get('alignment_mode', 'index')
            tooltip_lines.append("")
            tooltip_lines.append(f"⚙️ Alignment: {alignment_mode.title()}-based")
            
            # Additional alignment details if available
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                n_points = aligned_data.get('n_points', 0)
                if n_points > 0:
                    tooltip_lines.append(f"📈 Data points: {n_points:,}")
                
                if alignment_mode == 'time':
                    time_range = aligned_data.get('time_range')
                    if time_range:
                        tooltip_lines.append(f"⏱️ Time range: {time_range[0]:.3f} to {time_range[1]:.3f}s")
                elif alignment_mode == 'index':
                    index_range = aligned_data.get('index_range')
                    if index_range:
                        tooltip_lines.append(f"📍 Index range: {index_range[0]} to {index_range[1]}")
            
            # Join all lines and set tooltip
            tooltip_text = "\n".join(tooltip_lines)
            name_item.setToolTip(tooltip_text)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting pair name tooltip: {str(e)}")
            # Fallback simple tooltip
            if name_item:
                name_item.setToolTip(f"Pair: {pair_name}")
    
    def _try_create_time_data(self, ref_channel, test_channel):
        """Try to create time data for channels that don't have it"""
        try:
            channels_updated = []
            
            for channel, name in [(ref_channel, 'reference'), (test_channel, 'test')]:
                needs_time_data = False
                
                # Check if channel has no xdata at all
                if not hasattr(channel, 'xdata') or channel.xdata is None:
                    needs_time_data = True
                    print(f"[ComparisonWizard] No xdata found for {name} channel")
                
                # Check if xdata exists but contains unconvertible data
                elif hasattr(channel, 'xdata') and channel.xdata is not None:
                    try:
                        xdata_array = np.asarray(channel.xdata)
                        if not np.issubdtype(xdata_array.dtype, np.number):
                            # Check if this is datetime data that we should convert
                            if xdata_array.dtype == object or xdata_array.dtype.kind in ['U', 'S']:
                                # Check if it looks like datetime strings
                                sample_values = xdata_array[:min(5, len(xdata_array))]
                                datetime_like = any(isinstance(val, str) and any(char in str(val) for char in ['-', ':', ' ']) for val in sample_values)
                                
                                if datetime_like:
                                    print(f"[ComparisonWizard] Found datetime-like strings in {name} channel xdata, will handle in validation")
                                    # Don't mark as needing new time data - let validation handle the conversion
                                    continue
                            
                            # If not datetime-like, we need to create new time data
                            needs_time_data = True
                            print(f"[ComparisonWizard] Non-numeric, non-datetime xdata found for {name} channel")
                    except Exception as check_error:
                        needs_time_data = True
                        print(f"[ComparisonWizard] Error checking xdata for {name} channel: {check_error}")
                
                if needs_time_data:
                    print(f"[ComparisonWizard] Creating time data for {name} channel...")
                    # Create time data based on sampling rate or indices
                    
                    # Check if channel has sampling rate information
                    if hasattr(channel, 'sampling_rate') and channel.sampling_rate is not None and channel.sampling_rate > 0:
                        # Create time axis based on sampling rate
                        n_samples = len(channel.ydata)
                        dt = 1.0 / channel.sampling_rate
                        time_axis = np.arange(n_samples) * dt
                        channel.xdata = time_axis
                        print(f"  - Created time axis using sampling rate {channel.sampling_rate} Hz")
                        channels_updated.append(name)
                    
                    elif hasattr(channel, 'sample_period') and channel.sample_period is not None and channel.sample_period > 0:
                        # Create time axis based on sample period
                        n_samples = len(channel.ydata)
                        time_axis = np.arange(n_samples) * channel.sample_period
                        channel.xdata = time_axis
                        print(f"  - Created time axis using sample period {channel.sample_period} s")
                        channels_updated.append(name)
                    
                    else:
                        # Fallback: create time axis assuming 1 Hz sampling
                        n_samples = len(channel.ydata)
                        time_axis = np.arange(n_samples, dtype=float)
                        channel.xdata = time_axis
                        print(f"  - Created time axis using indices (assuming 1 Hz sampling)")
                        channels_updated.append(name)
                        
                        # Show warning to user
                        if hasattr(self, 'window'):
                            QMessageBox.information(
                                self.window,
                                "Time Data Created",
                                f"No usable time data found for {name} channel.\n"
                                f"Created time axis using sample indices (assuming 1 Hz).\n"
                                f"For better results, ensure your data includes proper time information."
                            )
            
            return len(channels_updated) > 0
            
        except Exception as e:
            print(f"[ComparisonWizard] Failed to create time data: {str(e)}")
            return False
    
    def _validate_time_data(self, ref_channel, test_channel):
        """Validate time data before alignment"""
        try:
            # Check if channels exist and have data
            for channel, name in [(ref_channel, 'reference'), (test_channel, 'test')]:
                # Check for basic attributes
                if not hasattr(channel, 'xdata') or not hasattr(channel, 'ydata'):
                    return {'valid': False, 'error': f'{name} channel missing time or data arrays'}
                
                # Check for None data
                if channel.xdata is None or channel.ydata is None:
                    return {'valid': False, 'error': f'{name} channel has None data'}
                
                # Check for empty data
                if len(channel.xdata) == 0 or len(channel.ydata) == 0:
                    return {'valid': False, 'error': f'{name} channel has empty data'}
                
                # Check for length mismatch
                if len(channel.xdata) != len(channel.ydata):
                    return {'valid': False, 'error': f'{name} channel time and data arrays have different lengths'}
                
                # Detailed inspection of time data
                xdata = channel.xdata
                print(f"[ComparisonWizard] {name} channel xdata info:")
                print(f"  - Type: {type(xdata)}")
                print(f"  - Shape: {getattr(xdata, 'shape', 'N/A')}")
                print(f"  - Dtype: {getattr(xdata, 'dtype', 'N/A')}")
                print(f"  - First 5 values: {xdata[:5] if len(xdata) > 0 else 'Empty'}")
                
                # Try to convert to numpy array if it's not already
                try:
                    xdata_array = np.asarray(xdata)
                    print(f"  - Converted dtype: {xdata_array.dtype}")
                    print(f"  - Is numeric: {np.issubdtype(xdata_array.dtype, np.number)}")
                except Exception as conv_error:
                    return {'valid': False, 'error': f'{name} channel time data cannot be converted to array: {conv_error}'}
                
                # Check if the data can be treated as numeric
                if not np.issubdtype(xdata_array.dtype, np.number):
                    # Try to convert string/object data to numeric
                    if xdata_array.dtype == object or xdata_array.dtype.kind in ['U', 'S']:
                        try:
                            # First try direct numeric conversion
                            xdata_numeric = pd.to_numeric(xdata_array, errors='coerce')
                            if not np.all(np.isnan(xdata_numeric)):
                                print(f"  - Successfully converted string/object data to numeric")
                                channel.xdata = xdata_numeric
                            else:
                                # Try datetime conversion if numeric conversion failed
                                print(f"  - Numeric conversion failed, trying datetime conversion...")
                                try:
                                    # Convert to pandas datetime then to timestamps
                                    datetime_series = pd.to_datetime(xdata_array, errors='coerce')
                                    if not datetime_series.isna().all():
                                        # Convert to seconds since the first timestamp (relative time)
                                        valid_datetimes = datetime_series.dropna()
                                        if len(valid_datetimes) == 0:
                                            raise ValueError("No valid datetime values found")
                                        
                                        # Get first valid datetime - handle both Series and Index
                                        if hasattr(valid_datetimes, 'iloc'):
                                            first_time = valid_datetimes.iloc[0]
                                        else:
                                            first_time = valid_datetimes[0]
                                        
                                        # Convert to seconds since the first timestamp (relative time)
                                        relative_seconds = (datetime_series - first_time).total_seconds()
                                        
                                        # Handle any NaT values by converting to NaN
                                        if hasattr(relative_seconds, 'values'):
                                            xdata_numeric = relative_seconds.values
                                        else:
                                            xdata_numeric = np.array(relative_seconds)
                                        
                                        # Replace any inf or -inf with NaN
                                        xdata_numeric = np.where(np.isfinite(xdata_numeric), xdata_numeric, np.nan)
                                        
                                        print(f"    - Successfully converted datetime strings to relative seconds")
                                        print(f"    - Time range: {np.nanmin(xdata_numeric):.3f} to {np.nanmax(xdata_numeric):.3f} seconds")
                                        print(f"    - Reference time: {first_time}")
                                        
                                        # Update the channel's xdata with converted values
                                        channel.xdata = xdata_numeric
                                        
                                        # Show info to user about datetime conversion
                                        if hasattr(self, 'window'):
                                            QMessageBox.information(
                                                self.window,
                                                "DateTime Conversion",
                                                f"Converted datetime strings to relative time for {name} channel.\n\n"
                                                f"Reference time: {first_time}\n"
                                                f"Time range: {np.nanmin(xdata_numeric):.1f} to {np.nanmax(xdata_numeric):.1f} seconds"
                                            )
                                    else:
                                        raise ValueError("All datetime values are invalid")
                                        
                                except Exception as dt_error:
                                    print(f"    - Datetime conversion also failed: {dt_error}")
                                    print(f"    - Attempting fallback: create time data from indices...")
                                    
                                    # Fallback: create time axis from indices
                                    try:
                                        n_samples = len(xdata_array)
                                        time_axis = np.arange(n_samples, dtype=float)
                                        channel.xdata = time_axis
                                        print(f"    - Created fallback time axis using indices (0 to {n_samples-1})")
                                        
                                        if hasattr(self, 'window'):
                                            QMessageBox.warning(
                                                self.window,
                                                "DateTime Conversion Failed",
                                                f"Could not convert datetime strings for {name} channel.\n\n"
                                                f"Error: {dt_error}\n\n"
                                                f"Created time axis using sample indices as fallback.\n"
                                                f"Time alignment may not be accurate."
                                            )
                                    except Exception as fallback_error:
                                        return {'valid': False, 'error': f'{name} channel time data conversion failed completely: {fallback_error}'}
                        except Exception as convert_error:
                            return {'valid': False, 'error': f'{name} channel time data is not numeric and cannot be converted: {convert_error}'}
                    else:
                        return {'valid': False, 'error': f'{name} channel time data is not numeric (dtype: {xdata_array.dtype})'}
                
                # Check for valid numeric data
                xdata_final = np.asarray(channel.xdata)
                ydata_final = np.asarray(channel.ydata)
                
                time_valid = np.isfinite(xdata_final)
                data_valid = np.isfinite(ydata_final)
                
                if np.sum(time_valid) < 2:
                    return {'valid': False, 'error': f'{name} channel has insufficient valid time points ({np.sum(time_valid)} valid)'}
                
                if np.sum(data_valid) < 2:
                    return {'valid': False, 'error': f'{name} channel has insufficient valid data points ({np.sum(data_valid)} valid)'}
                
                # Check time range
                valid_times = xdata_final[time_valid]
                time_range = valid_times.max() - valid_times.min()
                if time_range <= 0:
                    return {'valid': False, 'error': f'{name} channel has zero or negative time range ({time_range})'}
                
                print(f"  - Valid time points: {np.sum(time_valid)}/{len(time_valid)}")
                print(f"  - Time range: {valid_times.min():.3f} to {valid_times.max():.3f} ({time_range:.3f})")
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def _clean_and_sort_time_data(self, x_data, y_data):
        """Clean and sort time data, removing invalid values and duplicates"""
        # Remove NaN and infinite values
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) < 2:
            raise ValueError("Insufficient valid data points after cleaning")
        
        # Sort by time
        sort_indices = np.argsort(x_clean)
        x_sorted = x_clean[sort_indices]
        y_sorted = y_clean[sort_indices]
        
        # Handle duplicate time values by averaging
        if len(np.unique(x_sorted)) < len(x_sorted):
            # Find unique times and average corresponding y values
            unique_times, inverse_indices = np.unique(x_sorted, return_inverse=True)
            averaged_y = np.zeros_like(unique_times)
            
            for i in range(len(unique_times)):
                mask = inverse_indices == i
                averaged_y[i] = np.mean(y_sorted[mask])
            
            x_sorted = unique_times
            y_sorted = averaged_y
        
        return x_sorted, y_sorted
    
    def _interpolate_channel(self, x_data, y_data, time_grid, method, channel_name):
        """Robustly interpolate channel data to time grid"""
        try:
            if method == 'linear':
                # Use numpy's linear interpolation (fastest)
                return np.interp(time_grid, x_data, y_data)
            
            elif method == 'nearest':
                # Nearest neighbor interpolation
                f = interp1d(x_data, y_data, kind='nearest', 
                           bounds_error=False, fill_value=np.nan)
                return f(time_grid)
            
            elif method == 'cubic':
                # Cubic spline interpolation
                if len(x_data) < 4:
                    # Fall back to linear for insufficient points
                    warnings.warn(f"Not enough points for cubic interpolation in {channel_name} channel, using linear")
                    return np.interp(time_grid, x_data, y_data)
                
                f = interp1d(x_data, y_data, kind='cubic', 
                           bounds_error=False, fill_value=np.nan)
                return f(time_grid)
            
            else:
                # Default to linear for unknown methods
                warnings.warn(f"Unknown interpolation method '{method}', using linear")
                return np.interp(time_grid, x_data, y_data)
                
        except Exception as e:
            raise ValueError(f"Interpolation failed for {channel_name} channel: {str(e)}")
        
    # Legacy _calculate_statistics method removed - now using PairAnalyzer for all statistics
    
    # Legacy _calculate_simple_statistics method removed - now using PairAnalyzer
    
    # Legacy _validate_aligned_data method removed - validation now done in PairAnalyzer
    
    # Legacy _calculate_correlation method removed - now done in comparison methods
    
    # Legacy _calculate_difference_stats method removed - now done in comparison methods
        
    def _update_pair_statistics(self, pair_name, statistics):
        """Update statistics in the active pairs table"""
        print(f"[ComparisonWizard] Updating table statistics for pair '{pair_name}'...")
        
        table = self.window.active_pair_table
        row = self._find_pair_row(pair_name)
        
        if row is None:
            print(f"[ComparisonWizard] WARNING: Could not find table row for pair '{pair_name}'")
            return
        
        print(f"[ComparisonWizard] Found pair '{pair_name}' at table row {row}")
        
        # Set tooltip for pair name
        self._set_pair_name_tooltip(row, pair_name)
        
        # Create and update table items
        items = self._create_statistics_table_items(statistics)
        self._set_table_items(row, items)
        
        # Set comprehensive tooltips
        self._set_detailed_tooltips(row, statistics)
        
        print(f"[ComparisonWizard] Updated row {row}: r={items['r_text']}, RMS={items['rms_text']}, N={items['n_text']}")
        
        # Refresh the updated cells
        self._refresh_table_cells(row, [2, 3, 4])
    
    def _find_pair_row(self, pair_name):
        """Find the table row for a given pair name"""
        table = self.window.active_pair_table
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            if name_item and name_item.text() == pair_name:
                return row
        return None
    
    def _create_statistics_table_items(self, statistics):
        """Create formatted table items for statistics display"""
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtCore import Qt
        
        # Format correlation
        r_val = statistics.get('r', np.nan)
        if np.isnan(r_val):
            r_text, r_color = "N/A", None
        else:
            r_text = f"{r_val:.3f}"
            r_color = self._get_correlation_color(r_val)
        
        # Format RMS
        rms_val = statistics.get('rms', np.nan)
        if np.isnan(rms_val):
            rms_text, rms_color = "N/A", None
        else:
            rms_text = f"{rms_val:.3f}"
            rms_color = QColor(70, 130, 180)  # Steel blue
        
        # Format sample size
        n_val = statistics.get('n', 0)
        n_text = f"{n_val:,}"
        n_color = QColor(105, 105, 105) if n_val > 0 else QColor(220, 20, 60)
        
        # Create table items
        items = {}
        for key, (text, color) in [('r', (r_text, r_color)), ('rms', (rms_text, rms_color)), ('n', (n_text, n_color))]:
            item = QTableWidgetItem(text)
            if color:
                item.setForeground(color)
                item.setData(Qt.FontRole, self._get_bold_font())
            items[f'{key}_item'] = item
            items[f'{key}_text'] = text
        
        return items
    
    def _get_correlation_color(self, r_val):
        """Get color for correlation value"""
        if abs(r_val) >= 0.7:
            return QColor(34, 139, 34)  # Forest green
        elif abs(r_val) >= 0.3:
            return QColor(255, 140, 0)  # Dark orange
        else:
            return QColor(220, 20, 60)  # Crimson
    
    def _set_table_items(self, row, items):
        """Set table items for statistics columns"""
        table = self.window.active_pair_table
        table.setItem(row, 2, items['r_item'])
        table.setItem(row, 3, items['rms_item'])
        table.setItem(row, 4, items['n_item'])
    
    def _refresh_table_cells(self, row, columns):
        """Refresh specific table cells"""
        table = self.window.active_pair_table
        for col in columns:
            table.update(table.model().index(row, col))
                    
    def _set_pair_name_tooltip(self, row, pair_name):
        """Set tooltip for pair name showing file and channel details"""
        try:
            table = self.window.active_pair_table
            name_item = table.item(row, 1)
            
            if not name_item:
                return
            
            # Find the pair configuration
            active_pairs = self.window.get_active_pairs()
            pair_config = None
            for pair in active_pairs:
                if pair['name'] == pair_name:
                    pair_config = pair
                    break
            
            if not pair_config:
                name_item.setToolTip(f"Pair: {pair_name}\n(Configuration not found)")
                return
            
            # Build detailed tooltip
            tooltip_lines = []
            tooltip_lines.append(f"Comparison Pair: {pair_name}")
            tooltip_lines.append("")  # Empty line for spacing
            
            # Reference channel info
            ref_file = pair_config.get('ref_file', 'Unknown')
            ref_channel = pair_config.get('ref_channel', 'Unknown')
            tooltip_lines.append(f"📊 Reference:")
            tooltip_lines.append(f"   File: {ref_file}")
            tooltip_lines.append(f"   Channel: {ref_channel}")
            
            # Test channel info
            test_file = pair_config.get('test_file', 'Unknown')
            test_channel = pair_config.get('test_channel', 'Unknown')
            tooltip_lines.append(f"📊 Test:")
            tooltip_lines.append(f"   File: {test_file}")
            tooltip_lines.append(f"   Channel: {test_channel}")
            
            # Alignment info
            alignment_mode = pair_config.get('alignment_mode', 'index')
            tooltip_lines.append("")
            tooltip_lines.append(f"⚙️ Alignment: {alignment_mode.title()}-based")
            
            # Additional alignment details if available
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                n_points = aligned_data.get('n_points', 0)
                if n_points > 0:
                    tooltip_lines.append(f"📈 Data points: {n_points:,}")
                
                if alignment_mode == 'time':
                    time_range = aligned_data.get('time_range')
                    if time_range:
                        tooltip_lines.append(f"⏱️ Time range: {time_range[0]:.3f} to {time_range[1]:.3f}s")
                elif alignment_mode == 'index':
                    index_range = aligned_data.get('index_range')
                    if index_range:
                        tooltip_lines.append(f"📍 Index range: {index_range[0]} to {index_range[1]}")
            
            # Join all lines and set tooltip
            tooltip_text = "\n".join(tooltip_lines)
            name_item.setToolTip(tooltip_text)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting pair name tooltip: {str(e)}")
            # Fallback simple tooltip
            if name_item:
                name_item.setToolTip(f"Pair: {pair_name}")
    
    def _get_bold_font(self):
        """Get a bold font for table items"""
        try:
            font = QFont()
            font.setBold(True)
            return font
        except:
            return None
    
    def _set_detailed_tooltips(self, row, statistics):
        """Set detailed tooltips for table items"""
        try:
            table = self.window.active_pair_table
            
            # Build tooltip parts using helper methods
            tooltip_parts = []
            tooltip_parts.extend(self._get_basic_statistics_tooltip(statistics))
            tooltip_parts.extend(self._get_data_quality_tooltip(statistics))
            tooltip_parts.extend(self._get_additional_statistics_tooltip(statistics))
            tooltip_parts.extend(self._get_reference_test_tooltip(statistics))
            tooltip_parts.extend(self._get_error_tooltip(statistics))
            
            # Create tooltip text
            tooltip_text = "\n".join(tooltip_parts)
            
            # Apply tooltip to all statistics columns
            for col in [2, 3, 4]:  # r, RMS, N columns
                item = table.item(row, col)
                if item:
                    item.setToolTip(tooltip_text)
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error setting tooltips: {str(e)}")
    
    def _get_basic_statistics_tooltip(self, statistics):
        """Get basic statistics for tooltip"""
        parts = []
        
        r_val = statistics.get('r', np.nan)
        rms_val = statistics.get('rms', np.nan)
        n_val = statistics.get('n', 0)
        
        if not np.isnan(r_val):
            parts.append(f"Correlation: {r_val:.4f}")
            if 'r_pvalue' in statistics:
                parts.append(f"p-value: {statistics['r_pvalue']:.2e}")
        
        if not np.isnan(rms_val):
            parts.append(f"RMS difference: {rms_val:.4f}")
        
        parts.append(f"Sample size: {n_val:,}")
        
        return parts
    
    def _get_data_quality_tooltip(self, statistics):
        """Get data quality information for tooltip"""
        parts = []
        
        if 'valid_ratio' in statistics:
            parts.append(f"Valid data: {statistics['valid_ratio']*100:.1f}%")
        
        return parts
    
    def _get_additional_statistics_tooltip(self, statistics):
        """Get additional statistics for tooltip"""
        parts = []
        
        additional_stats = ['mean_diff', 'std_diff', 'median_diff', 'max_abs_diff']
        stat_labels = ['Mean difference', 'Std difference', 'Median difference', 'Max |difference|']
        
        for stat, label in zip(additional_stats, stat_labels):
            if stat in statistics:
                parts.append(f"{label}: {statistics[stat]:.4f}")
        
        return parts
    
    def _get_reference_test_tooltip(self, statistics):
        """Get reference and test data info for tooltip"""
        parts = []
        
        if 'ref_mean' in statistics and 'ref_std' in statistics:
            parts.append(f"Reference: μ={statistics['ref_mean']:.3f}, σ={statistics['ref_std']:.3f}")
        if 'test_mean' in statistics and 'test_std' in statistics:
            parts.append(f"Test: μ={statistics['test_mean']:.3f}, σ={statistics['test_std']:.3f}")
        
        return parts
    
    def _get_error_tooltip(self, statistics):
        """Get error information for tooltip"""
        parts = []
        
        if 'error' in statistics:
            parts.append(f"⚠️ Error: {statistics['error']}")
        
        return parts
        
    def _show_alignment_summary(self, pair_name, aligned_data, stats):
        """Show alignment summary and data quality information"""
        try:
            summary_parts = []
            warning_parts = []
            
            # Alignment method info
            method = aligned_data.get('alignment_method', 'unknown')
            n_points = aligned_data.get('n_points', 0)
            summary_parts.append(f"Alignment: {method} method, {n_points} data points")
            
            # Data quality info
            valid_ratio = aligned_data.get('valid_ratio', stats.get('valid_ratio', 1.0))
            if valid_ratio < 1.0:
                summary_parts.append(f"Data quality: {valid_ratio*100:.1f}% valid points")
                if valid_ratio < 0.8:
                    warning_parts.append(f"Low data quality ({valid_ratio*100:.1f}% valid)")
            
            # Time range info for time alignment
            if method == 'time':
                time_range = aligned_data.get('time_range')
                if time_range:
                    summary_parts.append(f"Time range: {time_range[0]:.3f} to {time_range[1]:.3f}")
                
                # Show round_to adjustments if any
                round_to_used = aligned_data.get('round_to_used')
                round_to_original = aligned_data.get('round_to_original')
                if round_to_used and round_to_original and round_to_used != round_to_original:
                    summary_parts.append(f"Time resolution adjusted: {round_to_original:.4f}s → {round_to_used:.4f}s")
                elif round_to_used:
                    summary_parts.append(f"Time resolution: {round_to_used:.4f}s")
            
            # Index range info for index alignment
            elif method == 'index':
                index_range = aligned_data.get('index_range')
                if index_range:
                    summary_parts.append(f"Index range: {index_range[0]} to {index_range[1]}")
                
                offset = aligned_data.get('offset_applied', 0)
                if offset != 0:
                    summary_parts.append(f"Offset applied: {offset} samples")
            
            # Statistical warnings
            if 'error' in stats:
                warning_parts.append(f"Statistics error: {stats['error']}")
            else:
                r_val = stats.get('r', np.nan)
                if np.isnan(r_val) and 'r_error' in stats:
                    warning_parts.append(f"Correlation: {stats['r_error']}")
                elif not np.isnan(r_val) and abs(r_val) < 0.1:
                    warning_parts.append(f"Low correlation (r = {r_val:.3f})")
                
                rms_val = stats.get('rms', np.nan)
                if not np.isnan(rms_val):
                    # Check if RMS is very large compared to data range
                    ref_std = stats.get('ref_std', np.nan)
                    test_std = stats.get('test_std', np.nan)
                    if not np.isnan(ref_std) and not np.isnan(test_std):
                        typical_scale = (ref_std + test_std) / 2
                        if typical_scale > 0 and rms_val > 2 * typical_scale:
                            warning_parts.append(f"Large RMS error ({rms_val:.3f}) relative to data scale")
            
            # Display summary in console or status
            summary_text = f"[{pair_name}] " + "; ".join(summary_parts)
            print(f"[ComparisonWizard] {summary_text}")
            
            # Show warnings if any
            if warning_parts and hasattr(self, 'window'):
                warning_text = f"Pair '{pair_name}' alignment completed with warnings:\n\n" + "\n".join(f"• {w}" for w in warning_parts)
                warning_text += f"\n\nSummary: {'; '.join(summary_parts)}"
                
                QMessageBox.information(self.window, "Alignment Summary", warning_text)
                
        except Exception as e:
            print(f"[ComparisonWizard] Error showing alignment summary: {str(e)}")
                

    def _update_cumulative_display(self):
        """Update cumulative statistics and preview plot"""
        # Get checked pairs
        checked_pairs = self.window.get_checked_pairs()
        
        if not checked_pairs:
            # No pairs checked
            self.window.update_cumulative_stats("Cumulative Stats: No pairs selected")
            self._clear_preview_plot()
            return
        
        # Calculate cumulative statistics
        cumulative_stats = self._calculate_cumulative_statistics(checked_pairs)
        
        # Update cumulative stats display
        stats_text = self._format_cumulative_stats(cumulative_stats, len(checked_pairs))
        self.window.update_cumulative_stats(stats_text)
        
        # Generate plot based on comparison method (not just cumulative preview)
        plot_type = self._determine_plot_type_from_pairs(checked_pairs)
        
        # Build comprehensive plot configuration including method parameters
        plot_config = {
            'plot_type': plot_type,
            'show_grid': True,
            'show_legend': False,  # Remove legend by default
            'checked_pairs': checked_pairs
        }
        
        # Add method-specific parameters from the first pair for plot configuration
        if checked_pairs:
            first_pair = checked_pairs[0]
            method_params = first_pair.get('method_parameters', {})
            
            # Map method parameters to plot config parameters
            if plot_type == 'bland_altman':
                plot_config['confidence_interval'] = method_params.get('show_ci', True)
                plot_config['agreement_limits'] = method_params.get('agreement_limits', 1.96)
                plot_config['proportional_bias'] = method_params.get('proportional_bias', False)
            elif plot_type == 'scatter' or plot_type == 'pearson':
                plot_config['confidence_level'] = method_params.get('confidence_level', 0.95)
                plot_config['correlation_type'] = method_params.get('correlation_type', 'pearson')
            elif plot_type == 'residual':
                plot_config['normality_test'] = method_params.get('normality_test', 'shapiro')
                plot_config['outlier_detection'] = method_params.get('outlier_detection', 'iqr')
        
        # Use the same plot generation as the Generate Plot button
        self._generate_multi_pair_plot(checked_pairs, plot_config)

    # REMOVED: _update_comprehensive_display - Not used in current implementation
        
    def _calculate_comprehensive_statistics(self, checked_pairs, plot_type='scatter'):
        """Calculate comprehensive statistics for step 2 display using statistics calculators"""
        if not checked_pairs:
            return {'error': 'No pairs selected'}
        
        # Use simple calculation
        return {'error': 'Comprehensive statistics not available'}
        
    def _prepare_pairs_for_statistics_calculator(self, checked_pairs):
        """Prepare pairs data for statistics calculator"""
        enhanced_pairs = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            enhanced_pair = pair.copy()
            
            # Add aligned data and statistics if available
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                enhanced_pair['statistics'] = {
                    'aligned_data': aligned_data
                }
                
                # Add individual pair statistics if available
                if pair_name in self.pair_statistics:
                    enhanced_pair['statistics']['calculated_stats'] = self.pair_statistics[pair_name]
            
            enhanced_pairs.append(enhanced_pair)
        
        return enhanced_pairs
    
    def _format_comprehensive_stats(self, stats, n_pairs, plot_type='scatter'):
        """Format comprehensive statistics for display in step 2"""
        if 'error' in stats:
            return f"Error: {stats['error']}"
        
        lines = []
        
        # Header with plot type
        plot_type_names = {
            'scatter': 'SCATTER PLOT',
            'bland_altman': 'BLAND-ALTMAN',
            'residual': 'RESIDUAL',
            'pearson': 'PEARSON CORRELATION'
        }
        plot_name = plot_type_names.get(plot_type, plot_type.upper())
        
        lines.append(f"{plot_name} ANALYSIS - {n_pairs} pairs, {stats['n_total']:,} total points")
        lines.append("=" * 70)
        
        # Pearson Correlation with enhanced information
        lines.append("CORRELATION ANALYSIS:")
        if 'error' in stats.get('pearson', {}):
            lines.append(f"  Error: {stats['pearson']['error']}")
        else:
            p = stats['pearson']
            sig_text = " ✓ significant" if p.get('significant', False) else " ✗ not significant"
            strength = p.get('strength', 'unknown')
            direction = p.get('direction', 'unknown')
            
            lines.append(f"  Pearson r = {p['r']:7.4f} ({strength} {direction} correlation)")
            lines.append(f"  R²        = {p['r2']:7.4f} ({p['r2']*100:5.1f}% of variance explained)")
            lines.append(f"  p-value   = {p['p_value']:.2e}{sig_text}")
            
            # Add correlation interpretation
            if not np.isnan(p['r']):
                if abs(p['r']) >= 0.7:
                    lines.append(f"  💡 Strong correlation indicates good agreement between datasets")
                elif abs(p['r']) >= 0.3:
                    lines.append(f"  ⚠️  Moderate correlation - consider examining data relationships")
                else:
                    lines.append(f"  ⚠️  Weak correlation - datasets may have different patterns")
        
        # Plot-type specific correlation analysis
        if plot_type == 'pearson' and 'correlation_analysis' in stats:
            if 'error' not in stats['correlation_analysis']:
                ca = stats['correlation_analysis']
                lines.append(f"  Spearman ρ = {ca['spearman_r']:7.4f} (rank correlation)")
                lines.append(f"  Kendall τ  = {ca['kendall_tau']:7.4f} (concordance)")
                linearity = "✓ linear" if ca.get('linearity_check', False) else "✗ potentially non-linear"
                lines.append(f"  Linearity: {linearity} relationship")
        
        # Enhanced Descriptive Statistics
        lines.append("\nDESCRIPTIVE STATISTICS:")
        if 'error' in stats.get('descriptive', {}):
            lines.append(f"  Error: {stats['descriptive']['error']}")
        else:
            d = stats['descriptive']
            
            # Reference data statistics
            lines.append(f"  Reference Dataset:")
            lines.append(f"    Mean ± SD:     {d['ref_mean']:8.3f} ± {d['ref_std']:8.3f}")
            lines.append(f"    Median (IQR):  {d['ref_quartiles'][1]:8.3f} ({d['ref_iqr']:8.3f})")
            lines.append(f"    Range:         [{d['ref_range'][0]:8.3f}, {d['ref_range'][1]:8.3f}]")
            if not np.isnan(d['ref_cv']):
                lines.append(f"    CV:            {d['ref_cv']:8.1f}% (coefficient of variation)")
            
            # Test data statistics  
            lines.append(f"  Test Dataset:")
            lines.append(f"    Mean ± SD:     {d['test_mean']:8.3f} ± {d['test_std']:8.3f}")
            lines.append(f"    Median (IQR):  {d['test_quartiles'][1]:8.3f} ({d['test_iqr']:8.3f})")
            lines.append(f"    Range:         [{d['test_range'][0]:8.3f}, {d['test_range'][1]:8.3f}]")
            if not np.isnan(d['test_cv']):
                lines.append(f"    CV:            {d['test_cv']:8.1f}% (coefficient of variation)")
            
            # Difference statistics
            lines.append(f"  Differences (Test - Reference):")
            lines.append(f"    Mean ± SD:     {d['diff_mean']:8.3f} ± {d['diff_std']:8.3f}")
            lines.append(f"    Median (IQR):  {d['diff_quartiles'][1]:8.3f} ({d['diff_iqr']:8.3f})")
            lines.append(f"    Range:         [{d['diff_range'][0]:8.3f}, {d['diff_range'][1]:8.3f}]")
            
            if not np.isnan(d.get('percent_diff_mean', np.nan)):
                lines.append(f"    % Difference:  {d['percent_diff_mean']:8.1f}% ± {d['percent_diff_std']:8.1f}%")
            
            # Data comparison insights
            if not np.isnan(d.get('data_spread_ratio', np.nan)):
                spread_ratio = d['data_spread_ratio']
                if spread_ratio > 1.2:
                    lines.append(f"  💡 Test data has {spread_ratio:.1f}x more variability than reference")
                elif spread_ratio < 0.8:
                    lines.append(f"  💡 Test data has {1/spread_ratio:.1f}x less variability than reference")
                else:
                    lines.append(f"  💡 Similar variability between datasets (ratio: {spread_ratio:.2f})")
        
        # Bland-Altman Analysis (enhanced for bland_altman plot type)
        if plot_type == 'bland_altman' or 'bland_altman' in stats:
            lines.append("\nBLAND-ALTMAN ANALYSIS:")
            if 'error' in stats.get('bland_altman', {}):
                lines.append(f"  Error: {stats['bland_altman']['error']}")
            else:
                ba = stats['bland_altman']
                lines.append(f"  Mean Bias:        {ba['mean_bias']:8.3f} ± {ba['std_bias']:8.3f}")
                
                if not np.isnan(ba.get('bias_relative_to_mean', np.nan)):
                    lines.append(f"  Relative Bias:    {ba['bias_relative_to_mean']:8.1f}% of mean")
                
                lines.append(f"  95% Limits of Agreement:")
                lines.append(f"    Upper LoA:      {ba['loa_upper']:8.3f}")
                lines.append(f"    Lower LoA:      {ba['loa_lower']:8.3f}")
                lines.append(f"    LoA Width:      {ba['loa_width']:8.3f}")
                
                if not np.isnan(ba.get('relative_loa_width', np.nan)):
                    lines.append(f"    Relative Width: {ba['relative_loa_width']:8.1f}% of mean")
                
                lines.append(f"  Within LoA:       {ba['percent_within_loa']:5.1f}% of points")
                
                # Proportional bias assessment
                if ba.get('has_proportional_bias', False):
                    lines.append(f"  ⚠️  Proportional bias detected (r={ba['proportional_bias_r']:.3f}, p={ba['proportional_bias_p']:.3f})")
                    lines.append(f"      Bias increases/decreases with measurement magnitude")
                else:
                    lines.append(f"  ✓ No significant proportional bias detected")
                
                # Clinical interpretation
                if ba['percent_within_loa'] >= 95:
                    lines.append(f"  💡 Excellent agreement - {ba['percent_within_loa']:.1f}% within expected range")
                elif ba['percent_within_loa'] >= 90:
                    lines.append(f"  💡 Good agreement - most points within limits")
                else:
                    lines.append(f"  ⚠️  Poor agreement - many outliers beyond limits")
        
        # Enhanced Error Metrics
        lines.append("\nERROR METRICS:")
        if 'error' in stats.get('error_metrics', {}):
            lines.append(f"  Error: {stats['error_metrics']['error']}")
        else:
            em = stats['error_metrics']
            
            # Basic error metrics
            lines.append(f"  Absolute Errors:")
            lines.append(f"    MAE (Mean):           {em['mae']:8.3f}")
            lines.append(f"    Median AE:            {em['median_abs_error']:8.3f}")
            lines.append(f"    Max AE:               {em['max_abs_error']:8.3f}")
            lines.append(f"    95th percentile AE:   {em['q95_abs_error']:8.3f}")
            lines.append(f"    99th percentile AE:   {em['q99_abs_error']:8.3f}")
            
            # Squared error metrics
            lines.append(f"  Squared Errors:")
            lines.append(f"    MSE (Mean Squared):   {em['mse']:8.3f}")
            lines.append(f"    RMSE (Root MSE):      {em['rmse']:8.3f}")
            
            # Normalized metrics
            if not np.isnan(em.get('nrmse_range', np.nan)):
                lines.append(f"  Normalized RMSE:")
                lines.append(f"    By range:             {em['nrmse_range']:8.3f} ({em['nrmse_range']*100:5.1f}%)")
            if not np.isnan(em.get('nrmse_mean', np.nan)):
                lines.append(f"    By mean:              {em['nrmse_mean']:8.3f} ({em['nrmse_mean']*100:5.1f}%)")
            
            # Percentage-based metrics
            if not np.isnan(em.get('mape', np.nan)):
                lines.append(f"  Percentage Errors:")
                lines.append(f"    MAPE (Mean Abs %):    {em['mape']:8.1f}%")
                lines.append(f"    SMAPE (Symmetric):    {em['smape']:8.1f}%")
            
            # Concordance correlation coefficient
            if not np.isnan(em.get('ccc', np.nan)):
                ccc = em['ccc']
                lines.append(f"  Concordance Corr Coef: {ccc:8.4f}")
                if ccc > 0.99:
                    lines.append(f"    💡 Almost perfect agreement")
                elif ccc > 0.95:
                    lines.append(f"    💡 Substantial agreement") 
                elif ccc > 0.90:
                    lines.append(f"    💡 Moderate agreement")
                else:
                    lines.append(f"    ⚠️  Poor agreement")
        
        # Residual Analysis (for residual plot type)
        if plot_type == 'residual' and 'residual_analysis' in stats:
            lines.append("\nRESIDUAL ANALYSIS:")
            if 'error' in stats['residual_analysis']:
                lines.append(f"  Error: {stats['residual_analysis']['error']}")
            else:
                ra = stats['residual_analysis']
                
                # Heteroscedasticity test
                if ra.get('has_heteroscedasticity', False):
                    lines.append(f"  ⚠️  Heteroscedasticity detected (r={ra['heteroscedasticity_r']:.3f})")
                    lines.append(f"      Error variance changes with measurement level")
                else:
                    lines.append(f"  ✓ Homoscedasticity - consistent error variance")
                
                # Durbin-Watson for autocorrelation
                if not np.isnan(ra.get('durbin_watson', np.nan)):
                    dw = ra['durbin_watson']
                    lines.append(f"  Durbin-Watson: {dw:.3f}")
                    if dw < 1.5:
                        lines.append(f"    ⚠️  Positive autocorrelation in residuals")
                    elif dw > 2.5:
                        lines.append(f"    ⚠️  Negative autocorrelation in residuals")
                    else:
                        lines.append(f"    ✓ No significant autocorrelation")
                
                # Normality test
                if ra.get('residuals_normal') is not None:
                    if ra['residuals_normal']:
                        lines.append(f"  ✓ Residuals are approximately normal (Shapiro p={ra['shapiro_p']:.3f})")
                    else:
                        lines.append(f"  ⚠️  Residuals deviate from normality (Shapiro p={ra['shapiro_p']:.3f})")
        
        # Individual Pair Summary with enhanced details
        lines.append(f"\nINDIVIDUAL PAIRS SUMMARY:")
        for pair_stat in stats.get('pair_stats', []):
            name = pair_stat['name']
            n = pair_stat['n']
            r = pair_stat.get('individual_r', np.nan)
            diff_mean = pair_stat.get('diff_mean', np.nan)
            diff_std = pair_stat.get('diff_std', np.nan)
            
            if not np.isnan(r):
                lines.append(f"  {name}: {n:,} pts, r={r:.3f}, bias={diff_mean:.3f}±{diff_std:.3f}")
            else:
                lines.append(f"  {name}: {n:,} pts, bias={diff_mean:.3f}±{diff_std:.3f}")
        
        # Add plot-specific recommendations
        lines.append(f"\nRECOMMendations for {plot_name}:")
        if plot_type == 'scatter':
            lines.append("  • Points near diagonal line indicate good agreement")
            lines.append("  • Scatter pattern shows precision; bias shown by offset from line")
            lines.append("  • Look for outliers and non-linear patterns")
        elif plot_type == 'bland_altman':
            lines.append("  • Points within LoA indicate acceptable agreement")
            lines.append("  • Horizontal pattern indicates consistent bias")
            lines.append("  • Fan-shaped pattern suggests proportional bias")
        elif plot_type == 'residual':
            lines.append("  • Random scatter around zero indicates good model fit")
            lines.append("  • Patterns indicate systematic errors or non-linearity")
            lines.append("  • Increasing spread suggests heteroscedasticity")
        elif plot_type == 'pearson':
            lines.append("  • r > 0.7 indicates strong linear relationship")
            lines.append("  • R² shows proportion of variance explained")
            lines.append("  • Compare Pearson vs Spearman for linearity assessment")
        
        return "\n".join(lines)

    # REMOVED: _generate_cumulative_preview - Not used in current implementation, replaced by _update_cumulative_display

    def _calculate_cumulative_statistics(self, checked_pairs):
        """Calculate cumulative statistics for checked pairs using statistics calculators"""
        if not checked_pairs:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': 0}
        
        # Use simple calculation
        return self._calculate_simple_cumulative_stats(checked_pairs)
    
    def _calculate_simple_cumulative_stats(self, checked_pairs):
        """Simple cumulative statistics calculation as fallback"""
        if not checked_pairs:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': 0}
        
        # Collect all data from checked pairs
        all_ref_data, all_test_data = self._collect_cumulative_data(checked_pairs)
        
        if not all_ref_data or not all_test_data:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': len(checked_pairs)}
        
        # Calculate statistics using the same helper methods
        ref_clean = np.array(all_ref_data)
        test_clean = np.array(all_test_data)
        
        correlation_stats = self._calculate_correlation(ref_clean, test_clean)
        difference_stats = self._calculate_difference_stats(ref_clean, test_clean)
        
        return {
            'r': correlation_stats.get('r', np.nan),
            'rms': difference_stats.get('rms', np.nan),
            'n': len(ref_clean),
            'pairs': len(checked_pairs)
        }
    
    def _collect_cumulative_data(self, checked_pairs):
        """Collect all data from checked pairs"""
        all_ref_data = []
        all_test_data = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                ref_data = aligned_data['ref_data']
                test_data = aligned_data['test_data']
                
                if ref_data is not None and test_data is not None:
                    # Filter valid data
                    valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                    all_ref_data.extend(ref_data[valid_mask])
                    all_test_data.extend(test_data[valid_mask])
        
        return all_ref_data, all_test_data
        
    def _format_cumulative_stats(self, stats, n_pairs):
        """Format cumulative statistics for display"""
        r_str = f"{stats['r']:.3f}" if not np.isnan(stats['r']) else "N/A"
        rms_str = f"{stats['rms']:.3f}" if not np.isnan(stats['rms']) else "N/A"
        
        return f"Cumulative Stats: r = {r_str}, RMS = {rms_str}, N = {stats['n']:,} points ({n_pairs} pairs shown)"
    
    def _enhance_plot_config_with_overlays(self, plot_config):
        """Enhance plot config with overlay configurations from window - direct passthrough"""
        try:
            if hasattr(self.window, '_get_overlay_parameters'):
                # Get overlay parameters from window (direct passthrough from method definitions)
                overlay_params = self.window._get_overlay_parameters()
                
                # Add overlay parameters to plot config
                plot_config.update(overlay_params)
                
                print(f"[ComparisonWizard] Enhanced plot config with overlay parameters: {overlay_params}")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error enhancing plot config with overlays: {e}")
            import traceback
            traceback.print_exc()
    
    def update_overlay_matplotlib_artists(self, overlay_id, config):
        """Update matplotlib artists for overlay changes"""
        try:
            # This method is called from the overlay wizard to update existing plots
            if not hasattr(self.window, 'canvas') or not self.window.canvas:
                return
                
            # Get overlay artists if they exist
            overlay_artists = getattr(self.window, 'overlay_artists', {})
            
            if overlay_id in overlay_artists:
                artist = overlay_artists[overlay_id]
                
                # Update artist properties based on overlay type
                if isinstance(artist, list):
                    for a in artist:
                        self._update_single_artist(a, config)
                else:
                    self._update_single_artist(artist, config)
                
                # Redraw canvas
                self.window.canvas.draw()
                
                print(f"[ComparisonWizard] Updated matplotlib artists for overlay: {overlay_id}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error updating overlay artists: {e}")
    
    def _update_single_artist(self, artist, config):
        """Update a single matplotlib artist with overlay config"""
        try:
            # Update common properties
            if hasattr(artist, 'set_color') and 'color' in config:
                artist.set_color(config['color'])
            
            if hasattr(artist, 'set_alpha') and 'alpha' in config:
                artist.set_alpha(config['alpha'])
            
            if hasattr(artist, 'set_linewidth') and 'linewidth' in config:
                artist.set_linewidth(config['linewidth'])
            
            if hasattr(artist, 'set_linestyle') and 'linestyle' in config:
                artist.set_linestyle(config['linestyle'])
            
            if hasattr(artist, 'set_markersize') and 'markersize' in config:
                artist.set_markersize(config['markersize'])
            
            if hasattr(artist, 'set_marker') and 'marker' in config:
                artist.set_marker(config['marker'])
            
            # Update text properties
            if hasattr(artist, 'set_fontsize') and 'fontsize' in config:
                artist.set_fontsize(config['fontsize'])
            
            if hasattr(artist, 'set_fontweight') and 'fontweight' in config:
                artist.set_fontweight(config['fontweight'])
            
            if hasattr(artist, 'set_bbox') and 'bbox' in config:
                artist.set_bbox(config['bbox'])
                
        except Exception as e:
            print(f"[ComparisonWizard] Error updating single artist: {e}")
    
    def on_overlay_visibility_changed(self, overlay_id, visible):
        """Handle overlay visibility changes"""
        try:
            # Update overlay artists visibility
            overlay_artists = getattr(self.window, 'overlay_artists', {})
            
            if overlay_id in overlay_artists:
                artist = overlay_artists[overlay_id]
                
                if isinstance(artist, list):
                    for a in artist:
                        a.set_visible(visible)
                else:
                    artist.set_visible(visible)
                
                # Redraw canvas
                if hasattr(self.window, 'canvas') and self.window.canvas:
                    self.window.canvas.draw()
                
                print(f"[ComparisonWizard] {'Showed' if visible else 'Hidden'} overlay: {overlay_id}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error changing overlay visibility: {e}")

    def _clear_figure_completely(self, fig):
        """Completely clear a matplotlib figure and reset its state"""
        try:
            # Clear all axes and their contents
            fig.clear()
            
            # Reset figure properties
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(1.0)
            
            # Ensure proper cleanup
            fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"[ComparisonWizard] Error clearing figure: {e}")

