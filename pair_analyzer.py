"""
Pair Analyzer

This module provides the PairAnalyzer class that processes pairs from a PairManager
using a specified method configuration, with intelligent caching for performance.
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from overlay import Overlay

class MethodConfigOp:
    """Represents a method configuration operation for caching purposes."""
    
    def __init__(self, method_name: str, parameters: Dict[str, Any], 
                 plot_script: Optional[str] = None, stats_script: Optional[str] = None,
                 performance_options: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[str] = None):
        self.method_name = method_name
        self.parameters = parameters
        self.plot_script = plot_script
        self.stats_script = stats_script
        self.performance_options = performance_options or {}
        self.timestamp = timestamp or str(time.time())
    
    def get_cache_key(self) -> str:
        """Generate a unique cache key for this method configuration."""
        config_data = {
            'method': self.method_name,
            'parameters': self.parameters,
            'plot_script': self.plot_script,
            'stats_script': self.stats_script,
            'performance_options': self.performance_options
        }
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


class PairAnalyzer:
    """
    Analyzes pairs using specified comparison methods with intelligent caching.
    
    The analyzer processes visible pairs from a PairManager, runs the appropriate
    comparison method, and generates both scatter data and global overlays.
    """
    
    def __init__(self, comparison_registry=None):
        """
        Initialize the PairAnalyzer.
        
        Args:
            comparison_registry: Registry containing comparison method classes
        """
        self.comparison_registry = comparison_registry
        self._cache = {}  # Internal cache: {cache_key: results}
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        
    def analyze(self, pair_manager, method_config: MethodConfigOp) -> Dict[str, Any]:
        """
        Analyze visible pairs using the specified method configuration.
        
        Args:
            pair_manager: PairManager instance containing pairs
            method_config: MethodConfigOp instance with method and parameters
            
        Returns:
            Dictionary containing:
            - overlays: List of global overlay objects
            - scatter_data: Combined plot data with styling
            - errors: Dictionary of any analysis errors
            - stats: Cache performance statistics
        """
        print(f"[PairAnalyzer] Starting analysis with method: {method_config.method_name}")
        
        # Get visible pairs
        visible_pairs = self._get_visible_pairs(pair_manager)
        if not visible_pairs:
            print("[PairAnalyzer] No visible pairs to analyze")
            return self._empty_results()
        
        print(f"[PairAnalyzer] Processing {len(visible_pairs)} visible pairs")
        
        # Get comparison method class
        comparison_cls = self._get_comparison_class(method_config.method_name)
        if not comparison_cls:
            return self._error_results(f"Comparison method '{method_config.method_name}' not found")
        
        # Get plot_type from comparison class
        plot_type = getattr(comparison_cls, 'plot_type', 'scatter')
        print(f"[PairAnalyzer] Using plot_type: {plot_type}")
        
        # Process each visible pair
        scatter_data = []
        all_stats_results = []
        errors = {}
        
        for pair in visible_pairs:
            try:
                # Check cache first
                cache_key = self._generate_pair_cache_key(pair.pair_id, method_config)
                cached_result = self._get_cached_result(cache_key)
                
                if cached_result:
                    self._cache_stats['hits'] += 1
                    print(f"[PairAnalyzer] Cache hit for pair: {pair.name}")
                    pair_scatter_data = cached_result['scatter_data']
                    pair_stats = cached_result['stats_results']
                else:
                    self._cache_stats['misses'] += 1
                    print(f"[PairAnalyzer] Cache miss for pair: {pair.name}, computing...")
                    
                    # Run analysis for this pair
                    pair_scatter_data, pair_stats = self._analyze_single_pair(
                        pair, comparison_cls, method_config
                    )
                    
                    # Cache the results
                    self._cache_result(cache_key, {
                        'scatter_data': pair_scatter_data,
                        'stats_results': pair_stats,
                        'computed_at': time.time()
                    })
                
                # Add to combined results
                scatter_data.append(pair_scatter_data)
                all_stats_results.append(pair_stats)
                
            except Exception as e:
                print(f"[PairAnalyzer] Error analyzing pair {pair.name}: {e}")
                errors[pair.pair_id] = str(e)
        
        # Generate global overlays from combined statistics
        overlays = self._generate_overlays(comparison_cls, all_stats_results, method_config, scatter_data)
        
        # Update cache statistics
        self._cache_stats['size'] = len(self._cache)
        
        print(f"[PairAnalyzer] Analysis complete. Cache stats: {self._cache_stats}")
        
        return {
            'overlays': overlays,
            'scatter_data': scatter_data,
            'errors': errors,
            'cache_stats': self._cache_stats.copy(),
            'method_name': method_config.method_name,
            'plot_type': plot_type,
            'n_pairs_processed': len(scatter_data)
        }
    
    def _get_visible_pairs(self, pair_manager) -> List[Any]:
        """Get list of pairs that should be processed (show=True)."""
        if not hasattr(pair_manager, 'get_visible_pairs'):
            print("[PairAnalyzer] PairManager missing get_visible_pairs method")
            return []
        
        visible_pairs = pair_manager.get_visible_pairs()
        print(f"[PairAnalyzer] Found {len(visible_pairs)} visible pairs")
        
        return visible_pairs
    
    def _get_comparison_class(self, method_name: str):
        """Get comparison class from registry."""
        if not self.comparison_registry:
            print("[PairAnalyzer] No comparison registry available")
            return None
        
        # Try to get the class from registry
        if hasattr(self.comparison_registry, 'get'):
            return self.comparison_registry.get(method_name)
        elif hasattr(self.comparison_registry, 'all'):
            all_methods = self.comparison_registry.all()
            return all_methods.get(method_name)
        else:
            print(f"[PairAnalyzer] Unknown registry interface: {type(self.comparison_registry)}")
            return None
    
    def _generate_pair_cache_key(self, pair_id: str, method_config: MethodConfigOp) -> str:
        """Generate cache key combining pair ID and method config."""
        config_key = method_config.get_cache_key()
        return f"{pair_id}_{config_key}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        return self._cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result."""
        self._cache[cache_key] = result
        
        # Simple cache size management - remove oldest if too large
        max_cache_size = 100  # Adjust based on memory constraints
        if len(self._cache) > max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = sorted(self._cache.keys(), 
                               key=lambda k: self._cache[k].get('computed_at', 0))
            for key in oldest_keys[:len(self._cache) - max_cache_size + 10]:
                del self._cache[key]
            print(f"[PairAnalyzer] Cache pruned to {len(self._cache)} entries")
    

    
    def _analyze_single_pair(self, pair, comparison_cls, method_config: MethodConfigOp) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Analyze a single pair using the comparison method.
        
        Returns:
            Tuple of (scatter_data, stats_results)
        """
        # Create comparison instance with parameters
        comparison = comparison_cls(**method_config.parameters)
        
        # Get aligned data from pair
        ref_data = pair.aligned_ref_data
        test_data = pair.aligned_test_data
        
        # Run plot_script to get scatter data
        x_data, y_data, plot_metadata = comparison.plot_script(ref_data, test_data, method_config.parameters)
        
        # Run stats_script to get statistical results
        stats_results = comparison.stats_script(x_data, y_data, ref_data, test_data, method_config.parameters)
        
        # Get plot_type from comparison class
        plot_type = getattr(comparison_cls, 'plot_type', 'scatter')
        
        # Prepare scatter data with pair styling
        scatter_data = {
            'x_data': x_data,
            'y_data': y_data,
            'pair_name': pair.name,
            'pair_id': pair.pair_id,
            'color': getattr(pair, 'color', '#1f77b4'),
            'alpha': getattr(pair, 'alpha', 0.6),
            'n_points': len(x_data),
            'metadata': plot_metadata
        }
        
        # Handle plot_type specific styling
        if plot_type == "line":
            scatter_data['marker'] = '-'  # Line style instead of marker
            scatter_data['line_style'] = 'solid'
            print(f"[PairAnalyzer] Line plot styling applied for pair '{pair.name}'")
        else:
            scatter_data['marker'] = getattr(pair, 'marker_type', 'o')
            print(f"[PairAnalyzer] Standard marker styling applied for pair '{pair.name}'")
        
        # Debug logging for marker styles
        print(f"[PairAnalyzer] Pair '{pair.name}' scatter data:")
        print(f"  - color: {scatter_data['color']}")
        print(f"  - marker: {scatter_data['marker']}")
        print(f"  - alpha: {scatter_data['alpha']}")
        print(f"  - n_points: {scatter_data['n_points']}")
        
        return scatter_data, stats_results
    
    def _generate_overlays(self, comparison_cls, all_stats_results: List[Dict[str, Any]], 
                          method_config: MethodConfigOp, scatter_data: List[Dict[str, Any]]) -> List[Overlay]:
        """
        Generate global overlay objects from combined statistics.
        
        Args:
            comparison_cls: Comparison method class
            all_stats_results: List of stats results from all pairs
            method_config: Method configuration
            scatter_data: List of scatter data from all pairs
            
        Returns:
            List of Overlay objects
        """
        overlays = []
        
        if not all_stats_results:
            return overlays
        
        # Combine statistics from all pairs using recomputation approach
        combined_stats = self._combine_statistics(all_stats_results, scatter_data, comparison_cls, method_config)
        
        # Create an instance of the comparison class to generate overlays
        try:
            comparison_instance = comparison_cls(**method_config.parameters)
            overlays = comparison_instance.generate_overlays(combined_stats)
            print(f"[PairAnalyzer] Generated {len(overlays)} overlays using {comparison_cls.name}")
        except Exception as e:
            print(f"[PairAnalyzer] Error generating overlays from {comparison_cls.name}: {e}")
        
        return overlays
    
    def _combine_statistics(self, all_stats_results: List[Dict[str, Any]], scatter_data: List[Dict[str, Any]], 
                           comparison_cls, method_config: MethodConfigOp) -> Dict[str, Any]:
        """
        Combine statistics from multiple pairs by recomputing on combined data.
        
        This method concatenates x,y data from all visible pairs and recomputes 
        global statistics using the comparison method's stats_script.
        """
        if not all_stats_results or not scatter_data:
            return {}
        
        try:
            # Concatenate all x_data and y_data from visible pairs
            combined_x_data = []
            combined_y_data = []
            combined_ref_data = []
            combined_test_data = []
            
            for pair_scatter in scatter_data:
                x_data = pair_scatter.get('x_data', [])
                y_data = pair_scatter.get('y_data', [])
                
                if len(x_data) > 0 and len(y_data) > 0:
                    combined_x_data.extend(x_data)
                    combined_y_data.extend(y_data)
                    # For ref/test data, we'll use x_data/y_data as approximation
                    # In a perfect world, we'd have access to original aligned data
                    combined_ref_data.extend(x_data)
                    combined_test_data.extend(y_data)
            
            if not combined_x_data or not combined_y_data:
                print("[PairAnalyzer] No valid combined data for statistics computation")
                return all_stats_results[0] if all_stats_results else {}
            
            # Convert to numpy arrays if they aren't already
            import numpy as np
            combined_x_data = np.array(combined_x_data)
            combined_y_data = np.array(combined_y_data)
            combined_ref_data = np.array(combined_ref_data)
            combined_test_data = np.array(combined_test_data)
            
            # Create comparison instance and run stats_script on combined data
            comparison_instance = comparison_cls(**method_config.parameters)
            combined_stats = comparison_instance.stats_script(
                combined_x_data, combined_y_data, 
                combined_ref_data, combined_test_data, 
                method_config.parameters
            )
            
            print(f"[PairAnalyzer] Recomputed combined statistics from {len(scatter_data)} visible pairs")
            print(f"[PairAnalyzer] Combined data points: {len(combined_x_data)}")
            
            return combined_stats
            
        except Exception as e:
            print(f"[PairAnalyzer] Error recomputing combined statistics: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to first result if recomputation fails
            print("[PairAnalyzer] Falling back to first pair's statistics")
            return all_stats_results[0] if all_stats_results else {}
    
    def get_cached_results(self, pair_id: str, method_config: MethodConfigOp) -> Optional[Dict[str, Any]]:
        """Get cached results for a specific pair and method configuration."""
        cache_key = self._generate_pair_cache_key(pair_id, method_config)
        return self._get_cached_result(cache_key)
    
    def get_all_cached_results(self, method_config: MethodConfigOp) -> List[Dict[str, Any]]:
        """Get all cached results for a method configuration."""
        cached_results = []
        for cache_key, result in self._cache.items():
            if method_config.get_cache_key() in cache_key:
                cached_results.append(result)
        return cached_results
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'overlays': [],
            'scatter_data': [],
            'errors': {},
            'cache_stats': self._cache_stats.copy(),
            'method_name': None,
            'n_pairs_processed': 0
        }
    
    def _error_results(self, error_message: str) -> Dict[str, Any]:
        """Return error results structure."""
        print(f"[PairAnalyzer] Error: {error_message}")
        return {
            'overlays': [],
            'scatter_data': [],
            'errors': {'analyzer': error_message},
            'cache_stats': self._cache_stats.copy(),
            'method_name': None,
            'n_pairs_processed': 0
        }
    
    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        print("[PairAnalyzer] Cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            'size': len(self._cache),
            'stats': self._cache_stats.copy(),
            'keys': list(self._cache.keys())
        }
    
    def recombine_visible_pairs(self, pair_manager, method_config: MethodConfigOp) -> Dict[str, Any]:
        """
        Recombine cached statistics from only visible pairs to update overlays.
        
        This method is optimized for visibility toggles - it uses cached individual
        pair results but recombines them based on current visibility state.
        
        Args:
            pair_manager: PairManager instance containing pairs
            method_config: MethodConfigOp instance with method and parameters
            
        Returns:
            Dictionary containing updated overlays and scatter data for visible pairs
        """
        print(f"[PairAnalyzer] Recombining visible pairs for method: {method_config.method_name}")
        
        # Get visible pairs
        visible_pairs = self._get_visible_pairs(pair_manager)
        if not visible_pairs:
            print("[PairAnalyzer] No visible pairs to recombine")
            return self._empty_results()
        
        print(f"[PairAnalyzer] Recombining {len(visible_pairs)} visible pairs")
        
        # Get comparison method class
        comparison_cls = self._get_comparison_class(method_config.method_name)
        if not comparison_cls:
            return self._error_results(f"Comparison method '{method_config.method_name}' not found")
        
        # Collect cached results for visible pairs
        visible_scatter_data = []
        visible_stats_results = []
        missing_pairs = []
        
        for pair in visible_pairs:
            cache_key = self._generate_pair_cache_key(pair.pair_id, method_config)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                visible_scatter_data.append(cached_result['scatter_data'])
                visible_stats_results.append(cached_result['stats_results'])
                print(f"[PairAnalyzer] Using cached result for visible pair: {pair.name}")
            else:
                missing_pairs.append(pair)
                print(f"[PairAnalyzer] Missing cached result for pair: {pair.name}")
        
        # If we have missing pairs, we need to compute them
        if missing_pairs:
            print(f"[PairAnalyzer] Computing {len(missing_pairs)} missing pairs...")
            for pair in missing_pairs:
                try:
                    # Compute the missing pair
                    pair_scatter_data, pair_stats = self._analyze_single_pair(
                        pair, comparison_cls, method_config
                    )
                    
                    # Cache the result
                    cache_key = self._generate_pair_cache_key(pair.pair_id, method_config)
                    self._cache_result(cache_key, {
                        'scatter_data': pair_scatter_data,
                        'stats_results': pair_stats,
                        'computed_at': time.time()
                    })
                    
                    # Add to visible results
                    visible_scatter_data.append(pair_scatter_data)
                    visible_stats_results.append(pair_stats)
                    
                except Exception as e:
                    print(f"[PairAnalyzer] Error computing missing pair {pair.name}: {e}")
        
        # Generate overlays from combined statistics of visible pairs only
        overlays = self._generate_overlays(comparison_cls, visible_stats_results, method_config, visible_scatter_data)
        
        print(f"[PairAnalyzer] Recombination complete: {len(visible_scatter_data)} scatter datasets, {len(overlays)} overlays")
        
        return {
            'overlays': overlays,
            'scatter_data': visible_scatter_data,
            'errors': {},
            'cache_stats': self._cache_stats.copy(),
            'method_name': method_config.method_name,
            'n_pairs_processed': len(visible_scatter_data),
            'recombined': True  # Flag to indicate this was a recombination
        } 